"""tools/dev/nc_remote.py: spot avoids the types it reclaimed lately; on-demand
only after reclaims of several types or when no spot pool has capacity."""
import pytest
import datetime
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "dev" / "nc_remote.py"
spec = importlib.util.spec_from_file_location("nc_remote", SCRIPT)
nc_remote = importlib.util.module_from_spec(spec)
sys.modules["nc_remote"] = nc_remote  # dataclasses look their module up by name
spec.loader.exec_module(nc_remote)

NOW = nc_remote.parse("2026-11-20T09:50:00Z")


def box(instance_type, ended, reclaimed=True, zone="eu-north-1b", market=nc_remote.SPOT):
    return nc_remote.Box(instance_id=f"i-{instance_type}-{ended}", instance_type=instance_type,
                         zone=zone, price=0.5, launched="2026-11-20T00:00:00Z",
                         expected_end="2026-11-20T23:00:00Z", ended=ended, market=market,
                         reclaimed=reclaimed)


def test_no_reclaims_offers_every_type_on_spot():
    plan = nc_remote.plan_market([box("c7a.24xlarge", "2026-11-20T09:00:00Z", False)], NOW)
    assert plan.market == nc_remote.SPOT
    assert plan.types == nc_remote.INSTANCE_TYPES


def test_two_reclaims_of_one_type_leave_out_that_type_only():
    # 2026-11-20: c7a reclaimed in two zones within 2 h; the old rule went on-demand.
    history = [box("c7a.24xlarge", "2026-11-20T07:51:24Z", zone="eu-north-1b"),
               box("c7a.24xlarge", "2026-11-20T09:36:10Z", zone="eu-north-1c")]
    plan = nc_remote.plan_market(history, NOW)
    assert plan.market == nc_remote.SPOT
    assert "c7a.24xlarge" not in plan.types
    assert set(plan.types) == set(nc_remote.INSTANCE_TYPES) - {"c7a.24xlarge"}
    assert "c7a.24xlarge" in plan.reason


def test_a_reclaim_older_than_the_window_no_longer_excludes():
    old = NOW - nc_remote.RECLAIM_WINDOW - datetime.timedelta(minutes=1)
    plan = nc_remote.plan_market([box("c6i.24xlarge", nc_remote.iso(old))], NOW)
    assert plan.types == nc_remote.INSTANCE_TYPES


def test_reclaimed_types_never_send_the_box_on_demand():
    """Owner, 2026-11-26: spot only. However many types spot reclaimed, the next box is spot
    (the reclaimed types left out while their window lasts)."""
    three = [box(t, "2026-11-20T09:40:00Z")
             for t in ("c7a.24xlarge", "c6i.24xlarge", "c7i.24xlarge")]
    assert nc_remote.plan_market(three, NOW).market == nc_remote.SPOT
    later = nc_remote.parse("2026-11-20T09:40:00Z") + nc_remote.RECLAIM_WINDOW
    assert nc_remote.plan_market(three, later + datetime.timedelta(seconds=1)).market \
        == nc_remote.SPOT


def test_reclaims_of_every_type_never_leave_the_fleet_empty():
    """2026-11-27: c7i and c6i, the fleet's two types, were both reclaimed within the
    window, and the build box could not launch until the window ran out."""
    both = [box(t, "2026-11-20T09:40:00Z") for t in nc_remote.INSTANCE_TYPES]
    plan = nc_remote.plan_market(both, NOW)
    assert plan.market == nc_remote.SPOT and plan.types == nc_remote.INSTANCE_TYPES
    assert "every type again" in plan.reason
    assert nc_remote.RECLAIM_WINDOW <= datetime.timedelta(hours=1)


class FakeEc2:
    """create_fleet's EC2 calls: two zones; each request answered from `capacity`
    (market -> types that launch)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.requests = []

    def describe_subnets(self, Filters):
        return {"Subnets": [{"SubnetId": "subnet-a", "AvailabilityZone": "eu-north-1a"},
                            {"SubnetId": "subnet-b", "AvailabilityZone": "eu-north-1b"}]}

    def create_fleet(self, **request):
        market = request["TargetCapacitySpecification"]["DefaultTargetCapacityType"]
        types = [o["InstanceType"] for c in request["LaunchTemplateConfigs"]
                 for o in c["Overrides"]]
        self.requests.append((market, types))
        fits = [t for t in types if t in self.capacity.get(market, ())]
        if fits:
            return {"Instances": [{"InstanceIds": [f"i-{market}-{fits[0]}"]}]}
        return {"Errors": [{"ErrorCode": "InsufficientInstanceCapacity"}]}


def fake_aws(capacity):
    aws = nc_remote.Aws.__new__(nc_remote.Aws)  # no boto3 clients
    aws.ec2, aws.region = FakeEc2(capacity), nc_remote.REGION
    aws.on_demand_price = lambda instance_type: 1.0  # within the price cap
    aws.zone_prices = lambda instance_type: {"eu-north-1a": 0.56, "eu-north-1b": 0.6}
    return aws


def spot_without_c7i():
    return nc_remote.plan_market([box("c7i.24xlarge", "2026-11-20T09:36:10Z")], NOW)


def test_the_spot_fleet_offers_no_pool_of_a_reclaimed_type():
    aws = fake_aws({nc_remote.SPOT: {"c7i.24xlarge", "c6i.24xlarge"}})
    instance, market = aws.fleet_instance("lt-1", spot_without_c7i(), "end")
    assert (instance, market) == ("i-spot-c6i.24xlarge", nc_remote.SPOT)
    [(_, offered)] = aws.ec2.requests
    assert "c7i.24xlarge" not in offered
    assert len(offered) == 2 * (len(nc_remote.INSTANCE_TYPES) - 1)  # every zone


def test_no_spot_capacity_in_any_remaining_pool_falls_back_to_on_demand():
    aws = fake_aws({nc_remote.SPOT: {"c7i.24xlarge"},
                    nc_remote.ON_DEMAND: set(nc_remote.INSTANCE_TYPES)})
    instance, market = aws.fleet_instance("lt-1", spot_without_c7i(), "end")
    assert market == nc_remote.ON_DEMAND
    assert [m for m, _ in aws.ec2.requests] == [nc_remote.SPOT, nc_remote.ON_DEMAND]


def test_no_on_demand_capacity_is_an_error():
    aws = fake_aws({})
    plan = nc_remote.MarketPlan(nc_remote.ON_DEMAND, nc_remote.INSTANCE_TYPES, "test")
    try:
        aws.fleet_instance("lt-1", plan, "end")
    except nc_remote.NoCapacity:
        return
    raise AssertionError("expected NoCapacity")


def test_the_fleet_weighs_each_type_by_its_vcpus_so_it_compares_price_per_vcpu():
    aws = fake_aws({nc_remote.SPOT: set(nc_remote.INSTANCE_TYPES)})
    seen = []
    original = aws.ec2.create_fleet
    aws.ec2.create_fleet = lambda **request: seen.append(request) or original(**request)
    aws.fleet_instance("lt-1", nc_remote.plan_market([], NOW), "end")
    request = seen[0]
    assert request["TargetCapacitySpecification"]["TotalTargetCapacity"] == nc_remote.MIN_VCPUS
    weights = {o["InstanceType"]: o["WeightedCapacity"]
               for o in request["LaunchTemplateConfigs"][0]["Overrides"]}
    assert set(weights.values()) == {96.0}
    assert all(nc_remote.TYPE_VCPUS[t] >= nc_remote.MIN_VCPUS for t in weights)


def test_a_full_spot_quota_never_falls_back_to_on_demand():
    class QuotaFull(FakeEc2):
        def create_fleet(self, **request):
            super().create_fleet(**request)
            return {"Errors": [{"ErrorCode": "MaxSpotInstanceCountExceeded"}]}
    aws = fake_aws({})
    aws.ec2 = QuotaFull({nc_remote.ON_DEMAND: set(nc_remote.INSTANCE_TYPES)})
    with pytest.raises(nc_remote.NoCapacity) as refused:
        aws.fleet_instance("lt-1", nc_remote.plan_market([], NOW), "end")
    assert refused.value.quota and [m for m, _ in aws.ec2.requests] == [nc_remote.SPOT]


def test_on_demand_above_the_price_cap_is_refused():
    aws = fake_aws({nc_remote.ON_DEMAND: set(nc_remote.INSTANCE_TYPES)})
    aws.on_demand_price = lambda instance_type: 4.37
    with pytest.raises(nc_remote.NoCapacity, match="within"):
        aws.create_fleet("lt-1", nc_remote.ON_DEMAND, nc_remote.INSTANCE_TYPES, "end")


def test_a_region_whose_spot_is_above_the_price_cap_offers_nothing():
    aws = fake_aws({nc_remote.SPOT: set(nc_remote.INSTANCE_TYPES)})
    aws.zone_prices = lambda instance_type: {"eu-west-1c": 1.93}
    with pytest.raises(nc_remote.NoCapacity, match="within"):
        aws.create_fleet("lt-1", nc_remote.SPOT, nc_remote.INSTANCE_TYPES, "end")
    assert aws.ec2.requests == []  # no fleet asked for


def test_a_pool_above_the_price_cap_is_not_offered_and_the_fleet_is_capped():
    aws = fake_aws({nc_remote.SPOT: set(nc_remote.INSTANCE_TYPES)})
    aws.zone_prices = lambda instance_type: {"eu-north-1a": 0.56, "eu-north-1b": 1.96}
    seen = []
    original = aws.ec2.create_fleet
    aws.ec2.create_fleet = lambda **request: seen.append(request) or original(**request)
    aws.create_fleet("lt-1", nc_remote.SPOT, nc_remote.INSTANCE_TYPES, "end")
    subnets = {o["SubnetId"] for o in seen[0]["LaunchTemplateConfigs"][0]["Overrides"]}
    assert subnets == {"subnet-a"} and seen[0]["SpotOptions"]["MaxTotalPrice"] == "1.5000"


def test_a_delisted_box_is_never_replaced_under_running_jobs():
    """2026-11-24: the build box relaunched as c7a after c7a left the list; it serves its
    jobs and ends once idle, while other reasons still replace a busy box."""
    act = nc_remote.reap_action
    assert act("c7a.24xlarge is no longer a build box type", True, idle=False) == ""
    assert act("c7a.24xlarge is no longer a build box type", True, idle=True) == nc_remote.END
    assert act("on-demand while spot is allowed", False, idle=False) == nc_remote.REPLACE
    assert act("", False, idle=True) == ""


def test_a_job_lost_to_a_reclaim_or_refused_on_a_notice_runs_again_on_a_new_box():
    retry = nc_remote.retry_after
    assert retry(nc_remote.SSH_FAILED, "shutting-down") and retry(nc_remote.SSH_FAILED, "terminated")
    assert retry(nc_remote.STATUS_SPOT_NOTICE, "terminated")
    assert not retry(nc_remote.SSH_FAILED, "running")  # a broken link, the box still up
    assert not retry(1, "terminated")  # the command failed: no retry


def test_the_box_side_names_the_same_notice_status():
    import nc_remote_job
    assert nc_remote_job.STATUS_SPOT_NOTICE == nc_remote.STATUS_SPOT_NOTICE


def test_a_running_job_is_told_once_when_its_box_gets_a_notice(capsys, monkeypatch):
    import nc_remote_job
    monkeypatch.setattr(nc_remote_job.time, "sleep", lambda s: None)
    polls = iter([None, None, '{"action": "terminate"}'])
    process = type("P", (), {"poll": lambda self: None})()
    nc_remote_job.warn_on_notice(process, notice=lambda: next(polls))
    assert capsys.readouterr().err.count("SPOT INTERRUPTION NOTICE") == 1
