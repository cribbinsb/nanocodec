"""tools/dev/aws_session.py: the instance-type fallback, on a fake EC2 (nothing is rented)."""
import time
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_session  # noqa: E402

ZONES = ("eu-north-1a", "eu-north-1b")


class FakeEc2:
    """Spot prices for the types in `prices`; run_instances answers each type with its
    error code from `refusals` (every zone), else launches."""

    def __init__(self, prices: dict[str, float], refusals: dict[str, str]):
        self.prices, self.refusals = prices, refusals
        self.launched: list[tuple[str, str, int]] = []

    def describe_spot_price_history(self, InstanceTypes, **_):
        price = self.prices.get(InstanceTypes[0])
        return {"SpotPriceHistory": [] if price is None else [
            {"AvailabilityZone": zone, "SpotPrice": str(price)} for zone in ZONES]}

    def describe_subnets(self, Filters):
        return {"Subnets": [{"SubnetId": f"subnet-{Filters[0]['Values'][0]}"}]}

    def run_instances(self, InstanceType, BlockDeviceMappings, NetworkInterfaces, **_):
        code = self.refusals.get(InstanceType)
        if code:
            raise ClientError({"Error": {"Code": code, "Message": code}}, "RunInstances")
        zone = NetworkInterfaces[0]["SubnetId"].removeprefix("subnet-")
        self.launched.append((InstanceType, zone, BlockDeviceMappings[0]["Ebs"]["VolumeSize"]))
        return {"Instances": [{"InstanceId": "i-fake", "SpotInstanceRequestId": "sir-fake"}]}


def session(ec2: FakeEc2, instance_type: str = "c7i.24xlarge", budget: float = 10.0,
            fallbacks: tuple[str, ...] | None = None) -> aws_session.Session:
    spec = aws_session.SessionSpec(
        region="eu-north-1", instance_type=instance_type, purpose="test", budget_usd=budget,
        lifetime_minutes=60, root_gb=40,
        fallback_types=aws_session.fallbacks_after(instance_type) if fallbacks is None
        else fallbacks)
    with patch("boto3.client", return_value=ec2):
        made = aws_session.Session(spec)
    made.ami = lambda: ("ami-fake", "/dev/sda1")
    made.await_running = lambda: None
    return made


PRICES = {"c7i.24xlarge": 0.56, "c6i.24xlarge": 0.44, "c7i.48xlarge": 2.6, "m7i.48xlarge": 2.4}
#: The quota tests' own types (their mechanics do not depend on the default ladder).
QUOTA_PRICES = {"c7a.24xlarge": 0.55, "c7a.16xlarge": 0.4}


def test_no_capacity_anywhere_falls_back_to_the_next_type(capsys):
    """2026-11-20: no c7a.metal-48xl in any eu-north-1 zone for two hours; the next type
    launched at once."""
    ec2 = FakeEc2(PRICES, {"c7i.24xlarge": "InsufficientInstanceCapacity"})
    run = session(ec2)
    run.launch(run.candidates())
    assert ec2.launched == [("c6i.24xlarge", "eu-north-1a", 40)]
    assert run.spec.instance_type == "c6i.24xlarge"  # the ledger and cost use it
    out = capsys.readouterr().out
    assert "c7i.24xlarge: no capacity in any zone; trying c6i.24xlarge" in out
    assert "fell back from c7i.24xlarge to c6i.24xlarge" in out


def test_a_full_spot_quota_tries_the_next_type_before_waiting(capsys):
    ec2 = FakeEc2(PRICES, {"c7i.24xlarge": aws_session.QUOTA_FULL})
    run = session(ec2)
    with patch.object(aws_session.time, "sleep", side_effect=AssertionError("waited")):
        run.launch(run.candidates())
    assert ec2.launched[0][0] == "c6i.24xlarge"
    assert "c7i.24xlarge: the spot instance quota is full; trying c6i.24xlarge" in \
        capsys.readouterr().out


def test_types_over_the_budget_or_without_a_price_are_skipped():
    prices = dict(PRICES)
    del prices["m7i.48xlarge"]  # not offered in the region
    ladder = ("c7i.24xlarge", "m7i.48xlarge", "c6i.24xlarge")
    run = session(FakeEc2(prices, {}), "c7i.48xlarge", budget=2.0, fallbacks=ladder)
    # worst case 1.5x the price over the hour: c7i 0.84 and c6i 0.66 fit; c7i.48xlarge 3.9
    # does not, and m7i has no price
    assert [c.instance_type for c in run.candidates()] == ["c7i.24xlarge", "c6i.24xlarge"]
    with pytest.raises(aws_session.BudgetExceeded):
        session(FakeEc2(prices, {}), budget=0.1).candidates()


def test_a_timing_session_never_changes_its_type():
    ec2 = FakeEc2({"c8a.large": 0.1}, {"c8a.large": "InsufficientInstanceCapacity"})
    run = session(ec2, "c8a.large", fallbacks=())
    with pytest.raises(aws_session.NoCapacity, match="c8a.large"):
        run.launch(run.candidates())
    assert ec2.launched == []


def test_a_quota_full_on_every_type_waits_and_retries_the_list():
    ec2 = FakeEc2(PRICES, {t: aws_session.QUOTA_FULL for t in PRICES})
    run = session(ec2)
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 2:  # the quota frees after two waits
            ec2.refusals = {}

    with patch.object(aws_session.time, "sleep", sleep):
        run.launch(run.candidates())
    assert len(waits) == 2 and ec2.launched[0][0] == "c7i.24xlarge"


def test_the_fallback_ladder_follows_the_type_asked_for():
    assert aws_session.fallbacks_after("c7i.24xlarge") == aws_session.THROUGHPUT_X86_TYPES[1:]
    assert aws_session.fallbacks_after("c7i.4xlarge") == aws_session.THROUGHPUT_X86_TYPES


def test_a_scoring_root_volume_holds_its_upload(tmp_path):
    """2026-11-20: a 24 GB root took a whole held-out set's ~17 GB of streams, then dpkg
    ran out of space (exit 100)."""
    import aws_heldout_chart as heldout
    data = tmp_path / "streams"
    data.mkdir()
    (data / "a.ivf").write_bytes(b"x" * 3000)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "streams").symlink_to(data)  # a linked directory is uploaded too (tar -h)
    (stage / "repo.tar").write_bytes(b"y" * 1000)
    assert heldout.staged_bytes(stage) == 4000
    gib = 1 << 30
    assert heldout.root_gb(17 * gib) == heldout.ROOT_SYSTEM_GB + 22  # 17 x 1.25, up
    assert heldout.root_gb(0) == heldout.ROOT_SYSTEM_GB


class QuotaEc2(FakeEc2):
    """Spot instances in use: (type, vCPUs, Purpose) per describe call, in turn."""

    def __init__(self, usage_by_call):
        super().__init__(QUOTA_PRICES, {})
        self.usage_by_call, self.calls = usage_by_call, 0

    def describe_instance_types(self, InstanceTypes):
        vcpus = {"c7a.24xlarge": 96, "c7a.16xlarge": 64}[InstanceTypes[0]]
        return {"InstanceTypes": [{"VCpuInfo": {"DefaultVCpus": vcpus}}]}

    def get_paginator(self, name):
        usage = self.usage_by_call[min(self.calls, len(self.usage_by_call) - 1)]
        self.calls += 1
        instances = [{"InstanceType": kind, "CpuOptions": {"CoreCount": cpus,
                                                            "ThreadsPerCore": 1},
                      "Tags": [{"Key": "Purpose", "Value": purpose}]}
                     for kind, cpus, purpose in usage]
        pages = [{"Reservations": [{"Instances": instances}]}]
        return type("P", (), {"paginate": lambda self, **_: pages})()


def regional(monkeypatch, tmp_path, usage: dict[str, list], instance_type="c7a.24xlarge",
             fallbacks=(), spill=True):
    """A session whose regions each answer with their own QuotaEc2 (usage per call); it may
    spill unless `spill` is False (the default without `NC_AWS_SPILL`)."""
    full = [("m7a.48xlarge", 192, "measure"), ("m7a.48xlarge", 192, "measure"),
            ("c7a.24xlarge", 96, "measure")]  # a region not named is full
    usage = {r: usage.get(r, [full]) for r in ("eu-north-1", "eu-west-1", "eu-central-1")}
    ecs = {region: QuotaEc2(calls) for region, calls in usage.items()}
    run = session(ecs["eu-north-1"], instance_type, fallbacks=fallbacks)
    run._client = lambda service, region: ecs[region]
    run.spec = aws_session.replace(run.spec, spill=spill)
    monkeypatch.setattr(aws_session, "spot_quota", lambda region: 512)
    monkeypatch.setattr(aws_session, "QUOTA_POLL_S", 0)
    monkeypatch.setattr(aws_session, "LAUNCH_LOCK_DIR", tmp_path)
    return run, ecs


BUSY = [("m7a.48xlarge", 192, "measure"), ("c7a.24xlarge", 96, "measure"),
        ("c7a.24xlarge", 96, "measure")]  # 384 of 512 in use
FULL = [("m7a.48xlarge", 192, "measure"), ("m7a.48xlarge", 192, "measure"),
        ("c7a.24xlarge", 96, "measure")]  # 480 of 512


def test_a_full_home_region_spills_to_the_next_region_with_room(monkeypatch, tmp_path, capsys):
    # eu-north-1: 384 in use and 96 kept for the absent build box: 32 free
    run, _ = regional(monkeypatch, tmp_path, {"eu-north-1": [BUSY], "eu-west-1": [FULL],
                                              "eu-central-1": [[]]})
    candidates = run.claim_room()
    out = capsys.readouterr().out
    assert run.spec.region == "eu-central-1" and candidates[0].instance_type == "c7a.24xlarge"
    assert "chosen over: eu-north-1" in out and "WAITING" not in out
    assert run.ec2 is run._client("ec2", "eu-central-1")


def test_without_the_spill_switch_a_full_home_region_waits_at_home(monkeypatch, tmp_path,
                                                                  capsys):
    monkeypatch.delenv(aws_session.SPILL_ENV, raising=False)
    assert not aws_session.SessionSpec(region="eu-north-1", instance_type="c7i.24xlarge",
                                      purpose="test").spill
    run, _ = regional(monkeypatch, tmp_path, {"eu-north-1": [BUSY, BUSY[1:]],
                                              "eu-central-1": [[]]}, spill=False)
    run.claim_room()
    assert run.spec.region == "eu-north-1"


def test_every_region_full_waits_then_takes_the_first_that_frees(monkeypatch, tmp_path, capsys):
    run, _ = regional(monkeypatch, tmp_path, {"eu-north-1": [BUSY, BUSY[1:]],
                                              "eu-west-1": [FULL], "eu-central-1": [FULL]})
    run.claim_room()
    out = capsys.readouterr().out
    assert out.count("WAITING for spot quota in every region") == 1
    assert "96 kept for the build box" in out and run.spec.region == "eu-north-1"


def test_the_build_box_share_stays_free_beside_a_running_build_box(monkeypatch, tmp_path,
                                                                    capsys):
    usage = [("c7a.16xlarge", 64, "buildbox"), *BUSY[1:], ("c7a.24xlarge", 96, "measure")]
    run, _ = regional(monkeypatch, tmp_path, {"eu-north-1": [usage], "eu-west-1": [[]]})
    run.claim_room()  # 352 in use and 96 kept: 64 free, so the 96-vCPU box spills
    assert run.spec.region == "eu-west-1"
    assert "64 vCPU free of 512 (352 in use, 96 kept for the build box)" in \
        capsys.readouterr().out


def test_waiting_too_long_fails_loudly(monkeypatch, tmp_path):
    run, _ = regional(monkeypatch, tmp_path, {r: [FULL] for r in
                                              ("eu-north-1", "eu-west-1", "eu-central-1")})
    run.spec = aws_session.replace(run.spec, launch_wait_minutes=0)
    with pytest.raises(aws_session.NoCapacity, match="waited 0 min"):
        run.claim_room()


def test_a_smaller_fallback_that_fits_ends_the_wait(monkeypatch, tmp_path, capsys):
    usage = [("m7a.48xlarge", 192, "measure"), *BUSY[1:2]]
    run, _ = regional(monkeypatch, tmp_path, {"eu-north-1": [usage]},
                      fallbacks=("c7a.16xlarge",))
    run.claim_room()  # 288 in use, 96 kept: 128 free... the 24xlarge fits at home
    out = capsys.readouterr().out
    assert "WAITING" not in out and run.spec.region == "eu-north-1"


def test_the_cheapest_pool_per_vcpu_wins_over_the_first_region_with_room(monkeypatch,
                                                                       tmp_path):
    run, ecs = regional(monkeypatch, tmp_path, {"eu-north-1": [[]], "eu-west-1": [[]],
                                                "eu-central-1": [[]]},
                        fallbacks=("c7a.16xlarge",))
    ecs["eu-north-1"].prices = {"c7a.24xlarge": 0.9, "c7a.16xlarge": 0.5}  # 0.0094 vs 0.0078
    ecs["eu-west-1"].prices = {"c7a.24xlarge": 0.6, "c7a.16xlarge": 0.6}   # 0.0063 per vCPU
    ecs["eu-central-1"].prices = {"c7a.24xlarge": 1.4, "c7a.16xlarge": 1.2}
    candidates = run.claim_room()
    assert run.spec.region == "eu-west-1" and candidates[0].instance_type == "c7a.24xlarge"


def test_a_launch_waits_for_this_machines_other_launch_at_home_rather_than_spill(
        monkeypatch, tmp_path, capsys):
    """2026-11-21: two scoring sessions started together; the second found eu-north-1's
    launch lock held, spilled to eu-west-1, found no capacity there and failed."""
    run, _ = regional(monkeypatch, tmp_path, {"eu-north-1": [[]], "eu-central-1": [[]]})
    monkeypatch.setattr(aws_session, "LAUNCH_LOCK_POLL_S", 0)
    other = aws_session.try_launch_lock("eu-north-1")
    polls = []
    real_sleep = aws_session.time.sleep

    def sleep(seconds):  # the other launch ends while this one waits
        polls.append(seconds)
        aws_session.release_launch_lock(other)
        real_sleep(0)
    monkeypatch.setattr(aws_session.time, "sleep", sleep)
    run.claim_room()
    out = capsys.readouterr().out
    assert polls and "waiting for this machine's other launch" in out
    assert run.spec.region == "eu-north-1"  # the second pass looks from home again
    # once it is free the regions are priced as usual, home among them
    assert "another launch from this machine" not in out.strip().splitlines()[-1]


def test_a_fleet_session_that_cannot_launch_within_its_wait_fails_loudly(monkeypatch, tmp_path):
    """heldout_fleet.py's sessions wait about a minute, then their driver relaunches them."""
    run, _ = regional(monkeypatch, tmp_path, {r: [FULL] for r in
                                              ("eu-north-1", "eu-west-1", "eu-central-1")})
    run.spec = aws_session.replace(run.spec, launch_wait_minutes=0.01)
    monkeypatch.setattr(aws_session.time, "sleep", lambda s: None)
    clock = iter(range(0, 10_000, 30))
    monkeypatch.setattr(aws_session.time, "monotonic", lambda: next(clock))
    with pytest.raises(aws_session.NoCapacity, match="waited 0.01 min"):
        run.claim_room()


def test_no_type_within_budget_in_any_region_fails_at_once(monkeypatch, tmp_path, capsys):
    """2026-11-21: $3 sessions waited silently where nothing cost under $3.28."""
    run, _ = regional(monkeypatch, tmp_path, {"eu-north-1": [[]]})
    run.spec = aws_session.replace(run.spec, budget_usd=0.01)
    with pytest.raises(aws_session.BudgetExceeded, match="in any region"):
        run.claim_room()
    assert "WAITING" not in capsys.readouterr().out


class FakeGroups:
    """One security group whose SSH ingress `authorize_security_group_ingress` extends."""

    def __init__(self, cidrs: list[str]):
        self.cidrs = list(cidrs)

    def describe_security_groups(self, GroupIds):
        return {"SecurityGroups": [{"GroupId": GroupIds[0], "IpPermissions": [
            {"IpRanges": [{"CidrIp": c} for c in self.cidrs]}]}]}

    def authorize_security_group_ingress(self, GroupId, IpPermissions):
        self.cidrs += [r["CidrIp"] for p in IpPermissions for r in p["IpRanges"]]


def test_a_reattached_box_lets_this_hosts_new_address_in_once():
    made = session(FakeEc2(PRICES, {}))
    made.ec2, made.group_id = FakeGroups(["198.51.100.1/32"]), "sg-fake"
    with patch.object(aws_session, "public_ip", return_value="203.0.113.7"):
        made.allow_this_host()
        made.allow_this_host()
    assert made.ec2.cidrs == ["198.51.100.1/32", "203.0.113.7/32"]


def test_ssh_rechecks_the_address_at_most_once_a_minute():
    made = session(FakeEc2(PRICES, {}))
    made.ec2, made.group_id, made.host = FakeGroups([]), "sg-fake", "192.0.2.5"
    addresses = iter(["203.0.113.7", "203.0.113.8"])
    clock = [1000.0]
    with patch.object(aws_session, "public_ip", side_effect=lambda: next(addresses)), \
            patch.object(aws_session.time, "monotonic", side_effect=lambda: clock[0]):
        made.ssh_base()
        made.ssh_base()  # within the minute: no second lookup
        clock[0] += aws_session.ACCESS_RECHECK_S
        made.ssh_base()  # the address moved meanwhile
    assert made.ec2.cidrs == ["203.0.113.7/32", "203.0.113.8/32"]


def test_a_spot_notice_pulls_at_once_and_once(monkeypatch, tmp_path):
    """Session.pulling: the interruption notice brings the files back before the next
    periodic pull is due (AWS gives two minutes)."""
    monkeypatch.setattr(aws_session, "NOTICE_POLL_SECONDS", 0.01)
    made = aws_session.Session.__new__(aws_session.Session)
    made.spec = aws_session.SessionSpec(region="eu-north-1", instance_type="c6i.24xlarge",
                                        purpose="t")
    pulls, notices = [], iter([False, True])
    made.spot_notice = lambda: next(notices, True)
    made.download_new = lambda *_: pulls.append(time.monotonic())
    with made.pulling("r", tmp_path, ".m", every_s=3600):
        time.sleep(0.2)
    assert len(pulls) == 2  # the notice's pull and the last one, no periodic pull


LEDGER_TEXT = """| logged | resources | type / region | purpose (event) | start | end | cost |
| 2026-11-24T07:41:38Z | instance (not launched) | c8i.2xlarge / eu-north-1 | bench (started) | 2026-11-24T07:41:38Z | — | cap $0.06 |
| 2026-11-24T07:42:05Z | i-1 | c8i.2xlarge / eu-north-1 | bench (ended) | 2026-11-24T07:15:05Z | 2026-11-24T07:42:04Z | $0.03 |
| 2026-11-24T08:00:00Z | i-2 | c8g.2xlarge / eu-north-1 | astra18-sve2-g4 (ended) | 2026-11-24T07:36:48Z | 2026-11-24T07:59:47Z | $0.01 |
| 2026-11-24T09:00:00Z | i-3 | c7i.24xlarge / eu-north-1 | astra18-kernels (ended) | 2026-11-24T08:00:00Z | 2026-11-24T09:00:00Z | $0.56 |
| 2026-11-23T09:00:00Z | i-0 | c7i.24xlarge / eu-north-1 | bench (ended) | 2026-11-23T08:00:00Z | 2026-11-23T09:00:00Z | $9.00 |
"""


def test_the_ledger_summary_books_ended_rows_by_purpose_in_the_window():
    """RCA action R6: spend by purpose for a time window (`aws_session.py ledger`)."""
    charges = aws_session.ledger_charges(LEDGER_TEXT)
    assert len(charges) == 4  # the started row's cap is not a cost
    text = aws_session.ledger_summary(charges, "2026-11-24T07:00", None, "kind")
    assert "$    0.57  astra18" in text and "$    0.03  bench" in text
    assert text.endswith("$    0.60  total")  # yesterday's bench row is outside the window
    by_purpose = aws_session.ledger_summary(charges, "2026-11-24", "2026-11-24T08:30", "purpose")
    assert "astra18-sve2-g4" in by_purpose and "astra18-kernels" not in by_purpose
