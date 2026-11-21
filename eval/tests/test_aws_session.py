"""tools/dev/aws_session.py: the instance-type fallback, on a fake EC2 (nothing is rented)."""
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


def session(ec2: FakeEc2, instance_type: str = "c7a.metal-48xl", budget: float = 10.0,
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


PRICES = {"c7a.metal-48xl": 1.1, "c7a.48xlarge": 1.8, "c7a.24xlarge": 0.55, "c7a.16xlarge": 0.4,
          "c7i.48xlarge": 2.0, "m7a.48xlarge": 2.2}


def test_no_capacity_anywhere_falls_back_to_the_next_type(capsys):
    """2026-11-20: no c7a.metal-48xl or c7a.48xlarge in any eu-north-1 zone; the
    c7a.24xlarge launched at once when asked for by hand, two hours later."""
    ec2 = FakeEc2(PRICES, {"c7a.metal-48xl": "InsufficientInstanceCapacity",
                           "c7a.48xlarge": "InsufficientInstanceCapacity"})
    run = session(ec2)
    run.launch(run.candidates())
    assert ec2.launched == [("c7a.24xlarge", "eu-north-1a", 40)]
    assert run.spec.instance_type == "c7a.24xlarge"  # the ledger and cost use it
    out = capsys.readouterr().out
    assert "c7a.metal-48xl: no capacity in any zone; trying c7a.48xlarge" in out
    assert "fell back from c7a.metal-48xl to c7a.24xlarge" in out


def test_a_full_spot_quota_tries_the_next_type_before_waiting(capsys):
    ec2 = FakeEc2(PRICES, {"c7a.metal-48xl": aws_session.QUOTA_FULL})
    run = session(ec2)
    with patch.object(aws_session.time, "sleep", side_effect=AssertionError("waited")):
        run.launch(run.candidates())
    assert ec2.launched[0][0] == "c7a.48xlarge"
    assert "c7a.metal-48xl: the spot instance quota is full; trying c7a.48xlarge" in \
        capsys.readouterr().out


def test_types_over_the_budget_or_without_a_price_are_skipped():
    prices = dict(PRICES)
    del prices["c7i.48xlarge"]  # not offered in the region
    run = session(FakeEc2(prices, {}), budget=2.0)
    # worst case 1.5x the price over the hour: metal 1.65, 24xlarge 0.83 fit; 48xlarge 2.7
    # and m7a 3.3 do not
    assert [c.instance_type for c in run.candidates()] == ["c7a.metal-48xl", "c7a.24xlarge"]
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
    assert len(waits) == 2 and ec2.launched[0][0] == "c7a.metal-48xl"


def test_the_fallback_ladder_follows_the_type_asked_for():
    assert aws_session.fallbacks_after("c7a.48xlarge") == aws_session.THROUGHPUT_X86_TYPES[2:]
    assert aws_session.fallbacks_after("c7a.4xlarge") == aws_session.THROUGHPUT_X86_TYPES


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
        super().__init__(PRICES, {})
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
             fallbacks=()):
    """A session whose regions each answer with their own QuotaEc2 (usage per call)."""
    full = [("m7a.48xlarge", 192, "measure"), ("m7a.48xlarge", 192, "measure"),
            ("c7a.24xlarge", 96, "measure")]  # a region not named is full
    usage = {r: usage.get(r, [full]) for r in ("eu-north-1", "eu-west-1", "eu-central-1")}
    ecs = {region: QuotaEc2(calls) for region, calls in usage.items()}
    run = session(ecs["eu-north-1"], instance_type, fallbacks=fallbacks)
    run._client = lambda service, region: ecs[region]
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
    monkeypatch.setattr(aws_session, "QUOTA_QUEUE_MINUTES", 0)
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
    # once it is free the regions are priced as usual, home among them
    assert "another launch from this machine" not in out.strip().splitlines()[-1]
