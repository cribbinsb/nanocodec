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


PRICES = {"c7a.metal-48xl": 1.1, "c7a.48xlarge": 1.8, "c7a.24xlarge": 0.55,
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
