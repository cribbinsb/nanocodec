"""tools/dev/nc_remote.py: the build box's SSH rule follows this PC's public address.
A reused box's rule is refreshed (it was only set at launch); a rule that already
admits the address is left alone. Fake EC2 security groups."""
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "dev" / "nc_remote.py"
spec = importlib.util.spec_from_file_location("nc_remote", SCRIPT)
nc_remote = importlib.util.module_from_spec(spec)
sys.modules["nc_remote"] = nc_remote  # dataclasses look their module up by name
spec.loader.exec_module(nc_remote)

OLD_IP, NEW_IP = "198.51.100.7", "203.0.113.9"


class FakeEc2:
    """One security group's SSH rules, as describe/authorize/revoke see them."""

    def __init__(self, cidrs):
        self.cidrs = list(cidrs)
        self.calls = []

    def describe_security_groups(self, **_):
        perms = [{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                  "IpRanges": [{"CidrIp": c}]} for c in self.cidrs]
        return {"SecurityGroups": [{"GroupId": "sg-1", "IpPermissions": perms}]}

    def create_tags(self, **_):
        pass

    def revoke_security_group_ingress(self, IpPermissions, **_):
        self.calls.append("revoke")
        gone = {r["CidrIp"] for p in IpPermissions for r in p["IpRanges"]}
        self.cidrs = [c for c in self.cidrs if c not in gone]

    def authorize_security_group_ingress(self, IpPermissions, **_):
        self.calls.append("authorize")
        self.cidrs += [r["CidrIp"] for p in IpPermissions for r in p["IpRanges"]]


def fake_aws(cidrs):
    aws = nc_remote.Aws.__new__(nc_remote.Aws)
    aws.region, aws.ec2 = nc_remote.REGION, FakeEc2(cidrs)
    aws._regional = {aws.region: aws}
    return aws


def box():
    return nc_remote.Box(instance_id="i-1", instance_type="c6i.24xlarge", zone="eu-north-1a",
                         price=0.5, launched="2026-11-24T01:00:00Z",
                         expected_end="2026-11-24T13:00:00Z", host="192.0.2.1")


def test_a_reused_box_admits_the_new_address(monkeypatch):
    monkeypatch.setattr(nc_remote, "public_ip", lambda: NEW_IP)
    aws = fake_aws([f"{OLD_IP}/32"])
    nc_remote.admit_this_host(aws, box())
    assert aws.ec2.cidrs == [f"{NEW_IP}/32"]


def test_an_unchanged_address_leaves_the_rule_alone(monkeypatch):
    monkeypatch.setattr(nc_remote, "public_ip", lambda: OLD_IP)
    aws = fake_aws([f"{OLD_IP}/32"])
    nc_remote.admit_this_host(aws, box())
    assert aws.ec2.cidrs == [f"{OLD_IP}/32"] and aws.ec2.calls == []


def test_ensure_box_refreshes_the_rule_of_the_box_it_reuses(monkeypatch):
    monkeypatch.setattr(nc_remote, "public_ip", lambda: NEW_IP)
    aws = fake_aws([f"{OLD_IP}/32"])
    live = box()
    monkeypatch.setattr(nc_remote, "Locked", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(nc_remote, "live_box", lambda _aws: live)
    monkeypatch.setattr(nc_remote, "day_cost", lambda: 0.0)
    monkeypatch.setattr(nc_remote, "provision", lambda _box: None)
    nc_remote.ensure_box(aws)
    assert aws.ec2.cidrs == [f"{NEW_IP}/32"]
