#!/usr/bin/env python3
"""nc-remote: run heavy agent work (builds, tests, clippy, callgrind, conformance,
fuzzing) on one shared AWS build box instead of the PC.

    nc-remote run [--worktree DIR] [--name NAME] [--sync PATH ...] [--fetch PATH ...]
                  [--fetch-to DIR] [--env KEY=VALUE ...] [--jobs N] -- COMMAND ...
    nc-remote status | stop | reap | bake | ssh

`run` starts the box if it is not up, rsyncs the worktree (tracked and
untracked sources, not target/, .git, ignored files or raw video) into
~/ncbox/jobs/NAME/src on the box, runs COMMAND there with bash in a job slot
(tools/dev/nc_remote_job.py: CARGO_TARGET_DIR is the agent's own directory
on the box, reachable as `target/`, so rebuilds are incremental), streams its
output, rsyncs each `--fetch` path back (relative to the worktree, `target/...`
included, or `~/...` for the same path under both homes) and exits with the
command's status. One argument is a shell command line; several are quoted
as one command. `--sync` mirrors extra paths under the PC's home to the same
place under the box's home (e.g. `~/nanocodec-data/anchors/x86_64/bin`);
paths with private clips are refused. NAME defaults to the worktree's
directory name.

Lifecycle (see also docs/PROCEDURES.md "Remote build box"):
- One persistent box from an instant EC2 Fleet over INSTANCE_TYPES x every zone of
  REGION (spot, price-capacity-optimized; on-demand after repeated reclaims),
  from the newest baked AMI (`nc-remote bake`) or plain Ubuntu, provisioned
  by nc_remote_provision.sh on first boot.
- Its user-data (nc_remote_box.sh) sets the hard lifetime and an idle
  watchdog on the box itself: terminate after IDLE_MINUTES without a job or
  a sync, so nothing outlives the PC. `nc-remote reap` (run from cron on the
  PC and at the start of every command) runs the same idle check over SSH and
  writes the ledger row of a box that has gone (idle, lifetime or spot
  reclaim). `run` restarts a lost box and retries the command once.
- Budget: the day's estimate (instance hours x price + volume + IPv4,
  per UTC day) must stay under DAILY_BUDGET_USD: no start once it is passed,
  and a running box is stopped.
- Every resource is tagged Project=nanocodec and ExpectedEnd; every launch,
  termination, AMI bake and AMI removal goes into the AWS ledger
  (tools/dev/aws_session.py). Nothing untagged is ever touched.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import fcntl
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aws_session import (  # noqa: E402
    GP3_USD_PER_GB_MONTH, HOURS_PER_MONTH, PROJECT_TAG, PUBLIC_IPV4_USD_PER_HOUR, OWNER_TAG,
    WORK_DIR, AMI_PARAMETER, UBUNTU_RELEASE, SessionSpec, iso, ledger_row, public_ip, say,
    utc_now)

HERE = Path(__file__).resolve().parent
REGION = "eu-north-1"
#: The fleet's 64-vCPU x86 types (EC2 picks the type and zone).
#: eu-north-1 because its spot prices were a quarter of eu-west-1's
#: (2026-11-16: c7a.8xlarge $0.23/h against $0.89/h). That day's 64-vCPU
#: prices there: c6i $0.29, m7a $0.55, c7i $0.60, c7a $0.62 per hour.
INSTANCE_TYPES = ("c7a.16xlarge", "c7i.16xlarge", "c6i.16xlarge", "m7a.16xlarge",
                  "m7i.16xlarge", "m6i.16xlarge")
VCPUS = 64
#: One heavy job per 8 vCPU; each job gets cargo -j JOBS_PER_SLOT.
JOBS_PER_SLOT = 8
SLOTS = VCPUS // JOBS_PER_SLOT
#: Conformance suites (~6 GB), anchors and up to ~20 agents' targets.
ROOT_GB = 200
LIFETIME_MINUTES = 12 * 60
IDLE_MINUTES = 10
#: Room for an on-demand day (RECLAIM_WINDOW): ~$2.9/h for up to 12 h.
DAILY_BUDGET_USD = 40.0
#: Markets. Spot by default; after two spot reclaims within RECLAIM_WINDOW
#: (2026-11-16: c6i.16xlarge was reclaimed at 12:04, 12:33 and 13:03 and agents
#: kept losing jobs), on-demand until RECLAIM_WINDOW after the later one
#: (owner, 2026-11-17: never pay on-demand longer than needed).
SPOT, ON_DEMAND = "spot", "on-demand"
RECLAIM_WINDOW = datetime.timedelta(hours=2)
SPOT_RECLAIM_CODE = "Server.SpotInstanceTermination"
#: The Price List API is served from us-east-1 (prices are still REGION's).
PRICING_API_REGION = "us-east-1"
PURPOSE = "buildbox"
NAME = "nanocodec-buildbox"
#: Bump when nc_remote_provision.sh changes, so boxes re-provision.
PROVISION_VERSION = "4"
#: AMIs and their snapshots have no lifetime of their own; their ExpectedEnd is reviewed then.
AMI_EXPECTED_END = "2026-12-31T00:00:00Z"

STATE_DIR = WORK_DIR / "buildbox"
STATE = STATE_DIR / "state.json"
HISTORY = STATE_DIR / "history.json"
LOCK = STATE_DIR / "lock"
KEY = STATE_DIR / "key.pem"
SSH_USER = "ubuntu"
BOX_HOME = f"/home/{SSH_USER}"
SSH_READY_SECONDS = 10 * 60
#: Paths never synced to the box: the private clips and their copies.
PRIVATE_PATHS = ("/mldata", str(Path.home() / "nanocodec-data/clips"),
                 str(WORK_DIR / "cache"))
RSYNC_EXCLUDES = ("--exclude=/target", "--exclude=/.git", "--exclude=*.yuv",
                  "--exclude=*.y4m", "--exclude=__pycache__", "--max-size=64m")


@dataclass
class Box:
    """The live box, as recorded on the PC (state.json) and in the history."""
    instance_id: str
    instance_type: str
    zone: str
    price: float
    launched: str
    expected_end: str
    host: str = ""
    ended: str = ""
    market: str = SPOT
    reclaimed: bool = False

    def hours(self, start: datetime.datetime, end: datetime.datetime) -> float:
        """Billed hours of this box inside [start, end)."""
        launched = parse(self.launched)
        ended = parse(self.ended) if self.ended else utc_now()
        overlap = (min(ended, end) - max(launched, start)).total_seconds()
        return max(overlap, 0) / 3600

    def cost(self, start: datetime.datetime, end: datetime.datetime) -> float:
        storage = ROOT_GB * GP3_USD_PER_GB_MONTH / HOURS_PER_MONTH
        return self.hours(start, end) * (self.price + storage + PUBLIC_IPV4_USD_PER_HOUR)

    def spec(self) -> SessionSpec:
        return SessionSpec(region=REGION, instance_type=self.instance_type, purpose=PURPOSE)


def parse(stamp: str) -> datetime.datetime:
    return datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)


# --- PC-side records ---------------------------------------------------------------------

def load_state() -> Box | None:
    return Box(**json.loads(STATE.read_text())) if STATE.exists() else None


def save_state(box: Box | None) -> None:
    if box is None:
        STATE.unlink(missing_ok=True)
    else:
        STATE.write_text(json.dumps(asdict(box), indent=1))


def history() -> list[Box]:
    return [Box(**b) for b in json.loads(HISTORY.read_text())] if HISTORY.exists() else []


def record(box: Box) -> None:
    """Add or replace `box` in the history (the daily budget's source)."""
    boxes = [b for b in history() if b.instance_id != box.instance_id] + [box]
    HISTORY.write_text(json.dumps([asdict(b) for b in boxes], indent=1))


def day_cost() -> float:
    start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    return sum(b.cost(start, start + datetime.timedelta(days=1)) for b in history())


def total_cost() -> float:
    epoch = parse("2000-01-01T00:00:00Z")
    return sum(b.cost(epoch, utc_now()) for b in history())


class Locked:
    """The PC-wide lock around starting, stopping and reaping the box."""

    def __enter__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.handle = open(LOCK, "w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        self.handle.close()


# --- AWS ---------------------------------------------------------------------------------

class Aws:
    """The calls nc-remote makes; every resource it creates or changes is tagged ours."""

    def __init__(self):
        import boto3
        self.ec2 = boto3.client("ec2", region_name=REGION)
        self.ssm = boto3.client("ssm", region_name=REGION)

    @staticmethod
    def tag_list(expected_end: str) -> list[dict]:
        tags = {"Project": PROJECT_TAG, "Owner": OWNER_TAG, "Purpose": PURPOSE,
                "ExpectedEnd": expected_end, "Name": NAME}
        return [{"Key": k, "Value": v} for k, v in tags.items()]

    def tags(self, resource: str, expected_end: str) -> list[dict]:
        return [{"ResourceType": resource, "Tags": self.tag_list(expected_end)}]

    @staticmethod
    def ours() -> list[dict]:
        return [{"Name": "tag:Project", "Values": [PROJECT_TAG]},
                {"Name": "tag:Purpose", "Values": [PURPOSE]}]

    def retag(self, resource_id: str, expected_end: str) -> None:
        self.ec2.create_tags(Resources=[resource_id], Tags=self.tag_list(expected_end))

    def key_pair(self, expected_end: str) -> str:
        """Our key pair, recreated when its private key is not on this PC."""
        found = self.ec2.describe_key_pairs(Filters=[{"Name": "key-name", "Values": [NAME]},
                                                     *self.ours()])["KeyPairs"]
        if found and KEY.exists():
            self.retag(found[0]["KeyPairId"], expected_end)
            return NAME
        if found:
            self.ec2.delete_key_pair(KeyPairId=found[0]["KeyPairId"])
        key = self.ec2.create_key_pair(KeyName=NAME, KeyType="ed25519",
                                       TagSpecifications=self.tags("key-pair", expected_end))
        KEY.touch(mode=0o600)
        KEY.write_text(key["KeyMaterial"])
        return NAME

    def security_group(self, expected_end: str) -> str:
        """Our security group, admitting SSH from this PC's current public IP only."""
        found = self.ec2.describe_security_groups(Filters=[
            {"Name": "group-name", "Values": [NAME]}, *self.ours()])["SecurityGroups"]
        if found:
            group = found[0]
            self.retag(group["GroupId"], expected_end)
        else:
            vpc = self.ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
            group_id = self.ec2.create_security_group(
                GroupName=NAME, Description="nanocodec build box: SSH from one host",
                VpcId=vpc["Vpcs"][0]["VpcId"],
                TagSpecifications=self.tags("security-group", expected_end))["GroupId"]
            group = {"GroupId": group_id, "IpPermissions": []}
        source = f"{public_ip()}/32"
        stale = [p for p in group["IpPermissions"]
                 if [r["CidrIp"] for r in p.get("IpRanges", [])] != [source]]
        if stale:
            self.ec2.revoke_security_group_ingress(GroupId=group["GroupId"], IpPermissions=stale)
        if len(stale) == len(group["IpPermissions"]):
            self.ec2.authorize_security_group_ingress(GroupId=group["GroupId"], IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                "IpRanges": [{"CidrIp": source, "Description": "this host"}]}])
        return group["GroupId"]

    def image(self) -> tuple[str, str]:
        """The newest baked AMI, else the official Ubuntu one: (image id, root device)."""
        baked = self.ec2.describe_images(Owners=["self"], Filters=[
            *self.ours(), {"Name": "state", "Values": ["available"]}])["Images"]
        if baked:
            image = max(baked, key=lambda i: i["CreationDate"])
        else:
            name = AMI_PARAMETER.format(release=UBUNTU_RELEASE, arch="amd64")
            image_id = self.ssm.get_parameter(Name=name)["Parameter"]["Value"]
            image = self.ec2.describe_images(ImageIds=[image_id])["Images"][0]
        say(f"image {image['ImageId']} ({image.get('Name', '')})")
        return image["ImageId"], image["RootDeviceName"]

    def zone_prices(self, instance_type: str) -> dict[str, float]:
        history = self.ec2.describe_spot_price_history(
            InstanceTypes=[instance_type], ProductDescriptions=["Linux/UNIX"],
            StartTime=utc_now())["SpotPriceHistory"]
        prices = {h["AvailabilityZone"]: float(h["SpotPrice"]) for h in history}
        return dict(sorted(prices.items(), key=lambda item: item[1]))

    def on_demand_price(self, instance_type: str) -> float:
        """The Linux on-demand list price in REGION (AWS Price List API)."""
        import boto3
        pricing = boto3.client("pricing", region_name=PRICING_API_REGION)
        products = pricing.get_products(ServiceCode="AmazonEC2", Filters=[
            {"Type": "TERM_MATCH", "Field": field, "Value": value} for field, value in (
                ("instanceType", instance_type), ("regionCode", REGION),
                ("operatingSystem", "Linux"), ("tenancy", "Shared"),
                ("preInstalledSw", "NA"), ("capacitystatus", "Used"))])["PriceList"]
        for product in products:
            for term in json.loads(product)["terms"]["OnDemand"].values():
                for dimension in term["priceDimensions"].values():
                    return float(dimension["pricePerUnit"]["USD"])
        raise RuntimeError(f"no on-demand price for {instance_type} in {REGION}")

    def launch(self, market: str) -> Box:
        """One instance from an instant EC2 Fleet over every INSTANCE_TYPES x zone:
        spot with the price-capacity-optimized strategy (the pools least likely
        to be reclaimed, among the cheap ones), or on-demand at the lowest price."""
        expected_end = iso(utc_now() + datetime.timedelta(minutes=LIFETIME_MINUTES + 15))
        key, group = self.key_pair(expected_end), self.security_group(expected_end)
        image_id, root_device = self.image()
        user_data = (HERE / "nc_remote_box.sh").read_text() \
            .replace("@LIFETIME_MINUTES@", str(LIFETIME_MINUTES)) \
            .replace("@IDLE_MINUTES@", str(IDLE_MINUTES)).replace("@SLOTS@", str(SLOTS))
        template = self.ec2.create_launch_template(
            LaunchTemplateName=f"{NAME}-{utc_now():%Y%m%d-%H%M%S}",
            TagSpecifications=self.tags("launch-template", expected_end),
            LaunchTemplateData={
                "ImageId": image_id, "KeyName": key,
                "UserData": base64.b64encode(user_data.encode()).decode(),
                "InstanceInitiatedShutdownBehavior": "terminate",
                "BlockDeviceMappings": [{"DeviceName": root_device, "Ebs": {
                    "VolumeSize": ROOT_GB, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                "NetworkInterfaces": [{"DeviceIndex": 0, "Groups": [group],
                                       "AssociatePublicIpAddress": True}],
                "TagSpecifications": [*self.tags("instance", expected_end),
                                      *self.tags("volume", expected_end)]})["LaunchTemplate"]
        try:
            instance_id = self.create_fleet(template["LaunchTemplateId"], market, expected_end)
        finally:  # the instance keeps its settings; the template is not needed again
            self.ec2.delete_launch_template(LaunchTemplateId=template["LaunchTemplateId"])
        self.ec2.get_waiter("instance_running").wait(
            InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 120})
        described = self.describe(instance_id)
        instance_type = described["InstanceType"]
        zone = described["Placement"]["AvailabilityZone"]
        price = (self.zone_prices(instance_type)[zone] if market == SPOT
                 else self.on_demand_price(instance_type))
        say(f"launched {market} {instance_type} {instance_id} in {zone} (${price:.4f}/h)")
        return Box(instance_id=instance_id, instance_type=instance_type, zone=zone,
                   price=price, launched=iso(described["LaunchTime"]),
                   expected_end=expected_end, host=described["PublicIpAddress"], market=market)

    def create_fleet(self, template_id: str, market: str, expected_end: str) -> str:
        subnets = self.ec2.describe_subnets(Filters=[
            {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
        overrides = [{"InstanceType": t, "SubnetId": subnet["SubnetId"]}
                     for t in INSTANCE_TYPES for subnet in subnets]
        reply = self.ec2.create_fleet(
            Type="instant",
            SpotOptions={"AllocationStrategy": "price-capacity-optimized",
                         "InstanceInterruptionBehavior": "terminate"},
            OnDemandOptions={"AllocationStrategy": "lowest-price"},
            TargetCapacitySpecification={"TotalTargetCapacity": 1,
                                         "DefaultTargetCapacityType": market},
            LaunchTemplateConfigs=[{"LaunchTemplateSpecification": {
                "LaunchTemplateId": template_id, "Version": "$Latest"},
                "Overrides": overrides}],
            TagSpecifications=self.tags("fleet", expected_end))
        launched = [i for group in reply.get("Instances", []) for i in group["InstanceIds"]]
        if not launched:
            errors = {e.get("ErrorCode") for e in reply.get("Errors", [])}
            raise RuntimeError(f"no {market} capacity for any of {INSTANCE_TYPES} in {REGION}: "
                               f"{sorted(c for c in errors if c)}")
        return launched[0]

    def describe(self, instance_id: str) -> dict | None:
        reply = self.ec2.describe_instances(InstanceIds=[instance_id])
        found = [i for r in reply["Reservations"] for i in r["Instances"]]
        return found[0] if found else None

    def state(self, instance_id: str) -> str:
        from botocore.exceptions import ClientError
        try:
            described = self.describe(instance_id)
        except ClientError:
            return "gone"
        return described["State"]["Name"] if described else "gone"

    def reclaimed(self, instance_id: str) -> bool:
        """Whether spot reclaimed the instance (while AWS still describes it, ~1 h)."""
        try:
            described = self.describe(instance_id) or {}
        except Exception:  # noqa: BLE001 - gone from the API: not known as a reclaim
            return False
        return described.get("StateReason", {}).get("Code") == SPOT_RECLAIM_CODE

    def end_time(self, instance_id: str) -> datetime.datetime:
        """When a gone instance stopped, from its transition reason, else now."""
        try:
            reason = (self.describe(instance_id) or {}).get("StateTransitionReason", "")
        except Exception:  # noqa: BLE001 - the record is gone; "now" over-counts, safely
            reason = ""
        found = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", reason)
        if not found:
            return utc_now()
        return datetime.datetime.strptime(found.group(1), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=datetime.timezone.utc)

    def terminate(self, box: Box) -> None:
        described = self.describe(box.instance_id) or {}
        self.ec2.terminate_instances(InstanceIds=[box.instance_id])
        request = described.get("SpotInstanceRequestId")
        if request:
            self.ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=[request])
        self.ec2.get_waiter("instance_terminated").wait(
            InstanceIds=[box.instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 120})
        say(f"{box.instance_id} verified terminated")


# --- the box over SSH --------------------------------------------------------------------

def ssh_options(box: Box) -> list[str]:
    return ["-i", str(KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            "-o", f"UserKnownHostsFile={STATE_DIR / ('known_hosts-' + box.instance_id)}",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ControlMaster=auto",
            "-o", f"ControlPath={STATE_DIR / 'cm-%C'}", "-o", "ControlPersist=120"]


def ssh(box: Box, command: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *ssh_options(box), f"{SSH_USER}@{box.host}", command], **kwargs)


#: rsync modes. Results come back as they are (-a). Sources go up by content
#: (--checksum) WITHOUT their mtimes (-a minus -t, -o, -g): a changed file lands
#: with the box's current time, so cargo always sees it newer than the last
#: build. Preserving mtimes left a file edited during a remote build older than
#: that build's outputs, and the next run reused the stale binary (2026-11-16).
FETCH_MODE = ("-a",)
UPLOAD_MODE = ("-rlpD", "--checksum")


def rsync(box: Box, mode: tuple[str, ...], *args: str) -> None:
    shell = shlex.join(["ssh", *ssh_options(box)])
    subprocess.run(["rsync", *mode, "--mkpath", "-e", shell, *args], check=True)


def remote(box: Box, path: str) -> str:
    return f"{SSH_USER}@{box.host}:{path}"


def wait_for_ssh(box: Box) -> None:
    deadline = time.monotonic() + SSH_READY_SECONDS
    while time.monotonic() < deadline:
        if ssh(box, "test -f ncbox/activity", capture_output=True).returncode == 0:
            return
        time.sleep(5)
    raise RuntimeError(f"{box.instance_id}: SSH and the box setup not ready in "
                       f"{SSH_READY_SECONDS // 60} min")


def pc_toolchains() -> dict[str, str]:
    """The PC's Rust toolchains, which the box must match exactly: the stable
    version, and the nightly's commit (its dist date is the day after the
    commit date)."""
    def rustc(toolchain: str) -> dict[str, str]:
        out = subprocess.run([str(Path.home() / ".cargo/bin/rustc"), f"+{toolchain}", "-vV"],
                             capture_output=True, text=True, check=True).stdout
        return dict(line.split(": ", 1) for line in out.splitlines() if ": " in line)
    stable, nightly = rustc("stable"), rustc("nightly")
    commit_date = datetime.date.fromisoformat(nightly["commit-date"])
    return {"RUST_STABLE": stable["release"], "NIGHTLY_HASH": nightly["commit-hash"],
            "NIGHTLY_DATE": (commit_date + datetime.timedelta(days=1)).isoformat()}


def provision(box: Box) -> None:
    """Set the box up unless it already matches this provisioning version and
    the PC's toolchains (a toolchain upgrade on the PC re-provisions the box)."""
    toolchains = pc_toolchains()
    tag = f"{PROVISION_VERSION}-{toolchains['RUST_STABLE']}-{toolchains['NIGHTLY_HASH'][:9]}"
    if ssh(box, f"test -f ncbox/provisioned-{tag}", capture_output=True).returncode == 0:
        return
    say(f"provisioning (toolchains {toolchains['RUST_STABLE']} and nightly "
        f"{toolchains['NIGHTLY_DATE']}, as on the PC)")
    script = (HERE / "nc_remote_provision.sh").read_text().replace("@PROVISION_VERSION@", tag)
    for key, value in toolchains.items():
        script = script.replace(f"@{key}@", value)
    ssh(box, "bash -s", input=script.encode(), check=True)


# --- lifecycle ---------------------------------------------------------------------------

def close_out(aws: Aws, box: Box, ended: datetime.datetime) -> None:
    """Record a box that has gone: history (with whether spot reclaimed it),
    ledger row, state cleared."""
    box.ended = iso(ended)
    box.reclaimed = aws.reclaimed(box.instance_id)
    if box.reclaimed:
        say(f"{box.instance_id} was reclaimed by spot")
    record(box)
    epoch = parse("2000-01-01T00:00:00Z")
    cost = box.cost(epoch, ended)
    event = "ended: spot reclaim" if box.reclaimed else "ended"
    ledger_row(box.instance_id, event, box.launched, box.ended, f"${cost:.2f}", box.spec(),
               spent=cost)
    current = load_state()
    if current and current.instance_id == box.instance_id:
        save_state(None)
    say(f"{box.instance_id} ended {box.ended}: ${cost:.2f}")


def market_now() -> str:
    """On-demand while within RECLAIM_WINDOW of the later of two spot reclaims
    that came within RECLAIM_WINDOW of each other; else spot."""
    reclaims = sorted(parse(b.ended) for b in history() if b.reclaimed and b.ended)
    now = utc_now()
    for earlier, later in zip(reclaims, reclaims[1:]):
        if later - earlier <= RECLAIM_WINDOW and now - later <= RECLAIM_WINDOW:
            return ON_DEMAND
    return SPOT


def check_budget() -> None:
    spent = day_cost()
    say(f"build box today (UTC): ${spent:.2f} of ${DAILY_BUDGET_USD:.2f}")
    if spent >= DAILY_BUDGET_USD:
        raise SystemExit(f"nc-remote: today's estimate ${spent:.2f} has reached the "
                         f"${DAILY_BUDGET_USD:.2f}/day limit; run the work locally")


def unended() -> list[Box]:
    """Boxes not yet recorded as gone: the current one and any still draining."""
    return [b for b in history() if not b.ended]


def live_box(aws: Aws) -> Box | None:
    """The current box if it is still running. Every box that has gone (the
    current one or a draining one) is closed out."""
    for box in unended():
        if aws.state(box.instance_id) not in ("pending", "running"):
            close_out(aws, box, aws.end_time(box.instance_id))
    return load_state()


def drain_if_outdated(box: Box | None) -> Box | None:
    """A running box of a type no longer in INSTANCE_TYPES gets no new jobs: it
    finishes the ones it has and stops itself when idle (closed out by reap)."""
    if box is None or box.instance_type in INSTANCE_TYPES:
        return box
    say(f"{box.instance_type} {box.instance_id} is no longer a build box type: draining it "
        "(its jobs finish, then its idle watchdog stops it); starting a new box")
    save_state(None)
    return None


def ensure_box(aws: Aws) -> Box:
    with Locked():
        box = drain_if_outdated(live_box(aws))
        if box is None:
            check_budget()
            started = time.monotonic()
            market = market_now()
            if market == ON_DEMAND:
                say("two spot reclaims within 2 h today: on-demand until the end of the UTC day")
            box = aws.launch(market)
            save_state(box)
            record(box)
            ledger_row(f"{box.instance_id}, {NAME} key pair + security group", "started",
                       box.launched, "—", f"{box.market} ${box.price:.4f}/h, "
                       f"12 h at that price ${box.price * LIFETIME_MINUTES / 60:.2f}",
                       box.spec())
            wait_for_ssh(box)
            provision(box)
            say(f"box ready in {(time.monotonic() - started) / 60:.1f} min")
        elif day_cost() >= DAILY_BUDGET_USD:
            stop_all(aws)
            check_budget()
        else:
            provision(box)  # a no-op unless the PC's toolchains changed
        return box


def stop_all(aws: Aws) -> None:
    """Terminate the current box and every draining one."""
    for box in unended():
        stop_box(aws, box)


def stop_box(aws: Aws, box: Box) -> None:
    say(f"terminating {box.instance_id}")
    aws.terminate(box)
    close_out(aws, box, utc_now())


# --- commands ----------------------------------------------------------------------------

def checked_sync_path(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(str(resolved).startswith(p) for p in PRIVATE_PATHS):
        raise SystemExit(f"nc-remote: {resolved} holds private clips; never uploaded")
    if not resolved.is_relative_to(Path.home()):
        raise SystemExit(f"nc-remote: --sync paths must be under {Path.home()}")
    return resolved


def sync(box: Box, worktree: Path, name: str, extra: list[Path]) -> None:
    started = time.monotonic()
    rsync(box, UPLOAD_MODE, "-z", "--delete", *RSYNC_EXCLUDES, "--filter=:- .gitignore",
          f"{worktree}/", remote(box, f"ncbox/jobs/{name}/src/"))
    for path in extra:
        relative = path.relative_to(Path.home())
        suffix = "/" if path.is_dir() else ""
        rsync(box, UPLOAD_MODE, "-z", *RSYNC_EXCLUDES, f"{path}{suffix}",
              remote(box, f"{relative}{suffix}"))
    say(f"synced in {time.monotonic() - started:.0f} s")


def fetch(box: Box, name: str, paths: list[str], fetch_to: Path) -> None:
    for path in paths:
        if path.startswith("~/"):
            source, local = path[2:], Path.home() / path[2:]
        elif path == "target" or path.startswith("target/"):
            source = f"ncbox/jobs/{name}/{path}"
            local = fetch_to / path
        else:
            source, local = f"ncbox/jobs/{name}/src/{path}", fetch_to / path
        local.parent.mkdir(parents=True, exist_ok=True)
        rsync(box, FETCH_MODE, remote(box, source.rstrip("/")), f"{local.parent}/")
        say(f"fetched {path} -> {local}")


def job_command(command: list[str]) -> str:
    return command[0] if len(command) == 1 else shlex.join(command)


def run_once(aws: Aws, args, worktree: Path, name: str, commit: str) -> int:
    box = ensure_box(aws)
    sync(box, worktree, name, [checked_sync_path(p) for p in args.sync])
    job = (HERE / "nc_remote_job.py").read_text()
    argv = ["python3", "-c", job, name, commit, str(args.jobs), job_command(args.command),
            *args.env]
    started = time.monotonic()
    # stdin stays open (and unused) for the job's life: its EOF tells the box
    # that this caller has gone (nc_remote_job.py kills the job then).
    job_ssh = subprocess.Popen(["ssh", *ssh_options(box), f"{SSH_USER}@{box.host}",
                                shlex.join(argv)], stdin=subprocess.PIPE)
    status = job_ssh.wait()
    say(f"exit {status} after {time.monotonic() - started:.0f} s")
    if args.fetch:
        fetch(box, name, args.fetch, args.fetch_to or worktree)
    return status


def cmd_run(args) -> int:
    if not args.command:
        raise SystemExit("nc-remote run: no command (put it after --)")
    worktree = Path(args.worktree).expanduser().resolve()
    name = args.name or worktree.name
    commit = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    aws = Aws()
    status = run_once(aws, args, worktree, name, commit)
    box = load_state()
    if status == 255 and box and aws.state(box.instance_id) not in ("pending", "running"):
        say("the box went away during the job (spot reclaim or limit): restarting, retrying once")
        status = run_once(aws, args, worktree, name, commit)
    return status


def cmd_status(_args) -> int:
    aws = Aws()
    with Locked():
        current = live_box(aws)
    say(f"cost: today (UTC) ${day_cost():.2f} of ${DAILY_BUDGET_USD:.2f}/day; "
        f"all build boxes ${total_cost():.2f}; next launch: {market_now()}")
    for image in aws.ec2.describe_images(Owners=["self"], Filters=aws.ours())["Images"]:
        say(f"image {image['ImageId']} {image['State']} (baked {image['CreationDate']})")
    if not unended():
        say("no box running")
    job = (HERE / "nc_remote_job.py").read_text()
    for box in unended():
        role = "current" if current and current.instance_id == box.instance_id else "draining"
        hours = box.hours(parse(box.launched), utc_now())
        say(f"{role}: {box.instance_type} {box.instance_id} at {box.host}, up {hours * 60:.0f} "
            f"min, {box.market} ${box.price:.4f}/h, hard end {box.expected_end}")
        ssh(box, shlex.join(["python3", "-c", job, "--status"]))
        ssh(box, "echo \"  idle for $(( $(date +%s) - $(stat -c %Y ncbox/activity) )) s\"; "
                 "df -h --output=avail,size / | tail -1 | sed 's/^/  disk free, size: /'")
    return 0


def cmd_stop(args) -> int:
    """Terminate the box, but only between jobs: the box is shared, and on
    2026-11-18 one agent's "stop after my batch" killed three other agents'
    jobs. `--force` terminates regardless (owner or coordinator only)."""
    aws = Aws()
    with Locked():
        live_box(aws)
        boxes = unended()
        if not boxes:
            say("no box running")
        busy = [box for box in boxes if not args.force and not between_jobs(box)]
        if busy:
            for box in busy:
                say(f"{box.instance_id}: other jobs are running; left up (the idle check "
                    f"stops it when they finish; --force terminates now)")
            return 0
        stop_all(aws)
    return 0


def cmd_reap(_args) -> int:
    """The PC's check: close out gone boxes, enforce the budget and the hard end,
    run the idle check on every live box (the current one and draining ones)."""
    aws = Aws()
    with Locked():
        live_box(aws)
        prune_images(aws)
        if day_cost() >= DAILY_BUDGET_USD:
            stop_all(aws)
            return 0
        for box in unended():
            if utc_now() > parse(box.expected_end):
                stop_box(aws, box)
            elif box.market == ON_DEMAND and market_now() == SPOT and between_jobs(box):
                # On-demand costs ~4x spot: once spot is allowed again, end the
                # box between jobs; the next `run` relaunches it on spot.
                say(f"{box.instance_id}: on-demand while spot is allowed and no job runs")
                stop_box(aws, box)
            elif ssh(box, "sudo /usr/local/sbin/ncbox-idle",
                     capture_output=True).returncode != 0:
                note_unreachable(aws, box)
            else:
                unreachable_marker(box).unlink(missing_ok=True)
    return 0


#: A box whose SSH fails this many reaps in a row is hung (2026-11-19: one sat
#: unreachable for 20 min, its jobs lost and every `run` blocked behind it).
UNREACHABLE_REAPS = 3


def unreachable_marker(box: Box) -> Path:
    return STATE_DIR / f"unreachable-{box.instance_id}"


def note_unreachable(aws: Aws, box: Box) -> None:
    """Count consecutive failed reaps; terminate the box at UNREACHABLE_REAPS.
    Nothing can reach its jobs, so ending it costs nothing that is not lost."""
    marker = unreachable_marker(box)
    count = int(marker.read_text()) + 1 if marker.exists() else 1
    if count >= UNREACHABLE_REAPS:
        say(f"{box.instance_id}: unreachable over SSH for {count} reaps; terminating")
        marker.unlink(missing_ok=True)
        stop_box(aws, box)
    else:
        marker.write_text(str(count))
        say(f"{box.instance_id}: idle check over SSH failed ({count}/{UNREACHABLE_REAPS})")


def between_jobs(box: Box) -> bool:
    """No job slot is held and no job is waiting on the box."""
    probe = ("for s in ncbox/slots/*; do flock -n \"$s\" true || exit 1; done; "
             "! ls ncbox/jobs/*/waiting.json >/dev/null 2>&1")
    return ssh(box, probe, capture_output=True).returncode == 0


def prune_images(aws: Aws) -> None:
    """Remove our baked images (and snapshots) older than the newest available one.
    Run by `bake` and `reap`: a snapshot of a busy box takes 30-60 min, so `bake`
    does not wait for it."""
    images = aws.ec2.describe_images(Owners=["self"], Filters=aws.ours())["Images"]
    available = [i for i in images if i["State"] == "available"]
    if not available:
        return
    newest = max(i["CreationDate"] for i in available)
    for image in images:
        if image["CreationDate"] >= newest or image["State"] == "pending":
            continue
        aws.ec2.deregister_image(ImageId=image["ImageId"])
        for device in image.get("BlockDeviceMappings", []):
            snapshot = device.get("Ebs", {}).get("SnapshotId")
            if snapshot:
                aws.ec2.delete_snapshot(SnapshotId=snapshot)
        ledger_row(f"{image['ImageId']} (AMI + snapshot)", "removed", "—", iso(utc_now()),
                   "$0.00", SessionSpec(region=REGION, instance_type="—", purpose=PURPOSE))
        say(f"removed the older image {image['ImageId']}")


def cmd_bake(_args) -> int:
    """Start imaging the running box (toolchains, data, warm targets) for fast
    restarts. Launches use the newest available image; older ones are pruned
    once a newer one is available."""
    aws = Aws()
    box = load_state()
    if box is None:
        raise SystemExit("nc-remote bake: no box running")
    prune_images(aws)
    stamp = f"{utc_now():%Y%m%d-%H%M%S}"
    ssh(box, "sync", check=True)
    image_id = aws.ec2.create_image(
        InstanceId=box.instance_id, Name=f"{NAME}-{stamp}", NoReboot=True,
        Description="nanocodec build box (nc-remote)",
        TagSpecifications=[*aws.tags("image", AMI_EXPECTED_END),
                           *aws.tags("snapshot", AMI_EXPECTED_END)])["ImageId"]
    ledger_row(f"{image_id} (AMI + snapshot)", "baked", iso(utc_now()), AMI_EXPECTED_END,
               "snapshot storage", box.spec())
    say(f"baking {image_id}: available in 10-60 min (`nc-remote status` shows it)")
    return 0


def cmd_ssh(args) -> int:
    box = load_state()
    if box is None:
        raise SystemExit("nc-remote ssh: no box running")
    return ssh(box, job_command(args.command) if args.command else "bash -l").returncode


def main() -> int:
    parser = argparse.ArgumentParser(prog="nc-remote", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="cmd", required=True)
    run = commands.add_parser("run", help="run a command on the build box")
    run.add_argument("--worktree", default=".", help="the agent's worktree (default: cwd)")
    run.add_argument("--name", help="job directory on the box (default: the worktree's name)")
    run.add_argument("--sync", action="append", default=[], help="extra path under ~ to mirror")
    run.add_argument("--fetch", action="append", default=[],
                     help="path to bring back (worktree-relative, target/..., or ~/...)")
    run.add_argument("--fetch-to", type=Path, help="local root for --fetch (default: worktree)")
    run.add_argument("--env", action="append", default=[], help="KEY=VALUE for the command")
    run.add_argument("--jobs", type=int, default=JOBS_PER_SLOT, help="CARGO_BUILD_JOBS")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=cmd_run)
    stop = commands.add_parser("stop", help="terminate the box when no job runs on it")
    stop.add_argument("--force", action="store_true", help="terminate even with jobs running")
    stop.set_defaults(handler=cmd_stop)
    for name, handler, text in (("status", cmd_status, "box, jobs and cost"),
                                ("reap", cmd_reap, "the PC's periodic check (cron)"),
                                ("bake", cmd_bake, "image the box for fast restarts")):
        commands.add_parser(name, help=text).set_defaults(handler=handler)
    shell = commands.add_parser("ssh", help="a shell (or a command) on the box")
    shell.add_argument("command", nargs=argparse.REMAINDER)
    shell.set_defaults(handler=cmd_ssh)
    args = parser.parse_args()
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    return args.handler(args)


#: Exit status after Ctrl-C (the remote job is killed when our SSH link closes).
STATUS_INTERRUPTED = 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("interrupted: the remote job is killed with the SSH link")
        sys.exit(STATUS_INTERRUPTED)
