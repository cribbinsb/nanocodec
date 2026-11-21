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
  REGION (spot, price-capacity-optimized, without the types spot reclaimed
  lately; on-demand only after reclaims of several types or no spot capacity),
  from the newest baked AMI (`nc-remote bake`) or plain Ubuntu, provisioned
  by nc_remote_provision.sh on first boot.
- Its user-data (nc_remote_box.sh) sets the hard lifetime and an idle
  watchdog on the box itself: terminate after IDLE_MINUTES without a job or
  a sync, so nothing outlives the PC. `nc-remote reap` (run from cron on the
  PC and at the start of every command) runs the same idle check over SSH and
  writes the ledger row of a box that has gone (idle, lifetime or spot
  reclaim). `run` restarts a lost box and retries the command once.
- Replacement, make before break: when the current box should give way (it is
  on-demand while the rule says spot again, or it is within REPLACE_BEFORE_END
  of its hard lifetime) and jobs are on it, `reap` launches a new box and makes
  it current once it is ready; every later `run` goes there. The old box
  drains: no new jobs, its running ones (and their --fetch) finish as before,
  and `reap` ends it once no `run` uses it and no slot is held. A box with no
  job is simply ended (the next `run` launches its successor).
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
import contextlib
import datetime
import fcntl
import json
import os
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
from build_stamp import diff_identity  # noqa: E402

HERE = Path(__file__).resolve().parent
REGION = "eu-north-1"
#: Every region has its own spot quota: a spot launch refused in REGION (no capacity, or
#: the quota full) is tried in the next of these (coordinator, 2026-11-21). The baked
#: AMI is REGION's; elsewhere the box starts from Ubuntu and is provisioned.
SPILL_REGIONS = (REGION, "eu-west-1", "eu-central-1")
#: Spot quota full (EC2 error codes): never a reason for on-demand.
QUOTA_CODES = frozenset({"MaxSpotInstanceCountExceeded"})
#: No build box above this price per hour, spot or on-demand, without the coordinator's
#: say (2026-11-21: an on-demand c6i.24xlarge at $4.37/h started on a full quota, then
#: a spilled eu-west-1 spot c6i.24xlarge at $1.93/h). A fleet offers only the types
#: priced at or below it in its region (spot: the cheapest zone); with none, the next
#: region, and finally the current box is kept. On-demand only for the owner's rule
#: (RECLAIMED_TYPES_FOR_ON_DEMAND types reclaimed, or no spot capacity in any region).
MAX_USD_PER_HOUR = 1.50
#: The fleet's x86 types and their vCPUs (EC2 picks the type and zone). Each is
#: weighted by its vCPUs against a target of MIN_VCPUS, so the price-capacity-optimized
#: fleet compares price per vCPU: a 96-vCPU box wins whenever it costs no more per
#: vCPU (owner, 2026-11-21: c7a.24xlarge spot $0.558/h against c7a.16xlarge $0.559/h).
#: eu-north-1 because its spot prices were a quarter of eu-west-1's
#: (2026-11-16: c7a.8xlarge $0.23/h against $0.89/h).
#: 96 vCPU only since 2026-11-21 (coordinator: the 64-vCPU box's 8 slots queued verify,
#: identity, callgrind and agents' jobs); a running 64-vCPU box drains (make before break).
TYPE_VCPUS = {"c7a.24xlarge": 96, "c7i.24xlarge": 96, "c6i.24xlarge": 96,
              "m7a.24xlarge": 96, "m7i.24xlarge": 96, "m6i.24xlarge": 96}
INSTANCE_TYPES = tuple(TYPE_VCPUS)
MIN_VCPUS = 96
#: One heavy job per 8 vCPU (the box makes nproc / JOBS_PER_SLOT slots); each job gets
#: cargo -j JOBS_PER_SLOT.
JOBS_PER_SLOT = 8
#: Conformance suites (~6 GB), anchors and up to ~20 agents' targets.
ROOT_GB = 200
LIFETIME_MINUTES = 12 * 60
#: The pool this process manages: "" the build box, OVERFLOW the second box that `run`
#: starts when every slot of the first has been busy for OVERFLOW_AFTER_SECONDS
#: (coordinator, 2026-11-21); each has its own state, key pair, history and idle limit.
POOL_ENV = "NC_REMOTE_POOL"
OVERFLOW = "overflow"
POOL = os.environ.get(POOL_ENV, "")
OVERFLOW_AFTER_SECONDS = 120
OVERFLOW_POLL_SECONDS = 10
IDLE_MINUTES = 20 if POOL == OVERFLOW else 10
#: A busy box this close to its hard lifetime is replaced (make before break),
#: so jobs started from then on do not die with it at the hard end; the jobs
#: already on it have this long to finish.
REPLACE_BEFORE_END = datetime.timedelta(hours=1)
#: Room for an on-demand day (RECLAIM_WINDOW): ~$2.9/h for up to 12 h.
DAILY_BUDGET_USD = 40.0
#: Markets. Spot by default. A type that spot reclaimed within RECLAIM_WINDOW is
#: left out of the next spot fleets, so the box moves to another type's pools
#: (owner, 2026-11-20: reclaims come in runs per type -- c6i at 12:04, 12:33 and
#: 13:03 on 2026-11-16, c7a in two zones at 07:51 and 09:36 on 2026-11-20 --
#: while other types' spot stayed cheap; the old "two reclaims of any type"
#: rule then paid on-demand at ~5x the spot price). The whole type is left
#: out, not one zone: the 2026-11-20 pair hit c7a in eu-north-1b and 1c.
#: On-demand only when RECLAIMED_TYPES_FOR_ON_DEMAND distinct types were
#: reclaimed within the window, or when no remaining spot pool has capacity
#: (owner, 2026-11-17: never pay on-demand longer than needed).
SPOT, ON_DEMAND = "spot", "on-demand"
RECLAIM_WINDOW = datetime.timedelta(hours=2)
RECLAIMED_TYPES_FOR_ON_DEMAND = 3
WINDOW_TEXT = f"{RECLAIM_WINDOW.total_seconds() / 3600:g} h"
SPOT_RECLAIM_CODE = "Server.SpotInstanceTermination"
#: The Price List API is served from us-east-1 (prices are still REGION's).
PRICING_API_REGION = "us-east-1"
PURPOSE = "buildbox"
NAME = "nanocodec-buildbox" + (f"-{POOL}" if POOL else "")
#: Bump when nc_remote_provision.sh changes, so boxes re-provision.
PROVISION_VERSION = "6"
#: What the browser checks need, pinned to the PC's (tools/web/build.sh, Cargo.lock):
#: substituted into nc_remote_provision.sh.
BROWSER_TOOLS = {"WASM_BINDGEN_VERSION": "0.2.105",
                 "WASM_BINDGEN_SHA256":
                     "b391448c4926ac4b11425a6752484d85164e72489d97804461d5e868c643b88a",
                 "PLAYWRIGHT_VERSION": "1.60.0",
                 # Playwright 1.60 refuses Ubuntu 26.04; its 24.04 Chromium runs there
                 # with six apt libraries (proved by the page tests, 2026-11-21).
                 "PLAYWRIGHT_PLATFORM": "ubuntu24.04-x64"}
#: AMIs and their snapshots have no lifetime of their own; their ExpectedEnd is reviewed then.
AMI_EXPECTED_END = "2026-12-31T00:00:00Z"

STATE_DIR = WORK_DIR / ("buildbox" + (f"-{POOL}" if POOL else ""))
STATE = STATE_DIR / "state.json"
HISTORY = STATE_DIR / "history.json"
LOCK = STATE_DIR / "lock"
#: Held (flock) while a replacement box starts; holds that box's instance id.
REPLACING = STATE_DIR / "replacing"
#: One flock'd file per `run` and the box it uses (BoxUse).
USERS_DIR = STATE_DIR / "users"
KEY = STATE_DIR / "key.pem"
SSH_USER = "ubuntu"
BOX_HOME = f"/home/{SSH_USER}"
SSH_READY_SECONDS = 10 * 60
#: Paths never synced by --sync: the private clips and their copies. Clips the
#: job needs go up with --clip instead (owner, 2026-11-20: clips may go to our AWS
#: boxes and are deleted there afterwards): the box deletes CLIP_ROOT before it
#: shuts itself down (nc_remote_box.sh), and `stop` before it terminates the box.
PRIVATE_PATHS = ("/mldata", str(Path.home() / "nanocodec-data/clips"),
                 str(WORK_DIR / "cache"))
CLIP_ROOT = Path.home() / "nanocodec-data/clips"
#: Clips keep their size and time (data, not build inputs), so one already there
#: is not sent again.
CLIP_MODE = ("-a",)
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
    region: str = REGION

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
        return SessionSpec(region=self.region, instance_type=self.instance_type, purpose=PURPOSE)


@dataclass(frozen=True)
class MarketPlan:
    """The next launch: its market, the instance types its fleet offers, and why."""
    market: str
    types: tuple[str, ...]
    reason: str


class NoCapacity(RuntimeError):
    """A fleet request launched nothing: no capacity in any pool it offered, or
    (`quota`) the region's spot quota is full."""

    def __init__(self, message: str, quota: bool = False):
        super().__init__(message)
        self.quota = quota


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


def flock_held(path: Path) -> bool:
    """Whether another open file holds an exclusive flock on `path`."""
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


class BoxUse:
    """Held by a `run` from the moment it is given its box (under Locked) until
    its --fetch has come back, so a draining box is never ended under a
    caller: not while its job runs, and not between the job and the fetch."""

    def __init__(self, box: Box):
        self.box = box
        USERS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = USERS_DIR / f"{box.instance_id}.{os.getpid()}"
        self.handle = open(self.path, "w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)

    def __enter__(self) -> Box:
        return self.box

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        self.handle.close()


def box_in_use(box: Box) -> bool:
    """Whether a `run` on this PC still uses `box`; files of runs that have gone
    (their flock died with them) are removed."""
    in_use = False
    for path in USERS_DIR.glob(f"{box.instance_id}.*"):
        if flock_held(path):
            in_use = True
        else:
            path.unlink(missing_ok=True)
    return in_use


class Replacing:
    """Held while a replacement box starts (one at a time, never blocking: a
    second reap finds it held and leaves the replacement to the first). The
    file names the new box once it is launched, so reap does not take that box
    for a draining one."""

    def __enter__(self) -> Replacing | None:
        """Self, or None when another replacement holds the lock."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.handle = open(REPLACING, "a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            return None
        self.name("")
        return self

    def name(self, instance_id: str) -> None:
        self.handle.truncate(0)
        self.handle.write(instance_id)
        self.handle.flush()

    def __exit__(self, *exc):
        if self.handle:
            self.name("")
            self.handle.close()


def starting_id() -> str:
    """The instance id of a replacement box that is still starting, else ""."""
    if not REPLACING.exists() or not flock_held(REPLACING):
        return ""
    return REPLACING.read_text().strip()


# --- AWS ---------------------------------------------------------------------------------

class Aws:
    """The calls nc-remote makes; every resource it creates or changes is tagged ours."""

    def __init__(self, region: str = REGION):
        import boto3
        self.region = region
        self.ec2 = boto3.client("ec2", region_name=region)
        self.ssm = boto3.client("ssm", region_name=region)
        self._regional = {region: self}

    def regional(self, region: str) -> "Aws":
        """The calls for `region` (a box's), one client set per region."""
        if region not in self._regional:
            self._regional[region] = Aws(region)
        return self._regional[region]

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
        key_file = key_path(self.region)
        if found and key_file.exists():
            self.retag(found[0]["KeyPairId"], expected_end)
            return NAME
        if found:
            self.ec2.delete_key_pair(KeyPairId=found[0]["KeyPairId"])
        key = self.ec2.create_key_pair(KeyName=NAME, KeyType="ed25519",
                                       TagSpecifications=self.tags("key-pair", expected_end))
        key_file.touch(mode=0o600)
        key_file.write_text(key["KeyMaterial"])
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
        """The Linux on-demand list price in this region (AWS Price List API)."""
        import boto3
        pricing = boto3.client("pricing", region_name=PRICING_API_REGION)
        products = pricing.get_products(ServiceCode="AmazonEC2", Filters=[
            {"Type": "TERM_MATCH", "Field": field, "Value": value} for field, value in (
                ("instanceType", instance_type), ("regionCode", self.region),
                ("operatingSystem", "Linux"), ("tenancy", "Shared"),
                ("preInstalledSw", "NA"), ("capacitystatus", "Used"))])["PriceList"]
        for product in products:
            for term in json.loads(product)["terms"]["OnDemand"].values():
                for dimension in term["priceDimensions"].values():
                    return float(dimension["pricePerUnit"]["USD"])
        raise RuntimeError(f"no on-demand price for {instance_type} in {self.region}")

    def launch(self, plan: MarketPlan, on_demand_fallback: bool = True) -> Box:
        """One instance from an instant EC2 Fleet over the plan's types x every
        zone: spot with the price-capacity-optimized strategy (the pools least
        likely to be reclaimed, among the cheap ones), or on-demand at the
        lowest price. A spot fleet with no capacity in any of its pools falls
        back to on-demand when `on_demand_fallback`, else raises NoCapacity."""
        expected_end = iso(utc_now() + datetime.timedelta(minutes=LIFETIME_MINUTES + 15))
        key, group = self.key_pair(expected_end), self.security_group(expected_end)
        image_id, root_device = self.image()
        user_data = (HERE / "nc_remote_box.sh").read_text() \
            .replace("@LIFETIME_MINUTES@", str(LIFETIME_MINUTES)) \
            .replace("@IDLE_MINUTES@", str(IDLE_MINUTES)).replace("@JOBS_PER_SLOT@", str(JOBS_PER_SLOT))
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
            instance_id, market = self.fleet_instance(template["LaunchTemplateId"], plan,
                                                      expected_end, on_demand_fallback)
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
                   expected_end=expected_end, host=described["PublicIpAddress"], market=market,
                   region=self.region)

    def fleet_instance(self, template_id: str, plan: MarketPlan, expected_end: str,
                       on_demand_fallback: bool = True) -> tuple[str, str]:
        """(instance id, market) from the plan's fleet, or from an on-demand
        fleet over every type when no spot pool in the plan has capacity (and
        `on_demand_fallback`; never when the spot quota is what refused)."""
        try:
            return self.create_fleet(template_id, plan.market, plan.types, expected_end), \
                plan.market
        except NoCapacity as missing:
            if plan.market != SPOT or not on_demand_fallback or missing.quota:
                raise
            say(f"{missing}: on-demand instead")
            return self.create_fleet(template_id, ON_DEMAND, INSTANCE_TYPES, expected_end), \
                ON_DEMAND

    def create_fleet(self, template_id: str, market: str, types: tuple[str, ...],
                     expected_end: str) -> str:
        types = self.affordable(market, types)
        subnets = self.ec2.describe_subnets(Filters=[
            {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
        # Spot: only the pools (type x zone) priced within the cap, since the fleet's
        # price-capacity-optimized choice may take a pricier zone (2026-11-21: m6i.24xlarge
        # at $1.96/h in eu-central-1 beside zones within $1.50/h).
        within = {t: ({z for z, price in self.zone_prices(t).items()
                       if price <= MAX_USD_PER_HOUR} if market == SPOT else None)
                  for t in types}
        overrides = [{"InstanceType": t, "SubnetId": subnet["SubnetId"],
                      "WeightedCapacity": float(TYPE_VCPUS[t])}
                     for t in types for subnet in subnets
                     if within[t] is None or subnet.get("AvailabilityZone") in within[t]]
        if not overrides:
            raise NoCapacity(f"no {market} pool within ${MAX_USD_PER_HOUR:.2f}/h in "
                             f"{self.region} among {', '.join(types)}")
        cap = f"{MAX_USD_PER_HOUR:.4f}"  # the fleet's one instance, whatever its pool
        reply = self.ec2.create_fleet(
            Type="instant",
            SpotOptions={"AllocationStrategy": "price-capacity-optimized",
                         "InstanceInterruptionBehavior": "terminate", "MaxTotalPrice": cap},
            OnDemandOptions={"AllocationStrategy": "lowest-price", "MaxTotalPrice": cap},
            TargetCapacitySpecification={"TotalTargetCapacity": MIN_VCPUS,
                                         "DefaultTargetCapacityType": market},
            LaunchTemplateConfigs=[{"LaunchTemplateSpecification": {
                "LaunchTemplateId": template_id, "Version": "$Latest"},
                "Overrides": overrides}],
            TagSpecifications=self.tags("fleet", expected_end))
        launched = [i for group in reply.get("Instances", []) for i in group["InstanceIds"]]
        if not launched:
            errors = {e.get("ErrorCode") for e in reply.get("Errors", [])}
            raise NoCapacity(f"no {market} capacity for any of {', '.join(types)} in {self.region} "
                             f"({', '.join(sorted(c for c in errors if c)) or 'no error code'})",
                             quota=bool(errors & QUOTA_CODES))
        return launched[0]

    def affordable(self, market: str, types: tuple[str, ...]) -> tuple[str, ...]:
        """The types priced within MAX_USD_PER_HOUR here (spot: the cheapest zone)."""
        def price(instance_type: str) -> float:
            if market == ON_DEMAND:
                return self.on_demand_price(instance_type)
            zones = self.zone_prices(instance_type)
            return min(zones.values()) if zones else float("inf")
        cheap = tuple(t for t in types if price(t) <= MAX_USD_PER_HOUR)
        if not cheap:
            raise NoCapacity(f"no {market} type within ${MAX_USD_PER_HOUR:.2f}/h in "
                             f"{self.region} among {', '.join(types)}")
        return cheap

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

def key_path(region: str) -> Path:
    """This pool's private key for `region` (a key pair exists per region)."""
    return KEY if region == REGION else STATE_DIR / f"key-{region}.pem"


def ssh_options(box: Box) -> list[str]:
    return ["-i", str(key_path(box.region)), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
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
    for key, value in {**toolchains, **BROWSER_TOOLS}.items():
        script = script.replace(f"@{key}@", value)
    ssh(box, "bash -s", input=script.encode(), check=True)


# --- lifecycle ---------------------------------------------------------------------------

def close_out(aws: Aws, box: Box, ended: datetime.datetime) -> None:
    """Record a box that has gone: history (with whether spot reclaimed it),
    ledger row, state cleared."""
    box.ended = iso(ended)
    box.reclaimed = aws.regional(box.region).reclaimed(box.instance_id)
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


def reclaimed_types(boxes: list[Box], now: datetime.datetime) -> set[str]:
    """The instance types spot reclaimed within RECLAIM_WINDOW before `now`."""
    return {b.instance_type for b in boxes
            if b.reclaimed and b.ended and now - parse(b.ended) <= RECLAIM_WINDOW}


def plan_market(boxes: list[Box], now: datetime.datetime) -> MarketPlan:
    """The next launch's market and fleet types, given the box history."""
    excluded = reclaimed_types(boxes, now)
    if len(excluded) >= RECLAIMED_TYPES_FOR_ON_DEMAND:
        return MarketPlan(ON_DEMAND, INSTANCE_TYPES,
                          f"on-demand: spot reclaimed {len(excluded)} types within "
                          f"{WINDOW_TEXT} ({', '.join(sorted(excluded))})")
    types = tuple(t for t in INSTANCE_TYPES if t not in excluded)
    reason = "spot"
    if excluded:
        reason += f" without {', '.join(sorted(excluded))} (reclaimed within {WINDOW_TEXT})"
    return MarketPlan(SPOT, types, reason)


def market_now() -> MarketPlan:
    return plan_market(history(), utc_now())


def check_budget() -> None:
    spent = day_cost()
    say(f"build box today (UTC): ${spent:.2f} of ${DAILY_BUDGET_USD:.2f}")
    if spent >= DAILY_BUDGET_USD:
        raise SystemExit(f"nc-remote: today's estimate ${spent:.2f} has reached the "
                         f"${DAILY_BUDGET_USD:.2f}/day limit; run the work locally")


def unended() -> list[Box]:
    """Boxes not yet recorded as gone: the current one, a replacement that is
    starting, and any still draining."""
    return [b for b in history() if not b.ended]


#: What `status` adds after each role.
ROLE_NOTES = {"current": "", "starting": " (replacement, current once ready)",
              "draining": " (no new jobs; reap ends it when its jobs and fetches are done)"}


def role(box: Box, current: Box | None) -> str:
    if current and current.instance_id == box.instance_id:
        return "current"
    return "starting" if box.instance_id == starting_id() else "draining"


def live_box(aws: Aws) -> Box | None:
    """The current box if it is still running. Every box that has gone (the
    current one or a draining one) is closed out."""
    for box in unended():
        regional = aws.regional(box.region)
        if regional.state(box.instance_id) not in ("pending", "running"):
            close_out(aws, box, regional.end_time(box.instance_id))
    return load_state()


def launch_anywhere(aws: Aws, plan: MarketPlan, on_demand_fallback: bool) -> Box:
    """A spot box in the first of SPILL_REGIONS that launches one (quota full or no
    capacity: the next region); on-demand only after every region refused spot for
    capacity (not quota), when `on_demand_fallback` (and within the price cap)."""
    if plan.market != SPOT:
        return aws.launch(plan, on_demand_fallback=False)
    refusals: list[NoCapacity] = []
    for region in SPILL_REGIONS:
        try:
            box = aws.regional(region).launch(plan, on_demand_fallback=False)
            if region != REGION:
                say(f"launched in {region} (spilled from {REGION}: "
                    f"{'; '.join(map(str, refusals))})")
            return box
        except NoCapacity as missing:
            say(f"{region}: {missing}")
            refusals.append(missing)
    # On-demand only when some region had quota room but no spot capacity.
    every_quota = all(r.quota for r in refusals)
    if on_demand_fallback and not every_quota:
        return aws.launch(MarketPlan(ON_DEMAND, INSTANCE_TYPES, "on-demand: no spot capacity "
                                     "in any region"), on_demand_fallback=False)
    raise NoCapacity("no spot box in any of " + ", ".join(SPILL_REGIONS) + ": "
                     + "; ".join(map(str, refusals)), quota=every_quota)


def start_box(aws: Aws, plan: MarketPlan, guard, on_demand_fallback: bool = True,
              on_launch=None) -> Box:
    """Launch a box, record it (history, ledger), wait for SSH and provision it.
    Writes to the PC records happen inside `guard()` (Locked, unless the caller
    already holds it). A box that launched but never became ready is ended."""
    started = time.monotonic()
    say(f"launching {plan.reason}")
    box = launch_anywhere(aws, plan, on_demand_fallback)
    with guard():
        record(box)
    if on_launch:
        on_launch(box)
    ledger_row(f"{box.instance_id}, {NAME} key pair + security group", "started",
               box.launched, "—", f"{box.market} ${box.price:.4f}/h, "
               f"12 h at that price ${box.price * LIFETIME_MINUTES / 60:.2f}", box.spec())
    try:
        wait_for_ssh(box)
        provision(box)
    except BaseException:
        with guard():
            stop_box(aws, box)
        raise
    say(f"box ready in {(time.monotonic() - started) / 60:.1f} min")
    return box


def ensure_box(aws: Aws) -> BoxUse:
    """The current box (started if there is none), held for one `run`."""
    with Locked():
        box = live_box(aws)
        if box is None:
            check_budget()
            box = start_box(aws, market_now(), contextlib.nullcontext)
            save_state(box)
        elif day_cost() >= DAILY_BUDGET_USD:
            stop_all(aws)
            check_budget()
        else:
            provision(box)  # a no-op unless the PC's toolchains changed
        return BoxUse(box)


def ends_soon(box: Box, now: datetime.datetime) -> bool:
    """Whether the box is within REPLACE_BEFORE_END of its hard lifetime."""
    hard_end = parse(box.launched) + datetime.timedelta(minutes=LIFETIME_MINUTES)
    return now >= hard_end - REPLACE_BEFORE_END


def replacement_reason(box: Box, plan: MarketPlan, now: datetime.datetime) -> str:
    """Why the current box should give way to a new one, or "" if it should not."""
    if ends_soon(box, now):
        return f"within {REPLACE_BEFORE_END.total_seconds() / 3600:g} h of its hard lifetime"
    if box.market == ON_DEMAND and plan.market == SPOT:
        return "on-demand while spot is allowed"
    if box.instance_type not in INSTANCE_TYPES:
        # A type no longer listed (the 64-vCPU boxes, 2026-11-21) keeps serving until a
        # listed type has started on spot: make before break, never on-demand for this.
        return f"{box.instance_type} is no longer a build box type"
    return ""


def unused(box: Box) -> bool:
    """No `run` on this PC uses the box and none of its slots is held or waited
    for: ending it now loses nothing."""
    return not box_in_use(box) and between_jobs(box)


def replace(aws: Aws, old: Box, reason: str) -> None:
    """Make before break: start a new box and make it current once it is ready;
    `old` then drains (reap ends it once it is unused). Runs given `old` before
    the switch keep it to the end of their fetch. Leaving on-demand only ever
    lands on spot (no spot capacity: the old box stays current); a box near its
    hard end is replaced on whatever market the rule gives."""
    with Replacing() as replacing:
        if replacing is None:
            say("a replacement box is already starting")
            return
        if day_cost() >= DAILY_BUDGET_USD:
            say(f"{old.instance_id}: {reason}, but today's budget is spent: not replaced")
            return
        plan = market_now()
        if not replacement_reason(old, plan, utc_now()):
            return  # the rule changed meanwhile
        say(f"{old.instance_id}: {reason} and jobs are on it: starting its replacement "
            "(make before break)")
        try:
            new = start_box(aws, plan, Locked, on_demand_fallback=ends_soon(old, utc_now()),
                            on_launch=lambda box: replacing.name(box.instance_id))
        except NoCapacity as missing:
            say(f"{missing}: {old.instance_id} stays the current box")
            return
        with Locked():
            live_box(aws)
            save_state(new)
            say(f"{new.instance_id} is now the current box; {old.instance_id} drains (no new "
                "jobs, its running jobs finish; reap ends it once none is left)")
            if old.instance_id in {b.instance_id for b in unended()} and unused(old):
                say(f"{old.instance_id}: drained")
                stop_box(aws, old)


def stop_all(aws: Aws) -> None:
    """Terminate the current box and every draining one."""
    for box in unended():
        stop_box(aws, box)


def stop_box(aws: Aws, box: Box) -> None:
    # The clips leave the box before it is terminated (owner, 2026-11-20).
    with contextlib.suppress(Exception):
        ssh(box, f"rm -rf -- {BOX_HOME}/{CLIP_ROOT.relative_to(Path.home())}", check=False,
            capture_output=True, timeout=60)
    say(f"terminating {box.instance_id}")
    aws.regional(box.region).terminate(box)
    close_out(aws, box, utc_now())


# --- commands ----------------------------------------------------------------------------

def checked_sync_path(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(str(resolved).startswith(p) for p in PRIVATE_PATHS):
        raise SystemExit(f"nc-remote: {resolved} holds private clips; never uploaded")
    if not resolved.is_relative_to(Path.home()):
        raise SystemExit(f"nc-remote: --sync paths must be under {Path.home()}")
    return resolved


def checked_clip_path(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(CLIP_ROOT) or not resolved.is_file():
        raise SystemExit(f"nc-remote: --clip takes a clip file under {CLIP_ROOT}: {path}")
    return resolved


def sync(box: Box, worktree: Path, name: str, extra: list[Path],
         clips: tuple[Path, ...] = ()) -> None:
    started = time.monotonic()
    rsync(box, UPLOAD_MODE, "-z", "--delete", *RSYNC_EXCLUDES, "--filter=:- .gitignore",
          f"{worktree}/", remote(box, f"ncbox/jobs/{name}/src/"))
    for path in extra:
        relative = path.relative_to(Path.home())
        suffix = "/" if path.is_dir() else ""
        rsync(box, UPLOAD_MODE, "-z", *RSYNC_EXCLUDES, f"{path}{suffix}",
              remote(box, f"{relative}{suffix}"))
    for clip in clips:
        rsync(box, CLIP_MODE, str(clip), remote(box, str(clip.relative_to(Path.home()))))
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
        if ssh(box, f"test -e {shlex.quote(source.rstrip('/'))}", capture_output=True).returncode:
            say(f"not fetched: {path} does not exist on the box (the job did not make it)")
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        rsync(box, FETCH_MODE, remote(box, source.rstrip("/")), f"{local.parent}/")
        say(f"fetched {path} -> {local}")


#: The PC's record of each box job's worktree (job name -> path), so a job whose
#: worktree is gone has its directory removed from the box (note_job).
JOB_TREES = WORK_DIR / "nc-remote-jobs.json"


def note_job(name: str, worktree: Path) -> list[str]:
    """Record `name`'s worktree; return (and forget) the recorded jobs whose worktree
    no longer exists."""
    JOB_TREES.parent.mkdir(parents=True, exist_ok=True)
    with open(JOB_TREES.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        trees = json.loads(JOB_TREES.read_text()) if JOB_TREES.exists() else {}
        trees[name] = str(worktree)
        gone = sorted(n for n, tree in trees.items() if not Path(tree).is_dir())
        for n in gone:
            del trees[n]
        partial = JOB_TREES.with_suffix(".partial")
        partial.write_text(json.dumps(trees, indent=1, sort_keys=True) + "\n")
        partial.rename(JOB_TREES)
    return gone


def job_command(command: list[str]) -> str:
    return command[0] if len(command) == 1 else shlex.join(command)


def run_once(aws: Aws, args, worktree: Path, name: str, commit: str,
             dirty: str) -> tuple[int, Box]:
    """Run the job on the current box (it stays this job's box to the end of
    its fetch, even if a replacement becomes current meanwhile)."""
    with ensure_box(aws) as box:
        return run_on(box, args, worktree, name, commit, dirty), box


def run_on(box: Box, args, worktree: Path, name: str, commit: str, dirty: str) -> int:
    job = (HERE / "nc_remote_job.py").read_text()
    gone = note_job(name, worktree)
    if gone:  # their directories on the box go too (targets: GBs each)
        ssh(box, shlex.join(["python3", "-c", job, "--drop", *gone]))
    sync(box, worktree, name, [checked_sync_path(p) for p in args.sync],
         tuple(checked_clip_path(p) for p in getattr(args, "clip", [])))
    # NC_BUILD_DIRTY stamps a build of local changes as such (build_stamp.py);
    # always set, so an empty value marks a clean tree.
    argv = ["python3", "-c", job, name, commit, str(args.jobs), job_command(args.command),
            f"NC_BUILD_DIRTY={dirty}", *args.env]
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
    dirty = diff_identity(worktree) if commit != "unknown" else ""
    if dirty:
        say(f"local changes: builds are stamped {commit[:12]}-dirty.{dirty}")
    aws = Aws()
    if not POOL and not name.startswith("verifier-") and main_box_full(aws):
        say(f"every slot of the build box has been busy for {OVERFLOW_AFTER_SECONDS} s: "
            f"running on the overflow box")
        return subprocess.call([sys.executable, __file__, *sys.argv[1:]],
                               env={**os.environ, POOL_ENV: OVERFLOW})
    status, box = run_once(aws, args, worktree, name, commit, dirty)
    if status == 255 and aws.regional(box.region).state(box.instance_id) not in ("pending",
                                                                                "running"):
        say("the box went away during the job (spot reclaim or limit): restarting, retrying once")
        status, _ = run_once(aws, args, worktree, name, commit, dirty)
    return status


def main_box_full(aws: Aws) -> bool:
    """Whether every job slot of the build box (verify's kept ones aside) stays busy
    for OVERFLOW_AFTER_SECONDS (a job already waiting that long counts at once)."""
    with Locked():
        box = live_box(aws)
    if box is None:
        return False
    job = (HERE / "nc_remote_job.py").read_text()
    deadline = time.monotonic() + OVERFLOW_AFTER_SECONDS
    while True:
        try:
            out = ssh(box, shlex.join(["python3", "-c", job, "--free"]), capture_output=True,
                      text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False
        if out.returncode != 0:
            return False  # an older box script or no answer: queue there as before
        free, waited = (int(x) for x in out.stdout.split()[:2])
        if free > 0:
            return False
        if waited >= OVERFLOW_AFTER_SECONDS or time.monotonic() > deadline:
            return True
        time.sleep(OVERFLOW_POLL_SECONDS)


def cmd_status(_args) -> int:
    aws = Aws()
    with Locked():
        current = live_box(aws)
    say(f"cost: today (UTC) ${day_cost():.2f} of ${DAILY_BUDGET_USD:.2f}/day; "
        f"all build boxes ${total_cost():.2f}; next launch: {market_now().reason}")
    for image in aws.ec2.describe_images(Owners=["self"], Filters=aws.ours())["Images"]:
        say(f"image {image['ImageId']} {image['State']} (baked {image['CreationDate']})")
    if not unended():
        say("no box running")
    job = (HERE / "nc_remote_job.py").read_text()
    for box in unended():
        hours = box.hours(parse(box.launched), utc_now())
        say(f"{role(box, current)}: {box.instance_type} {box.instance_id} at {box.host}, "
            f"up {hours * 60:.0f} min, {box.market} ${box.price:.4f}/h, "
            f"hard end {box.expected_end}{ROLE_NOTES[role(box, current)]}")
        ssh(box, shlex.join(["python3", "-c", job, "--status"]))
        ssh(box, "echo \"  idle for $(( $(date +%s) - $(stat -c %Y ncbox/activity) )) s\"; "
                 "df -h --output=avail,size / | tail -1 | sed 's/^/  disk free, size: /'")
    if not POOL:
        say("overflow box:")
        subprocess.call([sys.executable, __file__, "status"], env={**os.environ, POOL_ENV: OVERFLOW})
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
        busy = [box for box in boxes if not args.force and not unused(box)]
        if busy:
            for box in busy:
                say(f"{box.instance_id}: other jobs are running; left up (the idle check "
                    f"stops it when they finish; --force terminates now)")
            return 0
        stop_all(aws)
    return 0


def cmd_reap(_args) -> int:
    """The PC's check: close out gone boxes, enforce the budget and the hard end,
    end drained boxes, replace the current box when it should give way (make
    before break), and run the idle check on every live box."""
    aws = Aws()
    replacement = None
    with Locked():
        current = live_box(aws)
        prune_images(aws)
        if day_cost() >= DAILY_BUDGET_USD:
            stop_all(aws)
            return 0
        for box in unended():
            reason = ""
            if role(box, current) == "starting":
                continue  # its replacement run waits for it and handles its failure
            if utc_now() > parse(box.expected_end):
                stop_box(aws, box)
                continue
            if role(box, current) == "draining" and unused(box):
                say(f"{box.instance_id}: drained")
                stop_box(aws, box)
                continue
            if role(box, current) == "current":
                reason = replacement_reason(box, market_now(), utc_now())
            if reason and unused(box) and box.instance_type in INSTANCE_TYPES:
                say(f"{box.instance_id}: {reason} and no job runs: ending it")
                stop_box(aws, box)
                continue
            if reason:
                replacement = (box, reason)
            if ssh(box, "sudo /usr/local/sbin/ncbox-idle", capture_output=True).returncode != 0:
                note_unreachable(aws, box)
            else:
                unreachable_marker(box).unlink(missing_ok=True)
    if replacement:
        replace(aws, *replacement)
    if not POOL:  # the overflow box's records, budget and idle check too
        subprocess.call([sys.executable, __file__, "reap"], env={**os.environ, POOL_ENV: OVERFLOW})
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


#: A box counts as between jobs only this long after its last job ended: the
#: caller's --fetch follows the job's end, and a caller that holds no BoxUse
#: file (a `run` started by an older nc-remote) is covered by this alone.
QUIET_SECONDS = 120


def between_jobs(box: Box) -> bool:
    """No job slot is held, no job is waiting on the box, and none has ended in
    the last QUIET_SECONDS."""
    probe = ("for s in ncbox/slots/*; do flock -n \"$s\" true || exit 1; done; "
             # a waiting record from before this boot (baked, or a reclaim) is no job
             "[ -z \"$(find ncbox/jobs -maxdepth 2 -name waiting.json -newermt "
             "@$(awk '/^btime/ {print $2}' /proc/stat))\" ] && "
             f"[ $(( $(date +%s) - $(stat -c %Y ncbox/activity) )) -ge {QUIET_SECONDS} ]")
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
    run.add_argument("--clip", action="append", default=[],
                     help="a clip file under ~/nanocodec-data/clips the job reads (deleted on "
                          "the box before it stops)")
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
