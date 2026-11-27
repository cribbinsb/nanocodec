"""One short-lived AWS spot instance for a measurement session, cleaned up on every exit.

The launcher behind tools/dev/aws_timing.py (paired x86/arm timing on bare
metal) and tools/dev/aws_heldout_chart.py (held-out chart encodes). The AWS
account is shared with other projects, so this module only ever creates,
touches and deletes resources it made itself, each tagged Project=nanocodec,
and never lists anything else.

A session (`with Session(spec) as s:`):
  1. checks the budget: max spot price x the hard lifetime must fit
     `spec.budget_usd`, else nothing is created;
  2. creates a key pair (private key under WORK_DIR, mode 600) and a security
     group that admits SSH from this machine's public IP only;
  3. appends a "started" row to the ledger, then launches one ONE-TIME spot
     instance (no persistent request) on the official Ubuntu LTS AMI, of the
     type asked for or, when it has no capacity in any zone or the spot quota
     refuses it, of the next of `spec.fallback_types` within the budget (a
     throughput job's ladder, THROUGHPUT_X86_TYPES; a timing session has none:
     its type is the measurement):
     InstanceInitiatedShutdownBehavior=terminate, a root volume deleted on
     termination, and user-data whose first command is `shutdown -h +N`
     (the hard lifetime, even if this process dies);
  4. on EVERY exit (success, error, Ctrl-C, SIGTERM/SIGHUP) terminates the
     instance and cancels its spot request at once, waits until
     describe-instances reports the instance terminated, verifies the request
     cancelled or closed, deletes
     the security group and key pair, and appends the "ended" ledger row with
     the cost (billed seconds x the AZ's spot price + root volume + public
     IPv4) and updates "Spent so far".

Remote work runs detached (nohup) and is polled, so a dropped SSH link does
not lose it; files move as tar streams over SSH.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import fcntl
import hashlib
import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import tarfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

WORK_DIR = Path.home() / "nanocodec-data/work/aws"
LEDGER = WORK_DIR / "ledger.md"
PROJECT_TAG = "nanocodec"
OWNER_TAG = "coordinator"
#: Official Canonical AMIs, by architecture (SSM public parameters).
UBUNTU_RELEASE = "26.04"
AMI_PARAMETER = ("/aws/service/canonical/ubuntu/server/{release}/stable/current/"
                 "{arch}/hvm/ebs-gp3/ami-id")
SSH_USER = "ubuntu"
#: Max spot price we bid, relative to the current lowest AZ price: high enough
#: not to be outbid within a session, low enough to bound the cost.
MAX_PRICE_FACTOR = 1.5
#: A zone priced above this share of the session's bid is tried after the cheaper ones.
NEAR_BID_SHARE = 0.8
#: The AWS Pricing API's endpoint region (it serves every region's prices).
PRICING_REGION = "us-east-1"
#: The account's spot quota is full (EC2 error code): instances still shutting down count.
QUOTA_FULL = "MaxSpotInstanceCountExceeded"
#: How long a launch waits for the quota to free: a terminating metal instance takes
#: a minute or two to stop counting.
QUOTA_RETRIES, QUOTA_WAIT_S = 6, 30
#: The account's spot quota for the standard families, in vCPUs (Service Quotas code),
#: shared by every agent's sessions and the build box: a session waits in line for room
#: before it launches rather than being refused (coordinator, 2026-11-21: agents hit the
#: 512-vCPU quota and fell back to the PC).
SPOT_QUOTA_CODE = "L-34B43A08"
STANDARD_FAMILIES = frozenset("acdhimrtz")
#: The build box (nc-remote, Purpose tag `buildbox`): a box's worth of its region's quota
#: is always kept free (beside the running build box), so a measurement never keeps its
#: replacement or the overflow box from starting; measurements spill to other regions.
BUILD_BOX_PURPOSE = "buildbox"
BUILD_BOX_VCPUS = 96
BUILD_BOX_REGION = "eu-north-1"
#: Every region has its own spot quota (512 vCPU each, 2026-11-21): a spot session whose
#: home region is full launches in the next of these that has room instead of waiting
#: (coordinator, 2026-11-21); it waits only when every one is full.
SPILL_REGIONS = {"eu-north-1": ("eu-west-1", "eu-central-1")}
#: Set to 1 to let sessions spill when the home region is full (a result needed now).
SPILL_ENV = "NC_AWS_SPILL"
#: How often a waiting session re-reads the usage, and how long it waits in all before
#: it fails loudly (no silent fallback).
QUOTA_POLL_S = 30
#: One launch at a time from this machine per region: a waiting session holds its
#: region's lock from its check until its instance is running, so two cannot both count
#: the same free room. Per region because the quota is (2026-11-21: a c6g.large in
#: eu-west-1 waited behind sessions queued for eu-north-1's full quota).
LAUNCH_LOCK_DIR = Path.home() / "nanocodec-data" / "run"
#: How often a launch waiting for this machine's other launch in its home region looks
#: again (that lock is held from pricing until the instance runs: seconds to minutes).
LAUNCH_LOCK_POLL_S = 5


def launch_lock_path(region: str) -> Path:
    return LAUNCH_LOCK_DIR / f"aws-launch-{region}.lock"


#: run_instances errors that mean "not this type in this zone now": the next zone, then
#: the next type, is tried.
NO_CAPACITY = ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow", "Unsupported",
               "InsufficientCapacity")
#: Large x86 machines for throughput jobs (packed encodes, CPU scoring), in the order a
#: session falls back through them when the one asked for has no spot capacity in any
#: zone or the quota refuses it (docs/aws-cost-guide.md, measured 2026-11-24 in
#: eu-north-1 spot: c7i.24xlarge 165k packed encodes and 13.0k scored points per dollar,
#: c6i.24xlarge 179-216k and 11.7-14.1k by zone price). No c7a (owner: its AVX-512 is
#: double-pumped) and no 48xlarge or m-family size: 1.6-2.3x the price per vCPU that day.
THROUGHPUT_X86_TYPES = ("c7i.24xlarge", "c6i.24xlarge")
#: Running costs beside the instance (eu-north-1 list prices, USD).
GP3_USD_PER_GB_MONTH = 0.0836
PUBLIC_IPV4_USD_PER_HOUR = 0.005
#: Data out of AWS to the internet (downloads); uploads are free.
EGRESS_USD_PER_GB = 0.09
HOURS_PER_MONTH = 730
#: How often a detached remote job is polled, and SSH connection settings.
POLL_SECONDS = 20
#: How often `Session.pulling` brings a running job's finished files back.
PULL_SECONDS = 60
#: How often `Session.pulling` looks for the spot interruption notice (AWS gives two
#: minutes): on a notice it pulls at once, so a reclaim loses no finished file.
NOTICE_POLL_SECONDS = 15
#: The box's own view of a pending spot interruption (IMDSv2; empty when none is set).
SPOT_NOTICE_COMMAND = (
    "t=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token "
    "-H 'X-aws-ec2-metadata-token-ttl-seconds: 60') && "
    "curl -s -f -m 2 -H \"X-aws-ec2-metadata-token: $t\" "
    "http://169.254.169.254/latest/meta-data/spot/instance-action || true")
#: A remote job that prints nothing new this long is taken as stuck (fail fast:
#: the instance's hard lifetime is only the last-resort net).
STALL_MINUTES = 5
#: Remote log lines containing this count as failures; the log tail kept for errors.
ERROR_MARK = "FAILED"
TAIL_LINES = 30
#: Bare metal boots in ~5-10 min; a small instance in ~1.
SSH_READY_TIMEOUT_SECONDS = 15 * 60
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=6")
PUBLIC_IP_URL = "https://checkip.amazonaws.com"
#: Seconds between checks that this host's public address may still reach a box.
ACCESS_RECHECK_S = 60.0


@dataclass
class SessionSpec:
    """What to launch and why; `purpose` goes into the Purpose tag and the ledger."""
    region: str
    instance_type: str
    purpose: str
    lifetime_minutes: int = 90
    budget_usd: float = 5.0
    root_gb: int = 12
    arch: str = "amd64"  # "amd64" or "arm64" (the AMI's)
    #: Spot (cheap, may be reclaimed) or on demand (for a job an interruption would
    #: waste, such as timing: c8a spot in eu-west-1 was reclaimed 4 times on 2026-11-15).
    on_demand: bool = False
    tags: dict[str, str] = field(default_factory=dict)
    #: ExpectedEnd = launch + lifetime + this margin (the cleanup's slack; the
    #: instance itself stops at the lifetime).
    end_margin_minutes: int = 15
    #: Types tried in order after `instance_type` (no capacity, quota full), each within
    #: `budget_usd`. Empty for timing: there the type is what is measured.
    fallback_types: tuple[str, ...] = ()
    #: Whether a spot session may launch in SPILL_REGIONS when its region is full. Off
    #: unless the caller asks or `NC_AWS_SPILL=1` says a result is needed now: spill boxes
    #: cost 2.8-3.6x home (docs/aws-cost-guide.md rule 3), and on 2026-11-24 a quick
    #: screen started a $1.67/h box in eu-central-1 when eu-north-1 was briefly full.
    spill: bool = field(default_factory=lambda: os.environ.get(SPILL_ENV) == "1")
    #: At most this many vCPU of live instances with this session's purpose, in its region
    #: (None: no cap beyond the quota).
    purpose_vcpu_cap: int | None = None
    #: A spot bid of at least this many USD/h whatever the region's cheapest zone (None: the
    #: cheapest zone x MAX_PRICE_FACTOR only); the benchmark's metal machines bid it so a
    #: scarce type is taken in any zone up to it (owner 2026-11-26: "of course we will pay more").
    spot_cap_usd: float | None = None
    #: The purpose whose instances (it and its `-*` plans) count against that cap; None: this
    #: session's own. A benchmark plan (bench-history) shares the benchmark's one cap with
    #: the other plans running beside it (bench-remeasure).
    cap_purpose: str | None = None
    #: The zone tried first (a fleet spreads its boxes over zones: BP-3.11); the others
    #: follow, cheapest first.
    preferred_zone: str | None = None
    #: Zones never used (a fleet's measured reclaim rate there was too high).
    excluded_zones: tuple[str, ...] = ()
    #: How long a spot launch waits for quota room before it fails loudly (NoCapacity).
    #: A fleet's small sessions (heldout_fleet.py) wait about a minute and are relaunched
    #: by their driver, never left waiting silently (2026-11-21: sessions sat
    #: "WAITING for spot quota" in eu-central-1 until noticed by hand).
    launch_wait_minutes: float = 90.0
    #: Launch from our own image with these tags (newest available, owned by this account)
    #: when the region has one; else the stock Ubuntu image (measure_image: the baked
    #: measurement image).
    image_tags: dict[str, str] = field(default_factory=dict)


def fallbacks_after(instance_type: str, ladder: tuple[str, ...] = THROUGHPUT_X86_TYPES
                    ) -> tuple[str, ...]:
    """The ladder's types after `instance_type` (all of it when the type is not on it)."""
    return ladder[ladder.index(instance_type) + 1:] if instance_type in ladder else ladder


class NoCapacity(RuntimeError):
    """No type of the session's list could be launched in any zone."""


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Candidate:
    """One instance type a session may launch: its zones by price (cheapest first),
    the max price bid and the worst case over the lifetime."""
    instance_type: str
    zones: dict[str, float]
    max_price: float
    cap: float


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def iso(moment: datetime.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def say(line: str) -> None:
    print(f"[aws {utc_now():%H:%M:%S}] {line}", flush=True)


def public_ip() -> str:
    with urllib.request.urlopen(PUBLIC_IP_URL, timeout=10) as response:
        address = response.read().decode().strip()
    if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", address):
        raise RuntimeError(f"unexpected public IP lookup answer {address!r}")
    return address


class Session:
    """One spot instance and its key pair and security group (see the module doc)."""

    def __init__(self, spec: SessionSpec, stop: threading.Event | None = None,
                 clients: "SharedClients | None" = None):
        """A driver that runs many sessions as threads of one process passes `stop` (only
        its main thread gets signals; a set event fails the session's waits, so its cleanup
        runs) and `clients` (one set of boto3 clients for all its sessions)."""
        import boto3  # here: the instance side of a session imports `say` without boto3

        self.spec, self.stop = spec, stop
        self._client = clients or (lambda service, region: boto3.client(service,
                                                                        region_name=region))
        self.ec2 = self._client("ec2", spec.region)
        self.ssm = self._client("ssm", spec.region)
        # The pid too: two sessions of one purpose started in the same second (the
        # held-out chart's score parts, 2026-11-15) collided on the key pair name.
        # and a per-process count: threads of one driver start sessions in the same second
        self.name = (f"nanocodec-{spec.purpose}-{utc_now():%Y%m%d-%H%M%S}-{os.getpid()}"
                     f"-{next(_SESSION_COUNT)}")
        self.key_path = WORK_DIR / f"{self.name}.pem"
        self.known_hosts = WORK_DIR / f"{self.name}.known_hosts"
        self.key_name: str | None = None
        self.group_id: str | None = None
        self.instance_id: str | None = None
        self.host: str | None = None
        self.zone: str | None = None
        self.price: float | None = None
        self.launched: datetime.datetime | None = None
        self.egress_bytes = 0
        self.spot_request: str | None = None
        self.expected_end = utc_now() + datetime.timedelta(
            minutes=spec.lifetime_minutes + spec.end_margin_minutes)

    # --- lifecycle -------------------------------------------------------------------

    def __enter__(self) -> "Session":
        self._previous_handlers = {sig: signal.signal(sig, _raise_exit)
                                   for sig in (signal.SIGTERM, signal.SIGHUP)} \
            if _main_thread() else {}
        try:
            candidates = self.candidates() if self.spec.on_demand else self.claim_room()
            self.create_access()
            names = " or ".join(c.instance_type for c in candidates)
            ledger_row(self.row_resources(), "started", iso(utc_now()), "—",
                       f"cap ${max(c.cap for c in candidates):.2f}",
                       replace(self.spec, instance_type=names))
            self.launch(candidates)
            release_launch_lock(getattr(self, "_launch_lock", None))
            self.wait_for_ssh()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        # A second Ctrl-C must not interrupt the cleanup (a driver thread's main thread
        # holds its signals).
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN) if _main_thread() else None
        release_launch_lock(getattr(self, "_launch_lock", None))
        try:
            self.cleanup()
        finally:
            if previous_int is not None:
                signal.signal(signal.SIGINT, previous_int)
            for sig, handler in getattr(self, "_previous_handlers", {}).items():
                signal.signal(sig, handler)

    # --- a session that outlives its process (aws_warm: a warm measurement box) -----

    #: What `detach` keeps: enough for `attach` to ssh to the box and to clean it up.
    KEPT = ("name", "key_name", "group_id", "instance_id", "host", "zone", "price",
            "spot_request")

    def detach(self) -> dict:
        """Leave the box running past this process: its record (see `attach`). The
        caller owns it from here (aws_warm: an idle timer on the box, a reap here)."""
        release_launch_lock(getattr(self, "_launch_lock", None))
        for sig, handler in getattr(self, "_previous_handlers", {}).items():
            signal.signal(sig, handler)
        self._previous_handlers = {}
        return {**{k: getattr(self, k) for k in self.KEPT},
                "key_path": str(self.key_path), "known_hosts": str(self.known_hosts),
                "launched": iso(self.launched) if self.launched else None,
                "spec": {**self.spec.__dict__, "tags": dict(self.spec.tags),
                         "fallback_types": list(self.spec.fallback_types)}}

    @classmethod
    def attach(cls, record: dict) -> "Session":
        """The session of a detached box, to run on it or clean it up."""
        spec = SessionSpec(**{**record["spec"],
                              "fallback_types": tuple(record["spec"]["fallback_types"])})
        session = cls(spec)
        for name in cls.KEPT:
            setattr(session, name, record[name])
        session.key_path, session.known_hosts = Path(record["key_path"]), \
            Path(record["known_hosts"])
        session.launched = datetime.datetime.fromisoformat(record["launched"].replace(
            "Z", "+00:00")) if record.get("launched") else None
        session.allow_this_host()
        return session

    def recheck_access(self) -> None:
        """`allow_this_host` at most every ACCESS_RECHECK_S while the box is used: the
        address changed twice within half an hour on 2026-11-24 (after the reboot, and
        again mid-job), so a check at attach alone left a running job polling a box that
        no longer let it in. A failed check is logged; the SSH call then reports itself."""
        now = time.monotonic()
        if now - getattr(self, "_access_checked_at", -ACCESS_RECHECK_S) < ACCESS_RECHECK_S:
            return
        self._access_checked_at = now
        try:
            self.allow_this_host()
        except Exception as error:  # noqa: BLE001 - never stop a job over the check itself
            say(f"security group {self.group_id}: address check failed: {error}")

    def allow_this_host(self) -> None:
        """Let this host's current public address reach the box over SSH. A box outlives
        its process (`detach`), and this host's address can change meanwhile (a reboot,
        a new DHCP lease): 2026-11-24, after a reboot moved the PC from 51.179.192.42 to
        51.179.200.228, every job on the warm box and the build box timed out on SSH."""
        if not self.group_id:
            return
        source = f"{public_ip()}/32"
        groups = self.ec2.describe_security_groups(GroupIds=[self.group_id])["SecurityGroups"]
        allowed = {r["CidrIp"] for g in groups for p in g.get("IpPermissions", [])
                   for r in p.get("IpRanges", [])}
        if source in allowed:
            return
        self.ec2.authorize_security_group_ingress(
            GroupId=self.group_id, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                "IpRanges": [{"CidrIp": source, "Description": "this host"}]}])
        say(f"security group {self.group_id}: SSH from this host's new address {source}")

    def tags(self, resource: str) -> list[dict]:
        tags = {"Project": PROJECT_TAG, "Owner": OWNER_TAG, "Purpose": self.spec.purpose,
                "ExpectedEnd": iso(self.expected_end), "Name": self.name, **self.spec.tags}
        return [{"ResourceType": resource,
                 "Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}]

    def pause(self, seconds: float, waiting_for: str) -> None:
        """Sleep while waiting; a driver's stop ends the wait (and the launch) at once."""
        if self.stop is None:
            time.sleep(seconds)
        elif self.stop.wait(seconds):
            raise NoCapacity(f"stopped by the driver while waiting for {waiting_for}")

    def candidates(self) -> list[Candidate]:
        """The type asked for, then its fallbacks, each priced; a type without a price
        in the region, or whose worst case passes the budget, is left out (logged)."""
        found, skipped = [], []
        for instance_type in (self.spec.instance_type, *self.spec.fallback_types):
            try:
                zones = (self.on_demand_zones(instance_type) if self.spec.on_demand
                         else self.zone_prices(instance_type))
            except RuntimeError as error:
                say(f"{instance_type}: {error}: skipped")
                skipped.append(str(error))
                continue
            max_price = min(zones.values()) * (1 if self.spec.on_demand else MAX_PRICE_FACTOR)
            if self.spec.spot_cap_usd and not self.spec.on_demand:
                max_price = self.spec.spot_cap_usd  # a flat bid: any zone up to the cap
            cap = self.cost(max_price, self.spec.lifetime_minutes / 60)
            market = "on demand" if self.spec.on_demand else "spot"
            say(f"{instance_type} in {self.spec.region}: {market} from "
                f"${min(zones.values()):.4f}/h, cap ${max_price:.4f}/h, worst case ${cap:.2f} "
                f"for the {self.spec.lifetime_minutes}-minute lifetime")
            if cap > self.spec.budget_usd:
                say(f"{instance_type}: worst case ${cap:.2f} exceeds the session budget "
                    f"${self.spec.budget_usd:.2f}: skipped")
                skipped.append(f"{instance_type} worst case ${cap:.2f}")
                continue
            found.append(Candidate(instance_type, zones, max_price, cap))
        if not found:
            raise BudgetExceeded(f"no instance type fits the session budget "
                                 f"${self.spec.budget_usd:.2f}: {'; '.join(skipped)}")
        return found

    def on_demand_zones(self, instance_type: str) -> dict[str, float]:
        """The on-demand price (AWS Pricing API, Linux, shared tenancy) in every zone
        of the region that offers the type."""
        import boto3

        pricing = boto3.client("pricing", region_name=PRICING_REGION)
        filters = {"instanceType": instance_type, "regionCode": self.spec.region,
                   "operatingSystem": "Linux", "tenancy": "Shared", "preInstalledSw": "NA",
                   "capacitystatus": "Used"}
        products = pricing.get_products(ServiceCode="AmazonEC2", Filters=[
            {"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in filters.items()])
        prices = [float(dim["pricePerUnit"]["USD"]) for item in products["PriceList"]
                  for term in json.loads(item)["terms"]["OnDemand"].values()
                  for dim in term["priceDimensions"].values()]
        if not prices:
            raise RuntimeError(f"no on-demand price for {instance_type} in "
                               f"{self.spec.region}")
        zones = [o["Location"] for o in self.ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": [instance_type]}])[
                "InstanceTypeOfferings"]]
        return {zone: max(prices) for zone in sorted(zones)}

    def zone_prices(self, instance_type: str) -> dict[str, float]:
        history = self.ec2.describe_spot_price_history(
            InstanceTypes=[instance_type], ProductDescriptions=["Linux/UNIX"],
            StartTime=utc_now())["SpotPriceHistory"]
        prices = {h["AvailabilityZone"]: float(h["SpotPrice"]) for h in history}
        if not prices:
            raise RuntimeError(f"no spot price for {instance_type} in "
                               f"{self.spec.region}")
        return dict(sorted(prices.items(), key=lambda item: item[1]))

    def cost(self, price: float, hours: float) -> float:
        storage = self.spec.root_gb * GP3_USD_PER_GB_MONTH / HOURS_PER_MONTH
        return hours * (price + storage + PUBLIC_IPV4_USD_PER_HOUR)

    def create_access(self) -> None:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        key = self.ec2.create_key_pair(KeyName=self.name, KeyType="ed25519",
                                       TagSpecifications=self.tags("key-pair"))
        self.key_name = key["KeyName"]
        self.key_path.touch(mode=0o600)
        self.key_path.write_text(key["KeyMaterial"])
        self.key_path.chmod(0o600)
        source = f"{public_ip()}/32"
        vpc = self.ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
        group = self.ec2.create_security_group(
            GroupName=self.name, Description="nanocodec measurement session: SSH from one host",
            VpcId=vpc["Vpcs"][0]["VpcId"], TagSpecifications=self.tags("security-group"))
        self.group_id = group["GroupId"]
        self.ec2.authorize_security_group_ingress(
            GroupId=self.group_id, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                "IpRanges": [{"CidrIp": source, "Description": "this host"}]}])
        say(f"key pair {self.key_name}, security group {self.group_id} (SSH from {source})")

    def follow_host_address(self) -> None:
        """Let SSH in from this host's current public address when it has changed (an ISP
        change on 2026-11-24 cut every running session off: the stall watch then stopped 70
        boxes). A failed poll is the trigger; the old address's rule is left in place."""
        try:
            source = f"{public_ip()}/32"
            rules = self.ec2.describe_security_groups(GroupIds=[self.group_id])[
                "SecurityGroups"][0]["IpPermissions"]
            if any(r.get("CidrIp") == source for p in rules for r in p.get("IpRanges", [])):
                return
            self.ec2.authorize_security_group_ingress(
                GroupId=self.group_id, IpPermissions=[{
                    "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                    "IpRanges": [{"CidrIp": source, "Description": "this host (address changed)"}]}])
            say(f"this host's address changed: SSH now also from {source}")
        except Exception as e:  # noqa: BLE001  never fatal: the poll retries anyway
            say(f"could not follow the host address: {e}")

    def ami(self) -> tuple[str, str]:
        baked = self.baked_image()
        if baked is not None:
            say(f"image {baked['ImageId']} (baked: {baked.get('Name')})")
            return baked["ImageId"], baked["RootDeviceName"]
        name = AMI_PARAMETER.format(release=UBUNTU_RELEASE, arch=self.spec.arch)
        image_id = self.ssm.get_parameter(Name=name)["Parameter"]["Value"]
        image = self.ec2.describe_images(ImageIds=[image_id])["Images"][0]
        return image_id, image["RootDeviceName"]

    def baked_image(self) -> dict | None:
        """The newest available image of ours with `spec.image_tags` in this region."""
        if not self.spec.image_tags:
            return None
        filters = [{"Name": "state", "Values": ["available"]},
                   {"Name": "tag:Project", "Values": [PROJECT_TAG]},
                   *[{"Name": f"tag:{k}", "Values": [v]} for k, v in self.spec.image_tags.items()]]
        images = self.ec2.describe_images(Owners=["self"], Filters=filters)["Images"]
        return max(images, key=lambda i: i["CreationDate"]) if images else None

    def use_region(self, region: str) -> None:
        if region != self.spec.region:
            self.spec = replace(self.spec, region=region)
            self.ec2, self.ssm = self._client("ec2", region), self._client("ssm", region)

    def regions(self) -> list[str]:
        home = self.spec.region
        return [home, *(SPILL_REGIONS.get(home, ()) if self.spec.spill else ())]

    def room(self) -> tuple[int, str]:
        """(vCPU free in the session's region beside every live spot instance of the
        account and, in the build box's region, its share; a note for the log)."""
        limit = spot_quota(self.spec.region)
        used, _ = spot_usage(self.ec2)
        # Always kept in the build box's region, running or not: its make-before-break
        # replacement (or the overflow box) needs that room while it runs (2026-11-21: the
        # 96-vCPU replacement could not start beside measurement boxes at 512/512).
        reserve = BUILD_BOX_VCPUS if self.spec.region == BUILD_BOX_REGION else 0
        free = limit - used - reserve
        note = (f"{self.spec.region}: {free} vCPU free of {limit} ({used} in use"
                + (f", {reserve} kept for the build box" if reserve else "") + ")")
        if self.spec.purpose_vcpu_cap is not None:
            # a purpose's share of the quota (2026-11-24: the benchmark fleets held 1000 of
            # 1200 vCPU and starved the dev screens); checked under the launch lock, so this
            # machine's launches cannot overshoot it together
            capped = self.spec.cap_purpose or self.spec.purpose
            mine = purpose_vcpus(self.ec2, capped)
            free = min(free, self.spec.purpose_vcpu_cap - mine)
            note += f"; {capped} holds {mine} of its {self.spec.purpose_vcpu_cap} cap"
        return free, note

    def claim_room(self) -> list[Candidate]:
        """The region to launch in and its priced candidates (cheapest per vCPU first),
        holding that region's launch lock: of the regions (home, then SPILL_REGIONS)
        whose quota has room for a candidate, the one whose cheapest fitting candidate
        costs least per vCPU (2026-11-21: the first region that fitted was taken, and six
        spilled boxes started at 2.5x eu-north-1's price per vCPU); when none has room,
        wait and look again, failing loudly after `spec.launch_wait_minutes` (time spent
        waiting for this machine's own launch lock does not count). No region with any
        type within the budget fails at once: waiting cannot fix a price (2026-11-21:
        $3 sessions waited in eu-central-1, where nothing cost under $3.28)."""
        home, priced = self.spec.region, {}
        # once, at home: the loop moves the session's region as it looks (2026-11-21: a
        # second pass after waiting for this machine's own launch looked from the last
        # region looked at, saw only it, and launched there at 2.5x the price per vCPU)
        regions = self.regions()
        deadline = time.monotonic() + self.spec.launch_wait_minutes * 60
        while True:
            notes, options, home_busy = [], [], False
            for region in regions:
                self.use_region(region)
                if region not in priced:
                    try:
                        priced[region] = self.candidates()
                    except BudgetExceeded as error:
                        priced[region] = []
                        say(f"{region}: {error}")
                if not priced[region]:
                    notes.append(f"{region}: no type within budget")
                    continue
                handle = try_launch_lock(region, priced[region][0].instance_type.split(".")[0])
                if handle is None:
                    notes.append(f"{region}: another launch from this machine")
                    home_busy |= region == home
                    continue
                free, note = self.room()
                fitting = sorted((c for c in priced[region]
                                  if instance_vcpus(self.ec2, c.instance_type) <= free),
                                 key=lambda c: per_vcpu(self.ec2, c))
                if fitting:
                    options.append((per_vcpu(self.ec2, fitting[0]), region, fitting, handle,
                                    note))
                else:
                    release_launch_lock(handle)
                notes.append(note)
            if not any(priced.values()):
                self.use_region(home)
                raise BudgetExceeded("no type within the session budget "
                                     f"${self.spec.budget_usd:.2f} in any region: "
                                     + "; ".join(notes))
            if home_busy and time.monotonic() < deadline:
                # this machine's other launch holds the home region's lock for seconds:
                # wait for it rather than spill (2026-11-21: two scoring sessions started
                # together, one spilled to eu-west-1, found no capacity there and failed)
                for _, _, _, other, _ in options:
                    release_launch_lock(other)
                say(f"{home}: waiting for this machine's other launch there "
                    f"({LAUNCH_LOCK_POLL_S} s)")
                self.pause(LAUNCH_LOCK_POLL_S, "the launch lock")
                deadline += LAUNCH_LOCK_POLL_S
                continue
            if options:
                options.sort(key=lambda option: option[0])
                rate, region, fitting, handle, note = options[0]
                for _, _, _, other, _ in options[1:]:
                    release_launch_lock(other)
                self.use_region(region)
                self._launch_lock = handle
                say(f"spot quota: {note}: launching there, {fitting[0].instance_type} at "
                    f"${rate * 1000:.2f}/h per 1000 vCPU"
                    + ("" if region == home else f" (chosen over: {'; '.join(notes)})"))
                return fitting
            self.use_region(home)
            if time.monotonic() > deadline:
                raise NoCapacity(f"waited {self.spec.launch_wait_minutes:g} min for spot quota: "
                                 + "; ".join(notes))
            say("WAITING for spot quota in every region: " + "; ".join(notes)
                + f"; re-checking in {QUOTA_POLL_S} s")
            self.pause(QUOTA_POLL_S, "spot quota")

    def launch(self, candidates: list[Candidate]) -> None:
        """Launch the first candidate type with capacity in any zone. A type without
        capacity anywhere, or refused by the spot quota, gives way to the next; when
        only the quota refused, the whole list is tried again after QUOTA_WAIT_S (the
        quota counts instances still shutting down)."""
        image_id, root_device = self.ami()
        say(f"image {image_id}")  # the machine image, for the report's provenance
        # The hard lifetime comes first: it holds even if this process dies.
        user_data = f"#!/bin/bash\nshutdown -h +{self.spec.lifetime_minutes}\n"
        for attempt in range(QUOTA_RETRIES + 1):
            quota_full = False
            for index, candidate in enumerate(candidates):
                outcome = self.try_zones(candidate, image_id, root_device, user_data)
                if outcome == LAUNCHED:
                    # the quota now counts the pending instance: the next launch from this
                    # machine need not wait for this one to run (2026-11-24: 100 fleet slots
                    # queued for the lock about 15 s each)
                    release_launch_lock(getattr(self, "_launch_lock", None))
                    if candidate.instance_type != self.spec.instance_type:
                        say(f"fell back from {self.spec.instance_type} to "
                            f"{candidate.instance_type}")
                        self.spec = replace(self.spec, instance_type=candidate.instance_type)
                    self.await_running()
                    return
                quota_full |= outcome == QUOTA
                after = (f"trying {candidates[index + 1].instance_type}"
                         if index + 1 < len(candidates) else "no type left")
                reason = ("the spot instance quota is full" if outcome == QUOTA
                          else "no capacity in any zone")
                say(f"{candidate.instance_type}: {reason}; {after}")
            if not quota_full:
                break
            # The spot quota counts instances still shutting down (a session that just
            # ended): one launched right after it is refused for a minute or two
            # (2026-11-15: the second scoring session of a held-out refresh).
            if attempt < QUOTA_RETRIES:
                say(f"spot instance quota full; retry {attempt + 1}/{QUOTA_RETRIES} in "
                    f"{QUOTA_WAIT_S} s")
                time.sleep(QUOTA_WAIT_S)
        else:
            raise NoCapacity(f"the spot instance quota stayed full for "
                             f"{QUOTA_RETRIES * QUOTA_WAIT_S} s")
        raise NoCapacity(f"no zone of {self.spec.region} could launch "
                         f"{' or '.join(c.instance_type for c in candidates)}")

    def await_running(self) -> None:
        say(f"launched {self.instance_id} ({self.spec.instance_type}) in {self.zone} "
            f"(spot ${self.price:.4f}/h)")
        self.ec2.get_waiter("instance_running").wait(
            InstanceIds=[self.instance_id], WaiterConfig={"Delay": 10, "MaxAttempts": 90})
        instance = self.describe()
        self.host = instance["PublicIpAddress"]
        self.launched = instance["LaunchTime"]
        say(f"{self.instance_id} running at {self.host}")

    def try_zones(self, candidate: Candidate, image_id: str, root_device: str,
                  user_data: str) -> str:
        """Launch `candidate` in its cheapest zone with capacity: LAUNCHED, NO_ROOM (no
        zone had capacity) or QUOTA (the account's spot quota is full)."""
        from botocore.exceptions import ClientError
        # the preferred zone first, then cheapest first, a zone priced near the bid last (it is
        # reclaimed as soon as the price passes the bid); a zone without capacity is skipped
        def order(z: str) -> tuple:
            near_bid = candidate.zones[z] > NEAR_BID_SHARE * candidate.max_price
            return (z != self.spec.preferred_zone, near_bid, candidate.zones[z])
        for zone in sorted((z for z in candidate.zones if z not in self.spec.excluded_zones),
                           key=order):
            subnets = self.ec2.describe_subnets(Filters=[
                {"Name": "availability-zone", "Values": [zone]},
                {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
            if not subnets:
                continue
            try:
                reply = self.ec2.run_instances(
                    ImageId=image_id, InstanceType=candidate.instance_type, MinCount=1,
                    MaxCount=1, KeyName=self.key_name, UserData=user_data,
                    InstanceInitiatedShutdownBehavior="terminate",
                    **({} if self.spec.on_demand else {"InstanceMarketOptions": {
                        "MarketType": "spot", "SpotOptions": {
                            "SpotInstanceType": "one-time",
                            "MaxPrice": f"{candidate.max_price:.4f}",
                            "InstanceInterruptionBehavior": "terminate"}}}),
                    BlockDeviceMappings=[{"DeviceName": root_device, "Ebs": {
                        "VolumeSize": self.spec.root_gb, "VolumeType": "gp3",
                        "DeleteOnTermination": True}}],
                    NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnets[0]["SubnetId"],
                                        "Groups": [self.group_id],
                                        "AssociatePublicIpAddress": True}],
                    TagSpecifications=[*self.tags("instance"), *self.tags("volume"),
                                       *([] if self.spec.on_demand
                                         else self.tags("spot-instances-request"))])
            except ClientError as error:
                code = error.response["Error"]["Code"]
                if code in NO_CAPACITY:
                    say(f"{candidate.instance_type} {zone}: {code}, trying the next zone")
                    continue
                if code == QUOTA_FULL:
                    return QUOTA
                raise
            self.instance_id = reply["Instances"][0]["InstanceId"]
            self.spot_request = reply["Instances"][0].get("SpotInstanceRequestId")
            self.zone, self.price = zone, candidate.zones[zone]
            return LAUNCHED
        return NO_ROOM

    def describe(self) -> dict | None:
        reply = self.ec2.describe_instances(InstanceIds=[self.instance_id])
        found = [i for r in reply["Reservations"] for i in r["Instances"]]
        return found[0] if found else None

    def wait_for_ssh(self) -> None:
        deadline = time.monotonic() + SSH_READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.ssh("true", check=False, quiet=True).returncode == 0:
                say("SSH is up")
                return
            time.sleep(10)
        raise RuntimeError("SSH did not come up")

    def cleanup(self, ended_at: datetime.datetime | None = None) -> None:
        """Terminate (verified), delete the key pair and group, write the ledger row.
        `ended_at`: when a box that stopped itself ended (its cost runs to then)."""
        ended = min(ended_at, utc_now()) if ended_at else utc_now()
        if self.instance_id and ended_at is None:
            spot_request = self.spot_request or (self.describe() or {}).get(
                "SpotInstanceRequestId")
            say(f"terminating {self.instance_id}")
            self.ec2.terminate_instances(InstanceIds=[self.instance_id])
            # at once, not after the (metal: ~7 min) termination: the request would sit
            # "active/fulfilled" in an audit all that time
            if spot_request:
                self.ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request])
            self.ec2.get_waiter("instance_terminated").wait(
                InstanceIds=[self.instance_id], WaiterConfig={"Delay": 10, "MaxAttempts": 120})
            ended = utc_now()
            state = (self.describe() or {}).get("State", {}).get("Name", "gone")
            if state not in ("terminated", "gone"):
                raise RuntimeError(f"{self.instance_id} is still {state}")
            say(f"{self.instance_id} verified {state}")
            if spot_request:
                self.verify_spot_request_ended(spot_request)
        if self.group_id:
            self.delete_group()
        if self.key_name:
            self.ec2.delete_key_pair(KeyName=self.key_name)
            say(f"key pair {self.key_name} deleted")
        for path in (self.key_path, self.known_hosts):
            path.unlink(missing_ok=True)
        if self.instance_id:
            start = self.launched or ended
            cost = self.cost(self.price, (ended - start).total_seconds() / 3600) + \
                self.egress_bytes / 1e9 * EGRESS_USD_PER_GB
            ledger_row(self.row_resources(), "ended", iso(start), iso(ended), f"${cost:.2f}",
                       self.spec, spent=cost)
            say(f"session cost ${cost:.2f} ({(ended - start).total_seconds() / 60:.1f} min "
                f"at ${self.price:.4f}/h)")
        elif self.group_id or self.key_name:
            ledger_row(self.row_resources(), "ended (no instance)", "—", iso(ended), "$0.00",
                       self.spec, spent=0.0)

    def verify_spot_request_ended(self, request_id: str) -> None:
        """The request (cancelled with the terminate) is cancelled or closed."""
        for _ in range(30):
            state = self.ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[request_id])["SpotInstanceRequests"][0]["State"]
            if state in ("cancelled", "closed"):
                say(f"spot request {request_id} verified {state}")
                return
            time.sleep(2)
        raise RuntimeError(f"spot request {request_id} is still {state}")

    def delete_group(self) -> None:
        from botocore.exceptions import ClientError
        # The instance's network interface can take a few seconds to detach.
        for _ in range(30):
            try:
                self.ec2.delete_security_group(GroupId=self.group_id)
                say(f"security group {self.group_id} deleted")
                return
            except ClientError as error:
                if error.response["Error"]["Code"] != "DependencyViolation":
                    raise
                time.sleep(10)
        raise RuntimeError(f"security group {self.group_id} could not be deleted")

    def row_resources(self) -> str:
        parts = [self.instance_id or "instance (not launched)", self.group_id, self.key_name]
        return ", ".join(p for p in parts if p)

    # --- remote work -----------------------------------------------------------------

    def ssh_base(self) -> list[str]:
        self.recheck_access()
        return ["ssh", "-i", str(self.key_path), *SSH_OPTIONS,
                "-o", f"UserKnownHostsFile={self.known_hosts}",
                "-o", "StrictHostKeyChecking=accept-new", f"{SSH_USER}@{self.host}"]

    def ssh(self, command: str, check: bool = True, quiet: bool = False,
            stdin=None) -> subprocess.CompletedProcess:
        return subprocess.run([*self.ssh_base(), command], check=check, stdin=stdin,
                              capture_output=quiet, text=quiet)

    def upload(self, local_dir: Path, remote_dir: str) -> None:
        """`local_dir`'s contents (symlinks followed) into `remote_dir`, as one tar stream."""
        say(f"uploading {local_dir} to {remote_dir}")
        tar = subprocess.Popen(["tar", "-C", str(local_dir), "-chf", "-", "."],
                               stdout=subprocess.PIPE)
        self.ssh(f"mkdir -p {remote_dir} && tar -C {remote_dir} -xf -", stdin=tar.stdout)
        if tar.wait() != 0:
            raise RuntimeError("tar of the upload failed")

    def download(self, remote_dir: str, local_dir: Path) -> None:
        """`remote_dir`'s contents into `local_dir` (one tar stream; its size is billed)."""
        local_dir.mkdir(parents=True, exist_ok=True)
        size = self.ssh(f"du -sb {remote_dir} 2>/dev/null | cut -f1", quiet=True,
                        check=False).stdout.strip()
        if not size:  # a failure before the job wrote anything: keep its own error
            say(f"nothing to download: {remote_dir} does not exist")
            return
        self.egress_bytes += int(size or 0)
        say(f"downloading {remote_dir} ({int(size or 0) / 1e6:.0f} MB) to {local_dir}")
        ssh = subprocess.Popen([*self.ssh_base(), f"tar -C {remote_dir} -cf - ."],
                               stdout=subprocess.PIPE)
        subprocess.run(["tar", "-C", str(local_dir), "-xf", "-"], stdin=ssh.stdout, check=True)
        if ssh.wait() != 0:
            raise RuntimeError(f"download of {remote_dir} failed")

    def download_new(self, remote_dir: str, local_dir: Path, marker: str,
                     names: list[Path] | None = None) -> int:
        """The files under `remote_dir` changed since the last call with `marker` (all of
        them on the first) into `local_dir`: a checkpoint that a spot reclaim cannot
        take back. Returns the bytes moved (billed as egress); `names` gains the paths
        of the files brought back (under `local_dir`), also when a later batch fails.

        The new files go in batches over PULL_STREAMS parallel ssh links, one tar per batch:
        one ssh stream from a US box carried 2-3 MB/s on 2026-11-27 (a box's 46 GB of
        streams would have taken ~2 h more); 8 parallel links carried 24-32 MB/s from one
        us-west-2 box against 12 MB/s for one (docs/aws-cost-guide.md "Pulling results
        back"). Every file is verified on arrival against the size and SHA-256 the box took
        of the bytes it sent, and only then renamed into place (receive_batch): a cut or
        damaged copy never appears under its real name. The marker moves on only when every
        file has arrived whole; a failed call (a reclaim) keeps the files that landed, and the
        next call skips the files already here whole (same size and mtime), so a retry
        fetches only what is missing."""
        local_dir.mkdir(parents=True, exist_ok=True)
        listing = self.ssh(new_files_command(remote_dir, marker), check=False, quiet=True)
        if listing.returncode != 0:
            raise RuntimeError(f"checkpoint listing of {remote_dir} failed")
        todo = [f for f in parse_new_files(listing.stdout) if not f.arrived(local_dir)]
        moved, failed = self._fetch_batches(remote_dir, local_dir, pull_batches(todo), names)
        self.egress_bytes += moved
        if failed:
            raise RuntimeError(f"checkpoint download of {remote_dir} failed "
                               f"({failed} batches missing)")
        if self.ssh(advance_marker_command(remote_dir, marker), check=False,
                    quiet=True).returncode != 0:
            raise RuntimeError(f"checkpoint of {remote_dir} could not move on")
        return moved

    def _fetch_batches(self, remote_dir: str, local_dir: Path,
                       batches: list[list[RemoteFile]], names: list[Path] | None
                       ) -> tuple[int, int]:
        """`batches` over PULL_STREAMS parallel links, each tried PULL_ATTEMPTS times, a retry
        asking only for the files that did not arrive whole: (bytes moved, batches that never
        arrived)."""
        moved, failed = 0, 0
        lock = threading.Lock()

        def one(batch: list[RemoteFile]) -> None:
            nonlocal moved, failed
            todo = batch
            for _ in range(PULL_ATTEMPTS):
                settled = self._fetch_batch(remote_dir, local_dir, todo)
                with lock:
                    for f in todo:
                        if settled.get(f.path):
                            moved += f.size
                            if names is not None:
                                names.append(local_dir / f.path)
                todo = [f for f in todo if f.path not in settled]
                if not todo:
                    return
            with lock:
                failed += 1
        with concurrent.futures.ThreadPoolExecutor(PULL_STREAMS) as pool:
            list(pool.map(one, batches))
        return moved, failed

    def _fetch_batch(self, remote_dir: str, local_dir: Path,
                     batch: list[RemoteFile]) -> dict[str, bool]:
        """One batch over its own ssh link (receive_batch): each settled file's path, True
        when it landed whole, False when the box no longer had it as listed."""
        ssh = subprocess.Popen([*self.ssh_base(), batch_send_command(remote_dir)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
        try:
            ssh.stdin.write(b"".join(f"{f.size} {f.mtime} {f.path}".encode() + b"\0"
                                     for f in batch))
            ssh.stdin.close()
        except BrokenPipeError:
            pass  # the link died: what arrived says so
        try:
            return receive_batch(ssh.stdout, local_dir)
        finally:
            ssh.stdout.close()
            ssh.wait()

    @contextlib.contextmanager
    def pulling(self, remote_dir: str, local_dir: Path, marker: str,
                on_files: Callable[[list[Path]], None] | None = None,
                every_s: float = PULL_SECONDS):
        """While the block runs, what `remote_dir` gained since the last pull comes back
        every `every_s` (download_new), each batch handed to `on_files` as it lands; one
        last pull on the way out. Finished work leaves the box as it finishes, so a lost
        box costs only the minutes since the last pull (2026-11-21: a spot reclaim took a
        whole 13-minute round, then its "checkpoint download failed" took the rest). A
        failed pull is logged, never raised: the next one sends its files again, and what
        came back earlier stays."""
        lock, stop = threading.Lock(), threading.Event()
        noticed = False

        def pull(final: bool) -> None:
            with lock:
                files: list[Path] = []
                try:
                    self.download_new(remote_dir, local_dir, marker, files)
                except (RuntimeError, OSError) as error:
                    say(f"pull of {remote_dir} failed{' (the last)' if final else ''}: {error}")
                if files and on_files:
                    on_files(files)

        def loop() -> None:
            nonlocal noticed
            last = time.monotonic()
            while not stop.wait(min(NOTICE_POLL_SECONDS, every_s)):
                if not noticed and self.spot_notice():
                    noticed = True
                    say("spot interruption notice: pulling the finished files now")
                elif time.monotonic() - last < every_s:
                    continue
                pull(final=False)
                last = time.monotonic()
        puller = threading.Thread(target=loop, daemon=True)
        puller.start()
        try:
            yield
        finally:
            stop.set()
            puller.join()
            pull(final=True)

    def spot_notice(self) -> bool:
        """Whether the box has a spot interruption notice (never on demand; a failed
        look counts as none)."""
        if self.spec.on_demand:
            return False
        probe = self.ssh(SPOT_NOTICE_COMMAND, check=False, quiet=True)
        return probe.returncode == 0 and "action" in probe.stdout

    def run_detached(self, command: str, log: str, timeout_minutes: float,
                     stall_minutes: float = STALL_MINUTES, max_errors: int = 0) -> None:
        """Run `command` on the instance under nohup and wait for it, failing fast.

        The exit status lands in `log`.status, so a dropped SSH link loses nothing.
        Raises (with the log's tail) when the job fails, prints nothing new for
        `stall_minutes`, prints more than `max_errors` lines with ERROR_MARK, or
        runs past `timeout_minutes`."""
        self.start_detached(command, log)
        try:
            self._watch(log, timeout_minutes, stall_minutes, max_errors)
        except BaseException:
            self.ssh(f"kill -TERM -- -$(cat {log}.pid) 2>/dev/null; sleep 2; "
                     f"kill -KILL -- -$(cat {log}.pid) 2>/dev/null; true", check=False)
            raise

    def start_detached(self, command: str, log: str) -> None:
        """Start `command` on the instance under nohup; its exit status lands in `log`.status."""
        # a subshell, so a command that sets -e and fails still reaches the status line
        # (2026-11-23: a failed build left no status and was waited on until its stall)
        job = shlex.quote(f"( {command} ); echo $? > {log}.status")
        # its own process group (setsid), so a failure stops all of it before any download
        self.ssh(f"rm -f {log}.status; setsid nohup bash -c {job} > {log} 2>&1 < /dev/null & "
                 f"echo $! > {log}.pid")

    def state(self) -> str:
        """The instance's state ("gone" once it is no longer listed)."""
        return (self.describe() or {}).get("State", {}).get("Name", "gone")

    def _watch(self, log: str, timeout_minutes: float, stall_minutes: float,
               max_errors: int) -> None:
        deadline = time.monotonic() + timeout_minutes * 60
        shown, errors, last_news = 0, 0, time.monotonic()
        tail: list[str] = []
        while True:
            if self.stop is not None and self.stop.is_set():
                raise RemoteFailure("stopped by the driver", tail)
            if time.monotonic() > deadline:
                raise RemoteFailure(f"no result in {timeout_minutes} min", tail)
            if time.monotonic() - last_news > stall_minutes * 60:
                raise RemoteFailure(f"no progress for {stall_minutes} min", tail)
            time.sleep(POLL_SECONDS)
            probe = self.ssh(f"cat {log}.status 2>/dev/null; echo ---; tail -n +{shown + 1} {log}",
                             check=False, quiet=True)
            if probe.returncode != 0:
                instance = self.describe() or {}
                state = instance.get("State", {}).get("Name", "gone")
                if state != "running":
                    # a spot interruption: polling on would wait out the whole timeout
                    # (2026-11-15: 3 min of retries until stopped by hand)
                    reason = instance.get("StateReason", {}).get("Message", "")
                    raise RemoteFailure(f"the instance is {state}: {reason}", tail)
                self.follow_host_address()
                say("poll failed (retrying)")
                continue
            status, _, new = probe.stdout.partition("---\n")
            lines = new.splitlines()
            for line in lines:
                print(f"  remote: {line}", flush=True)
            if lines:
                shown, last_news = shown + len(lines), time.monotonic()
                tail = (tail + lines)[-TAIL_LINES:]
                errors += sum(ERROR_MARK in line for line in lines)
            if errors > max_errors:
                raise RemoteFailure(f"{errors} failures (more than {max_errors})", tail)
            if status.strip():
                if status.strip() != "0":
                    raise RemoteFailure(f"exit status {status.strip()}", tail)
                return


#: try_zones outcomes.
LAUNCHED, NO_ROOM, QUOTA = "launched", "no capacity", "quota full"


#: Parallel ssh links (one tar each) of one download_new; see its doc for the measurement.
PULL_STREAMS = 8
#: A batch is at most this big: a lost link resends at most this much.
PULL_BATCH_BYTES = 256 << 20
#: ... and at least this big when there is that much (one ssh setup is ~1 s across the
#: Atlantic: smaller batches would spend more time connecting than moving).
PULL_MIN_BATCH_BYTES = 8 << 20
#: Tries per batch within one call (a lost link is retried once; a reclaimed box fails fast).
PULL_ATTEMPTS = 2
#: The files a checkpoint may bring: another job's in-progress temporaries on a shared box
#: are dot files, renamed into place when complete, so they are left out (2026-11-21: one
#: job's checkpoint failed on another's `.cbr150-....json.qlpmzmbo.times.txt`).
#: A writer's `*.partial` temporaries (renamed into place when whole, nceval.cbr) are left out
#: too: copying one mid-write brought truncated streams to the PC.
CHECKPOINT_FILES = "find . -type f ! -name '.*' ! -name '*.partial'"
#: The member that ends every batch (batch_send_command): each sent file's size and SHA-256,
#: hashed from the bytes sent, and the listed files the box no longer had.
BATCH_MANIFEST = ".nanocodec-batch-manifest.json"
#: Bytes per read while a batch is received and hashed.
RECEIVE_CHUNK = 1 << 20


@dataclass(frozen=True)
class RemoteFile:
    """One file a checkpoint lists: its path under the pulled directory, size and mtime."""
    path: str
    size: int
    mtime: int

    def arrived(self, local_dir: Path) -> bool:
        """Whether an earlier (failed) call already brought this exact file whole: tar keeps
        the mtime, so the same size and whole-second mtime mean the same file."""
        try:
            here = (local_dir / self.path).stat()
        except OSError:
            return False
        return here.st_size == self.size and int(here.st_mtime) == self.mtime


def new_files_command(remote_dir: str, marker: str) -> str:
    """The box's list of the files under `remote_dir` newer than `marker` (all on the first
    call), NUL-separated `size mtime path` records. `marker`.next is stamped first, so a
    file written during the pull is listed again next time, never missed."""
    listed = f"{CHECKPOINT_FILES} -printf '%s %T@ %P\\0'"
    return (f"cd {remote_dir} && touch {marker}.next && "
            f"{{ if [ -f {marker} ]; then {listed} -newer {marker}; else {listed}; fi; }}")


def parse_new_files(listing: str) -> list[RemoteFile]:
    files = []
    for record in filter(None, listing.split("\0")):
        size, mtime, path = record.split(" ", 2)
        files.append(RemoteFile(path, int(size), int(float(mtime))))
    return files


def pull_batches(files: list[RemoteFile], streams: int = PULL_STREAMS) -> list[list[RemoteFile]]:
    """`files` in path order (a clip's streams together), cut into batches of about a
    `streams`-th of the total, kept within PULL_MIN_BATCH_BYTES..PULL_BATCH_BYTES, so a small
    pull still spreads over the links and a large one never risks more than a batch a link."""
    total = sum(f.size for f in files)
    target = min(PULL_BATCH_BYTES, max(PULL_MIN_BATCH_BYTES, -(-total // streams)))
    batches, batch, size = [], [], 0
    for f in sorted(files, key=lambda f: f.path):
        if batch and size + f.size > target:
            batches.append(batch)
            batch, size = [], 0
        batch.append(f)
        size += f.size
    return batches + ([batch] if batch else [])


#: The box's sender of one batch (batch_send_command): stdin holds the NUL-separated
#: `size mtime path` records of the listing; stdout is a tar of exactly the listed bytes of
#: each file (an appended log is sent as listed, not as it grew since), its listed mtime
#: kept, and BATCH_MANIFEST last. A file renamed away or shrunk since the listing is skipped
#: (and named so); the next listing has whatever replaced it.
SEND_SCRIPT = r"""
import hashlib, io, json, os, sys, tarfile

class Hashed:
    def __init__(self, f):
        self.f, self.sha = f, hashlib.sha256()
    def read(self, n=-1):
        data = self.f.read(n)
        self.sha.update(data)
        return data

sent, skipped = {}, []
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.GNU_FORMAT) as tar:
    for record in filter(None, sys.stdin.buffer.read().split(b"\0")):
        size, mtime, path = record.decode().split(" ", 2)
        try:
            f = open(path, "rb")
        except OSError:
            skipped.append(path)
            continue
        with f:
            if os.fstat(f.fileno()).st_size < int(size):
                skipped.append(path)
                continue
            info = tarfile.TarInfo(path)
            info.size, info.mtime = int(size), int(mtime)
            reader = Hashed(f)
            tar.addfile(info, reader)
            sent[path] = [int(size), reader.sha.hexdigest()]
    body = json.dumps({"files": sent, "skipped": skipped}).encode()
    info = tarfile.TarInfo(sys.argv[1])
    info.size = len(body)
    tar.addfile(info, io.BytesIO(body))
"""


def batch_send_command(remote_dir: str) -> str:
    """The box's command that sends one batch (SEND_SCRIPT)."""
    return (f"cd {remote_dir} && python3 -c {shlex.quote(SEND_SCRIPT)} "
            f"{shlex.quote(BATCH_MANIFEST)}")


def receive_batch(stream, local_dir: Path) -> dict[str, bool]:
    """One batch's tar (SEND_SCRIPT) from `stream` into `local_dir`: the settled paths, True
    for each that landed whole, False for each the box skipped. Every file is written under
    a dot temporary beside its destination and hashed on the way; only when the batch's
    manifest has arrived are the files whose size and SHA-256 match it renamed into place.
    A link that dies mid-batch, a truncated or damaged file, leaves nothing under its real
    name (2026-11-27: tar extraction of a cut link left streams truncated at 512-byte
    blocks, which then scored as decoder mismatches), so the retry fetches it again."""
    staged: dict[str, tuple[Path, Path, int, str, int]] = {}
    settled: dict[str, bool] = {}
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if member.name == BATCH_MANIFEST:
                    manifest = json.loads(tar.extractfile(member).read())
                    for path, (size, digest) in manifest["files"].items():
                        if path in staged and staged[path][2:4] == (size, digest):
                            temp, final, _, _, mtime = staged.pop(path)
                            os.utime(temp, (mtime, mtime))
                            os.replace(temp, final)
                            settled[path] = True
                    settled.update({path: False for path in manifest["skipped"]})
                    break
                if member.isfile():
                    staged[member.name] = stage_member(tar, member, local_dir)
    except (tarfile.TarError, OSError, ValueError, KeyError):
        pass  # a cut link: whatever had not been confirmed is fetched again
    finally:
        for temp, *_ in staged.values():
            temp.unlink(missing_ok=True)
    return settled


def stage_member(tar: tarfile.TarFile, member: tarfile.TarInfo, local_dir: Path
                 ) -> tuple[Path, Path, int, str, int]:
    """One received file under a dot temporary beside its destination: (temporary,
    destination, bytes received, their SHA-256, the listed mtime)."""
    relative = Path(member.name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe path in a batch: {member.name}")
    final = local_dir / relative
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = final.with_name(f".{final.name}.incoming")
    sha, size = hashlib.sha256(), 0
    source = tar.extractfile(member)
    try:
        with temp.open("wb") as sink:
            while chunk := source.read(RECEIVE_CHUNK):
                sha.update(chunk)
                size += len(chunk)
                sink.write(chunk)
    except BaseException:  # cut mid-file: the temporary goes too
        temp.unlink(missing_ok=True)
        raise
    return temp, final, size, sha.hexdigest(), int(member.mtime)


def advance_marker_command(remote_dir: str, marker: str) -> str:
    """The box's checkpoint moves on to the moment its last pull was listed."""
    return f"cd {remote_dir} && mv {marker}.next {marker}"


def try_launch_lock(region: str, family: str = ""):
    """This machine's launch lock of `region` (per instance family when given) if free now (an
    open handle), else None. Per family, so drivers hunting different chips do not queue behind
    each other (2026-11-26: 12 hunting runs let c8i try each zone only every ~7 min); the
    region's spot quota is far above two metal machines, so the race the lock guards is moot."""
    path = launch_lock_path(f"{region}-{family}" if family else region)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    if _try_lock(handle):
        return handle
    handle.close()
    return None


def _try_lock(handle) -> bool:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def release_launch_lock(handle) -> None:
    if handle is not None and not handle.closed:
        handle.close()


def per_vcpu(ec2, candidate: Candidate) -> float:
    """A candidate's cheapest zone price per vCPU (USD/h)."""
    return min(candidate.zones.values()) / instance_vcpus(ec2, candidate.instance_type)


def spot_quota(region: str) -> int:
    """The account's standard-family spot quota in `region` (vCPUs)."""
    import boto3
    quotas = boto3.client("service-quotas", region_name=region)
    return int(quotas.get_service_quota(ServiceCode="ec2", QuotaCode=SPOT_QUOTA_CODE)
               ["Quota"]["Value"])


_VCPUS: dict[str, int] = {}


def instance_vcpus(ec2, instance_type: str) -> int:
    if instance_type not in _VCPUS:
        info = ec2.describe_instance_types(InstanceTypes=[instance_type])["InstanceTypes"][0]
        _VCPUS[instance_type] = info["VCpuInfo"]["DefaultVCpus"]
    return _VCPUS[instance_type]


def purpose_vcpus(ec2, purpose: str) -> int:
    """vCPUs of this project's live instances tagged with `purpose` (every live state)."""
    used = 0
    pages = ec2.get_paginator("describe_instances").paginate(Filters=[
        {"Name": "tag:Purpose", "Values": [purpose, f"{purpose}-*"]},  # and its plans
        {"Name": "instance-state-name",
         "Values": ["pending", "running", "stopping", "shutting-down"]}])
    for page in pages:
        for reservation in page["Reservations"]:
            for instance in reservation["Instances"]:
                cpu = instance.get("CpuOptions") or {}
                used += cpu.get("CoreCount", 0) * cpu.get("ThreadsPerCore", 1)
    return used


def spot_usage(ec2) -> tuple[int, bool]:
    """(vCPUs the account's standard-family spot instances hold, whether the build box
    is among them): every live state counts, as the quota counts them."""
    used, build_box = 0, False
    pages = ec2.get_paginator("describe_instances").paginate(Filters=[
        {"Name": "instance-lifecycle", "Values": ["spot"]},
        {"Name": "instance-state-name",
         "Values": ["pending", "running", "stopping", "shutting-down"]}])
    for page in pages:
        for reservation in page["Reservations"]:
            for instance in reservation["Instances"]:
                if instance["InstanceType"][0] not in STANDARD_FAMILIES:
                    continue
                cpu = instance.get("CpuOptions") or {}
                used += cpu.get("CoreCount", 0) * cpu.get("ThreadsPerCore", 1)
                tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
                build_box |= tags.get("Purpose") == BUILD_BOX_PURPOSE
    return used, build_box


class RemoteFailure(RuntimeError):
    """A remote job failed, stalled or reported too many errors; carries the log tail."""

    def __init__(self, reason: str, tail: list[str]):
        super().__init__(reason + "\n  remote log tail:\n" + "\n".join(f"    {t}" for t in tail))


class SharedClients:
    """boto3 clients shared by the sessions of one process: a client pair costs about 30 MB,
    and botocore clients are safe to use from many threads once made (making them is not,
    hence the lock)."""

    def __init__(self) -> None:
        import boto3

        self._boto, self._lock, self._made = boto3.session.Session(), threading.Lock(), {}

    def __call__(self, service: str, region: str):
        with self._lock:
            if (service, region) not in self._made:
                self._made[service, region] = self._boto.client(service, region_name=region)
            return self._made[service, region]


_SESSION_COUNT = itertools.count()


def _main_thread() -> bool:
    """Signal handlers can be set only from the main thread."""
    return threading.current_thread() is threading.main_thread()


def _raise_exit(signum, _frame) -> None:
    raise SystemExit(f"signal {signum}")


def ledger_row(resources: str, event: str, start: str, end: str, cost: str, spec: SessionSpec,
               spent: float | None = None) -> None:
    """Append one row to the ledger table; with `spent`, add it to "Spent so far".
    Sessions run side by side, so the read-modify-write holds the ledger's lock."""
    with open(LEDGER.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _ledger_row_locked(resources, event, start, end, cost, spec, spent)


def _ledger_row_locked(resources: str, event: str, start: str, end: str, cost: str,
                       spec: SessionSpec, spent: float | None) -> None:
    text = LEDGER.read_text()
    row = (f"| {iso(utc_now())} | {resources} | {spec.instance_type} / {spec.region} | "
           f"{spec.purpose} ({event}) | {start} | {end} | {cost} |")
    marker = re.search(r"\n\*\*Spent so far: \$([0-9.]+) of \$([0-9.]+)\.\*\*", text)
    if marker is None:
        raise RuntimeError(f"{LEDGER} has no 'Spent so far' line")
    total = float(marker.group(1)) + (spent or 0.0)
    body = text[:marker.start()].rstrip("\n")
    LEDGER.write_text(f"{body}\n{row}\n\n**Spent so far: ${total:.2f} of ${marker.group(2)}.**\n")


# --- the ledger by purpose (`python3 tools/dev/aws_session.py ledger`) ---------------------

#: A ledger row's cells: logged, resources, type / region, purpose (event), start, end, cost.
LEDGER_ROW = re.compile(r"^\| (\S+) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|$")
#: How `ledger --by` groups rows (RCA action R6: spend by purpose for a time window).
LEDGER_GROUPS = ("purpose", "kind", "type", "region")


@dataclass(frozen=True)
class Charge:
    """One ledger row that books money: its end, cost and what it was for."""
    ended: str
    cost: float
    purpose: str
    instance_type: str
    region: str

    def group(self, by: str) -> str:
        if by == "kind":  # the purpose's first word: an agent's or a tool's runs together
            return re.split(r"[-_ ]", self.purpose, maxsplit=1)[0]
        return {"purpose": self.purpose, "type": self.instance_type,
                "region": self.region}[by]


def ledger_charges(text: str) -> list[Charge]:
    """The rows that book a cost (a session's end, a box's end, an image removed)."""
    charges = []
    for line in text.splitlines():
        row = LEDGER_ROW.match(line)
        if row is None or not row.group(7).startswith("$"):
            continue
        kind, _, region = row.group(3).partition(" / ")
        purpose = re.sub(r"\s*\(.*\)$", "", row.group(4))
        end = row.group(6) if row.group(6) not in ("—", "") else row.group(1)
        charges.append(Charge(end, float(row.group(7)[1:]), purpose, kind, region))
    return charges


def ledger_summary(charges: list[Charge], since: str, until: str | None, by: str) -> str:
    """Spend per group for the rows that ended in [since, until), largest first."""
    totals: dict[str, float] = {}
    for charge in charges:
        if charge.ended >= since and (until is None or charge.ended < until):
            totals[charge.group(by)] = totals.get(charge.group(by), 0.0) + charge.cost
    lines = [f"spend by {by}, rows ended from {since}" + (f" to {until}" if until else "")]
    lines += [f"  ${cost:8.2f}  {name}" for name, cost in
              sorted(totals.items(), key=lambda item: -item[1])]
    lines.append(f"  ${sum(totals.values()):8.2f}  total")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="the AWS ledger's spend, grouped")
    commands = parser.add_subparsers(dest="command", required=True)
    ledger = commands.add_parser("ledger", help="spend by purpose (or kind, type, region)")
    ledger.add_argument("--since", required=True,
                        help="UTC start, e.g. 2026-11-24T06:54 (ISO; compared as text)")
    ledger.add_argument("--until", help="UTC end (default: now)")
    ledger.add_argument("--by", choices=LEDGER_GROUPS, default="purpose")
    args = parser.parse_args(argv)
    print(ledger_summary(ledger_charges(LEDGER.read_text()), args.since, args.until, args.by))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
