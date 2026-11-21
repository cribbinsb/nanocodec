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

import datetime
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

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
#: How often a waiting session re-reads the usage, and how long it waits in all before
#: it fails loudly (no silent fallback).
QUOTA_POLL_S = 30
QUOTA_QUEUE_MINUTES = 90
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
#: Large x86 machines for throughput jobs (encoding, scoring: any many-core x86 does), in
#: the order a session falls back through them when the one asked for has no spot
#: capacity in any zone or the quota refuses it (2026-11-20: no c7a.metal-48xl or
#: c7a.48xlarge in any eu-north-1 zone for two hours; a c7a.24xlarge launched at once).
#: Same Zen 4 first, the smaller one before the other vendors' (same AVX-512 tier).
THROUGHPUT_X86_TYPES = ("c7a.metal-48xl", "c7a.48xlarge", "c7a.24xlarge", "c7i.48xlarge",
                        "m7a.48xlarge")
#: Running costs beside the instance (eu-north-1 list prices, USD).
GP3_USD_PER_GB_MONTH = 0.0836
PUBLIC_IPV4_USD_PER_HOUR = 0.005
#: Data out of AWS to the internet (downloads); uploads are free.
EGRESS_USD_PER_GB = 0.09
HOURS_PER_MONTH = 730
#: How often a detached remote job is polled, and SSH connection settings.
POLL_SECONDS = 20
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
    #: Whether a spot session may launch in SPILL_REGIONS when its region is full.
    spill: bool = True


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

    def __init__(self, spec: SessionSpec):
        import boto3  # here: the instance side of a session imports `say` without boto3

        self.spec = spec
        self._client = lambda service, region: boto3.client(service, region_name=region)
        self.ec2 = self._client("ec2", spec.region)
        self.ssm = self._client("ssm", spec.region)
        # The pid too: two sessions of one purpose started in the same second (the
        # held-out chart's score parts, 2026-11-15) collided on the key pair name.
        self.name = f"nanocodec-{spec.purpose}-{utc_now():%Y%m%d-%H%M%S}-{os.getpid()}"
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
                                   for sig in (signal.SIGTERM, signal.SIGHUP)}
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
        # A second Ctrl-C must not interrupt the cleanup.
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        release_launch_lock(getattr(self, "_launch_lock", None))
        try:
            self.cleanup()
        finally:
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
        return session

    def tags(self, resource: str) -> list[dict]:
        tags = {"Project": PROJECT_TAG, "Owner": OWNER_TAG, "Purpose": self.spec.purpose,
                "ExpectedEnd": iso(self.expected_end), "Name": self.name, **self.spec.tags}
        return [{"ResourceType": resource,
                 "Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}]

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

    def ami(self) -> tuple[str, str]:
        name = AMI_PARAMETER.format(release=UBUNTU_RELEASE, arch=self.spec.arch)
        image_id = self.ssm.get_parameter(Name=name)["Parameter"]["Value"]
        image = self.ec2.describe_images(ImageIds=[image_id])["Images"][0]
        return image_id, image["RootDeviceName"]

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
        return free, (f"{self.spec.region}: {free} vCPU free of {limit} ({used} in use"
                      + (f", {reserve} kept for the build box" if reserve else "") + ")")

    def claim_room(self) -> list[Candidate]:
        """The region to launch in and its priced candidates (cheapest per vCPU first),
        holding that region's launch lock: of the regions (home, then SPILL_REGIONS)
        whose quota has room for a candidate, the one whose cheapest fitting candidate
        costs least per vCPU (2026-11-21: the first region that fitted was taken, and six
        spilled boxes started at 2.5x eu-north-1's price per vCPU); when none has room,
        wait and look again, failing loudly after QUOTA_QUEUE_MINUTES."""
        home, priced = self.spec.region, {}
        deadline = time.monotonic() + QUOTA_QUEUE_MINUTES * 60
        while True:
            notes, options, home_busy = [], [], False
            for region in self.regions():
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
                handle = try_launch_lock(region)
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
            if home_busy and time.monotonic() < deadline:
                # this machine's other launch holds the home region's lock for seconds:
                # wait for it rather than spill (2026-11-21: two scoring sessions started
                # together, one spilled to eu-west-1, found no capacity there and failed)
                for _, _, _, other, _ in options:
                    release_launch_lock(other)
                say(f"{home}: waiting for this machine's other launch there "
                    f"({LAUNCH_LOCK_POLL_S} s)")
                time.sleep(LAUNCH_LOCK_POLL_S)
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
                raise NoCapacity(f"waited {QUOTA_QUEUE_MINUTES} min for spot quota: "
                                 + "; ".join(notes))
            say("WAITING for spot quota in every region: " + "; ".join(notes)
                + f"; re-checking in {QUOTA_POLL_S} s")
            time.sleep(QUOTA_POLL_S)

    def launch(self, candidates: list[Candidate]) -> None:
        """Launch the first candidate type with capacity in any zone. A type without
        capacity anywhere, or refused by the spot quota, gives way to the next; when
        only the quota refused, the whole list is tried again after QUOTA_WAIT_S (the
        quota counts instances still shutting down)."""
        image_id, root_device = self.ami()
        # The hard lifetime comes first: it holds even if this process dies.
        user_data = f"#!/bin/bash\nshutdown -h +{self.spec.lifetime_minutes}\n"
        for attempt in range(QUOTA_RETRIES + 1):
            quota_full = False
            for index, candidate in enumerate(candidates):
                outcome = self.try_zones(candidate, image_id, root_device, user_data)
                if outcome == LAUNCHED:
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
        for zone in candidate.zones:  # cheapest first; a zone without capacity is skipped
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
        of the files brought back (under `local_dir`)."""
        local_dir.mkdir(parents=True, exist_ok=True)
        ssh = subprocess.Popen([*self.ssh_base(), checkpoint_command(remote_dir, marker)],
                               stdout=subprocess.PIPE)
        untar = subprocess.Popen(["tar", "-C", str(local_dir), "-xvf", "-"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        listing = []
        reader = threading.Thread(target=lambda: listing.extend(
            untar.stdout.read().decode().splitlines()))
        reader.start()
        moved = 0
        while chunk := ssh.stdout.read(1 << 20):
            moved += len(chunk)
            untar.stdin.write(chunk)
        untar.stdin.close()
        status = untar.wait()
        reader.join()
        if ssh.wait() != 0 or status != 0:
            raise RuntimeError(f"checkpoint download of {remote_dir} failed")
        if names is not None:
            names.extend(local_dir / name for name in listing if not name.endswith("/"))
        self.egress_bytes += moved
        return moved

    def run_detached(self, command: str, log: str, timeout_minutes: float,
                     stall_minutes: float = STALL_MINUTES, max_errors: int = 0) -> None:
        """Run `command` on the instance under nohup and wait for it, failing fast.

        The exit status lands in `log`.status, so a dropped SSH link loses nothing.
        Raises (with the log's tail) when the job fails, prints nothing new for
        `stall_minutes`, prints more than `max_errors` lines with ERROR_MARK, or
        runs past `timeout_minutes`."""
        job = shlex.quote(f"{command}; echo $? > {log}.status")
        # its own process group (setsid), so a failure stops all of it before any download
        self.ssh(f"rm -f {log}.status; setsid nohup bash -c {job} > {log} 2>&1 < /dev/null & "
                 f"echo $! > {log}.pid")
        try:
            self._watch(log, timeout_minutes, stall_minutes, max_errors)
        except BaseException:
            self.ssh(f"kill -TERM -- -$(cat {log}.pid) 2>/dev/null; sleep 2; "
                     f"kill -KILL -- -$(cat {log}.pid) 2>/dev/null; true", check=False)
            raise

    def _watch(self, log: str, timeout_minutes: float, stall_minutes: float,
               max_errors: int) -> None:
        deadline = time.monotonic() + timeout_minutes * 60
        shown, errors, last_news = 0, 0, time.monotonic()
        tail: list[str] = []
        while True:
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


def checkpoint_command(remote_dir: str, marker: str) -> str:
    """The box's side of download_new: a tar of the files under `remote_dir` newer than
    `marker` (all on the first call). Other jobs on a warm box write the same point
    cache at the same time, so their in-progress temporary files (dot files, renamed
    into place when complete) are left out, and a file that vanishes between the list
    and the tar is skipped, not an error (2026-11-21: one job's checkpoint failed on
    another's `.cbr150-....json.qlpmzmbo.times.txt`)."""
    files = "find . -type f ! -name '.*'"
    return (f"cd {remote_dir} && touch {marker}.next && "
            f"{{ if [ -f {marker} ]; then {files} -newer {marker}; else {files}; fi; }}"
            f" | tar --ignore-failed-read -cf - -T - 2>/dev/null && mv {marker}.next {marker}")


def try_launch_lock(region: str):
    """This machine's launch lock of `region` if free now (an open handle), else None."""
    path = launch_lock_path(region)
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
