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
     instance (no persistent request) on the official Ubuntu LTS AMI:
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
import re
import shlex
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
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
    tags: dict[str, str] = field(default_factory=dict)


class BudgetExceeded(RuntimeError):
    pass


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
        self.ec2 = boto3.client("ec2", region_name=spec.region)
        self.ssm = boto3.client("ssm", region_name=spec.region)
        self.name = f"nanocodec-{spec.purpose}-{utc_now():%Y%m%d-%H%M%S}"
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
        self.expected_end = utc_now() + datetime.timedelta(minutes=spec.lifetime_minutes + 15)

    # --- lifecycle -------------------------------------------------------------------

    def __enter__(self) -> "Session":
        self._previous_handlers = {sig: signal.signal(sig, _raise_exit)
                                   for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            zones = self.zone_prices()
            max_price = min(zones.values()) * MAX_PRICE_FACTOR
            cap = self.cost(max_price, self.spec.lifetime_minutes / 60)
            say(f"{self.spec.instance_type} in {self.spec.region}: spot from "
                f"${min(zones.values()):.4f}/h, bid cap ${max_price:.4f}/h, worst case ${cap:.2f} "
                f"for the {self.spec.lifetime_minutes}-minute lifetime")
            if cap > self.spec.budget_usd:
                raise BudgetExceeded(f"worst case ${cap:.2f} exceeds the session budget "
                                     f"${self.spec.budget_usd:.2f}")
            self.create_access()
            ledger_row(self.row_resources(), "started", iso(utc_now()), "—",
                       f"cap ${cap:.2f}", self.spec)
            self.launch(zones, max_price)
            self.wait_for_ssh()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        # A second Ctrl-C must not interrupt the cleanup.
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.cleanup()
        finally:
            signal.signal(signal.SIGINT, previous_int)
            for sig, handler in getattr(self, "_previous_handlers", {}).items():
                signal.signal(sig, handler)

    def tags(self, resource: str) -> list[dict]:
        tags = {"Project": PROJECT_TAG, "Owner": OWNER_TAG, "Purpose": self.spec.purpose,
                "ExpectedEnd": iso(self.expected_end), "Name": self.name, **self.spec.tags}
        return [{"ResourceType": resource,
                 "Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}]

    def zone_prices(self) -> dict[str, float]:
        history = self.ec2.describe_spot_price_history(
            InstanceTypes=[self.spec.instance_type], ProductDescriptions=["Linux/UNIX"],
            StartTime=utc_now())["SpotPriceHistory"]
        prices = {h["AvailabilityZone"]: float(h["SpotPrice"]) for h in history}
        if not prices:
            raise RuntimeError(f"no spot price for {self.spec.instance_type} in "
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

    def launch(self, zones: dict[str, float], max_price: float) -> None:
        from botocore.exceptions import ClientError
        image_id, root_device = self.ami()
        # The hard lifetime comes first: it holds even if this process dies.
        user_data = f"#!/bin/bash\nshutdown -h +{self.spec.lifetime_minutes}\n"
        for zone in zones:  # cheapest first; a zone without capacity is skipped
            subnets = self.ec2.describe_subnets(Filters=[
                {"Name": "availability-zone", "Values": [zone]},
                {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
            if not subnets:
                continue
            try:
                reply = self.ec2.run_instances(
                    ImageId=image_id, InstanceType=self.spec.instance_type, MinCount=1,
                    MaxCount=1, KeyName=self.key_name, UserData=user_data,
                    InstanceInitiatedShutdownBehavior="terminate",
                    InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {
                        "SpotInstanceType": "one-time", "MaxPrice": f"{max_price:.4f}",
                        "InstanceInterruptionBehavior": "terminate"}},
                    BlockDeviceMappings=[{"DeviceName": root_device, "Ebs": {
                        "VolumeSize": self.spec.root_gb, "VolumeType": "gp3",
                        "DeleteOnTermination": True}}],
                    NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnets[0]["SubnetId"],
                                        "Groups": [self.group_id],
                                        "AssociatePublicIpAddress": True}],
                    TagSpecifications=[*self.tags("instance"), *self.tags("volume"),
                                       *self.tags("spot-instances-request")])
            except ClientError as error:
                code = error.response["Error"]["Code"]
                if code in ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow",
                            "Unsupported", "InsufficientCapacity"):
                    say(f"{zone}: {code}, trying the next zone")
                    continue
                raise
            self.instance_id = reply["Instances"][0]["InstanceId"]
            self.spot_request = reply["Instances"][0].get("SpotInstanceRequestId")
            self.zone, self.price = zone, zones[zone]
            break
        if self.instance_id is None:
            raise RuntimeError(f"no zone could launch {self.spec.instance_type}")
        say(f"launched {self.instance_id} in {self.zone} (spot ${self.price:.4f}/h)")
        self.ec2.get_waiter("instance_running").wait(
            InstanceIds=[self.instance_id], WaiterConfig={"Delay": 10, "MaxAttempts": 90})
        instance = self.describe()
        self.host = instance["PublicIpAddress"]
        self.launched = instance["LaunchTime"]
        say(f"{self.instance_id} running at {self.host}")

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

    def cleanup(self) -> None:
        ended = utc_now()
        if self.instance_id:
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

    def download_new(self, remote_dir: str, local_dir: Path, marker: str) -> int:
        """The files under `remote_dir` changed since the last call with `marker` (all of
        them on the first) into `local_dir`: a checkpoint that a spot reclaim cannot
        take back. Returns the bytes moved (billed as egress)."""
        local_dir.mkdir(parents=True, exist_ok=True)
        newer = f"-newer {marker}"
        command = (f"cd {remote_dir} && touch {marker}.next && "
                   f"{{ if [ -f {marker} ]; then find . -type f {newer}; else find . -type f; fi; }}"
                   f" | tar -cf - -T - && mv {marker}.next {marker}")
        ssh = subprocess.Popen([*self.ssh_base(), command], stdout=subprocess.PIPE)
        untar = subprocess.Popen(["tar", "-C", str(local_dir), "-xf", "-"],
                                 stdin=subprocess.PIPE)
        moved = 0
        while chunk := ssh.stdout.read(1 << 20):
            moved += len(chunk)
            untar.stdin.write(chunk)
        untar.stdin.close()
        if ssh.wait() != 0 or untar.wait() != 0:
            raise RuntimeError(f"checkpoint download of {remote_dir} failed")
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
