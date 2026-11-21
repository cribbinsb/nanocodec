#!/usr/bin/env python3
"""Warm measurement boxes: spot boxes that stay up between aws_measure jobs.

A one-off box (aws_measure --cold) pays ~3 min of upload and ~1.4 min of setup per
job. A warm box is set up once and keeps the clips, this machine's ffmpeg, libvmaf,
the Python environment and the point cache; a later job syncs only what changed
(rsync: its tree, builds and any clip the box does not have) and starts at once.

    python3 tools/dev/aws_warm.py status     # the warm boxes, their jobs, idle and cost
    python3 tools/dev/aws_warm.py stop       # terminate every warm box now
    python3 tools/dev/aws_warm.py reap       # clean up boxes that have stopped

Lifecycle (never left running):
  - Registry: one JSON record per box in REGISTRY (aws_session.Session.detach), under a
    lock; a job adds itself to its box's record and removes itself when it ends.
  - A box takes JOB_CPUS-wide jobs up to its vCPUs (the slot engine on the box shares
    its cores between them); when every box is full, or has less than
    JOB_MARGIN_MINUTES of its lifetime left, a new one is launched (quota-aware:
    aws_session waits in line for spot quota).
  - On the box an idle timer (`idle_watch_command`) deletes the clips, builds and points
    and shuts the box down (which terminates it) after IDLE_MINUTES without a job; the
    spot request's hard `shutdown -h +LIFETIME_MINUTES` is the last resort.
  - Here every aws_measure call (and `reap`) cleans up boxes that have stopped: their
    key pair and security group are deleted and the ledger row is written with the
    cost up to when the box stopped.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import re
import shlex
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aws_session import (PROJECT_TAG, SPILL_REGIONS, Session, SessionSpec,  # noqa: E402
                         instance_vcpus, say, utc_now)

#: The home region; warm boxes spill to SPILL_REGIONS when its quota is full
#: (aws_session.Session.claim_room). The registry holds every region's boxes.
HOME_REGION = "eu-north-1"

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
REGISTRY = DATA / "run" / "warm-boxes"
#: Box defaults: 96 vCPU (aws_measure's largest auto size), the smaller sizes when the
#: quota has no room for it; a root volume for the whole pool's clips (it grows job by
#: job); the hard lifetime, and the idle time after which the box stops itself.
WARM_TYPES = ("c7a.24xlarge", "c7i.24xlarge", "m7a.24xlarge", "c7a.16xlarge",
              "c7i.16xlarge", "m7a.16xlarge", "c7a.12xlarge", "c7i.12xlarge")
#: The sizes of an instance family WARM_TYPES does not list (a study pinned to it).
WARM_SIZES = ("24xlarge", "16xlarge", "12xlarge")
WARM_ROOT_GB = 120
LIFETIME_MINUTES = 240
IDLE_MINUTES = 20
#: The CPU one job is counted as (a box of V vCPUs takes V // JOB_CPUS jobs at once).
JOB_CPUS = 32
#: A box with less lifetime left than this takes no new job (a job's 30-min budget,
#: its setup and its download).
JOB_MARGIN_MINUTES = 50
WARM_PURPOSE = "measure-warm"
#: How often a waiting job looks again.
WAIT_SECONDS = 15
#: Jobs a launch in progress takes on (a 96-vCPU box's slots; the box keeps as many as
#: its size allows, the rest look again).
NOMINAL_JOBS = 3
#: The fleet-wide cap on the warm boxes' cost, $/h over every region (coordinator,
#: 2026-11-21; NC_WARM_SPEND_CAP or --warm-spend-cap), the cost assumed for a box still
#: being launched, and the root volume and IPv4 on top of an instance's price.
SPEND_CAP_USD_PER_HOUR = float(os.environ.get("NC_WARM_SPEND_CAP", "6"))
NEW_BOX_RATE_ESTIMATE = 1.0
BOX_OVERHEAD = 1.03
#: Draining: a box costing more than DRAIN_FACTOR times a home-region box per vCPU takes
#: no new job and idles out (coordinator, 2026-11-21: a $1.80/h spilled box stayed up
#: over an hour as jobs kept packing onto it, while home boxes cost $0.52-0.63/h). The
#: home reference is the cheapest home box registered, else HOME_USD_PER_VCPU_HOUR
#: (a 96-vCPU c7a/c7i spot box in eu-north-1 on 2026-11-21, $0.52-0.63/h).
DRAIN_FACTOR = 1.5
HOME_USD_PER_VCPU_HOUR = 0.60 / 96
#: No draining for this long after a launch spilled out of the home region (its quota
#: was full then: a new box would spill too, so draining would only launch more).
SPILL_MEMORY_MINUTES = 30
#: Where the box marks running jobs and its last activity (the idle timer reads them).
BUSY_FILE = "~/.warm-busy"
JOB_MARK = "RUNNING"


@dataclass
class Box:
    record: dict

    @property
    def instance_id(self) -> str:
        return self.record["instance_id"]

    @property
    def jobs(self) -> dict[str, int]:
        return self.record.setdefault("jobs", {})

    def capacity(self) -> int:
        return max(1, self.record["vcpus"] // JOB_CPUS)

    def minutes_left(self) -> float:
        launched = datetime.datetime.fromisoformat(self.record["launched"].replace("Z", "+00:00"))
        return LIFETIME_MINUTES - (utc_now() - launched).total_seconds() / 60

    def takes_job(self) -> bool:
        return (self.record.get("prepared") and len(self.jobs) < self.capacity()
                and self.minutes_left() > JOB_MARGIN_MINUTES)

    def per_vcpu(self) -> float:
        """$/vCPU-hour (unknown price: none, so never drained)."""
        return (self.record.get("price") or 0.0) / max(1, self.record.get("vcpus") or 1)

    def region(self) -> str:
        return self.record.get("spec", {}).get("region", HOME_REGION)

    def session(self) -> Session:
        return Session.attach(self.record)


def path_of(instance_id: str) -> Path:
    return REGISTRY / f"{instance_id}.json"


def save(box: Box) -> None:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    partial = path_of(box.instance_id).with_suffix(".partial")
    partial.write_text(json.dumps(box.record, indent=1) + "\n")
    partial.rename(path_of(box.instance_id))


def boxes() -> list[Box]:
    return [Box(json.loads(p.read_text())) for p in sorted(REGISTRY.glob("i-*.json"))]


@contextmanager
def registry_lock():
    REGISTRY.mkdir(parents=True, exist_ok=True)
    with (REGISTRY / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stopped_at(instance: dict | None) -> datetime.datetime | None:
    """When a box stopped (its state transition's time), else None while it runs."""
    if instance is None:
        return utc_now()  # gone from EC2 already: at most an hour ago
    if instance["State"]["Name"] in ("pending", "running"):
        return None
    found = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)",
                      instance.get("StateTransitionReason", ""))
    if not found:
        return utc_now()
    return datetime.datetime.strptime(found.group(1), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=datetime.timezone.utc)


def describe(session: Session) -> dict | None:
    try:
        return session.describe()
    except Exception as error:  # botocore's ClientError, without importing botocore
        code = (getattr(error, "response", None) or {}).get("Error", {}).get("Code", "")
        if "NotFound" in code:
            return None
        raise


def reap() -> None:
    """Clean up every registered box that has stopped, and terminate any box whose
    launch was abandoned (not prepared, its launcher gone). Call under the lock."""
    for box in boxes():
        session = box.session()
        ended = stopped_at(describe(session))
        abandoned = not box.record.get("prepared") and not alive(box.record["owner"])
        if ended is None and not abandoned:
            box.record["jobs"] = {j: p for j, p in box.jobs.items() if alive(p)}
            save(box)
            continue
        say(f"warm box {box.instance_id}: "
            + ("stopped itself" if ended else "abandoned while being set up") + "; cleaning up")
        session.cleanup(ended_at=ended)
        path_of(box.instance_id).unlink(missing_ok=True)


def acquire(job: str, spec: SessionSpec, setup,
            spend_cap: float = SPEND_CAP_USD_PER_HOUR,
            avoid: set[str] = frozenset(), family: str | None = None) -> tuple[Box, Session, bool]:
    """A warm box for `job`, the job registered on it. In order: a ready box with a free
    job slot; a box being set up, or a launch in progress, with a free slot (queued jobs
    pack onto it: on 2026-11-21 six waiters each launched a box within a minute, $8-10/h
    where two or three would have carried them); else a new box, launched and set up with
    `setup(session)`, unless the warm boxes together already cost `spend_cap` $/h (then
    the job waits, saying so). Boxes in `avoid` (one that would not take a sync) are not
    used. `family` (an instance family, e.g. c7a): only boxes of it, and only launches
    asked for it (a tune study pinned to one CPU). The lock is never held across a
    launch or a setup. Returns (box, session, fresh)."""
    waited = 0.0
    while True:
        capped = False
        with registry_lock():
            reap()
            found = registered(job) or place(job, avoid, family)
            if isinstance(found, Box) and found.record.get("prepared"):
                return found, found.session(), False
            if found is None:
                rate = spend_rate()
                if rate + NEW_BOX_RATE_ESTIMATE <= spend_cap:
                    marker = REGISTRY / f"launching-{os.getpid()}-{job}.json"
                    marker.write_text(json.dumps({"owner": os.getpid(), "family": family,
                                                  "jobs": {job: os.getpid()}}))
                    break
                capped = True
                say(f"WAITING: the warm boxes cost ${rate:.2f}/h; a new one would pass the "
                    f"${spend_cap:.2f}/h cap (--warm-spend-cap): queued ({waited:.0f} s)")
        if not capped:
            say(f"waiting for the warm box being set up or launched ({waited:.0f} s)")
        time.sleep(WAIT_SECONDS)
        waited += WAIT_SECONDS
    try:
        session = Session(spec)
        session.__enter__()  # quota-aware launch, the cheapest pool per vCPU
        with registry_lock():
            vcpus = instance_vcpus(session.ec2, session.spec.instance_type)
            box = Box({**session.detach(), "vcpus": vcpus, "owner": os.getpid(),
                       "prepared": False, "jobs": {}})
            note_launch(box.region())
            joined = json.loads(marker.read_text())["jobs"] if marker.exists() else {}
            # this job first, then the jobs that joined the launch, up to the box's slots
            for name, pid in [(job, os.getpid()), *((n, p) for n, p in joined.items()
                                                    if n != job and alive(p))]:
                if len(box.jobs) < box.capacity():
                    box.jobs[name] = pid
            save(box)
            marker.unlink(missing_ok=True)
    finally:
        marker.unlink(missing_ok=True)
    try:
        setup(session)
    except BaseException:
        say(f"warm box {box.instance_id}: setup failed; terminating it")
        session.cleanup()
        path_of(box.instance_id).unlink(missing_ok=True)
        raise
    with registry_lock():
        box = Box({**json.loads(path_of(box.instance_id).read_text()), "prepared": True})
        save(box)
    return box, session, True


def registered(job: str):
    """Where `job` is already counted: a box (ready or being set up), "launch" (a
    launch in progress), or None."""
    for box in boxes():
        if job in box.jobs:
            return box
    for marker in launching():
        if job in json.loads(marker.read_text())["jobs"]:
            return "launch"
    return None


def place(job: str, avoid: set[str] = frozenset(), family: str | None = None):
    """Count `job` on a ready box, a box being set up, or a launch in progress with a
    free slot (in that order; of `family` only, when given): the box or "launch" it
    went to, else None."""
    every = boxes()
    reference = home_rate(every)
    drain = not recently_spilled()
    usable = []
    for box in every:
        if box.instance_id in avoid or (family is not None and not of_family(
                box.record["spec"]["instance_type"], family)):
            continue
        if drain and box.per_vcpu() > DRAIN_FACTOR * reference:
            if not box.record.get("draining"):
                box.record["draining"] = True
                save(box)
                say(f"warm box {box.instance_id} ({box.region()}, ${box.record['price']:.2f}/h) "
                    f"draining: ${box.per_vcpu() * 96:.2f} per 96 vCPU-h against "
                    f"${reference * 96:.2f} at home; it takes no new job and idles out")
            continue
        usable.append(box)
    usable.sort(key=Box.per_vcpu)  # the cheapest per vCPU first
    for box in usable:
        if box.takes_job():
            box.jobs[job] = os.getpid()
            save(box)
            say(f"warm box {box.instance_id} ({box.record['spec']['instance_type']}, "
                f"{box.record['spec'].get('region', '?')}): job {len(box.jobs)} of "
                f"{box.capacity()}, {box.minutes_left():.0f} min of its lifetime left")
            return box
    for box in usable:
        if not box.record.get("prepared") and alive(box.record["owner"]) and \
                len(box.jobs) < box.capacity():
            box.jobs[job] = os.getpid()
            save(box)
            say(f"joining warm box {box.instance_id}, being set up")
            return box
    for marker in launching():
        state = json.loads(marker.read_text())
        if len(state["jobs"]) < NOMINAL_JOBS and (family is None
                                                  or state.get("family") == family):
            state["jobs"][job] = os.getpid()
            marker.write_text(json.dumps(state))
            say(f"joining a warm box being launched ({len(state['jobs'])} of {NOMINAL_JOBS})")
            return "launch"
    return None


def note_launch(region: str) -> None:
    (REGISTRY / "last-launch.json").write_text(json.dumps({"region": region,
                                                           "at": time.time()}))


def recently_spilled() -> bool:
    try:
        last = json.loads((REGISTRY / "last-launch.json").read_text())
    except (OSError, ValueError):
        return False
    return last["region"] != HOME_REGION and \
        time.time() - last["at"] < 60 * SPILL_MEMORY_MINUTES


def home_rate(every: list[Box]) -> float:
    """$/vCPU-hour of a home-region box: the cheapest registered, else the estimate."""
    home = [b.per_vcpu() for b in every if b.region() == HOME_REGION and b.per_vcpu() > 0]
    return min(home, default=HOME_USD_PER_VCPU_HOUR)


def spend_rate() -> float:
    """$/h of every warm box, running or being set up, and of the launches in progress."""
    running = sum((b.record.get("price") or 0.0) * BOX_OVERHEAD for b in boxes())
    return running + NEW_BOX_RATE_ESTIMATE * len(launching())


def launching() -> list[Path]:
    """Launch markers of live processes (a dead launcher's marker is removed)."""
    live = []
    for marker in REGISTRY.glob("launching-*.json"):
        try:
            owner = json.loads(marker.read_text())["owner"]
        except (OSError, ValueError, KeyError):
            continue
        if alive(owner):
            live.append(marker)
        else:
            marker.unlink(missing_ok=True)
    return live


def release(box: Box, job: str) -> None:
    with registry_lock():
        current = path_of(box.instance_id)
        if current.exists():
            fresh = Box(json.loads(current.read_text()))
            fresh.jobs.pop(job, None)
            save(fresh)


def of_family(instance_type: str, family: str) -> bool:
    return instance_type.split(".")[0] == family


def family_types(family: str | None) -> tuple[str, ...]:
    """WARM_TYPES, or those of `family` (its WARM_SIZES when WARM_TYPES has none)."""
    if family is None:
        return WARM_TYPES
    return tuple(t for t in WARM_TYPES if of_family(t, family)) or \
        tuple(f"{family}.{size}" for size in WARM_SIZES)


def warm_spec(region: str, budget: float, family: str | None = None) -> SessionSpec:
    types = family_types(family)
    return SessionSpec(region=region, instance_type=types[0], arch="amd64",
                       purpose=WARM_PURPOSE, lifetime_minutes=LIFETIME_MINUTES,
                       budget_usd=budget, root_gb=WARM_ROOT_GB,
                       fallback_types=types[1:])


def idle_watch_command(cleanup: str, idle_minutes: int = IDLE_MINUTES) -> str:
    """The box's idle timer, started once in its setup: every 30 s a job marked
    running (a JOB_MARK file touched in the last hour) renews BUSY_FILE; after
    `idle_minutes` without one, `cleanup` (the clips, builds and points) and shutdown,
    which terminates the box."""
    loop = (f"touch {BUSY_FILE}; while sleep 30; do "
            f"if [ -n \"$(find ~/jobs -name {JOB_MARK} -mmin -60 2>/dev/null)\" ]; then "
            f"touch {BUSY_FILE}; fi; "
            f"if [ $(( $(date +%s) - $(stat -c %Y {BUSY_FILE}) )) -ge {idle_minutes * 60} ]; "
            f"then {cleanup}; sudo shutdown -h now; fi; done")
    # Single-quoted (the loop's $(...) run on each pass, not once when it starts), and
    # grouped, so that the `; echo $?` run_detached appends follows a complete command.
    return f"{{ setsid nohup bash -c {shlex.quote(loop)} > ~/idle-watch.log 2>&1 < /dev/null & }}"


def status(_: argparse.Namespace) -> None:
    with registry_lock():
        reap()
        found = boxes()
    for box in found:
        session = box.session()
        idle = session.ssh(f"echo $(( $(date +%s) - $(stat -c %Y {BUSY_FILE}) ))",
                           check=False, quiet=True).stdout.strip()
        hours = (LIFETIME_MINUTES - box.minutes_left()) / 60
        print(f"{box.instance_id} {box.record['spec']['instance_type']} {box.record['host']}: "
              f"{len(box.jobs)} job(s) of {box.capacity()}, idle {idle or '?'} s, "
              f"{box.minutes_left():.0f} min of lifetime left, "
              f"${session.cost(box.record['price'] or 0.0, hours):.2f} so far, "
              f"${session.cost(box.record['price'] or 0.0, 1.0):.2f}/h"
              + (", draining" if box.record.get("draining") else ""))
    print(f"{len(found)} warm box(es)")
    known = {b.instance_id for b in found}
    for region in (HOME_REGION, *SPILL_REGIONS.get(HOME_REGION, ())):
        for instance in tagged_instances(region):
            if instance["InstanceId"] not in known:
                tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
                print(f"also running in {region}: {instance['InstanceId']} "
                      f"{instance['InstanceType']} (Purpose {tags.get('Purpose')}, "
                      f"ExpectedEnd {tags.get('ExpectedEnd')})")


def tagged_instances(region: str) -> list[dict]:
    """Every live instance tagged Project=nanocodec in `region` (any tool's)."""
    import boto3
    ec2 = boto3.client("ec2", region_name=region)
    pages = ec2.get_paginator("describe_instances").paginate(Filters=[
        {"Name": "tag:Project", "Values": [PROJECT_TAG]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]}])
    return [i for page in pages for r in page["Reservations"] for i in r["Instances"]]


def stop(_: argparse.Namespace) -> None:
    with registry_lock():
        for box in boxes():
            say(f"stopping warm box {box.instance_id}")
            box.session().cleanup()
            path_of(box.instance_id).unlink(missing_ok=True)


def reap_command(_: argparse.Namespace) -> None:
    with registry_lock():
        reap()
    print(f"{len(boxes())} warm box(es) running")


def main(argv: list[str]) -> None:
    sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))
    from nceval.cloud import check_fresh
    check_fresh()
    p = argparse.ArgumentParser(prog="aws_warm.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(run=status)
    sub.add_parser("stop").set_defaults(run=stop)
    sub.add_parser("reap").set_defaults(run=reap_command)
    args = p.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
