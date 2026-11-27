#!/usr/bin/env python3
"""nc-remote: run heavy agent work (builds, tests, clippy, callgrind, conformance,
fuzzing) on one shared AWS build box instead of the PC.

    nc-remote run [--worktree DIR] [--name NAME] [--sync PATH ...] [--data PATH ...]
                  [--fetch PATH ...] [--fetch-to DIR] [--env KEY=VALUE ...] [--jobs N]
                  -- COMMAND ...
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
paths with private clips are refused. `--data` uploads a file or directory
as it is, whatever its size (the worktree sync leaves out files over
SYNC_MAX_SIZE, raw video and ignored files): a path in the worktree lands at
the same place in the job's tree, any other path under ~ at the same place
under the box's home; `--fetch` the same path to bring it back. A command
that names a path the sync left out gets a warning. NAME defaults to the
worktree's directory name.

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
- Finished outputs leave the box before any planned stop (2026-11-24: 19 finished
  PGO builds were lost with an idle box). A job's `--fetch` paths are recorded on
  the box when it ends (nc_remote_job.py UNFETCHED) and where they go on the PC
  (OUTPUTS_DIR, held by the run until its fetch is done). `run` fetches them and
  drops the marker; if it has gone (killed, PC restart, dropped link), `reap`
  fetches them to the same place. Until then the box's idle watchdog keeps the box
  up, reap, replace and `stop` do not end it (reap and stop fetch first), and only
  `stop --force`, the budget or the hard lifetime end it, naming what is lost.
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
#: The cost guide's x86 pair, cheapest per vCPU first (docs/aws-cost-guide.md; coordinator,
#: 2026-11-24: the box relaunched as c7a.24xlarge at $0.63/h, $6.54 per 1000 vCPU-h, against
#: c6i $4.55 and c7i $5.85; m7a/m7i $9.79-10.74). No c7a (owner: double-pumped AVX-512).
TYPE_VCPUS = {"c6i.24xlarge": 96, "c7i.24xlarge": 96}
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
#: (coordinator, 2026-11-21); each has its own state, key pair and history.
POOL_ENV = "NC_REMOTE_POOL"
OVERFLOW = "overflow"
POOL = os.environ.get(POOL_ENV, "")
OVERFLOW_AFTER_SECONDS = 120
OVERFLOW_POLL_SECONDS = 10
#: 30 min (time audit, 2026-11-25): at 10, 42 of 124 relaunches in 10 days came 2-30 min
#: after an idle end, each paying ~1-3 min to start plus cold builds from the baked image;
#: the 20 extra idle minutes cost ~$0.15-0.20 an end (c6i/c7i.24xlarge spot, ~$0.44-0.56/h).
IDLE_MINUTES = 30
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
#: Owner, 2026-11-26: spot only, never on-demand ("don't spend 10x more"). With no spot
#: capacity anywhere, the launch fails and is retried later; the rules above that would
#: choose on-demand are kept for the record but never act while this is False.
ON_DEMAND_ALLOWED = False
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
#: Every finished job's slot record (nc_remote_job.OCCUPANCY_LOG on the box), copied here by
#: `reap` and before a box is terminated, with the box's id and type; `occupancy` sums it up.
OCCUPANCY = STATE_DIR / "occupancy.jsonl"
#: Per box: the bytes of its log already copied.
OCCUPANCY_OFFSETS = STATE_DIR / "occupancy-offsets.json"
LOCK = STATE_DIR / "lock"
#: Held (flock) while a replacement box starts; holds that box's instance id.
REPLACING = STATE_DIR / "replacing"
#: One flock'd file per `run` and the box it uses (BoxUse).
USERS_DIR = STATE_DIR / "users"
#: Per `run` with --fetch: where its outputs go on the PC (OutputsClaim), flock'd by the
#: run until it has fetched them; `reap` fetches the outputs of a run that has gone.
OUTPUTS_DIR = STATE_DIR / "outputs"
#: Where outputs go whose run left no record (e.g. its STATE_DIR was lost).
ORPHANS_DIR = STATE_DIR / "orphans"
KEY = STATE_DIR / "key.pem"
SSH_USER = "ubuntu"
#: ssh's own exit status for a failure to connect or authenticate (any other
#: status is the remote command's).
SSH_FAILED = 255
BOX_HOME = f"/home/{SSH_USER}"
SSH_READY_SECONDS = 10 * 60
#: ssh's multiplexing sockets, shared by every pool (`%C` hashes the host, so
#: pools never collide). A Unix socket path holds at most 107 bytes, and ssh
#: adds a 17-byte temporary suffix to the 40-hex `%C` while it binds: under a
#: pool's own STATE_DIR the path ran over (buildbox-overflow 113 bytes,
#: buildbox-nndata-hevc 116), every master failed with "too long for Unix
#: domain socket", and no named pool's box ever became ready (2026-11-22).
CONTROL_DIR = WORK_DIR / "mux"
CONTROL_PATH = CONTROL_DIR / "%C"
UNIX_SOCKET_PATH_MAX = 107
CONTROL_SOCKET_EXTRA = len("0" * 40 + ".0123456789abcdef") - len("%C")
#: Lines of the box's setup log quoted when it is not ready in time.
SETUP_LOG_LINES = 20
#: Paths never synced by --sync: the private clips and their copies. Clips the
#: job needs go up with --clip instead (owner, 2026-11-20: clips may go to our AWS
#: boxes and are deleted there afterwards): the box deletes CLIP_ROOT before it
#: shuts itself down (nc_remote_box.sh), and `stop` before it terminates the box.
PRIVATE_PATHS = ("/mldata", str(Path.home() / "nanocodec-data/clips"),
                 str(WORK_DIR.parent / "aws-clip-cache"))
CLIP_ROOT = Path.home() / "nanocodec-data/clips"
#: Clips keep their size and time (data, not build inputs), so one already there
#: is not sent again.
CLIP_MODE = ("-a",)
#: Larger files stay out of the worktree sync (they are data, not sources);
#: `--data` uploads one on request.
SYNC_MAX_BYTES = 64 * 1024 * 1024
RAW_VIDEO_SUFFIXES = (".yuv", ".y4m")
RSYNC_EXCLUDES = ("--exclude=/target", "--exclude=/.git",
                  *(f"--exclude=*{suffix}" for suffix in RAW_VIDEO_SUFFIXES),
                  "--exclude=__pycache__", f"--max-size={SYNC_MAX_BYTES // (1024 * 1024)}m")
#: --data keeps size and time (data, not build inputs: nothing already there is
#: sent again) and keeps a partial transfer, so an interrupted upload resumes.
DATA_MODE = ("-a", "--partial")


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


class OutputsClaim:
    """A `run`'s record of where its --fetch outputs go (OUTPUTS_DIR/NAME.RUN_ID.json),
    written before the job starts and flock'd until the run has fetched them. A record
    nobody holds belongs to a run that has gone: `reap` fetches its outputs."""

    def __init__(self, name: str, fetch_to: Path):
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = f"{utc_now():%Y%m%dT%H%M%S}-{os.getpid()}-{os.urandom(3).hex()}"
        self.path = outputs_record(name, self.run_id)
        self.handle = open(self.path, "w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        self.handle.write(json.dumps({"name": name, "run_id": self.run_id,
                                      "fetch_to": str(fetch_to), "written": iso(utc_now())}))
        self.handle.flush()

    def release(self, fetched: bool) -> None:
        """Forget the record once the outputs are on the PC and the box's marker is gone;
        otherwise leave it, unheld, for `reap`."""
        if fetched:
            self.path.unlink(missing_ok=True)
        self.handle.close()


def outputs_record(name: str, run_id: str) -> Path:
    return OUTPUTS_DIR / f"{name}.{run_id}.json"


def prune_output_records() -> None:
    """Drop records no run holds that are older than a box's whole life: their box has
    ended, and its outputs with it."""
    if not OUTPUTS_DIR.exists():
        return
    oldest = utc_now() - datetime.timedelta(minutes=LIFETIME_MINUTES) - REPLACE_BEFORE_END
    for path in OUTPUTS_DIR.glob("*.json"):
        if not flock_held(path) and \
                datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc) < oldest:
            path.unlink(missing_ok=True)


def unfetched(box: Box, name: str = "") -> list[dict]:
    """The box's markers of finished jobs whose outputs are not on the PC yet (all jobs,
    or `name`'s). Raises when the box cannot be asked."""
    job = (HERE / "nc_remote_job.py").read_text()
    out = ssh(box, shlex.join(["python3", "-c", job, "--unfetched"]), capture_output=True,
              text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"{box.instance_id}: cannot list unfetched outputs: "
                           f"{out.stderr.strip()}")
    markers = [json.loads(line) for line in (out.stdout or "").splitlines() if line.strip()]
    return [m for m in markers if not name or m["name"] == name]


def fetch_marked(box: Box, marker: dict) -> bool:
    """Fetch one finished job's outputs to where its run asked, then drop its marker.
    False when a live run still owns them (it fetches them itself)."""
    record = outputs_record(marker["name"], marker["run_id"])
    if record.exists():
        handle = open(record, "r+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        fetch_to = Path(json.loads(record.read_text() or "{}").get("fetch_to")
                        or ORPHANS_DIR / marker["name"] / marker["run_id"])
    else:
        handle = None
        fetch_to = ORPHANS_DIR / marker["name"] / marker["run_id"]
        say(f"{marker['name']} ({marker['run_id']}): no record of its run's --fetch-to on "
            f"this PC; its outputs go to {fetch_to}")
    try:
        say(f"fetching the outputs of {marker['name']} (run {marker['run_id']}, ended "
            f"{marker['ended']}, exit {marker['status']}): its caller has gone")
        fetch(box, marker["name"], marker["paths"], fetch_to)
        ssh(box, shlex.join(["python3", "-c", (HERE / "nc_remote_job.py").read_text(),
                             "--fetched", marker["name"], marker["run_id"]]),
            check=True, capture_output=True, timeout=120)
        record.unlink(missing_ok=True)
    finally:
        if handle:
            handle.close()
    return True


def fetch_unfetched(box: Box, name: str = "") -> list[dict]:
    """Fetch every finished job's outputs that no live run is fetching (all jobs, or
    `name`'s); return the markers still on the box (a live run's, or a failed fetch's).
    Best effort: an unreachable box returns []."""
    try:
        markers = unfetched(box, name)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        say(f"{box.instance_id}: {error}")
        return []
    left = []
    for marker in markers:
        try:
            if fetch_marked(box, marker):
                continue
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            say(f"{box.instance_id}: fetching {marker['name']} ({marker['run_id']}) failed: "
                f"{error}; kept on the box, the next reap tries again")
        left.append(marker)
    return left


def report_lost(box: Box) -> None:
    """Say loudly which finished outputs end with the box (stop --force, budget, hard end)."""
    with contextlib.suppress(Exception):
        for marker in unfetched(box):
            say(f"LOST: {box.instance_id} ends with the unfetched outputs of {marker['name']} "
                f"(run {marker['run_id']}, ended {marker['ended']}): "
                f"{', '.join(marker['paths'])}")


def idle_script() -> str:
    """The box's idle check (nc_remote_idle.sh) with this pool's idle limit."""
    return (HERE / "nc_remote_idle.sh").read_text().replace("@IDLE_MINUTES@", str(IDLE_MINUTES))


def idle_check(box: Box) -> subprocess.CompletedProcess:
    """Install this version of the idle check on the box (a box launched by an older
    nc-remote then keeps unfetched outputs too) and run it."""
    return ssh(box, "sudo tee /usr/local/sbin/ncbox-idle >/dev/null && "
                    "sudo chmod 755 /usr/local/sbin/ncbox-idle && sudo /usr/local/sbin/ncbox-idle",
               input=idle_script().encode(), capture_output=True)


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
            .replace("@IDLE_SCRIPT@", idle_script().rstrip("\n")) \
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


def control_path() -> Path:
    """The ssh ControlPath, its directory made and its length checked: an
    over-long socket path fails every ssh call, not just multiplexing."""
    length = len(str(CONTROL_PATH)) + CONTROL_SOCKET_EXTRA
    if length > UNIX_SOCKET_PATH_MAX:
        raise RuntimeError(f"ssh ControlPath {CONTROL_PATH} would be {length} bytes, "
                           f"over the {UNIX_SOCKET_PATH_MAX}-byte Unix socket limit")
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    return CONTROL_PATH


def ssh_options(box: Box) -> list[str]:
    return ["-i", str(key_path(box.region)), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            "-o", f"UserKnownHostsFile={STATE_DIR / ('known_hosts-' + box.instance_id)}",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control_path()}", "-o", "ControlPersist=120"]


def ssh(box: Box, command: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *ssh_options(box), f"{SSH_USER}@{box.host}", command], **kwargs)


#: rsync modes. Results come back as they are (-a). Sources go up by content
#: (--checksum) WITHOUT their mtimes (-a minus -t, -o, -g): a changed file lands
#: with the box's current time, so cargo always sees it newer than the last
#: build. Preserving mtimes left a file edited during a remote build older than
#: that build's outputs, and the next run reused the stale binary (2026-11-16).
FETCH_MODE = ("-a",)
UPLOAD_MODE = ("-rlpD", "--checksum")
#: `--sync` paths (anchor tool directories, builds) go up as real files: a link is
#: copied as its target (`-L`). Kept as links, the PC's absolute links (anchors/x86_64/bin
#: vpxdec, avmdec and dav2d -> /home/mark/...) landed dangling on the build box on
#: 2026-11-23 09:48 and every decode through them failed ("tool 'vpxdec' not found").
SYNC_MODE = ("-rLpD", "--checksum")


def rsync(box: Box, mode: tuple[str, ...], *args: str) -> None:
    shell = shlex.join(["ssh", *ssh_options(box)])
    subprocess.run(["rsync", *mode, "--mkpath", "-e", shell, *args], check=True)


def remote(box: Box, path: str) -> str:
    return f"{SSH_USER}@{box.host}:{path}"


def wait_for_ssh(box: Box) -> None:
    """Wait until the box accepts SSH and its user-data has made ncbox/activity.
    On timeout the error carries the real reason: ssh's last error, or, when
    SSH worked but the setup did not finish, the tail of the box's setup log."""
    deadline = time.monotonic() + SSH_READY_SECONDS
    last = None
    while time.monotonic() < deadline:
        last = ssh(box, "test -f ncbox/activity", capture_output=True, text=True)
        if last.returncode == 0:
            return
        time.sleep(5)
    if last is not None and last.returncode == SSH_FAILED:
        reason = f"ssh failed: {last.stderr.strip() or 'no error text'}"
    else:
        log = ssh(box, f"sudo tail -n {SETUP_LOG_LINES} /var/log/cloud-init-output.log",
                  capture_output=True, text=True)
        reason = f"ncbox/activity missing; setup log tail:\n{log.stdout or log.stderr}"
    raise RuntimeError(f"{box.instance_id}: SSH and the box setup not ready in "
                       f"{SSH_READY_SECONDS // 60} min: {reason}")


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
    if ON_DEMAND_ALLOWED and len(excluded) >= RECLAIMED_TYPES_FOR_ON_DEMAND:
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
    if ON_DEMAND_ALLOWED and on_demand_fallback and not every_quota:
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
            admit_this_host(aws, box)
            provision(box)  # a no-op unless the PC's toolchains changed
        return BoxUse(box)


def admit_this_host(aws: Aws, box: Box) -> None:
    """Point the box's SSH rule at this PC's current public address. The rule is set at
    launch; a reused box kept the old one, so after the PC's address changed (a reboot,
    2026-11-24) every run timed out on SSH until someone edited the group by hand."""
    aws.regional(box.region).security_group(box.expected_end)


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
        return delisted_reason(box)
    return ""


END, REPLACE = "end", "replace"


def reap_action(reason: str, delisted: bool, idle: bool) -> str:
    """What reap does to the current box given its replacement reason: END it when idle (the
    next run launches a listed type); REPLACE it (make before break) when busy, except a
    delisted type, which keeps serving its jobs until it idles (coordinator, 2026-11-24: a
    type change never replaces a box under running jobs); "" otherwise."""
    if not reason:
        return ""
    if idle:
        return END
    return "" if delisted else REPLACE


def delisted_reason(box: Box) -> str:
    """A type no longer listed keeps serving its jobs and ends once idle (cmd_reap), never
    replaced under running jobs; the next launch takes a listed type."""
    return f"{box.instance_type} is no longer a build box type"


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
            if old.instance_id in {b.instance_id for b in unended()} and \
                    not fetch_unfetched(old) and unused(old):
                say(f"{old.instance_id}: drained")
                stop_box(aws, old)


def stop_all(aws: Aws) -> None:
    """Terminate the current box and every draining one."""
    for box in unended():
        stop_box(aws, box)


def collect_occupancy(box: Box) -> None:
    """Copy the box's new job records (nc_remote_job.OCCUPANCY_LOG) to OCCUPANCY. Best
    effort: an unreachable box loses at most the jobs since the last reap."""
    offsets = json.loads(OCCUPANCY_OFFSETS.read_text()) if OCCUPANCY_OFFSETS.exists() else {}
    done = offsets.get(box.instance_id, 0)
    try:
        out = ssh(box, f"tail -c +{done + 1} ncbox/occupancy.jsonl 2>/dev/null || true",
                  capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return
    if out.returncode != 0 or not out.stdout:
        return
    whole = out.stdout[:out.stdout.rfind(b"\n") + 1]  # a line being written waits
    with OCCUPANCY.open("a") as log:
        for line in whole.decode().splitlines():
            record = json.loads(line)
            log.write(json.dumps({**record, "instance_id": box.instance_id,
                                  "instance_type": box.instance_type, "pool": POOL or "main"})
                      + "\n")
    offsets[box.instance_id] = done + len(whole)
    OCCUPANCY_OFFSETS.write_text(json.dumps(offsets))


def stop_box(aws: Aws, box: Box) -> None:
    """Terminate `box` after copying off its job records and any finished outputs no live
    run is fetching; outputs still there are named as lost. Callers that must lose
    nothing check `unused` first (it counts unfetched outputs as use)."""
    with contextlib.suppress(Exception):
        collect_occupancy(box)
    with contextlib.suppress(Exception):
        fetch_unfetched(box)
    report_lost(box)
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


def checked_data_path(path: str, worktree: Path) -> Path:
    resolved = (worktree / Path(path).expanduser()).resolve()
    checked_sync_path(str(resolved))  # the same private-clip and home rules as --sync
    if not resolved.exists():
        raise SystemExit(f"nc-remote: --data {path} does not exist")
    return resolved


def data_destination(path: Path, worktree: Path, name: str) -> str:
    """Where a --data path lands on the box: the same place in the job's tree
    for a worktree path, else the same place under the box's home."""
    if path.is_relative_to(worktree):
        return f"ncbox/jobs/{name}/src/{path.relative_to(worktree)}"
    return str(path.relative_to(Path.home()))


def sync_exclusion(path: Path, worktree: Path) -> str | None:
    """Why the worktree sync leaves `path` out, or None when it goes up (or is
    not the sync's business: outside the worktree, or target/)."""
    if not path.is_relative_to(worktree) or path == worktree:
        return None
    relative = path.relative_to(worktree)
    if relative.parts[0] in ("target", ".git"):
        return None
    if path.suffix in RAW_VIDEO_SUFFIXES:
        return "raw video"
    if path.is_file() and path.stat().st_size > SYNC_MAX_BYTES:
        return f"over {SYNC_MAX_BYTES // (1024 * 1024)} MB"
    ignored = subprocess.run(["git", "-C", str(worktree), "check-ignore", "-q", str(relative)])
    return "ignored by git" if ignored.returncode == 0 else None


def warn_excluded(command: str, worktree: Path, data: list[Path]) -> None:
    """Warn about each existing path the command names that the worktree sync
    leaves out and no --data uploads (the job would fail at once on a box
    that has never seen it)."""
    try:
        words = shlex.split(command)
    except ValueError:  # unbalanced quotes: bash will say so on the box
        return
    for word in {piece for w in words for piece in w.split("=")}:
        path = (worktree / Path(word).expanduser()).resolve()
        if any(path.is_relative_to(d) for d in data):
            continue
        try:
            reason = path.exists() and sync_exclusion(path, worktree)
        except OSError:  # a word too long or odd to be a path
            continue
        if reason:
            say(f"warning: {word} is not synced ({reason}); upload it with --data {word}")


def sync(box: Box, worktree: Path, name: str, extra: list[Path],
         clips: tuple[Path, ...] = (), data: tuple[Path, ...] = ()) -> None:
    started = time.monotonic()
    rsync(box, UPLOAD_MODE, "-z", "--delete", *RSYNC_EXCLUDES, "--filter=:- .gitignore",
          f"{worktree}/", remote(box, f"ncbox/jobs/{name}/src/"))
    for path in extra:
        relative = path.relative_to(Path.home())
        suffix = "/" if path.is_dir() else ""
        keep = [f"--exclude=/{name}" for name in box_links(box, str(relative))] if suffix else []
        rsync(box, SYNC_MODE, "-z", *RSYNC_EXCLUDES, *keep, f"{path}{suffix}",
              remote(box, f"{relative}{suffix}"))
    for clip in clips:
        rsync(box, CLIP_MODE, str(clip), remote(box, str(clip.relative_to(Path.home()))))
    for path in data:
        suffix = "/" if path.is_dir() else ""
        rsync(box, DATA_MODE, f"{path}{suffix}",
              remote(box, data_destination(path, worktree, name) + suffix))
    say(f"synced in {time.monotonic() - started:.0f} s")


def box_links(box: Box, directory: str) -> list[str]:
    """The names in the box's `directory` that are working links of the box's own (e.g.
    anchors/x86_64/bin/clang-format -> the box's venv): a `--sync` never replaces them.
    A dangling link (a PC path) is replaced by the real file."""
    found = ssh(box, f"test -d {shlex.quote(directory)} && find {shlex.quote(directory)} "
                     f"-mindepth 1 -maxdepth 1 -type l ! -xtype l -printf '%f\\n' || true",
                capture_output=True, text=True, check=True)
    return sorted(line for line in found.stdout.splitlines() if line.strip())


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
    # A finished job of this name whose caller went away: its outputs come back before this
    # job's sync and run can overwrite them.
    fetch_unfetched(box, name)
    data = tuple(checked_data_path(p, worktree) for p in args.data)
    warn_excluded(job_command(args.command), worktree, list(data))
    sync(box, worktree, name, [checked_sync_path(p) for p in args.sync],
         tuple(checked_clip_path(p) for p in getattr(args, "clip", [])), data)
    # NC_BUILD_DIRTY stamps a build of local changes as such (build_stamp.py);
    # always set, so an empty value marks a clean tree.
    argv = ["python3", "-c", job, name, commit, str(args.jobs), job_command(args.command),
            f"NC_BUILD_DIRTY={dirty}", *args.env]
    claim = None
    if args.fetch:
        # Where the outputs go is on record before the job starts, and the box records
        # them as unfetched when it ends, so they come back even if this run does not
        # live to fetch them (reap does).
        claim = OutputsClaim(name, args.fetch_to or worktree)
        argv[3:3] = ["--outputs", claim.run_id, json.dumps(args.fetch)]
    started = time.monotonic()
    # stdin stays open (and unused) for the job's life: its EOF tells the box
    # that this caller has gone (nc_remote_job.py kills the job then).
    job_ssh = subprocess.Popen(["ssh", *ssh_options(box), f"{SSH_USER}@{box.host}",
                                shlex.join(argv)], stdin=subprocess.PIPE)
    status = job_ssh.wait()
    say(f"exit {status} after {time.monotonic() - started:.0f} s")
    if claim:
        fetch(box, name, args.fetch, args.fetch_to or worktree)
        dropped = ssh(box, shlex.join(["python3", "-c", job, "--fetched", name, claim.run_id]),
                      capture_output=True, timeout=120)
        claim.release(dropped.returncode == 0)
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

    def state() -> str:
        return aws.regional(box.region).state(box.instance_id)
    if status == STATUS_SPOT_NOTICE:
        say("the box has a spot interruption notice: waiting for it to go, then running the "
            "command on a new box")
        deadline = time.monotonic() + NOTICE_WAIT_S
        while state() in LIVE_STATES and time.monotonic() < deadline:
            time.sleep(OVERFLOW_POLL_SECONDS)
    if retry_after(status, state()):
        say("the box went away during the job (spot reclaim or limit): restarting, retrying once")
        status, _ = run_once(aws, args, worktree, name, commit, dirty)
    return status


#: The box-side refusal of a job on a box with a spot interruption notice
#: (nc_remote_job.STATUS_SPOT_NOTICE), and how long `run` waits for that box to go.
STATUS_SPOT_NOTICE = 254
NOTICE_WAIT_S = 5 * 60
LIVE_STATES = ("pending", "running")


def retry_after(status: int, state: str) -> bool:
    """Whether a `run` goes again on a new box: its box went away under the job (SSH broke
    and the box is no longer live), or refused it on a spot notice and has gone since. The
    command is the unit: it runs once more, whole."""
    return status in (SSH_FAILED, STATUS_SPOT_NOTICE) and state not in LIVE_STATES


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
        # Finished outputs leave first; a box still holding some (a live run's, or a
        # failed fetch) is in use.
        busy = [box for box in boxes if not args.force and (fetch_unfetched(box)
                                                           or not unused(box))]
        if busy:
            for box in busy:
                say(f"{box.instance_id}: other jobs are running or their outputs are not "
                    f"fetched yet; left up (the idle check stops it when they finish; "
                    f"--force terminates now and loses them)")
            return 0
        stop_all(aws)
    return 0


def cmd_reap(_args) -> int:
    """The PC's check: close out gone boxes, enforce the budget and the hard end,
    end drained boxes, replace the current box when it should give way (make
    before break), and run the idle check on every live box."""
    aws = Aws()
    replacement = None
    # Finished outputs whose run has gone come back first, outside the lock (a fetch can
    # take minutes; runs need the lock to start), so nothing below ends a box holding them.
    for box in unended():
        if box.host and aws.regional(box.region).state(box.instance_id) in LIVE_STATES:
            fetch_unfetched(box)
    prune_output_records()
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
            action = reap_action(reason, reason == delisted_reason(box),
                                 bool(reason) and unused(box))
            if action == END:
                say(f"{box.instance_id}: {reason} and no job runs: ending it")
                stop_box(aws, box)
                continue
            if action == REPLACE:
                replacement = (box, reason)
            if idle_check(box).returncode != 0:
                note_unreachable(aws, box)
            else:
                unreachable_marker(box).unlink(missing_ok=True)
                collect_occupancy(box)
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
    """No job slot is held, no job is waiting on the box, no finished job's outputs
    wait to be fetched, and none has ended in the last QUIET_SECONDS."""
    probe = ("for s in ncbox/slots/*; do flock -n \"$s\" true || exit 1; done; "
             # a waiting record from before this boot (baked, or a reclaim) is no job
             "[ -z \"$(find ncbox/jobs -maxdepth 2 -name waiting.json -newermt "
             "@$(awk '/^btime/ {print $2}' /proc/stat))\" ] && "
             "[ -z \"$(find ncbox/jobs -mindepth 3 -maxdepth 3 -path '*/unfetched/*.json' "
             "-newermt @$(awk '/^btime/ {print $2}' /proc/stat))\" ] && "
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


def cmd_occupancy(args) -> int:
    """The job records of the last --hours (both pools): jobs, waits, and the share of
    time each number of agent slots was busy (docs/aws-cost-guide.md, build box)."""
    paths = [WORK_DIR / "buildbox" / OCCUPANCY.name, WORK_DIR / f"buildbox-{OVERFLOW}" / OCCUPANCY.name]
    since = time.time() - args.hours * 3600
    records = [json.loads(line) for path in paths if path.exists()
               for line in path.read_text().splitlines()]
    records = [r for r in records if r["ended"] >= since]
    print(occupancy_summary(records, since, time.time()))
    return 0


def occupancy_summary(records: list[dict], start: float, end: float) -> str:
    """Jobs, wait percentiles, and the time-weighted share of each count of running
    agent jobs (verifier-* jobs apart) over [start, end) on the main pool."""
    if not records:
        return "no job records in the window"
    waits = sorted(r["wait_s"] for r in records)
    pick = lambda q: waits[min(len(waits) - 1, int(q * len(waits)))]  # noqa: E731
    lines = [f"{len(records)} jobs; wait p50 {pick(0.5):.0f} s, p90 {pick(0.9):.0f} s, "
             f"max {waits[-1]:.0f} s; {sum(w > 0 for w in waits)} waited at all; "
             f"{sum(r.get('pool') == OVERFLOW for r in records)} on the overflow box"]
    events = sorted((t, d) for r in records
                    if r.get("pool", "main") == "main" and not r["name"].startswith("verifier-")
                    for t, d in ((max(r["started"], start), 1), (min(r["ended"], end), -1))
                    if r["ended"] > start)
    share: dict[int, float] = {}
    running, last = 0, start
    for t, step in events:
        share[running] = share.get(running, 0.0) + max(0.0, t - last)
        running, last = running + step, max(last, t)
    share[running] = share.get(running, 0.0) + max(0.0, end - last)
    total = sum(share.values())
    lines.append("agent jobs running at once (main box), share of the window: " + ", ".join(
        f"{k}: {100 * v / total:.1f} %" for k, v in sorted(share.items()) if v > 0))
    return "\n".join(lines)


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
    run.add_argument("--data", action="append", default=[],
                     help="a file or directory to upload as it is, whatever its size "
                          "(worktree-relative or under ~); --fetch it to bring it back")
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
    occupancy = commands.add_parser("occupancy", help="job slots and waits over a window")
    occupancy.add_argument("--hours", type=float, default=24.0)
    occupancy.set_defaults(handler=cmd_occupancy)
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
