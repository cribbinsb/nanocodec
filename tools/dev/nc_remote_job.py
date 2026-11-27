"""Box side of nc-remote: run one agent's command in a job slot (standard library only).

tools/dev/nc_remote.py sends this file's text over SSH (`python3 -c`), so the
box always runs the PC's version:

    python3 -c <this file> [--outputs RUN_ID PATHS_JSON] NAME COMMIT JOBS COMMAND [KEY=VALUE ...]

(nc-remote passes NC_BUILD_DIRTY=<hash of the local changes, or empty> as the
first KEY=VALUE: with NC_BUILD_COMMIT it stamps the build, tools/dev/build_stamp.py.)
    python3 -c <this file> --status      (the slots and the queue)
    python3 -c <this file> --drop NAME... (remove jobs whose worktree is gone)
    python3 -c <this file> --unfetched   (the finished jobs' outputs not yet fetched, JSON lines)
    python3 -c <this file> --fetched NAME RUN_ID (the PC has them: drop the marker)

It waits for a free slot (one flock'd file per slot under ~/ncbox/slots; the
idle watchdog of nc_remote_box.sh counts a held slot as busy), then runs
COMMAND with bash in ~/ncbox/jobs/NAME/src, the agent's synced worktree,
where `target` is a link to the agent's own CARGO_TARGET_DIR kept on the box
(~/ncbox/jobs/NAME/target), so rebuilds are incremental. Its output streams
straight back over SSH and this process exits with its status. When the SSH
link closes (the caller was interrupted), stdin reaches EOF and the job's
whole process group is killed, so nothing runs on unattended.

Outputs never stay behind (2026-11-24: 19 finished PGO builds, ~$2.7, were lost when an
idle box was replaced). With `--outputs`, the job's declared `--fetch` paths are recorded
when it ends, finished or killed, in ~/ncbox/jobs/NAME/unfetched/RUN_ID.json (UNFETCHED).
The calling `run` fetches them and drops the marker; if that caller has gone, `nc-remote
reap` does. While a marker of this boot exists, the box's idle watchdog
(nc_remote_idle.sh) does not shut it down, the PC does not stop or replace it, and
eviction keeps that job's directory. At the end, files the job left in the box's home
outside the job tree and the declared outputs are reported (undeclared_writes): they
would be lost with the box.

Disk: a starting job evicts the least recently used other job directories while
fewer than MIN_FREE_BYTES are free (never a running or waiting job's, nor PROTECTED).

A job records itself in ~/ncbox/jobs/NAME/waiting.json while it waits for a
slot and in running.json while it runs; `--status` reads them. A record lives only
as long as its process on this boot (live_records): one whose process is gone, or
which predates the box's boot (a record baked into the image, or left by a reclaim),
is removed when read, so it never shows as holding a slot or as waiting (2026-11-21:
a five-day-old av1-writer record, baked into the AMI, showed in a verifier slot).

Every finished job appends one line to ~/ncbox/occupancy.jsonl (OCCUPANCY_LOG: its
slot, wait for it, run time, status and the slots busy when it started); `nc-remote reap`
copies new lines to the PC and `nc-remote occupancy` summarises them, to size the box
(docs/aws-cost-guide.md, build box).
"""

import calendar
import fcntl
import select
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

BOX = Path.home() / "ncbox"
SLOT_POLL_SECONDS = 2
#: Below this much free disk the job warns (agents' targets fill the root volume).
LOW_DISK_BYTES = 20 << 30
#: Below this much free disk a starting job first evicts the least recently used other
#: jobs' directories (their targets: a rebuild there is incremental from scratch, the
#: sync re-sends the tree), never a running or waiting job's, nor PROTECTED ones
#: (2026-11-21: the 193 GB volume reached 2 GiB free, 156 GB of it job directories).
MIN_FREE_BYTES = 40 << 30
#: Kept by eviction: the verifier's targets and aws_cpu_ab's warm build directories.
PROTECTED = ("verifier", "cpuab-")
LAST_USED = "last-used"
OCCUPANCY_LOG = BOX / "occupancy.jsonl"
#: Exit status when the caller went away and the job was killed.
STATUS_ABANDONED = 255
#: Exit status of a job refused because the box has a spot interruption notice: nc-remote
#: waits for the box to go and runs the command on its replacement (nc_remote.cmd_run).
STATUS_SPOT_NOTICE = 254
#: Per finished job with declared outputs: jobs/NAME/UNFETCHED/RUN_ID.json until fetched.
UNFETCHED = "unfetched"
#: Home entries that are the box's own caches and tools, never a job's output.
HOME_CACHES = (".cargo", ".rustup", ".cache", ".cmake", ".local", ".config", ".npm", ".ssh",
               "ncbox", "nanocodec-data/clips")
#: Home files any shell leaves behind.
HOME_NOISE = (".bash_history", ".python_history", ".lesshst", ".viminfo", ".wget-hsts",
              ".sudo_as_admin_successful")
#: How many groups of undeclared files a warning names.
UNDECLARED_SHOWN = 8
#: The spot interruption notice (IMDSv2), posted about two minutes before a reclaim.
IMDS = "http://169.254.169.254/latest"
NOTICE_POLL_SECONDS = 10


def touch_activity() -> None:
    (BOX / "activity").touch()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def boot_time() -> float:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    return 0.0


def live_records(name: str) -> list[tuple[Path, dict]]:
    """The jobs' `name` records (running.json, waiting.json) whose process is alive and
    began on this boot; the others are removed."""
    booted = boot_time()
    live = []
    for path in (BOX / "jobs").glob(f"*/{name}"):
        try:
            record = json.loads(path.read_text())
            started = calendar.timegm(time.strptime(record.get("started") or record["since"],
                                                    "%Y-%m-%dT%H:%M:%SZ"))
            alive = started >= booted and Path(f"/proc/{int(record['pid'])}").exists()
        except (OSError, ValueError, KeyError):
            alive = False  # unreadable, or from before records carried their pid
        if alive:
            live.append((path, record))
        else:
            path.unlink(missing_ok=True)
    return live


def slot_held(path: Path) -> bool:
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def print_status() -> None:
    """One line per slot (free, or the job holding it) and the jobs waiting."""
    running = {job["slot"]: (path.parent.name, job) for path, job in live_records("running.json")}
    for slot in sorted((BOX / "slots").iterdir(), key=lambda p: int(p.name)):
        # A running.json whose slot lock is free belongs to a job that died
        # without cleanup (e.g. a spot reclaim): the slot is free.
        if slot.name in running and slot_held(slot):
            name, job = running[slot.name]
            print(f"  slot {slot.name}: {name} since {job['started']}: {job['command'][:100]}")
        else:
            print(f"  slot {slot.name}: {'busy' if slot_held(slot) else 'free'}")
    waiting = sorted((record["since"], path.parent.name)
                     for path, record in live_records("waiting.json"))
    print(f"  queue: {len(waiting)} waiting" + "".join(
        f"\n    {name} since {since}" for since, name in waiting))
    markers = [marker for _, marker in unfetched_markers()]
    print(f"  unfetched outputs: {len(markers)}" + "".join(
        f"\n    {m['name']} (run {m['run_id']}, ended {m['ended']}): {', '.join(m['paths'])}"
        for m in markers))


def caller_gone(timeout: float) -> bool:
    """Wait up to `timeout` s; True when our stdin (the SSH channel) has closed."""
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(readable) and not os.read(sys.stdin.fileno(), 1)


#: The first slots are kept for land.sh's verify parts (job names starting VERIFIER):
#: verify never queues behind agents' jobs; other jobs use the rest.
VERIFIER_SLOTS = 2
VERIFIER = "verifier-"


def job_slots(name: str) -> list[Path]:
    slots = sorted((BOX / "slots").iterdir(), key=lambda p: int(p.name))
    return slots if name.startswith(VERIFIER) else slots[VERIFIER_SLOTS:]


def print_free() -> None:
    """`free_slots oldest_wait_seconds`: the slots an agent's job could take now, and
    how long the oldest waiting job has waited (nc-remote's overflow decision)."""
    free = sum(1 for path in job_slots("") if not slot_held(path))
    since = [record["since"] for _, record in live_records("waiting.json")]
    waited = max((time.time() - calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
                  for s in since), default=0)
    print(free, int(waited))


def take_slot(waiting: Path, name: str):
    """Block until a slot file is locked; returns its open file (the lock)."""
    slots = job_slots(name)
    announced = False
    while True:
        for path in slots:
            handle = open(path, "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                waiting.unlink(missing_ok=True)
                return handle, path.name
            except BlockingIOError:
                handle.close()
        if not announced:
            waiting.write_text(json.dumps({"since": now(), "pid": os.getpid()}))
            print(f"[nc-remote] all {len(slots)} slots busy, waiting", file=sys.stderr, flush=True)
            announced = True
        touch_activity()
        if caller_gone(SLOT_POLL_SECONDS):
            waiting.unlink(missing_ok=True)
            sys.exit(STATUS_ABANDONED)


def job_env(target: Path, commit: str, jobs: str, extra: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS=jobs, NC_BUILD_COMMIT=commit,
               NC_REMOTE_BOX="1",
               # the page tests' Chromium: Playwright's Ubuntu 24.04 build (provisioning)
               PLAYWRIGHT_HOST_PLATFORM_OVERRIDE="ubuntu24.04-x64")
    # The provisioned venv (nceval's packages) and the Rust toolchains come first.
    env["PATH"] = f"{BOX / 'venv/bin'}:{Path.home() / '.cargo/bin'}:{env.get('PATH', '')}"
    for item in extra:
        key, _, value = item.partition("=")
        env[key] = value
    return env


def spot_notice() -> str | None:
    """The spot interruption notice, if one is posted; None off EC2 or with none."""
    import urllib.error
    import urllib.request
    try:
        token = urllib.request.urlopen(urllib.request.Request(
            f"{IMDS}/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=2).read().decode()
        return urllib.request.urlopen(urllib.request.Request(
            f"{IMDS}/meta-data/spot/instance-action",
            headers={"X-aws-ec2-metadata-token": token}), timeout=2).read().decode()
    except (urllib.error.URLError, OSError):
        return None


def warn_on_notice(process: subprocess.Popen, notice=spot_notice) -> None:
    """While the job runs, say once that the box is being reclaimed: the job dies with it
    and nc-remote runs the command again on a new box (its output so far is lost; outputs
    meant to survive go under remote-out/ and are fetched as they are made)."""
    while process.poll() is None:
        if notice():
            print("[nc-remote] SPOT INTERRUPTION NOTICE: this box ends within about 2 "
                  "minutes; the command will be run again on a new box", file=sys.stderr,
                  flush=True)
            return
        time.sleep(NOTICE_POLL_SECONDS)


def kill_on_hangup(process: subprocess.Popen, abandoned: threading.Event) -> None:
    """Kill the job's process group once our stdin (the SSH channel) closes."""
    sys.stdin.buffer.read()
    if process.poll() is None:
        abandoned.set()
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(5)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def unfetched_markers() -> list[tuple[Path, dict]]:
    """The markers of finished jobs whose outputs the PC has not fetched yet. A marker
    from before this boot (baked into the image) is removed: its box, and so its
    outputs' owner, is gone."""
    booted = boot_time()
    live = []
    for path in sorted((BOX / "jobs").glob(f"*/{UNFETCHED}/*.json")):
        try:
            if path.stat().st_mtime >= booted:
                live.append((path, json.loads(path.read_text())))
                continue
        except (OSError, ValueError):
            pass
        path.unlink(missing_ok=True)
    return live


def print_unfetched() -> None:
    for _, marker in unfetched_markers():
        print(json.dumps(marker))


def mark_fetched(name: str, run_id: str) -> None:
    (BOX / "jobs" / name / UNFETCHED / f"{run_id}.json").unlink(missing_ok=True)


def mark_unfetched(name: str, run_id: str, paths: list[str], status: int) -> None:
    """Record the job's declared outputs as not yet on the PC (written whole, then
    renamed, so a reader never sees half a marker)."""
    directory = BOX / "jobs" / name / UNFETCHED
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / f"{run_id}.partial"
    partial.write_text(json.dumps({"name": name, "run_id": run_id, "paths": paths,
                                   "status": status, "ended": now()}))
    partial.rename(directory / f"{run_id}.json")


def undeclared_writes(started: float, declared: list[str]) -> dict[str, int]:
    """Files in the box's home written since `started` outside the job trees, the
    caches and the declared `~/` outputs, counted per group (the first two path parts
    under home). Another job's or a `--sync` upload's writes in that time show too."""
    home = Path.home()
    prune = [str(home / p) for p in HOME_CACHES] + \
        [str(home / p[2:].rstrip("/")) for p in declared if p.startswith("~/")]
    expression = [x for p in prune for x in ("-o", "-path", p)][1:]
    try:
        found = subprocess.run(["find", str(home), "-xdev", "(", *expression, ")", "-prune",
                                "-o", "-type", "f", "-newermt", f"@{int(started)}", "-print"],
                               capture_output=True, text=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return {}
    groups: dict[str, int] = {}
    for line in found.splitlines():
        relative = Path(line).relative_to(home)
        if str(relative) in HOME_NOISE:
            continue
        group = "~/" + "/".join(relative.parts[:2])
        groups[group] = groups.get(group, 0) + 1
    return groups


def warn_undeclared(groups: dict[str, int]) -> None:
    if not groups:
        return
    shown = sorted(groups.items(), key=lambda item: -item[1])[:UNDECLARED_SHOWN]
    print(f"[nc-remote] WARNING: UNDECLARED WRITES: {sum(groups.values())} files in the box's "
          "home outside this job's tree (~/ncbox/jobs/NAME) and its --fetch paths were written "
          "during the job; the box's home does not outlive the box (idle stop, replacement, "
          "reclaim). Write outputs under the job's tree (remote-out/) and --fetch them. "
          "(Another job's or a --sync upload's writes in the same time show here too.)\n"
          + "".join(f"[nc-remote]   {group}: {count} files\n" for group, count in shown),
          file=sys.stderr, end="", flush=True)


def evictable(keep: set[str]) -> list[Path]:
    """Job directories that may go, least recently used first: not running or waiting,
    holding no unfetched outputs, not PROTECTED, not in `keep`."""
    live = {path.parent.name for kind in ("running.json", "waiting.json")
            for path, _ in live_records(kind)}
    live |= {marker["name"] for _, marker in unfetched_markers()}
    jobs = [d for d in (BOX / "jobs").iterdir() if d.is_dir() and d.name not in live
            and d.name not in keep and not d.name.startswith(PROTECTED)]
    used = lambda d: (d / LAST_USED).stat().st_mtime if (d / LAST_USED).exists() \
        else d.stat().st_mtime
    return sorted(jobs, key=used)


def remove_job(directory: Path, why: str) -> None:
    before = shutil.disk_usage(BOX).free
    shutil.rmtree(directory, ignore_errors=True)
    freed = (shutil.disk_usage(BOX).free - before) >> 30
    print(f"[nc-remote] removed job directory {directory.name} ({why}): {freed} GiB",
          file=sys.stderr, flush=True)


def make_room(current: str) -> None:
    """Evict least recently used job directories until MIN_FREE_BYTES are free."""
    for directory in evictable({current}):
        if shutil.disk_usage(BOX).free >= MIN_FREE_BYTES:
            return
        remove_job(directory, f"under {MIN_FREE_BYTES >> 30} GiB free, least recently used")


def drop(names: list[str]) -> None:
    """Remove the named jobs' directories (their worktrees are gone on the PC), unless
    running, waiting or PROTECTED."""
    allowed = {d.name: d for d in evictable(set())}
    for name in names:
        if name in allowed:
            remove_job(allowed[name], "its worktree is gone")


def main() -> int:
    if sys.argv[1:] == ["--status"]:
        print_status()
        return 0
    if sys.argv[1:] == ["--free"]:
        print_free()
        return 0
    if sys.argv[1:2] == ["--drop"]:
        drop(sys.argv[2:])
        return 0
    if sys.argv[1:] == ["--unfetched"]:
        print_unfetched()
        return 0
    if sys.argv[1:2] == ["--fetched"]:
        mark_fetched(*sys.argv[2:4])
        return 0
    args, run_id, outputs = sys.argv[1:], "", []
    if args[:1] == ["--outputs"]:
        run_id, outputs, args = args[1], json.loads(args[2]), args[3:]
    name, commit, jobs, command, *extra = args
    job_dir = BOX / "jobs" / name
    src, target = job_dir / "src", job_dir / "target"
    target.mkdir(parents=True, exist_ok=True)
    (job_dir / LAST_USED).touch()
    make_room(name)
    if not (src / "target").is_symlink():
        (src / "target").symlink_to(target)
    requested = time.time()
    if spot_notice():
        print("[nc-remote] this box has a spot interruption notice: not starting the job",
              file=sys.stderr, flush=True)
        return STATUS_SPOT_NOTICE
    slot, slot_name = take_slot(job_dir / "waiting.json", name)
    started = time.time()
    busy = sum(slot_held(path) for path in job_slots(name))  # ours included (another lock)
    touch_activity()
    free = shutil.disk_usage(BOX).free
    if free < LOW_DISK_BYTES:
        print(f"[nc-remote] warning: {free >> 30} GiB free on the box", file=sys.stderr)
    running = job_dir / "running.json"
    process = subprocess.Popen(["bash", "-c", command], cwd=src, stdin=subprocess.DEVNULL,
                               env=job_env(target, commit, jobs, extra), start_new_session=True)
    running.write_text(json.dumps({"slot": slot_name, "pid": process.pid, "command": command,
                                   "started": now()}))
    abandoned = threading.Event()
    threading.Thread(target=kill_on_hangup, args=(process, abandoned), daemon=True).start()
    threading.Thread(target=warn_on_notice, args=(process,), daemon=True).start()
    status = process.wait()
    status = STATUS_ABANDONED if abandoned.is_set() else status
    # The marker is written before the slot is let go, so the box is never idle (to its
    # watchdog or the PC) between the job's end and the record of its outputs. A killed
    # job's outputs are kept too: the units it finished are finished work.
    if run_id and outputs:
        mark_unfetched(name, run_id, outputs, status)
    running.unlink(missing_ok=True)
    touch_activity()
    slot.close()
    undeclared = undeclared_writes(started, outputs)
    warn_undeclared(undeclared)
    log_occupancy(name, slot_name, requested, started, status, busy, undeclared)
    return status


def log_occupancy(name: str, slot: str, requested: float, started: float, status: int,
                  busy: int, undeclared: dict[str, int]) -> None:
    """One line per finished job in OCCUPANCY_LOG (one short append: whole lines)."""
    record = {"name": name, "slot": int(slot), "requested": requested, "started": started,
              "ended": time.time(), "wait_s": round(started - requested, 1),
              "status": status, "busy_at_start": busy, "slots": len(job_slots(name)),
              "all_slots": len(list((BOX / "slots").iterdir())), "nproc": os.cpu_count(),
              "undeclared_files": sum(undeclared.values()), "undeclared": undeclared}
    with open(OCCUPANCY_LOG, "a") as log:
        log.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    # os._exit: the hang-up watcher is still blocked reading stdin, and a normal
    # interpreter exit aborts on that thread's buffered-reader lock.
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
