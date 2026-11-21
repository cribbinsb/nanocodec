"""Box side of nc-remote: run one agent's command in a job slot (standard library only).

tools/dev/nc_remote.py sends this file's text over SSH (`python3 -c`), so the
box always runs the PC's version:

    python3 -c <this file> NAME COMMIT JOBS COMMAND [KEY=VALUE ...]

(nc-remote passes NC_BUILD_DIRTY=<hash of the local changes, or empty> as the
first KEY=VALUE: with NC_BUILD_COMMIT it stamps the build, tools/dev/build_stamp.py.)
    python3 -c <this file> --status      (the slots and the queue)
    python3 -c <this file> --drop NAME... (remove jobs whose worktree is gone)

It waits for a free slot (one flock'd file per slot under ~/ncbox/slots; the
idle watchdog of nc_remote_box.sh counts a held slot as busy), then runs
COMMAND with bash in ~/ncbox/jobs/NAME/src, the agent's synced worktree,
where `target` is a link to the agent's own CARGO_TARGET_DIR kept on the box
(~/ncbox/jobs/NAME/target), so rebuilds are incremental. Its output streams
straight back over SSH and this process exits with its status. When the SSH
link closes (the caller was interrupted), stdin reaches EOF and the job's
whole process group is killed, so nothing runs on unattended.

Disk: a starting job evicts the least recently used other job directories while
fewer than MIN_FREE_BYTES are free (never a running or waiting job's, nor PROTECTED).

A job records itself in ~/ncbox/jobs/NAME/waiting.json while it waits for a
slot and in running.json while it runs; `--status` reads them. A record lives only
as long as its process on this boot (live_records): one whose process is gone, or
which predates the box's boot (a record baked into the image, or left by a reclaim),
is removed when read, so it never shows as holding a slot or as waiting (2026-11-21:
a five-day-old av1-writer record, baked into the AMI, showed in a verifier slot).
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
#: Exit status when the caller went away and the job was killed.
STATUS_ABANDONED = 255


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


def kill_on_hangup(process: subprocess.Popen, abandoned: threading.Event) -> None:
    """Kill the job's process group once our stdin (the SSH channel) closes."""
    sys.stdin.buffer.read()
    if process.poll() is None:
        abandoned.set()
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(5)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def evictable(keep: set[str]) -> list[Path]:
    """Job directories that may go, least recently used first: not running or waiting,
    not PROTECTED, not in `keep`."""
    live = {path.parent.name for kind in ("running.json", "waiting.json")
            for path, _ in live_records(kind)}
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
    name, commit, jobs, command, *extra = sys.argv[1:]
    job_dir = BOX / "jobs" / name
    src, target = job_dir / "src", job_dir / "target"
    target.mkdir(parents=True, exist_ok=True)
    (job_dir / LAST_USED).touch()
    make_room(name)
    if not (src / "target").is_symlink():
        (src / "target").symlink_to(target)
    slot, slot_name = take_slot(job_dir / "waiting.json", name)
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
    status = process.wait()
    running.unlink(missing_ok=True)
    touch_activity()
    slot.close()
    return STATUS_ABANDONED if abandoned.is_set() else status


if __name__ == "__main__":
    # os._exit: the hang-up watcher is still blocked reading stdin, and a normal
    # interpreter exit aborts on that thread's buffered-reader lock.
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
