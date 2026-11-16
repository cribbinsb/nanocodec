"""Box side of nc-remote: run one agent's command in a job slot (standard library only).

tools/dev/nc_remote.py sends this file's text over SSH (`python3 -c`), so the
box always runs the PC's version:

    python3 -c <this file> NAME COMMIT JOBS COMMAND [KEY=VALUE ...]
    python3 -c <this file> --status      (the slots and the queue)

It waits for a free slot (one flock'd file per slot under ~/ncbox/slots; the
idle watchdog of nc_remote_box.sh counts a held slot as busy), then runs
COMMAND with bash in ~/ncbox/jobs/NAME/src, the agent's synced worktree,
where `target` is a link to the agent's own CARGO_TARGET_DIR kept on the box
(~/ncbox/jobs/NAME/target), so rebuilds are incremental. Its output streams
straight back over SSH and this process exits with its status. When the SSH
link closes (the caller was interrupted), stdin reaches EOF and the job's
whole process group is killed, so nothing runs on unattended.

A job records itself in ~/ncbox/jobs/NAME/waiting.json while it waits for a
slot and in running.json while it runs; `--status` reads them.
"""

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
#: Exit status when the caller went away and the job was killed.
STATUS_ABANDONED = 255


def touch_activity() -> None:
    (BOX / "activity").touch()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slot_held(path: Path) -> bool:
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def print_status() -> None:
    """One line per slot (free, or the job holding it) and the jobs waiting."""
    running = {}
    for path in (BOX / "jobs").glob("*/running.json"):
        job = json.loads(path.read_text())
        running[job["slot"]] = (path.parent.name, job)
    for slot in sorted((BOX / "slots").iterdir(), key=lambda p: int(p.name)):
        if slot.name in running:
            name, job = running[slot.name]
            print(f"  slot {slot.name}: {name} since {job['started']}: {job['command'][:100]}")
        else:
            print(f"  slot {slot.name}: {'busy' if slot_held(slot) else 'free'}")
    waiting = sorted((json.loads(p.read_text())["since"], p.parent.name)
                     for p in (BOX / "jobs").glob("*/waiting.json"))
    print(f"  queue: {len(waiting)} waiting" + "".join(
        f"\n    {name} since {since}" for since, name in waiting))


def caller_gone(timeout: float) -> bool:
    """Wait up to `timeout` s; True when our stdin (the SSH channel) has closed."""
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(readable) and not os.read(sys.stdin.fileno(), 1)


def take_slot(waiting: Path):
    """Block until a slot file is locked; returns its open file (the lock)."""
    slots = sorted((BOX / "slots").iterdir())
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
            waiting.write_text(json.dumps({"since": now()}))
            print(f"[nc-remote] all {len(slots)} slots busy, waiting", file=sys.stderr, flush=True)
            announced = True
        touch_activity()
        if caller_gone(SLOT_POLL_SECONDS):
            waiting.unlink(missing_ok=True)
            sys.exit(STATUS_ABANDONED)


def job_env(target: Path, commit: str, jobs: str, extra: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS=jobs, NC_BUILD_COMMIT=commit,
               NC_REMOTE_BOX="1")
    env["PATH"] = f"{Path.home() / '.cargo/bin'}:{env.get('PATH', '')}"
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


def main() -> int:
    if sys.argv[1:] == ["--status"]:
        print_status()
        return 0
    name, commit, jobs, command, *extra = sys.argv[1:]
    job_dir = BOX / "jobs" / name
    src, target = job_dir / "src", job_dir / "target"
    target.mkdir(parents=True, exist_ok=True)
    if not (src / "target").is_symlink():
        (src / "target").symlink_to(target)
    slot, slot_name = take_slot(job_dir / "waiting.json")
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
