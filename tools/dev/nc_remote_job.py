"""Box side of nc-remote: run one agent's command in a job slot (standard library only).

tools/dev/nc_remote.py sends this file's text over SSH (`python3 -c`), so the
box always runs the PC's version:

    python3 -c <this file> NAME COMMIT JOBS COMMAND [KEY=VALUE ...]

It waits for a free slot (one flock'd file per slot under ~/ncbox/slots; the
idle watchdog of nc_remote_box.sh counts a held slot as busy), then runs
COMMAND with bash in ~/ncbox/jobs/NAME/src, the agent's synced worktree,
where `target` is a link to the agent's own CARGO_TARGET_DIR kept on the box
(~/ncbox/jobs/NAME/target), so rebuilds are incremental. Its output streams
straight back over SSH and this process exits with its status. When the SSH
link closes (the caller was interrupted), stdin reaches EOF and the job's
whole process group is killed, so nothing runs on unattended.
"""

import fcntl
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


def take_slot():
    """Block until a slot file is locked; returns its open file (the lock)."""
    slots = sorted((BOX / "slots").iterdir())
    announced = False
    while True:
        for path in slots:
            handle = open(path, "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle, path.name
            except BlockingIOError:
                handle.close()
        if not announced:
            print(f"[nc-remote] all {len(slots)} slots busy, waiting", file=sys.stderr, flush=True)
            announced = True
        touch_activity()
        time.sleep(SLOT_POLL_SECONDS)


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
    name, commit, jobs, command, *extra = sys.argv[1:]
    job_dir = BOX / "jobs" / name
    src, target = job_dir / "src", job_dir / "target"
    target.mkdir(parents=True, exist_ok=True)
    if not (src / "target").is_symlink():
        (src / "target").symlink_to(target)
    slot, slot_name = take_slot()
    touch_activity()
    free = shutil.disk_usage(BOX).free
    if free < LOW_DISK_BYTES:
        print(f"[nc-remote] warning: {free >> 30} GiB free on the box", file=sys.stderr)
    running = job_dir / "running.json"
    process = subprocess.Popen(["bash", "-c", command], cwd=src, stdin=subprocess.DEVNULL,
                               env=job_env(target, commit, jobs, extra), start_new_session=True)
    running.write_text(json.dumps({"slot": slot_name, "pid": process.pid, "command": command,
                                   "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
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
