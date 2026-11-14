#!/usr/bin/env python3
"""Run a staged standards-lens run (eval/nceval/ctc) on one rented spot instance.

    python3 tools/dev/aws_ctc.py --run-dir DIR --results DIR --instance-type c7a.metal-48xl \
        --lifetime 150 --budget 4 --minutes 130 [--workers 96] [--purpose ctc-full] [--dry-run]

`--run-dir` is a directory written by `python3 -m nceval.ctc stage` (with
`--source-dir /mnt/ctc`). The session (tools/dev/aws_session.py: spot,
Project=nanocodec and ExpectedEnd tags, self-termination at the lifetime,
ledger rows, cleanup on every exit) uploads it, then on the instance:

1. readiness: every tool links and runs, a tmpfs for the sources;
2. `worker.py prepare`: the sequences downloaded from their public URLs and
   checked against the CTC's MD5s (only results ever come back);
3. `worker.py run`, detached; every CHECKPOINT_MINUTES the new results come
   home (`Session.download_new`), so a spot reclaim loses at most that much.
   The run limit must leave LIFETIME_RESERVE_MINUTES of the lifetime (SSH
   closes 5 minutes before the self-termination).

Results land in `--results` (out/results/*.json, out/hashes/*.md5,
out/sources.json, the logs).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REMOTE_DIR = "ctc"
#: The sources' tmpfs (RAM, not /dev/shm: logind clears that when SSH sessions end).
SOURCE_MOUNT = "/mnt/ctc"
TMPFS_MARGIN = 1.2
TMPFS_FLOOR_MB = 256
PREPARE_MINUTES = 10
CHECKPOINT_MINUTES = 5
#: `shutdown -h +N` (the session's hard lifetime) closes SSH logins 5 minutes
#: before the end (pam_nologin), so the run, its last checkpoint and the logs
#: must be home by then: the run limit leaves this much of the lifetime.
LIFETIME_RESERVE_MINUTES = 9
#: ExpectedEnd margin over the lifetime: the instance stops itself at the
#: lifetime; this covers only the cleanup's API calls.
END_MARGIN_MINUTES = 5
POLL_SECONDS = 30
#: The worker prints a line per finished job; a 600-frame 720p HM point takes
#: over an hour, so "no news" is judged over that long, not the session default.
STALL_MINUTES = 90
CHECKPOINT_MARKER = ".ctc-checkpoint"
REGION = "eu-north-1"


def tmpfs_mb(run_dir: Path) -> int:
    sources = json.loads((run_dir / "sources.json").read_text())
    total = sum(s["frames"] * s["frame_bytes"] for s in sources)
    return round(total * TMPFS_MARGIN / 1e6) + TMPFS_FLOOR_MB


def prepare_command(run_dir: Path) -> str:
    """Readiness checks, the tmpfs and the sources, failing at the first problem."""
    return " && ".join([
        f"cd ~/{REMOTE_DIR}",
        "! ldd bin/* 2>/dev/null | grep 'not found'",
        "bin/nanocodec --version | tee out-version.txt",
        "bin/vmaf --version",
        f"sudo mkdir -p {SOURCE_MOUNT}",
        f"sudo mount -t tmpfs -o size={tmpfs_mb(run_dir)}m,mode=1777 tmpfs {SOURCE_MOUNT}",
        "python3 -I worker.py prepare --workers 8 --scratch /tmp/ctc-scratch",
        "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs | tee out/platform.txt",
        "mv out-version.txt out/nanocodec-version.txt",
    ])


def start_detached(session, command: str, log: str) -> None:
    job = shlex.quote(f"{command}; echo $? > {log}.status")
    session.ssh(f"rm -f {log}.status; setsid nohup bash -c {job} > {log} 2>&1 < /dev/null & "
                f"echo $! > {log}.pid")


def watch_with_checkpoints(session, log: str, results: Path, minutes: float) -> None:
    """Follow the detached run, bringing results home every CHECKPOINT_MINUTES, until it
    exits, stalls or runs past `minutes` (then it is stopped; finished jobs are kept)."""
    from aws_session import RemoteFailure, say

    started = last_checkpoint = last_news = time.monotonic()
    shown = 0
    while True:
        time.sleep(POLL_SECONDS)
        probe = session.ssh(f"cat {log}.status 2>/dev/null; echo ---; tail -n +{shown + 1} {log}",
                            check=False, quiet=True)
        if probe.returncode == 0:
            status, _, new = probe.stdout.partition("---\n")
            lines = new.splitlines()
            for line in lines:
                print(f"  remote: {line}", flush=True)
            if lines:
                shown, last_news = shown + len(lines), time.monotonic()
            if status.strip():
                if status.strip() != "0":
                    raise RemoteFailure(f"run exit status {status.strip()}", lines[-20:])
                return
        now = time.monotonic()
        if now - last_checkpoint > CHECKPOINT_MINUTES * 60:
            checkpoint(session, results)
            last_checkpoint = now
        if now - last_news > STALL_MINUTES * 60 or now - started > minutes * 60:
            session.ssh(f"kill -TERM -- -$(cat {log}.pid) 2>/dev/null; true", check=False)
            raise RemoteFailure("stalled or past the run limit; stopped", [])


def checkpoint(session, results: Path) -> None:
    """Bring new results home; a failed checkpoint is reported and retried next time."""
    from aws_session import say

    try:
        moved = session.download_new(f"~/{REMOTE_DIR}/out", results, CHECKPOINT_MARKER)
        say(f"checkpoint: {moved / 1e6:.1f} MB of results home")
    except RuntimeError as error:
        say(f"checkpoint failed ({error}); retrying at the next one")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--workers", type=int, required=True,
                        help="parallel jobs; each uses up to two cores (encoder + vmaf)")
    parser.add_argument("--lifetime", type=int, required=True, help="hard lifetime, minutes")
    parser.add_argument("--minutes", type=float, required=True, help="run limit, minutes")
    parser.add_argument("--budget", type=float, required=True, help="session cap, USD")
    parser.add_argument("--root-gb", type=int, default=24)
    parser.add_argument("--purpose", default="ctc")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_dir, results = Path(args.run_dir), Path(args.results)
    if args.minutes > args.lifetime - LIFETIME_RESERVE_MINUTES:
        raise SystemExit(f"--minutes must leave {LIFETIME_RESERVE_MINUTES} min of the lifetime")
    run = f"cd ~/{REMOTE_DIR} && python3 -I worker.py run --workers {args.workers}"
    if args.dry_run:
        print(f"prepare: {prepare_command(run_dir)}\nrun: {run}")
        return
    from aws_session import Session, SessionSpec, say

    spec = SessionSpec(region=REGION, instance_type=args.instance_type, purpose=args.purpose,
                       lifetime_minutes=args.lifetime, budget_usd=args.budget,
                       root_gb=args.root_gb, end_margin_minutes=END_MARGIN_MINUTES)
    with Session(spec) as session:
        session.upload(run_dir, f"~/{REMOTE_DIR}")
        try:
            say("remote: prepare")
            session.run_detached(prepare_command(run_dir), f"~/{REMOTE_DIR}/prepare.log",
                                 PREPARE_MINUTES)
            say("remote: run")
            start_detached(session, run, f"~/{REMOTE_DIR}/run.log")
            watch_with_checkpoints(session, f"~/{REMOTE_DIR}/run.log", results, args.minutes)
        finally:  # whatever finished comes home before the instance goes
            checkpoint(session, results)
            for log in ("prepare.log", "run.log"):
                probe = session.ssh(f"cat ~/{REMOTE_DIR}/{log} 2>/dev/null", check=False,
                                    quiet=True)
                (results / log).write_text(probe.stdout)
    failed = sorted((results / "results").rglob("*.failed"))
    say(f"{len(failed)} failed jobs" + "".join(f"\n  {p}" for p in failed[:20]))


if __name__ == "__main__":
    main(sys.argv[1:])
