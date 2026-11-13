#!/usr/bin/env python3
"""Run a list of encode jobs on a rented machine, many at once (standard library only).

    python3 remote_jobs.py --jobs jobs.json --workers 180 [--only N]

The worker side of tools/dev/aws_heldout_chart.py. jobs.json is a list of

    {"id": "<clip>/<family>/<label>-<identity>", "argv": [...],
     "recon": {"frame_bytes": B, "frames": N} | null}

run from the directory that argv's relative paths start from; every job's
outputs are OUT/<id>.* (the launcher wrote those paths into argv). A job with "recon" names `{RECON}`
in its argv: the encoder's displayed reconstruction goes into a pipe and only
its per-frame xxh3-128 digests are kept (out/<id>.recon.txt, one per input
frame), so the scoring machine can prove the decoded stream is the encoder's
picture without any video coming back.

Each job prints one progress line (`job k/n ok|FAILED id: elapsed, ETA`); a
failed job writes out/<id>.failed with its exit status and stderr tail and
the others go on. A job whose outputs are complete (out/<id>.done) is
skipped, so a rerun resumes. Jobs run in the order given (the launcher puts
the slowest first).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import xxhash  # python3-xxhash (the scoring side's yuv.FRAME_HASH)

#: Where every job's outputs go, relative to the working directory (argv names them).
OUT = Path("out")
STDERR_TAIL_LINES = 12
RECON_CHUNK = 1 << 20


def hash_frames(fd: int, frame_bytes: int) -> list[str]:
    """Per-frame xxh3-128 digests of the raw frames read from `fd` until EOF."""
    digests, frame = [], bytearray()
    with os.fdopen(fd, "rb", buffering=0) as pipe:
        while chunk := pipe.read(RECON_CHUNK):
            frame += chunk
            while len(frame) >= frame_bytes:
                digests.append(xxhash.xxh3_128_hexdigest(bytes(frame[:frame_bytes])))
                del frame[:frame_bytes]
    if frame:
        raise RuntimeError(f"reconstruction ends with a partial frame ({len(frame)} bytes)")
    return digests


def run_job(job: dict, out: Path) -> None:
    base = out / job["id"]
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{base}.failed").unlink(missing_ok=True)  # a rerun's outcome replaces the last
    recon = job.get("recon")
    read_fd = write_fd = None
    argv = job["argv"]
    if recon:
        read_fd, write_fd = os.pipe()
        argv = [f"/dev/fd/{write_fd}" if a == "{RECON}" else a for a in argv]
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, pass_fds=(write_fd,) if recon else ())
    digests: list[str] = []
    if recon:
        os.close(write_fd)
        digests = hash_frames(read_fd, recon["frame_bytes"])
    _, stderr = process.communicate()
    if process.returncode != 0:
        tail = stderr.decode(errors="replace").strip().splitlines()[-STDERR_TAIL_LINES:]
        Path(f"{base}.failed").write_text(json.dumps(
            {"status": process.returncode, "stderr": tail}) + "\n")
        raise RuntimeError(f"exit {process.returncode}: {' | '.join(tail[-2:])}")
    if recon:
        if len(digests) != recon["frames"]:
            raise RuntimeError(f"{len(digests)} reconstructed frames, expected {recon['frames']}")
        Path(f"{base}.recon.txt").write_text("\n".join(digests) + "\n")
    Path(f"{base}.done").touch()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--only", type=int, help="the first N jobs only (a pipe-cleaner)")
    args = parser.parse_args(argv)
    jobs = json.loads(Path(args.jobs).read_text())[:args.only]
    out = OUT
    todo = [j for j in jobs if not (out / f"{j['id']}.done").exists()]
    print(f"{len(todo)} of {len(jobs)} jobs to run on {args.workers} workers", flush=True)
    lock, started, done, failed = threading.Lock(), time.monotonic(), [0], [0]

    def one(job: dict) -> None:
        try:
            run_job(job, out)
            outcome = "ok"
        except Exception as error:  # recorded; the run goes on
            outcome = f"FAILED ({error})"
            with lock:
                failed[0] += 1
        with lock:
            done[0] += 1
            minutes = (time.monotonic() - started) / 60
            print(f"job {done[0]}/{len(todo)} {outcome} {job['id']}: {minutes:.1f} min, "
                  f"ETA {minutes / done[0] * (len(todo) - done[0]):.1f} min", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo))
    print(f"finished: {done[0] - failed[0]} ok, {failed[0]} failed", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
