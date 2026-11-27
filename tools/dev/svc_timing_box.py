#!/usr/bin/env python3
"""The box side of the SVC chart's timing (tools/dev/svc_timing.py): every plan entry's
encode, one at a time on one pinned core, in rotated rounds; standard library only.

    python3 -I svc_timing_box.py --plan plan.json --bin DIR --clips DIR --out results.jsonl \
        --rounds 3 --cpu 2

The plan is nceval.svc_chart's timing plan ({"points": {point: {clip: {"argv": [...]}}}},
with {BIN}, {INPUT} and {OUTPUT} placeholders). Each encode's CPU time (user + system of the
encoder and its children, from wait4) is appended to `out` as one JSON line the moment it
ends, so a reclaimed box loses only the encode in flight; a restart skips every (point, clip,
round) already in `out`. Round r runs the entries in an order rotated by r, so a slow drift
of the machine's clock spreads over every point alike.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def done_runs(out: Path) -> set[tuple[str, str, int]]:
    """The (point, clip, round) runs `out` already holds."""
    if not out.exists():
        return set()
    runs = set()
    for line in out.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line cut by a reclaim
            continue
        runs.add((record["point"], record["clip"], record["round"]))
    return runs


def command(argv: list[str], bin_dir: Path, clips: Path, clip: str, output: Path) -> list[str]:
    """The plan's argv with its placeholders filled."""
    return [a.replace("{BIN}", str(bin_dir)).replace("{INPUT}", str(clips / f"{clip}.yuv"))
            .replace("{OUTPUT}", str(output)) for a in argv]


def cpu_seconds(cmd: list[str], cpu: int) -> float:
    """CPU time of `cmd` pinned to `cpu`; refuses a failed encode."""
    process = subprocess.Popen(["taskset", "-c", str(cpu), *cmd], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, status, usage = os.wait4(process.pid, 0)
    stderr = process.stderr.read().decode(errors="replace")
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError(f"{cmd[0]} failed: {stderr[-400:]}")
    return usage.ru_utime + usage.ru_stime


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cpu", type=int, default=2)
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())["points"]
    entries = [(point, clip) for point, clips in sorted(plan.items()) for clip in sorted(clips)]
    done = done_runs(a.out)
    if a.out.exists() and not a.out.read_bytes().endswith(b"\n"):  # a line cut by a reclaim
        with a.out.open("a") as out:
            out.write("\n")
    stream = a.out.parent / "stream.bin"
    for round_ in range(a.rounds):
        shift = round_ * len(entries) // max(a.rounds, 1)
        for point, clip in entries[shift:] + entries[:shift]:
            if (point, clip, round_) in done:
                continue
            cmd = command(plan[point][clip]["argv"], a.bin, a.clips, clip, stream)
            seconds = cpu_seconds(cmd, a.cpu)
            size = stream.stat().st_size
            with a.out.open("a") as out:
                out.write(json.dumps({"point": point, "clip": clip, "round": round_,
                                      "cpu_seconds": seconds, "bytes": size,
                                      "ended": time.time()}) + "\n")
            print(f"{round_} {point} {clip} {seconds:.3f} s", flush=True)
    stream.unlink(missing_ok=True)
    print("all runs done", flush=True)


if __name__ == "__main__":
    main()
