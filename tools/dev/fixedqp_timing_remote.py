"""The timing machine's side of tools/dev/fixedqp_timing.py (standard library only).

    python3 -I fixedqp_timing_remote.py   # in the staged directory: plan.json, bin/, cfg/, clips/

Times each job of plan.json (one encode: an encoder at one quantizer on one
sequence) as CPU time (user + system, rusage) of the encoder process pinned
to one core, in rotated rounds: round 0 is a warm-up for the jobs timed
more than once, and a job's `rounds` are the timed rounds it takes part
in (the slow references run once: their encodes take minutes). Bitstreams
are written to scratch/ and deleted after their size is recorded.
out/times.json is rewritten after every round, so a stopped run keeps the
finished rounds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
OUT = HERE / "out"
SCRATCH = HERE / "scratch"


def concrete(argv: list[str], sequence: str) -> list[str]:
    """The plan's command with its placeholders filled for this directory."""
    fill = {"{BIN}": str(HERE / "bin"), "{CFG}": str(HERE / "cfg"),
            "{INPUT}": str(HERE / "clips" / f"{sequence}.yuv"),
            "{OUTPUT}": str(SCRATCH / "stream.bin")}
    out = []
    for arg in argv:
        for key, value in fill.items():
            arg = arg.replace(key, value)
        out.append(arg)
    return out


def main() -> int:
    plan = json.loads((HERE / "plan.json").read_text())
    jobs, core = plan["jobs"], plan["core"]
    OUT.mkdir(exist_ok=True)
    SCRATCH.mkdir(exist_ok=True)
    results = {job["id"]: {"cpu_s": [], "bytes": None} for job in jobs}
    rounds = max(job["rounds"] for job in jobs)
    for r in range(rounds + 1):
        # round 0 warms up the repeated jobs; a job takes part in rounds 1..its own count
        todo = [j for j in jobs if (r == 0 and j["rounds"] > 1) or 1 <= r <= j["rounds"]]
        shift = r % max(len(todo), 1)
        for job in todo[shift:] + todo[:shift]:
            seconds, size = timed(job, core)
            if seconds is None:
                continue
            print(f"r{r} {job['id']} {seconds:.2f}s", flush=True)
            if r:
                results[job["id"]]["cpu_s"].append(seconds)
            results[job["id"]]["bytes"] = size
        (OUT / "times.json").write_text(json.dumps(results, indent=1) + "\n")
        print(f"round {r} done ({len(todo)} encodes)", flush=True)
    print("TIME-DONE", flush=True)
    return 0


def timed(job: dict, core: int) -> tuple[float | None, int | None]:
    """CPU seconds (user + system of the encoder, from its rusage) and stream bytes."""
    stream = SCRATCH / "stream.bin"
    stream.unlink(missing_ok=True)
    cmd = ["taskset", "-c", str(core), *concrete(job["argv"], job["sequence"])]
    with open(SCRATCH / "stderr.txt", "wb") as err:
        proc = subprocess.Popen(cmd, cwd=SCRATCH, stdout=subprocess.DEVNULL, stderr=err)
        _, status, usage = os.wait4(proc.pid, 0)
    if status:
        tail = (SCRATCH / "stderr.txt").read_bytes()[-400:].decode(errors="replace")
        print(f"ERROR {job['id']}: status {status}: {tail}", flush=True)
        return None, None
    size = stream.stat().st_size if stream.exists() else 0
    stream.unlink(missing_ok=True)
    return usage.ru_utime + usage.ru_stime, size


if __name__ == "__main__":
    sys.exit(main())
