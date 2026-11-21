#!/usr/bin/env python3
"""The measuring side of tools/dev/aws_cpu_ab.py: interleaved A/B rounds on one core.

    python3 cpu_ab_box.py --plan plan.json --out runs.jsonl [--core 1] [--rounds 5]
        [--limit N] [--events EVENTS.json] [--probe EVENTS.json]

Runs on the rented box (or anywhere with the plan's binaries and clips). The plan's
jobs each name two or more variants (argv lists: a binary and its arguments); for
every job, each round runs every variant once, pinned to `--core`, in alternating
order (A B, then B A, ...) so a drift within the job falls on both sides. Each run
is counted with tools/dev/jperf.py's `measure`: user-space hardware events where the
kernel offers them (EVENTS, per architecture; a VM without a PMU gets none) and
always the child's CPU and wall time. One JSON line per run goes to `--out`, and one
progress line per job to stdout (the session's stall watch reads it).

`--probe FILE` finds what this machine counts without multiplexing: a VM's PMU may
offer fewer counters than its core has (2026-11-21, c6g/c7g/c8g.large: cycles,
instructions and one more event; seven at once left every count partial), so every
run counts BASE_EVENTS and the remaining counters take the supported extra events in
turn, round by round (`schedule`). It writes the schedule to FILE, which `--events`
reads. `--limit` measures only the first N jobs (the pipe-cleaner).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jperf import CYCLES, INSTRUCTIONS, PERF_TYPE_HARDWARE, PERF_TYPE_RAW, measure  # noqa: E402

#: Generic hardware event numbers (linux/perf_event.h) used on x86.
HW_CACHE_MISSES = 3
HW_BRANCH_MISSES = 5
#: Arm architectural PMU event numbers (Arm ARM, D17 "PMU events"), the same on the
#: Cortex-A78AE and Neoverse N1 / V1 / V2: the memory side first (what tells a
#: cache-bound change), then branches.
ARM_EVENTS = {"l2d_refill": 0x17, "ll_miss_rd": 0x37, "stall_backend": 0x24,
              "l1d_refill": 0x03, "branch_mispred": 0x10}
#: Counted on every run where the machine offers them; the rest of EVENTS are extras.
BASE_EVENTS = ("cycles", "instructions")
EVENTS = {
    "aarch64": {"cycles": CYCLES, "instructions": INSTRUCTIONS,
                **{name: (PERF_TYPE_RAW, number) for name, number in ARM_EVENTS.items()}},
    "x86_64": {"cycles": CYCLES, "instructions": INSTRUCTIONS,
               "cache_misses": (PERF_TYPE_HARDWARE, HW_CACHE_MISSES),
               "branch_mispred": (PERF_TYPE_HARDWARE, HW_BRANCH_MISSES)},
}
DEFAULT_CORE = 1
DEFAULT_ROUNDS = 5


def counts_together(argv: list[str], core: int, events: dict, names: list[str]) -> bool:
    run = measure(argv, core, {n: events[n] for n in names})
    return all(run.counts[n] is not None for n in names)


def probe_events(argv: list[str], core: int, events: dict) -> dict:
    """{"base": the BASE_EVENTS counted, "extras": the other events each counted beside
    them, "slots": how many extras count at once beside them} from runs of `argv`."""
    base = [n for n in BASE_EVENTS if counts_together(argv, core, events, [n])]
    if not counts_together(argv, core, events, base):
        base = base[:1]
    extras = [n for n in events if n not in BASE_EVENTS
              and counts_together(argv, core, events, [*base, n])]
    slots = 1 if extras else 0
    while slots < len(extras) and counts_together(argv, core, events,
                                                  [*base, *extras[:slots + 1]]):
        slots += 1
    return {"base": base, "extras": extras, "slots": slots}


def schedule(probe: dict, round_index: int) -> list[str]:
    """The events of one round: the base ones and the next `slots` extras in turn."""
    extras = probe["extras"]
    turn = [extras[(round_index * probe["slots"] + i) % len(extras)]
            for i in range(min(probe["slots"], len(extras)))]
    return [*probe["base"], *turn]


def round_order(variants: list[str], round_index: int) -> list[str]:
    """Every variant once, the order reversed on odd rounds."""
    return variants if round_index % 2 == 0 else variants[::-1]


def run_job(job: dict, rounds: int, core: int, events: dict, probe: dict) -> list[dict]:
    """All rounds of one job: one record per run."""
    names = list(job["variants"])
    records = []
    for index in range(rounds):
        counted = {n: events[n] for n in schedule(probe, index)}
        for name in round_order(names, index):
            run = measure(job["variants"][name], core, counted)
            records.append({"job": job["id"], "variant": name, "round": index,
                            "counts": run.counts, "user_s": run.user_s,
                            "system_s": run.system_s, "wall_s": run.wall_s})
    return records


def progress(job: dict, records: list[dict]) -> str:
    """One line: each variant's least cycles (else wall time) against the first's."""
    names = list(job["variants"])

    def least(name: str) -> float:
        runs = [r for r in records if r["variant"] == name]
        cycles = [r["counts"].get("cycles") for r in runs]
        if all(c is not None for c in cycles):
            return float(min(cycles))
        return min(r["wall_s"] for r in runs)

    base = least(names[0])
    return " ".join(f"{name} {100 * (least(name) / base - 1):+.2f}%" for name in names[1:])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--core", type=int, default=DEFAULT_CORE)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--limit", type=int, help="only the first N jobs")
    parser.add_argument("--probe", type=Path, help="write the events counted together, exit")
    parser.add_argument("--events", type=Path, help="the events to count (a --probe file)")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    events = EVENTS[platform.machine()]
    jobs = plan["jobs"][:args.limit] if args.limit else plan["jobs"]
    if args.probe:
        found = probe_events(next(iter(jobs[0]["variants"].values())), args.core, events)
        args.probe.write_text(json.dumps({"machine": platform.machine(), **found}))
        print(f"counted on every run: {', '.join(found['base']) or 'none'}; extras "
              f"{', '.join(found['extras']) or 'none'}, {found['slots']} at a time", flush=True)
        return
    probe = json.loads(args.events.read_text()) if args.events else {
        "base": [n for n in BASE_EVENTS if n in events],
        "extras": [n for n in events if n not in BASE_EVENTS], "slots": len(events)}
    if args.out is None:
        parser.error("--out is required")
    with args.out.open("a") as out:
        for number, job in enumerate(jobs, 1):
            records = run_job(job, args.rounds, args.core, events, probe)
            out.writelines(json.dumps(r) + "\n" for r in records)
            out.flush()
            wall = math.fsum(r["wall_s"] for r in records)
            print(f"job {number}/{len(jobs)} {job['id']}: {progress(job, records)} "
                  f"({wall:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
