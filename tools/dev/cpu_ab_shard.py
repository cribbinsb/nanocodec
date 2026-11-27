#!/usr/bin/env python3
"""One box of a CPU A/B (tools/dev/aws_cpu_ab.py): upload the staged jobs, check
readiness, run a pipe-cleaner round, then the timed rounds, and bring the results back.

    cpu_ab_shard.py --stage DIR --target zen5 --out DIR --rounds 5 --purpose P [--on-demand]

aws_cpu_ab starts one of these per box (aws_session's signal handling needs a main
thread). It imports aws_session only, not nceval, so a shard process stays small: 13
re-judge runs of 9 shards each once filled the PC's memory at ~110 MB a shard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aws_session import Session, SessionSpec, say  # noqa: E402

REMOTE_DIR = "cpuab"


@dataclass(frozen=True)
class Target:
    """One CPU target: the instance that has it, the encodes alone on one of its cores."""
    instance_type: str
    region: str
    ami_arch: str  # aws_session's AMI architecture
    machine: str  # what the binaries must be built for (uname -m)
    cpu: str
    #: On demand, not spot: a target the benchmark does not time on (no spot series kept).
    on_demand: bool = False
    #: The timed CPU: vCPU 1, as the benchmark's boxes (docs/benchmark-procedure.md section 6);
    #: the harness runs on another physical core (tools/bench/pinning.py).
    timed_cpu: int = 1
    #: A one-vCPU box: the harness shares the timed CPU and only waits during an encode
    #: (pinning.pin_harness, `single_cpu`).
    single_cpu: bool = False


#: The benchmark's timing machines (BP-3.10, docs/benchmark-procedure.md section 2), the same
#: instance sizes in the same region and market, so a decision's CPU is measured on the CPUs
#: the published speeds come from. The 2xlarge sizes give the harness a core of its own and
#: were measured stable (within 1.5 % between boxes, 0.5-1.1 % drift in 35 minutes); the
#: .large sizes used until 2026-11-24 shared the timed core's SMT sibling (Intel) or the
#: CCD's L3 with other tenants (c8a.large: up to 12.6 % between boxes, steps of about 2 %).
#: zen5 (c8a.2xlarge, on demand, the benchmark's superseded series) is optional, never a
#: default: decisions are charged on the default pair (nceval.decision_cpu).
TARGETS = {
    "graniterapids": Target("c8i.2xlarge", "eu-north-1", "amd64", "x86_64",
                            "Xeon 6 Granite Rapids"),
    "graviton4": Target("c8g.2xlarge", "eu-north-1", "arm64", "aarch64",
                        "Graviton4, Neoverse V2"),
    #: Graviton4 on one vCPU (c8g.medium, about 1/10 of c8g.2xlarge's spot price): measured as
    #: quiet as c8g.2xlarge on 2026-11-24 (docs/aws-cost-guide.md); not a default until the
    #: owner moves the benchmark's Graviton series.
    "graviton4medium": Target("c8g.medium", "eu-north-1", "arm64", "aarch64",
                          "Graviton4, Neoverse V2, one vCPU", timed_cpu=0, single_cpu=True),
    "zen5": Target("c8a.2xlarge", "eu-west-1", "amd64", "x86_64", "EPYC Turin, Zen 5",
                   on_demand=True),
}
#: Every decision measures the benchmark's two CPU families (owner, 2026-11-21: no material
#: regression on any target; 2026-11-24: dev measures on the benchmark's machines).
DEFAULT_TARGETS = ("graniterapids", "graviton4")
#: The spot vCPUs one A/B run may hold in a region: eu-north-1's 1200-vCPU spot quota is
#: shared with the benchmark fleet (2026-11-24), which must not be starved.
MAX_RUN_VCPUS = 200
#: vCPUs of each target's instance (the 2xlarge sizes).
TARGET_VCPUS = 8


#: Per box: budget cap and hard lifetime (a 2xlarge is about $0.1-0.5 an hour).
BOX_BUDGET_USD = 1.0
BOX_LIFETIME_MINUTES = 75
PREPARE_MINUTES = 8
#: The timed run prints a line per job (both variants), so silence spans a whole job: a
#: 900-frame veryslow pair on a Graviton core takes several minutes, past aws_session's
#: 5-minute default (2026-11-23: a veryslow re-judge shard failed twice on "no progress").
RUN_STALL_MINUTES = 20


# --- one box ---------------------------------------------------------------------------

def prepare_command(cpu: int) -> str:
    """Readiness: clips unpacked and read once (page cache), counters allowed, every
    binary runs, the CPU and the counted events recorded."""
    return (f"set -o pipefail; cd ~/{REMOTE_DIR} && for c in clips/*.zst; do zstd -d -q --rm \"$c\"; done && "
            # counters for a user process (a box without sudo just counts nothing)
            "{ sudo -n sysctl -q kernel.perf_event_paranoid=1 || true; } && "
            "for b in bin/*/nanocodec; do echo \"$b $($b --version)\"; done | tee out/versions.txt && "
            "lscpu > out/lscpu.txt && grep -m1 'Model name' out/lscpu.txt && "
            "cat clips/* > /dev/null && "
            f"python3 cpu_ab_box.py --plan plan.json --probe out/events.json --core {cpu}")


#: Job ids an earlier box of this shard finished (staged by aws_cpu_ab.launch_shards before
#: a retry), which the box leaves out.
DONE_FILE = "done.json"
#: download_new's marker for the run's incremental pulls (Session.pulling).
PULL_MARKER = ".pulled"


def box_command(cpu: int, rounds: int, out: str, limit: int | None = None,
                single_cpu: bool = False) -> str:
    return (f"cd ~/{REMOTE_DIR} && python3 cpu_ab_box.py --plan plan.json --out {out} "
            f"--events out/events.json --core {cpu} --rounds {rounds} --skip {DONE_FILE}"
            + (f" --limit {limit}" if limit else "") + (" --single-cpu" if single_cpu else ""))


def job_frames(job: dict) -> int:
    """The frames one job encodes (its base variant's `--frames`)."""
    argv = job["variants"]["base"]
    return int(argv[argv.index("--frames") + 1]) if "--frames" in argv else 1


def estimate_minutes(pipe_runs: Path, jobs: list[dict], rounds: int) -> float:
    """The full run's expected wall from the pipe-cleaner (one round of the first job),
    scaled by frames. Only printed: a shard mixes codecs and 150- to 900-frame windows,
    so the first job alone cannot bound the rest (2026-11-23: a 150-frame H.264 first
    job set a 19-minute limit on 900-frame AV1 and VVC jobs and failed a shard twice).
    The run is bounded by the box's lifetime and aws_session's stall watch instead."""
    seconds = sum(json.loads(line)["wall_s"] for line in pipe_runs.read_text().splitlines())
    per_frame = seconds / job_frames(jobs[0])
    return per_frame * sum(job_frames(job) for job in jobs) * rounds / 60


def run_shard(stage_dir: Path, target: Target, out: Path, rounds: int, on_demand: bool,
              purpose: str, session_for=Session, budget_usd: float = BOX_BUDGET_USD) -> None:
    """One box: upload, readiness, pipe-cleaner, the rounds, results back."""
    plan_jobs = json.loads((stage_dir / "plan.json").read_text())["jobs"]
    jobs = len(plan_jobs)
    spec = SessionSpec(region=target.region, instance_type=target.instance_type,
                       arch=target.ami_arch, purpose=purpose,
                       on_demand=on_demand or target.on_demand,
                       lifetime_minutes=BOX_LIFETIME_MINUTES, budget_usd=budget_usd,
                       root_gb=16, spill=False)
    remote_out = f"~/{REMOTE_DIR}/out"
    started = time.monotonic()
    with session_for(spec) as session:
        session.upload(stage_dir, f"~/{REMOTE_DIR}")
        try:
            session.run_detached(prepare_command(target.timed_cpu), f"{remote_out}/prepare.log",
                                 PREPARE_MINUTES)
            # The pipe-cleaner is one round of the first job, which may be the shard's
            # longest (a 900-frame slower pair: about 13 minutes on a Graviton 3 core).
            session.run_detached(box_command(target.timed_cpu, 1, "out/pipe.jsonl", 1, target.single_cpu),
                                 f"{remote_out}/pipe.log", RUN_STALL_MINUTES,
                                 stall_minutes=RUN_STALL_MINUTES)
            session.download(remote_out, out)
            timeout = BOX_LIFETIME_MINUTES - PREPARE_MINUTES - RUN_STALL_MINUTES
            say(f"{jobs} jobs x {rounds} rounds, about "
                f"{estimate_minutes(out / 'pipe.jsonl', plan_jobs, rounds):.0f} min "
                f"(first job's rate), limit {timeout} min")
            # Each finished job's records come back within PULL_SECONDS (and at once on a
            # spot notice: Session.pulling), so a reclaim loses only the job in flight.
            with session.pulling(remote_out, out, PULL_MARKER):
                session.run_detached(box_command(target.timed_cpu, rounds, "out/runs.jsonl",
                                                 single_cpu=target.single_cpu),
                                     f"{remote_out}/runs.log", timeout,
                                     stall_minutes=RUN_STALL_MINUTES)
        finally:
            session.download(remote_out, out)
    (out / "session.json").write_text(json.dumps({
        "instance_type": target.instance_type, "region": target.region,
        "minutes": round((time.monotonic() - started) / 60, 2),
        "on_demand": on_demand or target.on_demand,
        "jobs": jobs, "rounds": rounds}) + "\n")


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--target", required=True, choices=sorted(TARGETS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--purpose", required=True)
    p.add_argument("--on-demand", action="store_true")
    p.add_argument("--budget", type=float, default=BOX_BUDGET_USD, help="per-box cap, USD")
    a = p.parse_args(argv)
    run_shard(a.stage, TARGETS[a.target], a.out, a.rounds, a.on_demand, a.purpose,
              budget_usd=a.budget)


if __name__ == "__main__":
    main(sys.argv[1:])
