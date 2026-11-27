"""The tuner's steering CPU: callgrind on the trial's own commands.

Owner rule (2026-11-23): never combine measurements of different things; callgrind
may steer a search only if it runs the same commands as the quality it is weighed
against. Until then the tuner counted instructions at a fixed QP on 13 frames of three
fixed clips (tune.instruction_cost, the key frame amortised over a nominal period),
while its trials' quality is CBR at each clip's rung on the study's draw, 150 frames
with one key or a long window with periodic keys.

Here the counted commands are a trial's own (nceval.decision_cpu): the candidate's
single points on the trial's clips (`jobs_of`, Runner.points with the same rung
ladder, frames and key period as the quality), the declared subset (per group and
class, the longest clip at its first coded rate), each run whole under callgrind with
the setting's own arguments. A setting's CPU is the geometric mean over those jobs of
its instructions over the anchor's on the same job (`ratio`).

Steering only: adoption CPU is timed on rented cores (decision_cpu), never this.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import decision_cpu, realuse, slots
from .cbr import Family
from .clipframes import ClipFrames
from .clips import Clip
from .config import load_paths
from .singlepoint import rung_ladder
from .tune import _callgrind_total, run

#: What a study's stored steering anchor was counted with (a study counted otherwise is
#: not resumed: its trials' CPU would mix two measurements).
PROTOCOL = "own-commands-2026-11-23"


def jobs_of(runner: realuse.Runner, anchor: Family, clips: list[Clip], rungs: dict[str, int],
            frames: int, key_seconds: float | None = None,
            key_frames: int | None = None) -> list[decision_cpu.CpuJob]:
    """The declared subset of the single points a trial codes on `clips` (singlepoint:
    each clip's assigned rung) with `frames`, the key period as the quality's."""
    points = runner.points([anchor], clips, frames, None, key_seconds,
                           ladder=rung_ladder(rungs), key_frames=key_frames)
    return decision_cpu.declared_subset(decision_cpu.cpu_jobs(points, anchor.name))


def key_of(job: decision_cpu.CpuJob) -> str:
    return hashlib.sha256(json.dumps(job.record(), sort_keys=True).encode()).hexdigest()[:16]


def instructions(binary: Path, extra: tuple[str, ...], jobs: list[decision_cpu.CpuJob],
                 clips: dict[str, Clip]) -> dict[str, int]:
    """{job key: instructions} of each job run whole under callgrind with `extra` (the
    setting's arguments), side by side, one CPU slot each."""
    def count(job: decision_cpu.CpuJob) -> tuple[str, int]:
        work_root = load_paths().work
        work_root.mkdir(parents=True, exist_ok=True)
        clip = replace(clips[job.clip], start_frame=job.start_frame)
        with tempfile.TemporaryDirectory(prefix="nanocodec-steer-", dir=work_root) as work, \
                ClipFrames(clip, job.frames, Path(work) / "frames") as served, \
                slots.cpu_slots(1):
            out = Path(work) / "callgrind.out"
            fill = {decision_cpu.BIN: str(binary), decision_cpu.INPUT: str(served.reader()),
                    decision_cpu.OUTPUT: "/dev/null"}
            run(["valgrind", "--tool=callgrind", f"--callgrind-out-file={out}",
                 *(fill.get(a, a) for a in job.argv), *extra])
            return key_of(job), _callgrind_total(out)
    with ThreadPoolExecutor(max(1, len(jobs))) as threads:
        return dict(threads.map(count, jobs))


def ratio(test: dict[str, int], anchor: dict[str, int],
          keys: list[str] | None = None) -> float:
    """The geometric mean over the jobs (`keys`, default all) of test over anchor."""
    keys = keys if keys is not None else sorted(anchor)
    missing = [k for k in keys if k not in test or k not in anchor]
    if missing or not keys:
        raise ValueError(f"steering CPU: jobs not counted on both sides: {missing or 'none'}")
    return math.exp(sum(math.log(test[k] / anchor[k]) for k in keys) / len(keys))


def ratios_by_period(test: dict[str, int], anchor: dict[str, int],
                     jobs: list[decision_cpu.CpuJob]) -> dict[int, float]:
    """`ratio` per key period of the jobs (a key share's encodes)."""
    periods = sorted({j.intra for j in jobs})
    return {p: ratio(test, anchor, [key_of(j) for j in jobs if j.intra == p]) for p in periods}


def record(jobs: list[decision_cpu.CpuJob], anchor: dict[str, int]) -> dict:
    """A study's stored steering set: its jobs and the anchor's counts."""
    return {"protocol": PROTOCOL, "jobs": [j.record() for j in jobs], "anchor": anchor}


def jobs_from(stored: dict) -> list[decision_cpu.CpuJob]:
    return [decision_cpu.CpuJob.from_record(j) for j in stored["jobs"]]
