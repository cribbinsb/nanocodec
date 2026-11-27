"""The CBR charts' timing commands: a subset of the very encodes their quality points run.

A chart's speed axis must time what its quality axis scores. Its quality points are
real-time CBR on the real-use ladder (`Runner.points`: every family, every chart clip,
every rung, DEFAULT_FRAMES frames, one key frame) plus the low rungs on some clips; its
timing re-runs those same encodes (`every_job`: all of them, the published speed axis;
`timing_jobs`: one rung on each of three clips, `TIMING_SLOTS`, a cross-check only),
every encoder (ours and every reference) in its own CBR mode, so rate control
(per-frame and in-frame quantizer choice, key boost and caps, holds, re-codes, adaptive
structure) is part of the time.
Until 2026-11-23 the timing ran each encoder at a fixed quantizer matched to the
600 kbps rung's bytes (anchor_chart_timing.py `scan`): a different encoder mode from
the one the chart scores, so frames timed before then are fixed-QP speeds.

The timing commands are built here and only here, by the quality path's own job list
(`Runner.points`) and command (`Bench.command`), so the two cannot diverge;
tests/test_chart_timing.py checks the stored plan against them.

    every_job(runner, families, clips, low_clips) -> every quality encode of the chart
                                            (the speed axis: each point timed on all of them)
    timing_jobs(runner, families, clips) -> one rung on each of three clips (a cheap
                                            cross-check, never the published speed)
    plan_entry(bench, point)        -> one plan entry: the command with {BIN}/{INPUT}/
                                       {OUTPUT} placeholders (tools/dev/anchor_chart_timing.py)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .cbr import Bench, Family, Point, clip_targets
from .clips import Clip
from .realuse import DEFAULT_FRAMES, Runner

#: The timing slots: per content class, the rung it is timed at (720p30 kbps of the
#: real-use ladder, pixel-rate scaled like every chart point). Each slot takes the first
#: clip of its class (by name) in the chart's own quality clips whose ladder has that rung
#: (`timing_clips`): three classes the earlier timing set covered (low light, crowd,
#: bodycam), the rungs spread over the ladder (its cheapest rung, 300 since BP-3.10, is not
#: timed: three
#: encodes per point keep a release's timing session inside its lifetime,
#: aws_timing.REPEAT_TIMEOUT_MINUTES). On the main chart: lowlight_bdd_00ce95b0,
#: crowd_mot20_06, handheld_mot17_10.
TIMING_SLOTS: tuple[tuple[str, int], ...] = (("lowlight", 300), ("crowd", 800), ("bodycam", 1200))
#: Where full timing plans are kept (durable, outside the repository: an every-encode plan
#: holds thousands of commands); the repository keeps a small record of each
#: (`plan_record`), which names its plan here by content.
PLAN_STORE = (Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
              / "charts" / "timing-cbr" / "plans")
#: The plan's placeholders for the encoder's directory, the input clip and the stream.
BIN, INPUT, OUTPUT = "{BIN}", "{INPUT}", "{OUTPUT}"


def timing_rungs(clips: list[Clip]) -> dict[str, int]:
    """{clip name: reference rung} of the timing clips among a chart's quality `clips`.

    A slot takes the first clip of its class (by name) with that rung. A slot whose class
    the chart does not have (VC-1K's clips are all `videocall`) takes a clip of any class
    with that rung instead, evenly spaced by name over those left (`spread_pick`), so such
    a chart's slots spread over its strata, which its names lead with."""
    chosen = {}
    for index, (content_class, rung) in enumerate(TIMING_SLOTS):
        def fits(c: Clip) -> bool:
            return c.name not in chosen and bool(clip_targets(c, (rung,)))
        own = sorted((c for c in clips if c.content_class == content_class and fits(c)),
                     key=lambda c: c.name)
        if own:
            chosen[own[0].name] = rung
        elif not any(c.content_class == content_class for c in clips):
            chosen[spread_pick(sorted((c for c in clips if fits(c)), key=lambda c: c.name),
                               index, content_class, rung).name] = rung
        else:
            raise ValueError(f"no {content_class} clip with a {rung} kbps rung to time")
    return chosen


def spread_pick(candidates: list[Clip], index: int, content_class: str, rung: int) -> Clip:
    """The `index`-th of len(TIMING_SLOTS) evenly spaced positions over `candidates`."""
    if not candidates:
        raise ValueError(f"no clip with a {rung} kbps rung for the {content_class} slot")
    return candidates[index * len(candidates) // len(TIMING_SLOTS)]


def timing_kbps(clip: Clip, rung: int) -> int:
    """The clip's target for reference `rung` (cbr.clip_targets)."""
    return clip_targets(clip, (rung,))[0]


def quality_jobs(runner: Runner, families: list[Family], clips: list[Clip]
                 ) -> list[tuple[Bench, Point]]:
    """The chart's quality encodes on the real-use ladder, as nceval.chart measure
    lists them."""
    return runner.points(families, clips, DEFAULT_FRAMES, None)


def timing_jobs(runner: Runner, families: list[Family], clips: list[Clip]
                ) -> list[tuple[Bench, Point]]:
    """The timed encodes of a chart on quality `clips`: of its quality jobs on the timing
    clips, each clip's timed rung."""
    rungs = timing_rungs(clips)
    timed_clips = [c for c in clips if c.name in rungs]
    jobs = [(bench, point) for bench, point in quality_jobs(runner, families, timed_clips)
            if point.kbps == timing_kbps(point.clip, rungs[point.clip.name])]
    timed = {(p.family.name, p.clip.name) for _, p in jobs}
    absent = [(f.name, c) for f in families for c in rungs if (f.name, c) not in timed]
    if absent:
        raise ValueError(f"timed rungs missing from the quality ladder: {absent}")
    return jobs


def every_job(runner: Runner, families: list[Family], clips: list[Clip],
              low_clips: list[str]) -> list[tuple[Bench, Point]]:
    """Every quality encode of a chart: nceval.chart's own job list (chart_jobs: the
    real-use ladder and the upper rungs on every clip, the low rungs on `low_clips`)."""
    from .chart import chart_jobs

    low = [c for c in clips if c.name in low_clips]
    if len(low) != len(low_clips):
        raise ValueError(f"low-rung clips not among the chart's: {low_clips}")
    return chart_jobs(runner, families, clips, DEFAULT_FRAMES, None, low)


def plan_jobs(runner: Runner, families: list[Family], clips: list[Clip], plan: dict
              ) -> list[tuple[Bench, Point]]:
    """The jobs a plan's header names: every quality encode (`all_encodes`) or the
    timing clips' timed rungs."""
    if plan.get("all_encodes"):
        return every_job(runner, families, clips, plan.get("low_clips", []))
    return timing_jobs(runner, families, clips)


def entry_key(point: Point, all_encodes: bool) -> str:
    """A plan entry's key: the clip (one timed rung each), or clip@kbps (every encode)."""
    return f"{point.clip.name}@{point.kbps}" if all_encodes else point.clip.name


def placeholder_argv(command: list[str], clip: Clip) -> list[str]:
    """`command` with the tool's directory, the clip and the stream as placeholders."""
    head = f"{BIN}/{Path(command[0]).name}"
    return [head, *(INPUT if a == str(clip.yuv) else a for a in command[1:])]


def plan_entry(bench: Bench, point: Point) -> dict:
    """One timing plan entry: the quality encode's own command (`Bench.command`, without
    the reconstruction and logs only scoring needs), its rate and length."""
    command = bench.command(point, Path(OUTPUT), None)
    return {"argv": placeholder_argv(command, point.clip), "kbps": point.kbps,
            "input": point.clip.name,
            "frames": bench.frames, "intra_period": bench.intra, "layers": bench.layers,
            "fps": point.clip.fps}


def plan_digest(plan: dict) -> str:
    """The plan's content digest (its canonical JSON)."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def plan_record(plan: dict, plan_file: str, command: list[str]) -> dict:
    """The repository's record of a plan: its header, its file in PLAN_STORE and digest,
    the command that made it, and per point the encodes planned and one example command."""
    header = {k: v for k, v in plan.items() if k != "points"}
    return {**header, "plan_file": plan_file, "plan_sha256": plan_digest(plan),
            "command": command,
            "points": {name: {"encodes": len(entries),
                              "example": {"clip": next(iter(entries)),
                                          **next(iter(entries.values()))}}
                       for name, entries in plan["points"].items()}}


def load_plan(path: Path, store: Path | None = None) -> dict:
    """The full plan a record names (checked against its digest); a full plan file is
    returned as it is."""
    record = json.loads(Path(path).read_text())
    if "plan_file" not in record:
        return record
    full = (store or PLAN_STORE) / record["plan_file"]
    if not full.is_file():
        raise FileNotFoundError(f"{path}: its plan {full} is not here; regenerate it with "
                                f"{' '.join(record['command'])}")
    plan = json.loads(full.read_text())
    if plan_digest(plan) != record["plan_sha256"]:
        raise ValueError(f"{full}: content differs from the record {path}")
    return plan
