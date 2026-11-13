"""Single-point J_real: one ladder rung per clip, priced on the anchor's full curve.

At equal points, one rung on more clips predicts a disjoint clip set's
full-ladder J better than the full ladder on fewer clips (102 recorded
screens: r 0.38 / 0.43 vs 0.27 / 0.32 at 8 / 12 points; docs/handoff/
tuning.md). So the small-change measurements take one point per clip: the
tuner's trials (nceval.tune_pool), L1 screens and the single-point L3
(nceval.levels). Each clip's rung is fixed by its position in the set,
cycled low to high over the clip's legal ladder; the candidate's point is
turned into an equivalent rate delta on the anchor's full curve per metric
and averaged as J_real does (jreal.point_quality). The anchor's curves are
cached, so a run measures one candidate point per clip.

One point per clip cannot see a change of curve shape (better at low rate,
worse at high). `reshaping` flags a set whose per-clip deltas trend with
the rung (more than RESHAPE_PCT over the ladder at |t| > RESHAPE_T); the
callers then measure the full ladder instead.

A two-layer setting (spatial SVC) is priced the same way on its full-size
output at the total rate; its rows also carry `base_quality`, the base
layer's single-point J against the anchor's base curve (jreal.point_base_quality),
which `base_constraint` turns into the SVC base-layer guard.
"""

from __future__ import annotations

import math

import numpy as np

from . import jreal, realuse
from .cbr import Family, clip_targets
from .clips import Clip

FRAMES = realuse.DEFAULT_FRAMES
LADDER_RUNGS = len(realuse.LADDER_KBPS)
#: Curve-reshaping guard: J change over the ladder (percent) and its t-value.
RESHAPE_PCT = 1.0
RESHAPE_T = 2.5


def assign_rungs(clips: list[Clip], offset: int = 0) -> dict[str, int]:
    """One ladder rung per clip, cycled low to high over the set (strata interleave)."""
    return {c.name: (i + offset) % len(clip_targets(c, realuse.LADDER_KBPS))
            for i, c in enumerate(clips)}


def single_points(runner: realuse.Runner, anchor: Family, test: Family, clips: list[Clip],
                  rungs: dict[str, int], frames: int = FRAMES,
                  budgeted: bool = False, key_seconds: float | None = None) -> dict[str, dict]:
    """Per clip {"class", "rung", "quality" (= "quality_low", the clip's J for
    nceval.interval), "noncompliant", "base_quality" (two-layer settings; else
    None)}: the test's one rung against the anchor's
    full (cached) curve, plus both arms' ceiling terms at that rung
(jreal.ceiling_terms). A clip without a measured point (other than one our
    encoder refused) raises: the set is incomplete, so no estimate is made.
    `budgeted` splits a long run into parts (Runner.measure); `key_seconds` sets a
    key period (default one key)."""
    curves = runner.measure(f"{anchor.name} curves", [anchor], clips, frames, None,
                            key_seconds, budgeted=budgeted)
    target = lambda c: [clip_targets(c, realuse.LADDER_KBPS)[rungs[c.name]]]
    tests = runner.measure(f"{test.name} single points", [test], clips, frames, None,
                           key_seconds, budgeted=budgeted, ladder=target)
    refused = {entry.split("@")[0] for entry in runner.refused.get(test.name, ())}
    rows, missing = {}, []
    for clip in clips:
        points = tests[(test.name, clip.name)]
        if clip.name in refused:
            continue  # our encoder refused the clip: a property of the setting, listed
        quality = jreal.point_quality(curves[(anchor.name, clip.name)], points[0]) \
            if points else None
        if quality is None:
            missing.append(clip.name)
            continue
        anchor_curve = curves[(anchor.name, clip.name)]
        same_rung = [p for p in anchor_curve if p["target_kbps"] == points[0]["target_kbps"]]
        rows[clip.name] = {"class": clip.content_class, "rung": rungs[clip.name],
                           "quality": quality, "quality_low": quality,
                           "noncompliant": int(jreal.noncompliant(points[0])),
                           "base_quality": jreal.point_base_quality(anchor_curve, points[0]),
                           **jreal.ceiling_terms("test", points[:1]),
                           **jreal.ceiling_terms("anchor", same_rung)}
    if missing:
        # A point that silently failed (e.g. the GPU out of memory) is never data.
        raise ValueError(f"{test.name}: no point on {', '.join(missing)}")
    return rows


def balanced_j(rows: dict[str, dict], term: str = "quality") -> float | None:
    """The class-balanced mean of `term` over the rows that have it."""
    classes: dict[str, list[float]] = {}
    for row in rows.values():
        if row.get(term) is not None:
            classes.setdefault(row["class"], []).append(row[term])
    return sum(sum(v) / len(v) for v in classes.values()) / len(classes) if classes else None


def ceiling_constraint(rows: dict[str, dict]) -> list[float]:
    """The CBR ceiling guard (<= 0 when met; jreal.ceiling_excess) of single-point rows."""
    excess = jreal.ceiling_excess(jreal.compliance(rows))
    return [] if excess is None else [excess]


def base_constraint(rows: dict[str, dict]) -> list[float]:
    """The SVC base-layer guard (<= 0 when met): the class-balanced base-layer J
    against jreal.SVC_BASE_GUARD_PCT; no constraint for a single-layer setting."""
    base = balanced_j(rows, "base_quality")
    return [] if base is None else [base - jreal.SVC_BASE_GUARD_PCT]


def rung_trend(rows: dict[str, dict]) -> tuple[float, float]:
    """(J change from the lowest to the highest rung, its t-value) by least squares."""
    pairs = [(r["rung"], r["quality"]) for r in rows.values() if r["quality"] is not None]
    x = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)
    if len(pairs) < 4 or np.ptp(x) == 0:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    error = math.sqrt(residual @ residual / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean())))
    return float(slope * (LADDER_RUNGS - 1)), float(slope / error) if error > 0 else math.inf


def reshaping(rows: dict[str, dict]) -> bool:
    """Whether the per-clip deltas trend with the rung (see the module docstring)."""
    change, t_value = rung_trend(rows)
    return abs(change) > RESHAPE_PCT and abs(t_value) > RESHAPE_T
