"""Check 3: coding-tool activity, ours against the reference encoder's stream.

A tool that silently stops working rarely shows in a mean BD-rate, but it is
obvious in the stream: CDEF active on 0% of blocks, no skipped blocks, no
sub-pel vectors. This module defines a codec-neutral activity summary per
frame type, compares ours with the reference encoder's stream at a nearby
rate, and flags large divergences and tools that are effectively never on.

Per frame type, averaged over frames:
  skip_share      share of area coded without residual (AV1 skip_txfm,
                  H.264 P_Skip)
  intra_share     share of area intra coded
  histograms      block_sizes, tx_sizes (coded area), subpel_phases (inter
                  vectors by finest phase: integer/half/quarter/eighth),
                  codec extras (AV1 cdef_y_primary)
  filters         in-loop filter activity (loop filter level, CDEF share and
                  strengths, SAO / deblocking enable shares)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from .flags import Flag, Severity

PHASE_NAMES = {1: "integer", 2: "half", 4: "quarter", 8: "eighth"}
# Total-variation distance between two histograms above which they differ
# "completely" for this check's purposes (0 = identical, 1 = disjoint).
HISTOGRAM_DIVERGENCE = 0.5
# A reference share above this means the tool is in real use there.
REFERENCE_IN_USE = 0.10


@dataclass
class Activity:
    frames: int
    frame_type: str
    skip_share: float
    intra_share: float
    histograms: dict[str, dict[str, float]] = field(default_factory=dict)
    filters: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> Activity:
        return cls(**data)


def subpel_phase_shares(rows: np.ndarray, cols: np.ndarray, denominator: int) -> dict[str, float]:
    """Share of vectors by finest phase of their two components.

    A component's phase p (value mod denominator) needs a filter of
    denominator / gcd(p, denominator) steps per sample: 1 = integer, 2 = half...
    """
    if rows.size == 0:
        return {}
    levels = np.maximum(_phase_level(rows, denominator), _phase_level(cols, denominator))
    keys, counts = np.unique(levels, return_counts=True)
    return {PHASE_NAMES.get(int(k), f"1/{int(k)}"): float(c / levels.size)
            for k, c in zip(keys, counts)}


def _phase_level(values: np.ndarray, denominator: int) -> np.ndarray:
    return denominator // np.gcd(np.mod(values.astype(np.int64), denominator), denominator)


def mean_activity(frames: list[Activity]) -> dict[str, Activity]:
    """Average per-frame activity per frame type; missing histogram keys count as 0."""
    by_type: dict[str, list[Activity]] = defaultdict(list)
    for frame in frames:
        by_type[frame.frame_type].append(frame)
    return {kind: _mean(group) for kind, group in by_type.items()}


def _mean(group: list[Activity]) -> Activity:
    n = len(group)

    def mean_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
        acc: dict[str, float] = defaultdict(float)
        for d in dicts:
            for k, v in d.items():
                acc[k] += v / n
        return dict(sorted(acc.items()))

    names = sorted({h for a in group for h in a.histograms})
    return Activity(
        frames=n, frame_type=group[0].frame_type,
        skip_share=sum(a.skip_share for a in group) / n,
        intra_share=sum(a.intra_share for a in group) / n,
        histograms={h: mean_dict([a.histograms.get(h, {}) for a in group]) for h in names},
        filters=mean_dict([a.filters for a in group]))


def histogram_divergence(a: dict[str, float], b: dict[str, float]) -> float:
    """Total-variation distance between two share histograms."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


@dataclass(frozen=True)
class ToolLimits:
    skip_share_ratio: float
    share_gap: float
    never_active_share: float
    keyframe_size_ratio: float = 2.0


def compare_activity(ours: Activity, reference: Activity | None, limits: ToolLimits,
                     subject: str, codec_filters: tuple[str, ...]) -> list[Flag]:
    """Divergence and never-active flags for one frame type of one stream pair."""
    flags = []

    def flag(severity: Severity, finding: str, evidence: dict, cause: str) -> None:
        flags.append(Flag("tools", severity, subject, finding, evidence, cause))

    if ours.skip_share < limits.never_active_share:
        flag(Severity.HIGH, f"skip effectively never used ({ours.skip_share:.1%} of area)",
             {"skip_share": ours.skip_share,
              "reference": reference.skip_share if reference else None},
             "skip decision disabled or its threshold scaled wrongly")
    fractional = 1.0 - ours.histograms.get("subpel_phases", {}).get("integer", 1.0)
    if ours.histograms.get("subpel_phases") and fractional < limits.never_active_share:
        flag(Severity.HIGH, f"sub-pel vectors effectively never used ({fractional:.1%})",
             {"subpel_phases": ours.histograms["subpel_phases"]},
             "sub-pel refinement skipped by a gate at this QP")
    for name in codec_filters:
        value = ours.filters.get(name)
        ref_value = reference.filters.get(name) if reference else None
        if value is not None and value < limits.never_active_share and \
                (ref_value is None or ref_value > REFERENCE_IN_USE):
            flag(Severity.HIGH, f"{name} effectively never active ({value:.1%})",
                 {name: value, "reference": ref_value},
                 "in-loop filter gated off (e.g. a tuned content gate not firing on held-out content)")
    if reference is None:
        return flags
    _compare_shares(ours, reference, limits, flag)
    for name in ("block_sizes", "tx_sizes", "subpel_phases"):
        a, b = ours.histograms.get(name), reference.histograms.get(name)
        if a and b and histogram_divergence(a, b) > HISTOGRAM_DIVERGENCE:
            flag(Severity.MEDIUM,
                 f"{name} histogram diverges from the reference "
                 f"(TV distance {histogram_divergence(a, b):.2f})",
                 {"ours": _top(a), "reference": _top(b)},
                 "partition / transform / MV precision policy far from the reference's")
    return flags


def _compare_shares(ours: Activity, reference: Activity, limits: ToolLimits, flag) -> None:
    for name in ("skip_share", "intra_share"):
        a, b = getattr(ours, name), getattr(reference, name)
        ratio = (max(a, b) / min(a, b)) if min(a, b) > 0 else float("inf")
        if abs(a - b) > limits.share_gap or (ratio > limits.skip_share_ratio
                                             and abs(a - b) > limits.never_active_share * 5):
            flag(Severity.MEDIUM, f"{name} {a:.1%} vs reference {b:.1%}",
                 {"ours": a, "reference": b},
                 "skip/intra decision thresholds far from the reference's on this content")


def _top(histogram: dict[str, float], n: int = 6) -> dict[str, float]:
    return {k: round(v, 3) for k, v in sorted(histogram.items(), key=lambda kv: -kv[1])[:n]}
