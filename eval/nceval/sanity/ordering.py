"""Check 1: codec ordering per clip.

At matched quality a newer codec must not need more bits than an older one
from the same engine: bits(AV1) <= bits(HEVC) <= bits(H.264), within a small
tolerance. A violation means the newer backend is missing or misusing a tool
the older one has (e.g. AV1 losing to our own H.264 on a static clip was a
CDEF gate switching strong filtering off).

The comparison is a BD-style integral of log-rate over the overlapping
quality range, split into a low-rate and a high-rate half, so a loss
confined to one end of the curve is not averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .curves import CurveStore
from .flags import Flag, severity_by_excess

REGIONS = ("low", "high")  # halves of the shared quality range: low = low rate
MIN_POINTS = 3
# Excess bits (beyond the tolerance) that make a violation HIGH / CRITICAL.
HIGH_EXCESS = 0.10
CRITICAL_EXCESS = 0.35


@dataclass(frozen=True)
class RegionRatio:
    region: str
    quality_range: tuple[float, float]
    bits_ratio: float  # bits(test) / bits(reference) at matched quality


def region_ratios(test_kbps: list[float], test_quality: list[float],
                  ref_kbps: list[float], ref_quality: list[float]) -> list[RegionRatio]:
    """Bits ratio test/ref per half of the overlapping quality range.

    Returns [] when either curve is unusable (too few points, quality not
    strictly increasing with rate) or the curves do not overlap.
    """
    test, ref = _curve(test_kbps, test_quality), _curve(ref_kbps, ref_quality)
    if test is None or ref is None:
        return []
    lo, hi = max(test[0][0], ref[0][0]), min(test[0][-1], ref[0][-1])
    if hi <= lo:
        return []
    t_fit, r_fit = PchipInterpolator(*test), PchipInterpolator(*ref)
    edges = np.linspace(lo, hi, len(REGIONS) + 1)
    out = []
    for name, a, b in zip(REGIONS, edges[:-1], edges[1:]):
        mean_log_diff = (t_fit.integrate(a, b) - r_fit.integrate(a, b)) / (b - a)
        out.append(RegionRatio(name, (float(a), float(b)), float(10.0 ** mean_log_diff)))
    return out


def _curve(kbps: list[float], quality: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
    if len(kbps) < MIN_POINTS:
        return None
    q, r = np.asarray(quality, dtype=float), np.asarray(kbps, dtype=float)
    order = np.argsort(q)
    q, r = q[order], r[order]
    if np.any(np.diff(q) <= 0) or np.any(np.diff(r) <= 0):
        return None
    return q, np.log10(r)


def ordered_pairs(chain: tuple[str, ...]) -> list[tuple[str, str]]:
    """(newer, older) pairs a chain implies: every element must beat every later one."""
    return list(combinations(chain, 2))


def check_ordering(store: CurveStore, clips: list[str], chains: tuple[tuple[str, ...], ...],
                   metrics: tuple[str, ...], tolerance: float, label: str) -> list[Flag]:
    flags = []
    pairs = list(dict.fromkeys(p for chain in chains for p in ordered_pairs(chain)))
    for clip in clips:
        for newer, older in pairs:
            violations = []
            for metric in metrics:
                n_kbps, n_q = store.curve(newer, clip).series(metric)
                o_kbps, o_q = store.curve(older, clip).series(metric)
                for r in region_ratios(n_kbps, n_q, o_kbps, o_q):
                    if r.bits_ratio > 1.0 + tolerance:
                        violations.append({"metric": metric, "region": r.region,
                                           "bits_ratio": round(r.bits_ratio, 3),
                                           "quality_range": [round(x, 2) for x in r.quality_range]})
            if violations:
                worst = max(v["bits_ratio"] for v in violations) - 1.0 - tolerance
                flags.append(Flag(
                    check="ordering",
                    severity=severity_by_excess(worst, HIGH_EXCESS, CRITICAL_EXCESS),
                    subject=f"{label}/{clip}: {newer} vs {older}",
                    finding=(f"{newer} needs up to {100 * (worst + tolerance):.0f}% more bits "
                             f"than {older} at matched quality"),
                    evidence={"violations": violations, "tolerance": tolerance},
                    suspected_cause=ordering_cause(violations, newer)))
    return flags


def ordering_cause(violations: list[dict], newer: str) -> str:
    """A one-line hypothesis from the pattern of violations."""
    metrics = {v["metric"] for v in violations}
    regions = {v["region"] for v in violations}
    where = "at all rates" if regions == set(REGIONS) else f"at {regions.pop()} rates only"
    if metrics == {"vmaf_v1"}:
        what = "VMAF only: perceptual detail lost (over-filtering, too much skip/flat prediction)"
    elif metrics == {"psnr_yuv"}:
        what = "PSNR only: chroma or luma allocation tuned for VMAF, or a chroma tool off"
    else:
        what = "both metrics: a missing or broken coding tool / decision in " + newer
    return f"{what}; {where}. Compare tool activity and per-frame decay for this clip."
