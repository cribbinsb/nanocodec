"""Check 4: per-frame quality decay over P-frame chains.

Errors that accumulate from frame to frame (a loop filter switched off, drift
between encoder and decoder-side state, a rate model starving later frames)
show up as a steady decline of per-frame PSNR within a GOP. A sequence mean
hides it, and short GOPs reset it before it matters; this check fits a
straight line to the P frames of every GOP and flags a sustained decline
that the reference encoder, at the nearest rate, does not show. Long-GOP
sets (one key frame per clip) make the chains long enough to see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import CurveStore, FrameRow
from .flags import Flag, Severity

MIN_P_FRAMES = 30  # shorter chains give no meaningful slope
# Excess decline over the whole GOP (dB, versus the reference) for HIGH / CRITICAL.
HIGH_DROP_DB = 1.0
CRITICAL_DROP_DB = 3.0
# Second-half slope, as a fraction of the limit, that still counts as declining.
SUSTAINED_FRACTION = 0.5
PSNR_IDENTICAL_DB = 99.0  # capped PSNR of identical frames: not a quality sample


@dataclass(frozen=True)
class GopTrend:
    start: int            # index of the GOP's key frame
    p_frames: int
    slope: float          # metric units per frame, least squares over P frames
    first_half: float
    second_half: float
    first_mean: float
    last_mean: float      # mean of the last tenth of the P frames

    def sustained(self, limit: float) -> bool:
        """Still declining at a material rate in the second half of the chain.

        Excludes one dip, a scene change, and a key frame coded much better than
        the P frames whose quality then settles to the steady state (that is a
        key-frame quantizer offset, not accumulating error).
        """
        return self.first_half < 0 and self.second_half < limit * SUSTAINED_FRACTION


def gop_trends(frame_types: list[str], values: list[float | None]) -> list[GopTrend]:
    """Trend of a per-frame metric over the P frames of each GOP."""
    trends, start = [], 0
    starts = [i for i, t in enumerate(frame_types) if t == "I"] or [0]
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(frame_types)
        series = [(i, v) for i, v in enumerate(values[start + 1:end], start + 1)
                  if v is not None and v < PSNR_IDENTICAL_DB]
        if len(series) < MIN_P_FRAMES:
            continue
        x = np.array([i for i, _ in series], dtype=float)
        y = np.array([v for _, v in series], dtype=float)
        half = len(series) // 2
        tenth = max(1, len(series) // 10)
        trends.append(GopTrend(start=start, p_frames=len(series), slope=_slope(x, y),
                               first_half=_slope(x[:half], y[:half]),
                               second_half=_slope(x[half:], y[half:]),
                               first_mean=float(y[:tenth].mean()),
                               last_mean=float(y[-tenth:].mean())))
    return trends


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0


def decay_finding(test: GopTrend, reference: GopTrend | None, limit: float,
                  margin: float) -> float | None:
    """Excess decline in dB over the GOP if this is a decay flag, else None."""
    if test.slope >= limit or not test.sustained(limit):
        return None
    ref_slope = reference.slope if reference is not None else 0.0
    if test.slope >= ref_slope - margin:
        return None
    return (ref_slope - test.slope) * test.p_frames


def check_decay(store: CurveStore, clips: list[str], references: dict[str, str],
                encoders: tuple[str, ...], limit: float, margin: float,
                label: str) -> tuple[list[Flag], dict]:
    flags, table = [], {}
    for test, reference in references.items():
        if test not in encoders:
            continue
        for clip in clips:
            ref_curve = store.curve(reference, clip) if reference in encoders else None
            for point in store.curve(test, clip).points:
                rows = store.frames(point)
                ref_point = ref_curve.nearest(point.kbps) if ref_curve else None
                ref_rows = store.frames(ref_point) if ref_point else []
                worst = _worst_gop(rows, ref_rows, limit, margin)
                key = f"{test}/{clip}/q{point.quantizer}"
                table[key] = worst["summary"]
                if worst["excess_db"] is None:
                    continue
                excess = worst["excess_db"]
                flags.append(Flag(
                    check="decay",
                    severity=(Severity.CRITICAL if excess >= CRITICAL_DROP_DB else
                              Severity.HIGH if excess >= HIGH_DROP_DB else Severity.MEDIUM),
                    subject=f"{label}/{clip}: {test} q{point.quantizer} ({point.kbps:.0f} kbps)",
                    finding=(f"Y-PSNR declines {worst['summary']['slope']:+.3f} dB/frame over "
                             f"{worst['summary']['p_frames']} P frames; {reference} "
                             f"{_ref_text(worst['summary'], ref_point is not None)}"),
                    evidence={**worst["summary"], "reference": reference,
                              "reference_quantizer": ref_point.quantizer if ref_point else None,
                              "reference_kbps": round(ref_point.kbps, 1) if ref_point else None},
                    suspected_cause=("error accumulating along the P chain: a loop filter or "
                                     "CDEF gate off, reference/recon drift, or rate starving "
                                     "later frames")))
    return flags, table


def _worst_gop(rows: list[FrameRow], ref_rows: list[FrameRow], limit: float,
               margin: float) -> dict:
    tests = gop_trends([r.frame_type for r in rows], [r.psnr_y for r in rows])
    refs = gop_trends([r.frame_type for r in ref_rows], [r.psnr_y for r in ref_rows])
    vmaf = gop_trends([r.frame_type for r in rows], [r.vmaf_v1 for r in rows])
    best: dict = {"excess_db": None, "summary": {}}
    for i, trend in enumerate(tests):
        ref = next((r for r in refs if r.start == trend.start), None)
        excess = decay_finding(trend, ref, limit, margin)
        summary = {"gop_start": trend.start, "p_frames": trend.p_frames,
                   "slope": round(trend.slope, 4),
                   "halves": [round(trend.first_half, 4), round(trend.second_half, 4)],
                   "first_tenth_db": round(trend.first_mean, 2),
                   "last_tenth_db": round(trend.last_mean, 2),
                   "reference_slope": round(ref.slope, 4) if ref else None,
                   "reference_first_last_db": ([round(ref.first_mean, 2), round(ref.last_mean, 2)]
                                               if ref else None),
                   "vmaf_slope": round(vmaf[i].slope, 4) if i < len(vmaf) else None}
        if not best["summary"] or (excess or 0) > (best["excess_db"] or 0):
            best = {"excess_db": excess, "summary": summary}
    return best


def _ref_text(summary: dict, has_point: bool) -> str:
    if summary.get("reference_slope") is None:
        return "has no per-frame data at the nearest rate" if has_point else \
            "has no comparable curve"
    return f"{summary['reference_slope']:+.3f} dB/frame at the nearest rate"
