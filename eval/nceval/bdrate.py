"""Bjøntegaard delta rate (BD-rate).

Method, following current JVET/AOM practice and the pitfalls catalogued in
Herglotz et al., "The Bjøntegaard Bible" (IEEE TIP 2024):

- Interpolate log10(bitrate) as a function of quality with a monotone
  piecewise-cubic (PCHIP) curve through every RD point. The classic cubic
  polynomial fit over 4 points overshoots badly with 6+ points or
  irregular spacing.
- Integrate both curves over the *overlapping* quality range only, and report
  how much of each curve that overlap covers. A BD-rate computed from a small
  overlap is noise, and is flagged instead of trusted.
- For saturating metrics (VMAF, MS-SSIM) callers can transform quality first
  (e.g. -log10(1 - ssim)) so the interpolation sees a near-linear curve.

- `low_half=True` restricts the integral to the lower-rate half of the
  overlap (J_real's low-rate term, docs/PROCEDURES.md).

Result: average % bitrate change of `test` vs `anchor` at equal quality.
Negative means the test codec needs fewer bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

# Below this fraction of either curve's quality span the result is unreliable.
MIN_OVERLAP_FRACTION = 0.5


@dataclass(frozen=True)
class BdResult:
    bd_rate_percent: float | None
    overlap_fraction: float  # overlap / min(anchor span, test span)
    reliable: bool


# Reported BD-rates use four or more points. Staged screens use three outer
# points in their first stage; the error that adds was measured (nceval.screen).
MIN_POINTS = 4


def bd_rate(anchor_kbps: Sequence[float], anchor_quality: Sequence[float],
            test_kbps: Sequence[float], test_quality: Sequence[float],
            min_points: int = MIN_POINTS, low_half: bool = False) -> BdResult:
    """BD-rate over the curves' overlapping quality range.

    `low_half` integrates only the lower-rate half of that overlap (the
    anchor's log-rate midpoint; see `_low_half_ceiling`): the low-bitrate
    region the encoder is built for. It uses every point for the curve
    shape, so it needs no extra encodes. Reliability is the full overlap's.
    """
    a_q, a_lr = _sorted_curve(anchor_kbps, anchor_quality, min_points)
    t_q, t_lr = _sorted_curve(test_kbps, test_quality, min_points)

    lo = max(a_q[0], t_q[0])
    hi = min(a_q[-1], t_q[-1])
    smaller_span = min(a_q[-1] - a_q[0], t_q[-1] - t_q[0])
    if hi <= lo or smaller_span <= 0:
        return BdResult(bd_rate_percent=None, overlap_fraction=0.0, reliable=False)
    overlap = (hi - lo) / smaller_span

    anchor = PchipInterpolator(a_q, a_lr)
    test = PchipInterpolator(t_q, t_lr)
    if low_half:
        hi = _low_half_ceiling(anchor, lo, hi)
    mean_log_diff = (test.integrate(lo, hi) - anchor.integrate(lo, hi)) / (hi - lo)
    percent = (10.0 ** mean_log_diff - 1.0) * 100.0
    return BdResult(bd_rate_percent=float(percent), overlap_fraction=float(overlap),
                    reliable=overlap >= MIN_OVERLAP_FRACTION)


# Samples used to invert the anchor's quality -> log-rate curve.
_LOW_HALF_SAMPLES = 1024


def _low_half_ceiling(anchor: PchipInterpolator, lo: float, hi: float) -> float:
    """The quality at which the anchor's log rate is halfway across the overlap.

    The low-rate half is defined in log rate on the anchor curve, so both
    halves span the same rate ratio. Inverted by dense sampling: the anchor's
    PCHIP curve is monotone wherever its points are.
    """
    q = np.linspace(lo, hi, _LOW_HALF_SAMPLES)
    log_rate = anchor(q)
    middle = 0.5 * (log_rate[0] + log_rate[-1])
    above = np.nonzero(log_rate >= middle)[0]
    ceiling = float(q[above[0]]) if len(above) else hi
    return max(ceiling, q[1])  # never an empty interval


def _sorted_curve(kbps: Sequence[float], quality: Sequence[float],
                  min_points: int = MIN_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Return (quality, log10 rate) sorted by quality, requiring a usable curve."""
    if len(kbps) != len(quality):
        raise ValueError("rate and quality lists differ in length")
    if len(kbps) < min_points:
        raise ValueError(f"BD-rate needs at least {min_points} RD points per curve")
    q = np.asarray(quality, dtype=float)
    r = np.asarray(kbps, dtype=float)
    if np.any(r <= 0):
        raise ValueError("bitrates must be positive")
    order = np.argsort(q)
    q, r = q[order], r[order]
    if np.any(np.diff(q) <= 0):
        # Equal quality at two rates means the curve is not a function of quality;
        # usually a sign the QP sweep is too dense or saturated.
        raise ValueError("quality values must be strictly increasing across RD points")
    return q, np.log10(r)
