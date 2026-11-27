"""BD-J, the benchmark's decision score (BP-3 section 9), one definition for the
benchmark's report (tools/bench/report.py) and the dev screens (nceval.jreal, nceval.interval).

BD-J of a test curve against an anchor curve on one clip is the mean, over PSNR-YUV, VMAF v1
and MS-SSIM (in dB), of the BD-rate over the quality range the two curves share. Each metric's
curve is the points that carry that metric; it first loses the points another point of it
dominates (the BP-3.9 envelope: lower or equal rate, higher or equal quality), then the points
whose quality does not rise (a fall beyond MONOTONE_TOLERANCE refuses the curve), and must keep
MIN_POINTS points.

BP-3.14 (owner, 2026-11-25): the aim is a BD for every encoder on every clip against the one
baseline. So a curve needs 2 points, and no overlap is too thin to count: only two ranges
that do not touch give no BD (DisjointRanges). The overlap fraction is returned beside each
BD, for the callers to report (THIN_OVERLAP_FRACTION) rather than exclude.

A curve is a list of (rate kbps, {metric: raw value or None}) pairs; each caller maps its own
point records onto that (the report: coded rate and pooled scores; the screens: achieved rate
and summarised scores).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .bdrate import BdResult, bd_rate

METRICS = ("psnr_yuv", "vmaf_v1", "ms_ssim")
#: Section 9: the points a curve needs after the envelope. Two define the interpolation (PCHIP
#: in quality space is the straight line through them), so a clip whose ladder has fewer than
#: four rungs under the 50 %-of-source cap has a BD too (owner, 2026-11-25, BP-3.14; 4 before).
MIN_POINTS = 2
#: Section 9: an overlap under this fraction of the smaller curve's quality span is counted as
#: thin in the reports. It excluded the clip up to BP-3.13; since BP-3.14 (owner, 2026-11-25)
#: it only labels, and the BD is integrated over whatever range the curves share.
THIN_OVERLAP_FRACTION = 0.5
#: A metric's raw value may fall by up to this between rising rates (a quantizer ceiling's
#: noise); within it the point is dropped, beyond it the curve is refused.
MONOTONE_TOLERANCE = {"psnr_yuv": 0.05, "vmaf_v1": 0.1, "ms_ssim": 0.0005}

Curve = Sequence[tuple[float, dict[str, float | None]]]


class NoBd(ValueError):
    """There is no BD on this clip; the message says exactly why."""


class CurveRefusal(NoBd):
    """A curve that breaks the rules (too few rising points, a quality fall)."""


class DisjointRanges(NoBd):
    """The two curves' quality ranges do not touch: nothing to integrate over."""


def ms_ssim_db(value: float) -> float:
    """MS-SSIM linearised for BD-rate interpolation (it saturates near 1)."""
    return -10.0 * math.log10(max(1e-10, 1.0 - value))


QUALITY: dict[str, Callable[[float], float]] = {"ms_ssim": ms_ssim_db}


def has_metric(curve: Curve, metric: str) -> list[int]:
    """Indices of the points that carry `metric`. A point whose pooled value is missing (None:
    the scorer gave no figure for it) is not a point of that metric's curve; it stays a point
    of the others'. The report counts them."""
    return [i for i, (_, values) in enumerate(curve) if values.get(metric) is not None]


def envelope(curve: Curve, metric: str) -> list[int]:
    """Indices of the points carrying `metric` that no other such point dominates (BP-3.9)."""
    scored = has_metric(curve, metric)
    kept = []
    for i in scored:
        rate, value = curve[i][0], curve[i][1][metric]
        dominated = any(j != i and curve[j][0] <= rate and curve[j][1][metric] >= value
                        and (curve[j][0], curve[j][1][metric]) != (rate, value)
                        for j in scored)
        if not dominated:
            kept.append(i)
    return kept


def rd_curve(curve: Curve, metric: str, use_envelope: bool = True,
             label: str = "curve") -> tuple[list[float], list[float]]:
    """Rates and qualities (QUALITY's scale) by rising rate, the rules applied.

    With the envelope no curve is refused for a fall: a point at a higher rate with a lower or
    equal quality is dominated, so what the envelope keeps rises strictly. The fall refusal
    acts only on the view without it (`use_envelope=False`, the report's --no-envelope)."""
    points = sorted((curve[i] for i in (envelope(curve, metric) if use_envelope
                                         else has_metric(curve, metric))), key=lambda p: p[0])
    rates, quality, raw = [], [], []
    for rate, values in points:
        value = values[metric]
        if raw and value <= raw[-1]:
            if raw[-1] - value > MONOTONE_TOLERANCE[metric]:
                raise CurveRefusal(f"{label} {metric} falls {raw[-1] - value:.4f} as its rate "
                                   f"rises")
            continue
        rates.append(rate)
        quality.append(QUALITY.get(metric, lambda x: x)(value))
        raw.append(value)
    if len(rates) < MIN_POINTS:
        without = len(curve) - len(has_metric(curve, metric))
        raise CurveRefusal(
            f"{label} {metric}: {len(rates)} rising point{'' if len(rates) == 1 else 's'} of "
            f"{len(curve)} coded" + (f" ({without} without the metric)" if without else ""))
    return rates, quality


def bd(test: Curve, anchor: Curve, metric: str, use_envelope: bool = True) -> BdResult:
    """BD-rate (%) of `test` against `anchor` on `metric` over the quality range they share,
    with that range's fraction of the smaller curve's span. Raises CurveRefusal for a curve
    that breaks the rules and DisjointRanges when the two ranges do not touch."""
    ar, aq = rd_curve(anchor, metric, use_envelope, "anchor")
    tr, tq = rd_curve(test, metric, use_envelope, "test")
    if tq[0] >= aq[-1]:
        raise DisjointRanges(f"{metric}: ranges do not touch, the test curve lies above the "
                             f"anchor's (its lowest {tq[0]:.3f} against the anchor's highest "
                             f"{aq[-1]:.3f})")
    if tq[-1] <= aq[0]:
        raise DisjointRanges(f"{metric}: ranges do not touch, the test curve lies below the "
                             f"anchor's (its highest {tq[-1]:.3f} against the anchor's lowest "
                             f"{aq[0]:.3f})")
    return bd_rate(ar, aq, tr, tq, min_points=MIN_POINTS)


def bd_j(test: Curve, anchor: Curve, use_envelope: bool = True) -> float | None:
    """BD-J (%): the mean of the three metrics' BD-rates; None when any has no BD (a refused
    curve, or ranges that do not touch)."""
    try:
        values = [bd(test, anchor, metric, use_envelope).bd_rate_percent for metric in METRICS]
    except NoBd:
        return None
    return sum(values) / len(values)
