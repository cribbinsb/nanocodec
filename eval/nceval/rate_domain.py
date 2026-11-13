"""Pick scored quantizers inside the clip's source-bitrate ceiling.

Rate probes may bracket an out-of-range bitrate, but their quality is never
scored. Every chosen point is checked against its actual rate before use.
"""
from __future__ import annotations

from collections.abc import Callable

from .sweep import CAP_HEADROOM, fit_targets
MIN_POINTS = 4


def bounded_quantizers(rate_at: Callable[[int], float], bounds: tuple[int, int],
                       nominal: list[float], cap_kbps: float) -> list[int]:
    """Find four distinct legal QPs, pricing probes through a cached callback.

    Bitrate normally decreases with QP. Binary search finds the highest
    quality under each target; the actual chosen rate is checked again, so
    local nonmonotonicity cannot admit a point above the source ceiling.
    """
    lo, hi = bounds
    targets = fit_targets(nominal, rate_at(hi), cap_kbps * CAP_HEADROOM)
    if not targets:
        raise ValueError(f"no usable bitrate span below source ceiling {cap_kbps:g} kbps")
    chosen = set()
    for target in targets:
        left, right = lo, hi
        while left < right:
            middle = (left + right) // 2
            if rate_at(middle) <= target:
                right = middle
            else:
                left = middle + 1
        if rate_at(left) <= cap_kbps:
            chosen.add(left)
    if len(chosen) < MIN_POINTS:
        raise ValueError(f"only {len(chosen)} distinct legal quantizers below {cap_kbps:g} kbps")
    return sorted(chosen)
