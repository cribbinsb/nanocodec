"""Per-clip statistics for the pool levels: J_real's mean and a bootstrap interval.

A small draw can show a genuine gain as a loss (or the reverse) by chance, so
a level is not decided on its point estimate. From the per-clip J deltas
(J = (quality + quality_low) / 2 + k * dCPU, nceval.jreal) it computes the
class-balanced mean, as J_real does, and a CONFIDENCE interval by a
bootstrap over clips, resampled within each draw stratum (nceval.pool).

Decision (owner, 2026-11-11; judged as a guideline):
  - L1 / L2 stop a candidate only when it is confidently worse: the whole
    interval above +STOP_LOWER_PCT. Anything else goes on to the next level.
  - L3 adopts when the interval favours the candidate: mean < 0 and the
    upper bound at most +ADOPT_UPPER_PCT; it rejects (WORSE) when the
    whole interval is above 0; otherwise AMBIGUOUS (not adopted).
Every level reports mean, interval and n, so "ambiguous at n=12" never
reads as a false WORSE.
"""

from __future__ import annotations

import hashlib
import random

from .pool import STRATA

CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 2000
#: L1/L2 stop: the interval's lower bound above this (percent J).
STOP_LOWER_PCT = 0.5
#: L3 adopt: mean below -ADOPT_MIN_GAIN_PCT (identical builds measure
#: -0.00%, rounding) and the upper bound at most ADOPT_UPPER_PCT (percent J).
ADOPT_UPPER_PCT = 0.3
ADOPT_MIN_GAIN_PCT = 0.05
WORSE, BETTER, AMBIGUOUS, CONTINUE = "worse", "better", "ambiguous", "continue"


def clip_j(row: dict, dcpu_pct: float | None, k_cpu: float) -> float | None:
    """One clip's J delta (percent), or None if a term is missing."""
    full, low = row.get("quality"), row.get("quality_low")
    if full is None or low is None:
        return None
    return (full + low) / 2 + k_cpu * (dcpu_pct or 0.0)


def balanced_mean(values: list[tuple[str, float]]) -> float:
    """Mean over classes of each class's mean (J_real's class balance)."""
    classes: dict[str, list[float]] = {}
    for name, value in values:
        classes.setdefault(name, []).append(value)
    return sum(sum(v) / len(v) for v in classes.values()) / len(classes)


def interval(per_clip: dict[str, dict], dcpu_pct: float | None, k_cpu: float) -> dict:
    """{"mean", "lo", "hi", "n", "confidence"} of the class-balanced J delta."""
    scored = sorted((name, row["class"], clip_j(row, dcpu_pct, k_cpu))
                    for name, row in per_clip.items())
    scored = [(name, cls, j) for name, cls, j in scored if j is not None]
    result = {"n": len(scored), "confidence": CONFIDENCE, "mean": None, "lo": None, "hi": None}
    if not scored:
        return result
    groups = resampling_groups([(cls, j) for _, cls, j in scored])
    seed = hashlib.sha256(",".join(name for name, _, _ in scored).encode()).digest()
    rng = random.Random(int.from_bytes(seed[:8], "big"))
    means = sorted(
        balanced_mean([rng.choice(members) for members in groups
                       for _ in range(len(members))])
        for _ in range(BOOTSTRAP_DRAWS))
    tail = (1 - CONFIDENCE) / 2
    result.update(mean=balanced_mean([(cls, j) for _, cls, j in scored]),
                  lo=means[int(tail * (BOOTSTRAP_DRAWS - 1))],
                  hi=means[int((1 - tail) * (BOOTSTRAP_DRAWS - 1))])
    return result


def resampling_groups(values: list[tuple[str, float]]) -> list[list[tuple[str, float]]]:
    """(class, J) pairs grouped by draw stratum for the stratified bootstrap.

    A stratum of one clip would add no variance at all (a zero-width
    interval from one clip per class), so singleton strata are merged into
    one mixed group, and a lone leftover clip joins the largest group.
    """
    strata: dict[str, list[tuple[str, float]]] = {}
    for cls, j in values:
        strata.setdefault(_stratum(cls), []).append((cls, j))
    groups = [members for members in strata.values() if len(members) > 1]
    single = [members[0] for members in strata.values() if len(members) == 1]
    if len(single) > 1 or (single and not groups):
        groups.append(single)
    elif single:
        max(groups, key=len).extend(single)
    return groups


def decide(stats: dict, final: bool) -> str:
    """WORSE (stop), and at L3 BETTER or AMBIGUOUS; before L3 otherwise CONTINUE."""
    if stats["mean"] is None:
        return AMBIGUOUS if final else CONTINUE
    if stats["lo"] > STOP_LOWER_PCT:
        return WORSE
    if not final:
        return CONTINUE
    if stats["mean"] < -ADOPT_MIN_GAIN_PCT and stats["hi"] <= ADOPT_UPPER_PCT:
        return BETTER
    # At L3 a loss the interval is sure of is a loss, however small.
    return WORSE if stats["lo"] > 0 else AMBIGUOUS


def describe(stats: dict) -> str:
    if stats["mean"] is None:
        return f"J n/a (n={stats['n']})"
    return (f"J {stats['mean']:+.2f}% [{stats['lo']:+.2f}, {stats['hi']:+.2f}] "
            f"{stats['confidence']:.0%} CI, n={stats['n']}")


def _stratum(content_class: str) -> str:
    return next((name for name, classes in STRATA.items() if content_class in classes),
                content_class)
