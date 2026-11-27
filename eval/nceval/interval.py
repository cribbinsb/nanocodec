"""Per-clip statistics for the pool levels: J_real's mean and a bootstrap interval.

A small draw can show a genuine gain as a loss (or the reverse) by chance, so
a level is not decided on its point estimate. From the per-clip J deltas
(J = (quality + quality_low) / 2 + k * dCPU, nceval.jreal) it computes the
class-balanced mean, as J_real does, and a CONFIDENCE interval by a
bootstrap over clips, resampled within each draw stratum (nceval.pool).

Decision (owner, 2026-11-11; judged as a guideline):
  - L1 / L2 stop a candidate only when it is confidently worse: the whole
    interval above +STOP_LOWER_PCT. Anything else goes on to the next level.
  - L3 adopts when the whole interval is below 0 (and the mean below
    -ADOPT_MIN_GAIN_PCT); it rejects (WORSE) when the whole interval is above
    0; otherwise AMBIGUOUS (not adopted): an interval that crosses 0 is no
    evidence of a gain (coordinator, 2026-11-23; it allowed an upper bound up
    to +0.3 % before).
  - A level whose candidate and anchor code identical streams (every clip's J
    delta 0, so the whole interval within IDENTICAL_PCT of 0) is NEUTRAL at
    L3: it neither adopts nor rejects, and the decision falls to the other
    measurement (`nceval.train`: a candidate that only changes periodic keys
    leaves the 150-frame one-key L3 identical, coordinator 2026-11-22).
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
#: -0.00%, rounding) and the whole interval below 0 (percent J).
ADOPT_MIN_GAIN_PCT = 0.05
#: An interval this close to 0 (percent J) on both sides is identical streams:
#: the J deltas of identical points are 0 up to summation rounding (about 1e-14).
IDENTICAL_PCT = 1e-9
WORSE, BETTER, AMBIGUOUS, CONTINUE = "worse", "better", "ambiguous", "continue"
NEUTRAL = "neutral"


def clip_j(row: dict, dcpu_pct: float | None, k_cpu: float,
           score: str = "decision") -> float | None:
    """One clip's J delta (percent), or None if a term is missing.

    The decision score (2026-11-24, dev matches the benchmark) is the benchmark's BD-J
    (`bd_j`, nceval.bdj) where the row has a curve; a single-point row (L1, L3: short
    screens code one rung per clip, no curve) keeps its point quality. `score="j_real"`
    gives the earlier J_real, (full + lower-half BD) / 2, printed beside it."""
    if score == "decision" and "bd_j" in row:
        quality = row["bd_j"]
    else:
        full, low = row.get("quality"), row.get("quality_low")
        quality = None if full is None or low is None else (full + low) / 2
    return None if quality is None else quality + k_cpu * (dcpu_pct or 0.0)


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
    # the earlier J_real beside the decision, so a disagreement between the two shows
    previous = [(row["class"], j) for row in per_clip.values()
                if (j := clip_j(row, dcpu_pct, k_cpu, "j_real")) is not None]
    result["j_real_mean"] = balanced_mean(previous) if previous else None
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


def identical(stats: dict) -> bool:
    """Whether a level's interval is identical streams' (within IDENTICAL_PCT of 0)."""
    return all(stats.get(key) is not None and abs(stats[key]) <= IDENTICAL_PCT
               for key in ("mean", "lo", "hi"))


def decide(stats: dict, final: bool) -> str:
    """WORSE (stop), and at L3 BETTER, NEUTRAL (identical streams) or AMBIGUOUS;
    before L3 otherwise CONTINUE."""
    if stats["mean"] is None:
        return AMBIGUOUS if final else CONTINUE
    if identical(stats):
        return NEUTRAL if final else CONTINUE
    if stats["lo"] > STOP_LOWER_PCT:
        return WORSE
    if not final:
        return CONTINUE
    if stats["mean"] < -ADOPT_MIN_GAIN_PCT and stats["hi"] < 0:
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
