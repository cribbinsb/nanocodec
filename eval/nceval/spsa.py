"""SPSA steps and a sequential probability ratio test: the statistics of nceval.tune_spsa.

Why (docs/handoff/stop-network.md, 2026-11-22): a study that picks its best of 20-40 trials
on one fixed clip set reports the noise of that set (a trial's J has a standard error of
about 0.2 % on 16-23 long clips, and the best trial sits 0.3-0.5 % below its true value),
which is the whole size of most levers. Fishtest's answer, used here: steer the parameters
with SPSA (simultaneous perturbation stochastic approximation; Spall 1992, 1998), every
iteration on a FRESH draw, so the noise averages out over iterations instead of being
selected; and adopt only on a sequential test of the final point against the shipped rung,
on clips the steering never used.

SPSA. The parameters live in normalised units u in [-1, 1] (a domain's low to high). At
iteration k (from 0), with Spall's gains a_k = a / (k + 1 + A)^alpha and
c_k = c / (k + 1)^gamma and a Rademacher direction delta (each entry +-1), the objective f
(lower is better) is measured at u + c_k delta and u - c_k delta on the SAME clips, and
u <- clip(u - a_k (f+ - f-) / (2 c_k delta)). Measuring both sides on the same clips
cancels each clip's level from the difference (common random numbers).

SPRT. Per sample (a clip, key share and start), d = the candidate's score against the shipped
rung's (J + k dCPU, percent). H0: mean d = 0 (no gain); H1: mean d = -delta (a gain of delta).
With the samples' variance s^2 (floored), the log-likelihood ratio of H1 over H0 under a
normal model is LLR = (-delta / s^2) * sum(d_i + delta / 2), and the test stops at
LLR >= log((1 - beta) / alpha) (adopt) or LLR <= log(beta / (1 - alpha)) (no gain). The
variance is re-estimated from all samples at every look (the usual GSPRT plug-in).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

#: Spall's recommended gain exponents (1998, "Implementation of the simultaneous
#: perturbation algorithm"): the slowest decay that still converges.
ALPHA = 0.602
GAMMA = 0.101
#: The step size's stability constant A as a share of the planned iterations (Spall:
#: about 10 %).
STABILITY_SHARE = 0.1
#: Smallest per-sample standard deviation the SPRT divides by (percent of J): a handful of
#: samples underestimates the spread, and a 0.3 % effect is not 30 sigma.
SD_FLOOR = 0.3
#: Samples before the SPRT may decide (the variance estimate needs a few).
MIN_SAMPLES = 8


@dataclass(frozen=True)
class Gains:
    """Spall's gain sequences: a_k = a / (k + 1 + A)^ALPHA, c_k = c / (k + 1)^GAMMA."""
    a: float
    c: float
    stability: float

    @classmethod
    def plan(cls, iterations: int, c: float, first_step: float,
             first_gradient: float) -> "Gains":
        """Gains for `iterations` iterations: perturbation `c` (normalised units), and `a`
        such that the first step moves `first_step` for a gradient of `first_gradient`
        (objective units per normalised unit): Spall's rule of thumb."""
        stability = STABILITY_SHARE * iterations
        a = first_step * (1 + stability) ** ALPHA / max(first_gradient, 1e-9)
        return cls(a, c, stability)

    def step(self, k: int) -> float:
        return self.a / (k + 1 + self.stability) ** ALPHA

    def perturbation(self, k: int) -> float:
        return self.c / (k + 1) ** GAMMA


def direction(rng: random.Random, dimensions: int) -> list[int]:
    """A Rademacher direction: each entry +1 or -1."""
    return [rng.choice((-1, 1)) for _ in range(dimensions)]


def sides(u: list[float], delta: list[int], ck: float) -> tuple[list[float], list[float]]:
    """The two points measured at an iteration, each clipped to [-1, 1]."""
    plus = [clip(x + ck * d) for x, d in zip(u, delta)]
    minus = [clip(x - ck * d) for x, d in zip(u, delta)]
    return plus, minus


def update(u: list[float], delta: list[int], f_plus: float, f_minus: float, k: int,
           gains: Gains) -> list[float]:
    """One SPSA step from the two sides' objective values."""
    ck, ak = gains.perturbation(k), gains.step(k)
    return [clip(x - ak * (f_plus - f_minus) / (2 * ck * d)) for x, d in zip(u, delta)]


def clip(x: float) -> float:
    return min(1.0, max(-1.0, x))


def to_value(u: float, low: float, high: float) -> float:
    return low + (u + 1) / 2 * (high - low)


def to_unit(value: float, low: float, high: float) -> float:
    return clip(2 * (value - low) / (high - low) - 1) if high > low else 0.0


@dataclass
class Sprt:
    """A running sequential test of 'the candidate gains `delta`' against 'it gains 0'."""
    delta: float
    alpha: float = 0.05
    beta: float = 0.05
    samples: list[float] = field(default_factory=list)

    @property
    def upper(self) -> float:
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1 - self.alpha))

    def add(self, values: list[float]) -> None:
        self.samples.extend(values)

    def sd(self) -> float:
        n = len(self.samples)
        if n < 2:
            return SD_FLOOR
        mean = sum(self.samples) / n
        var = sum((x - mean) ** 2 for x in self.samples) / (n - 1)
        return max(SD_FLOOR, math.sqrt(var))

    def llr(self) -> float:
        """log(likelihood under H1 / under H0); positive favours the gain."""
        s2 = self.sd() ** 2
        return (-self.delta / s2) * sum(d + self.delta / 2 for d in self.samples)

    def decision(self) -> str | None:
        """'adopt', 'reject', or None (go on)."""
        if len(self.samples) < MIN_SAMPLES:
            return None
        llr = self.llr()
        if llr >= self.upper:
            return "adopt"
        if llr <= self.lower:
            return "reject"
        return None

    def summary(self) -> dict:
        n = len(self.samples)
        mean = sum(self.samples) / n if n else None
        return {"n": n, "mean": mean, "sd": self.sd(), "llr": self.llr() if n else 0.0,
                "bounds": [self.lower, self.upper], "decision": self.decision(),
                "delta": self.delta}
