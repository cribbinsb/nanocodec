"""Multi-fidelity trials for nceval.tune: rungs, dominance pruning, identity dedup.

A tune trial used to be measured once, at one fidelity. Most trials of a
Pareto search land far from the front, and the cheap first look already
says so. A *ladder* measures each trial on a sequence of rungs of rising
fidelity (fewer frames and quantizers first, the study's full fidelity
last) and stops at the first rung where the trial is clearly dominated:

  - every objective (PSNR BD, VMAF BD, CPU ratio) of the trial, minus its
    margin, is still no better than that of a trial that already reached
    the last rung (compared at the same rung); or
  - a class guard is exceeded by more than the margin.

The margin is MARGIN_SIGMAS times the rung's error against the last rung,
measured by the fidelity study (nceval.fidelity; docs/handoff/search.md).
A pruned trial completes with its rung values, so the sampler learns the
region is poor, but it is marked (`PRUNED_ATTR`) and never enters a front
or a preset.

Identity dedup: a trial whose first-rung streams are byte-identical to an
earlier trial's (same binary, same clips, frames and quantizers) encodes
exactly the same thing at every rung the rung-1 set covers; categorical
spaces and inactive parameters make this common. Such a trial reuses the
earlier trial's result instead of measuring again (`DUPLICATE_ATTR`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

#: A rung prunes a trial only when it is worse by this many of the rung's
#: standard errors (two-sided ~95 %: a front member is rarely pruned).
MARGIN_SIGMAS = 2.0
PRUNED_ATTR = "pruned_at_rung"
DUPLICATE_ATTR = "duplicate_of"
RUNG_VALUES_ATTR = "rung_values"
RUNG_KEY_ATTR = "rung1_streams"

#: Error of each rung's objective *differences* against the 60-frame, 4-QP
#: study fidelity (label -> (quality BD sigma %, CPU-ratio sigma)): the
#: larger of the PSNR and VMAF `pair_sigma` of the AV1 fidelity study
#: (nceval.fidelity, 2026-11-08: 11 configurations of the preset space on
#: the preset study's 6 clips; docs/handoff/search.md). The CPU objective is
#: measured identically on every rung (its callgrind frames do not depend
#: on the rung), so its sigma is 0. `python3 -m nceval.tune stats` re-derives
#: these on a running study's own promoted trials.
RUNG_SIGMAS: dict[str, tuple[float, float]] = {"30fx3": (1.66, 0.0), "30fx4": (1.59, 0.0)}


@dataclass(frozen=True)
class Rung:
    """One fidelity: `frames` per clip, the quantizers at `qp_indices` of the anchor grid.

    `sigma` is the rung's error (percent BD, per quality objective) against
    the last rung and `sigma_cpu` that of the CPU ratio; both 0 on the last.
    """
    frames: int
    qp_indices: tuple[int, ...]
    sigma: float = 0.0
    sigma_cpu: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.frames}fx{len(self.qp_indices)}"


#: Quantizer subsets of a fixed-QP rung, by count (indices into the anchor grid).
QP_SUBSETS = {"3": (0, 1, 3), "4": (0, 1, 2, 3)}


def parse_ladder(text: str, final_frames: int, sigmas: dict[str, tuple[float, float]],
                 subsets: dict[str, tuple[int, ...]] = QP_SUBSETS,
                 final: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[Rung, ...]:
    """'30x3,60x4' -> rungs; NxQ is N frames on Q points (`subsets`: 3 = the outer three).

    The last rung must be the study's own fidelity (`final_frames` on the
    `final` points). `sigmas` maps a rung label to its measured (quality,
    CPU) error; a rung without a measured error is refused, since its margin
    would be a guess.
    """
    rungs = []
    for item in text.split(","):
        frames, _, count = item.partition("x")
        indices = subsets.get(count)
        if indices is None or not frames.isdigit():
            raise ValueError(f"bad rung {item!r}: use FRAMESxN, N in {sorted(subsets)}")
        rungs.append(Rung(int(frames), indices))
    last = rungs[-1]
    if (last.frames, last.qp_indices) != (final_frames, final):
        raise ValueError(f"the last rung must be {final_frames}x{len(final)} "
                         "(the study's fidelity)")
    out = []
    for rung in rungs[:-1]:
        if rung.label not in sigmas:
            raise ValueError(f"rung {rung.label} has no measured error; known: {sorted(sigmas)}")
        sigma, sigma_cpu = sigmas[rung.label]
        out.append(Rung(rung.frames, rung.qp_indices, sigma, sigma_cpu))
    return (*out, last)


def clearly_dominated(values: tuple[float, ...], others: list[tuple[float, ...]],
                      rung: Rung) -> bool:
    """Whether some finished trial beats `values` in every objective by the margin.

    Objectives are (PSNR BD %, VMAF BD %, CPU ratio), all minimised, each
    pair measured at the same rung.
    """
    margins = (MARGIN_SIGMAS * rung.sigma, MARGIN_SIGMAS * rung.sigma,
               MARGIN_SIGMAS * rung.sigma_cpu)
    return any(all(v - m >= o for v, m, o in zip(values, margins, other)) for other in others)


def clearly_infeasible(constraint: list[float], rung: Rung) -> bool:
    """Whether a class guard (≤ 0 when met) is exceeded by more than the margin."""
    return max(constraint) > MARGIN_SIGMAS * rung.sigma


def stream_key(binary_sha256: str, streams: dict[str, list[tuple[int, str]]]) -> str:
    """Identity of a rung's encodes: binary and every (clip, quantizer, stream hash)."""
    payload = json.dumps([binary_sha256, sorted((clip, sorted(points))
                                                for clip, points in streams.items())])
    return hashlib.sha256(payload.encode()).hexdigest()
