"""The real-use objective for nceval.tune: CBR ladder curves judged by J_real terms.

`nceval.tune search --objective realuse` searches with the adoption rule's
own measurement (docs/PROCEDURES.md "Adoption rule: J_real") instead of the
fixed-QP proxy, which credits a temporal QP cadence real rate control does
not and has no flicker, worst-frame or key-pulse terms:

  - each trial codes the study's clips in CBR at the target ladder
    (nceval.realuse.LADDER_KBPS scaled per clip; `--real-targets` keeps a
    subset of rungs) for `--frames` frames, with the real-use metric set;
  - the default configuration is the anchor, measured once and cached by
    nceval.cbr like every point;
  - objectives (all minimised): class-balanced Q_full and Q_low (mean BD of
    PSNR-YUV, VMAF v1 and MS-SSIM over the whole overlap and its lower-rate
    half) and the CPU ratio of the fixed-QP instruction probe;
  - constraints (<= 0 when met; Optuna's): every J_real guard: each class's
    Q_full and Q_low within the class guard, worst-1 % BD within
    jreal.WORST_GUARD_PCT, flicker within jreal.FLICKER_GUARD_DB, and no more
    non-compliant buffer points than the anchor.

`--flicker-objective` makes flicker a search target rather than a guard:
the objectives become J_real quality ((Q_full + Q_low) / 2), class-balanced
dFlicker vs the default (dB) and the CPU ratio. `--flicker-anchor NAME`
(e.g. svtav1-m12, x265-superfast) adds an absolute constraint: flicker no
worse than that encoder's own real-time rate control at matched rate.

Trial points are cached by the configuration's parameters (not the trial
number), so a trial a 30-minute slice cut short, re-run by Optuna's retry
of stale trials, only encodes what is missing. Optional rungs
(`--ladder`, nceval.ladder) measure a trial at fewer frames or ladder
targets first; their errors against the final rung are REAL_RUNG_SIGMAS.

The periodic-key pulse guard needs a second (periodic-key) pass and is left
to the J_real screen of the selected presets. Points our encoder refuses
are listed on the trial (`refused`), never silently dropped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import jreal, realuse
from .cbr import Family, anchor_families
from .codecs import Codec
from .config import load_paths
from .gatepolicy import EVALUATION_GATE

#: Which J_real summary terms become constraints, with their limits.
GUARDS = (("bd_worst_psnr_y", jreal.WORST_GUARD_PCT),
          ("bd_worst_vmaf_v1", jreal.WORST_GUARD_PCT),
          ("d_flicker_psnr_y", jreal.FLICKER_GUARD_DB))


#: Default fidelity of a real-use study: 30 frames on two ladder targets
#: (the lowest and the highest), the cheapest set whose J_real terms track
#: 150 frames on all four (docs/handoff/search.md, CBR rung errors).
DEFAULT_FRAMES = 30
DEFAULT_TARGETS = (0, 3)
#: Ladder target subsets by count (indices into realuse.LADDER_KBPS).
TARGET_SUBSETS = {"2": DEFAULT_TARGETS, "3": (0, 1, 3), "4": (0, 1, 2, 3)}
#: Error of each real-use rung's objective differences against 150 frames
#: on all four targets (label -> (quality sigma %, CPU sigma)): the larger of
#: the Q_full and Q_low pairwise errors, AV1, 4 configurations of the preset
#: space and the default on the preset study's 6 clips (2026-11-09;
#: docs/handoff/search.md). The CPU probe does not depend on the rung (0).
#: With --flicker-objective the same margin applies to dFlicker (dB), whose
#: measured error is 0.01-0.02 dB: the rung then never stops a trial on
#: flicker alone, which is conservative.
REAL_RUNG_SIGMAS: dict[str, tuple[float, float]] = {
    "30fx2": (3.34, 0.0), "30fx3": (2.57, 0.0), "30fx4": (2.82, 0.0)}


@dataclass(frozen=True)
class Fidelity:
    frames: int
    targets: tuple[int, ...]


@dataclass
class RealUseResult:
    """One measurement's terms, constraints and the J_real report behind them."""
    quality_full: float
    quality_low: float
    flicker: float
    constraints: list[float]
    summary: dict
    refused: list[str]

    def objectives(self, cpu_ratio: float, flicker_objective: bool) -> tuple[float, float, float]:
        if flicker_objective:
            return (self.quality_full + self.quality_low) / 2, self.flicker, cpu_ratio
        return self.quality_full, self.quality_low, cpu_ratio


class RealUseObjective:
    """Measures configurations against the default in CBR, J_real terms."""

    def __init__(self, binary: Path, codec: Codec, clip_names: list[str],
                 fidelities: list[Fidelity], workers: int, class_guard: float,
                 flicker_objective: bool = False, flicker_anchor: str | None = None):
        paths = load_paths()
        clips = realuse.main_clips(paths)
        self.clips = [clips[name] for name in clip_names]
        self.codec, self.binary, self.class_guard = codec, binary, class_guard
        self.flicker_objective = flicker_objective
        self.runner = realuse.Runner(paths, EVALUATION_GATE, workers, over_budget=False)
        self.anchor = self.family({})
        self.external = None
        if flicker_anchor:
            known = {f.name: f for f in anchor_families(paths, realuse.OPENH264_HARNESS,
                                                        extra=["x264-veryfast", "x265-superfast"])}
            if flicker_anchor not in known:
                raise SystemExit(f"unknown --flicker-anchor {flicker_anchor}; known: {sorted(known)}")
            self.external = known[flicker_anchor]
        # Once, before any trial: concurrent trials would otherwise all
        # encode the anchors' points into the same files.
        for fidelity in fidelities:
            anchors = [self.anchor] + ([self.external] if self.external else [])
            self.runner.measure(f"real-use anchors {fidelity.frames}f", anchors, self.clips,
                                fidelity.frames, fidelity.targets, budgeted=False)

    def family(self, overrides: dict[str, float]) -> Family:
        """Our encoder with `overrides`, named by its parameters so its points are reusable."""
        pairs = [f"{name}={value!r}" for name, value in sorted(overrides.items())]
        label = (hashlib.sha256(" ".join(pairs).encode()).hexdigest()[:12] if pairs
                 else "default")
        return Family(f"nanocodec-{self.codec.value}@{label}", self.codec, self.binary,
                      realuse.param_args(pairs), own=True, cqp=False)

    def evaluate(self, overrides: dict[str, float], fidelity: Fidelity) -> RealUseResult:
        test = self.family(overrides)
        families = [self.anchor, test] + ([self.external] if self.external else [])
        curves = self.runner.measure(f"real-use {test.name}", families, self.clips,
                                     fidelity.frames, fidelity.targets, budgeted=False)
        min_points = min(jreal.BD_MIN_POINTS, len(fidelity.targets))
        per_clip = realuse.compare(curves, self.anchor.name, test.name, self.clips, min_points)
        summary = jreal.summarise(per_clip)
        full, low = summary["quality"]["balanced"], summary["quality_low"]["balanced"]
        flicker = summary["d_flicker_psnr_y"]["balanced"]
        if full is None or low is None or flicker is None:
            missing = sorted(set(summary["quality"]["missing"] + summary["quality_low"]["missing"]
                                 + summary["d_flicker_psnr_y"]["missing"]))
            raise ValueError(f"no J_real quality or flicker on {', '.join(missing)}")
        constraints = self.constraints(summary)
        if self.external:
            versus = jreal.summarise(realuse.compare(curves, self.external.name, test.name,
                                                     self.clips, min_points))
            constraints.append(versus["d_flicker_psnr_y"]["balanced"] or 0.0)
            summary["flicker_vs_" + self.external.name] = versus["d_flicker_psnr_y"]
        return RealUseResult(full, low, flicker, constraints, summary,
                             sorted(self.runner.refused.get(test.name, ())))

    def constraints(self, summary: dict) -> list[float]:
        """Every guard as Optuna wants it: <= 0 when met (a missing term counts as met).

        With the flicker objective, flicker vs the default is a target, not a guard.
        """
        out = [max(summary[term]["classes"].values()) - self.class_guard
               for term in ("quality", "quality_low")]
        out += [(summary[term]["balanced"] or 0.0) - limit for term, limit in GUARDS
                if not (self.flicker_objective and term == "d_flicker_psnr_y")]
        buffer = summary["compliance"]
        out.append(float(buffer["test_noncompliant"] - buffer["anchor_noncompliant"]))
        return out
