"""Preset studies on the current protocol: J_real vs CPU, broad pool clips, L3-checked finalists.

One study tunes at most MAX_PARAMS registered parameters (`nanocodec
params`) of one codec preset with Optuna's multi-objective TPE and keeps the
Pareto front of two objectives, both minimised, both relative to the preset:

  1. J: J_real quality (PSNR-YUV, VMAF v1 and MS-SSIM on every frame, GPU
     metrics, equal thirds; nceval.jreal), class-balanced over the study's
     clips, from the same streamed CBR points as `nceval.screen`
     (nceval.realuse Runner);
  2. CPU: x86 callgrind instructions of the trial's own stage-1 encodes (their
     declared subset, nceval.steering_cpu: the same CBR commands, clips, rungs,
     frames and key period as the quality), as a ratio to the preset's: exact on a
     loaded machine, for steering only (finalists are timed on rented cores).

Overfitting is the main risk of tuning on a fixed clip set (owner,
2026-11-11), so the protocol trades depth for breadth:

  - Breadth: TRIAL_CLIPS clips, each coded at ONE ladder rung (rungs
    cycled low to high over the set, fixed per study). The candidate's point
    becomes an equivalent rate delta on the anchor's cached full curve
    (jreal.point_quality: BD-rate's idea at one point, per metric, averaged
    as J_real does). On recorded screens one rung on four times as many
    clips predicted a disjoint clip set's full-ladder J better than the
    full ladder on a quarter of the clips (docs/handoff/tuning.md).
  - Curve-reshaping guard: a trial whose per-clip delta trends with the
    rung (slope over the ladder > RESHAPE_PCT at |t| > RESHAPE_T) changes
    the curve's shape, which one point per clip cannot see; it is
    escalated to the full ladder on its first LADDER_CLIPS clips, and that
    full-ladder J (BD, Q_full and Q_low) is its value (`fidelity` "ladder").
  - Trust region: each parameter's domain starts around the preset value
    (trust_region; an explicit `low high` in the space file wins) and is
    widened only when the front presses on an edge. The first trial is the
    preset; the other start-up trials each move a third of the parameters
    (sparse_moves), TPE takes over after TPE_STARTUP_TRIALS.
  - Re-checks: every RECHECK_EVERY trials the front is re-measured on the
    study's fresh clips (never trial clips); a point whose gain shrinks by
    more than half is dropped from the front.
  - L3 and the overfit gap: finalists are judged on L3 of the study's
    draw (disjoint from every study clip), and each finalist's trial J vs
    L3 J is reported; a study whose L3 gains are under half its trial gains
    is marked distrusted.
  - Agreement: `--half a|b` gives two studies disjoint clip halves (each
    re-checks on the other's); `agree` proposes only the parameter moves
    both studies' best points make in the same direction, after its own L3.

Study design (owner, 2026-11-11): presets were tuned one lever at a time,
so they are unlikely to be a local optimum.

  - Sensitivity first (`sensitivity`): each of ~30 candidate parameters at
    two values each side of the preset (half-way to and at its trust-region
    edges), one at a time, single-point J on the first STAGE_1_CLIPS trial
    clips plus x86 CPU. Live parameters (J clear of 0 by LIVE_J_PCT, or CPU
    by LIVE_CPU) go to a study space file, ranges widened WIDE_FACTOR times
    where the swept edge did not lose more than UNSAFE_J_PCT.
  - CPU as a constraint (`--cpu-max`): maximise J subject to x86 CPU <= a
    ratio to the preset (1.01: equal speed; 0.90: a faster point), which
    finds equal-speed gains a k-weighted score hides; 150-200 trials.

Clips. From the pool draw of the study's rotation key (nceval.pool,
default today): the train-split families of its L1 and L2 levels, taken
round-robin over strata; the first TRIAL_CLIPS are the trial set, the rest
the re-check set (with --half, alternate clips). L3 of the same key is
every other family, so finalists are checked on clips the study never
saw; TEST and adv12 never enter the pool.

Multi-fidelity inside the objective (Optuna's pruners do not support
multi-objective studies): stage 1 is the first STAGE_1_CLIPS trial clips
(while callgrind runs); a trial whose stage-1 J is a confident loss
(nceval.interval lower bound above +STOP_LOWER_PCT), or whose lower bound
is worse than a front point that costs no more CPU, stops there with its
stage-1 values, marked `fidelity` 1. Values rather than an infeasible flag:
both put the trial in TPE's "bad" density, but values keep its order
among bad trials and leave the constraint model to the real guards. Only
fidelity 2 and "ladder" trials enter the front.

A spatial SVC preset (l2-medium, l2-fast of AV1 or
VP9: two layers) is tuned on the same objective, which for a two-layer stream is J of
its full-size output (operating point 0) at the total stream rate; every
point also scores its base layer (operating point 1) at its own rate
(nceval.cbr, nceval.svcbase), and the base layer's single-point J is a
constraint (singlepoint.base_constraint, jreal.SVC_BASE_GUARD_PCT), so a
trial cannot buy the full layer's quality with the base's. The CPU objective
is the two-layer encode's (both layers).

J_real's class guard, the buffer guard and the CBR ceiling guard
(jreal.ceiling_excess: no more over target than the preset) are Optuna
constraints; worst-1 %,
flicker and pulse guards are full-curve terms left to L3.

Finalists: the top FINALISTS of the front by J + k * dCPU (k: --k, J_real's
K_CPU by default; 0.1 for Default quality arms), among those that beat the
preset on it (score below -RECHECK_MIN_GAIN), get a CPU A/B on rented single
cores of their L3s' own encodes (measure_l3_cpu, nceval.decision_cpu: Graviton 3
and 4 for Arm, Zen 5 for x86, equal weight; since 2026-11-23 the same commands
as the L3's quality, each level and key share at its own CPU) and L3 (nceval.levels, single-point over the
whole disjoint L3 pool, escalated to the full ladder when the change reshapes
the curve; candidate vs preset, J with that combined dCPU); only BETTER passes.
A finalist that moves a rate-control or structure parameter (rate.*, layers.*,
cadence, key frames, long-term references) must first pass a long-horizon
guard: the LONG_GUARD_CLIPS over LONG_GUARD_FRAMES (keys every 10 s, full
ladder), each term judged against its measured run-to-run noise
(judge_long_horizon), which the 150-frame levels cannot replace. Nothing lands
automatically: `write-presets` writes candidate preset files and a report
(front plot, finalist table with the overfit gap) for review.

Studies live in $NC_DATA_ROOT/tuning/<study>.sqlite (Optuna RDB storage;
rerun the same `search` command to resume). Points are streamed and cached
(no YUV files); CPU and GPU slots are machine-wide (nceval.slots).

    python3 -m nceval.tune search --study av1-fast-1 --codec av1 --preset fast \\
        --binary BIN --commit C --space tune_spaces/av1-fast.txt --trials 100
    python3 -m nceval.tune front --study av1-fast-1
    python3 -m nceval.tune finalists --study av1-fast-1      # search runs it at the end
    python3 -m nceval.tune write-presets --study av1-fast-1
    python3 -m nceval.tune agree --studies av1-fast-a,av1-fast-b
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna

from . import (buildinfo, checkrun, cloud, decision_cpu, interval, jreal, levels, levers, pool,
               realuse, steering_cpu)
from .singlepoint import (RESHAPE_PCT, RESHAPE_T, assign_rungs, balanced_j, base_constraint,
                          ceiling_constraint, rung_trend, single_points, worst_constraint)
from .cbr import Family
from .clips import Clip, load_clipset
from .codecs import Codec
from .config import load_paths
from .encoders.nanocodec import MEDIUM, preset_values
from .fingerprint import measurement_fingerprint
from .gatepolicy import EVALUATION_GATE, file_sha256
from .tune import (REPO_ROOT, TUNING_DIR,
                   serialize_trial_acquisition, specs_of, study_storage)

#: Cloud studies (cloud_search): a box job of SLICE_MINUTES of trials, plus the margin
#: for the trials in flight when the slice's timeout stops new ones; the default cap on
#: a study's cloud cost (coordinator, 2026-11-21: at most $10 unless overridden).
SLICE_MINUTES = 25.0
SLICE_MARGIN_MINUTES = 15.0
STUDY_MAX_USD = 10.0

PROTOCOL = "pool-jreal-2"
FRAMES = realuse.DEFAULT_FRAMES
TRIAL_CLIPS = 32
STAGE_1_CLIPS = 16
RECHECK_CLIPS = 16
LADDER_CLIPS = 12
MAX_PARAMS = 20
TRUST_FRACTION = 0.15
TRUST_FACTOR = 2.0
#: A domain edge the front comes within this share of the domain's width of
#: is widened by the domain's half-width (up to the registered limit).
EDGE_FRACTION = 0.1
RECHECK_EVERY = 25
#: Re-check: a gain that shrinks by more than this share is dropped.
RECHECK_SHRINK = 0.5
#: Re-checks and finalists skip points that gain less than this (percent J; the
#: preset itself measures about 0).
RECHECK_MIN_GAIN = 0.1
#: L3 gains under this share of the trial gains mark a study distrusted.
TRUST_GAIN_SHARE = 0.5
#: Agreement: a move smaller than this share of the domain width is no move.
AGREE_MIN_MOVE = 0.02
FINALISTS = 5
PARALLEL_TRIALS = 2
MAX_WORKERS = 8
#: How a finalist's x86 change is named in its result (levers.finalist_rows): the
#: rented x86 target of nceval.decision_cpu (the benchmark's c8i.2xlarge).
X86_TARGET = "graniterapids"
TPE_STARTUP_TRIALS = 10
#: Share of the parameters each start-up trial moves (sparse_moves).
SPARSE_SHARE = 1 / 3
#: Sizes of the sensitivity seeds: the best 1, 2, 3, 5 and 8 single moves combined.
SEED_SIZES = (1, 2, 3, 5, 8)
#: Sensitivity sweep: candidates per sweep, what counts as live, the widening of
#: the study range and the study size it feeds.
SENSITIVITY_MAX_PARAMS = 40
LIVE_J_PCT = 0.5
LIVE_CPU = 0.005
UNSAFE_J_PCT = 2.0
WIDE_FACTOR = 2.0
STUDY_PARAMS = 20
#: Finalists that move a rate-control or structure parameter first pass a
#: long-horizon guard on this long static clip (rate_or_structure).
RATE_STRUCTURE_PREFIXES = ("rate.", "layers.")
RATE_STRUCTURE_WORDS = ("cadence", "key", "ltr", "long_term", "intra_period")
LONG_GUARD_CLIPSET = "long.json"
LONG_GUARD_CLIPS = ("long_static_chirla_s024c2", "long_crowd_mot20_02", "long_pan_bdd_00c12bd0")
LONG_GUARD_FRAMES = 720
#: A long-window study's guard: the 150-frame (one key) J of its stage-1 clips may
#: lose at most this much (coordinator, 2026-11-13): a session's start still counts.
SHORT_WINDOW_GUARD_PCT = 1.5
#: A long-window trial costs ~4x a 150-frame one, so it is staged tighter: stage 1
#: on two clips of each class the long window hurts most (the diagnosis' worst:
#: crowd, static, pan, bodycam), 16 clips in all, and a trial whose stage-1 J is
#: more than WINDOW_REACH_PCT behind the best full trial stops there.
WINDOW_STAGE_1_CLASSES = ("crowd", "static", "pan", "bodycam")
WINDOW_STAGE_1_PER_CLASS = 2
WINDOW_TRIAL_CLIPS = 16
WINDOW_REACH_PCT = 1.0
#: Terms of the long-horizon guard: J first, then worst-1 % and the key pulse.
LONG_TERMS = ("quality", "bd_worst_psnr_y", "bd_worst_vmaf_v1", "d_periodic_pulse_psnr_y")
#: Null distribution (mean, sd) of each term per long clip and codec: the preset
#: vs six quality-neutral perturbations, 720 frames, full ladder, 2026-11-12
#: (docs/handoff/tuning.md). HEVC: HEVC Fast vs decision.lambda_scale
#: 0.995/1.005, hevc.inter_rounding 127/129, hevc.lambda_scale 0.997/1.003.
LONG_NULL = {
    Codec.HEVC: {
        "long_static_chirla_s024c2": {"quality": (-1.14, 2.09), "bd_worst_psnr_y": (11.79, 14.92),
                                      "bd_worst_vmaf_v1": (8.42, 13.95),
                                      "d_periodic_pulse_psnr_y": (0.49, 0.45)},
        "long_crowd_mot20_02": {"quality": (-0.08, 0.12), "bd_worst_psnr_y": (-0.38, 0.49),
                                "bd_worst_vmaf_v1": (0.19, 0.53),
                                "d_periodic_pulse_psnr_y": (-0.04, 0.03)},
        "long_pan_bdd_00c12bd0": {"quality": (-0.32, 0.48), "bd_worst_psnr_y": (-0.5, 0.75),
                                  "bd_worst_vmaf_v1": (-1.21, 2.12),
                                  "d_periodic_pulse_psnr_y": (0.0, 0.02)},
    },
    #: H.264 Fast vs decision.lambda_scale 0.771/0.779, h264.inter_rounding_divisor
    #: 5.575/5.631, decision.subpel_move_bias 1.196/1.206 (31c22ec2, VMAFx serve3).
    Codec.H264: {
        "long_static_chirla_s024c2": {"quality": (2.49, 3.25), "bd_worst_psnr_y": (6.73, 9.58),
                                      "bd_worst_vmaf_v1": (9.42, 14.26),
                                      "d_periodic_pulse_psnr_y": (0.24, 0.29)},
        "long_crowd_mot20_02": {"quality": (-0.0, 0.06), "bd_worst_psnr_y": (-0.17, 0.38),
                                "bd_worst_vmaf_v1": (-0.66, 0.87),
                                "d_periodic_pulse_psnr_y": (-0.02, 0.03)},
        "long_pan_bdd_00c12bd0": {"quality": (0.03, 0.04), "bd_worst_psnr_y": (-0.39, 0.32),
                                  "bd_worst_vmaf_v1": (-0.1, 0.18),
                                  "d_periodic_pulse_psnr_y": (0.0, 0.01)},
    },
}
#: Codecs not calibrated yet use HEVC's null.
LONG_NULL_DEFAULT = Codec.HEVC
#: Smallest sd a z-score divides by (percent, or dB for the pulse): six null
#: runs underestimate a clip's spread, and a real 0.5 % effect is not 10 sigma.
LONG_NULL_SD_FLOOR = {"quality": 0.5, "bd_worst_psnr_y": 1.0, "bd_worst_vmaf_v1": 1.0,
                      "d_periodic_pulse_psnr_y": 0.1}
LONG_GUARD_Z = 2.0
#: Any single clip this far outside its noise fails the guard too (one bad clip
#: must not hide behind two flat ones in the average).
LONG_CLIP_Z = 3.0
FIDELITY_ATTR = "fidelity"
CONSTRAINT_ATTR = "constraint"
FULL, LADDER, STOPPED = 2, "ladder", 1


# --- search space -------------------------------------------------------------------

def read_space(text: str, specs: dict[str, tuple[float, float, float]],
               preset: dict[str, float], max_params: int = MAX_PARAMS) -> dict[str, dict]:
    """Domains from `name [low high] [int|log]` lines; without bounds, the trust
    region around the preset value."""
    domains = {}
    for number, raw in enumerate(text.splitlines(), 1):
        words = raw.split("#")[0].split()
        if not words:
            continue
        name, rest = words[0], words[1:]
        if name not in specs:
            raise SystemExit(f"space line {number}: unknown parameter {name}")
        default, reg_low, reg_high = specs[name]
        kind = rest.pop() if rest and rest[-1] in ("int", "log") else "float"
        if rest:
            if len(rest) != 2:
                raise SystemExit(f"space line {number}: expected `name [low high] [int|log]`")
            low, high = float(rest[0]), float(rest[1])
        else:
            low, high = trust_region(preset.get(name, default), reg_low, reg_high, kind)
        if not reg_low <= low <= high <= reg_high:
            raise SystemExit(f"space line {number}: {name} [{low:g}, {high:g}] is outside "
                             f"its registered range [{reg_low:g}, {reg_high:g}]")
        if kind == "log" and low <= 0:
            raise SystemExit(f"space line {number}: a log domain needs low > 0")
        domains[name] = {"low": low, "high": high, "kind": kind,
                         "limits": [reg_low, reg_high]}
    if len(domains) > max_params:
        raise SystemExit(f"{len(domains)} parameters: at most {max_params} here")
    return domains


def trust_region(centre: float, low: float, high: float, kind: str) -> tuple[float, float]:
    """The preset value +- TRUST_FRACTION of the registered range, and for a positive
    value within a factor TRUST_FACTOR of it (a 0.1 in [0, 64] stays small); an
    integer gets at least one step either way."""
    half = TRUST_FRACTION * (high - low)
    lo, hi = centre - half, centre + half
    if centre > 0:
        lo, hi = max(lo, centre / TRUST_FACTOR), min(hi, centre * TRUST_FACTOR)
    if kind == "int":
        lo, hi = min(lo, centre - 1), max(hi, centre + 1)
    return max(low, lo), min(high, hi)


def suggest(trial: optuna.Trial, domains: dict[str, dict]) -> dict[str, float]:
    out = {}
    for name, d in domains.items():
        if d["kind"] == "int":
            out[name] = float(trial.suggest_int(name, math.ceil(d["low"]), math.floor(d["high"])))
        else:
            out[name] = trial.suggest_float(name, d["low"], d["high"], log=d["kind"] == "log")
    return out


def warm_start_point(domains: dict[str, dict], preset: dict[str, float],
                     specs: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    """The preset's own values (codec defaults where it sets none)."""
    point = {}
    for name, d in domains.items():
        value = min(max(preset.get(name, specs[name][0]), d["low"]), d["high"])
        point[name] = int(round(value)) if d["kind"] == "int" else value
    return point


def start_points(path: str | None, domains: dict[str, dict]) -> list[dict[str, float]]:
    """The parameter points of `path` (a JSON list of {name: value}; none without
    one), each clamped to the study's domains: warm starts carried over from an
    earlier study, e.g. a 150-frame study's front into a long-window one."""
    if not path:
        return []
    points = []
    for point in json.loads(Path(path).read_text()):
        unknown = sorted(set(point) - set(domains))
        if unknown:
            raise SystemExit(f"--start-points {path}: not in the space: {', '.join(unknown)}")
        points.append({name: min(max(value, domains[name]["low"]), domains[name]["high"])
                       for name, value in point.items()})
    return points


def sparse_moves(domains: dict[str, dict], count: int, seed: int) -> list[dict[str, float]]:
    """Random points of the trust region, each moving a random SPARSE_SHARE of the
    parameters (the rest stay at the preset).

    Moving all of them at once sums many small losses: in tp-av1-fast-v1 every
    one of 12 all-parameter random starts lost 2.5-7 % J, while single moves of
    the same size were within +-1.3 %.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        point = {}
        for name in rng.sample(sorted(domains), max(1, round(len(domains) * SPARSE_SHARE))):
            d = domains[name]
            value = rng.uniform(d["low"], d["high"])
            point[name] = float(round(value)) if d["kind"] == "int" else value
        out.append(point)
    return out


def widened(domains: dict[str, dict], front: list[optuna.trial.FrozenTrial]) -> dict[str, dict]:
    """Domains with every edge the front presses on moved out by the half-width."""
    out = {}
    for name, d in domains.items():
        d = dict(d)
        width = d["high"] - d["low"]
        values = [t.params[name] for t in front if name in t.params]
        if values and width > 0:
            if min(values) <= d["low"] + EDGE_FRACTION * width:
                d["low"] = max(d["limits"][0], d["low"] - width / 2)
            if max(values) >= d["high"] - EDGE_FRACTION * width:
                d["high"] = min(d["limits"][1], d["high"] + width / 2)
            if d["kind"] == "log":
                d["low"] = max(d["low"], domains[name]["low"])
        out[name] = d
    return out


# --- clips --------------------------------------------------------------------------

def draw_sets(key: str, half: str) -> tuple[list[Clip], list[Clip]]:
    """(trial clips, re-check clips) from the train-split L1+L2 families of draw `key`."""
    paths = load_paths()
    draw = pool.draw(pool.load_pool(paths, levels.usable), key, FRAMES)
    queues: dict[str, list[Clip]] = {s: [] for s in pool.STRATA}
    for clip in draw.levels[pool.QUICK] + draw.levels[pool.ADOPT]:
        if clip.split == "train":
            queues[pool.stratum(clip)].append(clip)
    ordered: list[Clip] = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                ordered.append(queue.pop(0))
    if half == "none":
        return ordered[:TRIAL_CLIPS], ordered[TRIAL_CLIPS:]
    mine, other = (ordered[0::2], ordered[1::2]) if half == "a" else (ordered[1::2], ordered[0::2])
    return mine[:TRIAL_CLIPS], other


def long_window_sets(key: str, frames: int) -> tuple[list[Clip], list[Clip]]:
    """(trial clips, re-check clips) of a long-window study: the pool clips with
    `frames` frames, measured from frame 0, outside the families of draw `key`'s L3
    (so its L3 stays disjoint), strata interleaved (window-fitting work,
    coordinator 2026-11-13: 150-frame trials reward front-loaded bits)."""
    paths = load_paths()
    clips = pool.load_pool(paths, levels.usable)
    final = {c.family for c in pool.draw(clips, key, FRAMES).levels[pool.FINAL]}
    queues: dict[str, list[Clip]] = {s: [] for s in pool.STRATA}
    for clip in sorted(clips, key=lambda c: c.name):
        if clip.frames >= frames and clip.family not in final:
            queues[pool.stratum(clip)].append(dataclasses.replace(clip, start_frame=0))
    ordered: list[Clip] = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                ordered.append(queue.pop(0))
    first = [c for cls in WINDOW_STAGE_1_CLASSES
             for c in [c for c in ordered if c.content_class == cls][:WINDOW_STAGE_1_PER_CLASS]]
    ordered = first + [c for c in ordered if c not in first]
    return ordered[:WINDOW_TRIAL_CLIPS], ordered[WINDOW_TRIAL_CLIPS:]


def clip_record(clips: list[Clip]) -> list[dict]:
    return [{"clip": c.name, "start_frame": c.start_frame, "class": c.content_class}
            for c in clips]


def clips_from_record(record: list[dict]) -> list[Clip]:
    by_name = {c.name: c for c in pool.load_pool(load_paths())}
    missing = [r["clip"] for r in record if r["clip"] not in by_name]
    if missing:
        raise SystemExit(f"study clips no longer in the pool: {', '.join(missing)}")
    return [replace(by_name[r["clip"]], start_frame=r["start_frame"]) for r in record]


# --- measurement --------------------------------------------------------------------

def param_pairs(values: dict[str, float]) -> list[str]:
    return [f"{name}={value!r}" for name, value in sorted(values.items())]


def family(codec: Codec, binary: Path, values: dict[str, float], name: str) -> Family:
    return Family(name, codec, binary, realuse.param_args(param_pairs(values)),
                  own=True, cqp=False)


def settings_label(values: dict[str, float]) -> str:
    return hashlib.sha256(" ".join(param_pairs(values)).encode()).hexdigest()[:12]


def class_constraints(rows: dict[str, dict]) -> list[float]:
    """Class guard and buffer guard (<= 0 when met)."""
    classes: dict[str, list[float]] = {}
    for row in rows.values():
        if row["quality"] is not None:
            classes.setdefault(row["class"], []).append(row["quality"])
    worst = max((sum(v) / len(v) for v in classes.values()), default=0.0)
    return [worst - jreal.CLASS_GUARD_PCT,
            float(sum(row.get("noncompliant", 0) for row in rows.values()))]


def steering_set(runner: realuse.Runner, binary: Path, anchor: Family, clips: list[Clip],
                 rungs: dict[str, int], frames: int, shares: dict[str, float | None]) -> dict:
    """A study's steering CPU sets (nceval.steering_cpu): per key share (its key period in
    seconds, None: one key), the stage-1 encodes' declared subset and the anchor's counts."""
    by_name = {c.name: c for c in clips}
    out = {}
    for share, seconds in shares.items():
        jobs = steering_cpu.jobs_of(runner, anchor, clips, rungs, frames, seconds)
        out[share] = steering_cpu.record(
            jobs, steering_cpu.instructions(binary, anchor.extra, jobs, by_name))
    return out


def recount_steering(setting: "Study") -> dict:
    """The study's steering anchor counted again on this CPU family (same jobs)."""
    if not setting.steering:
        raise SystemExit(f"study {setting.study.study_name} has no own-command steering CPU "
                         f"(counted before 2026-11-23): start a new study")
    by_name = {c.name: c for c in setting.clips}
    return {share: steering_cpu.record(steering_cpu.jobs_from(stored), steering_cpu.instructions(
        setting.binary, setting.anchor.extra, steering_cpu.jobs_from(stored), by_name))
        for share, stored in setting.steering.items()}


# --- the study ----------------------------------------------------------------------

class Study:
    """One study's fixed setting (its user attributes) and its objective."""

    def __init__(self, study: optuna.Study):
        attrs = study.user_attrs
        self.study = study
        self.codec = Codec(attrs["codec"])
        self.binary = Path(attrs["binary"])
        self.preset: dict[str, float] = attrs["preset_values"]
        self.domains: dict[str, dict] = attrs["domains"]
        self.clips = clips_from_record(attrs["clips"])
        self.recheck_clips = clips_from_record(attrs["recheck_clips"])
        self.rungs: dict[str, int] = attrs["rungs"]
        #: The trial window: FRAMES with one key, or a long-window study's frames
        #: with a key every `key_seconds` (long_window_sets).
        self.frames: int = attrs.get("frames", FRAMES)
        self.key_seconds: float | None = attrs.get("key_seconds")
        #: A long-window study's guard on the start: the stage-1 clips' 150-frame J
        #: (one key) may lose at most this much (percent; None: no guard).
        self.short_guard_pct: float | None = attrs.get("short_guard_pct")
        self.stage_1_clips: int = attrs.get("stage_1_clips", STAGE_1_CLIPS)
        #: Stop a trial whose stage-1 J trails the best full trial by more than this.
        self.reach_pct: float | None = attrs.get("reach_pct")
        #: The steering CPU's commands and the anchor's counts per key share ("first",
        #: "second"): callgrind on the trials' own stage-1 encodes (nceval.steering_cpu);
        #: None in a study counted before 2026-11-23 (fixed QP on other clips).
        self.steering: dict | None = attrs.get("steering")
        #: A second key share every trial is also measured at (--second-key-share
        #: SECONDS): quality and CPU with keys every SECONDS; J and CPU are the two
        #: shares' means, and the point may lose at neither (None: one share).
        self.second_share: dict | None = attrs.get("second_key_share")
        self.k: float = study_k(attrs)[0]
        self.workers: int = attrs["workers"]
        #: CPU-capped study: maximise J subject to x86 CPU <= cpu_max (a ratio).
        self.cpu_max: float | None = attrs.get("cpu_max")
        self.anchor = family(self.codec, self.binary, self.preset, f"nanocodec-{self.codec.value}")
        self.lock = threading.Lock()

    def runner(self, workers: int) -> realuse.Runner:
        return realuse.Runner(load_paths(), EVALUATION_GATE, workers, over_budget=False)

    def steering_cpu(self, values: dict[str, float], share: str = "first") -> float:
        """`values`' CPU over the preset's on the trials' own stage-1 encodes at `share`
        (callgrind; nceval.steering_cpu)."""
        stored = (self.steering or {}).get(share)
        if not stored or stored.get("protocol") != steering_cpu.PROTOCOL:
            raise SystemExit(f"study {self.study.study_name} counted its trials' CPU at a "
                             f"fixed QP on other clips (before 2026-11-23); its trials cannot "
                             f"be combined with CPU of their own encodes: start a new study")
        test = family(self.codec, self.binary, values, "steering")
        counts = steering_cpu.instructions(self.binary, test.extra,
                                           steering_cpu.jobs_from(stored),
                                           {c.name: c for c in self.clips})
        return steering_cpu.ratio(counts, stored["anchor"])

    def check_binary(self) -> None:
        if file_sha256(self.binary) != self.study.user_attrs["binary_sha256"]:
            raise SystemExit(f"{self.binary} changed since the study began: start a new study")

    def front(self) -> list[optuna.trial.FrozenTrial]:
        dropped = {int(n) for n, r in self.study.user_attrs.get("rechecks", {}).items()
                   if r["dropped"]}
        return [t for t in pareto_front(self.study.get_trials(deepcopy=False))
                if t.number not in dropped]

    def out_of_reach(self, stats: dict) -> bool:
        """Stage-1 J more than `reach_pct` behind the best full-fidelity trial."""
        full = [trial_j(t) for t in self.study.get_trials(deepcopy=False)
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.user_attrs.get(FIDELITY_ATTR) in (FULL, LADDER)]
        return (self.reach_pct is not None and stats["mean"] is not None and bool(full)
                and stats["mean"] > min(full) + self.reach_pct)

    def dominated(self, stats: dict, cpu: float) -> bool:
        return stats["lo"] is not None and any(
            trial_cpu(t) <= cpu and trial_j(t) < stats["lo"] for t in self.front())

    def objective(self, trial: optuna.Trial) -> float | tuple[float, float]:
        with self.lock:
            domains = self.domains
        values = {**self.preset, **suggest(trial, domains)}
        test = family(self.codec, self.binary, values,
                      f"nanocodec-{self.codec.value}@{settings_label(values)}")
        runner = self.runner(max(1, self.workers // PARALLEL_TRIALS))
        started = time.monotonic()
        stage_1, stage_2 = self.clips[:self.stage_1_clips], self.clips[self.stage_1_clips:]
        with ThreadPoolExecutor(1) as background:
            cost = background.submit(self.steering_cpu, values)
            rows = single_points(runner, self.anchor, test, stage_1, self.rungs, self.frames,
                                 key_seconds=self.key_seconds)
            cpu = cost.result()
        stats = interval.interval(rows, None, 0.0)
        trial.set_user_attr("stage_1", {"j": stats["mean"], "lo": stats["lo"],
                                        "hi": stats["hi"], "n": stats["n"],
                                        "seconds": round(time.monotonic() - started)})
        over_cap = self.cpu_max is not None and cpu > self.cpu_max
        # A confident loss on J + k dCPU, as L3 judges: a speed study's points
        # lose quality by design (all 120 of the first HEVC Faster speed
        # study's trials stopped here on J alone).
        # A between-rungs study (--seed-preset) judges it at the chord's slope
        # (study_k), and not before both rungs are measured; the rungs themselves
        # always run in full, being the chord's two ends.
        rung = trial.user_attrs.get("warm_start", "").startswith("preset")
        k = self.stop_k()
        credit = (k or 0.0) * 100 * (cpu - 1)
        loses = k is not None and stats["lo"] is not None and \
            stats["lo"] + credit > interval.STOP_LOWER_PCT
        fidelity = FULL if rung else STOPPED if (
            loses or over_cap or self.dominated(stats, cpu) or self.out_of_reach(stats)) \
            else FULL
        if fidelity == FULL:
            rows |= single_points(runner, self.anchor, test, stage_2, self.rungs, self.frames,
                                  key_seconds=self.key_seconds)
        j = balanced_j(rows)
        if j is None:
            raise ValueError("no J_real quality on any clip")
        constraint = class_constraints(rows) + base_constraint(rows) + ceiling_constraint(rows)
        # One entry whenever the study has a second share (Optuna needs equal lengths).
        shares = [] if self.second_share is None else [0.0]
        if self.second_share is not None and fidelity == FULL:
            j, cpu, shares = self.both_shares(trial, runner, test, values, (j, cpu))
        if self.short_guard_pct is not None:
            # A long-window study also guards the worst frames (worst-1%): a deeper QP
            # cadence bought J there and failed the long-horizon guard (H.264 #2:
            # static clip worst-1% PSNR-Y +53%, 2026-11-14).
            constraint += worst_constraint(rows)
            short = balanced_j(single_points(runner, self.anchor, test, stage_1, self.rungs))
            trial.set_user_attr("short_j", short)
            constraint.append((short or 0.0) - self.short_guard_pct)
        cap = [] if self.cpu_max is None else [cpu - self.cpu_max]
        change, t_value = rung_trend(rows)
        trial.set_user_attr("rung_trend", [change, t_value])
        if fidelity == FULL and abs(change) > RESHAPE_PCT and abs(t_value) > RESHAPE_T:
            fidelity = LADDER
            j, constraint = self.ladder_j(runner, test)
        constraint = constraint + cap + shares
        trial.set_user_attr(FIDELITY_ATTR, fidelity)
        trial.set_user_attr(CONSTRAINT_ATTR, constraint)
        trial.set_user_attr("cpu", cpu)
        trial.set_user_attr("seconds", round(time.monotonic() - started))
        refused = sorted(runner.refused.get(test.name, ()))
        if refused:
            trial.set_user_attr("refused", refused)
        print(f"trial {trial.number}: fidelity {fidelity}, J {j:+.2f}%, CPU x{cpu:.3f}, "
              f"rung trend {change:+.2f}% (t {t_value:+.1f}), "
              f"{time.monotonic() - started:.0f} s", flush=True)
        self.record(trial, values, j, cpu)
        return j if self.cpu_max is not None else (j, cpu)

    def both_shares(self, trial: optuna.Trial, runner: realuse.Runner, test: Family,
                    values: dict[str, float], first: tuple[float, float]
                    ) -> tuple[float, float, list[float]]:
        """J and CPU as the means over the study's key share and its second one
        (--second-key-share), and the constraint that the point loses at neither:
        each share's J + k dCPU at most 0 (owner, 2026-11-22: a key setting must be
        right at 1 key in 150 frames and at 1 in 300, weighted equally)."""
        share = self.second_share
        rows = single_points(runner, self.anchor, test, self.clips, self.rungs, self.frames,
                             key_seconds=share["seconds"])
        j2 = balanced_j(rows)
        if j2 is None:
            raise ValueError("no J_real quality at the second key share")
        cpu2 = self.steering_cpu(values, "second")
        (j1, cpu1), k = first, self.k
        trial.set_user_attr("shares", {"first": [j1, cpu1], "second": [j2, cpu2]})
        print(f"trial {trial.number}: key shares J {j1:+.2f} / {j2:+.2f}%, "
              f"CPU x{cpu1:.3f} / x{cpu2:.3f}", flush=True)
        scores = [jj + k * 100 * (cc - 1) for jj, cc in ((j1, cpu1), (j2, cpu2))]
        return (j1 + j2) / 2, (cpu1 + cpu2) / 2, [max(scores)]

    def record(self, trial: optuna.Trial, values: dict[str, float], j: float,
               cpu: float) -> None:
        """The trial into the lever ledger, against the study's preset point (trial 0)."""
        start = {**self.preset, **self.study.trials[0].params}
        row = levers.trial_row(self.study.user_attrs, trial.number, values, start, j, cpu,
                               dict(trial.user_attrs), study_source(self.study))
        if row["overrides"]:
            levers.record([row], f"tune trial {trial.number}")

    def ladder_j(self, runner: realuse.Runner, test: Family) -> tuple[float, list[float]]:
        """Full-ladder J_real and guards on the first LADDER_CLIPS trial clips."""
        clips = self.clips[:LADDER_CLIPS]
        curves = runner.measure(f"tune ladder {test.name}", [self.anchor, test], clips,
                                self.frames, None, self.key_seconds, budgeted=False)
        summary = jreal.summarise(realuse.compare(curves, self.anchor.name, test.name, clips,
                                                  paired=True))
        full, low = summary["quality"]["balanced"], summary["quality_low"]["balanced"]
        if full is None or low is None:
            raise ValueError("no full-ladder J_real quality")
        worst = max([v for term in ("quality", "quality_low")
                     for v in summary[term]["classes"].values()], default=0.0)
        buffer = summary["compliance"]
        excess = float(buffer["test_noncompliant"] - buffer["anchor_noncompliant"])
        base = summary["base_quality"]["balanced"]
        svc = [] if base is None else [base - jreal.SVC_BASE_GUARD_PCT]
        ceiling = jreal.ceiling_excess(buffer)
        return (full + low) / 2, [worst - jreal.CLASS_GUARD_PCT, excess, *svc,
                                  *([] if ceiling is None else [ceiling])]

    def recheck(self, checkpoint: int) -> None:
        """Re-measure unchecked front points on fresh clips; drop shrunk gains; widen
        domains the front presses on."""
        attrs = dict(self.study.user_attrs.get("rechecks", {}))
        front = self.front()
        rng = random.Random(checkpoint)
        fresh = rng.sample(self.recheck_clips, min(RECHECK_CLIPS, len(self.recheck_clips)))
        rungs = assign_rungs(fresh, checkpoint)
        runner = self.runner(self.workers)
        for trial in front:
            if str(trial.number) in attrs or trial_j(trial) > -RECHECK_MIN_GAIN:
                continue
            values = {**self.preset, **trial.params}
            test = family(self.codec, self.binary, values,
                          f"nanocodec-{self.codec.value}@{settings_label(values)}")
            try:
                j = balanced_j(single_points(runner, self.anchor, test, fresh, rungs,
                                             self.frames, key_seconds=self.key_seconds))
            except (ValueError, RuntimeError) as error:  # a failed re-check never ends a study
                print(f"re-check #{trial.number} failed: {error}", flush=True)
                continue
            dropped = j is None or j > trial_j(trial) * (1 - RECHECK_SHRINK)
            attrs[str(trial.number)] = {"j_trial": trial_j(trial), "j_fresh": j,
                                        "clips": len(fresh), "dropped": dropped}
            print(f"re-check #{trial.number}: J {trial_j(trial):+.2f}% -> fresh "
                  f"{'n/a' if j is None else f'{j:+.2f}%'}"
                  + (" DROPPED" if dropped else ""), flush=True)
        self.study.set_user_attr("rechecks", attrs)
        with self.lock:
            self.domains = widened(self.domains, self.front())
            self.study.set_user_attr("domains", self.domains)

    def stop_k(self) -> float | None:
        """The k of the stage-1 stop: the study's (study_k), or None while a
        between-rungs study's chord still lacks one of its two rungs."""
        with self.lock:
            attrs = self.study.user_attrs
            if attrs.get("seed_rungs") and "chord" not in attrs:
                return None
            self.k = study_k(attrs)[0]
            return self.k

    def after_trial(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.user_attrs.get("seed_rungs") and "chord" not in study.user_attrs:
            with self.lock:
                set_chord(study)
        done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if done and done % RECHECK_EVERY == 0 and \
                study.user_attrs.get("last_recheck", 0) < done:
            study.set_user_attr("last_recheck", done)
            self.recheck(done)


def rung_points(study: optuna.Study) -> list[tuple[str, float, float]] | None:
    """The two seed rungs' measured (rung, J, CPU) of a between-rungs study, from their
    full-fidelity trials; None until both are in."""
    out = []
    for seed in study.user_attrs.get("seed_rungs", [])[:2]:
        trial = next((t for t in study.get_trials(deepcopy=False)
                      if t.number == seed["trial"]), None)
        if trial is None or trial.state != optuna.trial.TrialState.COMPLETE or \
                trial.user_attrs.get(FIDELITY_ATTR) not in (FULL, LADDER):
            return None
        out.append((seed["rung"], trial_j(trial), trial_cpu(trial)))
    return out if len(out) == 2 else None


def chord_of(points: list[tuple[str, float, float]]) -> dict | None:
    """The line between two rungs in (log CPU, J): its slope as k (J points per percent
    of CPU, -dJ / (100 dln CPU): moving along it scores 0 on J + k dCPU), None when the
    second rung does not buy J with CPU (it is not slower and better)."""
    (_, j0, c0), (_, j1, c1) = points
    if c1 <= c0 or j1 >= j0:
        return None
    return {"k": -(j1 - j0) / (100 * math.log(c1 / c0)), "points": points}


def chord_j(chord: dict, cpu: float) -> float:
    """The chord's J at `cpu` (interpolated, or extended, in log CPU)."""
    (_, j0, c0), (_, j1, c1) = chord["points"]
    return j0 + (j1 - j0) * math.log(cpu / c0) / math.log(c1 / c0)


def set_chord(study: optuna.Study) -> None:
    """Once both seed rungs are measured: the chord into the study (its k for the stop
    and ranking, study_k), said aloud, loudly when an explicit --k exceeds its slope."""
    points = rung_points(study)
    if points is None:
        return
    chord = chord_of(points)
    attrs = study.user_attrs
    if chord is None:
        chord = {"k": None, "points": points}
        print(f"!!! {study.study_name}: rung {points[1][0]} (J {points[1][1]:+.2f}%, CPU "
              f"x{points[1][2]:.3f}) does not buy J with CPU over {points[0][0]}: no chord, "
              f"k stays {attrs['k']:g}", flush=True)
    study.set_user_attr("chord", chord)
    k, source = study_k(study.user_attrs)
    print(f"{study.study_name}: k {k:.4g} ({source})", flush=True)
    if chord["k"] is not None and attrs.get("k_explicit") and attrs["k"] > chord["k"]:
        print(f"!!! {study.study_name}: --k {attrs['k']:g} is above the chord's slope "
              f"{chord['k']:.4g}: the stop will reject moves along the chord toward "
              f"{points[1][0]}, which is the gap this study fills", flush=True)


def study_k(attrs: dict) -> tuple[float, str]:
    """The CPU weight a study stops and ranks by, and where it came from: an explicit
    --k; else a between-rungs study's chord slope once measured (owner/coordinator,
    2026-11-21: at the base rung's k, 42 of 44 moves toward the slower rung were
    stopped as confident losses); else the study's default k."""
    chord = attrs.get("chord") or {}
    if attrs.get("k_explicit"):
        return attrs["k"], "--k"
    if chord.get("k") is not None:
        (low, _, _), (high, _, _) = chord["points"]
        return chord["k"], f"the chord {low} -> {high}"
    return attrs["k"], "the default k" + (" (chord not measured yet)"
                                          if attrs.get("seed_rungs") else "")


def trial_j(trial: optuna.trial.FrozenTrial) -> float:
    return trial.values[0]


def trial_cpu(trial: optuna.trial.FrozenTrial) -> float:
    """x86 CPU ratio: the second objective, or (CPU-capped studies) a user attribute."""
    return trial.user_attrs["cpu"] if len(trial.values) == 1 else trial.values[1]


def pareto_front(trials: list[optuna.trial.FrozenTrial]) -> list[optuna.trial.FrozenTrial]:
    """Nondominated full-fidelity trials that meet the constraints."""
    done = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get(FIDELITY_ATTR) in (FULL, LADDER)
            and max(t.user_attrs.get(CONSTRAINT_ATTR, [0.0])) <= 0]
    point = lambda t: (trial_j(t), trial_cpu(t))
    if any(len(t.values) == 1 for t in done):
        return sorted(done, key=trial_j)  # CPU-capped: every feasible point, best J first
    return [t for t in done if not any(
        all(a <= b for a, b in zip(point(o), point(t))) and point(o) != point(t)
        for o in done)]


def score(trial: optuna.trial.FrozenTrial, k: float) -> float:
    """J + k * dCPU (percent), as L3 judges it: the order finalists are taken in."""
    return trial_j(trial) + k * 100 * (trial_cpu(trial) - 1)


def constraints_of(trial: optuna.trial.FrozenTrial) -> list[float]:
    return trial.user_attrs.get(CONSTRAINT_ATTR, [0.0, 0.0])


def load(name: str) -> optuna.Study:
    study = optuna.load_study(study_name=name, storage=study_storage(name))
    if study.user_attrs.get("protocol") != PROTOCOL:
        raise SystemExit(f"{name} is not a {PROTOCOL} study (legacy studies: `tune "
                         "front-legacy`)")
    return study


def sampler(seed: int) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed,
                                      n_startup_trials=TPE_STARTUP_TRIALS,
                                      constraints_func=constraints_of)


def create(args: argparse.Namespace) -> optuna.Study:
    """A new study: clips, rungs, domains and preset recorded, the anchor measured once."""
    if not (args.codec and args.binary and (args.space or args.params)):
        raise SystemExit("a new study needs --codec, --binary and --space or --params")
    codec = Codec(args.codec)
    binary, commit = levels.build(args.binary, load_paths(), "tune")
    specs = specs_of(binary, codec)
    presets = binary.parent / "presets"
    presets = presets if presets.is_dir() else REPO_ROOT / "crates/nc-encoder/presets"
    preset = preset_values(presets, codec, args.preset)
    space = Path(args.space).read_text() if args.space else "\n".join(args.params.split(","))
    domains = read_space(space, specs, preset)
    key = pool.rotation_key(args.rotation)
    window = getattr(args, "window_frames", None)
    clips, recheck = long_window_sets(key, window) if window else draw_sets(key, args.half)
    frames = window or FRAMES
    key_seconds = levels.LONG_KEY_SECONDS if window else None
    workers = min(args.workers, realuse.max_workers())  # the box's own on a rented box
    print(f"study {args.study}: {codec.value} {args.preset}, {len(domains)} parameters, "
          f"rotation {key}, {len(clips)} trial clips, {len(recheck)} re-check clips",
          flush=True)
    anchor = family(codec, binary, preset, f"nanocodec-{codec.value}")
    runner = realuse.Runner(load_paths(), EVALUATION_GATE, workers, over_budget=False)
    # The re-check clips' anchor curves are measured at the first re-check.
    runner.measure("tune anchor curves", [anchor], clips, frames, None, key_seconds,
                   budgeted=False)
    stage_1 = clips[:WINDOW_STAGE_1_PER_CLASS * len(WINDOW_STAGE_1_CLASSES) if window
                    else STAGE_1_CLIPS]
    second_share = None
    shares = {"first": key_seconds}
    if getattr(args, "second_key_share", None):
        second_share = {"seconds": float(args.second_key_share)}
        shares["second"] = second_share["seconds"]
    steering = steering_set(runner, binary, anchor, stage_1, assign_rungs(clips), frames,
                            shares)
    study = optuna.create_study(study_name=args.study, storage=study_storage(args.study),
                                directions=["minimize"] if args.cpu_max else ["minimize", "minimize"],
                                sampler=sampler(args.seed))
    attrs = {"protocol": PROTOCOL, "codec": codec.value, "preset": args.preset,
             "preset_values": preset, "binary": str(binary),
             "binary_sha256": file_sha256(binary), "commit": commit or args.commit or buildinfo.clean_commit(binary) or "",
             "domains": domains, "rotation": key, "half": args.half,
             "clips": clip_record(clips), "recheck_clips": clip_record(recheck),
             "rungs": assign_rungs(clips), "frames": frames, "key_seconds": key_seconds,
             "short_guard_pct": SHORT_WINDOW_GUARD_PCT if window else None,
             "stage_1_clips": WINDOW_STAGE_1_PER_CLASS * len(WINDOW_STAGE_1_CLASSES) if window
             else STAGE_1_CLIPS,
             "reach_pct": WINDOW_REACH_PCT if window else None,
             "k": jreal.K_CPU if args.k is None else args.k, "k_explicit": args.k is not None,
             # a between-rungs study: its chord's ends, trials 0 and 1 (set_chord)
             "seed_rungs": [{"rung": args.preset, "trial": 0},
                            {"rung": args.seed_preset[0], "trial": 1}]
             if getattr(args, "seed_preset", []) else [],
             "workers": workers, "measurement": measurement_fingerprint(),
             "cpu_max": args.cpu_max, "j_max": args.j_max,
             "steering": steering, "second_key_share": second_share,
             "cpu_family": cloud.cpu_family(),
             "cpu_families": [{"family": cloud.cpu_family(), "from_trial": 0}]}
    for name, value in attrs.items():
        study.set_user_attr(name, value)
    start = warm_start_point(domains, preset, specs)
    study.enqueue_trial(start, user_attrs={"warm_start": "preset"})
    # A study of the gap between two rungs starts from both ends (owner, 2026-11-21).
    for other in getattr(args, "seed_preset", []):
        study.enqueue_trial(warm_start_point(domains, preset_values(presets, codec, other), specs),
                            user_attrs={"warm_start": f"preset {other}"})
    sweep = Path(args.space).with_name("results.json") if args.space else None
    seeds = sensitivity_seeds(json.loads(sweep.read_text()), domains, args.cpu_max,
                              study_k(study.user_attrs)[0]) \
        if sweep and sweep.exists() else []
    for moves in seeds:
        study.enqueue_trial({**start, **moves}, user_attrs={"warm_start": "sensitivity"})
    # Points carried over from an earlier study of the same space (--start-points).
    for moves in start_points(getattr(args, "start_points", None), domains):
        study.enqueue_trial({**start, **moves}, user_attrs={"warm_start": "start points"})
    for moves in sparse_moves(domains, max(0, TPE_STARTUP_TRIALS - 1 - len(seeds)), args.seed):
        study.enqueue_trial({**start, **moves}, user_attrs={"warm_start": "sparse"})
    return study


def sensitivity_seeds(results: dict, domains: dict[str, dict], cpu_max: float | None,
                      k: float = 0.0) -> list[dict[str, float]]:
    """Start-up points from a sensitivity sweep's single moves (owner, 2026-11-11).

    Each parameter's best swept value that gained J (within the CPU cap, if
    any) is a single move; the seeds combine the best SEED_SIZES of them. A
    speed study (cap below 1) instead takes the moves that save CPU, ranked
    by J + k * dCPU, so their combinations reach the cap.
    Random joint moves of all parameters lose (they sum many small losses:
    tp-av1-fast-v1, and 111 of 120 trials of the first equal-CPU AV1 Fast
    study fell outside the cap or the class guard), so TPE starts from the
    measured improvements instead.
    """
    speed = cpu_max is not None and cpu_max < 1
    value = (lambda r: r["j"] + k * 100 * (r["cpu"] - 1)) if speed else (lambda r: r["j"])
    best: dict[str, dict] = {}
    for key, r in results.items():
        if key.startswith("_") or r["name"] not in domains or r["j"] is None or value(r) >= 0:
            continue
        if (speed and r["cpu"] >= 1) or (not speed and cpu_max is not None and r["cpu"] > cpu_max):
            continue
        d = domains[r["name"]]
        if d["low"] <= r["value"] <= d["high"] and (
                r["name"] not in best or value(r) < value(best[r["name"]])):
            best[r["name"]] = r
    ranked = sorted(best.values(), key=value)
    sizes = sorted({min(n, len(ranked)) for n in SEED_SIZES if ranked})
    return [{r["name"]: r["value"] for r in ranked[:n]} for n in sizes]


def search(args: argparse.Namespace) -> None:
    if not cloud.runs_here(args.local):
        return cloud_search(args)
    if args.study in optuna.study.get_all_study_names(study_storage(args.study)):
        study = load(args.study)
        study.sampler = sampler(args.seed)
        repin_binary(study, args.binary)
        pin_commit(study, args.commit)
        pin_study_family(study, args.re_anchor)
    else:
        study = create(args)
    setting = Study(serialize_trial_acquisition(study))
    setting.check_binary()
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.trials - done)
    print(f"{done} trials done, {remaining} to go ({PARALLEL_TRIALS} at a time)", flush=True)
    if remaining:
        study.optimize(setting.objective, n_trials=remaining, n_jobs=PARALLEL_TRIALS,
                       timeout=args.timeout * 60 if args.timeout else None,
                       catch=(ValueError, RuntimeError), callbacks=[setting.after_trial])
    print_front(study)
    # Finalists once the trial budget is spent, failed trials included (a run a
    # --timeout cut short resumes first).
    spent = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)
                for t in study.trials)
    if args.finalists and spent >= args.trials:
        run_finalists(study, args.finalists, args.workers)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"status": "done", "study": args.study, "trials": args.trials,
             "complete": completed(study), "spent": spent}) + "\n")


def pin_study_family(study: optuna.Study, re_anchor: bool) -> None:
    """A resumed study's CPU stays on the family it was counted on (check_family); a
    study from before the pin adopts this one."""
    attrs = study.user_attrs
    here = cloud.cpu_family()
    if "cpu_family" not in attrs:
        print(f"{study.study_name}: no CPU family recorded (older study): pinned to {here}",
              flush=True)
    elif check_family(attrs["cpu_family"], f"study {study.study_name}", re_anchor):
        setting = Study(study)
        setting.check_binary()
        study.set_user_attr("steering", recount_steering(setting))
    else:
        return
    history = [*attrs.get("cpu_families", []),
               {"family": here, "from_trial": len(study.trials)}]
    study.set_user_attr("cpu_family", here)
    study.set_user_attr("cpu_families", history)


def completed(study: optuna.Study) -> int:
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def study_family(name: str) -> str | None:
    """The instance family a stored study's CPU was counted on (None: none recorded,
    or this machine's)."""
    return cloud.box_family(stored_attrs(name).get("cpu_family"))


def local_completed(name: str) -> int:
    """Complete trials of study `name` in this machine's storage (0: no such study)."""
    if not (TUNING_DIR / f"{name}.sqlite").exists() or \
            name not in optuna.study.get_all_study_names(study_storage(name)):
        return 0
    return completed(optuna.load_study(study_name=name, storage=study_storage(name)))


def cloud_search(args: argparse.Namespace) -> None:
    """The study on rented boxes: slices of SLICE_MINUTES of trials, each one
    aws_measure job on a warm box (the study's storage goes up and comes back, its points
    stay in the cloud cache, CPU scoring, callgrind there), until the trials are done or
    the study has cost --max-usd. A resumed study asks for the instance family its CPU
    was counted on (check_family). Finalists (rented-core CPU and L3) are not run here:
    `tune finalists` afterwards."""
    check_started(stored_attrs(args.study).get("tools_version"))
    binary = cloud_binary(args.binary)
    passed = drop_options(sys.argv[2:] if sys.argv[1:2] == ["search"] else sys.argv[1:],
                          ("--binary", "--timeout", "--finalists", "--json", "--commit"), ("--local",))
    spent_usd, started = 0.0, time.monotonic()
    done = local_completed(args.study)
    print(f"cloud study {args.study}: {done} of {args.trials} trials done; slices of "
          f"{args.slice_minutes:g} min on warm boxes, at most ${args.max_usd:.2f} "
          f"(--max-usd; a warm 96-vCPU box costs about $0.6/h in eu-north-1)", flush=True)
    while done < args.trials:
        if spent_usd >= args.max_usd:
            print(f"cloud study {args.study}: STOPPED at its ${args.max_usd:.2f} cap "
                  f"(${spent_usd:.2f} spent, {done} of {args.trials} trials); raise --max-usd "
                  f"to go on", flush=True)
            break
        slice_started = time.monotonic()
        code, result = offload("search", [*passed, "--binary", str(binary), "--timeout",
                                          f"{args.slice_minutes:g}", "--finalists", "0"],
                               args.slice_minutes + SLICE_MARGIN_MINUTES,
                               None if args.re_anchor else study_family(args.study))
        if args.re_anchor:  # once: the slices after it stay on the new family
            passed = [a for a in passed if a != "--re-anchor"]
            args.re_anchor = False
        cost = (result.get("cloud") or {}).get("estimated_usd") or 0.0
        spent_usd += cost
        if local_completed(args.study):  # the box pinned its own path: back to this one
            repin_binary(load(args.study), str(binary))
            load(args.study).set_user_attr("tools_version", cloud.tools_version())
        before, done = done, local_completed(args.study)
        minutes = (time.monotonic() - slice_started) / 60
        per = f"{minutes / (done - before):.2f} min and ${cost / (done - before):.3f} a trial" \
            if done > before else "no trial finished"
        print(f"cloud study {args.study}: slice {done - before} trials in {minutes:.1f} min, "
              f"${cost:.2f} ({per}); {done} of {args.trials} done, ${spent_usd:.2f} spent",
              flush=True)
        if code != 0 or done == before:
            raise SystemExit(f"cloud study {args.study}: the slice failed (exit {code}) or made "
                             f"no progress; the study keeps what it has (resume: same command)")
    print(f"cloud study {args.study}: {done} trials, {(time.monotonic() - started) / 60:.1f} "
          f"min, ${spent_usd:.2f}. Next: `python3 -m nceval.tune front --study {args.study}`, "
          f"then `tune finalists --study {args.study}` (rented-core CPU and L3)", flush=True)


def pin_commit(study: optuna.Study, given: str | None) -> None:
    """Record the study's commit (the rented-core CPU step builds it): `given`
    (--commit), else the binary's own (buildinfo.clean_commit); a given commit that the
    binary's stamp contradicts is refused."""
    attrs = study.user_attrs
    binary = Path(attrs["binary"])
    stamped = buildinfo.clean_commit(binary) if binary.is_file() else None
    if given:
        resolved = levels.git(REPO_ROOT, "rev-parse", f"{given}^{{commit}}")
        if stamped and stamped != resolved:
            raise SystemExit(f"--commit {given} is {resolved[:12]}, but the study's binary says "
                             f"{stamped[:12]}")
        study.set_user_attr("commit", resolved)
    elif not attrs.get("commit") and stamped:
        print(f"{study.study_name}: commit {stamped[:12]} recorded from its binary", flush=True)
        study.set_user_attr("commit", stamped)


def check_started(recorded: str | None) -> None:
    """The stale-tools check (cloud.check_fresh) at a study's or sweep's start, and its
    pin for the slices; a resume from this tree at the tool version the study recorded
    (`recorded`) is the same run going on, not checked again."""
    if recorded and recorded == cloud.tools_version():
        os.environ[cloud.PINNED_ENV] = recorded
    cloud.check_fresh()


def stored_attrs(name: str) -> dict:
    """A stored study's user attributes ({}: none here)."""
    if not (TUNING_DIR / f"{name}.sqlite").exists() or \
            name not in optuna.study.get_all_study_names(study_storage(name)):
        return {}
    return optuna.load_study(study_name=name, storage=study_storage(name)).user_attrs


def cloud_binary(spec: str | None) -> Path:
    """A binary path as given, else the landed-commit cache's build of the commit
    (never a compile here)."""
    return Path(spec) if spec and Path(spec).is_file() else \
        checkrun.landed_binary(spec or "HEAD")


def drop_options(argv: list[str], valued: tuple[str, ...], flags: tuple[str, ...]) -> list[str]:
    """`argv` without the given options (`--x v`, `--x=v`) and flags."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in valued:
            skip = True
        elif arg in flags or any(arg.startswith(v + "=") for v in valued):
            continue
        else:
            out.append(arg)
    return out


# --- sensitivity sweep --------------------------------------------------------------

def sweep_values(centre: float, domain: dict) -> list[float]:
    """Two values each side of `centre`: half-way to and at the trust region's
    edges (a 0/1 switch: the other state); integers rounded, duplicates dropped."""
    low, high = domain["low"], domain["high"]
    if domain["limits"] == [0.0, 1.0] and domain["kind"] == "int":
        return [1.0 - centre]
    values = [low, (low + centre) / 2, (centre + high) / 2, high]
    if domain["kind"] == "int":
        values = [float(round(v)) for v in values]
    return sorted({v for v in values if v != centre})


def live(results: list[dict]) -> bool:
    """A parameter moves J (interval clear of 0 by LIVE_J_PCT) or CPU (LIVE_CPU)."""
    return any((r["lo"] is not None and (r["lo"] >= LIVE_J_PCT or r["hi"] <= -LIVE_J_PCT))
               or abs(r["cpu"] - 1) >= LIVE_CPU for r in results)


def effect(results: list[dict]) -> float:
    """The largest J or CPU change (percent) over a parameter's values."""
    return max(max(abs(r["j"] or 0.0), 100 * abs(r["cpu"] - 1)) for r in results)


def widened_bounds(centre: float, domain: dict, results: list[dict]) -> tuple[float, float]:
    """The study range: each side WIDE_FACTOR times the trust region unless its
    outermost swept value lost more than UNSAFE_J_PCT, then the trust region."""
    reg_low, reg_high = domain["limits"]
    by_value = {r["value"]: r for r in results}
    low, high = domain["low"], domain["high"]
    unsafe = lambda v: v in by_value and (by_value[v]["j"] or 0.0) > UNSAFE_J_PCT
    if not unsafe(min(by_value)):
        low = max(reg_low, centre - WIDE_FACTOR * (centre - low))
    if not unsafe(max(by_value)):
        high = min(reg_high, centre + WIDE_FACTOR * (high - centre))
    return low, high


def sensitivity_dir(root: Path, codec: str, preset: str, key: str) -> Path:
    """A sweep's directory (its results.json, table and space file)."""
    return root / f"sens-{codec}-{preset}-{key}"


def sensitivity(args: argparse.Namespace) -> None:
    """One-at-a-time sweep of a space file's parameters on the first STAGE_1_CLIPS
    trial clips (single-point J, x86 CPU); writes the table, the live parameters'
    study space file and the raw results (resumable). On a warm box unless --local
    (cloud_sensitivity); the table is then written here from what came back."""
    if not cloud.runs_here(args.local):
        args.binary = str(cloud_sensitivity(args))  # the table from what came back
        args.measure = False
    codec = Codec(args.codec)
    binary, commit = levels.build(args.binary, load_paths(), "tune")
    specs = specs_of(binary, codec)
    presets = binary.parent / "presets"
    presets = presets if presets.is_dir() else REPO_ROOT / "crates/nc-encoder/presets"
    preset = preset_values(presets, codec, args.preset)
    domains = read_space(Path(args.space).read_text(), specs, preset, SENSITIVITY_MAX_PARAMS)
    key = pool.rotation_key(args.rotation)
    clips = draw_sets(key, "none")[0][:STAGE_1_CLIPS]
    rungs = assign_rungs(clips)
    out = sensitivity_dir(TUNING_DIR, codec.value, args.preset, key)
    out.mkdir(parents=True, exist_ok=True)
    store = out / "results.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    jobs = sensitivity_jobs(domains, preset, specs, done, len(clips))
    if not getattr(args, "measure", True):
        if jobs:
            print(f"sensitivity: {len(jobs)} values still unmeasured (the table shows the "
                  f"rest; the same command resumes)", flush=True)
        return write_sensitivity(out, domains, preset, specs, done, args)
    started = time.monotonic()
    anchor = family(codec, binary, preset, f"nanocodec-{codec.value}")
    if "_anchor_cost" in done:
        raise SystemExit(f"{store}: its values' CPU was counted at a fixed QP on other clips "
                         f"(before 2026-11-23): start a new sensitivity directory")
    if check_family(done.get("_cpu_family"), f"sensitivity {out.name}", args.re_anchor):
        done.pop("_steering", None)
    if "_steering" not in done:  # callgrind on these values' own encodes (steering_cpu)
        done["_steering"] = steering_set(
            realuse.Runner(load_paths(), EVALUATION_GATE, 1, over_budget=False), binary,
            anchor, clips, rungs, FRAMES, {"first": None})["first"]
    steering = done["_steering"]
    done["_cpu_family"] = cloud.cpu_family()
    print(f"sensitivity {codec.value} {args.preset}: {len(domains)} parameters, "
          f"{len(jobs)} values to measure on {len(clips)} clips (rotation {key})", flush=True)
    lock = threading.Lock()

    def measure(job: tuple[str, float]) -> None:
        name, value = job
        if args.timeout and time.monotonic() - started > 60 * args.timeout:
            return  # a cloud slice's time is up: the next one resumes
        values = {**preset, name: value}
        test = family(codec, binary, values, f"nanocodec-{codec.value}@{settings_label(values)}")
        runner = realuse.Runner(load_paths(), EVALUATION_GATE,
                                max(1, args.workers // PARALLEL_TRIALS), over_budget=False)
        try:
            with ThreadPoolExecutor(1) as background:
                cost = background.submit(steering_cpu.instructions, binary, test.extra,
                                         steering_cpu.jobs_from(steering),
                                         {c.name: c for c in clips})
                stats = interval.interval(single_points(runner, anchor, test, clips, rungs),
                                          None, 0.0)
                cpu = steering_cpu.ratio(cost.result(), steering["anchor"])
        except (ValueError, RuntimeError) as error:  # not stored: a rerun measures it again
            print(f"{name} = {value:g}: failed ({error})", flush=True)
            return
        result = {"name": name, "value": value, "j": stats["mean"], "lo": stats["lo"],
                  "hi": stats["hi"], "n": stats["n"], "cpu": cpu}
        with lock:
            done[f"{name}={value!r}"] = result
            store.write_text(json.dumps(done, indent=1) + "\n")
        levers.record(levers.sensitivity_rows(
            codec.value, args.preset, {f"{name}={value!r}": result},
            {name: preset.get(name, specs[name][0])}, str(store),
            commit=commit or levers.binary_commit(binary), draw=key), "sensitivity")
        print(f"{name} = {value:g}: J {stats['mean']:+.2f}% [{stats['lo']:+.2f}, "
              f"{stats['hi']:+.2f}], CPU x{cpu:.3f}", flush=True)
    with ThreadPoolExecutor(PARALLEL_TRIALS) as executor:
        list(executor.map(measure, jobs))
    write_sensitivity(out, domains, preset, specs, done, args)
    if args.json:
        left = sensitivity_jobs(domains, preset, specs, done, len(clips))
        Path(args.json).write_text(json.dumps({"status": "done", "sweep": out.name,
                                               "measured": len(jobs) - len(left),
                                               "remaining": len(left)}) + "\n")


def sensitivity_jobs(domains: dict[str, dict], preset: dict[str, float],
                     specs: dict[str, tuple[float, float, float]], done: dict,
                     clips: int) -> list[tuple[str, float]]:
    """The (parameter, value) points of the sweep not yet measured on every clip."""
    return [(name, value) for name, d in domains.items()
            for value in sweep_values(preset.get(name, specs[name][0]), d)
            if f"{name}={value!r}" not in done or done[f"{name}={value!r}"]["n"] < clips]


def cloud_sensitivity(args: argparse.Namespace) -> Path:
    """The sweep on warm boxes, a slice of SLICE_MINUTES at a time (its results go up
    and come back, resuming) on the instance family its CPU was counted on, until no
    value is left or it has cost --max-usd; prints the measured wall time."""
    binary = cloud_binary(args.binary)
    key = pool.rotation_key(args.rotation)
    store = sensitivity_dir(TUNING_DIR, args.codec, args.preset, key) / "results.json"
    check_started(json.loads(store.read_text()).get("_tools_version") if store.exists()
                  else None)
    passed = drop_options(sys.argv[2:] if sys.argv[1:2] == ["sensitivity"] else sys.argv[1:],
                          ("--binary", "--timeout", "--json", "--max-usd", "--rotation"),
                          ("--local",))
    spent, started, remaining = 0.0, time.monotonic(), None
    while remaining != 0:
        if spent >= args.max_usd:
            raise SystemExit(f"sensitivity: STOPPED at its ${args.max_usd:.2f} cap with "
                             f"{remaining} values left; raise --max-usd (same command resumes)")
        recorded = json.loads(store.read_text()).get("_cpu_family") if store.exists() else None
        code, result = offload("sensitivity", [*passed, "--binary", str(binary), "--rotation",
                                               key, "--timeout", f"{SLICE_MINUTES:g}"],
                               SLICE_MINUTES + SLICE_MARGIN_MINUTES,
                               None if args.re_anchor else cloud.box_family(recorded))
        spent += (result.get("cloud") or {}).get("estimated_usd") or 0.0
        if store.exists():
            kept = json.loads(store.read_text())
            kept["_tools_version"] = cloud.tools_version()
            store.write_text(json.dumps(kept, indent=1) + "\n")
        if code != 0 or not result.get("measured") and result.get("remaining"):
            raise SystemExit(f"sensitivity: the box slice failed (exit {code}) or measured "
                             f"nothing; what came back is kept (same command resumes)")
        remaining = result.get("remaining", 0)
        if args.re_anchor:  # once: later slices stay on the new family
            passed = [a for a in passed if a != "--re-anchor"]
            args.re_anchor = False
    print(f"sensitivity {args.codec} {args.preset}: swept on boxes in "
          f"{(time.monotonic() - started) / 60:.1f} min, ${spent:.2f}", flush=True)
    return binary


def write_sensitivity(out: Path, domains: dict[str, dict], preset: dict[str, float],
                      specs: dict[str, tuple[float, float, float]], done: dict,
                      args: argparse.Namespace) -> None:
    by_name: dict[str, list[dict]] = {}
    for key, result in done.items():
        if not key.startswith("_"):
            by_name.setdefault(result["name"], []).append(result)
    ranked = sorted((n for n in domains if n in by_name), key=lambda n: -effect(by_name[n]))
    pct = lambda v: "n/a" if v is None else f"{v:+.2f}"
    lines = [f"# Sensitivity: {args.codec} {args.preset}", "",
             f"One parameter at a time vs the preset, single-point J (%, 90 % interval) on "
             f"{STAGE_1_CLIPS} trial clips and x86 CPU ratio; live: J clear of 0 by "
             f"{LIVE_J_PCT} % or CPU by {100 * LIVE_CPU:g} %.", "",
             "| parameter | preset | value | J | CPU | live |", "|---|---|---|---|---|---|"]
    space = [f"# Live parameters of the {args.preset} sensitivity sweep, widened where safe",
             "# (nceval.tune_pool sensitivity; ranked by effect)."]
    for name in ranked:
        results = sorted(by_name[name], key=lambda r: r["value"])
        centre = preset.get(name, specs[name][0])
        is_live = live(results)
        for r in results:
            lines.append(f"| {name} | {centre:g} | {r['value']:g} | {pct(r['j'])} "
                         f"[{pct(r['lo'])}, {pct(r['hi'])}] | x{r['cpu']:.3f} | "
                         f"{'yes' if is_live else ''} |")
        if is_live and len(space) - 2 < STUDY_PARAMS:
            low, high = widened_bounds(centre, domains[name], results)
            kind = domains[name]["kind"]
            space.append(f"{name} {low:g} {high:g}" + (f" {kind}" if kind != "float" else ""))
    (out / "sensitivity.md").write_text("\n".join(lines) + "\n")
    (out / "space.txt").write_text("\n".join(space) + "\n")
    print("\n".join(lines), flush=True)
    print(f"wrote {out / 'sensitivity.md'} and {out / 'space.txt'} "
          f"({len(space) - 2} live parameters)", flush=True)


# --- finalists, L3 and agreement ----------------------------------------------------

def l3_args(setting: "Study", long_window: bool) -> argparse.Namespace:
    """The L3 of the study's draw as a finalist is checked: single-point (a
    parameter-only change; the full ladder if it reshapes the curve), on the long window
    at every key share when `long_window` (a rate, key, cadence or structure move)."""
    return argparse.Namespace(frames=FRAMES, rotation=setting.study.user_attrs["rotation"],
                              max_level=len(pool.LEVELS), dcpu=None, cpu=None, k=setting.k,
                              all_stages=False, final_estimator=levels.SINGLE,
                              final_frames=levels.LONG_FRAMES if long_window else None,
                              key_shares=None)


def l3_plan(setting: "Study", values: dict[str, float], name: str, long_window: bool) -> dict:
    """The CPU plan of a candidate's L3 (nceval.decision_cpu): its own encodes."""
    args = l3_args(setting, long_window)
    test = family(setting.codec, setting.binary, values,
                  f"nanocodec-{setting.codec.value}@{settings_label(values)}")
    draw = levels.draw_for(load_paths(), args)
    jobs = levels.quality_jobs(setting.runner(1), test, draw, (pool.FINAL,), args)
    build = setting.study.user_attrs.get("commit") or str(setting.binary)
    return decision_cpu.plan_entry(name, setting.codec.value, setting.study.user_attrs["preset"],
                                   draw.key, jobs, setting.anchor, test, build, build)


def measure_l3_cpu(setting: "Study", candidates: dict[str, tuple[dict[str, float], bool]],
                   purpose: str) -> dict[str, dict]:
    """Per candidate {name: (values, long_window)}, its CPU charge on its L3's own encodes
    (a decision_cpu record), timed on the rented targets in one aws_cpu_ab run."""
    if not setting.study.user_attrs.get("commit"):
        raise SystemExit("no commit recorded (bare --binary): give --commit, so that the "
                         "rented targets can build it")
    entries = {name: l3_plan(setting, values, name, long)
               for name, (values, long) in candidates.items()}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summary = decision_cpu.measure(list(entries.values()),
                                   load_paths().work / "cpu-ab" / f"{purpose}-{stamp}", purpose)
    return {name: decision_cpu.record(decision_cpu.load_charge(summary, name))
            for name in entries}


def cpu_fields(found: decision_cpu.Charge | None) -> dict:
    """A finalist result's CPU fields (levers.finalist_rows): the combined change and its
    x86 and Arm parts, each the mean over the L3's groups (key shares)."""
    if found is None:
        return {"dcpu": None, "dcpu_x86": None, "dcpu_arm": None, "cpu": None}
    return {"dcpu": found.mean(), "dcpu_x86": found.mean("x86"), "dcpu_arm": found.mean("arm"),
            "dcpu_x86_how": X86_TARGET, "cpu": decision_cpu.record(found)}


def l3_check(setting: Study, values: dict[str, float], label: str, cpu: dict | None,
             workers: int, long_window: bool = False) -> dict:
    """L3 of the study's draw for one candidate (l3_args), J + k dCPU with `cpu`, the
    charge measured on this L3's own encodes (measure_l3_cpu; refused if measured on
    others), each level and key share at its own group's CPU."""
    test = family(setting.codec, setting.binary, values, f"nanocodec-{setting.codec.value}@{label}")
    check_args = l3_args(setting, long_window)
    found = decision_cpu.from_record(cpu)
    if found is not None:
        expected = l3_plan(setting, values, label, long_window)["digest"]
        check_args.cpu = decision_cpu.verify(found, expected, f"finalist {label}")
    print(f"== {label}: dCPU {decision_cpu.describe(found)}; L3", flush=True)
    level = levels.run(setting.runner(workers), (setting.anchor, test), load_paths(),
                       check_args, (pool.FINAL,))["levels"].get(pool.FINAL)
    return {**cpu_fields(found), "k": setting.k, "l3": level,
            "passed": bool(level) and level["decision"] == interval.BETTER}


def finalist_todo(study: optuna.Study, count: int,
                  k: float) -> dict[str, optuna.trial.FrozenTrial]:
    """The best `count` front points by J + k dCPU that beat the preset on it and have
    no finalist result yet."""
    setting = Study(study)
    j_max = study.user_attrs.get("j_max")
    better = [t for t in setting.front() if score(t, k) < -RECHECK_MIN_GAIN
              and (j_max is None or trial_j(t) <= j_max)]
    picks = sorted(better, key=lambda t: score(t, k))[:count]
    if not picks:
        print(f"no front point beats the preset at k {k:g}: no finalists", flush=True)
    results = study.user_attrs.get("finalists", {})
    return {str(t.number): t for t in picks if str(t.number) not in results}


def finalist_long_window(setting: Study, trial: optuna.trial.FrozenTrial) -> bool:
    """Whether a finalist's L3 is the long window at every key share: a long-window
    study, or a rate, key, cadence or structure move."""
    start = setting.study.trials[0].params
    moved = [n for n, v in trial.params.items() if start.get(n) != v]
    return setting.key_seconds is not None or any(rate_or_structure(n) for n in moved)


def measure_finalist_cpu(study: optuna.Study, count: int, k: float | None) -> None:
    """The CPU of the finalists still to check, on their L3s' own encodes
    (measure_l3_cpu), kept in the study (`finalist_cpu`) for run_finalists, wherever
    that runs."""
    setting = Study(study)
    todo = finalist_todo(study, count, setting.k if k is None else k)
    stored: dict = dict(study.user_attrs.get("finalist_cpu", {}))
    missing = {n: ({**setting.preset, **t.params}, finalist_long_window(setting, t))
               for n, t in todo.items() if not (stored.get(n) or {}).get("digest")}
    if not missing:
        return
    stored.update(measure_l3_cpu(setting, {f"t{n}": v for n, v in missing.items()},
                                 "tune-finalists"))
    study.set_user_attr("finalist_cpu", {n.removeprefix("t"): v for n, v in stored.items()})


def require_box_cpu(study: optuna.Study, count: int, k: float | None) -> None:
    """Refuse to rank finalists whose CPU was not measured on their L3s' own encodes."""
    stored = study.user_attrs.get("finalist_cpu", {})
    todo = finalist_todo(study, count, study_k(study.user_attrs)[0] if k is None else k)
    missing = [n for n in todo if not (stored.get(n) or {}).get("digest")]
    if missing:
        raise SystemExit(f"finalists {', '.join('#' + n for n in missing)} have no CPU "
                         f"measured on their L3s' own encodes (study commit "
                         f"{study.user_attrs.get('commit') or 'none'}): run `tune finalists` "
                         f"from the PC, with --commit if the study has none")


def run_finalists(study: optuna.Study, count: int, workers: int, k: float | None = None,
                  timeout_minutes: float | None = None) -> dict:
    """The rented cores' CPU (measured first where this can rent them, else as stored),
    then L3, for the best `count` front points (resumable); `k` re-weights CPU for
    another role (e.g. 0.1: a quality arm for Default). `timeout_minutes`: start no
    finalist after that (a cloud slice; the next one resumes)."""
    started = time.monotonic()
    setting = Study(study)
    setting.check_binary()
    setting.k = setting.k if k is None else k
    if not os.environ.get(cloud.ON_BOX_ENV):
        measure_finalist_cpu(study, count, setting.k)
    require_box_cpu(study, count, setting.k)
    results: dict = dict(study.user_attrs.get("finalists", {}))
    todo = finalist_todo(study, count, setting.k)
    values = {n: {**setting.preset, **t.params} for n, t in todo.items()}
    # The preset point (trial 0): what a finalist moved, for the long-horizon guard.
    start = study.trials[0].params
    box_measured = study.user_attrs.get("finalist_cpu", {})
    for number, trial in todo.items():
        if timeout_minutes is not None and time.monotonic() - started > 60 * timeout_minutes:
            print(f"finalists: {timeout_minutes:g} min passed; the rest next time", flush=True)
            break
        moved = [n for n, v in trial.params.items() if start.get(n) != v]
        long_guard = long_horizon_guard(setting, values[number], workers) \
            if any(rate_or_structure(n) for n in moved) else None
        if long_guard and long_guard["guards"]:
            print(f"#{number}: long-horizon guard failed: {'; '.join(long_guard['guards'])}",
                  flush=True)
            results[number] = {"j": trial_j(trial),
                               **cpu_fields(decision_cpu.from_record(box_measured.get(number))),
                               "l3": None, "passed": False, "long_horizon": long_guard}
        else:
            result = l3_check(setting, values[number], f"t{number}", box_measured.get(number),
                              workers, long_window=finalist_long_window(setting, trial))
            results[number] = {"j": trial_j(trial), **result,
                               **({"long_horizon": long_guard} if long_guard else {})}
        study.set_user_attr("finalists", results)
    levers.record(levers.finalist_rows(study.user_attrs, results,
                                       {t.number: t.params for t in study.trials}, start,
                                       study_source(study)), "tune finalists")
    print_finalists(results)
    return results


def study_source(study: optuna.Study) -> str:
    return str(TUNING_DIR / f"{study.study_name}.sqlite")


def rate_or_structure(name: str) -> bool:
    """A rate-control or frame-structure parameter (cadence, layers, rate.*, key
    frames, long-term references): its effect builds up over long P chains."""
    base = name.removeprefix("nonbase.")
    return base.startswith(RATE_STRUCTURE_PREFIXES) or any(
        word in base for word in RATE_STRUCTURE_WORDS)


def long_guard_clips() -> list[Clip]:
    clips = {c.name: c for c in load_clipset(load_paths().clips / LONG_GUARD_CLIPSET,
                                              load_paths().data_root).clips}
    return [clips[name] for name in LONG_GUARD_CLIPS]


def long_horizon_guard(setting: Study, values: dict[str, float], workers: int) -> dict:
    """The long-horizon check of `values` vs the preset, calibrated on run-to-run noise.

    The LONG_GUARD_CLIPS over LONG_GUARD_FRAMES (keys every 10 s, full ladder):
    the check CLAUDE.md requires before a rate or structure change lands, which
    150-frame levels cannot replace. Each term is turned into a z-score against
    its null distribution on that clip (LONG_NULL: the preset vs six tiny,
    quality-neutral perturbations, HEVC Fast, 2026-11-12) and averaged over the
    clips; a term fails when that mean z exceeds LONG_GUARD_Z, or one clip's z
    exceeds LONG_CLIP_Z. J is the primary
    term; worst-1 % and the key pulse are guards. On the static clip alone those
    two are noise (null worst-1 % PSNR BD +11.8 % sd 14.9, pulse +0.49 sd 0.45
    dB), so a single clip's raw J_real guards misjudge (HEVC Fast's finalists
    all "failed" them; H.264 Fast's cadence 1.3, reverted, also lost J there).
    """
    chosen = long_guard_clips()
    test = family(setting.codec, setting.binary, values,
                  f"nanocodec-{setting.codec.value}@{settings_label(values)}")
    curves = setting.runner(workers).measure("long-horizon guard", [setting.anchor, test], chosen,
                                             LONG_GUARD_FRAMES, None, budgeted=False)
    rows = realuse.compare(curves, setting.anchor.name, test.name, chosen, paired=True)
    return judge_long_horizon(rows, setting.codec)


def judge_long_horizon(rows: dict[str, dict], codec: Codec = LONG_NULL_DEFAULT) -> dict:
    """Mean z-score per term over the long clips (the codec's LONG_NULL); failing terms
    as guards."""
    nulls = LONG_NULL.get(codec, LONG_NULL[LONG_NULL_DEFAULT])
    missing = [clip for clip in nulls if rows.get(clip, {}).get("quality") is None]
    if missing:  # an incomplete long-horizon check is never evidence
        raise ValueError(f"long-horizon guard: no J on {', '.join(missing)}")
    z_scores, guards = {}, []
    for term in LONG_TERMS:
        zs = []
        for clip, null in nulls.items():
            value = rows.get(clip, {}).get(term)
            if value is not None:
                mean, sd = null[term]
                zs.append((value - mean) / max(sd, LONG_NULL_SD_FLOOR[term]))
                if zs[-1] > LONG_CLIP_Z:
                    guards.append(f"{term} on {clip} z {zs[-1]:+.1f} > {LONG_CLIP_Z:g}")
        if zs:
            z_scores[term] = sum(zs) / len(zs)
            if z_scores[term] > LONG_GUARD_Z:
                guards.append(f"{term} z {z_scores[term]:+.1f} > {LONG_GUARD_Z:g}")
    j = [rows[clip]["quality"] for clip in rows if rows[clip].get("quality") is not None]
    mean_j = sum(j) / len(j) if j else None
    print("long-horizon guard: J " + ("n/a" if mean_j is None else f"{mean_j:+.2f}%") + "; z "
          + ", ".join(f"{t} {z:+.1f}" for t, z in z_scores.items())
          + (f"; FAILS: {'; '.join(guards)}" if guards else ""), flush=True)
    return {"clips": list(rows), "frames": LONG_GUARD_FRAMES, "j": mean_j, "z": z_scores,
            "guards": guards}


def finalists(args: argparse.Namespace) -> None:
    """Finalists on a rented box (cloud_finalists) unless --local; here: re-pin the
    study's build (a box's path of the same file), then run_finalists."""
    if not cloud.runs_here(args.local):
        return cloud_finalists(args)
    study = load(args.study)
    repin_binary(study, args.binary)
    if not os.environ.get(cloud.ON_BOX_ENV):
        pin_commit(study, args.commit)
    results = run_finalists(study, args.count, args.workers, args.k, args.timeout)
    if args.json:
        todo = finalist_todo(study, args.count, study_k(study.user_attrs)[0]
                             if args.k is None else args.k)
        Path(args.json).write_text(json.dumps({"status": "done", "study": args.study,
                                               "finalists": results,
                                               "remaining": len(todo)}, default=str) + "\n")


def cloud_finalists(args: argparse.Namespace) -> None:
    """The finalists' CPU on rented single cores from here (measure_l3_cpu: it rents its own
    boxes, which a measurement box cannot), then their long-horizon guards and L3s on
    warm boxes, a slice at a time (each SLICE_MINUTES; the study goes up and comes back,
    resuming), until none is left or the study has cost --max-usd."""
    study = load(args.study)
    check_started(study.user_attrs.get("tools_version"))
    study.set_user_attr("tools_version", cloud.tools_version())
    repin_binary(study, args.binary)
    if not Path(study.user_attrs["binary"]).is_file() and study.user_attrs.get("commit"):
        # a study last run on a box names the box's copy: the commit's build here
        repin_binary(study, str(cloud_binary(study.user_attrs["commit"])))
    Study(study).check_binary()
    pin_commit(study, args.commit)
    measure_finalist_cpu(study, args.count, args.k)
    require_box_cpu(study, args.count, args.k)
    passed = drop_options(sys.argv[2:] if sys.argv[1:2] == ["finalists"] else sys.argv[1:],
                          ("--binary", "--timeout", "--json", "--max-usd", "--commit"),
                          ("--local",))
    spent = 0.0
    while True:
        remaining = len(finalist_todo(load(args.study), args.count,
                                      study_k(study.user_attrs)[0] if args.k is None
                                      else args.k))
        if not remaining:
            break
        if spent >= args.max_usd:
            raise SystemExit(f"tune finalists {args.study}: STOPPED at its ${args.max_usd:.2f} "
                             f"cap with {remaining} finalists left; raise --max-usd")
        code, result = offload("finalists", [*passed, "--binary", study.user_attrs["binary"],
                                             "--timeout", f"{SLICE_MINUTES:g}"],
                               SLICE_MINUTES + SLICE_MARGIN_MINUTES)
        spent += (result.get("cloud") or {}).get("estimated_usd") or 0.0
        if code != 0 or result.get("remaining", remaining) >= remaining:
            raise SystemExit(f"tune finalists {args.study}: the box slice failed (exit {code}) "
                             f"or checked none; the study keeps what it has (same command "
                             f"resumes)")
    print(f"tune finalists {args.study}: ${spent:.2f} on boxes", flush=True)
    print_finalists(load(args.study).user_attrs.get("finalists", {}))


def offload(sub: str, argv: list[str], minutes: float,
            family: str | None = None) -> tuple[int, dict]:
    """`python3 -m nceval.tune SUB ARGV` as one aws_measure job on a warm box (of
    `family` when given), `minutes` its limit there: (exit code, its result JSON)."""
    work = load_paths().work / "cloud-results"
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"tune-{sub}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.json"
    code = subprocess.call([sys.executable, str(cloud.AWS_MEASURE), "--run-minutes",
                            f"{minutes:g}", *(["--instance-family", family] if family else []),
                            "--", "python3", "-m", "nceval.tune", sub, *argv,
                            "--json", str(out)])
    return code, (json.loads(out.read_text()) if out.exists() else {})


def repin_binary(study: optuna.Study, given: str | None) -> None:
    """Point the study at `given` when it is the study's build at another path (a cloud
    job's directory); the sha check (Study.check_binary) still guards it."""
    path = Path(given) if given else None
    if path and path.is_file() and str(path) != study.user_attrs["binary"] and \
            file_sha256(path) == study.user_attrs["binary_sha256"]:
        study.set_user_attr("binary", str(path))


def check_family(recorded: str | None, what: str, re_anchor: bool) -> bool:
    """Whether the CPU counts here (cloud.cpu_family) may join `what`'s, counted on
    `recorded`: True when the anchor must be counted again here (`--re-anchor` on
    another family). Callgrind counts the kernels the CPU's dispatch picks (AVX-512 on
    c7a and c7i, AVX2 elsewhere), so a ratio against another family's anchor is not a
    ratio: a different family is refused unless re-anchored."""
    here = cloud.cpu_family()
    if recorded is None or recorded == here:
        return False
    if not re_anchor:
        raise SystemExit(f"{what}: its CPU was counted on {recorded}, this is {here}: run it "
                         f"there (the cloud driver asks for that family), or --re-anchor to "
                         f"count the anchor again here (later CPU ratios against that)")
    print(f"{what}: re-anchored on {here} (was {recorded})", flush=True)
    return True


def best_point(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """A study's best passed finalist, else its best front point by score."""
    setting = Study(study)
    passed = [int(n) for n, r in study.user_attrs.get("finalists", {}).items() if r["passed"]]
    candidates = [t for t in setting.front() if t.number in passed] or setting.front()
    if not candidates:
        raise SystemExit(f"{study.study_name} has no front yet")
    return min(candidates, key=lambda t: score(t, setting.k))


def agreed_moves(preset: dict[str, float], domains: dict[str, dict],
                 points: list[dict[str, float]]) -> dict[str, float]:
    """The moves every point makes in the same direction, each by the smallest step."""
    agreed = {}
    for name, d in domains.items():
        base = preset.get(name)
        if base is None:
            continue
        moves = [p[name] - base for p in points]
        floor = AGREE_MIN_MOVE * (d["high"] - d["low"])
        if all(m > floor for m in moves) or all(m < -floor for m in moves):
            step = min(moves, key=abs)
            agreed[name] = round(base + step) if d["kind"] == "int" else base + step
    return agreed


def agree(args: argparse.Namespace) -> None:
    """Moves two (or more) studies of one preset agree on, checked on L3."""
    studies = [load(name) for name in args.studies.split(",")]
    first = Study(studies[0])
    if len({(s.user_attrs["preset"], s.user_attrs["binary_sha256"]) for s in studies}) > 1:
        raise SystemExit("agreement needs studies of one preset and one binary")
    points = [best_point(s).params for s in studies]
    moves = agreed_moves(first.preset, first.domains, points)
    print(f"agreed moves: {moves or 'none'}", flush=True)
    report = {"studies": args.studies, "points": points, "moves": moves}
    if moves:
        values = {**first.preset, **moves}
        long_window = any(rate_or_structure(n) for n in moves)
        measured = measure_l3_cpu(first, {"agreed": (values, long_window)}, "tune-agree")
        report |= {"values": values,
                   **l3_check(first, values, "agreed", measured["agreed"], args.workers,
                              long_window=long_window)}
    path = TUNING_DIR / f"agree-{'-'.join(s.study_name for s in studies)}.json"
    path.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {path}" + (f"; L3 {report['l3'] and report['l3']['decision']}"
                             if moves else ""))


# --- reports ------------------------------------------------------------------------

def overfit(results: dict) -> dict | None:
    """Trial-set J vs L3 J over the finalists: the study's overfit gap."""
    pairs = [(r["j"], r["l3"]["mean"]) for r in results.values()
             if r.get("l3") and r["l3"].get("mean") is not None]
    if not pairs:
        return None
    trial = float(np.median([p[0] for p in pairs]))
    final = float(np.median([p[1] for p in pairs]))
    return {"trial_j": trial, "l3_j": final, "gap": final - trial,
            "distrusted": trial < 0 and final > trial * TRUST_GAIN_SHARE}


def print_front(study: optuna.Study) -> None:
    trials = study.trials
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    count = lambda f: sum(t.user_attrs.get(FIDELITY_ATTR) == f for t in complete)
    seconds = [t.user_attrs["seconds"] for t in complete if "seconds" in t.user_attrs]
    k, source = study_k(study.user_attrs)
    chord = (study.user_attrs.get("chord") or {})
    chord = chord if chord.get("k") is not None else None
    print(f"study {study.study_name} ({study.user_attrs['codec']} {study.user_attrs['preset']}): "
          f"{len(complete)} trials ({count(STOPPED)} stopped at stage 1, {count(LADDER)} "
          f"escalated to the full ladder), median {np.median(seconds) if seconds else 0:.0f} s; "
          f"front (J vs preset, x86 CPU ratio, score J + {k:g} dCPU; k from {source}"
          + ("; vs chord: J minus the chord's J at that CPU, negative is better: the "
             "acceptance test" if chord else "") + "):")
    for t in sorted(Study(study).front(), key=trial_cpu):
        versus = f"vs chord {trial_j(t) - chord_j(chord, trial_cpu(t)):+6.2f}  " if chord else ""
        print(f"  #{t.number:3d}  J {trial_j(t):+6.2f}%  CPU x{trial_cpu(t):.3f}  "
              f"score {score(t, k):+6.2f}  {versus}"
              + ", ".join(f"{n}={v:.4g}" for n, v in sorted(t.params.items())))
    print(WINNERS_CURSE_NOTE)


#: Printed under every front: the best of many noisy trials overstates its gain
#: (docs/PROCEDURES.md "Tunable parameters and tuning").
WINNERS_CURSE_NOTE = ("note: trial-clip J and x86 callgrind CPU are selection estimates, not "
                      "results: re-measure a finalist on fresh clips (tune finalists / the "
                      "long-window L3) and its CPU on rented graniterapids and graviton4 "
                      "(aws_cpu_ab) before quoting or landing it")


def front(args: argparse.Namespace) -> None:
    print_front(load(args.study))


def print_finalists(results: dict) -> None:
    for line in finalist_table(results):
        print(line)


def finalist_table(results: dict) -> list[str]:
    pct = lambda v: "n/a" if v is None else f"{v:+.2f}%"
    lines = ["| trial | J (trial clips) | L3 J (90% interval, n) | overfit gap | dCPU x86 | "
             "dCPU Arm | L3 | passed |", "|---|---|---|---|---|---|---|---|"]
    for number, r in sorted(results.items(), key=lambda kv: kv[0]):
        l3 = r["l3"] or {}
        stats = (f"{pct(l3.get('mean'))} [{pct(l3.get('lo'))}, {pct(l3.get('hi'))}], "
                 f"n {l3.get('n')}") if l3 else "n/a"
        gap = l3["mean"] - r["j"] if l3.get("mean") is not None else None
        lines.append(f"| #{number} | {pct(r['j'])} | {stats} | {pct(gap)} | "
                     f"{pct(r['dcpu_x86'])} | {pct(r.get('dcpu_arm', r.get('dcpu_jetson')))} | "
                     f"{l3.get('decision', 'n/a')} | {'yes' if r['passed'] else 'no'} |")
    gap = overfit(results)
    if gap:
        lines.append("")
        lines.append(f"Overfit: median trial J {gap['trial_j']:+.2f}% vs L3 {gap['l3_j']:+.2f}% "
                     f"(gap {gap['gap']:+.2f}%)"
                     + (": DISTRUSTED (L3 keeps under half the trial gain)"
                        if gap["distrusted"] else ""))
    return lines


def write_presets(args: argparse.Namespace) -> None:
    """Candidate preset files (passed finalists, or --trials) and the study report."""
    study = load(args.study)
    attrs = study.user_attrs
    results = attrs.get("finalists", {})
    numbers = ([int(n) for n in args.trials.split(",")] if args.trials
               else sorted(int(n) for n, r in results.items() if r["passed"]))
    out = Path(args.out) if args.out else REPO_ROOT / "crates/nc-encoder/presets" / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    report_dir = TUNING_DIR / study.study_name
    report_dir.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        trial = study.trials[number]
        values = {**attrs["preset_values"], **trial.params}
        result = results.get(str(number))
        path = out / f"{attrs['codec']}-{attrs['preset']}-{study.study_name}-t{number}.toml"
        header = [f"# Candidate for {attrs['preset']} from tuning study {study.study_name}, "
                  f"trial {number} ({PROTOCOL}, rotation {attrs['rotation']}).",
                  f"# Trial clips vs {attrs['preset']}: J {trial_j(trial):+.2f}%, "
                  f"x86 CPU x{trial_cpu(trial):.3f}."]
        if result and result["l3"]:
            l3 = result["l3"]
            header.append(f"# L3: {l3['decision']}, J {l3['mean']:+.2f}% "
                          f"[{l3['lo']:+.2f}, {l3['hi']:+.2f}], n {l3['n']}; dCPU "
                          f"{result['dcpu']:+.2f}% (x86 {result['dcpu_x86']:+.2f}%).")
        header.append("# Review before landing: nothing here is adopted automatically.")
        path.write_text("\n".join(header + [f"{k} = {v!r}" for k, v in sorted(values.items())])
                        + "\n")
        print(f"wrote {path}")
    plot = plot_front(study, report_dir / "front.png", numbers)
    report = report_dir / "report.md"
    report.write_text("\n".join(
        [f"# Study {study.study_name}", "",
         f"{attrs['codec']} {attrs['preset']}, {len(attrs['domains'])} parameters, rotation "
         f"{attrs['rotation']}, {len(attrs['clips'])} trial clips (one rung each), "
         f"{len(attrs['recheck_clips'])} re-check clips.", "",
         f"![front]({plot.name})", "", "## Finalists (L3 of the study's draw)", "",
         *finalist_table(results), "", "## Re-checks on fresh clips", "",
         *[f"- #{n}: {r['j_trial']:+.2f}% -> {r['j_fresh']}"
           + (" (dropped)" if r["dropped"] else "")
           for n, r in sorted(attrs.get("rechecks", {}).items())], "",
         "## Written candidates", "", *[f"- trial {n}" for n in numbers]]) + "\n")
    print(f"wrote {report}")


def plot_front(study: optuna.Study, path: Path, chosen: list[int]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    for fidelity, colour, label in ((STOPPED, "#c8c8c8", "stopped at stage 1"),
                                    (FULL, "#8fa3b8", "all trial clips"),
                                    (LADDER, "#c98f5a", "escalated to the full ladder")):
        points = [t for t in trials if t.user_attrs.get(FIDELITY_ATTR) == fidelity]
        if points:
            ax.scatter([trial_cpu(t) for t in points], [trial_j(t) for t in points], s=14,
                       color=colour, label=label)
    front_points = sorted(Study(study).front(), key=trial_cpu)
    ax.plot([trial_cpu(t) for t in front_points], [trial_j(t) for t in front_points],
            "-o", color="#1f5fa8", markersize=5, label="front")
    for t in front_points:
        if t.number in chosen:
            ax.annotate(f"#{t.number}", (trial_cpu(t), trial_j(t)), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
    ax.axhline(0, color="#888", linewidth=0.6)
    ax.axvline(1, color="#888", linewidth=0.6)
    ax.set_xlabel("x86 instructions per frame (ratio to preset)")
    ax.set_ylabel("J_real quality vs preset (%, lower is better)")
    ax.set_title(f"{study.study_name}: {study.user_attrs['preset']}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def cloud_options(parser, what: str) -> None:
    parser.add_argument("--local", action="store_true",
                        help=f"run {what} here (default: on warm rented boxes, nceval.cloud)")
    parser.add_argument("--max-usd", type=float, default=STUDY_MAX_USD,
                        help=f"cloud: stop starting slices once this has cost so much "
                             f"(default ${STUDY_MAX_USD:g})")
    parser.add_argument("--timeout", type=float,
                        help="start nothing new after N minutes (a cloud slice)")
    parser.add_argument("--json", help="write a summary here (the cloud driver reads it)")


def add_commands(sub) -> None:
    """The protocol's subcommands on nceval.tune's parser (and nceval.tune_spsa's)."""
    from . import tune_spsa
    tune_spsa.add_commands(sub)
    s = sub.add_parser("search", help="run or resume a study (J_real vs CPU, pool clips)")
    s.add_argument("--study", required=True)
    s.add_argument("--codec", choices=[c.value for c in Codec])
    s.add_argument("--preset", default=MEDIUM, help="the preset tuned and measured against")
    s.add_argument("--binary", help="nanocodec binary, or a commit (built and cached)")
    s.add_argument("--commit", help="the binary's commit (default: its --version stamp, or "
                                    "the COMMIT beside it), for the rented-core CPU builds; "
                                    "also recorded on a resumed study that has none")
    space = s.add_mutually_exclusive_group()
    space.add_argument("--space", help="search-space file (`name [low high] [int|log]` lines)")
    space.add_argument("--params", help="comma-separated parameter names (trust region)")
    s.add_argument("--half", choices=("none", "a", "b"), default="none",
                   help="a/b: disjoint clip halves for two agreeing studies")
    s.add_argument("--trials", type=int, default=100)
    s.add_argument("--workers", type=int, default=MAX_WORKERS)
    s.add_argument("--rotation", help="pool rotation key (default: today, UTC)")
    s.add_argument("--cpu-max", type=float,
                   help="maximise J subject to x86 CPU <= this ratio to the preset (e.g. "
                        "1.01 equal speed, 0.90 a faster point) instead of the (J, CPU) front")
    s.add_argument("--window-frames", type=int, nargs="?", const=levels.LONG_FRAMES,
                   help="a long-window study: trials on the pool clips with this many frames "
                        f"(default {levels.LONG_FRAMES}), from frame 0, a key every "
                        f"{levels.LONG_KEY_SECONDS:g} s, outside the rotation's L3 families "
                        "(long_window_sets); for rate, key and cadence parameters")
    s.add_argument("--j-max", type=float,
                   help="finalists lose at most this much J (percent; a speed study's "
                        "accepted quality cost)")
    s.add_argument("--k", type=float,
                   help=f"CPU weight for the stage-1 stop, finalist order and L3's J (default "
                        f"{jreal.K_CPU:g}; with --seed-preset, the slope of the chord between "
                        f"the two rungs once both are measured; 0.1: Default quality arms)")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--second-key-share", metavar="SECONDS",
                   help="also measure every full trial with keys every SECONDS (quality, and "
                        "CPU on those encodes); J and CPU are the means of both shares and a "
                        "point may lose at neither (e.g. 5 beside a long-window study's 10 s)")
    s.add_argument("--start-points", metavar="JSON",
                   help="also start from these points (a JSON list of {parameter: value}), "
                        "e.g. an earlier study's front")
    s.add_argument("--seed-preset", action="append", default=[], metavar="RUNG",
                   help="also start from this rung's values (a study of the gap between "
                        "--preset and RUNG); repeatable")
    s.add_argument("--timeout", type=float, help="stop starting trials after N minutes")
    s.add_argument("--finalists", type=int, default=FINALISTS,
                   help="front points checked on Jetson and L3 once the trials are done")
    s.add_argument("--local", action="store_true",
                   help="run the trials here (default: on warm rented boxes, nceval.cloud)")
    s.add_argument("--max-usd", type=float, default=STUDY_MAX_USD,
                   help=f"cloud: stop starting slices once the study has cost this (default "
                        f"${STUDY_MAX_USD:g})")
    s.add_argument("--slice-minutes", type=float, default=SLICE_MINUTES,
                   help="cloud: trials per box job, in minutes (the study resumes after each)")
    s.add_argument("--re-anchor", action="store_true",
                   help="resume on another CPU family than the study's: count the anchor "
                        "again there (default: refused; the cloud driver asks for its family)")
    s.add_argument("--json", help="write a summary here (the cloud driver reads it)")
    s.set_defaults(func=search)
    f = sub.add_parser("front", help="a study's Pareto front")
    f.add_argument("--study", required=True)
    f.set_defaults(func=front)
    n = sub.add_parser("finalists",
                       help="rented-core CPU (Arm and x86) and L3 for the front's best points")
    n.add_argument("--study", required=True)
    n.add_argument("--count", type=int, default=FINALISTS)
    n.add_argument("--k", type=float, help="CPU weight for this run (default: the study's)")
    n.add_argument("--workers", type=int, default=MAX_WORKERS)
    cloud_options(n, "the L3s")
    n.add_argument("--binary", help="the study's build at another path (a box's)")
    n.add_argument("--commit", help="the study's commit, when it has none recorded (the "
                                    "rented-core CPU step builds it)")
    n.set_defaults(func=finalists)
    w = sub.add_parser("write-presets", help="candidate preset files and the study report")
    w.add_argument("--study", required=True)
    w.add_argument("--trials", help="comma-separated trial numbers (default: passed finalists)")
    w.add_argument("--out", help="directory (default: presets/candidates)")
    w.set_defaults(func=write_presets)
    v = sub.add_parser("sensitivity", help="one-at-a-time sweep: live parameters, safe ranges")
    v.add_argument("--codec", required=True, choices=[c.value for c in Codec])
    v.add_argument("--preset", default=MEDIUM)
    v.add_argument("--binary", required=True)
    v.add_argument("--space", required=True, help="candidate parameters (space file)")
    v.add_argument("--rotation", help="pool rotation key (default: today, UTC)")
    v.add_argument("--workers", type=int, default=MAX_WORKERS)
    cloud_options(v, "the sweep")
    v.add_argument("--re-anchor", action="store_true",
                   help="resume on another CPU family than the sweep's: count the anchor "
                        "again there")
    v.set_defaults(func=sensitivity)
    a = sub.add_parser("agree", help="L3-check the moves two studies agree on")
    a.add_argument("--studies", required=True, help="comma-separated, e.g. X-a,X-b")
    a.add_argument("--workers", type=int, default=MAX_WORKERS)
    a.set_defaults(func=agree)
