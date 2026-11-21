"""Decompose what CBR rate control costs against fixed QP, by replaying quantizer sequences.

    python3 -m nceval.rc_oracle --binary BIN --codec av1 [--clips val|quick|adopt|final] [--workers N]
        [--frames 600 --key-seconds 10] [--per-class N] [--variants replay,startup,...] --json OUT

`nceval.rc_cost` measures the whole cost: an encoder's CBR curve against its
own fixed-QP curve (BD-J). This tool splits it. Every CBR point is encoded
once more with `--rate-log`; its logged quantizers are then replayed through
`nanocodec encode --qp-replay` (the controller runs as usual, only each
frame's quantizer is imposed, never re-coded), each time with one part of
the controller's choices taken out:

- `replay`: the logged sequence itself (a check: equals CBR, except frames
  CBR re-coded, which the replay codes once at their final quantizer);
- `trailing`: each inter frame's level replaced by the mean of the second
  up to it (what a causal controller could do: smoothing without foresight);
- `startup`: the levels of the first second after each key replaced by the
  mean level of the rest of that key's period, the rest as logged: what the
  start-up descent after a key costs, apart from the steady-state drift;
- `plateau2`, `plateau4`: the levels as logged for the first 2 (4) seconds
  after each key, then held at the mean level of that window's last second
  until the next key: causal, what stopping the refinement cycle once the
  picture has recovered from its key would do (the stream then runs under
  its target); `plateau2_f2`, `_f4`, `_f6` hold a level 2, 4 or 6 QP units
  finer than `plateau2` does: where a held level that spends the whole
  target would stand;
- `smooth`: each inter frame's level replaced by its centred one-second
  mean in the step domain (the cadence and layer offsets kept): what the
  level's frame-to-frame jitter costs, including the size model's misses;
- `flat`: every inter level at the sequence's mean step (start-up transient,
  content-driven drift and jitter all gone), the key as logged;
- `reuse1`, `reuse2`, `reuse_anti`: a frame-level reuse weighting with hindsight
  (`reuse_offsets`): each inter level moved finer by 1 (2, -1: the control)
  QP unit per doubling of how much the following second of frames inherits
  from the frame, measured from the coded sizes, at the sequence's mean
  offset (so the mean level is kept): the upper bound of spending frame
  bits where later frames copy them;
- `key_finer`, `key_coarser`: every key 4 QP units finer (coarser), the inter
  frames as logged: the local slope of the key's size (the stream's rate
  moves with it, and the BD is at equal rate); `key1_2`, `key1_4`, `key1_6`
  move only the first (start-up) key 2, 4 or 6 units finer, `keyp_2`,
  `keyp_4`, `keyp_6` only the later (periodic) keys;
- `floor`, `ceil`: each inter level bounded by its key period's median level
  (hindsight): never finer than it (`floor`: the descents onto refinement
  bursts taken out), or never coarser (`ceil`: the climbs after them);
- `flat_key`: `flat` with the key at the fixed-QP key's step ratio to the
  inter level (`key_qp.rs`; H.264 `keyframe_qp_offset` of its fixed-QP profile; AV1
  `keyframe_step_permille`).

The chain CBR -> replay -> smooth -> flat -> flat_key -> fixed QP adds up to
the whole cost; what is left between `flat_key` and fixed QP is what the
rate-control-only tools (the reuse term's level, held frames' budgets) and
the remaining schedule differences (cadence strength) do. Per-point
controller statistics from the logs (size-model error per frame role,
level jitter, key share, re-codes) come with each class.

`--arm NAME:param=value,...` adds a live CBR arm (the preset plus those
parameters, no replay), reported against CBR per clip, so a per-clip best
over several arms bounds what choosing the setting from content could gain.
`--preset` measures a shipped preset instead of the codec's medium.

`--clips`: `val` the natural VAL clips (rc_cost's set), or a pool level of
a draw (`quick`, `adopt`, `final`; not adoption evidence here, only a
broader sample per class), at most `--per-class` clips of each class; or
`set:NAME:N`, a class-spread sample of N clips of another clip set.
`--rotation KEY` names the draw (default: the day's; a diagnostic that
precedes an adoption screen takes its own key, so it does not look at the
clips the screen will judge on).
`--frames` and `--key-seconds` set the window (default: 150 frames, one key;
the product window is 600 frames with keys every 10 s, as L3's long window).
Every point is cached by the shared engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import jreal, key_inherit, levels, pool
from .cbr import Family, Point, measure_all
from .clips import load_clipset
from .codecs import Codec
from .config import load_paths
from .gatepolicy import EVALUATION_GATE, uniform_check
from .rc_cost import clip_terms
from .realuse import (DEFAULT_FRAMES, LADDER_KBPS, MAX_WORKERS_ENV, Runner, max_workers,
                      natural_val, own_family, param_args)

WORKERS = 8
VARIANTS = ("replay", "trailing", "startup", "smooth", "flat", "flat_key")
#: The opt-in reuse-weighted variants: QP units finer per doubling of what
#: the frames after a frame inherit from it (`reuse_offsets`).
REUSE_STRENGTH = {"reuse1": 1.0, "reuse2": 2.0, "reuse_anti": -1.0}
#: The opt-in key variants: which keys ("all", "first": the start-up key, "periodic":
#: every later one) are moved by how many QP units (negative is finer), the inter
#: frames as logged: the slope of the key's size, for the two kinds of key apart.
KEY_SHIFT = {"key_finer": ("all", -4.0), "key_coarser": ("all", 4.0),
             **{f"key1_{n}": ("first", -float(n)) for n in (2, 4, 6)},
             **{f"keyp_{n}": ("periodic", -float(n)) for n in (2, 4, 6)}}
#: How far ahead the reuse weighting follows a frame's content, in seconds.
REUSE_SECONDS = 1.0
#: Inter frames per budget cycle (`rate.cadence_period`'s default; a three-layer mini-GOP).
CYCLE_FRAMES = 4
#: The rate-log columns kept per CBR point in the report, for offline analysis.
LOG_COLUMNS = ("type", "qp", "level_q", "bytes", "budget_bits", "predicted_bits",
               "cadence_offset", "static_fraction", "buffer_fullness", "codings",
               "spatial_hint", "temporal_hint", "camera_motion", "compensable")
#: Quantizer units per doubling of the step (`qp_cadence.rs`).
QP_UNITS_PER_OCTAVE = 6.0
#: Seconds after each key that the opt-in `plateau` variants keep as logged.
PLATEAU_SECONDS = {"plateau2": 2.0, "plateau4": 4.0}
#: `plateau2` held this many QP units finer than the level it would hold: where a
#: held level that spends the whole target would stand, against the same buffer.
PLATEAU_FINER = {"plateau2_f2": 2.0, "plateau2_f4": 4.0, "plateau2_f6": 6.0}
#: Frames of the centred mean of `smooth`: one second at 30 fps.
SMOOTH_FRAMES = 31
#: The start-up window after each key that `startup` flattens, in seconds.
STARTUP_SECONDS = 1.0
#: AV1/VP9/AV2 AC quantizer step per qindex (8-bit; AV1 §7.12.2 `Ac_Qlookup[0]`).
AC_QLOOKUP = (
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134,
    136, 138, 140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182,
    185, 188, 191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251,
    255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 311, 317, 323, 329, 335, 341, 347, 353,
    359, 366, 373, 380, 387, 394, 401, 408, 416, 424, 432, 440, 448, 456, 465, 474, 483, 492, 501,
    510, 520, 530, 540, 550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676, 689, 702, 715,
    729, 743, 757, 771, 786, 801, 816, 832, 848, 864, 881, 898, 915, 933, 951, 969, 988, 1007,
    1026, 1046, 1066, 1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343,
    1369, 1396, 1423, 1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793,
    1828)
#: Units past the table's end over which the encoder extrapolates its slope
#: (`encoder.rs` STEP_EXTRAPOLATION_UNITS), for levels in the λ headroom.
EXTRAPOLATION_UNITS = 8
#: Planned quantizers past the codec's largest (the λ headroom) this tool maps.
HEADROOM_UNITS = 200
#: Fixed-QP key step relative to the configured quantizer, per codec (the
#: fixed-QP profile's, `CodecBackend::FIXED_QP_PARAMS`): H.264
#: `h264.keyframe_qp_offset` 8 QP; HEVC `key_qp.rs` 12 QP at a 60-frame
#: period +3 QP per factor of 5 (at most +3: one key in 150 frames); AV1
#: `av1.keyframe_step_permille` 240.
FIXED_KEY_STEP_RATIO = {Codec.H264: 2 ** (-8 / 6), Codec.HEVC: 2 ** (-15 / 6), Codec.AV1: 0.24}


def step_table(codec: Codec) -> list[float]:
    """Quantizer step per planned quantizer, including the λ headroom."""
    if codec in (Codec.AV1, Codec.VP9, Codec.AV2):
        steps = [float(s) for s in AC_QLOOKUP]
        ratio = (steps[-1] / steps[-1 - EXTRAPOLATION_UNITS]) ** (1 / EXTRAPOLATION_UNITS)
        return steps + [steps[-1] * ratio ** (i + 1) for i in range(HEADROOM_UNITS)]
    return [2 ** (q / 6) for q in range(64 + HEADROOM_UNITS)]


def nearest(steps: list[float], step: float, low: int) -> int:
    """The quantizer (at least `low`) whose step is nearest `step` in the log domain."""
    return min(range(low, len(steps)), key=lambda q: abs(math.log(steps[q] / step)))


def read_log(path: Path) -> list[dict]:
    with path.open() as f:
        header = f.readline().strip().split(",")
        return [dict(zip(header, line.strip().split(","))) for line in f if line.strip()]


def coded(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["type"] in ("Key", "Inter")]


def startup_levels(rows: list[dict], log_level: dict[int, float], fps: float) -> dict[int, float]:
    """`startup`'s level per inter frame of a start-up window: the mean (log step)
    of the logged levels after the window up to the next key."""
    window = round(STARTUP_SECONDS * fps)
    keys = [i for i, r in enumerate(rows) if r["type"] == "Key"] + [len(rows)]
    levels_: dict[int, float] = {}
    for key, end in zip(keys, keys[1:]):
        steady = [log_level[i] for i in range(key + window + 1, end) if i in log_level]
        if steady:
            mean = math.exp(statistics.fmean(steady))
            levels_.update({i: mean for i in range(key + 1, key + window + 1) if i in log_level})
    return levels_


def plateau_levels(rows: list[dict], log_level: dict[int, float], fps: float,
                   seconds: float) -> dict[int, float]:
    """A `plateau` variant's level per inter frame later than `seconds` after its
    key: the mean (log step) of the logged levels of the last second before that."""
    window, second = round(seconds * fps), round(fps)
    keys = [i for i, r in enumerate(rows) if r["type"] == "Key"] + [len(rows)]
    levels_: dict[int, float] = {}
    for key, end in zip(keys, keys[1:]):
        last = [log_level[i] for i in range(key + window - second + 1, key + window + 1)
                if i in log_level]
        if last:
            mean = math.exp(statistics.fmean(last))
            levels_.update({i: mean for i in range(key + window + 1, end) if i in log_level})
    return levels_


def reuse_offsets(rows: list[dict], steps: list[float], fps: float) -> dict[int, float]:
    """Per inter frame, QP units (at strength 1) by which a frame-level reuse
    weighting moves its level; negative is finer, and the mean is 0.

    A frame's coded size times its step is its cost at a common quantizer.
    Against the cost its key would have had on the frame's own picture (the
    key's, scaled by the spatial hints), `f = 1 - inter / intra` is the share
    the frame took from its reference (the propagation fraction of a
    macroblock tree, per frame). What frame t hands on is
    `A = sum_k prod_{j<=k} f[t+j]` over the next `REUSE_SECONDS`: nothing
    passes a key, and past the end of the clip the last second's mean share
    stands in, so the clip's end is not read as the content's. The offset is
    `-log2(1 + A)` less its mean."""
    frames = [i for i, r in enumerate(rows) if r["type"] in ("Key", "Inter")]
    cost = {i: int(rows[i]["bytes"]) * 8 * steps[int(rows[i]["qp"])] for i in frames}
    spatial = {i: max(1.0, float(rows[i].get("spatial_hint") or 1.0)) for i in frames}
    share: dict[int, float] = {}
    key = None
    for i in frames:
        if rows[i]["type"] == "Key":
            key, share[i] = i, 0.0
        elif key is None:
            share[i] = 0.0
        else:
            intra = cost[key] * spatial[i] / spatial[key]
            share[i] = min(1.0, max(0.0, 1.0 - cost[i] / max(intra, 1.0)))
    horizon = max(1, round(REUSE_SECONDS * fps))
    tail = [share[i] for i in frames[-horizon:] if rows[i]["type"] == "Inter"]
    beyond = statistics.fmean(tail) if tail else 0.0
    handed: dict[int, float] = {}
    for position, i in enumerate(frames):
        if rows[i]["type"] != "Inter":
            continue
        total, carried = 0.0, 1.0
        for k in range(1, horizon + 1):
            later = position + k
            carried *= share[frames[later]] if later < len(frames) else beyond
            total += carried
        handed[i] = math.log2(1.0 + total)
    mean = statistics.fmean(handed.values()) if handed else 0.0
    return {i: mean - value for i, value in handed.items()}


def bounded_levels(rows: list[dict], log_level: dict[int, float], finer: bool) -> dict[int, float]:
    """Each inter level bounded by the median level (log step) of its key period:
    never finer than it (`finer` False: the `floor` variant, no descent below the
    level the period sustains) or never coarser (`ceil`: no climb above it)."""
    keys = [i for i, r in enumerate(rows) if r["type"] == "Key"] + [len(rows)]
    bounded: dict[int, float] = {}
    for key, end in zip(keys, keys[1:]):
        period = [i for i in range(key + 1, end) if i in log_level]
        if not period:
            continue
        median = statistics.median(log_level[i] for i in period)
        pick = min if finer else max
        bounded.update({i: math.exp(pick(log_level[i], median)) for i in period})
    return bounded


def schedules(rows: list[dict], codec: Codec, fps: float) -> dict[str, list[str]]:
    """Each variant's `--qp-replay` lines (one per input frame)."""
    steps = step_table(codec)
    low = 1 if codec in (Codec.AV1, Codec.VP9, Codec.AV2) else 0
    inter = [i for i, r in enumerate(rows) if r["type"] == "Inter"]
    log_level = {i: math.log(steps[int(rows[i]["level_q"])]) for i in inter}
    flat_level = math.exp(statistics.fmean(log_level.values())) if inter else 1.0
    half = SMOOTH_FRAMES // 2

    def smooth(i: int, before: int, after: int) -> float:
        window = [log_level[j] for j in inter if -before <= j - i <= after]
        return math.exp(statistics.fmean(window))

    def line(i: int, level_step: float | None, key_step: float | None) -> str:
        row = rows[i]
        q, level = int(row["qp"]), int(row["level_q"])
        if row["type"] == "Key":
            return str(q if key_step is None else nearest(steps, key_step, low))
        if row["type"] != "Inter":
            return "-"
        if level_step is None:
            return f"{q} {level}"
        offset = steps[q] / steps[level]
        return f"{nearest(steps, level_step * offset, low)} {nearest(steps, level_step, low)}"

    key_flat = flat_level * FIXED_KEY_STEP_RATIO.get(codec, 1.0)
    every = range(len(rows))
    startup = startup_levels(rows, log_level, fps)
    plateaus = {name: plateau_levels(rows, log_level, fps, seconds)
                for name, seconds in PLATEAU_SECONDS.items()}
    reuse = reuse_offsets(rows, steps, fps)
    floors = bounded_levels(rows, log_level, finer=False)
    ceils = bounded_levels(rows, log_level, finer=True)

    def reused(i: int, strength: float) -> float | None:
        if i not in reuse:
            return None
        return math.exp(log_level[i]) * 2 ** (strength * reuse[i] / QP_UNITS_PER_OCTAVE)

    first_key = next((i for i, r in enumerate(rows) if r["type"] == "Key"), None)

    def shifted_key(i: int, scope: str, shift: float) -> float | None:
        """Key `i`'s step moved by `shift` QP units when `scope` covers it."""
        if rows[i]["type"] != "Key" or scope == ("periodic" if i == first_key else "first"):
            return None
        return steps[int(rows[i]["qp"])] * 2 ** (shift / QP_UNITS_PER_OCTAVE)

    held2 = plateaus["plateau2"]
    return {
        **{name: [line(i, held2[i] * 2 ** (-finer / QP_UNITS_PER_OCTAVE) if i in held2 else None,
                       None) for i in every] for name, finer in PLATEAU_FINER.items()},
        **{name: [line(i, held.get(i), None) for i in every] for name, held in plateaus.items()},
        **{name: [line(i, reused(i, strength), None) for i in every]
           for name, strength in REUSE_STRENGTH.items()},
        **{name: [line(i, None, shifted_key(i, scope, shift)) for i in every]
           for name, (scope, shift) in KEY_SHIFT.items()},
        "floor": [line(i, floors.get(i), None) for i in every],
        "ceil": [line(i, ceils.get(i), None) for i in every],
        "replay": [line(i, None, None) for i in every],
        "trailing": [line(i, smooth(i, 2 * half, 0) if i in log_level else None, None)
                     for i in every],
        "startup": [line(i, startup.get(i), None) for i in every],
        "smooth": [line(i, smooth(i, half, half) if i in log_level else None, None)
                   for i in every],
        "flat": [line(i, flat_level, None) for i in every],
        "flat_key": [line(i, flat_level, key_flat) for i in every],
    }


def percentile(values: list[float], share: float) -> float | None:
    """The `share` quantile (nearest rank) of `values`; None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def role_of(row: dict) -> str:
    """The frame's size-model role as the log shows it: key, the QP cadence's
    base (a negative cadence offset), or any other inter frame."""
    if row["type"] == "Key":
        return "key"
    return "base" if float(row["cadence_offset"]) < 0 else "inter"


def conformity(rows: list[dict], fps: float) -> dict:
    """How well one CBR point's frames and windows met their targets, from its
    rate log. Sizes are in frame budgets (target / fps), windows in targets.

    The target is a ceiling: a window or stream under it is allowed, over it
    is not, so the window columns are the p95 and the largest one-second
    window, and `under_target` is how far the whole stream ran below."""
    frames = coded(rows)
    if not frames:
        return {}
    bits = [int(r["bytes"]) * 8 for r in frames]
    target = float(frames[-1]["target_bps"])
    unit = target / fps
    inter = [r for r in frames if r["type"] == "Inter"]
    to_budget = [int(r["bytes"]) * 8 / float(r["budget_bits"]) for r in inter
                 if float(r["budget_bits"]) > 0]
    # Mini-GOPs of CYCLE_FRAMES inter frames from each key (the QP cadence's cycle, a
    # layered mini-GOP): the budget is met over a cycle, not by each frame of it.
    cycles: list[float] = []
    run: list[dict] = []
    for r in frames:
        run = [] if r["type"] == "Key" else run + [r]
        if len(run) == CYCLE_FRAMES:
            budget = sum(float(x["budget_bits"]) for x in run)
            if budget > 0:
                cycles.append(sum(int(x["bytes"]) * 8 for x in run) / budget)
            run = []
    second = max(1, round(fps))
    # One-second windows over every input frame (a held or dropped frame is 0 bits).
    per_input = [int(r["bytes"]) * 8 if r["type"] in ("Key", "Inter") else 0 for r in rows]
    windows = [sum(per_input[i:i + second]) / target * fps / second
               for i in range(0, max(1, len(per_input) - second + 1))]
    # Windows that hold no key: what the inter frames alone do.
    keys = [i for i, r in enumerate(rows) if r["type"] == "Key"]
    keyless = [w for i, w in enumerate(windows) if not any(i <= k < i + second for k in keys)]
    fullness = [float(r["buffer_fullness"]) for r in frames]
    misses: dict[str, list[float]] = {}
    for r in frames:
        if float(r["predicted_bits"]) > 0 and int(r["bytes"]) > 0:
            misses.setdefault(role_of(r), []).append(
                math.log2(int(r["bytes"]) * 8 / float(r["predicted_bits"])))
    result = {
        "rate_over_target": sum(bits) / len(rows) * fps / target,
        "inter_to_budget_p50": percentile(to_budget, 0.5),
        "inter_to_budget_p95": percentile(to_budget, 0.95),
        "inter_to_budget_max": max(to_budget, default=None),
        "inter_to_budget_log2_sd": statistics.pstdev(math.log2(max(v, 1e-6)) for v in to_budget)
        if len(to_budget) > 1 else None,
        "cycle_to_budget_p50": percentile(cycles, 0.5),
        "cycle_to_budget_p95": percentile(cycles, 0.95),
        "cycle_to_budget_log2_sd": statistics.pstdev(math.log2(max(v, 1e-6)) for v in cycles)
        if len(cycles) > 1 else None,
        "inter_frame_max_units": max((int(r["bytes"]) * 8 / unit for r in inter), default=None),
        "window_mean": statistics.fmean(windows),
        "window_p95": percentile(windows, 0.95),
        "window_max": max(windows),
        "keyless_window_p95": percentile(keyless, 0.95),
        "keyless_window_max": max(keyless, default=None),
        "fullness_max": max(fullness),
        "fullness_over_90": sum(1 for f in fullness if f > 0.9) / len(fullness),
        "buffer_overflows": int(frames[-1]["buffer_overflows"]),
        "recodes_per_1000": 1000 * int(frames[-1]["recoded_frames"]) / len(frames),
    }
    for role, values in misses.items():
        result[f"model_{role}_log2_bias"] = statistics.fmean(values)
        result[f"model_{role}_log2_sd"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        result[f"model_{role}_misses_2x"] = sum(1 for m in values if abs(m) >= 1) / len(values)
    return result


def log_stats(rows: list[dict], fps: float) -> dict:
    """The controller's behaviour on one point, from its rate log."""
    frames = coded(rows)
    bits = [int(r["bytes"]) * 8 for r in frames]
    inter = [r for r in frames if r["type"] == "Inter"]
    levels_ = [int(r["level_q"]) for r in inter]
    misses = [math.log2(int(r["bytes"]) * 8 / float(r["predicted_bits"]))
              for r in inter if float(r["predicted_bits"]) > 0 and int(r["bytes"]) > 0]
    keys = [r for r in frames if r["type"] == "Key"]
    def mean_ms(kind: str) -> float | None:
        """Mean wall time of coding a frame of `kind` (the box is shared: a ratio
        between arms of one run, not a timing)."""
        times = [int(r["encode_us"]) / 1000 for r in frames
                 if r["type"] == kind and r.get("encode_us")]
        return statistics.fmean(times) if times else None

    return {
        **conformity(rows, fps),
        "key_encode_ms": mean_ms("Key"),
        "inter_encode_ms": mean_ms("Inter"),
        "encode_ms_total": sum(int(r["encode_us"]) / 1000 for r in frames if r.get("encode_us")),
        "key_share": sum(int(r["bytes"]) * 8 for r in keys) / max(1, sum(bits)),
        "level_mean": statistics.fmean(levels_) if levels_ else None,
        "level_sd": statistics.pstdev(levels_) if len(levels_) > 1 else 0.0,
        "level_jump": statistics.fmean(abs(a - b) for a, b in zip(levels_, levels_[1:]))
        if len(levels_) > 1 else 0.0,
        "key_minus_level": int(keys[0]["qp"]) - statistics.fmean(levels_[:30])
        if keys and levels_ else None,
        "model_log2_bias": statistics.fmean(misses) if misses else None,
        "model_log2_sd": statistics.pstdev(misses) if len(misses) > 1 else None,
        "model_misses_4x": sum(1 for m in misses if abs(m) >= 2) / max(1, len(misses)),
        "recoded": int(frames[-1]["recoded_frames"]) if frames else 0,
    }


def clip_list(spec: str, paths, frames: int, per_class: int | None,
              rotation: str | None = None, classes: tuple[str, ...] = ()) -> list:
    if spec == "val":
        clips = natural_val(paths)
    elif spec.startswith("set:"):
        return clipset_sample(spec, paths, frames, rotation)
    else:
        draw = pool.draw(pool.load_pool(paths, levels.usable), pool.rotation_key(rotation),
                         DEFAULT_FRAMES)
        clips = [c for level in spec.split("+") for c in draw.levels[level]]
    if frames > DEFAULT_FRAMES:
        clips = levels.long_clips(clips, frames)
    if classes:
        clips = [c for c in clips if c.content_class in classes]
    if per_class is None:
        return clips
    kept: dict[str, list] = {}
    for clip in clips:
        kept.setdefault(clip.content_class, []).append(clip)
    return [c for group in kept.values() for c in group[:per_class]]


def clipset_sample(spec: str, paths, frames: int, rotation: str | None) -> list:
    """`set:NAME:N`: N clips of clip set NAME (a file of the clips directory), dealt
    round-robin over its content classes from a shuffle seeded by the set and
    `rotation`: a class-spread sample of a set the pool does not draw (a chart's)."""
    _, name, count = spec.split(":")
    clips = [c for c in load_clipset(paths.clips / f"{name}.json", paths.data_root).clips
             if levels.usable(c) and c.frames >= frames]
    by_class: dict[str, list] = {}
    for clip in sorted(clips, key=lambda c: c.name):
        by_class.setdefault(clip.content_class, []).append(clip)
    rng = random.Random(f"{name}:{rotation or 'sample'}")
    for group in by_class.values():
        rng.shuffle(group)
    sample: list = []
    while len(sample) < int(count) and any(by_class.values()):
        for cls in sorted(by_class):
            if by_class[cls] and len(sample) < int(count):
                sample.append(by_class[cls].pop())
    return sample


def bd_rows(curves: dict, clips: list, anchor: str, test: str) -> dict:
    """Per-clip J terms of `test` against `anchor` and the class-balanced BD-J."""
    per_clip = {c.name: {"class": c.content_class, **clip_terms(curves[(anchor, c.name)],
                                                              curves[(test, c.name)])}
                for c in clips if curves.get((anchor, c.name)) and curves.get((test, c.name))}
    full, low = (jreal.aggregate(per_clip, t) for t in ("full", "low"))
    j = None if full["balanced"] is None or low["balanced"] is None else \
        (full["balanced"] + low["balanced"]) / 2
    classes = {name: (full["classes"][name] + low["classes"].get(name, full["classes"][name])) / 2
               for name in full["classes"]}
    per_clip_j = {name: (row["full"] + row["low"]) / 2 for name, row in per_clip.items()
                  if row["full"] is not None and row["low"] is not None}
    return {"bd_j": j, "classes": classes, "scored": full["scored"], "per_clip": per_clip_j}


def best_arm(arms: dict[str, dict], clips: list) -> dict:
    """The class-balanced mean, over clips, of each clip's best J among `arms`
    (each a `bd_rows` against CBR) and 0 (CBR itself): what choosing among
    them per clip with hindsight gains. It picks on the measured J, so it
    includes selection noise: an upper bound, not an estimate."""
    by_class: dict[str, list[float]] = {}
    chosen: dict[str, int] = {}
    for clip in clips:
        options = {"cbr": 0.0, **{name: arm["per_clip"][clip.name] for name, arm in arms.items()
                                  if clip.name in arm["per_clip"]}}
        pick = min(options, key=options.get)
        chosen[pick] = chosen.get(pick, 0) + 1
        by_class.setdefault(clip.content_class, []).append(options[pick])
    classes = {name: statistics.fmean(values) for name, values in by_class.items()}
    return {"bd_j": statistics.fmean(classes.values()) if classes else None,
            "classes": classes, "chosen": chosen}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", required=True)
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--clips", default="val")
    parser.add_argument("--work", help="schedule files and rate logs (default: a "
                        "directory beside --json)")
    parser.add_argument("--json", required=True)
    parser.add_argument("--workers", type=int, default=None,
                        help=f"default: {MAX_WORKERS_ENV} when set (a rented box), else {WORKERS}")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--key-seconds", type=float, default=None,
                        help="periodic keys (default: one key)")
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--classes", default="", help="only these content classes (a,b,...)")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--preset", default="", help="a shipped preset (default: medium)")
    parser.add_argument("--rotation", default=None, help="the pool draw's key (default: today)")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME:P=V,...",
                        help="a live CBR arm: the preset plus these parameters")
    parser.add_argument("--key-map", metavar="ARM:S1,S2,...",
                        help="the inheritance oracle: ARM's keys with hindsight MB-tree "
                        "offsets at these strengths (cN: centred; nceval.key_inherit), "
                        "against --base-arm")
    parser.add_argument("--base-arm", metavar="NAME",
                        help="also report every other arm against this --arm")
    args = parser.parse_args(argv)
    variants = [v for v in args.variants.split(",") if v]
    args.workers = args.workers or (max_workers() if os.environ.get(MAX_WORKERS_ENV) else WORKERS)
    args.work = args.work or str(Path(args.json).with_suffix("")) + "-work"
    paths, codec = load_paths(), Codec(args.codec)
    encoder = f"nanocodec-{codec.value}" + (f"@{args.preset}" if args.preset else "")
    base = replace(own_family(encoder, Path(args.binary)), cqp=True)
    clips = clip_list(args.clips, paths, args.frames, args.per_class, args.rotation,
                      tuple(c for c in args.classes.split(",") if c))
    runner = Runner(paths, EVALUATION_GATE, args.workers, over_budget=True)
    benches = {}

    def bench_of(clip) -> "Bench":
        """The clip's bench: its window and, with `--key-seconds`, its key period."""
        intra = round(args.key_seconds * clip.fps) if args.key_seconds else None
        frames = min(args.frames, clip.frames)
        if (frames, intra) not in benches:
            benches[(frames, intra)] = (runner.bench(frames, intra) if intra
                                        else runner.bench(frames))
        return benches[(frames, intra)]
    cbr, cqp = {}, {}
    for group in {id(bench_of(c)): [d for d in clips if bench_of(d) is bench_of(c)]
                  for c in clips}.values():
        group_cbr, group_cqp = measure_all(bench_of(group[0]), [base], group, args.workers,
                                           EVALUATION_GATE, LADDER_KBPS)
        cbr |= group_cbr
        cqp |= group_cqp
    print(f"measured CBR and fixed QP: {len(cbr)} + {len(cqp)} points", flush=True)
    work = Path(args.work) / (f"{codec.value}-{args.preset}" if args.preset else codec.value)
    work.mkdir(parents=True, exist_ok=True)
    by_name = {c.name: c for c in clips}

    def logged(key: tuple, family: Family = base, tag: str = "") -> list[dict]:
        """The rate log of `family`'s CBR point `key` (coded once more for it)."""
        _, clip, kbps = key
        log = work / f"{clip}-{kbps}{tag}.rate.csv"
        if not log.exists():
            stream = work / f"{clip}-{kbps}{tag}.bin"
            bench_of(by_name[clip]).encode(Point(family, by_name[clip], kbps=kbps), stream, log)
            stream.unlink()
        return read_log(log)

    def inheritance_maps(key: tuple, family: Family, strengths: list[str]) -> dict:
        """`--key-offsets` text per strength for `family`'s CBR point `key`, from
        the analysis of a first coding (nceval.key_inherit)."""
        _, clip, kbps = key
        analysis = work / f"{clip}-{kbps}-{family.name}.analysis"
        if not analysis.exists():
            pass1 = replace(family, extra=family.extra + ("--analysis-log", str(analysis)))
            stream = work / f"{clip}-{kbps}-{family.name}-pass1.bin"
            bench_of(by_name[clip]).encode(Point(pass1, by_name[clip], kbps=kbps), stream)
            stream.unlink()
        return key_inherit.key_maps(key_inherit.read_analysis_log(analysis), strengths)["texts"]

    with ThreadPoolExecutor(args.workers) as threads:
        logs = dict(zip(cbr, threads.map(logged, cbr)))
    jobs, owners = [], []
    for key, rows in logs.items():
        _, clip, kbps = key
        clip_variants = schedules(rows, codec, by_name[clip].fps)
        for variant in variants:
            lines = clip_variants[variant]
            text = "\n".join(lines) + "\n"
            path = work / f"{clip}-{kbps}-{variant}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.txt"
            path.write_text(text)
            family = replace(base, name=f"{base.name}~oracle-{variant}", variant=True, cqp=False,
                             extra=base.extra + ("--qp-replay", str(path)))
            # Scored like every evaluation point (our reconstruction, no decode):
            # conformance is the landing check's.
            jobs.append((bench_of(by_name[clip]), Point(family, by_name[clip], kbps=kbps,
                                      check=uniform_check(EVALUATION_GATE))))
            owners.append((variant, clip))
    arm_families = {}
    for spec in args.arm:
        name, _, pairs = spec.partition(":")
        family = replace(base, name=f"{base.name}~arm-{name}", variant=True, cqp=False,
                         extra=base.extra + param_args(pairs.split(",")))
        arm_families[f"arm-{name}"] = family
        for (_, clip, kbps) in cbr:
            jobs.append((bench_of(by_name[clip]), Point(family, by_name[clip], kbps=kbps,
                                      check=uniform_check(EVALUATION_GATE))))
            owners.append((f"arm-{name}", clip))
    arms = [f"arm-{spec.partition(':')[0]}" for spec in args.arm]
    key_maps = []
    if args.key_map:  # the inheritance oracle: a first coding, then keys with its maps
        map_arm, _, strengths = args.key_map.partition(":")
        if f"arm-{map_arm}" not in arm_families:
            raise SystemExit(f"--key-map {map_arm}: no such --arm")
        family = arm_families[f"arm-{map_arm}"]
        key_maps = [f"map-{s}" for s in strengths.split(",")]
        with ThreadPoolExecutor(args.workers) as threads:
            texts = dict(zip(cbr, threads.map(lambda key: inheritance_maps(
                key, family, strengths.split(",")), cbr)))
        for (_, clip, kbps), maps in texts.items():
            for strength, text in maps.items():
                path = work / f"{clip}-{kbps}-keymap-{map_arm}-{strength}.txt"
                path.write_text(text)
                variant = f"map-{strength}"
                mapped = replace(family, name=f"{base.name}~{variant}",
                                 extra=family.extra + ("--key-offsets", str(path)))
                jobs.append((bench_of(by_name[clip]), Point(mapped, by_name[clip], kbps=kbps,
                                          check=uniform_check(EVALUATION_GATE))))
                owners.append((variant, clip))
    results = runner.run_jobs(jobs)
    curves: dict = {}
    for (family, clip, _), point in cbr.items():
        curves.setdefault(("cbr", clip), []).append(point)
    for (family, clip, _), point in cqp.items():
        curves.setdefault(("qp", clip), []).append(point)
    for (variant, clip), result in zip(owners, results):
        if result is not None:
            curves.setdefault((variant, clip), []).append(result)
    curves = {k: sorted(v, key=lambda p: p["kbps"]) for k, v in curves.items()}
    measured = {"qp", "cbr", *variants}
    chain = [v for v in ("qp", "flat_key", "flat", "smooth", "replay", "cbr") if v in measured]
    pairs = list(zip(chain, chain[1:])) + [
        pair for pair in (("replay", "trailing"), ("smooth", "trailing"), ("replay", "startup"),
                          ("replay", "flat"),
                          *(("replay", name) for name in (*PLATEAU_SECONDS, *PLATEAU_FINER, *REUSE_STRENGTH,
                                                          *KEY_SHIFT,
                                                          "floor", "ceil")))
        if set(pair) <= measured]
    arm_rows = {arm: bd_rows(curves, clips, "cbr", arm) for arm in arms}
    # `--base-arm`: every other arm also against that arm (a change on a changed base)
    base_arm = f"arm-{args.base_arm}" if args.base_arm else None
    if base_arm and base_arm not in arms:
        raise SystemExit(f"--base-arm {args.base_arm}: no such --arm")
    arm_vs_base = {arm: bd_rows(curves, clips, base_arm, arm)
                   for arm in [*arms, *key_maps] if base_arm and arm != base_arm}
    report = {"codec": codec.value, "clips": args.clips, "binary": str(args.binary),
              "frames": args.frames, "key_seconds": args.key_seconds,
              "total": bd_rows(curves, clips, "qp", "cbr"),
              "vs_qp": {v: bd_rows(curves, clips, "qp", v) for v in chain[1:]},
              "preset": args.preset or "medium", "rotation": pool.rotation_key(args.rotation),
              "steps": {f"{b}->{a}": bd_rows(curves, clips, b, a) for b, a in pairs},
              "arms": arm_rows, "best_arm": best_arm(arm_rows, clips) if arm_rows else None,
              "base_arm": base_arm, "arms_vs_base": arm_vs_base}
    stats: dict = {}
    for (_, clip, kbps), rows in logs.items():
        cls = by_name[clip].content_class
        for name, value in log_stats(rows, by_name[clip].fps).items():
            if value is not None:
                stats.setdefault(cls, {}).setdefault(name, []).append(value)
    report["log_stats"] = {cls: {k: statistics.fmean(v) for k, v in values.items()}
                           for cls, values in stats.items()}
    # The same table for each live arm, from its own rate logs: what the arm does to
    # conformity, beside what it does to J.
    report["arm_log_stats"] = {}
    for arm, family in arm_families.items():
        with ThreadPoolExecutor(args.workers) as threads:
            arm_logs = dict(zip(cbr, threads.map(lambda key: logged(key, family, f"-{arm}"), cbr)))
        arm_stats: dict = {}
        for (_, clip, _), rows in arm_logs.items():
            for name, value in log_stats(rows, by_name[clip].fps).items():
                if value is not None:
                    arm_stats.setdefault(by_name[clip].content_class, {}).setdefault(
                        name, []).append(value)
        report["arm_log_stats"][arm] = {cls: {k: statistics.fmean(v) for k, v in values.items()}
                                        for cls, values in arm_stats.items()}
        report.setdefault("arm_logs", {})[arm] = {
            f"{clip}@{kbps}": {"class": by_name[clip].content_class, "fps": by_name[clip].fps,
                               "columns": LOG_COLUMNS,
                               "rows": [[row.get(c, "") for c in LOG_COLUMNS] for row in rows]}
            for (_, clip, kbps), rows in arm_logs.items()}
    rates = {}
    for (variant, clip), result in zip(owners, results):
        if result is not None:
            rates.setdefault(variant, []).append(result["kbps"] / result["target_kbps"])
    report["rate_over_target"] = {v: statistics.median(r) for v, r in rates.items()}
    # Per point: what each variant achieved, and whether its buffer stayed legal.
    def point_row(variant: str, clip: str, result: dict) -> dict:
        return {"variant": variant, "clip": clip, "class": by_name[clip].content_class,
                **{k: result.get(k) for k in ("target_kbps", "kbps", "psnr_yuv", "vmaf_v1",
                                              "ms_ssim", "max_fullness", "max_window_ratio")},
                "worst_psnr_yuv": result.get("worst_psnr_yuv"),
                "frame_psnr_y": [round(v, 2) for v in result.get("frame_psnr_y") or ()],
                "frame_bits": list(result.get("frame_bits") or ()),
                "frame_key": [int(k) for k in result.get("frame_key") or ()]}
    report["points"] = [point_row(variant, clip, result)
                        for (variant, clip), result in zip(owners, results)
                        if result is not None]
    report["points"] += [point_row(mode, clip, result)
                         for mode, measured_ in (("cbr", cbr), ("qp", cqp))
                         for (_, clip, _), result in measured_.items()]
    report["logs"] = {f"{clip}@{kbps}": {"class": by_name[clip].content_class,
                                         "fps": by_name[clip].fps, "columns": LOG_COLUMNS,
                                         "rows": [[row.get(c, "") for c in LOG_COLUMNS]
                                                  for row in rows]}
                      for (_, clip, kbps), rows in logs.items()}
    Path(args.json).write_text(json.dumps(report, indent=1))
    def brief(row: dict | None) -> dict | None:
        return row and {k: v for k, v in row.items() if k != "per_clip"}
    print(json.dumps({"total": brief(report["total"]),
                      **{part: {k: brief(v) for k, v in report[part].items()}
                         for part in ("vs_qp", "steps", "arms", "arms_vs_base")},
                      "best_arm": report["best_arm"]}, indent=1))
    print(json.dumps(report["log_stats"], indent=1))
    print(json.dumps(report["rate_over_target"], indent=1))


if __name__ == "__main__":
    main()
