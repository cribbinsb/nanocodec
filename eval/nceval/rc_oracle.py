"""Decompose what CBR rate control costs against fixed QP, by replaying quantizer sequences.

    python3 -m nceval.rc_oracle --binary BIN --codec av1 [--clips val|quick|adopt] --json OUT

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
- `smooth`: each inter frame's level replaced by its centred one-second
  mean in the step domain (the cadence and layer offsets kept): what the
  level's frame-to-frame jitter costs, including the size model's misses;
- `flat`: every inter level at the sequence's mean step (start-up transient,
  content-driven drift and jitter all gone), the key as logged;
- `flat_key`: `flat` with the key at the fixed-QP key's step ratio to the
  inter level (`key_qp.rs`; H.264 `keyframe_qp_offset`; AV1 `keyframe_step_permille`).

The chain CBR -> replay -> smooth -> flat -> flat_key -> fixed QP adds up to
the whole cost; what is left between `flat_key` and fixed QP is what the
rate-control-only tools (the reuse term's level, held frames' budgets) and
the remaining schedule differences (cadence strength) do. Per-point
controller statistics from the logs (size-model error per frame role,
level jitter, key share, re-codes) come with each class.

`--clips`: `val` the natural VAL clips (rc_cost's set), or a pool level of
the day's draw (`quick`, `adopt`; not adoption evidence here, only a
broader sample per class). Every point is cached by the shared engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import jreal, levels, pool
from .cbr import Family, Point, measure_all
from .codecs import Codec
from .config import load_paths
from .gatepolicy import EVALUATION_GATE
from .rc_cost import clip_terms
from .realuse import DEFAULT_FRAMES, LADDER_KBPS, Runner, natural_val, own_family

WORKERS = 8
VARIANTS = ("replay", "trailing", "smooth", "flat", "flat_key")
#: Frames of the centred mean of `smooth`: one second at 30 fps.
SMOOTH_FRAMES = 31
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
#: Fixed-QP key step relative to the configured quantizer, per codec: H.264
#: `h264.keyframe_qp_offset` 6 QP; HEVC `key_qp.rs` 12 QP at a 60-frame
#: period +3 QP per factor of 5 (at most +3: one key in 150 frames); AV1
#: `av1.keyframe_step_permille` 240.
FIXED_KEY_STEP_RATIO = {Codec.H264: 2 ** (-6 / 6), Codec.HEVC: 2 ** (-15 / 6), Codec.AV1: 0.24}


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


def schedules(rows: list[dict], codec: Codec) -> dict[str, list[str]]:
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
    return {
        "replay": [line(i, None, None) for i in every],
        "trailing": [line(i, smooth(i, 2 * half, 0) if i in log_level else None, None)
                     for i in every],
        "smooth": [line(i, smooth(i, half, half) if i in log_level else None, None)
                   for i in every],
        "flat": [line(i, flat_level, None) for i in every],
        "flat_key": [line(i, flat_level, key_flat) for i in every],
    }


def log_stats(rows: list[dict]) -> dict:
    """The controller's behaviour on one point, from its rate log."""
    frames = coded(rows)
    bits = [int(r["bytes"]) * 8 for r in frames]
    inter = [r for r in frames if r["type"] == "Inter"]
    levels_ = [int(r["level_q"]) for r in inter]
    misses = [math.log2(int(r["bytes"]) * 8 / float(r["predicted_bits"]))
              for r in inter if float(r["predicted_bits"]) > 0 and int(r["bytes"]) > 0]
    keys = [r for r in frames if r["type"] == "Key"]
    return {
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


def clip_list(spec: str, paths) -> list:
    if spec == "val":
        return natural_val(paths)
    draw = pool.draw(pool.load_pool(paths, levels.usable), pool.rotation_key(None),
                     DEFAULT_FRAMES)
    return draw.levels[spec]


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
    return {"bd_j": j, "classes": classes, "scored": full["scored"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", required=True)
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--clips", default="val")
    parser.add_argument("--work", required=True, help="schedule files and rate logs")
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    paths, codec = load_paths(), Codec(args.codec)
    base = replace(own_family(f"nanocodec-{codec.value}", Path(args.binary)), cqp=True)
    clips = clip_list(args.clips, paths)
    runner = Runner(paths, EVALUATION_GATE, WORKERS, over_budget=True)
    bench = runner.bench(DEFAULT_FRAMES)
    cbr, cqp = measure_all(bench, [base], clips, WORKERS, EVALUATION_GATE, LADDER_KBPS)
    print(f"measured CBR and fixed QP: {len(cbr)} + {len(cqp)} points", flush=True)
    work = Path(args.work) / codec.value
    work.mkdir(parents=True, exist_ok=True)
    by_name = {c.name: c for c in clips}

    def logged(key: tuple) -> list[dict]:
        _, clip, kbps = key
        log = work / f"{clip}-{kbps}.rate.csv"
        if not log.exists():
            stream = work / f"{clip}-{kbps}.bin"
            bench.encode(Point(base, by_name[clip], kbps=kbps), stream, log)
            stream.unlink()
        return read_log(log)

    with ThreadPoolExecutor(WORKERS) as threads:
        logs = dict(zip(cbr, threads.map(logged, cbr)))
    jobs, owners = [], []
    for key, rows in logs.items():
        _, clip, kbps = key
        for variant, lines in schedules(rows, codec).items():
            text = "\n".join(lines) + "\n"
            path = work / f"{clip}-{kbps}-{variant}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.txt"
            path.write_text(text)
            family = replace(base, name=f"{base.name}~oracle-{variant}", variant=True, cqp=False,
                             extra=base.extra + ("--qp-replay", str(path)))
            jobs.append((bench, Point(family, by_name[clip], kbps=kbps)))
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
    chain = ("qp", "flat_key", "flat", "smooth", "replay", "cbr")
    extra = {"replay->trailing": bd_rows(curves, clips, "replay", "trailing"),
             "smooth->trailing": bd_rows(curves, clips, "smooth", "trailing")}
    report = {"codec": codec.value, "clips": args.clips, "binary": str(args.binary),
              "total": bd_rows(curves, clips, "qp", "cbr"),
              "vs_qp": {v: bd_rows(curves, clips, "qp", v) for v in chain[1:]},
              "steps": {f"{b}->{a}": bd_rows(curves, clips, b, a)
                        for b, a in zip(chain, chain[1:])} | extra}
    stats: dict = {}
    for (_, clip, kbps), rows in logs.items():
        cls = by_name[clip].content_class
        for name, value in log_stats(rows).items():
            if value is not None:
                stats.setdefault(cls, {}).setdefault(name, []).append(value)
    report["log_stats"] = {cls: {k: statistics.fmean(v) for k, v in values.items()}
                           for cls, values in stats.items()}
    rates = {}
    for (variant, clip), result in zip(owners, results):
        if result is not None:
            rates.setdefault(variant, []).append(result["kbps"] / result["target_kbps"])
    report["rate_over_target"] = {v: statistics.median(r) for v, r in rates.items()}
    Path(args.json).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("total", "vs_qp", "steps")}, indent=1))
    print(json.dumps(report["log_stats"], indent=1))
    print(json.dumps(report["rate_over_target"], indent=1))


if __name__ == "__main__":
    main()
