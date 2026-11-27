"""Configuration arms in CBR and fixed QP, kept per frame: the oracle for choosing a
configuration from content (docs/analysis/learned-decisions-2026-11-22.md, Part 2).

    python3 -m nceval.arm_oracle --binary BIN --codec av1 --clips adopt --per-class 2 \\
        --frames 600 --key-seconds 10 --arm cad8:rate.cadence_period=8,... --json OUT

Every arm (the preset plus its parameters; `base` is the preset itself) is measured
like a ladder family: CBR at the ladder rates, then fixed QP at quantizers that span
each CBR curve (`cbr.measure_all`). The report keeps every point's per-frame bits,
PSNR-YUV, VMAF v1 and key flags, so the offline analysis can score each key period
apart: a per-clip choice made on one key period and scored on the next has no
selection noise in it, unlike `rc_oracle`'s hindsight `best arm per clip`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from . import pool
from .cbr import measure_all
from .codecs import Codec
from .config import load_paths
from .gatepolicy import EVALUATION_GATE
from .rc_oracle import WORKERS, bd_rows, clip_list
from .realuse import LADDER_KBPS, MAX_WORKERS_ENV, Runner, max_workers, own_family, param_args

#: Per-point scalar terms kept beside the per-frame series.
POINT_TERMS = ("target_kbps", "kbps", "qp", "psnr_yuv", "vmaf_v1", "ms_ssim")
#: Per-frame series kept for the per-key-period scoring.
FRAME_SERIES = ("frame_bits", "frame_psnr_yuv", "frame_vmaf_v1", "frame_key")


def arm_families(base, specs: list[str]) -> dict:
    """`base` plus one ladder family per `NAME:p=v,...` arm."""
    families = {"base": base}
    for spec in specs:
        name, _, pairs = spec.partition(":")
        families[name] = replace(base, name=f"{base.name}~arm-{name}",
                                 extra=base.extra + param_args(pairs.split(",")))
    return families


def point_row(arm: str, mode: str, clip, result: dict) -> dict:
    return {"arm": arm, "mode": mode, "clip": clip.name, "class": clip.content_class,
            "source_kbps": clip.source_kbps, **{k: result.get(k) for k in POINT_TERMS},
            **{k: list(result.get(k) or ()) for k in FRAME_SERIES}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", required=True)
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--clips", default="adopt")
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--key-seconds", type=float, default=10.0)
    parser.add_argument("--preset", help="a shipped preset (default: medium)")
    parser.add_argument("--rotation", help="the pool draw's key (default: today)")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME:P=V,...")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    workers = args.workers or (max_workers() if os.environ.get(MAX_WORKERS_ENV) else WORKERS)
    paths, codec = load_paths(), Codec(args.codec)
    encoder = f"nanocodec-{codec.value}" + (f"@{args.preset}" if args.preset else "")
    base = replace(own_family(encoder, Path(args.binary)), cqp=True)
    families = arm_families(base, args.arm)
    clips = clip_list(args.clips, paths, args.frames, args.per_class, args.rotation)
    runner = Runner(paths, EVALUATION_GATE, workers, over_budget=True)
    rows, curves = [], {}
    benches: dict = {}
    for clip in clips:
        key = (min(args.frames, clip.frames), round(args.key_seconds * clip.fps))
        benches.setdefault(key, []).append(clip)
    for (frames, intra), group in benches.items():
        cbr, cqp = measure_all(runner.bench(frames, intra), list(families.values()), group,
                               workers, EVALUATION_GATE, LADDER_KBPS)
        by_family = {f.name: arm for arm, f in families.items()}
        by_name = {c.name: c for c in group}
        for mode, points in (("cbr", cbr), ("qp", cqp)):
            for (family, clip, _), result in points.items():
                arm = by_family[family]
                rows.append(point_row(arm, mode, by_name[clip], result))
                curves.setdefault((f"{mode}:{arm}", clip), []).append(result)
    curves = {k: sorted(v, key=lambda p: p["kbps"]) for k, v in curves.items()}
    whole = {mode: {arm: bd_rows(curves, clips, f"{mode}:base", f"{mode}:{arm}")
                    for arm in families if arm != "base"} for mode in ("cbr", "qp")}
    report = {"codec": codec.value, "preset": args.preset or "medium", "clips": args.clips,
              "rotation": pool.rotation_key(args.rotation), "frames": args.frames,
              "key_seconds": args.key_seconds, "arms": args.arm, "whole_clip": whole,
              "points": rows}
    Path(args.json).write_text(json.dumps(report))
    print(json.dumps({mode: {arm: {k: v for k, v in row.items() if k != "per_clip"}
                             for arm, row in arms.items()} for mode, arms in whole.items()},
                     indent=1))


if __name__ == "__main__":
    main()
