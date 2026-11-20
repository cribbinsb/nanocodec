#!/usr/bin/env python3
"""Targeted periodic-key and worst-frame check of one preset change.

Keys every `--key-seconds` (2 s), `--frames` (600) frames from frame 0, on 6
static and 6 moving pool clips (a fixed hash draw, or `--clips`), at the two
periodic-key CBR rungs (`--all-rungs`: every rung). Per clip: the
periodic-key pulse delta (key against the P frames before it, dB) and the
BD terms (PSNR-YUV, VMAF v1, worst-1 % PSNR-Y and VMAF) of the candidate
against the anchor, then their means. This ranks a change's key behaviour
quickly; adoption still comes from the pool levels and the final-check.

The arms (`arms`):
- anchor: `--anchor-binary` (default `--binary`) with the preset's values
  and `--anchor-param`s;
- candidate: `--binary` with the preset's values and `--param`s.
Each arm is built from the preset alone: the candidate never inherits the
anchor's `--anchor-param`s (the first copies did, so `--anchor-param` alone
measured one configuration against itself). Identical arms are refused.

    python3 tools/dev/pulsecheck.py --codec vvc --binary BIN \\
        --anchor-param vvc.tx64=0 --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import jreal, pool  # noqa: E402
from nceval.cbr import Family  # noqa: E402
from nceval.config import load_paths  # noqa: E402
from nceval.gatepolicy import GatePolicy  # noqa: E402
from nceval.realuse import PERIODIC_RUNGS, Runner, own_family, param_args  # noqa: E402

#: Clips per class in the default draw.
CLIPS_PER_CLASS = 6
#: Seed of the default draw (the first runs', so their numbers compare).
DRAW_SEED = "2026-11-19"
#: The per-clip terms reported (jreal.compare_clip keys).
TERMS = ("d_periodic_pulse_psnr_y", "bd_psnr_yuv", "bd_vmaf_v1", "bd_worst_psnr_y",
         "bd_worst_vmaf_v1")
#: Suffix of the candidate arm's family name.
CANDIDATE_SUFFIX = "+cand"


def encoder_name(codec: str, preset: str | None) -> str:
    """The nceval family name of our codec at `preset` (medium: the Default)."""
    return f"nanocodec-{codec}" + (f"@{preset}" if preset and preset != "medium" else "")


def arms(name: str, binary: Path, anchor_binary: Path | None, params: list[str],
         anchor_params: list[str], family=own_family) -> tuple[Family, Family]:
    """(anchor, candidate), each from the preset's own values (module docs)."""
    anchor_base = family(name, Path(anchor_binary or binary))
    candidate_base = family(name, Path(binary))
    anchor = replace(anchor_base, extra=anchor_base.extra + param_args(anchor_params))
    candidate = replace(candidate_base, name=name + CANDIDATE_SUFFIX,
                        extra=candidate_base.extra + param_args(params))
    if anchor.tool == candidate.tool and anchor.extra == candidate.extra:
        raise SystemExit("pulsecheck: the anchor and the candidate are the same encoder and "
                         "parameters; give --param, --anchor-param or --anchor-binary")
    return anchor, candidate


def draw(clips: list, seed: str = DRAW_SEED, per_class: int = CLIPS_PER_CLASS) -> list:
    """`per_class` static and `per_class` other clips, by a seeded hash order."""
    def pick(subset):
        return sorted(subset, key=lambda c: hashlib.sha256(
            f"{seed}{c.name}".encode()).hexdigest())[:per_class]
    return (pick([c for c in clips if c.content_class == "static"])
            + pick([c for c in clips if c.content_class != "static"]))


def mean_terms(rows: dict) -> dict:
    """Each term's mean over the clips that have it, with that count (None: none do)."""
    summary = {}
    for term in TERMS:
        values = [r[term] for r in rows.values() if isinstance(r.get(term), (int, float))]
        summary[term] = (round(statistics.mean(values), 3), len(values)) if values else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--codec", required=True)
    parser.add_argument("--preset", help="preset stem (default: medium, the Default)")
    parser.add_argument("--binary", type=Path, required=True, help="the candidate's build")
    parser.add_argument("--anchor-binary", type=Path, help="the anchor's build (default --binary)")
    parser.add_argument("--param", action="append", default=[], help="candidate name=value")
    parser.add_argument("--anchor-param", action="append", default=[], help="anchor name=value")
    parser.add_argument("--json", type=Path, help="write the rows and the summary here")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clips", help="comma-separated pool clip names instead of the draw")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--key-seconds", type=float, default=2.0)
    parser.add_argument("--all-rungs", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    name = encoder_name(args.codec, args.preset)
    anchor, candidate = arms(name, args.binary, args.anchor_binary, args.param,
                             args.anchor_param)
    every = {c.name: replace(c, start_frame=0) for c in pool.load_pool(paths)}
    if args.clips:
        chosen = [every[clip] for clip in args.clips.split(",")]
    else:
        chosen = draw([c for c in every.values() if c.frames >= args.frames])
    runner = Runner(paths, GatePolicy("off"), args.workers, over_budget=False)
    curves = runner.measure("pulse check", [anchor, candidate], chosen, args.frames,
                            None if args.all_rungs else PERIODIC_RUNGS, args.key_seconds)
    rows = {}
    for clip in chosen:
        x, y = curves.get((anchor.name, clip.name)), curves.get((candidate.name, clip.name))
        if not x or not y:
            print(f"{clip.name}: no curve for one arm, skipped", flush=True)
            continue
        x, y, _, _ = jreal.pair_rungs(x, y)
        terms = jreal.compare_clip(x, y, min_points=2)
        rows[clip.name] = {"class": clip.content_class, **{t: terms.get(t) for t in TERMS}}
        shown = {t: round(v, 3) if isinstance(v, float) else v for t, v in rows[clip.name].items()
                 if t != "class"}
        print(clip.name, clip.content_class, shown, flush=True)
    summary = mean_terms(rows)
    print("SUMMARY", summary, flush=True)
    if args.json:
        args.json.write_text(json.dumps({
            "codec": args.codec, "preset": args.preset, "params": args.param,
            "anchor_params": args.anchor_param,
            "anchor_binary": str(args.anchor_binary or args.binary), "binary": str(args.binary),
            "rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
