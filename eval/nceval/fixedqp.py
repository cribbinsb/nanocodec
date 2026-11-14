"""Fixed-QP coding efficiency on the graded pool: the gap map and paired ablations.

    python3 -m nceval.fixedqp gap --binary BIN --families nanocodec-av1,svtav1-m12,... \\
        [--pairs nanocodec-av1:svtav1-m12,...] [--rotation KEY] [--level quick] [--clips N]
    python3 -m nceval.fixedqp ab --codec av1 --binary BIN [--preset av1-fast] \\
        --param a.b=1 [--anchor-param ...] [--rotation KEY] [--level quick] [--clips N]

Rate control costs our encoders 10-22 % BD-J against their own fixed QP
(nceval.rc_cost); this module measures the other half, each codec's pure coding
efficiency, with rate control out of the picture.

Clips come from the pool draw (nceval.pool) of a rotation key, by default
`fixedqp-<UTC date>`: stratified and family-disjoint like the screens' levels,
but a different draw from the day's adoption levels, so ranking ideas here does
not spend the clips that later judge them. Nothing here picks clips by name.

Every family is coded at fixed QP (our `--qp`, one key frame; anchors in their
fixed-quantiser modes, AQ off) at CURVE_POINTS quantizers per clip spanning the
real-use ladder's rates (nceval.realuse.LADDER_KBPS, scaled by pixel rate, the
top capped at half the source bitrate). The quantizers are found by a walk from
the coarsest quantizer on cached rate probes (no scoring), so the finest, slowest
encodes are made only when a goal rate needs them.

- `gap`: BD-J (nceval.jreal: the mean of the full and lower-half BD-rates over
  PSNR-YUV, VMAF v1 and MS-SSIM) and each metric's BD-rate of every requested
  pair, class-balanced, with a bootstrap interval over clips (nceval.interval).
- `ab`: candidates (our encoder with --param, or each line of --sets) against
  one anchor (the same binary or --anchor-binary, the preset's values) at the
  ANCHOR's quantizers: matched QP, paired per clip; all in one worker pool. This ranks tools quickly and without rate-control
  noise; it is not adoption evidence (nceval.screen --clips adopt is).
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from . import buildinfo, interval, pool
from .cbr import CQP_SPAN_MARGIN, Bench, Family, clip_targets, curve_points, preset_anchor
from .clips import Clip
from .codecs import Codec
from .config import Paths, load_paths
from .gatepolicy import EVALUATION_GATE
from .levels import usable
from .rc_cost import clip_terms
from .realuse import LADDER_KBPS, PRESETS_DIR, Runner, own_family, param_args

#: Rotation keys of this program's draws: never the day's adoption key.
ROTATION_PREFIX = "fixedqp-"
#: Quantizers per curve (the BD-rate minimum is three; four as the chart).
CURVE_POINTS = 4
FRAMES = 150
WORKERS = 8
OWN_PREFIX = "nanocodec-"
#: The J terms and the per-metric BD-rates reported for each pair.
TERMS = ("full", "low", "psnr_yuv", "vmaf_v1", "ms_ssim")


def default_rotation() -> str:
    return ROTATION_PREFIX + datetime.now(timezone.utc).strftime("%Y-%m-%d")


def draw_clips(paths: Paths, rotation: str, level: str, count: int | None,
               frames: int = FRAMES) -> list[Clip]:
    """The first `count` clips of one level of the rotation's pool draw (strata interleaved)."""
    draw = pool.draw(pool.load_pool(paths, usable), rotation, frames)
    clips = draw.levels[level]
    return clips[:count] if count else clips


def family_of(name: str, binary: Path, paths: Paths) -> Family:
    """`nanocodec-<codec>[@preset]` (ours) or an anchor family name (cbr.preset_anchor)."""
    if name.startswith(OWN_PREFIX):
        return own_family(name, binary)
    return preset_anchor(name, paths, cqp=True)


class Benches:
    """The bench for each clip: a clip shorter than the run is measured whole,
    as the screens do (realuse.Runner.points)."""

    def __init__(self, paths: Paths, frames: int):
        self.runner = Runner(paths, EVALUATION_GATE, WORKERS, over_budget=True)
        self.frames = frames

    def of(self, clip: Clip) -> Bench:
        return self.runner.bench(min(self.frames, clip.frames))


class Quantizers:
    """Per (family, clip): CURVE_POINTS quantizers spanning the ladder's rates.

    Rate is monotonic in the quantizer, so each goal rate is bracketed by
    walking the family's grid from its coarsest end on cached rate probes
    (Bench.price), and the nearer grid neighbour in log rate is taken.
    """

    def __init__(self, benches: Benches):
        self.benches = benches

    def goals(self, clip: Clip) -> list[float]:
        targets = clip_targets(clip, LADDER_KBPS)
        low = min(targets) / CQP_SPAN_MARGIN
        high = min(max(targets) * CQP_SPAN_MARGIN, clip.max_test_kbps)
        return [low * (high / low) ** (i / (CURVE_POINTS - 1)) for i in range(CURVE_POINTS)]

    def choose(self, family: Family, clip: Clip) -> list[int]:
        grid = list(family.quantizer_grid)
        rates: dict[int, float] = {}

        def rate(index: int) -> float:
            q = grid[index]
            if q not in rates:
                rates[q] = self.benches.of(clip).price(family, clip, q)
            return rates[q]
        chosen = set()
        # Goals from the lowest rate up, each walk starting where the last
        # ended: the finest quantizers (the slowest encodes, and rates far
        # above the ladder) are probed only if a goal needs them.
        index = len(grid) - 1  # rate falls as the index rises
        for goal in self.goals(clip):
            while index > 0 and rate(index) < goal:
                index -= 1
            if rate(index) < goal:  # even the finest quantizer stays below the goal
                chosen.add(grid[index])
                continue
            # rate(index) >= goal; rate(index + 1) < goal when it exists
            candidates = [i for i in (index, index + 1) if i < len(grid)]
            near = min(candidates, key=lambda i: abs(math.log(rate(i) / goal)))
            # never above the clip's rate cap (encode at <= half the source rate)
            if rate(near) > clip.max_test_kbps and near + 1 < len(grid):
                near += 1
            chosen.add(grid[near])
        return sorted(chosen)


def measure(benches: Benches, families: list[Family], clips: list[Clip],
            quantizers: dict[tuple[str, str], list[int]] | None = None,
            workers: int = WORKERS) -> dict[tuple[str, str], list[dict]]:
    """{(family, clip): fixed-QP points sorted by rate}. `quantizers` fixes the
    grid per (family name, clip name); otherwise each family's own (Quantizers).

    A curve that fails (the encoder refuses a setting, or a pool clip cannot
    serve its frames) is reported and left out, never silently: pair_rows
    then drops that clip from the pairs it is missing from.
    """
    chooser = Quantizers(benches)
    failed: dict[tuple[str, str], str] = {}

    def collect(jobs: dict) -> dict:
        done = {}
        for key, job in jobs.items():
            try:
                done[key] = job.result()
            except RuntimeError as error:
                failed.setdefault(key[:2], str(error).splitlines()[0][:200])
        return done
    with ThreadPoolExecutor(max_workers=workers) as executor:
        grids = collect({(f.name, c.name): executor.submit(chooser.choose, f, c)
                         for f in families for c in clips
                         if not quantizers or (f.name, c.name) not in quantizers})
        grids |= quantizers or {}
        results = collect({(f.name, c.name, p.qp): executor.submit(benches.of(c).measure, p)
                           for f in families for c in clips if (f.name, c.name) in grids
                           and (f.name, c.name) not in failed
                           for p in curve_points(benches.of(c), f, c, EVALUATION_GATE,
                                                 quantizers=grids[(f.name, c.name)])})
    for (family, clip), reason in sorted(failed.items()):
        print(f"failed: {family} {clip}: {reason}", flush=True)
    curves: dict[tuple[str, str], list[dict]] = {}
    for (family, clip, _), point in results.items():
        if (family, clip) not in failed:
            curves.setdefault((family, clip), []).append(point)
    return {k: sorted(v, key=lambda p: p["kbps"]) for k, v in curves.items()}


def pair_rows(curves: dict, clips: list[Clip], anchor: str, test: str) -> dict[str, dict]:
    """Per-clip terms of `test` against `anchor`, in the shape nceval.interval reads
    (clips missing either curve are left out)."""
    rows = {}
    for clip in clips:
        if (anchor, clip.name) not in curves or (test, clip.name) not in curves:
            continue
        terms = clip_terms(curves[(anchor, clip.name)], curves[(test, clip.name)])
        rows[clip.name] = {"class": clip.content_class, "quality": terms["full"],
                           "quality_low": terms["low"],
                           **{m: terms[m] for m in TERMS[2:]}}
    return rows


def summary(rows: dict[str, dict]) -> dict:
    """BD-J with its interval, and each metric's class-balanced BD-rate."""
    stats = interval.interval(rows, None, 0.0)
    out = {"j": stats["mean"], "lo": stats["lo"], "hi": stats["hi"], "n": stats["n"]}
    for metric in TERMS[2:]:
        values = [(r["class"], r[metric]) for r in rows.values() if r[metric] is not None]
        out[metric] = interval.balanced_mean(values) if values else None
    return out


def describe(label: str, result: dict) -> str:
    def pct(value):
        return "   n/a" if value is None else f"{value:+6.1f}"
    return (f"{label:52s} J {pct(result['j'])} [{pct(result['lo'])},{pct(result['hi'])}] "
            f"PSNR {pct(result['psnr_yuv'])} VMAF {pct(result['vmaf_v1'])} "
            f"MS-SSIM {pct(result['ms_ssim'])}  n={result['n']}")


def gap(args: argparse.Namespace) -> dict:
    paths, binary = load_paths(), Path(args.binary)
    names = args.families.split(",")
    families = [family_of(n, binary, paths) for n in names]
    clips = draw_clips(paths, args.rotation, args.level, args.clips, args.frames)
    print(f"draw {args.rotation} {args.level}: {len(clips)} clips; "
          f"{len(families)} families", flush=True)
    curves = measure(Benches(paths, args.frames), families, clips, workers=args.workers)
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")] if args.pairs else []
    results = {}
    for anchor, test in pairs:
        rows = pair_rows(curves, clips, anchor, test)
        results[f"{test} vs {anchor}"] = {"summary": summary(rows), "per_clip": rows}
        print(describe(f"{test} vs {anchor}", results[f"{test} vs {anchor}"]["summary"]),
              flush=True)
    return {"rotation": args.rotation, "level": args.level, "frames": args.frames,
            "clips": [{"clip": c.name, "start_frame": c.start_frame, "class": c.content_class}
                      for c in clips],
            "binary_commit": buildinfo.info_of(binary).get("commit"),
            "pairs": results,
            "rates": {f"{f}|{c}": [round(p["kbps"], 1) for p in pts]
                      for (f, c), pts in curves.items()}}


def candidate_sets(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """(label, [name=value, ...]) per candidate: `--param` pairs form one candidate,
    each line `label: a=1 b=2` of `--sets` another (# comments)."""
    sets = [(" ".join(args.param), args.param)] if args.param else []
    if args.sets:
        for line in Path(args.sets).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                label, _, pairs = line.partition(":")
                sets.append((label.strip(), pairs.split()))
    if not sets:
        raise SystemExit("no candidate: give --param or --sets")
    return sets


def ab(args: argparse.Namespace) -> dict:
    """Every candidate against one anchor at the anchor's quantizers (matched QP)."""
    paths = load_paths()
    codec = Codec(args.codec)
    name = OWN_PREFIX + codec.value + (f"@{args.preset}" if args.preset else "")
    anchor = own_family(name, Path(args.anchor_binary or args.binary))
    anchor = replace(anchor, extra=anchor.extra + param_args(args.anchor_param))
    base = own_family(name, Path(args.binary))
    sets = candidate_sets(args)
    tests = [replace(base, name=f"{name}+{index}", extra=anchor.extra + param_args(pairs))
             for index, (_, pairs) in enumerate(sets)]
    clips = draw_clips(paths, args.rotation, args.level, args.clips, args.frames)
    print(f"draw {args.rotation} {args.level}: {len(clips)} clips; "
          f"{len(sets)} candidates", flush=True)
    benches = Benches(paths, args.frames)
    chooser = Quantizers(benches)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        grids = {(anchor.name, c.name): executor.submit(chooser.choose, anchor, c)
                 for c in clips}
        grids = {k: j.result() for k, j in grids.items()}
    grids |= {(test.name, clip): q for test in tests for (_, clip), q in grids.items()}
    curves = measure(benches, [anchor, *tests], clips, grids, args.workers)
    results = []
    for (label, pairs), test in zip(sets, tests):
        if not any(family == test.name for family, _ in curves):  # refused everywhere
            results.append({"label": label, "param": pairs, "refused": True})
            continue
        rows = pair_rows(curves, clips, anchor.name, test.name)
        results.append({"label": label, "param": pairs, "summary": summary(rows),
                        "per_clip": rows})
        print(describe(f"{codec.value}{'@' + args.preset if args.preset else ''} {label}",
                       results[-1]["summary"]), flush=True)
    return {"codec": codec.value, "preset": args.preset, "anchor_param": args.anchor_param,
            "rotation": args.rotation, "level": args.level, "candidates": results,
            "binary_commit": buildinfo.info_of(Path(args.binary)).get("commit")}


#: Concurrent callgrind runs of `cost` (each is one core for minutes).
COST_WORKERS = 6


def cost(args: argparse.Namespace) -> dict:
    """x86 P-frame instructions of each candidate relative to the anchor settings
    (tune.instruction_cost: callgrind, key + COST_P_FRAMES P frames on
    tune.COST_CLIPS, geometric mean): the CPU column of the ablation table."""
    import tempfile
    from . import tune
    from .tune import read_preset
    codec, binary = Codec(args.codec), Path(args.binary)
    preset = read_preset(PRESETS_DIR / f"{args.preset}.toml") if args.preset else {}
    sets = [("anchor", [])] + candidate_sets(args)
    if args.only:
        sets = [s for s in sets if s[0] == "anchor" or s[0] in args.only.split(",")]

    def one(item: tuple[str, list[str]]) -> tuple[str, float]:
        label, pairs = item
        overrides = preset | {k: float(v) for k, v in (pair.split("=") for pair in pairs)}
        with tempfile.TemporaryDirectory(dir=load_paths().work) as work:
            return label, tune.instruction_cost(binary, codec, overrides, Path(work)).p
    with ThreadPoolExecutor(max_workers=COST_WORKERS) as executor:
        costs = dict(executor.map(one, sets))
    base = costs["anchor"]
    result = {label: {"p_minstr": value, "dcpu_pct": 100 * (value / base - 1)}
              for label, value in costs.items()}
    for label, row in result.items():
        print(f"{codec.value} {label:24s} {row['p_minstr']:9.1f} M instr/P  "
              f"{row['dcpu_pct']:+6.1f}%", flush=True)
    return {"codec": codec.value, "preset": args.preset, "costs": result}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python3 -m nceval.fixedqp",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("gap", "ab", "cost"):
        p = sub.add_parser(name)
        p.add_argument("--binary", required=True)
        p.add_argument("--json", help="write the result here")
    for name in ("gap", "ab"):
        p = sub.choices[name]
        p.add_argument("--rotation", default=default_rotation())
        p.add_argument("--level", default=pool.QUICK, choices=pool.LEVELS)
        p.add_argument("--clips", type=int, help="the first N clips of the level")
        p.add_argument("--workers", type=int, default=WORKERS)
        p.add_argument("--frames", type=int, default=FRAMES,
                       help="frames per clip (fewer only to pipe-clean a run)")
    g = sub.choices["gap"]
    g.add_argument("--families", required=True,
                   help="nanocodec-<codec>[@preset] and anchor family names (cbr)")
    g.add_argument("--pairs", help="anchor:test,... to report")
    for name in ("ab", "cost"):
        a = sub.choices[name]
        a.add_argument("--codec", required=True, choices=[c.value for c in Codec])
        a.add_argument("--preset", help="the preset both sides start from (default: Default)")
        a.add_argument("--param", action="append", default=[], help="one candidate's name=value")
        a.add_argument("--sets", help="file of candidates, one per line: `label: a=1 b=2`")
    sub.choices["ab"].add_argument("--anchor-binary")
    sub.choices["ab"].add_argument("--anchor-param", action="append", default=[])
    sub.choices["cost"].add_argument("--only", help="comma-separated candidate labels")
    args = parser.parse_args(argv)
    result = {"gap": gap, "ab": ab, "cost": cost}[args.command](args)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
