#!/usr/bin/env python3
"""Identity and speed A/B of two builds of the reference encoders, on one pinned core.

    reference_ab.py --plan FULL-PLAN.json --clip NAME[@KBPS] --yuv CLIP.yuv
        --old DIR --new DIR [--frames N] [--repeats 3] [--cpu 1] --out AB.json

Runs on the timing box (tools/dev/aws_reference_builds.py; standard library only).
For every reference family of the chart's full timing plan (the one
docs/results/bd-vs-speed-plan.json names; its points are keyed `clip@kbps`), the chart's
own command lines, the old and the new build of its program code the
same clip with the same command, `--repeats` rounds, the order alternating each round
(old new, new old, ...), pinned to `--cpu` with taskset. Every encode's stream is
hashed: a rebuild that only changes how the code was compiled (LTO, PGO) must give
the old build's bytes. Time is the encode's user + system CPU time (rusage).

The summary per family: identical (every round), each round's new/old CPU-time
ratio, their median and range, and the first stderr lines that name the CPU
features the program dispatched to (x264, x265 and SVT-AV1 print them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

#: Our own encoder's families, which this A/B leaves out.
OWN_PREFIX = "nanocodec"
#: How each program's argv names the frame count (the plan's 150 is cut to --frames).
FRAME_OPTIONS = ("--frames", "-n")
FRAME_PREFIXES = ("--limit=",)
#: stderr words that show the CPU features a program dispatched to.
DISPATCH_WORDS = ("cpu capabilities", "asm", "CPU", "SIMD", "AVX", "simd")
#: Extra variants timed on both builds: the family, a suffix and the added options.
#: x265 uses its AVX-512 kernels only when asked (`--asm avx512`, x265 --help).
VARIANTS = {"x265-medium": ("+asm-avx512", ("--asm", "avx512"))}


def set_frames(argv: list[str], frames: int) -> list[str]:
    out = list(argv)
    for index, token in enumerate(out):
        if token in FRAME_OPTIONS and index + 1 < len(out):
            out[index + 1] = str(frames)
        for prefix in FRAME_PREFIXES:
            if token.startswith(prefix):
                out[index] = f"{prefix}{frames}"
    return out


def encode(argv: list[str], cpu: str) -> tuple[float, str]:
    """(CPU seconds, stderr) of one pinned encode."""
    with tempfile.TemporaryFile() as err:
        process = subprocess.Popen(["taskset", "-c", cpu, *argv], stdout=subprocess.DEVNULL,
                                   stderr=err)
        _, status, usage = os.wait4(process.pid, 0)
        err.seek(0)
        text = err.read().decode(errors="replace")
    if os.waitstatus_to_exitcode(status) != 0:
        code = os.waitstatus_to_exitcode(status)
        raise RuntimeError(f"exit {code}: {' '.join(argv)}\n{text[-2000:]}")
    return usage.ru_utime + usage.ru_stime, text


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dispatch_lines(stderr: str) -> list[str]:
    return [line.strip() for line in stderr.splitlines()
            if any(word in line for word in DISPATCH_WORDS)][:4]


def run_rounds(template: list[str], builds: dict[str, Path], args: argparse.Namespace,
               work: Path, times: dict, hashes: dict, dispatch: dict) -> None:
    """`args.repeats` rounds of old and new, the order alternating each round."""
    for round_index in range(args.repeats):
        order = ("old", "new") if round_index % 2 == 0 else ("new", "old")
        for build in order:
            out = work / f"{build}.bin"
            argv = [part.replace("{BIN}", str(builds[build])).replace(
                "{INPUT}", str(args.yuv)).replace("{OUTPUT}", str(out)) for part in template]
            seconds, stderr = encode(set_frames(argv, args.frames), args.cpu)
            times[build].append(seconds)
            hashes[build].add(md5(out))
            dispatch.setdefault(build, dispatch_lines(stderr))
            out.unlink()


def family_commands(plan: dict, clip: str, extra: list[str]) -> dict[str, list[str]]:
    """The plan's reference commands for `clip`, plus VARIANTS and the `extra` variants
    (`FAMILY+SUFFIX=OPTIONS`, e.g. `x265-medium+neon=--asm NEON`)."""
    commands = {name: points[clip]["argv"] for name, points in plan["points"].items()
                if not name.startswith(OWN_PREFIX) and clip in points}
    if not commands:
        raise SystemExit(f"no reference point of the plan codes {clip!r}")
    variants = dict(VARIANTS)
    for spec in extra:
        name, options = spec.split("=", 1)
        base, suffix = name.split("+", 1)
        variants[base] = (f"+{suffix}", tuple(options.split()))
    for base, (suffix, options) in variants.items():
        if base in commands:
            argv = commands[base]
            commands[base + suffix] = [argv[0], *options, *argv[1:]]
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--yuv", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu", default="1")
    parser.add_argument("--only", help="comma-separated families (default: all)")
    parser.add_argument("--variant", action="append", default=[],
                        help="FAMILY+SUFFIX=OPTIONS: also time FAMILY with OPTIONS added")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commands = family_commands(json.loads(args.plan.read_text()), args.clip, args.variant)
    if args.only:
        commands = {k: v for k, v in commands.items() if k in args.only.split(",")}
    work = Path(tempfile.mkdtemp(prefix="nanocodec-refab-"))
    summary = {"clip": args.clip, "frames": args.frames, "repeats": args.repeats,
               "cpu": args.cpu, "families": {}}
    for family, template in commands.items():
        builds = {"old": args.old, "new": args.new}
        times: dict[str, list[float]] = {"old": [], "new": []}
        hashes: dict[str, set[str]] = {"old": set(), "new": set()}
        dispatch = {}
        try:
            run_rounds(template, builds, args, work, times, hashes, dispatch)
        except RuntimeError as error:  # e.g. an old build without an option the plan uses
            summary["families"][family] = {"error": str(error)[:600]}
            print(f"{family:28s} skipped, a build could not run it: "
                  f"{str(error).splitlines()[1] if len(str(error).splitlines()) > 1 else ''}",
                  flush=True)
            for leftover in work.iterdir():
                leftover.unlink()
            continue
        ratios = [n / o for n, o in zip(times["new"], times["old"])]
        summary["families"][family] = {
            "identical": len(hashes["old"]) == 1 and hashes["old"] == hashes["new"],
            "md5": {k: sorted(v) for k, v in hashes.items()},
            "cpu_s": times, "ratio_new_over_old": ratios,
            "ratio_median": statistics.median(ratios),
            "ratio_range": [min(ratios), max(ratios)], "dispatch": dispatch}
        row = summary["families"][family]
        print(f"{family:28s} identical={row['identical']} ratio={row['ratio_median']:.3f} "
              f"[{min(ratios):.3f}, {max(ratios):.3f}] old={statistics.median(times['old']):.2f}s",
              flush=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    os.rmdir(work)


if __name__ == "__main__":
    main()
