#!/usr/bin/env python3
"""Sanity checks of the bitrate-vs-speed chart's speed axis (chart_refresh.py's `check`).

A loaded PC can mis-time one point while its anchor-paired rescaling still looks
plausible (2026-11-12: AV1 Fast timed level with AV1 Faster on x86, 153.7 fps,
while it runs 44% more instructions and is 20% slower on the Jetson). Two
cross-checks catch that, per codec:

  ratio         the x86/Jetson fps ratios of adjacent presets of one codec
                (Default/Fast, Fast/Faster, ...): apart by more than
                RATIO_TOLERANCE is flagged (neighbouring presets share most of
                their work, so the platforms' speed ratio moves little between them).
  instructions  callgrind instruction counts of each preset on one timing clip
                (CALLGRIND_CLIP, its first CALLGRIND_FRAMES frames, the timing
                plan's own command on the family's recorded x86 timed build): a
                preset that runs clearly more instructions (INSTRUCTION_MARGIN)
                but is not slower on x86 is flagged.

Flags are warnings for whoever refreshes the chart (re-time the flagged points,
e.g. `chart_refresh.py C --families F --from time-x86 --x86-rounds 21`), not
failures.

    python3 tools/dev/chart_sanity.py [--codecs av1,h264] [--no-instructions]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs/results"
JETSON_CSV = RESULTS / "bd-vs-speed.csv"
X86_CSV = RESULTS / "bd-vs-speed-x86.csv"
PLAN_FILE = RESULTS / "bd-vs-speed-plan.json"
BUILDS_FILE = RESULTS / "bd-vs-speed-builds.json"
X86_TIMING_FILE = RESULTS / "bd-vs-speed-timing-x86.json"
BUILDS_ROOT = Path.home() / "nanocodec-data/charts/builds"
#: Binaries of a build directory, preferred first: the x86 timed build, then the
#: scoring build (same streams; only the instruction mix of LTO/PGO differs).
TIMED_BINARIES = ("nanocodec-timed-pgo", "nanocodec")
FAMILY_PREFIX = "nanocodec-"
#: Adjacent presets' x86/Jetson fps ratios may differ this much (relative). The
#: ratio drifts along a codec's presets (HEVC Default 2.29 vs Faster 1.74 at
#: 4ff6a3c1: AVX2 helps the heavier search more than NEON does), but adjacent
#: presets stay closer (AV1 Default/Fast 2.48/2.07 at 31d1a480, 19%). At
#: introduction this flagged HEVC Default/Fast (2.29/1.80, 27%: Default runs
#: 1.74x Fast's instructions, 1.50x slower on the Jetson, only 1.18x on x86).
RATIO_TOLERANCE = 0.20
#: A codec's presets from slowest to fastest (the family suffix after the codec).
PRESET_ORDER = ("", "fast", "faster", "fastest")
#: The instruction check's clip and length (crowd: the timing set's middle clip).
CALLGRIND_CLIP = "crowd_mot17_09"
CALLGRIND_FRAMES = 30
#: Concurrent callgrind runs (each one core; ~30 s per preset).
CALLGRIND_JOBS = 4
#: More instructions by this fraction must show as lower x86 fps; smaller gaps
#: are within what cache and branch effects can reverse.
INSTRUCTION_MARGIN = 0.05
COLLECTED = re.compile(r"Collected\s*:\s*(\d+)")


def codec_of(family: str) -> str:
    return family.split("@")[0].removeprefix(FAMILY_PREFIX)


def preset_rank(family: str) -> int:
    """Position in PRESET_ORDER: nanocodec-av1 0, nanocodec-av1@av1-fast 1, ..."""
    preset = family.split("@")[1] if "@" in family else ""
    return PRESET_ORDER.index(preset.removeprefix(f"{codec_of(family)}-"))


def fps_table() -> dict[str, tuple[float, float]]:
    """Our families' (Jetson fps, x86 fps) from the rendered CSVs."""
    jetson = {r["family"]: float(r["fps_jetson"]) for r in csv.DictReader(JETSON_CSV.open())}
    x86 = {r["family"]: float(r["fps_x86"]) for r in csv.DictReader(X86_CSV.open())}
    return {f: (jetson[f], x86[f]) for f in jetson
            if f.startswith(FAMILY_PREFIX) and f in x86}


def by_codec(families, codecs: set[str] | None) -> dict[str, list[str]]:
    """Codec -> its families from slowest to fastest preset (codecs with one preset dropped)."""
    groups: dict[str, list[str]] = {}
    for family in sorted(families, key=preset_rank):
        if codecs is None or codec_of(family) in codecs:
            groups.setdefault(codec_of(family), []).append(family)
    return {c: f for c, f in groups.items() if len(f) > 1}


def ratio_flags(fps: dict[str, tuple[float, float]], groups: dict[str, list[str]]) -> list[str]:
    flags = []
    for families in groups.values():
        ratio = {f: fps[f][1] / fps[f][0] for f in families}
        for slower, faster in zip(families, families[1:]):
            if max(ratio[slower], ratio[faster]) / min(ratio[slower], ratio[faster]) - 1 \
                    > RATIO_TOLERANCE:
                flags.append(f"ratio: {slower} x86/Jetson {ratio[slower]:.2f} "
                             f"({fps[slower][1]:.1f}/{fps[slower][0]:.1f} fps) vs {faster} "
                             f"{ratio[faster]:.2f} ({fps[faster][1]:.1f}/{fps[faster][0]:.1f})")
    return flags


def timed_binary(family: str, record: dict) -> Path:
    build = BUILDS_ROOT / record["families"][family]
    for name in TIMED_BINARIES:
        if (build / name).is_file():
            return build / name
    raise SystemExit(f"{family}: no x86 binary in {build}")


def callgrind_instructions(argv: list[str], binary: Path, clip: Path) -> int:
    """Instructions of the planned command on the clip's first CALLGRIND_FRAMES frames."""
    with tempfile.TemporaryDirectory(dir=Path.home() / "nanocodec-data/work") as tmp:
        cmd = [str(binary)] + [str(clip) if a == "{INPUT}" else f"{tmp}/s.bin" if a == "{OUTPUT}"
                               else a for a in argv[1:]]
        cmd[cmd.index("--frames") + 1] = str(CALLGRIND_FRAMES)
        run = subprocess.run(["valgrind", "--tool=callgrind", f"--callgrind-out-file={tmp}/cg",
                              *cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        found = COLLECTED.search(run.stderr)
        if run.returncode or not found:
            raise SystemExit(f"callgrind failed on {binary}: {run.stderr[-400:]}")
        return int(found.group(1))


def instruction_flags(fps: dict[str, tuple[float, float]],
                      groups: dict[str, list[str]]) -> tuple[list[str], dict[str, int]]:
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.config import load_paths
    from nceval.realuse import main_clips

    clip = main_clips(load_paths())[CALLGRIND_CLIP].yuv
    plan = json.loads(PLAN_FILE.read_text())["points"]
    record = json.loads(BUILDS_FILE.read_text())
    families = [f for fs in groups.values() for f in fs]
    with ThreadPoolExecutor(CALLGRIND_JOBS) as pool:
        counts = dict(zip(families, pool.map(
            lambda f: callgrind_instructions(plan[f][CALLGRIND_CLIP]["argv"],
                                             timed_binary(f, record), clip), families)))
    flags = []
    for members in groups.values():
        for a, b in combinations(members, 2):
            heavy, light = (a, b) if counts[a] > counts[b] else (b, a)
            if counts[heavy] > counts[light] * (1 + INSTRUCTION_MARGIN) \
                    and fps[heavy][1] >= fps[light][1]:
                flags.append(f"instructions: {heavy} runs {counts[heavy] / counts[light]:.2f}x "
                             f"{light}'s instructions but times {fps[heavy][1]:.1f} vs "
                             f"{fps[light][1]:.1f} fps on x86")
    return flags, counts


def same_session_ratios() -> dict[str, dict[str, float]]:
    """Family -> {second anchor: geometric-mean per-clip time ratio}, from the x86
    timing file (anchor_chart_timing.py time-host: timed in the same rounds)."""
    points = json.loads(X86_TIMING_FILE.read_text())["points"]
    return {family: {ref: statistics.geometric_mean(list(ratios.values()))
                     for ref, ratios in entry["same_session_time_ratio"].items()}
            for family, entry in points.items() if "same_session_time_ratio" in entry}


def check(codecs: set[str] | None, instructions: bool = True) -> list[str]:
    """Every flag for the given codecs (None: all), each one line; prints the counts
    and the same-session time ratios to the second anchor."""
    fps = fps_table()
    groups = by_codec(fps, codecs)
    for family, ratios in sorted(same_session_ratios().items()):
        if codecs is None or codec_of(family) in codecs:
            print(f"  {family:30s} same session: " + ", ".join(
                f"{r:.3f}x {ref}'s time" for ref, r in ratios.items()), flush=True)
    flags = ratio_flags(fps, groups)
    if instructions:
        instruction_list, counts = instruction_flags(fps, groups)
        flags += instruction_list
        for family, count in counts.items():
            print(f"  {family:30s} {count / 1e9:6.2f}G instructions ({CALLGRIND_CLIP}, "
                  f"{CALLGRIND_FRAMES} frames), x86 {fps[family][1]:.1f}, "
                  f"Jetson {fps[family][0]:.1f} fps", flush=True)
    return flags


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codecs", default="", help="comma-separated codecs, e.g. av1,h264 (all)")
    p.add_argument("--no-instructions", action="store_true", help="the ratio check only")
    args = p.parse_args()
    flags = check(set(args.codecs.split(",")) if args.codecs else None,
                  not args.no_instructions)
    for flag in flags:
        print(f"FLAG {flag}")
    print(f"{len(flags)} flag(s)")


if __name__ == "__main__":
    main()
