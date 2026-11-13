"""Relative bitrate vs speed: every codec, preset and reference encoder on one scale.

An AOM/MSU-style chart. Every encoder (our codecs and presets, openh264,
SVT-AV1, x264, x265) runs its own real-time rate control on the real-use CBR
ladder (nceval.realuse: 150/300/600/1200 kbps at 720p30, pixel-rate scaled,
1 s buffer, one key frame, 150 frames, natural VAL clips; our streams scored
on the reconstruction of a build that passed the landing conformance check). Each is compared with ONE neutral anchor, x264 `--preset medium
--tune zerolatency` without B-frames, by class-balanced BD-rate on the
achieved-rate curves:

  - BD-J: the J_real quality composite, (CB(Q_full) + CB(Q_low)) / 2 with
    Q = mean BD-rate over PSNR-YUV, VMAF v1 and MS-SSIM (nceval.jreal);
  - BD-PSNR, BD-VMAF v1: the same with PSNR-YUV or VMAF v1 alone;
  - spread: the interquartile range of the per-clip BD-J;
  - on clips where some curve has no reliable overlap with the anchor's
    (bdrate.MIN_OVERLAP_FRACTION), every encoder gets extra low rungs
    (`LOW_LADDER_KBPS`, `low_rung_clips`) that reach into its quality range;
  - every point is averaged over ONE clip set: the clips where every point
    overlaps the anchor directly (`anchor_points`); the others are left out
    for all and listed, unless a whole class would vanish, whose clips are
    then kept by chaining every unscored point through ONE intermediate
    encoder (`chain_clip`, marked ‡); a point with too few scored clips is
    flagged, drawn hollow and kept off the Pareto front; nothing is
    extrapolated.

Speed comes from a timing file (tools/dev/anchor_chart_timing.py: Jetson
core 3, single thread, rate-matched to the 600 kbps rung), as frames per
second and as speed relative to the anchor. A second timing file from the
PC (`--timing-x86`) gives the same chart with x86 speed (`-x86` outputs).

    python3 -m nceval.chart measure --binary BIN [--points a,b] [--part I/N]
    python3 -m nceval.chart render --binary BIN --timing FILE [--timing-x86 FILE]
        [--meta FILE] [--out DIR]
    (both take --binary-for FAMILY=PATH and --presets-for FAMILY=DIR for a family on
    another build; tools/dev/chart_refresh.py refreshes the families a commit changes)

`measure` caches points (each invocation is budgeted under 30 min; split
with --part or --points); `render` reads the cache and writes
bd-vs-speed.{svg,csv,md}.

Hardware encoders (`HARDWARE`: NVIDIA NVENC, measured with `measure --points
nvenc-...` on this PC's GPU) are on the quality axis only: scored on the
chart's clip set and rungs without changing them, listed in their own table
(`bd-vs-speed-hardware.csv`) with their wall-clock fps (`render
--timing-hardware`, tools/dev/nvenc_timing.py), and drawn on the x86 chart only,
at that fps in their own style (past the right end only if faster than the
axis), never on the Pareto front.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from . import buildinfo, jreal
from .budget import MAX_MINUTES, estimate_minutes
from .cbr import Family, anchor_families, clip_targets
from .clips import Clip
from .codecs import Codec
from .config import load_paths
from .encoders.nanocodec import PRESETS_DIR
from .gatepolicy import EVALUATION_GATE
from .realuse import (DEFAULT_FRAMES, DEFAULT_WORKERS, LADDER_KBPS, OPENH264_HARNESS, Runner,
                      clipset_clips, compare, natural_val, own_family)
from .stages import balanced_parts

#: The neutral anchor every point is compared with (BD 0, speed 1).
ANCHOR = "x264-medium"
#: Our points: (family, chart label).
OURS = (("nanocodec-h264", "H.264"), ("nanocodec-h264@h264-fast", "H.264 Fast"),
        ("nanocodec-h264@h264-faster", "H.264 Faster"),
        ("nanocodec-h264@h264-fastest", "H.264 Fastest"), ("nanocodec-hevc", "HEVC"),
        ("nanocodec-hevc@hevc-fast", "HEVC Fast"), ("nanocodec-hevc@hevc-faster", "HEVC Faster"),
        ("nanocodec-av1", "AV1"),
        ("nanocodec-av1@av1-fast", "AV1 Fast"), ("nanocodec-av1@av1-faster", "AV1 Faster"),
        ("nanocodec-vp9", "VP9"),
        ("nanocodec-vp9@vp9-fast", "VP9 Fast"), ("nanocodec-vp9@vp9-faster", "VP9 Faster"),
        ("nanocodec-vp9@vp9-fastest", "VP9 Fastest"), ("nanocodec-av2", "AV2"),
        ("nanocodec-av2@av2-fast", "AV2 Fast"), ("nanocodec-vvc", "VVC"),
        ("nanocodec-vvc@vvc-fast", "VVC Fast"))
#: Reference encoders: (family, chart label). Families are cbr.anchor_families.
REFERENCES = (("openh264-baseline", "openh264"),
              ("x264-ultrafast", "x264 ultrafast"), ("x264-veryfast", "x264 veryfast"),
              ("x264-medium", "x264 medium"),
              ("x265-ultrafast", "x265 ultrafast"), ("x265-superfast", "x265 superfast"),
              ("x265-medium", "x265 medium"),
              ("svtav1-m8", "SVT-AV1 M8"), ("svtav1-m10", "SVT-AV1 M10"),
              ("svtav1-m12", "SVT-AV1 M12"), ("svtav1-m13", "SVT-AV1 M13"),
              ("libvpx-vp9-cpu7", "libvpx VP9 rt 7"), ("libvpx-vp9-cpu9", "libvpx VP9 rt 9"),
              ("libaom-av1-cpu8", "libaom rt 8"), ("libaom-av1-cpu10", "libaom rt 10"))
#: Hardware encoders (cbr.NVENC_FAMILIES): (family, chart label). On the quality axis
#: only: they are scored on the chart's own clip set and rungs, but never change
#: which clips or low rungs the other points use, and their fps (the GPU's wall
#: time) is not CPU speed: a table of their own, and on the PC's (x86) chart only,
#: in their own style and off the Pareto front.
#: The GPU they ran on (the PC's only GPU), named in the legend and the table.
HARDWARE_GPU = "RTX 5090"
#: Short labels: p4 is preset P4 with -tune ll, p7 preset P7 with -tune hq.
HARDWARE = (("nvenc-h264", "NVENC H.264 p4"), ("nvenc-h264-p7hq", "NVENC H.264 p7"),
            ("nvenc-hevc", "NVENC HEVC p4"), ("nvenc-hevc-p7hq", "NVENC HEVC p7"),
            ("nvenc-av1", "NVENC AV1 p4"), ("nvenc-av1-p7hq", "NVENC AV1 p7"))
HARDWARE_FAMILIES = frozenset(name for name, _ in HARDWARE)
#: Families that anchor_families builds without being asked for by name.
BUILT_IN_ANCHORS = ("svtav1-m12", "openh264-baseline")
#: A point is flagged (insufficient curve overlap: hollow, off the Pareto
#: front) when fewer than this fraction of the clips have a BD-J. A point
#: with a whole content class unscored is "partial" (marked †): its
#: class-balanced BD is over the classes it has.
MIN_SCORED_FRACTION = 0.5
PARTIAL_MARK = "†"
#: Extra low rungs (720p30 kbps, scaled like the ladder) for EVERY encoder on the
#: clips where some curve has no reliable overlap with the anchor's. There x264
#: medium undershoots and has lower quality at each rate than most points
#: (static clips, bodycam_airport), so their curves start above its top; lower
#: rungs bring every curve down into its quality range. The 50 % source cap
#: does not limit rates below the ladder. Two rungs, halving like the ladder.
LOW_LADDER_KBPS = (40, 75)
#: Marks a point with clips chained through an intermediate encoder.
CHAIN_MARK = "‡"
#: Codec lineages: a chained clip prefers an intermediate of the point's own
#: lineage (AV1 for AV2) and of its own side (ours or a reference program),
#: whose curves are most alike, before any other.
LINEAGES = (frozenset({Codec.H264}), frozenset({Codec.HEVC, Codec.VVC}),
            frozenset({Codec.VP9, Codec.AV1, Codec.AV2}))
#: The per-metric BD terms composed through the intermediate: the inputs of
#: BD-J (jreal.headline_quality) and of BD-PSNR-YUV.
CHAINED_TERMS = tuple(f"{prefix}{metric}" for prefix in ("bd_", "low_")
                      for metric in jreal.HEADLINE_METRICS)
OUTPUT_STEM = "bd-vs-speed"
#: Direct comparisons shown under the chart: (ours, reference), each BD on the
#: same real-use points, without the neutral anchor in between.
SELECTED_PAIRS = (
    ("nanocodec-h264", "openh264-baseline"), ("nanocodec-h264", "x264-veryfast"),
    ("nanocodec-h264", "x264-medium"), ("nanocodec-h264@h264-fast", "x264-veryfast"),
    ("nanocodec-h264@h264-fast", "openh264-baseline"), ("nanocodec-h264@h264-fast", "nanocodec-h264"),
    ("nanocodec-h264@h264-faster", "x264-ultrafast"), ("nanocodec-h264@h264-faster", "nanocodec-h264@h264-fast"),
    ("nanocodec-h264@h264-fastest", "x264-ultrafast"),
    ("nanocodec-h264@h264-fastest", "nanocodec-h264@h264-faster"),
    ("nanocodec-hevc", "x265-medium"), ("nanocodec-hevc@hevc-fast", "x265-superfast"),
    ("nanocodec-hevc", "svtav1-m12"), ("nanocodec-hevc@hevc-fast", "nanocodec-hevc"),
    ("nanocodec-av1", "openh264-baseline"), ("nanocodec-av1@av1-fast", "openh264-baseline"),
    ("nanocodec-av1", "svtav1-m8"), ("nanocodec-av1", "svtav1-m12"),
    ("nanocodec-av1@av1-fast", "svtav1-m12"), ("nanocodec-av1", "libaom-av1-cpu8"),
    ("nanocodec-av1@av1-fast", "nanocodec-av1"),
    ("nanocodec-av1@av1-faster", "svtav1-m12"), ("nanocodec-av1@av1-faster", "nanocodec-av1@av1-fast"),
    ("nanocodec-hevc@hevc-faster", "svtav1-m13"), ("nanocodec-hevc@hevc-faster", "nanocodec-hevc@hevc-fast"),
    ("nanocodec-vp9", "libvpx-vp9-cpu7"), ("nanocodec-vp9@vp9-fast", "libvpx-vp9-cpu9"),
    ("nanocodec-vp9@vp9-fast", "nanocodec-vp9"),
    ("nanocodec-vp9@vp9-faster", "libvpx-vp9-cpu9"), ("nanocodec-vp9@vp9-faster", "nanocodec-vp9@vp9-fast"),
    ("nanocodec-vp9@vp9-fastest", "libvpx-vp9-cpu9"),
    ("nanocodec-vp9@vp9-fastest", "nanocodec-vp9@vp9-faster"),
    ("nanocodec-av2", "svtav1-m8"), ("nanocodec-av2", "nanocodec-av1"),
    ("nanocodec-av2@av2-fast", "nanocodec-av2"),
    ("nanocodec-vvc", "x265-medium"), ("nanocodec-vvc", "nanocodec-hevc"),
    ("nanocodec-vvc@vvc-fast", "nanocodec-vvc"))


@dataclass(frozen=True)
class ChainedClip:
    """A clip whose BD vs the anchor goes through one intermediate encoder."""
    clip: str
    intermediate: str
    to_intermediate: float  # BD-J of the point vs the intermediate
    intermediate_to_anchor: float  # the intermediate's direct BD-J vs the anchor
    j: float  # the composed BD-J of the point vs the anchor


@dataclass(frozen=True)
class ChartPoint:
    family: str
    label: str
    ours: bool
    bd_j: float | None
    bd_psnr: float | None
    #: BD-rate on VMAF v1 alone, the same (full + low) / 2 as bd_j and bd_psnr.
    bd_vmaf: float | None
    iqr: tuple[float, float] | None
    scored: int
    clips: int
    missing_classes: tuple[str, ...]
    rate_invalid: int
    refused: int
    fps: float | None = None
    speed: float | None = None  # relative to the anchor (anchor = 1)
    chained: tuple[ChainedClip, ...] = ()
    #: BD-J over the directly compared clips only (equal to bd_j without chained clips).
    direct_bd_j: float | None = None

    @property
    def flagged(self) -> bool:
        return self.bd_j is None or self.scored < MIN_SCORED_FRACTION * self.clips

    @property
    def partial(self) -> bool:
        """A whole content class without a BD: the class-balanced mean misses it."""
        return bool(self.missing_classes)


# --- families and measurement ----------------------------------------------------------

def chart_families(binary: Path, presets: Path, names: list[str] | None = None,
                   binaries: dict[str, Path] | None = None,
                   presets_for: dict[str, Path] | None = None) -> list[Family]:
    """The anchor first, then every requested point's family.

    `binaries` gives some of our families another build than `binary` (a
    codec merged after the frozen build, e.g. nanocodec-vvc), `presets_for`
    that build's own presets directory (tools/dev/chart_refresh.py keeps
    every family on its own build)."""
    paths = load_paths()
    wanted = names or [name for name, _ in OURS + REFERENCES]
    extra = [n for n, _ in REFERENCES + HARDWARE if n not in BUILT_IN_ANCHORS]
    anchors = {f.name: f for f in anchor_families(paths, OPENH264_HARNESS, extra=extra)}
    binaries, presets_for = binaries or {}, presets_for or {}
    ours = {name: own_family(name, binaries.get(name, binary), presets_for.get(name, presets))
            for name, _ in OURS}
    unknown = [n for n in wanted if n not in anchors and n not in ours]
    if unknown:
        raise SystemExit(f"unknown points: {', '.join(unknown)}")
    chosen = [ours.get(n) or anchors[n] for n in wanted if n != ANCHOR]
    return [anchors[ANCHOR], *chosen]


def chart_clips(clipset: str | None) -> list[Clip]:
    paths = load_paths()
    return clipset_clips(clipset, paths) if clipset else natural_val(paths)


def low_ladder(clip: Clip) -> list[int]:
    return clip_targets(clip, LOW_LADDER_KBPS)


def low_rung_clips(curves: dict, families: list[Family], clips: list[Clip]) -> list[Clip]:
    """The clips on which some point (families[1:]) has no reliable direct overlap
    with the anchor (families[0]) on the real-use ladder."""
    anchor = families[0].name
    return [c for c in clips if any(
        per_clip_j(jreal.compare_clip(curves[(anchor, c.name)], curves[(f.name, c.name)]))
        is None for f in families[1:])]


def chart_curves(runner: Runner, families: list[Family], clips: list[Clip], frames: int,
                 label: str, rungs: tuple[int, ...] | None = None) -> tuple[dict, list[Clip]]:
    """Every family's curves, with the low rungs on the clips that need them
    (low_rung_clips, judged on the real-use ladder, or its `rungs` indices), and
    those clips."""
    curves = runner.measure(label, families, clips, frames, rungs)
    low = low_rung_clips(curves, families, clips)
    if low:
        extra = runner.measure(f"{label} low rungs", families, low, frames, ladder=low_ladder)
        for key, points in extra.items():
            curves[key] = curves[key] + points
    return curves, low


def measure(args: argparse.Namespace) -> None:
    """Measure (cache) the points of one part; refuse a part over the time budget."""
    families = chart_families(Path(args.binary).resolve(), Path(args.presets_dir),
                              args.points.split(",") if args.points else None,
                              binary_overrides(args.binary_for),
                              binary_overrides(args.presets_for))
    clips = chart_clips(args.clipset)
    if args.part:
        index, count = (int(x) for x in args.part.split("/"))
        clips = balanced_parts(clips, count)[index - 1]
    runner = Runner(load_paths(), EVALUATION_GATE, args.workers, args.over_budget,
                    stored_ms_ssim=not args.rescore, prebuilt=args.prebuilt)
    jobs = runner.points(families, clips, args.frames, args.rungs)
    minutes = estimate_minutes(runner.price(jobs))
    print(f"chart measure: {len(clips)} clips, {len(families)} families, {len(jobs)} points, "
          f"~{minutes:.0f} min", flush=True)
    if minutes > MAX_MINUTES and not args.over_budget:
        raise SystemExit(f"over the {MAX_MINUTES:g}-min budget: split with --part I/N "
                         f"(N >= {math.ceil(minutes / MAX_MINUTES)}) or fewer --points")
    # Which clips need low rungs is judged on the stored curves of every family
    # (any MS-SSIM protocol: the decision is about rate overlap), even on --rescore.
    judge = Runner(load_paths(), EVALUATION_GATE, args.workers, args.over_budget,
                   stored_ms_ssim=True, prebuilt=args.prebuilt)
    low = expected_low_rung_clips(args, judge, families, clips)
    if low:  # one pool: the low rungs fill the ladder's drain instead of following it
        print(f"low rungs expected on {', '.join(c.name for c in low)}", flush=True)
    runner.run_jobs(jobs + runner.points(families, low, args.frames, None, ladder=low_ladder))
    measure_low_rungs(args, runner, judge, families, clips)
    print("refused:", {k: sorted(v) for k, v in runner.refused.items()} or "none")


def measure_low_rungs(args: argparse.Namespace, runner: Runner, judge: Runner,
                      families: list[Family], clips: list[Clip]) -> None:
    """The low rungs of `families` on the clips that need them (measured by
    `runner`), once every chart point's real-use ladder is cached (which clips
    need them depends on all; judged by `judge`)."""
    every = chart_families(Path(args.binary).resolve(), Path(args.presets_dir),
                           binaries=binary_overrides(args.binary_for),
                           presets_for=binary_overrides(args.presets_for))
    if judge.price(judge.points(every, clips, args.frames, args.rungs)):
        print("low rungs: wait until every point's real-use ladder is cached")
        return
    low = low_rung_clips(judge.measure("chart", every, clips, args.frames, args.rungs), every,
                         clips)
    print(f"low rungs on {', '.join(c.name for c in low) or 'no clip'}", flush=True)
    if low:
        runner.measure("chart low rungs", families, low, args.frames, budgeted=False,
                       ladder=low_ladder)


def expected_low_rung_clips(args: argparse.Namespace, judge: Runner, families: list[Family],
                            clips: list[Clip]) -> list[Clip]:
    """The clips that will need low rungs whatever `families` measure: those the
    other, fully cached families already need them on (low_rung_clips takes the
    union over families, so this is a subset of the final set)."""
    measured = {f.name for f in families[1:]}
    every = chart_families(Path(args.binary).resolve(), Path(args.presets_dir),
                           binaries=binary_overrides(args.binary_for),
                           presets_for=binary_overrides(args.presets_for))
    known = [f for f in every if f.name not in measured]
    if len(known) < 2 or judge.price(judge.points(known, clips, args.frames, args.rungs)):
        return []
    return low_rung_clips(judge.measure("chart", known, clips, args.frames, args.rungs), known,
                          clips)


def binary_overrides(pairs: list[str]) -> dict[str, Path]:
    """`--binary-for` / `--presets-for` FAMILY=PATH pairs."""
    overrides = {}
    for pair in pairs:
        family, sep, path = pair.partition("=")
        if not sep:
            raise SystemExit(f"--binary-for / --presets-for need FAMILY=PATH, got {pair!r}")
        overrides[family] = Path(path).resolve()
    return overrides


# --- quality ---------------------------------------------------------------------------

def per_clip_j(row: dict) -> float | None:
    """One clip's BD-J: the mean of its full-overlap and low-rate quality BD."""
    if row.get("quality") is None or row.get("quality_low") is None:
        return None
    return (row["quality"] + row["quality_low"]) / 2


def balanced_pair(per_clip: dict, full: str, low: str) -> float | None:
    """(CB(full) + CB(low)) / 2, CB = class-balanced mean (jreal.aggregate)."""
    values = [jreal.aggregate(per_clip, term)["balanced"] for term in (full, low)]
    return None if None in values else sum(values) / 2


def interquartile(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def per_clip_rows(curves: dict, anchor: str, test: str, clips: list[Clip],
                  paired: bool = False) -> dict[str, dict]:
    """compare's per-clip terms of `test` vs `anchor`, each with its BD-J (`j`)."""
    rows = compare(curves, anchor, test, clips, paired=paired)
    for row in rows.values():
        row["j"] = per_clip_j(row)
    return rows


def compose(first: float | None, second: float | None) -> float | None:
    """Two BD-rates in percent chained: bitrate ratios multiply, (1+a)(1+b) - 1."""
    if first is None or second is None:
        return None
    return ((1 + first / 100) * (1 + second / 100) - 1) * 100


def chained_row(to_intermediate: dict, intermediate_to_anchor: dict) -> dict:
    """One clip's BD terms vs the anchor through an intermediate: every headline
    metric's full and low-rate BD composed, then BD-J from them as for a direct row.

    Each leg's low-rate half is that leg's own (its anchor's lower log-rate half),
    so the chained low-rate term approximates the direct one."""
    row = {"class": to_intermediate["class"]}
    for term in CHAINED_TERMS:
        row[term] = compose(to_intermediate.get(term), intermediate_to_anchor.get(term))
    row.update(jreal.headline_quality(row))
    row["j"] = per_clip_j(row)
    return row


@dataclass(frozen=True)
class Candidate:
    """A possible intermediate on one clip: its codec and side, the point's row vs
    it, its DIRECT row vs the anchor, and the smaller of the two legs' curve overlaps."""
    name: str
    codec: Codec
    own: bool
    to_intermediate: dict
    to_anchor: dict
    overlap: float


def lineage(codec: Codec) -> frozenset[Codec]:
    return next((group for group in LINEAGES if codec in group), frozenset({codec}))


def choose_intermediate(test: Family, candidates: list[Candidate]) -> Candidate | None:
    """The intermediate for one clip: among the candidates with a BD-J on both legs,
    the point's own lineage first, then its own side (ours / reference), then the
    best overlap, then by name (deterministic)."""
    usable = [c for c in candidates
              if c.to_intermediate.get("j") is not None and c.to_anchor.get("j") is not None]
    if not usable:
        return None
    return min(usable, key=lambda c: (c.codec not in lineage(test.codec), c.own != test.own,
                                      -c.overlap, c.name))


def chain_clip(curves: dict, anchor: str, test: Family, clip: Clip, families: list[Family],
               direct: dict[str, dict[str, dict]]) -> tuple[ChainedClip, dict] | None:
    """`test` vs `anchor` on `clip` through one intermediate, or None.

    `direct[family][clip]` holds every family's DIRECT row vs the anchor; the
    intermediate's leg comes only from there, so a chain never has more than one
    hop (a clip chained for one family never serves another's chain)."""
    candidates = []
    for family in families:
        if family.name in (anchor, test.name) or direct[family.name][clip.name]["j"] is None:
            continue
        a, t = curves[(family.name, clip.name)], curves[(test.name, clip.name)]
        leg = {"class": clip.content_class, **jreal.compare_clip(a, t)}
        leg["j"] = per_clip_j(leg)
        overlap = min(jreal.curve_overlap(a, t),
                      jreal.curve_overlap(curves[(anchor, clip.name)], a))
        candidates.append(Candidate(family.name, family.codec, family.own, leg,
                                    direct[family.name][clip.name], overlap))
    chosen = choose_intermediate(test, candidates)
    if chosen is None:
        return None
    row = chained_row(chosen.to_intermediate, chosen.to_anchor)
    if row["j"] is None:
        return None
    return ChainedClip(clip.name, chosen.name, chosen.to_intermediate["j"],
                       chosen.to_anchor["j"], row["j"]), row


def chart_point(test: Family, label: str, per_clip: dict[str, dict], clips: list[Clip],
                runner: Runner, chained: dict[str, tuple[ChainedClip, dict]] | None = None
                ) -> ChartPoint:
    """One chart point from its direct per-clip rows and any chained clips' rows."""
    chained = chained or {}
    rows = {clip: chained[clip][1] if clip in chained else row for clip, row in per_clip.items()}
    scored = [row["j"] for row in rows.values() if row["j"] is not None]
    scored_classes = {row["class"] for row in rows.values() if row["j"] is not None}
    return ChartPoint(
        family=test.name, label=label, ours=test.own,
        bd_j=balanced_pair(rows, "quality", "quality_low"),
        bd_psnr=balanced_pair(rows, "bd_psnr_yuv", "low_psnr_yuv"),
        bd_vmaf=balanced_pair(rows, "bd_vmaf_v1", "low_vmaf_v1"),
        iqr=interquartile(scored), scored=len(scored), clips=len(clips),
        missing_classes=tuple(sorted({c.content_class for c in clips} - scored_classes)),
        rate_invalid=len(runner.unreachable.get(test.name, ())),
        refused=len(runner.refused.get(test.name, ())),
        chained=tuple(chained[clip][0] for clip in sorted(chained)),
        direct_bd_j=balanced_pair(per_clip, "quality", "quality_low"))


def versus(curves: dict, anchor: str, test: Family, clips: list[Clip], runner: Runner,
           label: str, paired: bool = False) -> ChartPoint:
    """`test` vs `anchor` on the cached curves, directly, as one chart point
    (`paired`: ours vs ours, so rungs either arm overshot are dropped, as on the board)."""
    return chart_point(test, label, per_clip_rows(curves, anchor, test.name, clips, paired),
                       clips, runner)


@dataclass(frozen=True)
class Excluded:
    """A clip left out for every point: some point has no reliable direct overlap
    with the anchor on it. `direct` holds each point's direct BD-J (None: none)."""
    clip: str
    content_class: str
    unscored: tuple[str, ...]
    direct: tuple[tuple[str, float | None], ...]


def common_clips(direct: dict[str, dict[str, dict]], points: list[str], clips: list[Clip]
                 ) -> list[Clip]:
    """The clips on which every point has a direct BD-J vs the anchor."""
    return [c for c in clips if all(direct[name][c.name]["j"] is not None for name in points)]


def rescue_classes(curves: dict, anchor: str, families: list[Family], clips: list[Clip],
                   common: list[Clip], direct: dict[str, dict[str, dict]]
                   ) -> dict[str, dict[str, tuple[ChainedClip, dict]]]:
    """Chains for the clips of classes the common set would lose entirely: a clip is
    rescued only when EVERY point unscored on it chains (chain_clip), so all points
    still share one clip set. {clip: {family: (ChainedClip, row)}}."""
    kept = {c.content_class for c in common}
    rescued = {}
    for clip in clips:
        if clip in common or clip.content_class in kept:
            continue
        links = {f.name: chain_clip(curves, anchor, f, clip, families, direct)
                 for f in families[1:] if direct[f.name][clip.name]["j"] is None}
        if all(links.values()):
            rescued[clip.name] = links
    return rescued


def anchor_points(curves: dict, families: list[Family], clips: list[Clip], runner: Runner,
                  labels: dict[str, str]) -> tuple[list[ChartPoint], list[Excluded]]:
    """Every family vs the anchor (families[0]) on ONE clip set shared by all points:
    the clips where every point overlaps the anchor directly, plus, only for a class
    that would otherwise vanish, its clips on which every unscored point chains
    (rescue_classes). Returns the points and the clips left out for everyone."""
    anchor = families[0].name
    direct = {f.name: per_clip_rows(curves, anchor, f.name, clips) for f in families}
    common = common_clips(direct, [f.name for f in families], clips)
    rescued = rescue_classes(curves, anchor, families, clips, common, direct)
    used = [c for c in clips if c in common or c.name in rescued]
    points = []
    for family in families:
        chained = {clip: links[family.name] for clip, links in rescued.items()
                   if family.name in links}
        rows = {c.name: direct[family.name][c.name] for c in used}
        points.append(chart_point(family, labels[family.name], rows, used, runner, chained))
    excluded = [Excluded(c.name, c.content_class,
                         tuple(f.name for f in families if direct[f.name][c.name]["j"] is None),
                         tuple((f.name, direct[f.name][c.name]["j"]) for f in families[1:]))
                for c in clips if c not in used]
    return points, excluded


@dataclass(frozen=True)
class Quality:
    """The quality side of the chart: points vs the anchor, the selected pairs as
    (reference label, pair), the clips left out for all, the clips with low rungs,
    and the curves and clip set the points were averaged over."""
    points: list[ChartPoint]
    pairs: list[tuple[str, ChartPoint]]
    excluded: list[Excluded]
    low_rungs: list[Clip]
    curves: dict
    used: list[Clip]


def quality_points(runner: Runner, families: list[Family], clips: list[Clip],
                   frames: int, rungs: tuple[int, ...] | None = None) -> Quality:
    """BD of every family vs the anchor (families[0]) and of SELECTED_PAIRS, from
    cached points (`rungs`: those indices of the real-use ladder only)."""
    curves, low = chart_curves(runner, families, clips, frames, "chart render", rungs)
    labels = dict(OURS + REFERENCES)
    by_name = {f.name: f for f in families}
    points, excluded = anchor_points(curves, families, clips, runner, labels)
    pairs = [(labels[ref], versus(curves, ref, by_name[ours], clips, runner, labels[ours],
                                  paired=by_name[ref].own))
             for ours, ref in SELECTED_PAIRS if ours in by_name and ref in by_name]
    left_out = {e.clip for e in excluded}
    return Quality(points, pairs, excluded, low, curves,
                   [c for c in clips if c.name not in left_out])


def hardware_points(runner: Runner, quality: Quality, families: list[Family],
                    hardware: list[Family], clips: list[Clip], frames: int,
                    rungs: tuple[int, ...] | None = None) -> list[ChartPoint]:
    """BD of each hardware family vs the anchor (families[0]) on exactly the chart's
    clip set and rungs (`quality`: its low rungs where the chart has them). A clip
    where it has no direct overlap is chained through the chart's own families,
    as for a rescued clip (chain_clip); they never change the chart's clip set."""
    anchor = families[0].name
    curves = {**quality.curves, **runner.measure("chart hardware", hardware, clips, frames,
                                                 rungs)}
    if quality.low_rungs:
        extra = runner.measure("chart hardware low rungs", hardware, quality.low_rungs, frames,
                               ladder=low_ladder)
        for key, points in extra.items():
            curves[key] = curves[key] + points
    direct = {f.name: per_clip_rows(curves, anchor, f.name, clips) for f in families}
    labels = dict(HARDWARE)
    points = []
    for family in hardware:
        rows = per_clip_rows(curves, anchor, family.name, quality.used)
        chained = {}
        for clip in quality.used:
            if rows[clip.name]["j"] is None:
                link = chain_clip(curves, anchor, family, clip, families, direct)
                if link:
                    chained[clip.name] = link
        points.append(chart_point(family, labels[family.name], rows, quality.used, runner,
                                  chained))
    return points


def pairs_table(pairs: list[tuple[str, ChartPoint]]) -> str:
    lines = ["| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |",
             "|---|---|---:|---:|---:|---|---:|"]
    for reference, p in pairs:
        iqr = f"{p.iqr[0]:+.1f} … {p.iqr[1]:+.1f}" if p.iqr else "n/a"
        mark = PARTIAL_MARK if p.partial else ""
        lines.append(f"| {p.label} | {reference} | {fmt(p.bd_j)}{mark} | {fmt(p.bd_psnr)} | "
                     f"{fmt(p.bd_vmaf)} | {iqr} | {p.scored}/{p.clips} |")
    return "\n".join(lines) + "\n"


# --- speed -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeedPlatform:
    """Where the speed axis was timed: output suffix, CSV column, axis title, and the
    core named in the chart title."""
    suffix: str
    column: str
    axis: str
    core: str


JETSON = SpeedPlatform("", "fps_jetson", "single-thread encode fps, 720p, one Jetson Orin "
                       "Nano core (log scale; faster →)",
                       "one Arm Cortex-A78AE core (NVIDIA Jetson Orin Nano)")
X86 = SpeedPlatform("-x86", "fps_x86", "single-thread encode fps, 720p, one Ryzen 9 5950X "
                    "core (log scale; faster →)", "one x86-64 core (AMD Ryzen 9 5950X)")


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def with_speed(points: list[ChartPoint], timing: dict) -> list[ChartPoint]:
    """Attach fps (geometric mean over the timing clips) and speed vs the anchor.

    `timing["points"][family]["ms_per_frame"][clip]` holds medians; speed is the
    geometric mean of the per-clip anchor/point time ratios, against the anchor
    runs paired with the point (`anchor_ms_per_frame`: timed in the same lock,
    rotated with it) when present.
    """
    entries = timing["points"]
    result = []
    for point in points:
        entry = entries.get(point.family)
        if not entry:
            result.append(point)
            continue
        ms = entry["ms_per_frame"]
        anchor = entry.get("anchor_ms_per_frame") or entries[ANCHOR]["ms_per_frame"]
        speed = geomean([anchor[clip] / ms[clip] for clip in ms])
        fps = 1000.0 / geomean(list(ms.values()))
        result.append(ChartPoint(**{**point.__dict__, "fps": fps, "speed": speed}))
    return result


@dataclass(frozen=True)
class HardwareSpeed:
    """A hardware encoder's speeds (tools/dev/nvenc_timing.py), frames per second at
    720p: the best aggregate throughput and its stream count, one stream encoding
    frames already in GPU memory, and one stream through ffmpeg's read-and-upload
    pipeline (the earlier, pipeline-bound measure); the last two are geometric means
    over the timing clips."""
    aggregate: float
    streams: int
    encode: float
    pipeline: float | None


def hardware_speeds(timing: dict) -> dict[str, HardwareSpeed]:
    speeds = {}
    for family, entry in timing["points"].items():
        streams, aggregate = max(((int(n), fps) for n, fps in entry["aggregate_fps"].items()),
                                 key=lambda pair: pair[1])
        pipeline = entry.get("pipeline_fps")
        speeds[family] = HardwareSpeed(aggregate, streams,
                                       geomean(list(entry["encode_fps"].values())),
                                       geomean(list(pipeline.values())) if pipeline else None)
    return speeds


def with_hardware_fps(points: list[ChartPoint], speeds: dict[str, HardwareSpeed]
                      ) -> list[ChartPoint]:
    """Attach the hardware encoders' aggregate throughput as their fps (what the x86
    chart plots). Their speed vs the anchor stays None: GPU time is not CPU time."""
    return [ChartPoint(**{**p.__dict__, "fps": speeds[p.family].aggregate})
            if p.family in speeds else p for p in points]


def pareto(points: list[ChartPoint]) -> list[ChartPoint]:
    """Unflagged timed points no other point beats on both speed and BD-J."""
    usable = [p for p in points if not p.flagged and p.fps is not None]
    front, best = [], math.inf
    for point in sorted(usable, key=lambda p: (-p.fps, p.bd_j)):
        if point.bd_j < best:
            front.append(point)
            best = point.bd_j
    return front


# --- output: CSV, markdown, SVG --------------------------------------------------------

def fmt(value: float | None, spec: str = "+.1f", unit: str = "%") -> str:
    return "n/a" if value is None else f"{value:{spec}}{unit}"


def hardware_csv(points: list[ChartPoint], speeds: dict[str, HardwareSpeed]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["family", "label", "bd_j_pct", "bd_psnr_yuv_pct", "bd_vmaf_v1_pct",
                     "iqr_low_pct", "iqr_high_pct", "clips_scored", "clips",
                     "fps_aggregate", "aggregate_streams", "fps_encode_1stream",
                     "fps_ffmpeg_pipeline", "flagged", "rate_invalid_points", "clips_chained",
                     "bd_j_direct_pct"])
    num = (lambda v, d: "" if v is None else f"{v:.{d}f}")
    for p in points:
        speed = speeds.get(p.family)
        writer.writerow([p.family, p.label, num(p.bd_j, 2), num(p.bd_psnr, 2),
                         num(p.bd_vmaf, 2), num(p.iqr and p.iqr[0], 2),
                         num(p.iqr and p.iqr[1], 2), p.scored, p.clips,
                         num(speed and speed.aggregate, 1), speed.streams if speed else "",
                         num(speed and speed.encode, 1), num(speed and speed.pipeline, 1),
                         int(p.flagged), p.rate_invalid, len(p.chained),
                         num(p.direct_bd_j, 2)])
    return out.getvalue()


def csv_text(points: list[ChartPoint], front: list[ChartPoint],
             platform: SpeedPlatform = JETSON) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["family", "label", "ours", "bd_j_pct", "bd_psnr_yuv_pct", "bd_vmaf_v1_pct", "iqr_low_pct",
                     "iqr_high_pct", "clips_scored", "clips", platform.column,
                     "speed_vs_anchor", "pareto", "flagged", "rate_invalid_points",
                     "clips_chained", "bd_j_direct_pct", "chained"])
    for p in points:
        num = (lambda v, d: "" if v is None else f"{v:.{d}f}")
        chained = ";".join(f"{c.clip}:{c.intermediate}:{c.j:.2f}" for c in p.chained)
        writer.writerow([p.family, p.label, int(p.ours), num(p.bd_j, 2), num(p.bd_psnr, 2), num(p.bd_vmaf, 2),
                         num(p.iqr and p.iqr[0], 2), num(p.iqr and p.iqr[1], 2), p.scored,
                         p.clips, num(p.fps, 2), num(p.speed, 3), int(p in front),
                         int(p.flagged), p.rate_invalid, len(p.chained),
                         num(p.direct_bd_j, 2), chained])
    return out.getvalue()


def markdown_table(points: list[ChartPoint], front: list[ChartPoint],
                   x86: list[ChartPoint] | None = None) -> str:
    """The chart's table; `x86` (the same points with x86 speed) adds its columns."""
    on_x86 = {p.family: p for p in x86 or []}
    x86_front = {p.family for p in pareto(x86 or [])}
    x86_head = " fps (x86, 1 core) | x86 speed vs x264 medium |" if x86 else ""
    chains = any(p.chained for p in points)  # the direct-only column only beside chains
    lines = ["| Encoder | BD-J |" + (" BD-J direct only |" if chains else "")
             + " BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) "
             f"| fps (Jetson, 1 core) | speed vs x264 medium |{x86_head} clips | notes |",
             "|---|---:|" + ("---:|" if chains else "") + "---:|---:|---|---:|---:|"
             + ("---:|---:|" if x86 else "") + "---:|---|"]
    for p in sorted(points, key=lambda p: (p.bd_j is None, p.bd_j or 0)):
        name = f"**{p.label}**" if p.ours else p.label
        iqr = f"{p.iqr[0]:+.1f} … {p.iqr[1]:+.1f}" if p.iqr else "n/a"
        notes = []
        if p in front:
            notes.append("Pareto")
        if p.family in x86_front:
            notes.append("Pareto x86")
        if p.flagged:
            notes.append("FLAGGED: too little curve overlap")
        elif p.partial:
            notes.append(f"{PARTIAL_MARK} no {', '.join(p.missing_classes)} clip with a "
                         f"reliable overlap: mean over the other classes")
        if p.chained:
            notes.append(f"{CHAIN_MARK} {chained_count(p)}")
        if p.rate_invalid:
            notes.append(f"{p.rate_invalid} rate-invalid pts")
        if p.refused:
            notes.append(f"{p.refused} pts refused")
        direct = (fmt(p.direct_bd_j) if p.chained else "") + " | " if chains else ""
        lines.append(f"| {name} | {fmt(p.bd_j)}{CHAIN_MARK if p.chained else ''} | {direct}"
                     f"{fmt(p.bd_psnr)} | {fmt(p.bd_vmaf)} | {iqr} | "
                     f"{fmt(p.fps, '.1f', '')} | {fmt(p.speed, '.2f', '×')} | "
                     + (f"{fmt(on_x86[p.family].fps, '.1f', '')} | "
                        f"{fmt(on_x86[p.family].speed, '.2f', '×')} | " if x86 else "")
                     + f"{p.scored}/{p.clips} | {'; '.join(notes)} |")
    return "\n".join(lines) + "\n"


def hardware_table(points: list[ChartPoint], speeds: dict[str, HardwareSpeed],
                   platform: str) -> str:
    """The hardware encoders' section: BD vs x264 medium on the chart's clips and
    their speeds (HardwareSpeed)."""
    lines = ["## Hardware encoders\n",
             f"NVIDIA NVENC ({platform or 'GPU'}) through ffmpeg, in low-delay CBR on the "
             "same clips, rungs and buffer as every point above, no B-frames; scored the "
             "same way; p4 is preset P4 with `-tune ll`, p7 preset P7 with `-tune hq`. "
             "Speeds are GPU wall-clock frames per second at 720p (fixed-function hardware, "
             "not CPU time; tools/dev/nvenc_timing.py): the aggregate throughput of N "
             "parallel streams (the best of 1, 2, 4 and 8) and one stream, both encoding "
             "frames already in GPU memory, and one stream through ffmpeg's "
             "read-convert-upload pipeline (the earlier figure, bound by that pipeline). "
             "The x86 chart plots the aggregate, never on the Pareto front; the Jetson "
             "chart leaves them out.\n",
             "| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) "
             "| aggregate fps (N streams) | 1-stream encode fps | 1-stream fps via ffmpeg "
             "pipeline | clips | notes |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for p in sorted(points, key=lambda p: (p.bd_j is None, p.bd_j or 0)):
        iqr = f"{p.iqr[0]:+.1f} … {p.iqr[1]:+.1f}" if p.iqr else "n/a"
        notes = [f"{CHAIN_MARK} {chained_count(p)}"] if p.chained else []
        if p.flagged:
            notes.append("FLAGGED: too little curve overlap")
        if p.rate_invalid:
            notes.append(f"{p.rate_invalid} rate-invalid pts")
        speed = speeds.get(p.family)
        rates = (f"{speed.aggregate:.0f} ({speed.streams}) | {speed.encode:.0f} | "
                 f"{fmt(speed.pipeline, '.0f', '')}" if speed else "n/a | n/a | n/a")
        lines.append(f"| {p.label} | {fmt(p.bd_j)}{CHAIN_MARK if p.chained else ''} | "
                     f"{fmt(p.bd_psnr)} | {fmt(p.bd_vmaf)} | {iqr} | {rates} | "
                     f"{p.scored}/{p.clips} | {'; '.join(notes)} |")
    return "\n".join(lines) + "\n"


def chained_count(p: ChartPoint) -> str:
    return f"{len(p.chained)} clip{'s' if len(p.chained) > 1 else ''} chained"


def chained_table(points: list[ChartPoint]) -> str:
    """Every chained clip: the intermediate and both legs (empty when none)."""
    rows = [(p, c) for p in sorted(points, key=lambda p: (p.bd_j is None, p.bd_j or 0))
            for c in p.chained]
    if not rows:
        return ""
    labels = {p.family: p.label for p in points}
    lines = [f"## Chained clips ({CHAIN_MARK})\n",
             "Clips without a reliable overlap with x264 medium, measured through one "
             "intermediate encoder (see Method).\n",
             "| Encoder | clip | via | BD-J vs via | via vs x264 medium | chained BD-J |",
             "|---|---|---|---:|---:|---:|"]
    for p, c in rows:
        lines.append(f"| {p.label} | {c.clip} | {labels.get(c.intermediate, c.intermediate)} | "
                     f"{fmt(c.to_intermediate)} | {fmt(c.intermediate_to_anchor)} | {fmt(c.j)} |")
    return "\n".join(lines) + "\n"


def coverage_note(quality: Quality) -> str:
    """Which clips got the low rungs, and how many are left out or chained."""
    low = ", ".join(c.name for c in quality.low_rungs) or "none"
    chained = sum(len(p.chained) for p in quality.points)
    return (f"Low rungs ({' / '.join(map(str, LOW_LADDER_KBPS))} kbps, every encoder) on: "
            f"{low}. Clips left out for every encoder: {len(quality.excluded)}; "
            f"chained (point, clip) pairs: {chained}.\n")


def excluded_table(excluded: list[Excluded], points: list[ChartPoint]) -> str:
    """The clips left out for every point, why, and each point's direct BD-J there."""
    labels = {p.family: p.label for p in points}
    lines = ["## Excluded clips\n",
             "Left out for every encoder, so that all points are averaged over the same "
             "clips: on these, some encoders' curves have no reliable overlap with x264 "
             "medium's (even with its extra rungs, which the 50% source cap limits).\n"]
    for e in excluded:
        names = ", ".join(labels.get(n, n) for n in e.unscored)
        lines.append(f"- **{e.clip}** ({e.content_class}): no reliable overlap for {names}.")
    lines += ["", "Direct BD-J vs x264 medium on them, where it exists:\n",
              "| Encoder | " + " | ".join(e.clip for e in excluded) + " |",
              "|---|" + "---:|" * len(excluded)]
    for p in sorted(points, key=lambda p: (p.bd_j is None, p.bd_j or 0)):
        if p.family == ANCHOR:
            continue
        cells = [fmt(dict(e.direct).get(p.family)) for e in excluded]
        lines.append(f"| {p.label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Frame:
    """Plot geometry: the data window and the pixel box it maps to."""
    fps_lo: float
    fps_hi: float
    bd_lo: float
    bd_hi: float
    left: int = 70
    top: int = 40
    width: int = 900
    height: int = 700

    def x(self, fps: float) -> float:
        span = math.log(self.fps_hi) - math.log(self.fps_lo)
        return self.left + (math.log(fps) - math.log(self.fps_lo)) / span * self.width

    def y(self, bd: float) -> float:
        return self.top + (self.bd_hi - bd) / (self.bd_hi - self.bd_lo) * self.height


#: Rounds per clip of the x86 timing (anchor_chart_timing.HOST_ROUNDS), for the note.
X86_ROUNDS = 5
#: Colours (dataviz reference palette, light mode): ours vs reference encoders;
#: marker shape names the reference program, so identity is never colour alone.
OURS_COLOUR, REFERENCE_COLOUR, FRONT_COLOUR = "#2a78d6", "#eb6834", "#52514e"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MARKER_RADIUS = 6
#: Hardware encoders: their marker colour, the gutter (px) right of the plot that holds
#: their off-scale markers and labels, and the least vertical gap between two labels.
HARDWARE_COLOUR = "#2e8540"
HARDWARE_GUTTER = 200
HARDWARE_LABEL_GAP = 14
#: Height (px) of one more footnote line.
NOTE_LINE = 16
#: Plot-window padding around the data: log-fps factor and BD percentage points.
FPS_PAD, BD_PAD = 1.4, 6.0


def is_hardware(p: ChartPoint) -> bool:
    return p.family in HARDWARE_FAMILIES


def marker(p: ChartPoint, cx: float, cy: float) -> str:
    """Circle for ours; square x264, diamond x265, triangle SVT, inverted triangle
    openh264, X libvpx, + libaom, green hexagon a hardware encoder. Flagged points
    are hollow."""
    r = MARKER_RADIUS
    if is_hardware(p):
        pts = " ".join(f"{cx + (r + 1) * math.cos(math.pi / 3 * k):.1f},"
                       f"{cy + (r + 1) * math.sin(math.pi / 3 * k):.1f}" for k in range(6))
        fill = SURFACE if p.flagged else HARDWARE_COLOUR
        return f'<polygon points="{pts}" fill="{fill}" stroke="{HARDWARE_COLOUR}" stroke-width="2"/>'

    colour = OURS_COLOUR if p.ours else REFERENCE_COLOUR
    width = 1 if p.family.startswith(("libvpx", "libaom")) else 2
    style = f'fill="{SURFACE if p.flagged else colour}" stroke="{colour}" stroke-width="{width}"'
    if p.ours:
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" {style}/>'
    if p.family.startswith("x264"):
        return f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" {style}/>'
    if p.family.startswith("x265"):
        pts = f"{cx},{cy - r - 1} {cx + r + 1},{cy} {cx},{cy + r + 1} {cx - r - 1},{cy}"
        return f'<polygon points="{pts}" {style}/>'
    if p.family.startswith("svtav1"):
        pts = f"{cx},{cy - r - 1} {cx + r + 1},{cy + r} {cx - r - 1},{cy + r}"
        return f'<polygon points="{pts}" {style}/>'
    if p.family.startswith(("libvpx", "libaom")):
        # a thick X for libvpx, a thick + for libaom (outlined like the others)
        arm, half = r + 1, 2.5
        cross = [(-half, -arm), (half, -arm), (half, -half), (arm, -half), (arm, half),
                 (half, half), (half, arm), (-half, arm), (-half, half), (-arm, half),
                 (-arm, -half), (-half, -half)]
        turn = 0.7071 if p.family.startswith("libvpx") else 1.0
        sin = 0.7071 if p.family.startswith("libvpx") else 0.0
        pts = " ".join(f"{cx + x * turn - y * sin:.1f},{cy + x * sin + y * turn:.1f}"
                       for x, y in cross)
        return f'<polygon points="{pts}" {style}/>'
    pts = f"{cx},{cy + r + 1} {cx + r + 1},{cy - r} {cx - r - 1},{cy - r}"
    return f'<polygon points="{pts}" {style}/>'


#: Label positions tried around a marker, nearest first: (dx, dy, text-anchor).
#: Positions from LEADER_FROM on are far enough away to need a leader line.
LABEL_OFFSETS = ((10, 4, "start"), (-10, 4, "end"), (0, -11, "middle"), (0, 20, "middle"),
                 (10, -10, "start"), (10, 18, "start"), (-10, -10, "end"), (-10, 18, "end"),
                 (0, -26, "middle"), (0, 34, "middle"), (16, -24, "start"), (-16, -24, "end"),
                 (16, 32, "start"), (-16, 32, "end"), (0, -40, "middle"), (0, 48, "middle"),
                 (24, -38, "start"), (-24, -38, "end"), (24, 46, "start"), (-24, 46, "end"),
                 (0, -54, "middle"), (0, 62, "middle"), (32, -52, "start"), (-32, -52, "end"),
                 (32, 60, "start"), (-32, 60, "end"), (0, -70, "middle"), (0, 78, "middle"))
LEADER_FROM = 8
CHAR_WIDTH = {False: 6.6, True: 7.3}  # px per character at 12 px, regular / bold
#: A label without a leader must be at least this much nearer its own marker than
#: any other marker (px), so that a reader pairs it with the right point.
LABEL_CLEARANCE = 4.0


def chart_label(p: ChartPoint) -> str:
    """A point's label on the chart, with its flagged / partial mark. Chained clips
    are marked in the table only: most points have one, and the marks crowd the plot."""
    return p.label + (" (flagged)" if p.flagged else PARTIAL_MARK if p.partial else "")


def box_distance(box: tuple[float, float, float, float], x: float, y: float) -> float:
    """Distance from a point to the nearest edge of a box (0 inside it)."""
    return math.hypot(max(box[0] - x, 0.0, x - box[2]), max(box[1] - y, 0.0, y - box[3]))


Box = tuple[float, float, float, float]


def overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def marker_box(cx: float, cy: float) -> Box:
    return (cx - 8, cy - 8, cx + 8, cy + 8)


def place_labels(items: list[tuple[ChartPoint, float, float]], bars: list[Box]
                 ) -> dict[str, tuple[float, float, str, bool]]:
    """Greedy label placement: our points first, then top to bottom; each label
    takes the nearest LABEL_OFFSETS spot that overlaps no marker, spread bar
    (boxes in `bars`) or earlier label and, unless it gets a leader line, is unambiguous (nearer its own
    marker than any other by LABEL_CLEARANCE). Returns (x, y, text-anchor,
    needs a leader line) per family.

    Hardware encoders (is_hardware) do not move the CPU encoders' labels from
    where they would be without them, except a label that would lie on a hardware
    marker: only that one is placed again, against everything placed so far. The
    hardware labels come last."""
    cpu = [i for i in items if not is_hardware(i[0])]
    hardware = [i for i in items if is_hardware(i[0])]
    taken = [marker_box(cx, cy) for _, cx, cy in cpu] + list(bars)
    placed: dict[str, tuple[float, float, str, bool]] = {}
    boxes: dict[str, Box] = {}
    for item in sorted(cpu, key=lambda i: (not i[0].ours, i[2])):
        boxes[item[0].family], placed[item[0].family] = _place_label(item, cpu, taken)
        taken.append(boxes[item[0].family])
    hardware_markers = [marker_box(cx, cy) for _, cx, cy in hardware]
    taken += hardware_markers
    for item in sorted(cpu, key=lambda i: (not i[0].ours, i[2])):
        family = item[0].family
        if any(overlaps(boxes[family], m) for m in hardware_markers):
            taken.remove(boxes[family])
            boxes[family], placed[family] = _place_label(item, items, taken,
                                                         avoid=hardware_markers)
            taken.append(boxes[family])
    for item in sorted(hardware, key=lambda i: i[2]):
        box, placed[item[0].family] = _place_label(item, items, taken)
        taken.append(box)
    return placed


def _place_label(item: tuple[ChartPoint, float, float],
                 neighbours: list[tuple[ChartPoint, float, float]], taken: list[Box],
                 avoid: list[Box] = ()) -> tuple[Box, tuple[float, float, str, bool]]:
    """The nearest usable LABEL_OFFSETS spot for one point's label (place_labels),
    unambiguous among `neighbours`: its box and its (x, y, text-anchor, leader).
    Without one it takes the first spot, or with `avoid`, the nearest spot that
    overlaps none of those boxes and the fewest others."""
    p, cx, cy = item
    w, h = CHAR_WIDTH[p.ours] * (len(chart_label(p)) + 1) + 4, 14
    others = [(ox, oy) for q, ox, oy in neighbours if q.family != p.family]

    def box(index: int) -> Box:
        dx, dy, anchor = LABEL_OFFSETS[index]
        x0 = cx + dx - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
        return (x0, cy + dy - h + 3, x0 + w, cy + dy + 3)

    def usable(index: int) -> bool:
        b = box(index)
        if any(overlaps(b, t) for t in taken):
            return False
        own = box_distance(b, cx, cy)
        return index >= LEADER_FROM or all(
            own + LABEL_CLEARANCE < box_distance(b, ox, oy) for ox, oy in others)

    index = next((i for i in range(len(LABEL_OFFSETS)) if usable(i)), None)
    if index is None:
        index = min(range(len(LABEL_OFFSETS)), key=lambda i: (
            sum(overlaps(box(i), a) for a in avoid),
            sum(overlaps(box(i), t) for t in taken), i)) if avoid else 0
    dx, dy, anchor = LABEL_OFFSETS[index]
    return box(index), (cx + dx, cy + dy, anchor, index >= LEADER_FROM)


#: Spacing (px) of the small boxes that stand for a polyline in label placement.
POLYLINE_STEP = 4.0


def polyline_boxes(vertices: list[tuple[float, float]]) -> list[tuple[float, float, float, float]]:
    """A polyline as small boxes every POLYLINE_STEP px, so that labels avoid it."""
    boxes = []
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:]):
        steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / POLYLINE_STEP))
        for i in range(steps + 1):
            x, y = x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps
            boxes.append((x - 1, y - 1, x + 1, y + 1))
    return boxes


def log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for decade in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        ticks += [m * 10 ** decade for m in (1, 2, 5) if lo <= m * 10 ** decade <= hi]
    return ticks


def updated_line(meta: dict, date: str) -> str:
    """The "Updated <date>, build <sha>" stamp of every rendered output: the render date and
    the build(s) measured (`--meta`'s "build", else its "commit")."""
    build = meta.get("build") or meta.get("commit")
    return f"Updated {date}" + (f", build {build}" if build else "")


#: The chart's footnote on what its bars and markers show.
BARS_NOTE = ("Bars: interquartile range of per-clip BD. Every point is averaged over the same "
             "clips (see the table for excluded clips). Hollow: flagged.")


def hardware_chevron(cx: float, cy: float) -> str:
    """A right-pointing chevron: a point whose speed is off the axis's right end."""
    r = MARKER_RADIUS
    pts = (f"{cx - r:.1f},{cy - r:.1f} {cx:.1f},{cy - r:.1f} {cx + r + 2:.1f},{cy:.1f} "
           f"{cx:.1f},{cy + r:.1f} {cx - r:.1f},{cy + r:.1f} {cx + 2 - r / 2:.1f},{cy:.1f}")
    return f'<polygon points="{pts}" fill="{HARDWARE_COLOUR}" stroke="{HARDWARE_COLOUR}"/>'


def off_scale_markers(hardware: list[ChartPoint], frame: Frame) -> list[str]:
    """Hardware encoders faster than the axis's right end: chevrons past the plot's
    edge, each at its own BD-J height, labelled with its fps. Labels that would
    overlap move down, joined to their marker by a leader."""
    edge = frame.left + frame.width
    out = [f'<line x1="{edge}" x2="{edge}" y1="{frame.top}" y2="{frame.top + frame.height}" '
           f'stroke="{MUTED}" stroke-dasharray="2 3"/>',
           f'<text x="{edge + 10}" y="{frame.top - 6}" fill="{HARDWARE_COLOUR}">'
           f'GPU hardware, off scale →</text>']
    floor = -math.inf
    for p in sorted((p for p in hardware if p.bd_j is not None), key=lambda p: -p.bd_j):
        cy = frame.y(p.bd_j)
        ty = max(cy, floor + HARDWARE_LABEL_GAP)
        floor = ty
        fps = f" · {p.fps:.0f} fps" if p.fps else ""
        out.append(f'<g><title>{p.label}: BD-J {p.bd_j:+.1f}%, BD-PSNR {fmt(p.bd_psnr)}, '
                   f'BD-VMAF v1 {fmt(p.bd_vmaf)}{fps} (GPU wall time)</title>'
                   f'{hardware_chevron(edge + 14, cy)}</g>')
        if ty - cy > MARKER_RADIUS:
            out.append(f'<line x1="{edge + 22}" y1="{cy:.1f}" x2="{edge + 27}" '
                       f'y2="{ty - 4:.1f}" stroke="{MUTED}" stroke-width="0.75"/>')
        out.append(f'<text x="{edge + 28}" y="{ty + 4:.1f}" fill="{INK}">{p.label}{fps}</text>')
    return out


#: The chart's footnote on hardware points (with `hardware`).
def hardware_note(speeds: dict[str, HardwareSpeed]) -> str:
    """The chart's footnote on its hardware points: what their fps is."""
    counts = sorted({s.streams for s in speeds.values()})
    streams = (f"{counts[0]} parallel stream{'s' if counts[0] > 1 else ''}" if len(counts) == 1
               else f"{'/'.join(map(str, counts))} parallel streams (the best of each)")
    return (f"NVENC ({HARDWARE_GPU} GPU): aggregate encode throughput, {streams}, frames in "
            "GPU memory; GPU time, not CPU, and never on the Pareto front.")


def svg_chart(points: list[ChartPoint], front: list[ChartPoint], subtitle: str,
              platform: SpeedPlatform = JETSON, stamp: str = "",
              heading: str = "Bitrate vs single-threaded encode speed",
              note: str = BARS_NOTE, hardware: list[ChartPoint] = (),
              hardware_footnote: str = "") -> str:
    """The chart. `hardware` points (GPU encoders, the PC's chart only) are drawn at
    their fps in their own style, or past the right edge when faster than the axis
    (off_scale_markers); the axis is set by `points` alone."""
    timed = [p for p in points if p.fps is not None and p.bd_j is not None]
    bds = [v for p in timed for v in ((p.iqr or (p.bd_j, p.bd_j)) + (p.bd_j,))]
    hardware = [p for p in hardware if p.fps is not None and p.bd_j is not None]
    bds += [p.bd_j for p in hardware]
    frame = Frame(min(p.fps for p in timed) / FPS_PAD, max(p.fps for p in timed) * FPS_PAD,
                  min(bds) - BD_PAD, max(bds) + BD_PAD)
    off_scale = [p for p in hardware if p.fps > frame.fps_hi]
    on_axis = [p for p in hardware if p not in off_scale and p.fps >= frame.fps_lo]
    W = frame.left + frame.width + (HARDWARE_GUTTER if off_scale else 30)
    H = frame.top + frame.height + 110 + (NOTE_LINE if hardware else 0)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
           f'height="{H}" font-family="system-ui, -apple-system, Segoe UI, sans-serif" '
           f'font-size="12">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="{frame.left}" y="20" font-size="15" font-weight="600" fill="{INK}">'
           f'{heading}: {platform.core}</text>',
           f'<text x="{frame.left}" y="35" fill="{MUTED}">{subtitle}</text>']
    for tick in range(math.ceil(frame.bd_lo / 10) * 10, math.floor(frame.bd_hi / 10) * 10 + 1, 10):
        y = frame.y(tick)
        out.append(f'<line x1="{frame.left}" x2="{frame.left + frame.width}" y1="{y:.1f}" '
                   f'y2="{y:.1f}" stroke="{MUTED if tick == 0 else GRID}" '
                   f'stroke-width="{1.5 if tick == 0 else 1}"/>')
        out.append(f'<text x="{frame.left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   f'fill="{MUTED}">{tick:+d}%</text>' if tick else
                   f'<text x="{frame.left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   f'fill="{MUTED}">0%</text>')
    for tick in log_ticks(frame.fps_lo, frame.fps_hi):
        x = frame.x(tick)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{frame.top}" '
                   f'y2="{frame.top + frame.height}" stroke="{GRID}"/>')
        out.append(f'<text x="{x:.1f}" y="{frame.top + frame.height + 16}" '
                   f'text-anchor="middle" fill="{MUTED}">{tick:g}</text>')
    out.append(f'<text x="{frame.left + frame.width / 2}" y="{frame.top + frame.height + 36}" '
               f'text-anchor="middle" fill="{INK}">{platform.axis}</text>')
    out.append(f'<text transform="translate(18,{frame.top + frame.height / 2}) rotate(-90)" '
               f'text-anchor="middle" fill="{INK}">BD-rate vs x264 medium (J composite; '
               f'lower is better)</text>')
    path = " ".join(f"{frame.x(p.fps):.1f},{frame.y(p.bd_j):.1f}" for p in front)
    out.append(f'<polyline points="{path}" fill="none" stroke="{FRONT_COLOUR}" '
               f'stroke-width="1.5" stroke-dasharray="5 4"/>')
    items = [(p, frame.x(p.fps), frame.y(p.bd_j)) for p in timed + on_axis]
    for p, cx, _ in items:
        if p.iqr and not is_hardware(p):
            colour = OURS_COLOUR if p.ours else REFERENCE_COLOUR
            out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{frame.y(p.iqr[0]):.1f}" '
                       f'y2="{frame.y(p.iqr[1]):.1f}" stroke="{colour}" stroke-opacity="0.45" '
                       f'stroke-width="2"/>')
    # the spread bars, the anchor's 0% line and the Pareto front, which labels must not cross
    zero = frame.y(0)
    bars = [(cx - 1, frame.y(p.iqr[1]), cx + 1, frame.y(p.iqr[0]))
            for p, cx, _ in items if p.iqr and not is_hardware(p)] + [
        (frame.left, zero - 1, frame.left + frame.width, zero + 1)] + polyline_boxes(
        [(frame.x(p.fps), frame.y(p.bd_j)) for p in front])
    labels = place_labels(items, bars)
    for p, cx, cy in items:
        speed = (" (GPU aggregate throughput, frames in GPU memory)" if is_hardware(p) else
                 f", {p.speed:.2f}× x264 medium" if p.speed else "")
        out.append(f'<g><title>{p.label}: BD-J {p.bd_j:+.1f}%, BD-PSNR {fmt(p.bd_psnr)}, BD-VMAF v1 {fmt(p.bd_vmaf)}, '
                   f'{p.fps:.1f} fps{speed}</title>{marker(p, cx, cy)}</g>')
        tx, ty, anchor, leader = labels[p.family]
        if leader:  # from the marker's edge to the label's near side
            end_y = ty - 12 if ty > cy else ty + 3
            start_y = cy + (MARKER_RADIUS + 2) * (1 if end_y > cy else -1)
            out.append(f'<line x1="{cx:.1f}" y1="{start_y:.1f}" x2="{tx:.1f}" '
                       f'y2="{end_y:.1f}" stroke="{MUTED}" stroke-width="0.75"/>')
        weight = ' font-weight="600"' if p.ours else ""
        out.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" fill="{INK}"'
                   f'{weight}>{chart_label(p)}</text>')
    if off_scale:
        out += off_scale_markers(off_scale, frame)
    legend_y = frame.top + frame.height + 62
    keys = [(ChartPoint("nanocodec", "nanocodec (ours)", True, 0, 0, 0, None, 0, 0, (), 0, 0),
             "nanocodec (ours)"),
            (ChartPoint("x264", "", False, 0, 0, 0, None, 0, 0, (), 0, 0), "x264"),
            (ChartPoint("x265", "", False, 0, 0, 0, None, 0, 0, (), 0, 0), "x265"),
            (ChartPoint("svtav1", "", False, 0, 0, 0, None, 0, 0, (), 0, 0), "SVT-AV1 rtc"),
            (ChartPoint("openh264", "", False, 0, 0, 0, None, 0, 0, (), 0, 0), "openh264"),
            (ChartPoint("libvpx", "", False, 0, 0, 0, None, 0, 0, (), 0, 0), "libvpx rt"),
            (ChartPoint("libaom", "", False, 0, 0, 0, None, 0, 0, (), 0, 0), "libaom rt")]
    x = frame.left
    for key, text in keys:
        out.append(marker(key, x + 6, legend_y - 4))
        out.append(f'<text x="{x + 18}" y="{legend_y}" fill="{INK}">{text}</text>')
        x += 30 + 7 * len(text)
    out.append(f'<line x1="{x}" x2="{x + 26}" y1="{legend_y - 4}" y2="{legend_y - 4}" '
               f'stroke="{FRONT_COLOUR}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{x + 32}" y="{legend_y}" fill="{INK}">Pareto front</text>')
    x += 32 + 7 * len("Pareto front") + 12
    if on_axis:
        key = ChartPoint(HARDWARE[0][0], "", False, 0, 0, 0, None, 0, 0, (), 0, 0)
        out.append(marker(key, x + 6, legend_y - 4))
        text = f"NVENC ({HARDWARE_GPU} GPU)"
        out.append(f'<text x="{x + 18}" y="{legend_y}" fill="{INK}">{text}</text>')
        x += 30 + 7 * len(text)
    if off_scale:
        out.append(hardware_chevron(x + 6, legend_y - 4))
        out.append(f'<text x="{x + 18}" y="{legend_y}" fill="{INK}">NVENC ({HARDWARE_GPU} GPU), '
                   f'off scale</text>')
    out.append(f'<text x="{frame.left}" y="{legend_y + 22}" fill="{MUTED}">{note}</text>')
    if hardware:
        out.append(f'<text x="{frame.left}" y="{legend_y + 22 + NOTE_LINE}" fill="{MUTED}">'
                   f'{hardware_footnote}</text>')
    if stamp:  # small and unobtrusive, bottom right
        out.append(f'<text x="{W - 10}" y="{H - 8}" font-size="10" text-anchor="end" '
                   f'fill="{MUTED}">{stamp}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


#: The methodology's x86 speed paragraph (with --timing-x86).
X86_NOTE = """
- **x86 speed (`{stem}-x86.svg`, `.csv`):** the same streams (same quantizers and
  binaries' x86 builds) on a Ryzen 9 5950X, a shared machine, so each run is pinned
  to one logical CPU (one per physical core) and every point is paired with an x264
  medium run on the same CPU in interleaved rounds (machine load slows both); the
  minimum of {rounds} rounds per clip, after a warm-up."""

METHODOLOGY = """\
## Method

- **Anchor:** x264 {x264} `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build {commit}, its presets), openh264
  realtime (medium complexity, CAVLC), SVT-AV1 {svt} `--rtc --rc 2` (CBR, 1 s buffer),
  x264 / x265 {x265} as the anchor (zero latency, no B-frames, one thread / no pools),
  libvpx {libvpx} VP9 and libaom {libaom} AV1 in realtime CBR (`--rt --end-usage=cbr
  --lag-in-frames=0 --threads=1`, `--cpu-used` as named, 1 s buffer, frame dropping off).
  A point is counted as rate-invalid (it stays on its curve) when its 1 s leaky bucket,
  drained at the target, peaks above 1.5x the buffer, or its average rate exceeds the
  target by more than 10% plus the buffer's share of the window (+30% at 150 frames);
  ours are judged without the first key frame's outstanding loan (up to about 1.4
  buffers, repaid over the following seconds), the references with a strict 1 s VBV.
  The BD axis uses every bit. No encoder here drops frames; ours do not code input
  frames that would not change the picture (holds: no bits, scored as the repeated
  picture), the references code every frame.
- **Ladder and clips:** 150 / 300 / 600 / 1200 kbps at 720p30, scaled by pixel rate and
  capped at half the source bitrate; {clips} natural VAL clips in {classes} content
  classes, {frames} frames, one key frame. Metrics are scored on our encoder's
  reconstruction (the references' on their standard-decoded output); the build passed
  the landing conformance check (short streams of every shipped preset decoded by two
  standard decoders and ours, bit-exact with the reconstruction); the chart streams
  themselves are not each decoded.
- **Quality axis:** BD-rate on the achieved-rate curves (PCHIP, reliable-overlap rule).
  BD-J = (CB(Q_full) + CB(Q_low)) / 2, Q = mean BD-rate over PSNR-YUV, VMAF v1 and MS-SSIM,
  CB = mean over content classes of the class means, Q_low = the lower-rate half of
  the overlap (the J_real quality composite, eval/nceval/jreal.py). BD-PSNR-YUV and BD-VMAF v1 are the
  same with PSNR-YUV or VMAF v1 alone. The bars are the interquartile
  range of the per-clip BD-J. **Low rungs:** x264 medium undershoots its targets and has
  lower quality than most points at a given rate, so on some clips (static,
  bodycam_airport) their whole curves lay above its quality range, with no reliable
  overlap. On every clip where some point has no reliable overlap on the ladder above,
  EVERY encoder (ours and the references, the anchor too) also runs {low} kbps (720p30,
  scaled the same way), which reaches down into the anchor's range; BD is then direct
  over the overlap. **One clip set for all:** every point is averaged over the same clips,
  those on which every point overlaps the anchor directly; a clip where some point still
  has none is left out for every point (listed, with each point's direct BD there, under
  "Excluded clips"), unless that would remove a whole content class: then its clips are
  kept by chaining (‡) every unscored point through ONE intermediate encoder that
  overlaps both, BD(point→anchor) = (1 + BD(point→intermediate)) × (1 + BD(intermediate→
  anchor)) − 1, per metric and term, the second leg always direct (never two hops), the
  intermediate of the point's own codec lineage first, then of its own side, then the best
  overlap, then by name, and only when every unscored point chains.
  Nothing is extrapolated; a point with BD on fewer than {min_scored:.0%} of the clips is
  flagged.
- **Speed axis:** {platform}, encode CPU time per frame (rusage), median of three
  rotated rounds after a warm-up, the anchor timed in the same lock window as the point
  or, since the core's clock is locked and the anchor repeats within ~0.3 ms/frame
  across lock windows, one fresh anchor timing reused for several points; fixed
  quantizer chosen on the PC to land each encoder at about {rung} kbps
  (150 frames, 720p) on lowlight_city_night, crowd_mot17_09 and bodycam_bc1;
  fps = 1000 / geometric-mean ms per frame; speed = geometric mean of the per-clip
  anchor / point time ratios.{x86_note}
- **Caveats:** our VVC and AV2 are young (rate control and quality work are
  ongoing); the main speed axis is single-core on one ARM core (x86 ratios differ,
  mostly in SIMD coverage); fixed-QP timing switches x264/x265 adaptive quantisation off
  (cheaper than their CBR mode by a few percent); the quality axis is 11 short clips,
  so differences of a few percent are within the per-clip spread, and these VAL clips
  were also used to rank our candidate changes (see heldout-chart-*.md for 71 other
  clips); the reference encoders' real-time modes are tuned for latency, not for these
  metrics, x264 / x265 keep their default psychovisual settings (no `--tune psnr` /
  `ssim`), and their other modes (threads, lookahead, B-frames) may do better.
"""


def render(args: argparse.Namespace) -> None:
    families = chart_families(Path(args.binary).resolve(), Path(args.presets_dir),
                              binaries=binary_overrides(args.binary_for),
                              presets_for=binary_overrides(args.presets_for))
    clips = chart_clips(args.clipset)
    runner = Runner(load_paths(), EVALUATION_GATE, args.workers, args.over_budget,
                    stored_ms_ssim=True,  # untouched families keep their stored numbers
                    prebuilt=args.prebuilt)
    quality = quality_points(runner, families, clips, args.frames, args.rungs)
    points, pairs, excluded = quality.points, quality.pairs, quality.excluded
    hardware, hardware_timing, speeds = [], {}, {}
    if args.timing_hardware:  # the hardware encoders run only on the PC's own GPU
        hardware_timing = json.loads(Path(args.timing_hardware).read_text())
        speeds = hardware_speeds(hardware_timing)
        hardware = with_hardware_fps(hardware_points(
            runner, quality, families, chart_families(
                Path(args.binary).resolve(), Path(args.presets_dir),
                [name for name, _ in HARDWARE], binary_overrides(args.binary_for),
                binary_overrides(args.presets_for))[1:], clips, args.frames, args.rungs),
            speeds)
    untimed = points
    if args.timing:
        points = with_speed(points, json.loads(Path(args.timing).read_text()))
    x86 = (with_speed(untimed, json.loads(Path(args.timing_x86).read_text()))
           if args.timing_x86 else None)
    front = pareto(points)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (OUTPUT_STEM + (f"-{Path(args.clipset).stem}" if args.clipset else ""))
    stem.with_suffix(".csv").write_text(csv_text(points, front))
    if hardware:
        stem.with_name(stem.name + "-hardware.csv").write_text(hardware_csv(hardware, speeds))
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    stamp = updated_line(meta, args.date or datetime.date.today().isoformat())
    page = [f"# {args.title}\n", f"<sub>{stamp}</sub>\n", f"![bitrate vs speed]({stem.name}.svg)\n",
            *([f"The same on x86 (one Ryzen 9 5950X core):\n\n"
               f"![bitrate vs x86 speed]({stem.name}{X86.suffix}.svg)\n"] if x86 else []),
            markdown_table(points, front, x86), coverage_note(quality),
            *([hardware_table(hardware, speeds, hardware_timing.get("platform", ""))]
              if hardware else []),
            *([chained_table(points)] if any(p.chained for p in points) else []),
            *([excluded_table(excluded, points)] if excluded else []),
            "## Selected pairs\n\nDirect BD-rate of our encoders against the reference "
            "encoders and against each other, on the same points (no anchor in between; "
            "ours vs ours drops the rungs either arm overshot, as on the board).\n", pairs_table(pairs)]
    if meta:
        timing = json.loads(Path(args.timing).read_text()) if args.timing else {}
        page.append(METHODOLOGY.format(
            clips=len(clips), classes=len({c.content_class for c in clips}),
            frames=args.frames, min_scored=MIN_SCORED_FRACTION,
            low=" / ".join(map(str, LOW_LADDER_KBPS)),
            platform=timing.get("platform", "n/a"), rung=timing.get("rung_kbps", "n/a"),
            x86_note=(X86_NOTE.format(stem=stem.name, rounds=X86_ROUNDS)
                      + meta.get("x86_timing_note", "")) if x86 else "",
            **meta))
    stem.with_suffix(".md").write_text("\n".join(page))
    if any(p.fps for p in points):
        stem.with_suffix(".svg").write_text(svg_chart(points, front, args.subtitle, stamp=stamp))
    if x86:
        x86_stem = stem.with_name(stem.name + X86.suffix)
        x86_front = pareto(x86)
        x86_stem.with_suffix(".csv").write_text(csv_text(x86, x86_front, X86))
        x86_stem.with_suffix(".svg").write_text(svg_chart(x86, x86_front, args.subtitle, X86,
                                                          stamp, hardware=hardware,
                                                          hardware_footnote=hardware_note(speeds)))
        print(f"wrote {x86_stem}.{{csv,svg}}")
    print(markdown_table(points, front, x86))
    print(pairs_table(pairs))
    print(f"wrote {stem}.{{csv,md{',svg' if any(p.fps for p in points) else ''}}}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python3 -m nceval.chart",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("measure", "render"):
        p = sub.add_parser(name)
        p.add_argument("--binary", required=True, help="our nanocodec build")
        p.add_argument("--presets-dir", default=str(PRESETS_DIR),
                       help="that build's own presets/ (a frozen build)")
        p.add_argument("--clipset", help="another clip set (e.g. adv12) instead of VAL natural")
        p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
        p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        p.add_argument("--over-budget", action="store_true")
        p.add_argument("--binary-for", action="append", default=[], metavar="FAMILY=PATH",
                       help="another build for one of our families (e.g. a codec merged "
                            "after --binary's commit)")
        p.add_argument("--presets-for", action="append", default=[], metavar="FAMILY=DIR",
                       help="that build's own presets directory")
        p.add_argument("--prebuilt", type=Path, help="score streams encoded elsewhere, from "
                       "this directory (cbr.Bench.prebuilt; tools/dev/aws_heldout_chart.py)")
        p.add_argument("--rungs", type=lambda v: tuple(int(i) for i in v.split(",")),
                       help="comma-separated real-use ladder indices (default: all)")
    m = sub.choices["measure"]
    m.add_argument("--points", help="comma-separated families (default: all); the anchor "
                                    "is always included")
    m.add_argument("--part", help="I/N: only part I of N class-balanced clip parts")
    m.add_argument("--rescore", action="store_true",
                   help="re-measure stored points of an older MS-SSIM protocol too "
                        "(default: reuse them; only points not stored are measured)")
    r = sub.choices["render"]
    r.add_argument("--timing", help="timing JSON (tools/dev/anchor_chart_timing.py)")
    r.add_argument("--timing-x86", help="the PC's timing JSON (anchor_chart_timing.py "
                   "time-host): adds the x86 chart")
    r.add_argument("--timing-hardware", help="the hardware encoders' timing JSON "
                   "(tools/dev/nvenc_timing.py): adds them (measured on this PC's GPU), "
                   "with their fps, in a table of their own and past the chart's edge")
    r.add_argument("--out", default="../docs/results")
    r.add_argument("--subtitle", default="")
    r.add_argument("--title", default="Bitrate vs encode speed")
    r.add_argument("--date", help="the date in the outputs' Updated line (default: today)")
    r.add_argument("--meta", help="JSON with the methodology note's build facts "
                                  "(commit, x264, x265, svt versions)")
    m.add_argument("--expect-commit",
                   help="fail at once unless --binary was built from this commit "
                        f"(also ${buildinfo.EXPECT_ENV})")
    args = parser.parse_args(argv)
    if args.command == "measure":
        buildinfo.check(Path(args.binary), "binary", args.expect_commit)
        for pair in args.binary_for:
            family, _, path = pair.partition("=")
            buildinfo.check(Path(path), f"binary for {family}", measured=False)
    measure(args) if args.command == "measure" else render(args)


if __name__ == "__main__":
    main(sys.argv[1:])
