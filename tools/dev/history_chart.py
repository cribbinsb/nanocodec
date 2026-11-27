#!/usr/bin/env python3
"""The headline chart's history: one frame per day, appended to the page's stored frames.

The results site's headline chart (view `fresh-psnr`: BD-J against x264 medium --tune
psnr on the held-out fresh clips, x86 fps on the c8a) changed its clip set, zero line,
timing machine and preset names several times, so the dated chart files are not
comparable. The history page (web/results/history.html) shows one frame per day, each a
build measured with the chart's own engine (nceval.chart: the real-use CBR ladder, low
rungs, class-balanced BD-J, our reconstruction scored) on a fixed slice of the fresh clips
and timed on the chart's x86 timing machine (AWS c8a.large, tools/dev/aws_timing.py).

The page is append-only (owner, 2026-11-21): its stored frames (web/results/data/
history.json) are never re-measured. A new frame is computed alone, on the basis the
stored frames recorded (`BASIS`: the slice's clips, the clips with low rungs, the clips
left out), and appended; the stored frames are read through the rung renames made since
they were written (nanocodec.RUNG_RENAMES), so each series stays one ladder position:

  points  the new frame's points on rented machines (tools/dev/aws_heldout_chart.py run and
          score-cloud on the slice, every family on its build in RECORD, the low rungs on
          the basis's clips). Points already in the shared store (the release's held-out
          fresh chart measures the slice's ladder) are neither encoded nor scored again.
  time    the new frame's x86 fps: the plain release build (--binary, the kind every frame
          was timed as) on the chart's timing plan, as chart_refresh's time-x86 does
          (TIMING_SESSIONS c8a.large sessions, one repeat). With --baseline PLAIN (the
          previous frame's commit built the same way), each point is timed paired with
          that build on the same core (aws_timing.py --baseline-binary), and the frame's
          fps is the previous frame's times this build's speed over the baseline's: the
          day-to-day change is the code's alone, not the sessions' or the builds' (owner,
          2026-11-22, after a 6-11 % frame-to-frame drop was two thirds code).
  report  the stored frames plus the new one into web/results/data/history.json and
          docs/results/history-<last day>/ (history.json, README.md); `--out DIR` writes a
          scratch copy instead; also regenerates the screens' front slopes (nceval.front_k,
          eval/configs/front_k.json). Fails when a point is missing, the basis would change, or
          the references (recomputed from the store) differ from the stored ones.
  frames  PNG per day, GIF and MP4 from the page.
  add-day points, time, report and frames in order (each skips what is done).

    python3 tools/dev/history_chart.py add-day --day 2026-11-21 --commit R --record RECORD \\
        --binary PLAIN [--label 0.6.0] [--out DIR]

The frames up to 2026-11-20 were measured by this tool's earlier form (each day's build
re-measured locally, timed in one c8a.large per preset holding every day's build: git
history before the append-only change).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aws_cpu_ab import SHARE_SUFFIX, shard_runs  # noqa: E402
from aws_timing import BASELINE_SUFFIX  # noqa: E402
from nceval import chart, front_k, key_shares  # noqa: E402
from nceval.cbr import Family  # noqa: E402
from nceval.clips import Clip  # noqa: E402
from nceval.config import load_paths  # noqa: E402
from nceval.gatepolicy import EVALUATION_GATE  # noqa: E402
from nceval.realuse import DEFAULT_FRAMES, Runner  # noqa: E402
from nceval.encoders.nanocodec import RUNG_RENAMES, renamed_stem  # noqa: E402

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
WORK = DATA / "work" / "history-anim"
CODECS = ("h264", "hevc", "vvc", "av1", "vp9", "av2")
CODEC_LABELS = {"h264": "H.264", "hevc": "HEVC", "vvc": "VVC", "av1": "AV1", "vp9": "VP9",
                "av2": "AV2"}
ZERO = "x264-medium~psnr"
FRESH = "independent_fresh_2026-11-13.json"
#: The headline view, its data (labels, owners, the references' c8a fps) and the page's copy.
VIEW = "fresh-psnr"
SITE_CHART = REPO / "web" / "results" / "data" / "chart.json"
PAGE_DATA = REPO / "web" / "results" / "data" / "history.json"
#: Each frame's main landings, by date, written by hand from git log.
CAPTIONS = REPO / "docs" / "results" / "history-2026-11-20" / "captions.json"
RESULTS = REPO / "docs" / "results"
#: The page and what its version stamp covers (`stamp_page`).
PAGE = REPO / "web" / "results" / "history.html"
PAGE_ASSETS = (REPO / "web" / "results" / "history.js", REPO / "web" / "results" / "site.css",
               PAGE_DATA)
#: The record's fields that fix what every frame is computed on.
BASIS = ("slice", "excluded", "low_rung_clips")
#: How many RUNG_RENAMES a stored record's names already carry (absent: none, the record
#: of 2026-11-20). Stored frames are read through the renames made since.
RENAMES_KEY = "rung_renames"
#: A rung renamed away (H.264's old Faster, 2026-11-21): its series keeps its old stem
#: behind this prefix and ends with the last frame that had it.
RETIRED = "retired-"
#: The whole-front change is measured against this frame's frontier: the first with every
#: codec and more than one preset (2026-11-07 is H.264 alone, one point).
FRONT_BASE_DAY = "2026-11-09"
#: Evenly spaced points over an overlap range at which two fronts are compared.
FRONT_SAMPLES = 200
#: The slice as a clip set (the held-out cloud path's input; written by `points`).
SLICE_CLIPSET = WORK / "slice-clipset.json"
#: A new frame's x86 timing (tools/dev/aws_timing.py, as chart_refresh's time-x86): the
#: points split over this many c8a.large sessions side by side, one repeat (the 2026-11-15
#: probe's repeats agreed within 0.4 %), against x264 medium in the same session.
TIMING_DIR = WORK / "timing"
TIMING_SESSIONS = 4
TIMING_PLAN = RESULTS / "bd-vs-speed-plan.json"
TIMING_X86 = RESULTS / "bd-vs-speed-timing-x86.json"
ANCHOR_TIMED = "x264-medium"
#: The repository's first commit (421c8142d): the first frame's activity counts from it.
FIRST_COMMIT_DAY = "2026-11-06"
#: Agent families as the commit trailers name them (first match wins); a commit without
#: one of these trailers is counted as `unmarked`.
AGENT_TRAILERS = (("Claude", r"^(Co-Authored-By: Claude|Claude-Session:)"),
                  ("Codex", r"^(Co-Authored-By: (OpenAI )?Codex|Assisted-by: OpenAI Codex)"),
                  ("Gemini", r"^(Co-Authored-By|Assisted-by):.*Gemini"))
COMMIT_MARK = "\x1ecommit\x1e"
#: The animation: frame width (px), seconds per day, and the last day's extra hold.
FRAME_WIDTH = 1000
FRAME_SECONDS = 1
LAST_FRAME_HOLD = 3


# --- the stored frames -----------------------------------------------------------------

def renamed_key(key: str, renames) -> str:
    """A stored preset key (`h264@fast`) by today's rung names; a retired rung keeps its
    old stem behind RETIRED (`h264@retired-faster`)."""
    codec, at, stem = key.partition("@")
    if not at or stem.startswith(RETIRED):
        return key
    today = renamed_stem(codec, stem, renames)
    return f"{codec}@{today}" if today else f"{codec}@{RETIRED}{stem}"


def stored_record(path: Path) -> dict:
    """The page's stored record, its preset keys by today's rung names (`renamed_key`)."""
    record = json.loads(path.read_text())
    renames = RUNG_RENAMES[record.get(RENAMES_KEY, 0):]
    rename = functools.partial(renamed_key, renames=renames)
    for day in record["days"]:
        day["values"] = {rename(k): v for k, v in day["values"].items()}
        day["frontier"] = [rename(k) for k in day["frontier"]]
        day["codec_fronts"] = {c: [rename(k) for k in keys]
                               for c, keys in day.get("codec_fronts", {}).items()}
    unchanged = record["timing_noise"]["unchanged"]
    unchanged["fps_change_pct"] = {rename(k): v for k, v in unchanged["fps_change_pct"].items()}
    record[RENAMES_KEY] = len(RUNG_RENAMES)
    return record


def basis_clips(record: dict) -> tuple[list[Clip], list[Clip]]:
    """The slice's clips and those with low rungs, as the record fixes them."""
    by_name = {c.name: c for c in chart.chart_clips(str(load_paths().clips / FRESH))}
    clips = [by_name[c["name"]] for c in record["slice"]["clips"]]
    low = set(record["low_rung_clips"])
    return clips, [c for c in clips if c.name in low]


# --- the new frame's points ------------------------------------------------------------

def record_families(record_file: Path) -> list[Family]:
    """Our chart families on their builds in a chart's build record (as
    aws_heldout_chart.py --record measures them, so their points are those)."""
    from aws_heldout_chart import record_builds

    build, binaries, presets = record_builds(str(record_file))
    names = [n for n, _ in chart.OURS if n in binaries]
    return chart.chart_families(build / "nanocodec", build / "presets", names, binaries,
                                presets)[1:]


def reference_families() -> list[Family]:
    """The zero line first, then the chart's reference encoders."""
    families = chart.chart_families(Path(sys.executable), chart.PRESETS_DIR,
                                    [ZERO, *(name for name, _ in chart.REFERENCES)])
    return sorted(families, key=lambda f: f.name != ZERO)


def write_slice_clipset(record: dict) -> Path:
    """The slice's clips as a clip set file (the fresh set's own entries)."""
    fresh = json.loads((load_paths().clips / FRESH).read_text())["clips"]
    names = [c["name"] for c in record["slice"]["clips"]]
    by_name = {c["name"]: c for c in fresh}
    SLICE_CLIPSET.parent.mkdir(parents=True, exist_ok=True)
    SLICE_CLIPSET.write_text(json.dumps({"name": SLICE_CLIPSET.stem,
                                         "clips": [by_name[n] for n in names]}, indent=1) + "\n")
    return SLICE_CLIPSET


def cloud_args(record: dict, args: argparse.Namespace, families: str) -> list[str]:
    """aws_heldout_chart.py's options for the frame's points: the slice, the build
    record, the basis's low-rung clips, `families`."""
    return ["--clipset", str(write_slice_clipset(record)), "--record", str(args.record),
            "--low-clips", ",".join(record["low_rung_clips"]), "--families", families]


def points_cmd(args: argparse.Namespace) -> None:
    """The frame's points on rented machines: encoded, then scored (each a no-op when
    every point is in the store already)."""
    record = stored_record(PAGE_DATA)
    families = ",".join(f.name for f in [*reference_families(), *record_families(args.record)])
    tool = [sys.executable, str(REPO / "tools/dev/aws_heldout_chart.py")]
    for command in ("run", "score-cloud"):
        extra = ["--dry-run"] if args.dry_run else []
        subprocess.run([*tool, command, *cloud_args(record, args, families),
                        "--purpose", f"history-{command}", "--budget", str(args.budget), *extra],
                       check=True)


# --- the new frame's speed -------------------------------------------------------------

def timing_out(commit: str, index: int) -> Path:
    return TIMING_DIR / f"x86-{commit}-{index + 1}"


def time_cmd(args: argparse.Namespace) -> None:
    """Our families of the record timed on the plain build `--binary`, TIMING_SESSIONS
    c8a.large sessions side by side (a session whose summary is here is not rerun)."""
    names = [f.name for f in record_families(args.record)]
    parts = [names[i::TIMING_SESSIONS] for i in range(TIMING_SESSIONS)]
    sessions = []
    for index, part in enumerate(parts):
        out = timing_out(args.commit, index)
        if not part or (out / "summary.json").is_file():
            continue
        baseline = ["--baseline-binary", str(args.baseline)] if args.baseline else []
        command = [sys.executable, str(REPO / "tools/dev/aws_timing.py"), "--points",
                   ",".join(part), "--binary", str(args.binary), "--plan", str(args.plan),
                   "--repeats", "1", "--purpose", "history-timing", "--budget", "1",
                   "--out", str(out), *baseline] + (["--dry-run"] if args.dry_run else [])
        TIMING_DIR.mkdir(parents=True, exist_ok=True)
        log = (TIMING_DIR / f"x86-{args.commit}-{index + 1}.log").open("w")
        sessions.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    failed = [p.args for p, _ in sessions if p.wait()]
    if failed:
        raise SystemExit(f"{len(failed)} timing sessions failed: see {TIMING_DIR}/x86-{args.commit}-*.log")


def timed_fps(commit: str, family: str, anchor_ms: dict[str, float],
              previous_fps: float | None = None) -> tuple[float | None, bool]:
    """The chart's x86 fps of `family` at `commit` (`time`): the median over its repeats of
    the time ratio to x264 medium in the same session, times x264 medium's stored time per
    clip (`anchor_ms`, as every stored frame's), as the chart. Timed paired with the
    previous frame's build (`time --baseline`), it is `previous_fps` over the median of
    this build's time over the baseline's. The flag says the fps is chained so (its
    absolute level is the previous frame's)."""
    for index in range(TIMING_SESSIONS):
        summary = timing_out(commit, index) / "summary.json"
        rows = json.loads(summary.read_text()) if summary.is_file() else {}
        paired = rows.get(f"{family}{BASELINE_SUFFIX}") or {}
        over_base = [v for k, v in paired.items() if re.fullmatch(r"repeat-\d+-over-base", k)]
        if over_base and previous_fps:
            return previous_fps / statistics.median(over_base), True
        entry = rows.get(family)
        ratios = [v["ratio_x264_medium"] for k, v in (entry or {}).items()
                  if re.fullmatch(r"repeat-\d+", k)]
        if ratios:
            return 1000.0 / (statistics.median(ratios) * chart.geomean(list(anchor_ms.values()))), \
                False
    return None, False


# --- the 1-in-300 key-share view ---------------------------------------------------------
#
# The owner's key-share rule (2026-11-22, eval/nceval/key_shares.py; docs/PROCEDURES.md
# "Key shares"): the frames above are 150 f with one key, which never shows a periodic
# key. From 2026-11-22 each frame also carries the 1-in-300 share: 900 f from frame 0, a
# key every 300 f, our families of the frame's build and of the baseline build (0.6.0)
# both scored and both timed, in the same sessions. The fresh slice has no clip that long
# (164-168 f), so this view's clips are the long clips of SHARE_SOURCE, SHARE_PER_CLASS a
# class: tuning-pool clips, not held out. Its speed is CPU time on SHARE_TIMING_TARGET
# (aws_cpu_ab.py: base and candidate on the same core in interleaved rounds, CBR 600 on
# its three 900-frame clips), so `fps_x86` here is not the chart's timing.

SHARE = key_shares.KeyShare(300)
SHARE_KEY = "key_share_300"
SHARE_SOURCE = "heldout_chart_2026-11-12.json"
SHARE_PER_CLASS = 2
SHARE_CLIPSET = WORK / "share300-clipset.json"
SHARE_TIMING_DIR = WORK / "share300-timing"
SHARE_TIMING_TARGET = "zen5"
SHARE_TIMING_BOXES = 8
SHARE_TIMING_ROUNDS = 3
#: The pair name aws_cpu_ab gives its one --base/--cand pair (`<pair>@k<period>`).
SHARE_PAIR = "ab"


def share_clip_entries(record: dict) -> list[dict]:
    """The view's clips as SHARE_SOURCE entries: the stored view's, else the long-enough
    clips, SHARE_PER_CLASS of each class by name."""
    source = json.loads((load_paths().clips / SHARE_SOURCE).read_text())["clips"]
    stored = record.get(SHARE_KEY, {}).get("clips")
    if stored:
        by_name = {c["name"]: c for c in source}
        return [by_name[n] for n in stored]
    chosen, per_class = [], {}
    for clip in sorted((c for c in source if c["frames"] >= SHARE.window),
                       key=lambda c: (c["content_class"], c["name"])):
        if per_class.get(clip["content_class"], 0) < SHARE_PER_CLASS:
            per_class[clip["content_class"]] = per_class.get(clip["content_class"], 0) + 1
            chosen.append(clip)
    return chosen


def write_share_clipset(record: dict) -> Path:
    SHARE_CLIPSET.parent.mkdir(parents=True, exist_ok=True)
    SHARE_CLIPSET.write_text(json.dumps({"name": SHARE_CLIPSET.stem,
                                         "clips": share_clip_entries(record)}, indent=1) + "\n")
    return SHARE_CLIPSET


def share_window_args(clips: list[dict]) -> list[str]:
    """aws_heldout_chart.py's window: the share's frames and key period, low rungs on
    every clip (one basis for both builds, fixed in advance)."""
    return ["--frames", str(SHARE.window), "--key-frames", str(SHARE.period),
            "--low-clips", ",".join(c["name"] for c in clips)]


def share_points_cmd(args: argparse.Namespace) -> None:
    """The view's points of the frame's build and of the baseline build (`--record`,
    `--baseline-record`) on rented machines, encoded, then scored."""
    record = stored_record(PAGE_DATA)
    clipset = write_share_clipset(record)
    clips = share_clip_entries(record)
    tool = [sys.executable, str(REPO / "tools/dev/aws_heldout_chart.py")]
    for record_file in (args.record, args.baseline_record):
        families = ",".join(f.name for f in [*reference_families(),
                                             *record_families(record_file)])
        for command in ("run", "score-cloud"):
            extra = ["--dry-run"] if args.dry_run else []
            subprocess.run([*tool, command, "--clipset", str(clipset), "--record",
                            str(record_file), "--families", families, *share_window_args(clips),
                            "--purpose", f"history-share-{command}", "--budget",
                            str(args.budget), *extra], check=True)


def family_rung(name: str) -> str:
    """`nanocodec-h264@fast` -> `h264:fast` (the Default, unsuffixed, is `medium`)."""
    codec, _, preset = name.removeprefix("nanocodec-").partition("@")
    return f"{codec}:{preset or 'medium'}"


def share_rungs(*record_files: Path) -> list[str]:
    """The rungs every record has (a rung new tonight has no baseline to pair with)."""
    sets = [{family_rung(f.name) for f in record_families(r)} for r in record_files]
    return sorted(set.intersection(*sets))


def share_timing_out(commit: str) -> Path:
    return SHARE_TIMING_DIR / f"{SHARE_TIMING_TARGET}-{commit}"


def share_time_cmd(args: argparse.Namespace) -> None:
    """The frame's build (`--binary`) paired with the baseline build (`--baseline`) on the
    same core, at every key share (aws_cpu_ab.py --key-shares; the view reads 1/300)."""
    out = share_timing_out(args.commit)
    if (out / "summary.json").is_file():
        return
    rungs = share_rungs(args.record, args.baseline_record)
    command = [sys.executable, str(REPO / "tools/dev/aws_cpu_ab.py"), "--base",
               str(args.baseline), "--cand", str(args.binary), "--rungs", ",".join(rungs),
               "--key-shares", "--targets", SHARE_TIMING_TARGET,
               "--boxes", str(SHARE_TIMING_BOXES), "--rounds", str(SHARE_TIMING_ROUNDS),
               "--purpose", "history-share-timing", "--out", str(out)]
    subprocess.run(command + (["--dry-run"] if args.dry_run else []), check=True)


def share_fps(run_dir: Path, period: int) -> dict[str, dict[str, float]]:
    """Per rung, each side's CPU-time fps at the share: the geometric mean over the clips
    of the frames over the median CPU time of the rounds (aws_cpu_ab's runs)."""
    times: dict[tuple[str, str], dict[str, list[float]]] = {}
    pair = f"{SHARE_PAIR}{SHARE_SUFFIX}{period}"
    # a shard's runs may span a reclaimed box and its retry (aws_cpu_ab.keep_finished)
    for runs in (f for files in shard_runs(run_dir).values() for f in files):
        for line in runs.read_text().splitlines():
            run = json.loads(line)
            name, rung, clip = run["job"].split("|")
            if name == pair and run["user_s"]:
                times.setdefault((rung, run["variant"]), {}).setdefault(clip, []).append(
                    run["user_s"])
    window = key_shares.KeyShare(period).window
    fps: dict[str, dict[str, float]] = {}
    for (rung, side), clips in times.items():
        per_clip = [window / statistics.median(v) for v in clips.values()]
        fps.setdefault(rung, {})[side] = chart.geomean(per_clip)
    return fps


def share_quality(clips: list[Clip], families: list[Family]) -> dict[str, dict]:
    """`families`' BD-J against the zero line on the view's clips at the share, from the
    store (the `share-points` step measures them; nothing is measured here)."""
    run = runner()
    every = [*reference_families(), *families]
    jobs = (run.points(every, clips, SHARE.window, None, key_frames=SHARE.period)
            + run.points(every, clips, SHARE.window, None, ladder=chart.low_ladder,
                         key_frames=SHARE.period))
    missing = [p for bench, p in jobs if not bench.cached(p)]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(jobs)} share points are not in the store "
                         f"(first: {missing[0].family.name} {missing[0].clip.name}): run "
                         f"share-points first")
    curves = run.measure("share", every, clips, SHARE.window, budgeted=False,
                         key_frames=SHARE.period)
    low = run.measure("share low rungs", every, clips, SHARE.window, budgeted=False,
                      ladder=chart.low_ladder, key_frames=SHARE.period)
    for key, points in low.items():
        curves[key] = curves[key] + points
    points, excluded = chart.anchor_points(curves, every, clips, run,
                                           {f.name: f.name for f in every})
    by_name = {p.family: p for p in points}
    return {"values": {f.name: bd_values(by_name[f.name]) for f in every},
            "excluded": sorted(e.clip for e in excluded)}


def share_day(record: dict, args: argparse.Namespace) -> dict:
    """The view's frame: our families of the build and of the baseline at 1/300, each with
    its BD-J and CPU-time fps, and the build's front against the baseline's."""
    clips = chart.chart_clips(str(write_share_clipset(record)))
    timing = share_timing_out(args.commit)
    fps = {p: share_fps(timing, p) for p in key_shares.REQUIRED_PERIODS}
    sides = {}
    for side, record_file, timed in (("build", args.record, "cand"),
                                     ("baseline", args.baseline_record, "base")):
        quality = share_quality(clips, record_families(record_file))
        values = {}
        for family in record_families(record_file):
            rung = family_rung(family.name)
            values[family.name.removeprefix("nanocodec-")] = {
                **quality["values"][family.name],
                "fps_x86": rounded((fps[SHARE.period].get(rung) or {}).get(timed)),
                # the same pairing at the other share, for the record
                "fps_1in150": rounded((fps[150].get(rung) or {}).get(timed))}
        sides[side] = {"values": values, "excluded": quality["excluded"],
                       "references": {f.name: quality["values"][f.name]
                                      for f in reference_families()}}
    build = {"date": args.day, "values": sides["build"]["values"]}
    baseline = {"date": args.baseline_label, "values": sides["baseline"]["values"]}
    for day in (build, baseline):
        day["frontier"] = pareto_front(day["values"])
    return {"date": args.day, "commit": args.commit, "baseline": args.baseline_label,
            "values": build["values"], "baseline_values": baseline["values"],
            "excluded": sides["build"]["excluded"],
            "references": sides["build"]["references"],
            "frontier": build["frontier"], "baseline_frontier": baseline["frontier"],
            "front_vs_baseline": front_change(build, baseline)}


def share_report_cmd(args: argparse.Namespace) -> None:
    """The view's frame into the page's data (history.json `key_share_300`)."""
    record = stored_record(PAGE_DATA)
    view = record.setdefault(SHARE_KEY, {
        "share": {"period": SHARE.period, "window": SHARE.window},
        "clips": [c["name"] for c in share_clip_entries(record)],
        "clip_rule": f"{SHARE_SOURCE}: clips with {SHARE.window} frames or more, "
                     f"{SHARE_PER_CLASS} per class by name (tuning-pool clips, not held out)",
        "timing": f"aws_cpu_ab.py on {SHARE_TIMING_TARGET}, CBR, {SHARE_TIMING_ROUNDS} "
                  "rounds, base and build on one core; CPU-time fps",
        "days": []})
    day = share_day(record, args)
    view["days"] = sorted([d for d in view["days"] if d["date"] != args.day] + [day],
                          key=lambda d: d["date"])
    change = day["front_vs_baseline"]
    print(f"1/{SHARE.period} view {args.day}: front against {args.baseline_label} "
          f"J {change['j_points']} points, speed x{change['speed_factor']}", flush=True)
    text = json.dumps(record, indent=1) + "\n"
    outs = [Path(args.out) / "history.json"] if args.out else [
        RESULTS / f"history-{record['days'][-1]['date']}" / "history.json", PAGE_DATA]
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"{SHARE_KEY} -> {out}")


# --- activity --------------------------------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True).stdout


def path_area(path: str) -> str:
    """ACTIVITY_AREAS: Markdown anywhere is docs; then by top-level directory."""
    if path.endswith(".md"):
        return "docs"
    top = path.split("/", 1)[0]
    return {"crates": "rust", "eval": "eval_tools", "tools": "eval_tools",
            "docs": "docs"}.get(top, "other")


def commit_agent(trailers: str) -> str:
    """The agent family a commit's trailers name (AGENT_TRAILERS), else `unmarked`."""
    for agent, pattern in AGENT_TRAILERS:
        if re.search(pattern, trailers, re.I):
            return agent
    return "unmarked"


def rust_lines(commit: str) -> int:
    """Lines of the .rs files under crates/ at `commit`."""
    blobs = [line.split()[2] for line in git("ls-tree", "-r", commit, "--", "crates").splitlines()
             if line.endswith(".rs")]
    batch = subprocess.run(["git", "-C", str(REPO), "cat-file", "--batch"], input="\n".join(blobs).encode(),
                           capture_output=True, check=True).stdout
    # each object comes as a header line, its content, then a newline of its own
    return batch.count(b"\n") - 2 * len(blobs)


def activity(day: str, commit: str, since: str) -> dict:
    """The integration branch's first-parent commits from `since` to the end of `day` (UTC)
    up to `commit`: their count, lines added / removed per area, commits per agent by
    trailer; Rust lines at `commit`."""
    log = git("log", f"{commit}", "--first-parent", "--diff-merges=first-parent", "--numstat",
              f"--since={since} 00:00:00 +0000", f"--until={day} 23:59:59 +0000",
              f"--format={COMMIT_MARK}%(trailers:only,unfold)")
    lines = {area: [0, 0] for area in ("rust", "eval_tools", "docs", "other")}
    agents: dict[str, int] = {}
    commits = 0
    for chunk in log.split(COMMIT_MARK)[1:]:
        commits += 1
        trailers, *rows = chunk.split("\n")
        stats = [r.split("\t") for r in rows if r.count("\t") == 2]
        trailer_text = "\n".join([trailers, *(r for r in rows if r.count("\t") != 2)])
        agent = commit_agent(trailer_text)
        agents[agent] = agents.get(agent, 0) + 1
        for added, removed, path in stats:
            if added != "-":  # binary files
                lines[path_area(path)][0] += int(added)
                lines[path_area(path)][1] += int(removed)
    return {"since": since, "commits": commits, "lines": lines, "agents": agents,
            "rust_lines_total": rust_lines(commit)}




# --- fronts ----------------------------------------------------------------------------

def pareto_front(values: dict) -> list[str]:
    """Our points no other of the day beats on both timed speed and BD-J, slowest
    first (as web/results/frontier.js)."""
    points = {k: (v["fps_x86"], v["j"]) for k, v in values.items()
              if v["j"] is not None and v["fps_x86"]}
    front = [k for k, (x, y) in points.items()
             if not any((x2 >= x and y2 <= y and (x2 > x or y2 < y))
                        for k2, (x2, y2) in points.items() if k2 != k)]
    return sorted(front, key=lambda k: points[k][0])


def front_curve(day: dict, front: list[str]) -> list[tuple[float, float]]:
    """A frontier of the day (preset keys) as (log10 fps, BD-J) vertices, slowest first: J
    rises with speed."""
    return [(math.log10(day["values"][k]["fps_x86"]), day["values"][k]["j"]) for k in front]


def interpolate(curve: list[tuple[float, float]], x: float) -> float:
    """The piecewise-linear curve at x (inside its range)."""
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def mean_gap(curve: list[tuple[float, float]], base: list[tuple[float, float]]) -> float | None:
    """The mean of curve - base over the overlap of their x ranges (FRONT_SAMPLES evenly
    spaced points; no extrapolation beyond either), or None without an overlap."""
    if len(curve) < 2 or len(base) < 2:
        return None
    lo, hi = max(curve[0][0], base[0][0]), min(curve[-1][0], base[-1][0])
    if hi <= lo:
        return None
    xs = [lo + (hi - lo) * i / (FRONT_SAMPLES - 1) for i in range(FRONT_SAMPLES)]
    return sum(interpolate(curve, x) - interpolate(base, x) for x in xs) / FRONT_SAMPLES


def front_change(day: dict, base: dict, codec: str | None = None) -> dict:
    """`day`'s frontier against `base`'s (over every codec, or `codec`'s presets alone): the
    mean BD-J difference at equal speed (points, over the overlapping log-fps range) and the
    mean speed factor at equal BD-J (over the overlapping BD-J range; the front read as log
    fps against J). None for both where either front has fewer than two presets."""
    pick = (lambda d: d["frontier"]) if codec is None else (lambda d: d["codec_fronts"][codec])
    curve, reference = front_curve(day, pick(day)), front_curve(base, pick(base))
    j_gap = mean_gap(curve, reference)
    speed_gap = mean_gap([(y, x) for x, y in curve], [(y, x) for x, y in reference])
    return {"base": base["date"], "j_points": rounded(j_gap),
            "speed_factor": rounded(10 ** speed_gap, 3) if speed_gap is not None else None}


def codec_front_changes(days: list[dict]) -> dict[str, str]:
    """Each codec's own frontier per day (`codec_fronts`, its presets alone) and its change
    against the codec's first day with two presets or more and against the previous frame
    (`codec_vs_base`, `codec_vs_previous`). Returns each codec's base day."""
    bases = {}
    for day in days:
        day["codec_fronts"] = {c: pareto_front({k: v for k, v in day["values"].items()
                                                if k.partition("@")[0] == c})
                               for c in CODECS}
    for codec in CODECS:
        base = next((d for d in days if len(d["codec_fronts"][codec]) >= 2), None)
        if base is None:
            continue
        bases[codec] = base["date"]
        for index, day in enumerate(days):
            day.setdefault("codec_vs_base", {})[codec] = front_change(day, base, codec)
            day.setdefault("codec_vs_previous", {})[codec] = (
                front_change(day, days[index - 1], codec) if index else None)
    return bases


def bd_values(point) -> dict:
    return {"j": rounded(point.bd_j),
            "iqr": [rounded(v) for v in point.iqr] if point.iqr else None,
            "clips_scored": point.scored, "chained": len(point.chained),
            "flagged": point.flagged}


# --- report ----------------------------------------------------------------------------

def runner() -> Runner:
    # J only (no XPSNR): the history reports BD-J; a result with XPSNR serves it as it is
    return Runner(load_paths(), EVALUATION_GATE, 1, over_budget=True)


def frame_curves(run: Runner, families: list[Family], clips: list[Clip],
                 low: list[Clip]) -> dict:
    """The families' cached curves on the clips, the low rungs added on `low`. Refuses to
    measure here: a missing point is the `points` step's (the PC carries no compute)."""
    jobs = (run.points(families, clips, DEFAULT_FRAMES, None)
            + run.points(families, low, DEFAULT_FRAMES, None, ladder=chart.low_ladder))
    missing = [p for bench, p in jobs if not bench.cached(p)]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(jobs)} points are not in the store "
                         f"(first: {missing[0].family.name} {missing[0].clip.name} "
                         f"{missing[0].label}): run the points step first")
    curves = run.measure("history", families, clips, DEFAULT_FRAMES, budgeted=False)
    extra = run.measure("history low rungs", families, low, DEFAULT_FRAMES, budgeted=False,
                        ladder=chart.low_ladder)
    for key, points in extra.items():
        curves[key] = curves[key] + points
    return curves


def frame_quality(record: dict, families: list[Family]):
    """The references' and `families`' BD-J vs the zero line on the record's basis
    (nceval.chart.anchor_points). A new family without a direct overlap on a basis clip
    does not change the basis: it is a side point there, chained through the others
    (nceval.chart.chain_clip, as the chart's side points). Fails when a recomputed
    reference differs from the stored one."""
    clips, low = basis_clips(record)
    run = runner()
    refs = reference_families()
    curves = frame_curves(run, [*refs, *families], clips, low)
    stored_out = sorted(e["clip"] for e in record["excluded"])
    _, excluded = chart.anchor_points(curves, [*refs, *families], clips, run,
                                      {f.name: f.name for f in [*refs, *families]})
    side_names = {n for e in excluded if e.clip not in stored_out for n in e.unscored}
    if side_names & {f.name for f in refs}:
        raise SystemExit(f"a reference lost its overlap on the basis: {sorted(side_names)}")
    core = [*refs, *(f for f in families if f.name not in side_names)]
    labels = {f.name: f.name for f in [*refs, *families]}
    points, excluded = chart.anchor_points(curves, core, clips, run, labels)
    if sorted(e.clip for e in excluded) != stored_out:
        raise SystemExit(f"the new frame would change the basis: clips left out "
                         f"{sorted(e.clip for e in excluded)}, stored {stored_out}")
    used = [c for c in clips if c.name not in stored_out]
    direct = {f.name: chart.per_clip_rows(curves, ZERO, f.name, clips) for f in core}
    for family in (f for f in families if f.name in side_names):
        rows = chart.per_clip_rows(curves, ZERO, family.name, used)
        chained = {c.name: link for c in used if rows[c.name]["j"] is None
                   and (link := chart.chain_clip(curves, ZERO, family, c, core, direct))}
        points.append(chart.chart_point(family, family.name, rows, used, run, curves, chained))
        print(f"{family.name}: no direct overlap with the zero line on "
              f"{sorted(c.name for c in used if rows[c.name]['j'] is None)}: chained "
              f"({len(chained)} clips)", flush=True)
    by_name = {p.family: p for p in points}
    moved = {name: (stored["j"], bd_values(by_name[name])["j"])
             for name, stored in record["references"].items()
             if (stored["j"], stored["iqr"]) != (bd_values(by_name[name])["j"],
                                                 bd_values(by_name[name])["iqr"])}
    if moved:
        raise SystemExit(f"references recomputed from the store differ from the stored "
                         f"ones (stored, now): {moved}")
    print(f"basis: {len(clips)} clips, low rungs on {len(low)}, left out {stored_out or 'none'}; "
          f"the {len(record['references'])} references reproduce", flush=True)
    return [by_name[f.name] for f in families]


def new_day(record: dict, args: argparse.Namespace) -> dict:
    """The frame of `args.day`: our record families' BD-J and fps on the record's basis."""
    families = record_families(args.record)
    anchor_ms = record["timing"]["x264_medium_ms_per_frame"]
    values, chained = {}, False
    previous = [d for d in record["days"] if d["date"] < args.day]
    last = previous[-1]["values"] if previous else {}
    for point in frame_quality(record, families):
        short = point.family.removeprefix("nanocodec-")
        fps, paired = timed_fps(args.commit, point.family, anchor_ms,
                                (last.get(short) or {}).get("fps_x86"))
        chained |= paired
        values[short] = {**bd_values(point), "fps_x86": rounded(fps)}
    captions = json.loads(CAPTIONS.read_text()) if CAPTIONS.is_file() else {}
    since = FIRST_COMMIT_DAY if not previous else day_after(previous[-1]["date"])
    day = {"date": args.day, "commit": args.commit, "values": values,
           "caption": captions.get(args.day, []), "activity": activity(args.day, args.commit, since),
           # what the fps time (nceval.front_k refuses slopes from any other timing than
           # the screens' CBR, owner 2026-11-23): the plan's rate mode, and the frame whose
           # absolute fps a paired timing is chained to
           front_k.TIMING_MODE_KEY: json.loads(args.plan.read_text()).get("rate_mode")}
    if chained:
        day[front_k.CHAINED_KEY] = previous[-1]["date"]
    if args.label:
        day["release"] = args.label
    if args.note:
        day["note"] = args.note
    return day


def day_after(date: str) -> str:
    import datetime

    return (datetime.date.fromisoformat(date) + datetime.timedelta(days=1)).isoformat()


def add_fronts(days: list[dict]) -> dict[str, str]:
    """Every frame's frontier, codec fronts and front changes (from its values). Returns
    each codec's base day."""
    for day in days:
        day["frontier"] = pareto_front(day["values"])
    base = next(d for d in days if d["date"] == FRONT_BASE_DAY)
    for index, day in enumerate(days):
        day["front_vs_base"] = front_change(day, base)
        day["front_vs_previous"] = front_change(day, days[index - 1]) if index else None
    return codec_front_changes(days)


def report_cmd(args: argparse.Namespace) -> None:
    record = stored_record(PAGE_DATA)
    stored_days = [d for d in record["days"] if d["date"] != args.day]
    # the anchor time the stored frames' fps were computed with, fixed from here on
    record["timing"].setdefault("x264_medium_ms_per_frame", json.loads(
        TIMING_X86.read_text())["points"][ANCHOR_TIMED]["ms_per_frame"])
    days = sorted([*stored_days, new_day(record, args)], key=lambda d: d["date"])
    before = {d["date"]: json.dumps({k: d[k] for k in DERIVED if k in d}, sort_keys=True)
              for d in stored_days}
    record["codec_front_base_days"] = add_fronts(days)
    changed = [d["date"] for d in days if d["date"] in before and before[d["date"]]
               != json.dumps({k: d[k] for k in DERIVED if k in d}, sort_keys=True)]
    if changed:  # a front is a function of its own frame's values (and the base frames')
        raise SystemExit(f"stored frames' fronts changed on recomputing: {changed}")
    record["days"] = days
    out = Path(args.out) if args.out else RESULTS / f"history-{days[-1]['date']}"
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1) + "\n"
    (out / "history.json").write_text(text)
    (out / "README.md").write_text(markdown(record))
    if not args.out:
        PAGE_DATA.write_text(text)
        stamp_page()
        # the acceptance screens' CPU charge follows the page's newest front
        front_k.TABLE.write_text(json.dumps(front_k.generate(PAGE_DATA), indent=1) + "\n")
        print(f"front slopes -> {front_k.TABLE} (commit it with the frame)")
    print(f"history: {len(days)} frames, {args.day} new -> {out / 'history.json'}"
          + ("" if args.out else f", {PAGE_DATA}"))


#: A frame's fields computed from its values (and the base frames'): `add_fronts`.
DERIVED = ("frontier", "front_vs_base", "front_vs_previous", "codec_fronts", "codec_vs_base",
           "codec_vs_previous")


def add_day_cmd(args: argparse.Namespace) -> None:
    points_cmd(args)
    time_cmd(args)
    if not args.dry_run:
        report_cmd(args)
        frames_cmd(args)


def preset_label(key: str) -> str:
    """`hevc@fast` -> HEVC Fast; `hevc` -> HEVC (its Default); `h264@retired-faster` ->
    H.264 Faster (retired)."""
    codec, _, stem = key.partition("@")
    retired = stem.startswith(RETIRED)
    stem = stem.removeprefix(RETIRED)
    return (CODEC_LABELS[codec] + (f" {stem.capitalize()}" if stem else "")
            + (" (retired)" if retired else ""))


def noise_text(noise: dict) -> str:
    """The stored frames' timing noise in one sentence (history.js noiseText says the same)."""
    moves = list(noise["unchanged"]["fps_change_pct"].values())
    text = (f"within a session a preset's repeats deviate from their median by "
            f"{noise['within_session_median_deviation_pct']:.1f} % (median over presets)")
    if moves:
        text += (f"; the {len(moves)} presets whose streams did not change from "
                 f"{noise['unchanged']['from']} to {noise['unchanged']['to']}, each timed in one "
                 f"session, moved by {min(moves):+.1f} to {max(moves):+.1f} % in fps")
    return text + "."


def markdown(record: dict) -> str:
    """The docs page: what was measured, then one row per day (J, fps, captions)."""
    days = record["days"]
    present = [c for c in CODECS if any(c in d["values"] for d in days)]
    slice_ = record["slice"]
    notes = [f"- **{d['date']}{' (release ' + d['release'] + ')' if d.get('release') else ''}:** "
             f"{d['note']}" for d in days if d.get("note")]
    lines = [
        f"# History of the headline chart, {days[0]['date']} to {days[-1]['date']}",
        "",
        "Generated by `tools/dev/history_chart.py report` (do not edit by hand); the animation is "
        "[web/results/history.html](../../../web/results/history.html), also as "
        "[history.gif](history.gif) and [history.mp4](history.mp4), one PNG per day beside them. "
        "The page is append-only: each frame was measured once, when it was added, on the "
        "basis below, and is kept as it was (preset names follow later rung renames).",
        "",
        "What was measured:",
        "",
        "- **Builds:** the last commit of each UTC day on the integration branch, as plain "
        "release builds (2026-11-06's build has no CBR rate control, so the history starts on "
        "2026-11-07, where H.264 alone is shown: that build's HEVC and AV1 refuse frame sizes "
        "that are not multiples of 8, most of the slice).",
        "- **Points:** every speed-ladder preset each build shipped, each with that build's own "
        "values, by today's rung names (0.5.0's Slow is Slower, Veryslow on HEVC and VVC; the old "
        "H.264 Fast is Faster, and the old H.264 Faster, retired on 2026-11-21, ends there), "
        "real-time CBR on the chart's ladder "
        f"({'/'.join(map(str, record['ladder_kbps']))} kbps at 720p30 scaled per clip, plus "
        f"{'/'.join(map(str, record['low_ladder_kbps']))} kbps on the {len(record['low_rung_clips'])} "
        f"clips whose curves need them), {record['frames']} frames, one key frame, our "
        "reconstruction scored (nceval.chart's engine).",
        f"- **Quality:** BD-J (PSNR-YUV, VMAF v1, MS-SSIM in equal thirds, class-balanced) against "
        f"{record['zero_label']}, on a fixed slice of {len(slice_['clips'])} of the "
        f"{record['fresh_clips']} held-out fresh clips ({slice_['rule']}: "
        + ", ".join(f"{n} {k}" for k, n in slice_["per_class"].items())
        + f"), averaged over the {record['clips_used']} clips on which every point and every "
        "reference overlaps the zero line"
        + (f" (left out: {', '.join(e['clip'] for e in record['excluded'])})"
           if record["excluded"] else "") + ".",
        "- **Speed:** fps timed on AWS c8a (Zen 5), one core, by the chart's own timing "
        "(`aws_timing.py`): every stored frame was timed with the chart's former timing "
        "commands (three 720p training clips, 150 frames, fixed QP found per build for about "
        f"{record['timing_rung_kbps']} kbps): a different encoder mode and other clips than the "
        "CBR quality points, so these fps are not the speed of the encodes scored (since "
        "2026-11-23 chart timing re-runs the quality encodes themselves, nceval.chart_timing; "
        "these frames are being re-timed that way), each run back to back "
        "with x264 medium, the median of its time ratios to x264 medium times the chart's stored "
        "x264 medium time; every frame is a plain release build (the overview's are PGO). The "
        "frames to 2026-11-20 were timed in one c8a.large per preset holding that preset's build "
        "of every one of those days; later frames each in their own sessions.",
        "- **Timing noise:** " + noise_text(record["timing_noise"]),
        *notes,
        "- **References:** the overview chart's encoders at today's versions, their quality "
        "re-averaged on the same clips, their fps as timed on the c8a.",
        "- **Activity:** from git on the integration branch (first parent): commits "
        f"(each row since the day after the previous row; the first from the first commit, "
        f"{FIRST_COMMIT_DAY}), lines added and "
        "removed by area (history.json `activity`), commits per agent where a trailer names it "
        "(Co-Authored-By / Claude-Session, Codex, Gemini; others carry no agent trailer), and "
        "the lines of the `.rs` files under crates/ at the day's build. "
        "The code is written and maintained by AI coding agents under the direction of one person.",
        "- **Captions:** each day's landings with the largest effect their own commit messages "
        "state, by codec (the numbers are those commits' own measurements on their own clip "
        "sets, not this chart's).",
        "",
        "| Day | Build | " + " | ".join(f"{c} Default J" for c in present) + " | "
        + " | ".join(f"{c} Default fps" for c in present)
        + " | Commits | Rust lines | Main changes since the previous row |",
        "|---|---|" + "---:|" * (2 * len(present) + 2) + "---|",
    ]
    for day in days:
        values = day["values"]
        j = [f"{values[c]['j']:+.1f} %" if c in values and values[c]["j"] is not None else "–"
             for c in present]
        speed = [f"{values[c]['fps_x86']:.0f}" if c in values
                 and values[c]["fps_x86"] else "–" for c in present]
        work = day["activity"]
        lines.append(f"| {day['date']} | `{day['commit']}` | " + " | ".join(j + speed)
                     + f" | {work['commits']:,} | {work['rust_lines_total']:,} | "
                     + "<br>".join(day["caption"]) + " |")
    lines += ["", "Our Pareto frontier each day (every codec and preset; BD-J, fps on c8a), and how "
              "it moved. Each front is BD-J as a piecewise-linear function of log10 fps; "
              "**J gap** is the mean difference in BD-J at equal speed over the overlapping speed "
              "range (negative: fewer bits), **speed** the mean speed factor at equal BD-J over "
              "the overlapping BD-J range (above 1: faster), never extrapolated beyond either "
              f"front; against {record['front_base_day']} (the first frame with every codec and "
              "several presets) and against the previous frame.", "",
              "| Day | J gap vs base | speed vs base | J gap vs previous | speed vs previous "
              "| Frontier, slowest first |", "|---|---:|---:|---:|---:|---|"]

    def change(value, key, fmt):
        return fmt.format(value[key]) if value and value[key] is not None else "–"
    for day in days:
        base, previous = day["front_vs_base"], day["front_vs_previous"]
        lines.append(f"| {day['date']} | {change(base, 'j_points', '{:+.1f}')} | "
                     f"{change(base, 'speed_factor', '{:.2f}x')} | "
                     f"{change(previous, 'j_points', '{:+.1f}')} | "
                     f"{change(previous, 'speed_factor', '{:.2f}x')} | " + "; ".join(
            f"{preset_label(k)} {day['values'][k]['j']:+.1f} % at "
            f"{day['values'][k]['fps_x86']:.0f} fps"
            for k in day["frontier"]) + " |")
    lines += ["", "References (fixed in every frame):", "",
              "| Encoder | J | fps (c8a) |", "|---|---:|---:|"]
    lines += [f"| {r['label']} | {r['j']:+.1f} % | {r['fps_x86']:.0f} |"
              for r in record["references"].values() if r["j"] is not None]
    return "\n".join(lines) + "\n"


# --- frames ----------------------------------------------------------------------------

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def page_handler(data: Path):
    """The site's web/ directory, its history data replaced by `data`."""
    class Handler(QuietHandler):
        def translate_path(self, path: str) -> str:
            if path.split("?")[0] == "/results/data/history.json":
                return str(data)
            return super().translate_path(path)
    return functools.partial(Handler, directory=str(REPO / "web"))


def frames_cmd(args: argparse.Namespace) -> None:
    """Screenshots of history.html?frame=N&still=1 (the page's own drawing, so the frames
    are the page's style), then the GIF and MP4 (ffmpeg), into the report's directory."""
    from playwright.sync_api import sync_playwright

    data = Path(args.out) / "history.json" if args.out else PAGE_DATA
    days = json.loads(data.read_text())["days"]
    out = Path(args.out) if args.out else RESULTS / f"history-{days[-1]['date']}"
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), page_handler(data))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/results/history.html"
    frames = out / "frames" if args.out else WORK / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": FRAME_WIDTH, "height": 900},
                                device_scale_factor=1, color_scheme="light")
        for index in range(len(days)):
            page.goto(f"{base}?frame={index}&still=1")
            page.wait_for_selector("body[data-ready=true]")
            page.locator(".chart-card").screenshot(path=str(frames / f"frame-{index:02d}.png"))
        browser.close()
    server.shutdown()
    pattern = str(frames / "frame-%02d.png")
    # each day FRAME_SECONDS, the last held longer
    timing = ["-framerate", f"1/{FRAME_SECONDS}", "-i", pattern]
    palette = (f"[0:v]tpad=stop_mode=clone:stop_duration={LAST_FRAME_HOLD},split[a][b];"
               "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *timing, "-filter_complex", palette,
                    "-r", "1", str(out / "history.gif")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *timing, "-vf",
                    f"tpad=stop_mode=clone:stop_duration={LAST_FRAME_HOLD},"
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-r", "10",
                    "-c:v", "libx264", "-crf", "28", "-movflags", "+faststart",
                    str(out / "history.mp4")], check=True)
    for index in range(len(days)):
        shutil.copyfile(frames / f"frame-{index:02d}.png",
                        out / f"frame-{index:02d}-{days[index]['date']}.png")
    print(f"frames: {len(days)} -> {out}")


def stamp_page() -> None:
    """history.html loads history.js and site.css with ?v=<hash of them and the data>, and
    history.js reads the data with the same version: after a publish a browser never pairs
    a cached page with newer code or data."""
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in PAGE_ASSETS)).hexdigest()[:10]
    text = PAGE.read_text()
    for asset in ("history.js", "site.css"):
        text = re.sub(rf'"{re.escape(asset)}(\?v=[0-9a-f]+)?"', f'"{asset}?v={digest}"', text)
    PAGE.write_text(text)


def rounded(value: float | None, places: int = 2) -> float | None:
    return None if value is None or math.isnan(value) else round(value, places)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("points", "time", "report", "frames", "add-day"):
        p = sub.add_parser(name)
        p.add_argument("--out", help="a scratch directory for the report and frames (default: "
                       "docs/results/history-<last day>/ and the page's data)")
        if name == "frames":
            continue
        p.add_argument("--record", type=Path, required=True,
                       help="the chart build record our families' points are on (e.g. "
                            "docs/results/bd-vs-speed-independent_fresh_2026-11-13-builds.json)")
        p.add_argument("--commit", required=name != "points", help="the frame's commit")
        p.add_argument("--day", required=name in ("report", "add-day"), help="the frame's date")
        p.add_argument("--binary", type=Path, required=name in ("time", "add-day"),
                       help="the commit's plain release build (timed)")
        p.add_argument("--baseline", type=Path,
                       help="the previous frame's commit as a plain release build, built the "
                            "same way as --binary: fps timed paired with it (see `time`)")
        p.add_argument("--plan", type=Path, default=TIMING_PLAN,
                       help="the chart's timing plan, scanned at the frame's build (default: "
                            "this tree's; a release's charts tree before it lands)")
        p.add_argument("--label", help="a release the frame is (e.g. 0.6.0)")
        p.add_argument("--note", help="how the frame was measured, where it differs")
        p.add_argument("--budget", type=float, default=2.0, help="each points session's cap, USD")
        p.add_argument("--dry-run", action="store_true", help="print the sessions; launch none")
    for name in ("share-points", "share-time", "share-report"):
        p = sub.add_parser(name, help=f"the 1-in-{SHARE.period} key-share view's {name[6:]}")
        p.add_argument("--record", type=Path, required=True, help="the frame's build record")
        p.add_argument("--baseline-record", type=Path, required=True,
                       help="the baseline's (0.6.0's) build record")
        p.add_argument("--commit", required=name != "share-points")
        p.add_argument("--binary", type=Path, required=name == "share-time",
                       help="the commit's plain release build (x86, timed)")
        p.add_argument("--baseline", type=Path, required=name == "share-time",
                       help="the baseline's plain release build (x86, timed beside it)")
        p.add_argument("--baseline-label", default="0.6.0")
        p.add_argument("--day", required=name == "share-report")
        p.add_argument("--out", help="a scratch directory for the report")
        p.add_argument("--budget", type=float, default=3.0, help="each points session's cap, USD")
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    {"points": points_cmd, "time": time_cmd, "report": report_cmd, "frames": frames_cmd,
     "add-day": add_day_cmd, "share-points": share_points_cmd, "share-time": share_time_cmd,
     "share-report": share_report_cmd}[args.command](args)


if __name__ == "__main__":
    main()
