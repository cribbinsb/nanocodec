#!/usr/bin/env python3
"""The headline chart's history: one build per day, re-measured under today's protocol.

The results site's headline chart (view `fresh-psnr`: BD-J against x264 medium --tune
psnr on the held-out fresh clips, x86 fps on the c8a) changed its clip set, zero line,
timing machine and preset names several times, so the dated chart files are not
comparable. This tool re-measures the last integration commit of each day with the
chart's own engine (nceval.chart: the real-use CBR ladder, low rungs, class-balanced
BD-J, our reconstruction scored, streamed) on a fixed slice of the fresh clips, and
times each build on the chart's x86 timing machine (AWS c8a.large, tools/dev/aws_timing.py
--binaries: one session per preset holding every day's build of it):

  slice    choose the slice once (SLICE_PER_CLASS clips per content class, within a
           class by a hash of the name) into OUT/slice.json.
  measure  one day's presets of every codec on the slice (the real-use ladder;
           the zero line's points are the chart's, cached). `--low` adds the low rungs
           once every day's ladder is cached (the chart's rule: the clips where some
           point lacks a reliable overlap with the zero line, plus those the published
           chart gave low rungs).
  scan     one day's timing quantizers: anchor_chart_timing.py's scan of each preset
           on the chart's three timing clips (fixed QP closest to the 600 kbps
           rung), into the work directory.
  time     the timing on AWS c8a.large (tools/dev/aws_timing.py, the chart's x86 timing):
           one session per preset, its scanned commands of every day's build side by side
           with x264 medium and SVT-AV1 M13, five repeats each.
  report   OUT/history.json: per day and preset, BD-J over the clips every point
           shares (nceval.chart.anchor_points over every day's points at once) and the
           timed x86 fps (`timed_fps`).

Builds are plain release builds of each commit (`--builds DIR`: DIR/<commit>/nanocodec),
made on the build box from `git archive` of each commit's crates, manifests and presets
(`cargo build --release -p nc-cli|nanocodec-cli` with NC_BUILD_COMMIT set, so each binary is
stamped with its commit; the 2026-11-10 build predates the stamp). Old builds that lack an
option the harness passes run through `compat_wrapper`. Our points are every speed-ladder
preset each build shipped (`day_presets`), each with that build's own preset values.
Output: docs/results/history-2026-11-20/ (the page: web/results/history.html).

    python3 tools/dev/history_chart.py slice
    python3 tools/dev/history_chart.py measure --day 2026-11-20 --builds DIR [--low]
    python3 tools/dev/history_chart.py scan --day 2026-11-20 --builds DIR
    python3 tools/dev/history_chart.py time --builds DIR
    python3 tools/dev/history_chart.py report --builds DIR
    python3 tools/dev/history_chart.py frames     # PNG per day, GIF and MP4 from the page
"""

from __future__ import annotations

import argparse
import datetime
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
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nceval import chart  # noqa: E402
from nceval.cbr import Family  # noqa: E402
from nceval.clips import Clip  # noqa: E402
from nceval.config import load_paths  # noqa: E402
from nceval.gatepolicy import EVALUATION_GATE  # noqa: E402
from nceval.realuse import DEFAULT_FRAMES, Runner, own_family  # noqa: E402
from nceval.encoders.nanocodec import MEDIUM, preset_path, presets_in  # noqa: E402

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
OUT = REPO / "docs" / "results" / "history-2026-11-20"
WORK = DATA / "work" / "history-anim"
#: The last integration-branch commit of each UTC day (git log --first-parent --before),
#: from the first day with CBR rate control: 2026-11-06's build (d9a7a04d8) takes only --qp.
DAYS = (
    ("2026-11-07", "e9e1c0cb6"), ("2026-11-08", "2ff69c9ad"), ("2026-11-09", "adaa8531b"),
    ("2026-11-10", "0f9a76433"), ("2026-11-11", "31d1a4809"), ("2026-11-12", "f4680a7a1"),
    ("2026-11-13", "f29b59338"), ("2026-11-14", "10844b79c"), ("2026-11-15", "594793b33"),
    ("2026-11-16", "2860f897e"), ("2026-11-17", "1f85eff44"), ("2026-11-18", "891c3f8a0"),
    ("2026-11-19", "da018cbbf"), ("2026-11-20", "5d33d7e2a"),
)
CODECS = ("h264", "hevc", "vvc", "av1", "vp9", "av2")
CODEC_LABELS = {"h264": "H.264", "hevc": "HEVC", "vvc": "VVC", "av1": "AV1", "vp9": "VP9",
                "av2": "AV2"}
#: The speed ladder's presets beside each codec's Default (medium), as the chart shows them.
LADDER_STEMS = ("slow", "fast", "faster", "veryfast")
#: Days whose build codes only some of its codecs on the slice: on 2026-11-07 HEVC and AV1
#: refuse sizes that are not multiples of 8 (960x540, most of the slice), and a refused
#: point would drop its clip for every point (anchor_points), so that day is H.264 only.
DAY_CODECS = {"2026-11-07": ("h264",)}
ZERO = "x264-medium~psnr"
FRESH = "independent_fresh_2026-11-13.json"
#: Clips per content class in the slice (the fresh set has 36-42 of each of crowd, pan
#: and static, 13 bodycam and 8 low-light ones). Class-balanced BD-J weights classes
#: equally, so the small classes keep enough clips to mean something.
SLICE_PER_CLASS = {"crowd": 6, "pan": 6, "static": 6, "bodycam": 5, "lowlight": 5}
TIMING_PLAN = REPO / "docs" / "results" / "bd-vs-speed-plan.json"
#: The headline view, its data (labels, owners, the references' c8a fps) and the page's copy.
VIEW = "fresh-psnr"
SITE_CHART = REPO / "web" / "results" / "data" / "chart.json"
PAGE_DATA = REPO / "web" / "results" / "data" / "history.json"
CAPTIONS = OUT / "captions.json"
#: The page and what its version stamp covers (`stamp_page`).
PAGE = REPO / "web" / "results" / "history.html"
PAGE_ASSETS = (REPO / "web" / "results" / "history.js", REPO / "web" / "results" / "site.css",
               PAGE_DATA)
#: The whole-front change is measured against this frame's frontier: the first with every
#: codec and more than one preset (2026-11-07 is H.264 alone, one point).
FRONT_BASE_DAY = "2026-11-09"
#: Evenly spaced points over an overlap range at which two fronts are compared.
FRONT_SAMPLES = 200
#: The c8a.large timing sessions of each day (one per part of its presets) and their anchor.
TIMING_DIR = WORK / "timing"
ANCHOR_TIMED = "x264-medium"
SECOND_ANCHOR_TIMED = "svtav1-m13"
#: Each preset is timed this many times in a row in its session, and the median ratio kept:
#: one repeat moved presets whose streams did not change by up to 5.6 % between frames.
TIMING_REPEATS = 5
#: The run's label in the session directories.
TIMING_RUN = "same-box"
#: The slice clips with the low rungs: the chart's rule (nceval.chart.low_rung_clips) applied
#: to every day's Default and the references (the Default-only history, 65a803718): the 13
#: the published chart gave low rungs plus ind_bwc_videotext_video190.
LOW_RUNG_CLIPS = frozenset((
    "ind_personpath22_uid_vid_00177", "ind_mot_mot20_01", "ind_personpath22_uid_vid_00195",
    "ind_bdd100k_mot_bdd_00d1bafa_1b47b41c", "ind_jaad_video_0042", "ind_otw_homes_00310",
    "ind_personpath22_uid_vid_00079", "ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508",
    "ind_bwc_videotext_video317", "ind_bwc_videotext_video190", "ind_bwc_videotext_video321",
    "ind_bdd100k_mot_bdd_00cea101_06f21d5e", "ind_bwc_videotext_video28",
    "ind_bwc_videotext_video354"))
#: Agent families as the commit trailers name them (first match wins); a commit without
#: one of these trailers is counted as `unmarked`.
AGENT_TRAILERS = (("Claude", r"^(Co-Authored-By: Claude|Claude-Session:)"),
                  ("Codex", r"^(Co-Authored-By: (OpenAI )?Codex|Assisted-by: OpenAI Codex)"),
                  ("Gemini", r"^(Co-Authored-By|Assisted-by):.*Gemini"))
COMMIT_MARK = "\x1ecommit\x1e"
#: The repository's first commit (421c8142d): the first frame's activity counts from it.
FIRST_COMMIT_DAY = "2026-11-06"
ZERO_LABEL = "x264 medium --tune zerolatency,psnr"
#: The animation: frame width (px), seconds per day, and the last day's extra hold.
FRAME_WIDTH = 1000
FRAME_SECONDS = 1
LAST_FRAME_HOLD = 3
TIMING_X86 = REPO / "docs" / "results" / "bd-vs-speed-timing-x86.json"
#: The build before 2026-11-11 has no --display-recon: it writes the same pictures with
#: --recon (no frame drops or holds without --frame-drop, one layer, no reordering).
#: It also seeks in its input, which the harness streams through a pipe.
DISPLAY_RECON = "--display-recon"
OLD_RECON = "--recon"
#: Flags the harness passes that builds before 2026-11-08 lack (compat_wrapper).
LAYERS_FLAG = "--temporal-layers"
TIMES_FLAG = "--timestamps-out"
LOAN_COLUMN = "key_loan_bits"


def day_commit(day: str) -> str:
    return dict(DAYS)[day]


def binary(builds: Path, commit: str) -> Path:
    """The commit's CLI, wrapped when it predates --display-recon (`compat_wrapper`)."""
    path = builds / commit / "nanocodec"
    if not path.is_file():
        raise SystemExit(f"no build of {commit} at {path}")
    return path if DISPLAY_RECON in encode_usage(path) else compat_wrapper(path)


def encode_usage(tool: Path) -> str:
    """`nanocodec --help` (on stdout or stderr, by build)."""
    done = subprocess.run([str(tool), "--help"], capture_output=True, text=True)
    return done.stdout + done.stderr


def compat_wrapper(path: Path) -> Path:
    """A script running `path` with --display-recon spelled --recon and a piped --input
    first copied to a file under WORK (removed when the encode ends). For a build before
    2026-11-08 it also drops `--temporal-layers 1` (one layer was all such a build coded)
    and writes --timestamps-out itself (every input frame coded at nominal times: such a
    build had no frame drop or hold); before the rate log's key_loan_bits column
    (98663eae4, 2026-11-08) it adds that column as 0, so a key loan such a build carried at
    the end counts as overshoot (rate validity only; BD uses the achieved rate). It names the binary's SHA-256, so its own bytes (the
    measurement caches' key for a build without a --version stamp) identify the binary it
    runs."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    usage = encode_usage(path)
    layers, times = LAYERS_FLAG not in usage, TIMES_FLAG not in usage
    loan = LOAN_COLUMN.encode() not in path.read_bytes()
    adapted = "".join([f", without {LAYERS_FLAG}" if layers else "",
                       f", {TIMES_FLAG} written here" if times else "",
                       f", {LOAN_COLUMN} 0 added to --rate-log" if loan else ""])
    wrapper = path.with_name("nanocodec-compat")
    loop_extra = (f"""  if [ "$previous" = {LAYERS_FLAG} ]; then [ "$a" = 1 ]; previous=$a; continue; fi
  if [ "$a" = {LAYERS_FLAG} ]; then previous=$a; continue; fi
""" if layers else "") + (f"""  if [ "$previous" = {TIMES_FLAG} ]; then times=$a; previous=$a; continue; fi
  if [ "$a" = {TIMES_FLAG} ]; then previous=$a; continue; fi
  [ "$previous" = --frames ] && frames=$a
  [ "$previous" = --fps ] && rate=$a
""" if times else "") + ("""  [ "$previous" = --rate-log ] && log=$a
""" if loan else "")
    after = """if [ -n "$times" ]; then
  awk -v n="$frames" -v r="$rate" 'BEGIN { split(r, f, "/"); d = (2 in f) ? f[2] : 1;
    for (i = 0; i < n; i++) printf "%.6f\\n", i * 1000 * d / f[1] }' > "$times"
fi
""" if times else ""
    after += f"""if [ -n "$log" ]; then
  awk -F, 'NR == 1 {{ print $0 ",{LOAN_COLUMN}"; next }} {{ print $0 ",0" }}' "$log" > "$log.loan"
  mv -- "$log.loan" "$log"
fi
""" if loan else ""
    wrapper.write_text(f"""#!/usr/bin/env bash
# runs {path} (sha256 {digest}): {DISPLAY_RECON} as {OLD_RECON}, a piped --input as a file{adapted}
set -euo pipefail
copy=""
trap '[ -z "$copy" ] || rm -f -- "$copy"' EXIT
args=(); previous=""{' times=""' if times else ""}{' log=""' if loan else ""}
for a in "$@"; do
{loop_extra}  if [ "$previous" = --input ] && [ ! -f "$a" ]; then
    copy=$(mktemp -p {WORK} input-XXXXXX.yuv); cat -- "$a" > "$copy"; previous=$a; a=$copy
  else
    previous=$a
  fi
  [ "$a" = {DISPLAY_RECON} ] && a={OLD_RECON}
  args+=("$a")
done
{path} "${{args[@]}}"
{after}""")
    wrapper.chmod(0o755)
    return wrapper


# --- slice -----------------------------------------------------------------------------

def name_hash(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def fresh_clips() -> list[Clip]:
    return chart.chart_clips(str(load_paths().clips / FRESH))


def choose_slice(clips: list[Clip]) -> list[Clip]:
    """SLICE_PER_CLASS clips of each class, the first by a hash of the name."""
    chosen = []
    for content_class, count in SLICE_PER_CLASS.items():
        members = sorted((c for c in clips if c.content_class == content_class),
                         key=lambda c: name_hash(c.name))
        chosen += members[:count]
    return chosen


def slice_cmd(args: argparse.Namespace) -> None:
    clips = choose_slice(fresh_clips())
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"clipset": FRESH, "rule": "per content class, the first clips by SHA-256 of the "
                                        "clip name", "per_class": SLICE_PER_CLASS,
              "clips": [{"name": c.name, "class": c.content_class, "width": c.width,
                         "height": c.height, "fps": c.fps} for c in clips]}
    (OUT / "slice.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"slice: {len(clips)} clips -> {OUT / 'slice.json'}")


def slice_clips() -> list[Clip]:
    names = [c["name"] for c in json.loads((OUT / "slice.json").read_text())["clips"]]
    by_name = {c.name: c for c in fresh_clips()}
    return [by_name[n] for n in names]


# --- measure ---------------------------------------------------------------------------

def reference_families() -> list[Family]:
    """The zero line first, then the chart's reference encoders (their points on the
    slice are the published chart's, cached)."""
    families = chart.chart_families(Path(sys.executable), chart.PRESETS_DIR,
                                    [ZERO, *(name for name, _ in chart.REFERENCES)])
    return sorted(families, key=lambda f: f.name != ZERO)


def zero_family() -> Family:
    return reference_families()[0]


def codecs_of(tool: Path) -> list[str]:
    """The codecs a build's encoder takes (all six from 2026-11-09; read from its usage)."""
    listed = re.search(r"--codec ([a-z0-9|]+)", encode_usage(tool))
    return [c for c in CODECS if listed and c in listed.group(1).split("|")]


def day_families(builds: Path, day: str) -> list[Family]:
    """Every speed-ladder preset of every codec the day's build has, Default (medium)
    first, each with that build's own preset values (`day_presets`)."""
    commit = day_commit(day)
    tool, presets = binary(builds, commit), presets_dir(builds, commit)
    return [own_family(family_name(codec, stem), tool, presets)
            for codec in day_codecs(day, tool) for stem in day_presets(presets, codec)]


def presets_dir(builds: Path, commit: str) -> Path:
    """The commit's presets (DIR/<commit>/tree: `git archive` of its presets directory)."""
    return presets_in(builds / commit / "tree")


def day_presets(presets: Path, codec: str) -> list[str]:
    """medium, then each LADDER_STEMS preset the build ships for `codec`. Builds before
    2026-11-15 keep them flat as presets/<codec>-fast|faster|fastest.toml (the speed
    ladder's fast, faster, veryfast: nanocodec.preset_path maps them); their other files
    (h264-balanced, hevc-better, default) are not ladder presets and are left out."""
    return [MEDIUM, *(stem for stem in LADDER_STEMS
                      if preset_path(presets, codec, stem).is_file())]


def family_name(codec: str, stem: str) -> str:
    return f"nanocodec-{codec}" + ("" if stem == MEDIUM else f"@{stem}")


def day_codecs(day: str, tool: Path) -> list[str]:
    return [c for c in codecs_of(tool) if c in DAY_CODECS.get(day, CODECS)]


def runner(workers: int) -> Runner:
    # J only (no XPSNR): the history reports BD-J
    return Runner(load_paths(), EVALUATION_GATE, workers, over_budget=True)


def low_clips(builds: Path, run: Runner, clips: list[Clip]) -> list[Clip]:
    """LOW_RUNG_CLIPS: fixed, so every day's low rungs can be measured while other days'
    ladders are still running. A clip some preset needs beyond them is left out or chained
    by anchor_points, and the report lists it."""
    return [c for c in clips if c.name in LOW_RUNG_CLIPS]


def renamed(family: Family, day: str) -> Family:
    return replace(family, name=f"{family.name}#{day}")


def all_ladders(builds: Path, run: Runner, clips: list[Clip],
                ladder=None) -> tuple[dict, list[Family]]:
    """The references' and every day's cached curves (ours under day-qualified names),
    the zero line first."""
    families = reference_families()
    curves = run.measure("history references", families, clips, DEFAULT_FRAMES,
                         budgeted=False, ladder=ladder)
    for day, _ in DAYS:
        own = day_families(builds, day)
        measured = run.measure(f"history {day}", own, clips, DEFAULT_FRAMES, budgeted=False,
                               ladder=ladder)
        for family in own:
            families.append(renamed(family, day))
            for clip in clips:
                curves[(families[-1].name, clip.name)] = measured[(family.name, clip.name)]
    return curves, families


def measure_cmd(args: argparse.Namespace) -> None:
    builds, run, clips = Path(args.builds), runner(args.workers), slice_clips()
    families = [*reference_families(), *day_families(builds, args.day)]
    if args.low:
        clips = low_clips(builds, run, clips)
        jobs = run.points(families, clips, DEFAULT_FRAMES, None, ladder=chart.low_ladder)
    else:
        jobs = run.points(families, clips, DEFAULT_FRAMES, None)
    todo = [j for j in jobs if not j[0].cached(j[1])]
    print(f"history measure {args.day}{' low rungs' if args.low else ''}: {len(clips)} clips, "
          f"{len(families)} families, {len(todo)} of {len(jobs)} points to measure", flush=True)
    run.run_jobs(todo)
    print("refused:", {k: sorted(v) for k, v in run.refused.items()} or "none", flush=True)


# --- speed -----------------------------------------------------------------------------

def plan_path(commit: str) -> Path:
    return WORK / "plans" / f"plan-{commit}.json"


def scan_cmd(args: argparse.Namespace) -> None:
    commit = day_commit(args.day)
    tool = binary(Path(args.builds), commit)
    plan_path(commit).parent.mkdir(parents=True, exist_ok=True)
    points = ",".join(f.name for f in day_families(Path(args.builds), args.day))
    subprocess.run([sys.executable, str(REPO / "tools/dev/anchor_chart_timing.py"), "scan",
                    "--binary", str(tool), "--presets-dir",
                    str(presets_dir(Path(args.builds), commit)),
                    "--plan", str(plan_path(commit)), "--points", points],
                   check=True, cwd=REPO / "eval", env={**os.environ, "PYTHONPATH": "."})


def cloud_plan(builds: Path, commit: str) -> Path:
    """The day's scanned plan for aws_timing.py: our argv run as `{BIN}/nanocodec` (the build
    itself, not compat_wrapper's script; `--temporal-layers 1` left out where the build lacks
    the flag) and the chart's x264 medium and SVT-AV1 M13 entries it pairs every point with."""
    plan = json.loads(plan_path(commit).read_text())
    usage = encode_usage(builds / commit / "nanocodec")
    points = {}
    for family, clips in plan["points"].items():
        for clip, entry in clips.items():
            argv = ["{BIN}/nanocodec", *entry["argv"][1:]]
            if LAYERS_FLAG not in usage and LAYERS_FLAG in argv:
                at = argv.index(LAYERS_FLAG)
                del argv[at:at + 2]
            points.setdefault(family, {})[clip] = {**entry, "argv": argv}
    published = json.loads(TIMING_PLAN.read_text())["points"]
    for anchor in (ANCHOR_TIMED, SECOND_ANCHOR_TIMED):
        points[anchor] = published[anchor]
    path = TIMING_DIR / f"cloud-plan-{commit}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**plan, "points": points, "anchor": ANCHOR_TIMED}, indent=1))
    return path


def family_plan(builds: Path, family: str) -> tuple[Path, dict[str, Path]]:
    """One preset's timing plan across every day: each day's scanned commands for it
    (cloud_plan) as point `<family>#<commit>`, run by that day's build staged as
    bin/nc-<commit>, beside the x264 medium and SVT-AV1 M13 anchors. Returns the plan and the
    builds it names."""
    points, binaries = {}, {}
    for _, commit in DAYS:
        day_plan = json.loads(cloud_plan(builds, commit).read_text())
        if family not in day_plan["points"]:
            continue
        name = f"nc-{commit}"
        binaries[name] = builds / commit / "nanocodec"
        points[f"{family}#{commit}"] = {clip: {**entry, "argv": [f"{{BIN}}/{name}", *entry["argv"][1:]]}
                                        for clip, entry in day_plan["points"][family].items()}
    for anchor in (ANCHOR_TIMED, SECOND_ANCHOR_TIMED):
        points[anchor] = day_plan["points"][anchor]
    path = TIMING_DIR / f"family-plan-{family}.json"
    path.write_text(json.dumps({**day_plan, "points": points}, indent=1))
    return path, binaries


def time_cmd(args: argparse.Namespace) -> None:
    """Every preset in its own c8a.large session (all side by side), its builds from every
    day in that one session, each timed TIMING_REPEATS times in a row (aws_timing.py
    --repeats): the day-to-day differences are then the same machine's."""
    builds = Path(args.builds)
    families = sorted({f for _, commit in DAYS for f in json.loads(plan_path(commit).read_text())["points"]
                       if f.startswith("nanocodec-")})
    sessions = []
    for family in families:
        plan, binaries = family_plan(builds, family)
        points = [p for p in json.loads(plan.read_text())["points"] if p.startswith("nanocodec-")]
        sessions.append(subprocess.Popen(
            [sys.executable, str(REPO / "tools/dev/aws_timing.py"), "--plan", str(plan),
             "--points", ",".join(points), *(f"--binaries={n}={p}" for n, p in binaries.items()),
             "--repeats", str(TIMING_REPEATS), "--out", str(TIMING_DIR / f"fam-{family}-{TIMING_RUN}"),
             "--budget", "2", "--lifetime", "150", "--purpose", "history-timing"],
            stdout=(TIMING_DIR / f"fam-{family}.log").open("w"), stderr=subprocess.STDOUT))
    failed = [s.args for s in sessions if s.wait()]
    if failed:
        raise SystemExit(f"timing sessions failed: {len(failed)}")


def timed_fps(commit: str, family: str, published: dict) -> float | None:
    """The chart's x86 fps of `family` at `commit` from that preset's same-box session
    (TIMING_DIR/fam-<family>-<run>, tools/dev/aws_timing.py): the median over the repeats of
    the geometric-mean time ratio to x264 medium (run back to back with it), times the
    stored x264 medium time (bd-vs-speed-timing-x86.json), as the chart."""
    summary = TIMING_DIR / f"fam-{family}-{TIMING_RUN}" / "summary.json"
    entry = json.loads(summary.read_text()).get(f"{family}#{commit}") if summary.is_file() else None
    ratios = [v["ratio_x264_medium"] for k, v in (entry or {}).items() if k.startswith("repeat-")]
    if not ratios:
        return None
    anchor_ms = list(published[ANCHOR_TIMED]["ms_per_frame"].values())
    return 1000.0 / (statistics.median(ratios) * chart.geomean(anchor_ms))


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


def quality_points(builds: Path, run: Runner, clips: list[Clip]):
    """Every reference and every day's point vs the zero line on one shared clip set
    (nceval.chart.anchor_points), with the low rungs where the chart's rule puts them."""
    curves, families = all_ladders(builds, run, clips)
    low = low_clips(builds, run, clips)
    if low:
        extra, _ = all_ladders(builds, run, low, ladder=chart.low_ladder)
        for key, points in extra.items():
            curves[key] = curves[key] + points
    points, excluded = chart.anchor_points(curves, families, clips, run,
                                           {f.name: f.name for f in families})
    return points, excluded, low


def period_start(day: str) -> str:
    """The first day a frame's activity counts: the day after the previous frame's (the
    first frame: the repository's first commit)."""
    index = [d for d, _ in DAYS].index(day)
    if index == 0:
        return FIRST_COMMIT_DAY
    after = datetime.date.fromisoformat(DAYS[index - 1][0]) + datetime.timedelta(days=1)
    return after.isoformat()


def pareto_front(values: dict) -> list[str]:
    """Our points no other of the day beats on both timed speed and BD-J, slowest
    first (as web/results/frontier.js)."""
    points = {k: (v["fps_x86"], v["j"]) for k, v in values.items()
              if v["j"] is not None and v["fps_x86"]}
    front = [k for k, (x, y) in points.items()
             if not any((x2 >= x and y2 <= y and (x2 > x or y2 < y))
                        for k2, (x2, y2) in points.items() if k2 != k)]
    return sorted(front, key=lambda k: points[k][0])


def unchanged_streams(day: dict, previous: dict, family: str) -> bool:
    """Whether `family` codes the same streams on both days: the same BD-J and, on every
    timing clip, the same scanned quantizer and stream size."""
    if day["values"][family]["j"] != previous["values"][family]["j"]:
        return False
    plans = [json.loads(plan_path(d["commit"]).read_text())["points"] for d in (day, previous)]
    entries = [p.get(f"nanocodec-{family}") for p in plans]
    return all(entries) and all(
        (entries[0][c]["qp"], entries[0][c]["bytes"]) == (entries[1][c]["qp"], entries[1][c]["bytes"])
        for c in entries[0])


def timing_noise(days: list[dict]) -> dict:
    """The timing's own noise: within a session, each preset's repeats' deviation from their
    median (median over presets); and how far the fps of the presets coding unchanged
    streams (`unchanged_streams`) moved from the second-last to the last frame, both timed
    in that preset's one session."""
    deviations = []
    for summary in TIMING_DIR.glob(f"fam-*-{TIMING_RUN}/summary.json"):
        for name, entry in json.loads(summary.read_text()).items():
            ratios = [v["ratio_x264_medium"] for k, v in entry.items() if k.startswith("repeat-")]
            if name.startswith("nanocodec-") and len(ratios) > 1:
                middle = statistics.median(ratios)
                deviations.append(statistics.median(abs(r / middle - 1) * 100 for r in ratios))
    previous, day = days[-2], days[-1]
    unchanged = {f: round((day["values"][f]["fps_x86"] / previous["values"][f]["fps_x86"] - 1) * 100, 2)
                 for f in sorted(set(day["values"]) & set(previous["values"]))
                 if day["values"][f]["fps_x86"] and previous["values"][f]["fps_x86"]
                 and unchanged_streams(day, previous, f)}
    return {"within_session_median_deviation_pct": round(statistics.median(deviations), 2)
            if deviations else None,
            "unchanged": {"from": previous["date"], "to": day["date"], "fps_change_pct": unchanged}}


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


def report_cmd(args: argparse.Namespace) -> None:
    clips = slice_clips()
    points, excluded, low = quality_points(Path(args.builds), runner(args.workers), clips)
    site = json.loads(SITE_CHART.read_text())
    view = next(v for v in site["views"] if v["id"] == VIEW)
    published = json.loads(TIMING_X86.read_text())["points"]
    # each day's main landings, from its commit messages (CAPTIONS: written by hand from git log)
    captions = json.loads(CAPTIONS.read_text()) if CAPTIONS.is_file() else {}
    references = {p.family: {**bd_values(p), "label": site["points"][p.family]["label"],
                             "owner": site["points"][p.family]["owner"],
                             "fps_x86": view["values"][p.family]["fps"]["x86"]}
                  for p in points if "#" not in p.family}
    days = []
    for day, commit in DAYS:
        values = {}
        for p in points:
            family, _, at = p.family.partition("#")
            if at == day:
                values[family.removeprefix("nanocodec-")] = {
                    **bd_values(p),
                    "fps_x86": rounded(timed_fps(commit, family, published))}
        days.append({"date": day, "commit": commit, "values": values,
                     "frontier": pareto_front(values),
                     "caption": captions.get(day, []), "activity": activity(day, commit, period_start(day))})
    noise = timing_noise(days)
    base = next(d for d in days if d["date"] == FRONT_BASE_DAY)
    for index, day in enumerate(days):
        day["front_vs_base"] = front_change(day, base)
        day["front_vs_previous"] = front_change(day, days[index - 1]) if index else None
    codec_bases = codec_front_changes(days)
    record = {
        "view": VIEW, "front_base_day": FRONT_BASE_DAY, "codec_front_base_days": codec_bases, "zero": ZERO, "zero_label": ZERO_LABEL, "fresh_clips": view["clips"],
        "ladder_kbps": site["ladder_kbps"], "low_ladder_kbps": site["low_ladder_kbps"],
        "frames": site["frames"], "timing_rung_kbps": site["timing_rung_kbps"],
        "slice": json.loads((OUT / "slice.json").read_text()),
        "clips_used": len(clips) - len(excluded),
        "excluded": [{"clip": e.clip, "class": e.content_class, "unscored": list(e.unscored)}
                     for e in excluded],
        "low_rung_clips": [c.name for c in low],
        "timing": {"anchor_times": str(TIMING_X86.relative_to(REPO)),
                   "machine": "AWS c8a.large (Zen 5), one core"},
        "references": references,
        "timing_noise": noise,
        "days": days,
    }
    text = json.dumps(record, indent=1) + "\n"
    for path in (OUT / "history.json", PAGE_DATA):
        path.write_text(text)
    (OUT / "README.md").write_text(markdown(record))
    stamp_page()
    print(f"history: {len(days)} days -> {OUT / 'history.json'}, {PAGE_DATA}")


def preset_label(key: str) -> str:
    """`hevc@fast` -> HEVC Fast; `hevc` -> HEVC (its Default)."""
    codec, _, stem = key.partition("@")
    return CODEC_LABELS[codec] + (f" {stem.capitalize()}" if stem else "")


def noise_text(noise: dict) -> str:
    """The timing noise in one sentence (the page says the same, history.js noiseText)."""
    moves = list(noise["unchanged"]["fps_change_pct"].values())
    text = (f"within a session a preset's {TIMING_REPEATS} repeats deviate from their median by "
            f"{noise['within_session_median_deviation_pct']:.1f} % (median over presets)")
    if moves:
        text += (f"; the {len(moves)} presets whose streams did not change from "
                 f"{noise['unchanged']['from']} to {noise['unchanged']['to']}, each timed in one "
                 f"session, moved by {min(moves):+.1f} to {max(moves):+.1f} % in fps")
    return text + "."


def markdown(record: dict) -> str:
    """The docs page: what was measured, then one row per day (J, fps, captions)."""
    present = [c for c in CODECS if any(c in d["values"] for d in record["days"])]
    slice_ = record["slice"]
    lines = [
        "# History of the headline chart, 2026-11-07 to 2026-11-20",
        "",
        "Generated by `tools/dev/history_chart.py report` (do not edit by hand); the animation is "
        "[web/results/history.html](../../../web/results/history.html), also as "
        "[history.gif](history.gif) and [history.mp4](history.mp4), one PNG per day beside them.",
        "",
        "What was measured:",
        "",
        "- **Builds:** the last commit of each UTC day on the integration branch (2026-11-20: "
        "its head at 07:52), built now as plain release builds; 2026-11-06's build has no CBR rate control (`--qp` only), so the "
        "history starts on 2026-11-07, where H.264 alone is shown (that build's HEVC and AV1 "
        "refuse frame sizes that are not multiples of 8, most of the slice).",
        "- **Points:** every speed-ladder preset each build shipped (Default, and Slow, Fast, "
        "Faster, Veryfast where present; before 2026-11-15 the files were `<codec>-fast`, "
        "`-faster`, `-fastest`, mapped to Fast, Faster, Veryfast; other files such as "
        "`h264-balanced` and `hevc-better` are not on the ladder and are left out), each with "
        "that build's own values, real-time CBR on the chart's ladder "
        f"({'/'.join(map(str, record['ladder_kbps']))} kbps at 720p30 scaled per clip, plus "
        f"{'/'.join(map(str, record['low_ladder_kbps']))} kbps on the {len(record['low_rung_clips'])} "
        f"clips whose curves need them), {record['frames']} frames, one key frame, our "
        "reconstruction scored (nceval.chart's engine).",
        f"- **Quality:** BD-J (PSNR-YUV, VMAF v1, MS-SSIM in equal thirds, class-balanced) against "
        f"{record['zero_label']}, on a fixed slice of {len(slice_['clips'])} of the "
        f"{record['fresh_clips']} held-out fresh clips ({slice_['rule']}: "
        + ", ".join(f"{n} {k}" for k, n in slice_["per_class"].items())
        + f"), averaged over the {record['clips_used']} clips on which every point of every day "
        "and every reference overlaps the zero line"
        + (f" (left out: {', '.join(e['clip'] for e in record['excluded'])})"
           if record["excluded"] else "") + ".",
        "- **Speed:** fps timed on AWS c8a (Zen 5), one core, by the chart's own timing "
        "(`aws_timing.py`): the chart's timing commands (three 720p clips, 150 frames, fixed QP "
        f"found per build for about {record['timing_rung_kbps']} kbps), each run back to back "
        "with x264 medium in rotated rounds, one c8a.large per preset holding that preset's "
        f"build of every day (so the day-to-day differences are the same machine's), each timed "
        f"{TIMING_REPEATS} times in a row and the median of its time ratios to x264 medium kept, "
        "times the chart's stored x264 medium time; every day "
        "is a plain release build (the overview's are PGO).",
        "- **Timing noise:** " + noise_text(record["timing_noise"]),

        "- **References:** the overview chart's encoders at today's versions, their quality "
        "re-averaged on the same clips, their fps as timed on the c8a.",
        "- **Activity:** from git on the integration branch (first parent): commits "
        f"(each row since the day after the previous row; the first from the first commit, "
        f"{FIRST_COMMIT_DAY}), lines added and "
        "removed by area (history.json `activity`), commits per agent where a trailer names it "
        "(Co-Authored-By / Claude-Session, Codex; none name Gemini in this period, others carry "
        "no agent trailer), and the lines of the `.rs` files under crates/ at the day's build. "
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
    for day in record["days"]:
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
    for day in record["days"]:
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


def frames_cmd(args: argparse.Namespace) -> None:
    """Screenshots of history.html?frame=N&still=1 (the page's own drawing, so the frames
    are the page's style), then the GIF and MP4 (ffmpeg) into OUT."""
    from playwright.sync_api import sync_playwright

    handler = functools.partial(QuietHandler, directory=str(REPO / "web"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/results/history.html"
    days = json.loads(PAGE_DATA.read_text())["days"]
    frames = WORK / "frames"
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
                    "-r", "1", str(OUT / "history.gif")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *timing, "-vf",
                    f"tpad=stop_mode=clone:stop_duration={LAST_FRAME_HOLD},"
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-r", "10",
                    "-c:v", "libx264", "-crf", "28", "-movflags", "+faststart",
                    str(OUT / "history.mp4")], check=True)
    for index in range(len(days)):
        shutil.copyfile(frames / f"frame-{index:02d}.png",
                        OUT / f"frame-{index:02d}-{days[index]['date']}.png")
    print(f"frames: {len(days)} -> {OUT}")


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
    sub.add_parser("slice")
    sub.add_parser("frames")
    for name in ("measure", "scan", "time", "report"):
        p = sub.add_parser(name)
        p.add_argument("--builds", required=True, help="DIR/<commit>/nanocodec")
        p.add_argument("--workers", type=int, default=8)
        if name not in ("report", "time"):
            p.add_argument("--day", required=True,
                           help="a DAYS date")
        if name == "measure":
            p.add_argument("--low", action="store_true", help="the low rungs (LOW_RUNG_CLIPS)")
    args = parser.parse_args()
    {"slice": slice_cmd, "frames": frames_cmd, "measure": measure_cmd, "scan": scan_cmd, "time": time_cmd,
     "report": report_cmd}[args.command](args)


if __name__ == "__main__":
    main()
