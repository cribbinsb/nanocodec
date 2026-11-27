#!/usr/bin/env python3
"""Refresh the bitrate-vs-speed charts for the families one commit changes.

    python3 tools/dev/chart_refresh.py COMMIT --families nanocodec-h264,nanocodec-h264@fast \\
        [--rescore] [--from STEP] [--workers 8] [--cpus 14,15] [--dry-run]

Run it from your own worktree: it writes that tree's docs/results (the
charts, their plan, timing and build record), which you then commit and land.
Every chart family is measured on its own build, recorded in
docs/results/bd-vs-speed-builds.json (family -> build id; build id ->
description); a build lives in BUILDS_ROOT/<id>/ on the PC (`nanocodec`, the
scoring build; `nanocodec-timed-release`, the plain release build x86
timing uses; the
commit's `presets/`) and on the Jetson (`nanocodec`, release), each with a
`COMMIT` file written last (a build without it is incomplete). The steps
(`--from` resumes at one; the steps before it are skipped):

  build        COMMIT built for x86 in the tool's own worktrees and target dirs:
               the scoring build (`gate` profile, incremental: seconds to a minute)
               first, the timed release build (fat LTO, minutes) in the
               background, and for the Jetson, also in the background (git archive
               over ssh into a persistent source tree, under jetson-build.sh;
               only changed files get new times, so cargo rebuilds incrementally).
               A family whose sources (crates/, Cargo.*, presets/) did not change
               since its recorded build keeps that build: its cached points stay
               valid. A build that already exists is kept; one whose binary
               sources (all but presets/) equal a recorded build's reuses that
               binary on both machines and only takes COMMIT's presets.
  verify       for --retime families: their streams at COMMIT must equal their
               recorded build's (tools/dev/stream_check.py: the timing plan's
               CBR timing encodes, by SHA-256); they keep their quality points and are timed on COMMIT's
               build (`timed_on` in bd-vs-speed-builds.json).
  delete       the replaced build's cached chart points of each refreshed family
               (VAL ladder and low rungs, by exact identity: cbr.Bench.result_path),
               so each curve keeps one result version.
  plan         the timing plan of the refreshed and retimed families (anchor_chart_timing.py
               plan --all-encodes, on the build box): every one of their CBR quality
               encodes (nceval.chart_timing.every_job: all clips and rungs, the low rungs on
               --low-clips), into the chart's plan (docs/results/bd-vs-speed-plan.json;
               bd-vs-speed-<set>-plan.json on a clip set).
  measure      every point the render reads (ladder and low rungs of every family of
               the record, the references, both zero lines, the metric-tuned side
               points; cached ones cost nothing), encoded and scored on rented machines
               in small sessions side by side (tools/dev/heldout_fleet.py; the main
               chart's VAL clips as a clip-set file), the results into this PC's
               point cache under the same identities a local measure gives them;
               `--local`: nceval.chart measure here, in parts that fit the 30-minute
               budget, the last part adding the low rungs.
  time-x86     tools/dev/aws_timing.py on the x86 timing machine (a rented Zen 5,
               aws_timing.TIMING_MACHINE; 2026-11-15, the PC before), each pair on
               its own core, merged into bd-vs-speed-timing-x86.json on the stored
               session's footing: each per-clip time is the median per-round time
               ratio to x264-medium (run back to back with it) times the stored
               x264-medium time (the raw times are kept in raw_ms_per_frame, and the
               same-session ratio to SVT-AV1 M13 in same_session_time_ratio).
  time-jetson  the same streams timed on the Jetson (anchor_chart_timing.py time
               under jetson-timed.sh, which retries until the board is quiet),
               merged into bd-vs-speed-timing.json.
  render       nceval.chart render of both charts, every family on its build, and the
               hardware encoders (NVENC, on this PC's GPU) beside them; then nceval.site
               regenerates the results site's data (web/results/data) and the README's
               headline facts from them.
  check        tools/dev/chart_sanity.py on the refreshed families' codecs: flags a
               preset whose x86/Jetson fps ratio is off its codec's other presets, or
               whose x86 speed contradicts its callgrind instruction count. Warnings,
               not failures: re-time a flagged point (`--from time-x86 --x86-rounds 21`).

With `--pgo` (a release, tools/release/release.py) both timing steps time the
release's profile-guided builds instead of plain release builds: BUILDS_ROOT/
COMMIT/nanocodec-timed-pgo on x86 and nanocodec-pgo on the Jetson, which the
release's `pgo` stage placed there (streams checked identical to the plain
build); the steps fail loudly when they are missing and never build them.
Between releases charts are not refreshed (docs/PROCEDURES.md "Release").

With `--clipset FILE` (the held-out chart, e.g. ~/nanocodec-data/clips/
independent_fresh_2026-11-13.json) the same steps refresh a chart on that clip set,
whose families and builds are recorded apart (docs/results/bd-vs-speed-<set>-builds.json;
`--seed` starts the record with every family on COMMIT's build). Its points are too many
for this PC: measure encodes and scores them on rented machines (tools/dev/
heldout_fleet.py: lanes of ~10 box-minutes, `--cloud-boxes` side by side, each
aws_heldout_chart.py `run` then `score-cloud` on its own clips, a lost session relaunched
on its lane's remaining work; `--cloud-budget` caps each session), every point the render
reads (ours, the references, both zero lines, the metric-tuned side points), and render
only reads the store (it fails listing any point missing); `delete` removes a replaced
build's streams with its points. The speed axis stays the main chart's: timed on
its timing clips, said so under the chart. A build that any record still uses is never
removed.

Independent steps overlap, so the wall time is the measure (or the Jetson's
lock wait), not the sum:

  x86 scoring build -> delete -> measure ---------------------.
                              `-> plan -> time-x86 ---------|-> render
  x86 release build (background) -----'                     |
  Jetson build (background) -> (after plan) time-jetson ----'   render -> check

time-x86 runs beside measure: it times on a rented machine, so this PC's load does
not reach it (on the PC it had to wait for measure). time-jetson times only the
point, against the stored anchor time.

Each step prints its wall time; a failed step names the `--from` to resume at.

Families not named in --families keep their stored results as they are, even
of an older MS-SSIM protocol (eeab2c2b moved real-use MS-SSIM from every 4th
frame to every frame, which barely moves a sequence mean): render and measure
re-measure only points that are not stored. `--rescore` (explicit opt-in, not
for routine refreshes) re-measures every chart point on its recorded build,
references included, in parts.

This replaces the per-refresh scripts of the chart5 work (h264q.sh, run.py,
check_landed.sh, del_av1.py).
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

# The sibling tools in tools/dev, also when this file is loaded by path (eval/tests).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_sanity  # noqa: E402
import stream_check  # noqa: E402
from anchor_chart_timing import HOST_ROUNDS  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
#: Chart builds on both machines (the Jetson's home has the same layout).
BUILDS_ROOT = "nanocodec-data/charts/builds"
#: x86 build trees and target directories: the timed `release` build (fat LTO) and
#: the scoring `gate` build (Cargo.toml: no LTO, incremental; LTO and unit
#: splitting change no stream, so its points are the release build's, and both
#: timing steps check every timed stream against the plan's bytes).
BUILD_TREE = DATA / "worktrees" / "chart-build"
BUILD_TARGET = DATA / "targets" / "chart-build"
SCORING_TREE = DATA / "worktrees" / "chart-build-gate"
SCORING_TARGET = DATA / "targets" / "chart-build-gate"
#: The gate build the landing verifier keeps warm (land.sh --verify): a clean one of COMMIT codes the same streams as our own scoring
#: build, so it is copied instead of building one (7 min in a cold tree).
GATE_DONORS = (DATA / "targets" / "verifier" / "gate" / "nanocodec",)
SCORING_PROFILE = "gate"
#: A build directory's binaries: the scoring build (plan, measure, render) and
#: the timed plain release build (time-x86), what users build. It was a
#: profile-guided build until 2026-11-15 (`nanocodec-timed-pgo`: 1.4-3.2 % less
#: CPU, ~25 min a refresh); the new name keeps a build without it from lending
#: its binaries. The Jetson's `nanocodec` is a plain release build too.
SCORING_BINARY = "nanocodec"
TIMED_BINARY = "nanocodec-timed-release"
#: A release's profile-guided timing builds (--pgo), x86 and Jetson (tools/release/release.py).
PGO_TIMED_BINARY = "nanocodec-timed-pgo"
JETSON_PGO_BINARY = "nanocodec-pgo"
WORK = DATA / "work" / "chart-refresh"
JETSON_WORK = "nanocodec-data/work/chart-refresh"
RESULTS = "docs/results"
BUILDS_FILE = f"{RESULTS}/bd-vs-speed-builds.json"
#: A chart on another clip set (--clipset) keeps its own build record beside the main one.
CLIPSET_BUILDS = RESULTS + "/bd-vs-speed-{stem}-builds.json"
#: The held-out chart's rented sessions: streams (bitstreams, logs) under the data root,
#: scoring sessions side by side and each one's limit (aws_heldout_chart.py).
CLIPSET_STREAMS = DATA / "work" / "heldout-chart" / "streams-{stem}"
#: The remote build box (plan) and the plan's encodes side by side there (the box has
#: 96 vCPU among its job slots).
NC_REMOTE = DATA / "bin" / "nc-remote"
BOX_PLAN_WORKERS = 24
#: The encoders a plan's streams are sized with, synced to the build box (vpxenc is a
#: link into the vp9-av2 anchors; the chart's openh264 harness is the rate-control one).
PLAN_TOOLS = (DATA / "work/ratecontrol/bin", DATA / "anchors/x86_64/bin",
              DATA / "anchors/vp9-av2/bin")
#: The main chart's clips as a clip-set file for the rented machines (under WORK).
MAIN_CLIPSET = "main-val.json"
#: The charts on other clip sets than the main chart's are the results site's chart
#: views (nceval.site.CHART_VIEWS: `clipset_views`, `clipset_chains`), refreshed after
#: the main chart with --heldout, each also rendered against the PSNR-tuned x264 zero
#: line the site and the README quote.
HELDOUT_TUNED_ANCHOR = "x264-medium~psnr"
#: Families whose streams are compared at once (each check encodes on one CPU).
STREAM_CHECK_WORKERS = 3
#: The USD cap of each rented session (heldout_fleet.py: a c7a.24xlarge for ~10 minutes,
#: $0.57/h spot in eu-north-1 on 2026-11-21, so its 45-minute worst case is ~$0.65; the
#: cap leaves room for the throughput ladder's larger fallbacks). Spend is the time used.
CLOUD_SESSION_BUDGET = 3.0
#: Lanes side by side per chart (heldout_fleet.BOXES): with a release's other chains,
#: within eu-north-1's 1,200 spot vCPU.
CLOUD_BOXES = 5
#: The fleet's status lines, echoed from its log into this refresh's own output.
FLEET_MARK = "[fleet]"
PLAN_FILE = f"{RESULTS}/bd-vs-speed-plan.json"
TIMING_FILE = f"{RESULTS}/bd-vs-speed-timing.json"
TIMING_X86_FILE = f"{RESULTS}/bd-vs-speed-timing-x86.json"
#: A chart on another clip set times its own clips: its plan and timing records beside its
#: build record (each chart's speed axis times the encodes its quality axis scores).
CLIPSET_RESULT = RESULTS + "/bd-vs-speed-{stem}-{kind}.json"
#: The hardware encoders' (NVENC) wall-clock fps (tools/dev/nvenc_timing.py); render
#: adds them, measured on this PC's GPU, beside the chart.
TIMING_HARDWARE_FILE = f"{RESULTS}/bd-vs-speed-timing-nvenc.json"
#: The paths whose change can change an encoder build's output.
#: The shipped presets (in the crate that embeds them) and, in commits before
#: 2026-11-15, where they were. A chart point takes its preset's values from
#: the build directory's copy, so a preset change alone needs no new binary.
PRESETS_PATH, OLD_PRESETS_PATH = "crates/nc-encoder/presets", "presets"
BINARY_INPUTS = ("crates", f":(exclude){PRESETS_PATH}", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo")
#: Everything a build directory depends on: the binary's sources and its presets. Not
#: `(*BINARY_INPUTS, PRESETS_PATH, ...)`: git applies an `:(exclude)` pathspec to the
#: whole list, so the presets path stayed excluded and presets-only commits were taken
#: for the same build (2026-11-19, binary_for.sh b51ddbb40 returned b9c7a6382's build).
BUILD_INPUTS = ("crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo",
                OLD_PRESETS_PATH)
#: Written last into a finished build directory (on both machines).
COMPLETE_MARKER = "COMMIT"
#: x86 timing machines side by side (aws_timing.TIMING_MACHINE, one pair at a time on
#: each): a whole-chart re-time takes about 10 min instead of 40.
X86_SESSIONS = 4
STEPS = ("build", "verify", "delete", "plan", "measure", "time-x86", "time-jetson", "render",
         "check")
MEMORY_SCOPE = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=8G",
                "-p", "MemorySwapMax=0"]
ANCHOR = "x264-medium"
#: The stored Jetson anchor timing, in `anchor_chart_timing.py time --anchor-from` form.
STORED_ANCHOR = "t-stored-anchor.json"
#: The full timing plan as the Jetson's timing reads it (nceval.chart_timing.load_plan).
JETSON_PLAN = "plan.json"
#: Times a quiet-board timing slot is retried before giving up (jetson-timed.sh
#: refuses after 5 minutes of load).
JETSON_ATTEMPTS = 6
#: Measure workers: a point holds its slots only while encoding and scoring, so
#: more workers than the slot share fits keep that share busy through each
#: point's unslotted setup and summary (nceval.realuse caps workers at 8).
DEFAULT_WORKERS = 8
#: `--families all`: every shipped preset (nceval.plotted).
ALL_FAMILIES = "all"


class Refresh:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.commit = git("rev-parse", "--short=8", f"{args.commit}^{{commit}}")
        self.families = family_list(args.families)
        self.builds_file = REPO / builds_file(args.clipset)
        self.plan_file, self.timing_file, self.timing_x86_file = (
            REPO / chart_result(args.clipset, kind) for kind in ("plan", "timing", "timing-x86"))
        if self.builds_file.exists():
            self.record = json.loads(self.builds_file.read_text())
        elif args.seed:  # written with the first step that saves the record
            self.record = seed_record(self.commit)
        else:
            raise SystemExit(f"no build record {self.builds_file}: start one with --seed")
        shipped = shipped_families()
        unknown = [f for f in self.families if f not in self.record["families"] and f not in shipped]
        if unknown:
            raise SystemExit(f"not chart families: {', '.join(unknown)}")
        recorded = self.record["families"]
        #: Shipped presets the chart has never measured (nceval.plotted): built and
        #: measured on COMMIT, then recorded on it like any refreshed family.
        self.new = [f for f in self.families if f not in recorded]
        #: Families still on an older build whose sources differ: built and
        #: their old points deleted. One whose sources did not change keeps its
        #: build and cached points (a later step never touches it).
        self.stale = [f for f in self.families if f in self.new or (
            recorded[f] != self.commit and not sources_equal(recorded[f], self.commit))]
        for family in sorted(set(self.families) - set(self.stale)):
            if recorded[family] != self.commit:
                print(f"{family}: sources unchanged since {recorded[family]}: keeps that build")
        #: Families measured and timed on this commit's build.
        self.targets = [f for f in self.families if f in self.stale or recorded[f] == self.commit]
        #: Families only timed on this commit's build (--retime): their streams must be
        #: bit-identical to their recorded build's (checked after the build), so their
        #: quality points stay on that build.
        self.retimed = args.retime.split(",") if args.retime else []
        clash = [f for f in self.retimed if f not in recorded or f in self.families]
        if clash:
            raise SystemExit(f"--retime: not chart families, or also in --families: "
                             f"{', '.join(clash)}")
        #: Every family timed on this commit's build.
        #: Reference encoders to plan and time on COMMIT's session (--time-anchors):
        #: new zero-line neighbours (the fixed-QP chart's slower settings).
        self.anchors = [a for a in getattr(args, "time_anchors", "").split(",") if a]
        self.timed = self.targets + self.retimed + self.anchors
        self.print_lock = threading.Lock()
        #: Logs and render metadata, one directory per chart: the clip-set refreshes of a
        #: release run side by side (their meta.json once crossed over the shared one).
        self.work = WORK / Path(args.clipset).stem if args.clipset else WORK
        self.work.mkdir(parents=True, exist_ok=True)

    # --- helpers ---------------------------------------------------------------------

    def say(self, line: str) -> None:
        with self.print_lock:
            print(line, flush=True)

    def run(self, cmd: list[str], log: str, cwd: Path = REPO, scope: bool = False,
            env: dict[str, str] | None = None, echo: str | None = None) -> None:
        """`cmd` with its output in `log`; lines containing `echo` also go to this
        refresh's own output (a fleet's status, so a stall shows in the top-level log)."""
        cmd = (MEMORY_SCOPE + cmd) if scope else cmd
        self.say(f"  $ {shlex.join(cmd)}  [{self.work / 'logs' / log}]")
        if self.args.dry_run:
            return
        (self.work / "logs").mkdir(exist_ok=True)
        with (self.work / "logs" / log).open("a") as out:
            process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       text=True, env={**os.environ, **(env or {})})
            for line in process.stdout:
                out.write(line)
                out.flush()
                if echo and echo in line:
                    self.say(f"  {line.rstrip()}")
            status = process.wait()
        if status:
            raise SystemExit(f"failed ({status}); see {self.work / 'logs' / log}")

    def build_dir(self, build: str) -> Path:
        return Path.home() / BUILDS_ROOT / build

    def builds(self) -> tuple[dict[str, Path], dict[str, Path]]:
        """Every family's (binary, presets directory), as recorded."""
        dirs = {f: self.build_dir(b) for f, b in self.record["families"].items()}
        return ({f: d / SCORING_BINARY for f, d in dirs.items()},
                {f: d / "presets" for f, d in dirs.items()})

    def chart_args(self) -> list[str]:
        """nceval.chart's build arguments: every family on its recorded build."""
        base = self.build_dir(self.record["base"])
        args = ["--binary", str(base / SCORING_BINARY), "--presets-dir", str(base / "presets")]
        binaries, presets = self.builds()
        for family in sorted(binaries):
            args += ["--binary-for", f"{family}={binaries[family]}",
                     "--presets-for", f"{family}={presets[family]}"]
        return args

    def save_record(self) -> None:
        if not self.args.dry_run:
            self.builds_file.write_text(json.dumps(self.record, indent=1) + "\n")

    # --- steps -----------------------------------------------------------------------

    def build(self) -> None:
        """COMMIT's x86 scoring build (the release builds run alongside: `build_timed`,
        `build_jetson`)."""
        if not self.needs_build():
            return
        complete = (self.build_dir(self.commit) / COMPLETE_MARKER).is_file()
        if complete or self.commit in self.record["builds"]:
            self.say(f"  build {self.commit} exists: kept")
        elif (donor := self.binary_donor()) is not None:
            self.say(f"  build {self.commit}: binary sources equal build {donor}'s: its binaries "
                     f"with {self.commit}'s presets")
            if not self.args.dry_run:
                place(self.build_dir(donor) / SCORING_BINARY,
                      self.build_dir(self.commit) / SCORING_BINARY)
        elif (warm := self.gate_donor()) is not None:
            self.say(f"  build {self.commit}: the landing scripts' gate build {warm}")
            if not self.args.dry_run:  # the copy is checked: a landing may rebuild it meanwhile
                place(warm, self.build_dir(self.commit) / SCORING_BINARY)
                verify_build(self.build_dir(self.commit) / SCORING_BINARY, self.commit)
        else:
            self.cargo_build(SCORING_TREE, SCORING_TARGET, SCORING_PROFILE,
                             self.build_dir(self.commit) / SCORING_BINARY)
        # a complete build is never re-extracted: chains of one release share it and
        # read its presets meanwhile (2026-11-19: three chains extracting at once failed
        # one with tar's "File exists")
        if not self.args.dry_run and not complete:
            self.install_presets()
        self.record["builds"].setdefault(self.commit, self.commit)
        self.save_record()

    def build_timed(self) -> None:
        """COMMIT's x86 release build (time-x86), unless it exists or a donor's applies;
        with --pgo the release's PGO build, which must exist."""
        if not self.needs_build():
            return
        if self.args.pgo:
            self.require_pgo_builds()
            return
        timed = self.build_dir(self.commit) / TIMED_BINARY
        if timed.is_file():
            self.say(f"  release build {self.commit} exists: kept")
            return
        donor = self.binary_donor()
        if donor is None:
            self.cargo_build(BUILD_TREE, BUILD_TARGET, "release", timed)
        elif not self.args.dry_run:
            place(self.build_dir(donor) / TIMED_BINARY, timed)

    def cargo_build(self, tree: Path, target: Path, profile: str, destination: Path) -> None:
        """COMMIT checked out in `tree` (its own worktree: unchanged files keep their
        times, so the build is incremental), built with `profile`, its binary placed
        at `destination`."""
        log = f"build-{profile}.log"
        env = {"CARGO_TARGET_DIR": str(target),
               "PATH": f"{Path.home() / '.cargo/bin'}:{os.environ['PATH']}"}
        with build_tree_lock(tree):
            if not tree.exists():
                self.run(["git", "worktree", "add", "--detach", str(tree), self.commit], log)
            # forced: the tree is this tool's own, and a build may leave Cargo.lock
            # rewritten (an older commit's stale lockfile), which blocks a plain checkout
            self.run(["git", "checkout", "-q", "--force", "--detach", self.commit], log, tree)
            built = target / profile / "nanocodec"
            self.run(["cargo", "build", "--profile", profile, "-j8", "--bin", "nanocodec"], log,
                     tree, env=env)
            if not self.args.dry_run:  # placed while locked: another refresh may rebuild next
                verify_build(built, self.commit)
                place(built, destination)

    def needs_build(self) -> bool:
        return (bool(self.stale or self.retimed)
                or self.commit in self.record["families"].values())

    def install_presets(self) -> None:
        """COMMIT's presets into its build directory, then the marker. One refresh at a
        time (the others find the marker and keep its presets); extracted beside and
        renamed into place, so no reader sees a half-written directory."""
        target = self.build_dir(self.commit)
        target.mkdir(parents=True, exist_ok=True)
        with build_tree_lock(target):
            if (target / COMPLETE_MARKER).is_file():
                return
            presets, staging = target / "presets", target / "presets.partial"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            moved = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                                    f"{self.commit}:{PRESETS_PATH}"], capture_output=True)
            source = PRESETS_PATH if moved.returncode == 0 else OLD_PRESETS_PATH
            archive = subprocess.run(["git", "-C", str(REPO), "archive", self.commit, source],
                                     check=True, capture_output=True).stdout
            subprocess.run(["tar", "-x", f"--strip-components={source.count('/') + 1}", "-C",
                            str(staging)], input=archive, check=True)
            shutil.rmtree(presets, ignore_errors=True)
            staging.rename(presets)
            (target / COMPLETE_MARKER).write_text(self.commit + "\n")

    def gate_donor(self) -> Path | None:
        """A landing script's gate build that is exactly COMMIT (clean), if any."""
        if SCORING_PROFILE != "gate":
            return None
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.buildinfo import info_of, stamp_problem
        full = git("rev-parse", f"{self.commit}^{{commit}}")
        return next((binary for binary in GATE_DONORS
                     if binary.is_file() and stamp_problem(info_of(binary), full) is None), None)

    def binary_donor(self) -> str | None:
        """A recorded build (complete on the PC) whose binaries COMMIT would rebuild unchanged."""
        for build in sorted(self.record["builds"]):
            directory = self.build_dir(build)
            if (build != self.commit and (directory / SCORING_BINARY).is_file()
                    and (directory / TIMED_BINARY).is_file()
                    and sources_equal(build, self.commit, BINARY_INPUTS)):
                return build
        return None

    def build_jetson(self) -> None:
        """COMMIT's Jetson build, unless it exists; a binary donor's copy when one applies.
        With --pgo the Jetson times the release's PGO build (checked in build_timed)."""
        if not self.needs_build() or self.args.pgo:
            return
        build = f"{BUILDS_ROOT}/{self.commit}"
        if not self.args.dry_run and self.jetson_build_done():
            self.say(f"  Jetson build {self.commit} exists: kept")
            return
        donor = self.binary_donor()
        if donor is not None:
            self.say(f"  $ ssh {jetson()} cp {BUILDS_ROOT}/{donor}/nanocodec {build}/")
            if not self.args.dry_run:
                ssh(f"mkdir -p ~/{build} && cp ~/{BUILDS_ROOT}/{donor}/nanocodec ~/{build}/ && "
                    f"echo {self.commit} > ~/{build}/{COMPLETE_MARKER}")
        else:
            self.run(["bash", "-c", self.jetson_build_command()], "jetson-build.log")
        if not self.args.dry_run and not self.jetson_build_done():
            raise SystemExit(f"the Jetson build of {self.commit} is incomplete")

    def jetson_build_command(self) -> str:
        """The board's build of COMMIT (nceval.jetson.build_command, shared with the tuner)."""
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.jetson import build_command
        return build_command(self.commit)

    def timed_binary(self) -> Path:
        """The x86 binary time-x86 times: the plain release build, or the release's PGO one."""
        return self.build_dir(self.commit) / (PGO_TIMED_BINARY if self.args.pgo else TIMED_BINARY)

    def jetson_binary(self) -> str:
        """The Jetson binary time-jetson times (home-relative), as timed_binary."""
        name = JETSON_PGO_BINARY if self.args.pgo else SCORING_BINARY
        return f"{BUILDS_ROOT}/{self.commit}/{name}"

    def require_pgo_builds(self) -> None:
        """--pgo: both PGO builds of COMMIT are in place (release.py `pgo` puts them there)."""
        if self.args.dry_run:
            return
        missing = [] if self.timed_binary().is_file() else [str(self.timed_binary())]
        if ssh(f"test -x ~/{self.jetson_binary()} && echo yes || true").strip() != "yes":
            missing.append(f"{jetson()}:~/{self.jetson_binary()}")
        if missing:
            raise SystemExit(f"--pgo: no PGO build {' or '.join(missing)}: run "
                             f"tools/release/release.py's pgo stage first")

    def jetson_build_done(self) -> bool:
        marker = f"~/{BUILDS_ROOT}/{self.commit}/{COMPLETE_MARKER}"
        return ssh(f"cat {marker} 2>/dev/null || true").strip() == self.commit

    def delete(self) -> None:
        """Delete the replaced builds' cached chart points (and, on a clip set, their
        rented encodes' streams); record the new build."""
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.chart import chart_clips, low_ladder
        from nceval.config import load_paths
        from nceval.gatepolicy import EVALUATION_GATE
        from nceval.realuse import DEFAULT_FRAMES, Runner, own_family

        runner = Runner(load_paths(), EVALUATION_GATE, 1, over_budget=True)
        # a clip set's streams, encoded on rented machines, are kept per build too: the
        # replaced build's go with its points (2026-11-20: 41 GiB of streams of builds no
        # chart showed any more pushed work/ to its budget)
        streams = (Runner(load_paths(), EVALUATION_GATE, 1, over_budget=True,
                          prebuilt=self.streams()) if self.on_boxes() else None)
        clips = chart_clips(self.args.clipset)
        for family in self.stale:
            if family in self.new:
                print(f"  {family}: new chart family: no old points")
                self.record["families"][family] = self.commit
                continue
            old = self.build_dir(self.record["families"][family])
            if not (old / SCORING_BINARY).is_file():
                print(f"  {family}: old build {old} is gone; nothing to delete")
            else:
                # the points were measured under the name as the record has it, in its
                # build's own preset names (a rung renamed since, e.g. 0.5.0's `slow`,
                # is today's `slower`; that build has no rung today's `slow` names)
                stale = [own_family(family, old / SCORING_BINARY, old / "presets",
                                    recorded=True)]
                jobs = (runner.points(stale, clips, DEFAULT_FRAMES, None)
                        + runner.points(stale, clips, DEFAULT_FRAMES, None, ladder=low_ladder))
                removed = 0
                for bench, point in jobs:
                    path = bench.result_path(point)
                    for entry in (path, path.with_name(path.name + ".lock")):
                        if entry.exists():
                            removed += 1
                            if not self.args.dry_run:
                                entry.unlink()
                print(f"  {family}: {'would delete' if self.args.dry_run else 'deleted'} "
                      f"{removed} cached files of {len(jobs)} old points")
                if streams is not None:
                    self.delete_streams(streams, stale, clips)
            self.record["families"][family] = self.commit
        self.drop_unused_builds()
        self.save_record()

    def delete_streams(self, runner, families: list, clips: list) -> None:
        """The rented encodes of `families` (a replaced build) under the clip set's
        streams: each point's files, every suffix (stream, rate log, times, .done)."""
        from nceval.chart import low_ladder
        from nceval.realuse import DEFAULT_FRAMES
        jobs = (runner.points(families, clips, DEFAULT_FRAMES, None)
                + runner.points(families, clips, DEFAULT_FRAMES, None, ladder=low_ladder))
        removed = 0
        for bench, point in jobs:
            base = bench.prebuilt_base(point)
            if not base.parent.is_dir():
                continue
            for entry in base.parent.glob(f"{base.name}.*"):
                removed += entry.stat().st_size
                if not self.args.dry_run:
                    entry.unlink()
        print(f"  {families[0].name}: {'would delete' if self.args.dry_run else 'deleted'} "
              f"{removed / 1e9:.2f} GB of its old streams")

    def verify(self) -> None:
        """--retime's guard: each retimed family's streams at COMMIT equal its recorded
        build's (stream_check.py: the timing plan's fixed-QP streams and a CBR stream per
        timing clip); then its timing build is recorded beside its quality build.

        First, a refreshed family whose streams did not change (its sources did, its
        output did not: a speed change, a lever that lands off) becomes a retimed one:
        its quality points stay valid on its recorded build and are not measured again.
        A refresh after a one-codec change then measures that codec's families only
        (2026-11-15: re-measuring all 18 took 49 min)."""
        self.retime_unchanged()
        if not self.retimed:
            return
        new = self.build_dir(self.commit)
        changed = []
        for family in self.retimed:
            old = self.build_dir(self.record["families"][family])
            self.say(f"  {family}: streams of {old.name} vs {self.commit}")
            if not self.args.dry_run and not stream_check.identical(
                    family, (old / SCORING_BINARY, old / "presets"),
                    (new / SCORING_BINARY, new / "presets"), self.plan_file):
                changed.append(family)
        if changed:
            raise SystemExit(f"streams changed, give them in --families: {','.join(changed)}")
        # Recorded once they are timed (`render`): a refresh stopped before its
        # timing must not read as already timed on COMMIT when resumed.

    def retime_unchanged(self) -> None:
        """Move each stale family whose streams at COMMIT equal its recorded build's
        from the measured families to the retimed ones."""
        new = self.build_dir(self.commit)

        def unchanged(family: str) -> bool:
            old = self.build_dir(self.record["families"][family])
            return (not self.args.dry_run and (old / SCORING_BINARY).is_file()
                    and stream_check.identical(family, (old / SCORING_BINARY, old / "presets"),
                                               (new / SCORING_BINARY, new / "presets"),
                                               self.plan_file))
        # Side by side: one at a time the checks of 18 families took 10 min.
        recorded = [f for f in self.stale if f not in self.new]  # a new family has no old build
        with ThreadPoolExecutor(STREAM_CHECK_WORKERS) as pool:
            same = dict(zip(recorded, pool.map(unchanged, recorded)))
        timed_on = self.record.get("timed_on", {})
        for family in [f for f, equal in same.items() if equal]:
            self.stale.remove(family)
            self.targets.remove(family)
            timed_build = timed_on.get(family, self.record["families"][family])
            if sources_equal(timed_build, self.commit, speed_inputs(family)):
                self.say(f"  {family}: streams and its code unchanged since {timed_build}: "
                         f"kept as it is")
                continue
            self.say(f"  {family}: streams unchanged since {self.record['families'][family]}: "
                     f"its quality points are kept, it is only timed")
            self.retimed.append(family)
        self.timed = self.targets + self.retimed + self.anchors

    def drop_unused_builds(self) -> None:
        for family in self.targets:  # measured and timed on one build again
            self.record.get("timed_on", {}).pop(family, None)
        # COMMIT stays while this refresh times on it: a retime-only refresh records its
        # families on it only at render (2026-11-16: it was removed before the timing)
        used = ({self.commit} if self.timed else set()) | set(self.record["families"].values()) \
            | {self.record["base"]} | set(self.record.get("timed_on", {}).values()) \
            | other_records_builds(self.builds_file)
        for build in sorted(set(self.record["builds"]) - used):
            print(f"  build {build} is no longer used: removed on both machines")
            del self.record["builds"][build]
            if not self.args.dry_run:
                shutil.rmtree(self.build_dir(build), ignore_errors=True)
                ssh(f"rm -rf ~/{BUILDS_ROOT}/{build}")

    def plan(self) -> None:
        if not self.timed:
            return
        build = self.build_dir(self.commit)
        clipset = ["--clipset", str(Path(self.args.clipset).resolve())] if self.args.clipset else []
        if self.args.local:
            self.run(["python3", "../tools/dev/anchor_chart_timing.py", "plan",
                      "--binary", str(build / SCORING_BINARY), "--presets-dir",
                      str(build / "presets"), "--plan", str(self.plan_file), *clipset,
                      *self.encode_set(), "--points", ",".join(self.timed)],
                     "plan.log", REPO / "eval", scope=True, env={"PYTHONPATH": "."})
            return
        # On the build box (nc-remote: this tree synced, the build and the timing clips
        # up, the plan back): our streams are bit-exact across machines, and each entry's
        # key hashes the tool and the clip file.
        self.run(self.box_plan_command(build), "plan.log")

    def low_clips(self) -> str:
        """The chart's low-rung clips: --low-clips, else the stored plan's."""
        if self.args.low_clips is not None:
            return self.args.low_clips
        stored = json.loads(self.plan_file.read_text()) if self.plan_file.is_file() else {}
        return ",".join(stored.get("low_clips", []))

    def encode_set(self) -> list[str]:
        """anchor_chart_timing plan's options for every quality encode of the chart."""
        low = self.low_clips()
        return ["--all-encodes", *(["--low-clips", low] if low else [])]

    def box_plan_command(self, build: Path) -> list[str]:
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.chart import chart_clips
        from nceval.chart_timing import PLAN_STORE
        from nceval.config import load_paths
        timed = chart_clips(self.args.clipset)
        clip_file = (Path(self.args.clipset).resolve() if self.args.clipset
                     else load_paths().clips / "main.json")
        on_box = f"~/{BUILDS_ROOT}/{build.name}"
        plan_path = self.plan_file.relative_to(REPO)
        plan = shlex.join(["python3", "../tools/dev/anchor_chart_timing.py", "plan",
                           "--plan", f"../{plan_path}", "--points", ",".join(self.timed),
                           *(["--clipset", str(clip_file)] if self.args.clipset else []),
                           *self.encode_set(), "--workers", str(BOX_PLAN_WORKERS)])
        # the full plans go up and come back (a refresh of some families keeps the others'
        # entries); the repository keeps the plan's record
        PLAN_STORE.mkdir(parents=True, exist_ok=True)
        store = f"~/{PLAN_STORE.relative_to(Path.home())}"
        return [str(NC_REMOTE), "run", "--worktree", str(REPO), "--name",
                f"chart-plan-{self.commit}", "--sync", str(build), "--data", store,
                "--fetch", store,
                *(a for tools in PLAN_TOOLS for a in ("--sync", str(tools))),
                *(a for clip in timed for a in ("--clip", str(clip.yuv))),
                "--clip", str(clip_file), "--fetch", str(plan_path), "--",
                f"cd eval && PYTHONPATH=. {plan} --binary {on_box}/{SCORING_BINARY} "
                f"--presets-dir {on_box}/presets --store {store}"]

    def measured(self) -> list[str] | None:
        """The families measure takes: every chart point (None) on --rescore."""
        return None if self.args.rescore else self.targets

    def measure(self) -> None:
        if not (self.targets or self.args.rescore):
            return
        if self.on_boxes():
            self.cloud_measure()
            return
        # Sized to the budget (measure_parts), and at least --part-jobs parts so that many
        # run side by side (each measure process holds at most its own slot share).
        parts = max(self.args.part_jobs,
                    1 if self.args.dry_run else self.measure_parts())  # needs the build

        def measure_part(index: int) -> None:
            part = ["--part", f"{index}/{parts}"] if parts > 1 else []
            # measure's own 30-minute check prices at the slots free at that instant, so it
            # would refuse a part whenever the machine got busier after sizing; it guards
            # hand-run jobs.
            self.run(["python3", "-m", "nceval.chart", "measure", *self.chart_args(),
                      *([] if self.args.rescore else ["--points", ",".join(self.targets)]),
                      "--workers", str(self.args.workers), "--over-budget", *part,
                      *(["--rescore"] if self.args.rescore else [])],
                     f"measure-{index}.log" if parts > 1 else "measure.log", REPO / "eval",
                     scope=True)

        with ThreadPoolExecutor(self.args.part_jobs) as pool:
            for job in [pool.submit(measure_part, i) for i in range(1, parts + 1)]:
                job.result()

    def cloud_measure(self) -> None:
        """Every point the render reads, encoded and scored on rented machines by
        tools/dev/heldout_fleet.py (small sessions side by side, a lost one relaunched
        on its own clips, results back in this PC's point cache as they finish): every
        family of the chart's record on its build (`--record`, so the low rungs are
        judged on the whole chart, as the render judges them; unchanged families' points
        are cached and cost nothing), the references and both zero lines, and the
        metric-tuned side points (cached from the last release where their tools did not
        change: 0.6.0's render coded 15,878 of them on this PC)."""
        if self.args.rescore:
            raise SystemExit("--rescore on --clipset: give the families instead")
        families = [*self.record["families"], *main_references(), *self.side_families()]
        common = ["--clipset", str(self.cloud_clipset()), "--streams", str(self.streams()),
                  "--record", str(self.builds_file),
                  "--families", ",".join(dict.fromkeys(families))]
        self.run(["python3", "tools/dev/heldout_fleet.py", "measure",
                  "--state", str(self.work / "logs" / "fleet.json"),
                  "--boxes", str(self.args.cloud_boxes),
                  "--budget", f"{self.args.cloud_budget:.2f}", "--purpose", "chart-refresh",
                  *(["--dry-run"] if self.args.dry_run else []), "--", *common],
                 "cloud-fleet.log", echo=FLEET_MARK)

    def side_families(self) -> list[str]:
        """The metric-tuned references the render draws as side points: the clip-set
        view's `tuned` list, else every one (nceval.chart.TUNED)."""
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.chart import TUNED
        view = clipset_view(self.args.clipset) if self.args.clipset else None
        return view.tuned.split(",") if view and view.tuned else [n for n, _ in TUNED]

    def on_boxes(self) -> bool:
        """Whether measure encodes and scores on rented machines: a clip set always, the
        main chart unless --local (the PC carries no encode compute, owner 2026-11-21)."""
        return bool(self.args.clipset) or not self.args.local

    def cloud_clipset(self) -> Path:
        """The clip-set file the rented machines measure: --clipset, or the main chart's
        VAL clips written from main.json (realuse.write_natural_val_clipset: the same
        clips, so the same point identities as a local measure)."""
        if self.args.clipset:
            return Path(self.args.clipset).resolve()
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.config import load_paths
        from nceval.realuse import write_natural_val_clipset
        return write_natural_val_clipset(load_paths(), WORK / MAIN_CLIPSET)

    def streams(self) -> Path:
        stem = Path(self.args.clipset).stem if self.args.clipset else Path(MAIN_CLIPSET).stem
        return Path(str(CLIPSET_STREAMS).format(stem=stem))

    def measure_parts(self) -> int:
        """How many class-balanced clip parts keep each measure under the budget.

        Priced at this process's slot share (what one measure process may hold), not at
        the slots free at this instant, so the part count does not swing with the
        machine's momentary load."""
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.budget import MAX_MINUTES, estimate_minutes
        from nceval.slots import owner_share
        from nceval.chart import chart_clips, chart_families
        from nceval.config import load_paths
        from nceval.gatepolicy import EVALUATION_GATE
        from nceval.realuse import DEFAULT_FRAMES, Runner
        from nceval.stages import PART_FILL

        binaries, presets = self.builds()
        base = self.build_dir(self.record["base"])
        families = chart_families(base / SCORING_BINARY, base / "presets", self.measured(),
                                  binaries, presets)
        runner = Runner(load_paths(), EVALUATION_GATE, self.args.workers, over_budget=True)
        minutes = estimate_minutes(runner.price(runner.points(families, chart_clips(None),
                                                              DEFAULT_FRAMES, None)),
                                   slots=owner_share())
        return max(1, math.ceil(minutes / (MAX_MINUTES * PART_FILL)))

    def time_x86(self) -> None:
        if not self.timed:
            return
        if not self.args.dry_run and not self.timed_binary().is_file():
            raise SystemExit(f"no timing build {self.timed_binary()}: resume with --from build")
        # every timed point is on COMMIT's build; one repeat (the 2026-11-15 probe's
        # repeats agreed within 0.4%), the points split over X86_SESSIONS machines side
        # by side, each timing one pair at a time (machines agreed within 1.3%)
        parts = [self.timed[i::X86_SESSIONS] for i in range(X86_SESSIONS)]
        outs = [WORK / f"x86-{self.commit}-{i + 1}" for i, part in enumerate(parts) if part]
        for out in outs:
            if out.exists() and not self.args.dry_run:
                shutil.rmtree(out)  # a re-run times afresh
        with ThreadPoolExecutor(len(outs)) as pool:
            for job in [pool.submit(self.run, [
                    "python3", "tools/dev/aws_timing.py", "--points", ",".join(part),
                    "--binary", str(self.timed_binary()), "--repeats", "1",
                    "--plan", str(self.plan_file),
                    "--rounds", str(self.args.x86_rounds),
                    "--purpose", "chart-refresh-x86-timing", "--out", str(out)],
                    f"time-x86-{out.name.rsplit('-', 1)[1]}.log")
                    for part, out in zip([p for p in parts if p], outs)]:
                job.result()
        if not self.args.dry_run:
            merge_timing(self.timing_x86_file,
                         sorted(r for out in outs for r in (out / "repeat-1").glob("t-*.json")),
                         rescale=True, build=self.commit)
            for out in outs:
                shutil.rmtree(out)

    def time_jetson(self) -> None:
        if not self.timed:
            return
        if not self.args.dry_run and not self.args.pgo and not self.jetson_build_done():
            raise SystemExit(f"no Jetson build of {self.commit}: resume with --from build")
        local = WORK / f"jetson-{self.commit}"
        self.jetson_records(local)
        if not self.args.dry_run:
            merge_timing(self.timing_file, sorted(local.glob("t-*.json")), rescale=False,
                         build=self.commit)
            shutil.rmtree(local)

    def jetson_clips(self) -> None:
        """The plan's timing clips on the Jetson (anchor_chart_timing.JETSON_CLIP_DIR), each
        the plan's frames, copied as aws_timing stages them where the board lacks them."""
        sys.path.insert(0, str(REPO / "eval"))
        from anchor_chart_timing import JETSON_CLIP_DIR, input_frames, plan_clips, plan_input
        from aws_timing import clip_archive
        from nceval.chart import chart_clips
        plan = json.loads(self.plan_file.read_text())  # the record: its header suffices
        clips = {c.name: c for c in chart_clips(plan["clipset"])}
        remote = JETSON_CLIP_DIR.relative_to(Path.home())
        for name in sorted({plan_input(plan, key) for key in plan_clips(plan)}):
            target = f"{remote}/{name}.yuv"
            if subprocess.run(["ssh", "-o", "BatchMode=yes", jetson(), f"test -f {target}"]
                              ).returncode == 0:
                continue
            archive = clip_archive(clips[name], input_frames(plan, name))
            scp([str(archive)], f"{jetson()}:{remote}/")
            ssh(f"zstd -d -q --rm {remote}/{archive.name} -o {target}")

    def jetson_records(self, local: Path) -> None:
        """Time every timed family on the Jetson; its records (`t-FAMILY.json`) into `local`
        (a release's pipe-clean keeps them there instead of merging them)."""
        binary = f"~/{self.jetson_binary()}"
        if not self.args.dry_run:
            # The timing core's clock is locked and x264 medium repeats within ~0.3 ms/frame
            # across lock windows (anchor_chart_timing.py time), so the stored anchor time
            # stands in for a paired one: each lock window times only the point (about a
            # quarter of the window the pair took).
            anchor = WORK / STORED_ANCHOR
            stored = json.loads(self.timing_file.read_text())["points"][ANCHOR]
            anchor.write_text(json.dumps({"clips": {
                clip: {"ms_per_frame": {"anchor": ms}}
                for clip, ms in stored["ms_per_frame"].items()}}) + "\n")
            ssh(f"rm -rf {JETSON_WORK}/timing && mkdir -p {JETSON_WORK}/timing")
            self.jetson_clips()
            from nceval.chart_timing import load_plan
            full = WORK / JETSON_PLAN
            full.write_text(json.dumps(load_plan(self.plan_file)) + "\n")
            scp([str(full), str(REPO / "tools/dev/anchor_chart_timing.py"),
                 str(anchor)], f"{jetson()}:{JETSON_WORK}/timing/")
            anchor.unlink()
            full.unlink()
        for family in self.timed:
            command = (f"cd {JETSON_WORK}/timing && for attempt in $(seq {JETSON_ATTEMPTS}); "
                       f"do ~/nanocodec-data/jetson-timed.sh python3 anchor_chart_timing.py time "
                       f"--plan {JETSON_PLAN} --point {shlex.quote(family)} "
                       f"--nanocodec {binary} --out t-{shlex.quote(family)}.json "
                       f"--anchor-from {STORED_ANCHOR} && exit 0; sleep 60; done; exit 1")
            self.run(["ssh", "-o", "BatchMode=yes", jetson(), command], "time-jetson.log")
        if not self.args.dry_run:
            local.mkdir(parents=True, exist_ok=True)
            scp([f"{jetson()}:{JETSON_WORK}/timing/t-{f}.json" for f in self.timed], str(local))
            ssh(f"rm -rf {JETSON_WORK}/timing")

    def render(self) -> None:
        if self.retimed and not self.args.metrics_only:
            self.record.setdefault("timed_on", {}).update({f: self.commit for f in self.retimed})
            self.save_record()
        meta = self.work / "meta.json"
        if not self.args.dry_run:
            meta.write_text(json.dumps(self.meta(), indent=1) + "\n")
        clipset = (["--clipset", str(Path(self.args.clipset).resolve()),
                    "--prebuilt", str(self.streams())] if self.args.clipset else [])
        # The main chart is rendered against both zero lines the site offers (the
        # held-out ones get the tuned line from refresh_heldout): without it the
        # site's Development + tuned-x264 view fell back to an older Defaults-only run.
        anchors = ([self.args.anchor] if getattr(self.args, "anchor", None) else
                   [None] if self.args.clipset else [None, HELDOUT_TUNED_ANCHOR])
        view = clipset_view(self.args.clipset) if self.args.clipset else None
        for zero in anchors:
            anchor = ["--anchor", zero] if zero else []
            if view:
                title = f"Bitrate vs encode speed, {view.heading} ({Path(self.args.clipset).stem})"
                anchor += ["--title", title + (f", zero line {zero}" if zero else "")]
                anchor += ["--tuned", view.tuned] if view.tuned else []
            self.run(["python3", "-m", "nceval.chart", "render", *self.chart_args(), *clipset,
                      *anchor, "--timing", str(self.timing_file),
                      "--timing-x86", str(self.timing_x86_file),
                      # NVENC runs only on this PC's GPU: never on a rented clip set's streams
                      *([] if self.args.clipset else
                        ["--timing-hardware", str(REPO / TIMING_HARDWARE_FILE)]),
                      "--meta", str(meta), "--out", str(REPO / RESULTS),
                      # a local chart's render measures what its measure left (the PC
                      # carries no compute otherwise: render reads the store only)
                      *([] if self.on_boxes() else ["--measure-missing"]),
                      "--workers", str(self.args.workers)], "render.log", REPO / "eval",
                     scope=True)
        # The results site's data follows the new charts.
        self.run(["python3", "-m", "nceval.site"], "site.log", REPO / "eval")

    def check(self) -> None:
        """chart_sanity.py on the refreshed families' codecs: flags printed, never fatal."""
        if not self.timed or self.args.dry_run:
            return
        codecs = {chart_sanity.codec_of(f) for f in self.timed if f.startswith("nanocodec-")}
        try:
            flags = chart_sanity.check(codecs, instructions=not self.args.metrics_only)
        except (Exception, SystemExit) as error:  # advisory: it never stops a refresh
            flags = [f"the sanity check itself failed: {str(error)[:300]}"]
        for flag in flags:
            self.say(f"  FLAG {flag}")
        if flags:
            self.say("  check the flagged points; re-time with --from time-x86 --x86-rounds N")

    # --- schedule --------------------------------------------------------------------

    def execute(self, start: str) -> None:
        """Every step from `start` on, each as soon as the steps it needs are done."""
        wanted = set(STEPS[STEPS.index(start):])
        if self.args.metrics_only:
            wanted -= {"plan", "time-x86", "time-jetson"}
        if "verify" not in wanted and self.stale:
            # A resume past verify must still know which stale families' streams did
            # not change, or it measures them again on the new build (2026-11-15: a
            # held-out resume rented 13156 encodes and two scoring sessions for them).
            self.retime_unchanged()
        with ThreadPoolExecutor(len(STEPS)) as pool:
            def launch(step: str, *after: Future, part_of: str = "") -> Future:
                """`step` after `after`; skipped when it (or the step it is part of) is
                before `start`."""
                skipped = self.args.metrics_only and step in ("build-timed", "build-jetson")
                return pool.submit(self.step, None if skipped or (part_of or step) not in wanted
                                   else step, after)
            jetson_build = launch("build-jetson", part_of="build")
            timed_build = launch("build-timed", part_of="build")
            x86_build = launch("build")
            verify = launch("verify", x86_build)
            delete = launch("delete", verify)
            measure = launch("measure", delete)
            plan = launch("plan", delete)
            time_x86 = launch("time-x86", plan, timed_build)
            time_jetson = launch("time-jetson", plan, jetson_build)
            render = launch("render", measure, time_x86, time_jetson)
            check = launch("check", render)
            steps = {"build": x86_build, "verify": verify, "build-jetson": jetson_build,
                     "build-timed": timed_build, "delete": delete,
                     "measure": measure, "plan": plan, "time-x86": time_x86,
                     "time-jetson": time_jetson, "render": render, "check": check}
            failed = [name for name, job in steps.items() if job.exception() is not None]
        if failed:
            first = min(STEPS.index("build" if f.startswith("build") else f) for f in failed)
            for name in failed:
                self.say(f"failed: {name}: {steps[name].exception()}")
            raise SystemExit(f"resume with --from {STEPS[first]}")

    def step(self, name: str | None, after: tuple[Future, ...]) -> None:
        for job in after:
            if job.exception() is not None:
                raise SystemExit("an earlier step failed")
        if not name:
            return
        self.say(f"== {name}")
        began = time.monotonic()
        getattr(self, name.replace("-", "_"))()
        minutes, seconds = divmod(round(time.monotonic() - began), 60)
        self.say(f"== {name} done in {minutes}:{seconds:02d}")

    def meta(self) -> dict:
        """The methodology note's build facts, from the build record: the base
        build, and the families on other builds by chart label."""
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.chart import OURS

        labels = dict(OURS)
        base = self.record["base"]
        others: dict[str, list[str]] = {}
        for family, build in self.record["families"].items():
            if build != base:
                others.setdefault(build, []).append(labels[family])
        describe = self.record["builds"]
        line = describe[base] + (" (" + "; ".join(
            f"{', '.join(names)}: {describe[build]}" for build, names in sorted(others.items()))
            + ")" if others else "")
        timed: dict[str, list[str]] = {}
        for family, build in self.record.get("timed_on", {}).items():
            timed.setdefault(build, []).append(labels[family])
        line += "".join(f"; {', '.join(names)} timed on {describe[build]} (same streams)"
                        for build, names in sorted(timed.items()))
        return {**self.record["tools"], "commit": line, "build": line,
                "x86_timing_note": self.record.get("x86_timing_note", "")}


def main_references() -> list[str]:
    """The main chart's reference encoders and both zero lines (nceval.chart)."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.chart import ANCHOR, REFERENCES
    return list(dict.fromkeys([ANCHOR, HELDOUT_TUNED_ANCHOR, *(n for n, _ in REFERENCES)]))


def clipset_views() -> list:
    """The results site's chart views on clip sets of their own (nceval.site.CHART_VIEWS),
    one per clip set: every chart a release refreshes besides the main one."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.site import CHART_VIEWS
    return list({view.clips: view for view in CHART_VIEWS if view.clips}.values())


def clipset_path(view) -> Path:
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.config import load_paths
    return load_paths().clips / view.clips


def clipset_view(clipset: str):
    """The site's chart view of the clip set file `clipset`."""
    view = next((v for v in clipset_views() if v.clips == Path(clipset).name), None)
    if view is None:
        raise SystemExit(f"{clipset} is no chart view's clip set (nceval.site.CHART_VIEWS)")
    return view


def clipset_chains(views: list | None = None) -> list[list[Path]]:
    """The clip-set charts' files in groups that share clips, the largest first in each:
    a group is refreshed in that order, so a smaller set finds its shared clips' points
    already cached (one point cache: cbr.Bench.identity has no clip-set part). The
    all-held-out set holds the 138 fresh ones: run side by side, both scored those
    clips (0.3.0)."""
    clips = {clipset_path(v): {c["name"] for c in json.loads(clipset_path(v).read_text())["clips"]}
             for v in (clipset_views() if views is None else views)}
    groups: list[list[Path]] = []
    for path in sorted(clips, key=lambda p: -len(clips[p])):
        group = next((g for g in groups if any(clips[path] & clips[other] for other in g)), None)
        if group is None:
            groups.append([path])
        else:
            group.append(path)
    return groups


def chart_result(clipset: str | None, kind: str) -> str:
    """The chart's `kind` record (plan, timing, timing-x86) on `clipset` (the main chart's
    without one)."""
    if clipset:
        return CLIPSET_RESULT.format(stem=Path(clipset).stem, kind=kind)
    return {"plan": PLAN_FILE, "timing": TIMING_FILE, "timing-x86": TIMING_X86_FILE}[kind]


def builds_file(clipset: str | None) -> str:
    """The build record of the chart on `clipset` (the main chart's without one)."""
    return CLIPSET_BUILDS.format(stem=Path(clipset).stem) if clipset else BUILDS_FILE


def other_records_builds(own: Path) -> set[str]:
    """The builds every other chart's record uses: never removed by this one."""
    used: set[str] = set()
    for path in (REPO / RESULTS).glob("bd-vs-speed*builds.json"):
        if path.resolve() != own.resolve():
            record = json.loads(path.read_text())
            used |= {record["base"], *record["families"].values(),
                     *record.get("timed_on", {}).values()}
    return used


def shipped_families() -> list[str]:
    """Our chart families: every shipped single-layer preset (nceval.plotted)."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.plotted import own_names

    return own_names()


def family_list(text: str) -> list[str]:
    """A `--families` value: comma-separated names, or `all` (shipped_families)."""
    return shipped_families() if text == ALL_FAMILIES else [f for f in text.split(",") if f]


def seed_record(commit: str) -> dict:
    """A new chart's build record: every family of ours on `commit`'s build, the
    reference tools' versions as the main chart records them."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.chart import OURS

    main = json.loads((REPO / BUILDS_FILE).read_text())
    return {"base": commit, "families": {name: commit for name, _ in OURS},
            "builds": {commit: commit}, "tools": main["tools"]}


def place(source: Path, destination: Path) -> None:
    """Copy a binary into place atomically: present means complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, partial)
    partial.rename(destination)


def verify_build(binary: Path, commit: str) -> None:
    """Fail loudly unless the fresh build `binary` says (--version) it is `commit`."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.buildinfo import verify_built
    verify_built(binary, commit)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@contextmanager
def build_tree_lock(tree: Path):
    """One refresh at a time works in a shared tree: checks out and builds in a build
    tree, or installs a build directory's presets."""
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / f"{tree.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


#: The preset rung of a family named without one (presets/<codec>/medium.toml).
DEFAULT_RUNG = "medium"
#: The crates every family's encode runs (the engine, kernels and front end).
SHARED_SPEED_CRATES = ("nc-common", "nc-dsp", "nc-analysis", "nc-encoder", "nc-cli")


def speed_inputs(family: str) -> tuple[str, ...]:
    """What a family's encode speed depends on: the shared crates, its own codec's
    crate, its preset file and the build manifests. A change elsewhere (another
    codec) leaves its time as it is, so it is not timed again (x86 timing took
    36 min of a refresh)."""
    rung = family.split("@")[1] if "@" in family else DEFAULT_RUNG
    codec = chart_sanity.codec_of(family)
    return (*(f"crates/{c}" for c in (*SHARED_SPEED_CRATES, f"nc-{codec}")),
            f":(exclude){PRESETS_PATH}",
            "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo",
            *(f"{presets}/{codec}/{rung}.toml" for presets in (PRESETS_PATH, OLD_PRESETS_PATH)))


def sources_equal(old: str, new: str, inputs: tuple[str, ...] = BUILD_INPUTS) -> bool:
    """Whether two commits build the same encoder (no change under `inputs`);
    False when `old` is not a commit (a hand-made build)."""
    try:
        git("rev-parse", "--verify", f"{old}^{{commit}}")
    except subprocess.CalledProcessError:
        return False
    return subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", old, new, "--",
                           *inputs]).returncode == 0


@functools.cache
def jetson() -> str:
    """The Jetson's SSH target: NC_JETSON, from the environment or the machine's local.env."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.localenv import require
    return require("NC_JETSON")


def ssh(command: str) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", jetson(), command], check=True,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout


def scp(sources: list[str], destination: str) -> None:
    subprocess.run(["scp", "-q", *sources, destination], check=True, stdin=subprocess.DEVNULL)


def stream_mismatches(records: list[dict]) -> list[str]:
    """`point entry` for every timed encode whose stream is not the plan's (the quality
    stream: anchor_chart_timing records planned_bytes / planned_sha256 on a mismatch)."""
    return [f"{r['point']} {entry}" for r in records for entry, v in r["clips"].items()
            if "planned_bytes" in v.get("streams", {}).get("point", {})]


def combined_records(records: list[dict]) -> list[dict]:
    """One record a point: the records of a point timed in several sessions, each on its
    own clips (aws_timing --inputs), joined. An entry timed twice is an error."""
    by_point: dict[str, dict] = {}
    for record in records:
        joined = by_point.setdefault(record["point"], {**record, "clips": {}})
        twice = joined["clips"].keys() & record["clips"].keys()
        if twice:
            raise SystemExit(f"{record['point']}: entries timed twice: {sorted(twice)[:3]}")
        joined["clips"].update(record["clips"])
    return list(by_point.values())


def merge_timing(stored_path: Path, records: list[Path], rescale: bool, build: str) -> None:
    """Replace the timed points' entries in a stored timing file, each naming the `build`
    it was timed on (the site's captions say which builds a machine's speeds come from).

    `rescale` (x86) puts each per-clip time on the stored session's footing: the
    point's median per-round time ratio to the anchor times the stored anchor time.
    The raw median times and the same-session ratio to the second anchor are kept. A
    stored file of another rate mode or other clips is replaced by a new record.
    """
    loaded = combined_records([json.loads(path.read_text()) for path in records])
    refused = stream_mismatches(loaded)
    if refused:
        raise SystemExit("timed streams that are not the scored ones (owner 2026-11-23: one "
                         "binary per point for quality and timing): " + "; ".join(refused))
    settings = {k: loaded[0][k] for k in ("rate_mode", "frames")}
    # every record's entries (a point timed on part of the plan, e.g. a failed pair, adds
    # what it has)
    settings["clips"] = {c: v["kbps"] for r in loaded for c, v in r["clips"].items()}
    stored = json.loads(stored_path.read_text()) if stored_path.is_file() else {}
    if any(stored.get(k) != v for k, v in settings.items()):
        # another rate mode or clips (a record from before 2026-11-23 timed fixed QP):
        # a new record, its anchor time this session's (the median over the records)
        stored = {**settings, "anchor": ANCHOR, "platform": loaded[0].get("platform"), "points": {
            ANCHOR: {"ms_per_frame": {c: statistics.median(
                r["clips"][c]["ms_per_frame"]["anchor"] for r in loaded if c in r["clips"])
                for c in settings["clips"]}, "build": build}}}
    reference = stored["points"][ANCHOR]["ms_per_frame"]
    for record in loaded:
        clips = record["clips"]
        raw = {c: v["ms_per_frame"]["point"] for c, v in clips.items()}
        entry = {"rate_fraction": {c: v["rate_fraction"] for c, v in clips.items()},
                 "build": build}
        if rescale:
            entry.update(ms_per_frame={c: v["time_ratio"][ANCHOR] * reference[c]
                                       for c, v in clips.items()},
                         anchor_ms_per_frame=dict(reference), raw_ms_per_frame=raw,
                         same_session_time_ratio={
                             ref: {c: v["time_ratio"][ref] for c, v in clips.items()}
                             for ref in next(iter(clips.values()))["time_ratio"]
                             if ref != ANCHOR})
        else:
            entry.update(ms_per_frame=raw, anchor_ms_per_frame={
                c: v["ms_per_frame"]["anchor"] for c, v in clips.items()})
        stored["points"][record["point"]] = entry
        print(f"  {stored_path.name}: {record['point']} re-timed")
    stored_path.write_text(json.dumps(stored, indent=1) + "\n")


def refresh_heldout(args: argparse.Namespace) -> None:
    """Each clip-set chart (clipset_chains, in order) refreshed for the main refresh's
    families (a family whose streams did not change keeps its points:
    Refresh.retime_unchanged), then rendered against HELDOUT_TUNED_ANCHOR too."""
    families = ",".join([*family_list(args.families), *family_list(args.retime)])
    for clipset in [path for chain in clipset_chains() for path in chain]:
        common = [sys.executable, str(Path(__file__).resolve()), args.commit,
                  "--clipset", str(clipset), "--cloud-budget", str(args.cloud_budget),
                  "--cloud-boxes", str(args.cloud_boxes),
                  *(["--dry-run"] if args.dry_run else [])]
        print(f"== held-out chart {clipset.stem}", flush=True)
        subprocess.run([*common, "--families", families, "--metrics-only"], check=True)
        subprocess.run([*common, "--families", families, "--from", "render",
                        "--anchor", HELDOUT_TUNED_ANCHOR], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("commit", help="the commit to build (e.g. the landed change)")
    p.add_argument("--families", default="",
                   help="comma-separated chart families it changes, e.g. nanocodec-h264; "
                        "`all`: every shipped preset (nceval.plotted), new ones included")
    p.add_argument("--retime", default="",
                   help="comma-separated chart families only re-timed on COMMIT's build: their "
                        "streams must equal their recorded build's (checked), so their quality "
                        "points stay")
    p.add_argument("--time-anchors", default="",
                   help="comma-separated reference encoders (cbr.preset_anchor names, e.g. "
                        "x264-veryslow) also planned and timed on both machines")
    p.add_argument("--low-clips", help="comma-separated clips with the chart's low rungs, "
                   "for the timing plan (default: the stored plan's)")
    p.add_argument("--metrics-only", action="store_true",
                   help="no plan and no timing (nor the release and Jetson builds they need): "
                        "quality only, the stored timings stay")
    p.add_argument("--part-jobs", type=int, default=1,
                   help="measure parts run side by side (each its own process and slot share)")
    p.add_argument("--rescore", action="store_true",
                   help="measure every chart point on its recorded build (after a metric "
                        "protocol change), in parts that fit the budget")
    p.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0],
                   help="resume at this step")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="measure workers (at most 8)")
    p.add_argument("--x86-rounds", type=int, default=HOST_ROUNDS,
                   help=f"interleaved x86 timing rounds per clip (default {HOST_ROUNDS})")
    p.add_argument("--anchor", help="render (only) against another zero line, e.g. "
                   "x264-medium~psnr: nceval.chart render --anchor (outputs get a -vs- suffix)")
    p.add_argument("--clipset", help="refresh the chart on this clip set (the held-out chart) "
                   "instead of the main one; its points are measured on rented machines")
    p.add_argument("--seed", action="store_true",
                   help="with --clipset: start its build record (every family on COMMIT)")
    p.add_argument("--cloud-budget", type=float, default=CLOUD_SESSION_BUDGET,
                   help="with --clipset: USD cap of each rented session")
    p.add_argument("--cloud-boxes", type=int, default=CLOUD_BOXES,
                   help="rented sessions side by side (heldout_fleet.py --boxes)")
    p.add_argument("--heldout", action="store_true",
                   help="after the main chart, refresh the clip-set charts (clipset_chains) "
                        "for the same families, incrementally, so the site data "
                        "follows")
    p.add_argument("--pgo", action="store_true",
                   help="a release's refresh: time the release's PGO builds of COMMIT (x86 and "
                        "Jetson), placed by tools/release/release.py; never builds them")
    p.add_argument("--local", action="store_true",
                   help="measure the main chart on this PC (nceval.chart measure) instead of "
                        "on rented machines")
    p.add_argument("--dry-run", action="store_true", help="print the steps' commands only")
    args = p.parse_args()
    if not (args.families or args.rescore or args.retime or args.time_anchors):
        p.error("give --families, --retime, --rescore or a combination")
    began = time.monotonic()
    Refresh(args).execute(args.start)
    if args.heldout and not args.clipset:
        refresh_heldout(args)
    minutes, seconds = divmod(round(time.monotonic() - began), 60)
    print(f"done in {minutes}:{seconds:02d}: commit {RESULTS} and "
          f"{builds_file(args.clipset)}", flush=True)


if __name__ == "__main__":
    main()
