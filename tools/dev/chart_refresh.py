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
               fixed-QP streams and a 600 kbps CBR stream on each timing clip, by
               SHA-256); they keep their quality points and are timed on COMMIT's
               build (`timed_on` in bd-vs-speed-builds.json).
  delete       the replaced build's cached chart points of each refreshed family
               (VAL ladder and low rungs, by exact identity: cbr.Bench.result_path),
               so each curve keeps one result version.
  scan         the timing plan's quantizers of the refreshed and retimed families
               (our encoders by `--preset NAME`, so each build applies its own preset)
               (anchor_chart_timing.py scan) in docs/results/bd-vs-speed-plan.json.
  measure      nceval.chart measure of the refreshed families, in parts that fit the
               30-minute budget; the last part adds the low rungs.
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

With `--clipset FILE` (the held-out chart, e.g. ~/nanocodec-data/clips/
independent_fresh_2026-11-13.json) the same steps refresh a chart on that clip set,
whose families and builds are recorded apart (docs/results/bd-vs-speed-<set>-builds.json;
`--seed` starts the record with every family on COMMIT's build). Its points are too many
for this PC: measure encodes and scores them on rented machines (tools/dev/
aws_heldout_chart.py `run`, then `score-cloud`, checkpointed; `--cloud-budget`), and
render reads them from those streams. The speed axis stays the main chart's: timed on
its timing clips, said so under the chart. A build that any record still uses is never
removed.

Independent steps overlap, so the wall time is the measure (or the Jetson's
lock wait), not the sum:

  x86 scoring build -> delete -> measure ---------------------.
                              `-> scan -> time-x86 ---------|-> render
  x86 release build (background) -----'                     |
  Jetson build (background) -> (after scan) time-jetson ----'   render -> check

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
#: timing steps check every timed stream against the scan's bytes).
BUILD_TREE = DATA / "worktrees" / "chart-build"
BUILD_TARGET = DATA / "targets" / "chart-build"
SCORING_TREE = DATA / "worktrees" / "chart-build-gate"
SCORING_TARGET = DATA / "targets" / "chart-build-gate"
#: The gate build the landing verifier keeps warm (land.sh --verify): a clean one of COMMIT codes the same streams as our own scoring
#: build, so it is copied instead of building one (7 min in a cold tree).
GATE_DONORS = (DATA / "targets" / "verifier" / "gate" / "nanocodec",)
SCORING_PROFILE = "gate"
#: A build directory's binaries: the scoring build (scan, measure, render) and
#: the timed plain release build (time-x86), what users build. It was a
#: profile-guided build until 2026-11-15 (`nanocodec-timed-pgo`: 1.4-3.2 % less
#: CPU, ~25 min a refresh); the new name keeps a build without it from lending
#: its binaries. The Jetson's `nanocodec` is a plain release build too.
SCORING_BINARY = "nanocodec"
TIMED_BINARY = "nanocodec-timed-release"
WORK = DATA / "work" / "chart-refresh"
JETSON_WORK = "nanocodec-data/work/chart-refresh"
RESULTS = "docs/results"
BUILDS_FILE = f"{RESULTS}/bd-vs-speed-builds.json"
#: A chart on another clip set (--clipset) keeps its own build record beside the main one.
CLIPSET_BUILDS = RESULTS + "/bd-vs-speed-{stem}-builds.json"
#: The held-out chart's rented sessions: streams (bitstreams, logs) under the data root,
#: scoring sessions side by side and each one's limit (aws_heldout_chart.py).
CLIPSET_STREAMS = DATA / "work" / "heldout-chart" / "streams-{stem}"
#: The held-out charts the README's headline facts come from (nceval.site), refreshed
#: after the main chart with --heldout, each also rendered against the PSNR-tuned
#: x264 zero line the README quotes.
HELDOUT_CLIPSETS = (DATA / "clips" / "independent_fresh_2026-11-13.json",
                    DATA / "clips" / "independent_2026-11-13.json")
HELDOUT_TUNED_ANCHOR = "x264-medium~psnr"
#: Families whose streams are compared at once (each check encodes on one CPU).
STREAM_CHECK_WORKERS = 3
CLOUD_SESSIONS = 2
CLOUD_SCORE_MINUTES = 120
PLAN_FILE = f"{RESULTS}/bd-vs-speed-plan.json"
TIMING_FILE = f"{RESULTS}/bd-vs-speed-timing.json"
TIMING_X86_FILE = f"{RESULTS}/bd-vs-speed-timing-x86.json"
#: The hardware encoders' (NVENC) wall-clock fps (tools/dev/nvenc_timing.py); render
#: adds them, measured on this PC's GPU, beside the chart.
TIMING_HARDWARE_FILE = f"{RESULTS}/bd-vs-speed-timing-nvenc.json"
#: The paths whose change can change an encoder build's output.
#: The shipped presets (in the crate that embeds them) and, in commits before
#: 2026-11-15, where they were. A chart point takes its preset's values from
#: the build directory's copy, so a preset change alone needs no new binary.
PRESETS_PATH, OLD_PRESETS_PATH = "crates/nc-encoder/presets", "presets"
BINARY_INPUTS = ("crates", f":(exclude){PRESETS_PATH}", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo")
BUILD_INPUTS = (*BINARY_INPUTS, PRESETS_PATH, OLD_PRESETS_PATH)
#: Written last into a finished build directory (on both machines).
COMPLETE_MARKER = "COMMIT"
#: x86 timing machines side by side (aws_timing.TIMING_MACHINE, one pair at a time on
#: each): a whole-chart re-time takes about 10 min instead of 40.
X86_SESSIONS = 4
STEPS = ("build", "verify", "delete", "scan", "measure", "time-x86", "time-jetson", "render",
         "check")
MEMORY_SCOPE = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=8G",
                "-p", "MemorySwapMax=0"]
ANCHOR = "x264-medium"
#: The stored Jetson anchor timing, in `anchor_chart_timing.py time --anchor-from` form.
STORED_ANCHOR = "t-stored-anchor.json"
#: Times a quiet-board timing slot is retried before giving up (jetson-timed.sh
#: refuses after 5 minutes of load).
JETSON_ATTEMPTS = 6
#: Measure workers: a point holds its slots only while encoding and scoring, so
#: more workers than the slot share fits keep that share busy through each
#: point's unslotted setup and summary (nceval.realuse caps workers at 8).
DEFAULT_WORKERS = 8


class Refresh:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.commit = git("rev-parse", "--short=8", f"{args.commit}^{{commit}}")
        self.families = args.families.split(",") if args.families else []
        self.builds_file = REPO / builds_file(args.clipset)
        if self.builds_file.exists():
            self.record = json.loads(self.builds_file.read_text())
        elif args.seed:  # written with the first step that saves the record
            self.record = seed_record(self.commit)
        else:
            raise SystemExit(f"no build record {self.builds_file}: start one with --seed")
        unknown = [f for f in self.families if f not in self.record["families"]]
        if unknown:
            raise SystemExit(f"not chart families: {', '.join(unknown)}")
        recorded = self.record["families"]
        #: Families still on an older build whose sources differ: built and
        #: their old points deleted. One whose sources did not change keeps its
        #: build and cached points (a later step never touches it).
        self.stale = [f for f in self.families
                      if recorded[f] != self.commit and not sources_equal(recorded[f], self.commit)]
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
        #: Reference encoders to scan and time on COMMIT's session (--time-anchors):
        #: new zero-line neighbours (the fixed-QP chart's slower settings).
        self.anchors = [a for a in getattr(args, "time_anchors", "").split(",") if a]
        self.timed = self.targets + self.retimed + self.anchors
        self.print_lock = threading.Lock()
        WORK.mkdir(parents=True, exist_ok=True)

    # --- helpers ---------------------------------------------------------------------

    def say(self, line: str) -> None:
        with self.print_lock:
            print(line, flush=True)

    def run(self, cmd: list[str], log: str, cwd: Path = REPO, scope: bool = False,
            env: dict[str, str] | None = None) -> None:
        cmd = (MEMORY_SCOPE + cmd) if scope else cmd
        self.say(f"  $ {shlex.join(cmd)}  [{WORK / 'logs' / log}]")
        if self.args.dry_run:
            return
        (WORK / "logs").mkdir(exist_ok=True)
        with (WORK / "logs" / log).open("a") as out:
            status = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL,
                                    env={**os.environ, **(env or {})}).returncode
        if status:
            raise SystemExit(f"failed ({status}); see {WORK / 'logs' / log}")

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
        if ((self.build_dir(self.commit) / COMPLETE_MARKER).is_file()
                or self.commit in self.record["builds"]):
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
        if not self.args.dry_run:
            self.install_presets()
        self.record["builds"].setdefault(self.commit, self.commit)
        self.save_record()

    def build_timed(self) -> None:
        """COMMIT's x86 release build (time-x86), unless it exists or a donor's applies."""
        if not self.needs_build():
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
            self.run(["git", "checkout", "-q", "--detach", self.commit], log, tree)
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
        """COMMIT's presets into its build directory, then the marker."""
        target = self.build_dir(self.commit)
        presets = target / "presets"
        shutil.rmtree(presets, ignore_errors=True)
        presets.mkdir(parents=True)
        moved = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                                f"{self.commit}:{PRESETS_PATH}"], capture_output=True)
        source = PRESETS_PATH if moved.returncode == 0 else OLD_PRESETS_PATH
        archive = subprocess.run(["git", "-C", str(REPO), "archive", self.commit, source],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", f"--strip-components={source.count('/') + 1}", "-C",
                        str(presets)], input=archive,
                       check=True)
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
        """COMMIT's Jetson build, unless it exists; a binary donor's copy when one applies."""
        if not self.needs_build():
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

    def jetson_build_done(self) -> bool:
        marker = f"~/{BUILDS_ROOT}/{self.commit}/{COMPLETE_MARKER}"
        return ssh(f"cat {marker} 2>/dev/null || true").strip() == self.commit

    def delete(self) -> None:
        """Delete the replaced builds' cached chart points; record the new build."""
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.chart import chart_clips, low_ladder
        from nceval.config import load_paths
        from nceval.gatepolicy import EVALUATION_GATE
        from nceval.realuse import DEFAULT_FRAMES, Runner, own_family

        runner = Runner(load_paths(), EVALUATION_GATE, 1, over_budget=True)
        clips = chart_clips(self.args.clipset)
        for family in self.stale:
            old = self.build_dir(self.record["families"][family])
            if not (old / SCORING_BINARY).is_file():
                print(f"  {family}: old build {old} is gone; nothing to delete")
            else:
                stale = [own_family(family, old / SCORING_BINARY, old / "presets")]
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
            self.record["families"][family] = self.commit
        self.drop_unused_builds()
        self.save_record()

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
                    (new / SCORING_BINARY, new / "presets"), REPO / PLAN_FILE):
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
                                               REPO / PLAN_FILE))
        # Side by side: one at a time the checks of 18 families took 10 min.
        with ThreadPoolExecutor(STREAM_CHECK_WORKERS) as pool:
            same = dict(zip(self.stale, pool.map(unchanged, self.stale)))
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

    def scan(self) -> None:
        if not self.timed:
            return
        build = self.build_dir(self.commit)
        self.run(["python3", "../tools/dev/anchor_chart_timing.py", "scan",
                  "--binary", str(build / SCORING_BINARY), "--presets-dir", str(build / "presets"),
                  "--plan", str(REPO / PLAN_FILE), "--points", ",".join(self.timed)],
                 "scan.log", REPO / "eval", scope=True, env={"PYTHONPATH": "."})

    def measured(self) -> list[str] | None:
        """The families measure takes: every chart point (None) on --rescore."""
        return None if self.args.rescore else self.targets

    def measure(self) -> None:
        if not (self.targets or self.args.rescore):
            return
        if self.args.clipset:
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
        """The refreshed families on --clipset: encoded on one rented machine, scored on
        CLOUD_SESSIONS more side by side (aws_heldout_chart.py), every point's result
        brought back into this PC's point cache."""
        if self.args.rescore:
            raise SystemExit("--rescore on --clipset: give the families instead")
        build = self.build_dir(self.commit)
        common = ["--clipset", str(Path(self.args.clipset).resolve()), "--streams",
                  str(self.streams()), "--build", str(build),
                  "--families", ",".join(self.targets)]
        budget = self.args.cloud_budget / (CLOUD_SESSIONS + 1)
        self.run(["python3", "tools/dev/aws_heldout_chart.py", "run", *common,
                  "--budget", f"{budget:.2f}", "--purpose", "chart-refresh-encode"],
                 "cloud-encode.log")
        sessions = [f"{index}/{CLOUD_SESSIONS}" for index in range(1, CLOUD_SESSIONS + 1)]
        with ThreadPoolExecutor(CLOUD_SESSIONS) as pool:
            jobs = [pool.submit(self.run, [
                "python3", "tools/dev/aws_heldout_chart.py", "score-cloud", *common,
                "--session", session, "--minutes", str(CLOUD_SCORE_MINUTES),
                "--budget", f"{budget:.2f}", "--purpose", "chart-refresh-score"],
                f"cloud-score-{session.replace('/', '-')}.log") for session in sessions]
            for job in jobs:
                job.result()

    def streams(self) -> Path:
        return Path(str(CLIPSET_STREAMS).format(stem=Path(self.args.clipset).stem))

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
        build = self.build_dir(self.commit)
        if not self.args.dry_run and not (build / TIMED_BINARY).is_file():
            raise SystemExit(f"no release build of {self.commit}: resume with --from build")
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
                    "--binary", str(build / TIMED_BINARY), "--repeats", "1",
                    "--rounds", str(self.args.x86_rounds),
                    "--purpose", "chart-refresh-x86-timing", "--out", str(out)],
                    f"time-x86-{out.name.rsplit('-', 1)[1]}.log")
                    for part, out in zip([p for p in parts if p], outs)]:
                job.result()
        if not self.args.dry_run:
            merge_timing(REPO / TIMING_X86_FILE,
                         sorted(r for out in outs for r in (out / "repeat-1").glob("t-*.json")),
                         rescale=True, build=self.commit)
            for out in outs:
                shutil.rmtree(out)

    def time_jetson(self) -> None:
        if not self.timed:
            return
        if not self.args.dry_run and not self.jetson_build_done():
            raise SystemExit(f"no Jetson build of {self.commit}: resume with --from build")
        local = WORK / f"jetson-{self.commit}"
        binary = f"~/{BUILDS_ROOT}/{self.commit}/nanocodec"
        if not self.args.dry_run:
            # The timing core's clock is locked and x264 medium repeats within ~0.3 ms/frame
            # across lock windows (anchor_chart_timing.py time), so the stored anchor time
            # stands in for a paired one: each lock window times only the point (about a
            # quarter of the window the pair took).
            anchor = WORK / STORED_ANCHOR
            stored = json.loads((REPO / TIMING_FILE).read_text())["points"][ANCHOR]
            anchor.write_text(json.dumps({"clips": {
                clip: {"ms_per_frame": {"anchor": ms}}
                for clip, ms in stored["ms_per_frame"].items()}}) + "\n")
            ssh(f"rm -rf {JETSON_WORK}/timing && mkdir -p {JETSON_WORK}/timing")
            scp([str(REPO / PLAN_FILE), str(REPO / "tools/dev/anchor_chart_timing.py"),
                 str(anchor)], f"{jetson()}:{JETSON_WORK}/timing/")
            anchor.unlink()
        for family in self.timed:
            command = (f"cd {JETSON_WORK}/timing && for attempt in $(seq {JETSON_ATTEMPTS}); "
                       f"do ~/nanocodec-data/jetson-timed.sh python3 anchor_chart_timing.py time "
                       f"--plan bd-vs-speed-plan.json --point {shlex.quote(family)} "
                       f"--nanocodec {binary} --out t-{shlex.quote(family)}.json "
                       f"--anchor-from {STORED_ANCHOR} && exit 0; sleep 60; done; exit 1")
            self.run(["ssh", "-o", "BatchMode=yes", jetson(), command], "time-jetson.log")
        if not self.args.dry_run:
            local.mkdir(parents=True, exist_ok=True)
            scp([f"{jetson()}:{JETSON_WORK}/timing/t-{f}.json" for f in self.timed], str(local))
            merge_timing(REPO / TIMING_FILE, sorted(local.glob("t-*.json")), rescale=False,
                         build=self.commit)
            shutil.rmtree(local)
            ssh(f"rm -rf {JETSON_WORK}/timing")

    def render(self) -> None:
        if self.retimed and not self.args.metrics_only:
            self.record.setdefault("timed_on", {}).update({f: self.commit for f in self.retimed})
            self.save_record()
        meta = WORK / "meta.json"
        if not self.args.dry_run:
            meta.write_text(json.dumps(self.meta(), indent=1) + "\n")
        clipset = (["--clipset", str(Path(self.args.clipset).resolve()),
                    "--prebuilt", str(self.streams())] if self.args.clipset else [])
        # The main chart is rendered against both zero lines the site offers (the
        # held-out ones get the tuned line from refresh_heldout): without it the
        # site's Development + tuned-x264 view fell back to an older Defaults-only run.
        anchors = ([self.args.anchor] if getattr(self.args, "anchor", None) else
                   [None] if self.args.clipset else [None, HELDOUT_TUNED_ANCHOR])
        for zero in anchors:
            anchor = ["--anchor", zero] if zero else []
            if self.args.clipset:
                title = f"Bitrate vs encode speed, held-out clips ({Path(self.args.clipset).stem})"
                anchor += ["--title", title + (f", zero line {zero}" if zero else "")]
            self.run(["python3", "-m", "nceval.chart", "render", *self.chart_args(), *clipset,
                      *anchor, "--timing", str(REPO / TIMING_FILE),
                      "--timing-x86", str(REPO / TIMING_X86_FILE),
                      # NVENC runs only on this PC's GPU: never on a rented clip set's streams
                      *([] if self.args.clipset else
                        ["--timing-hardware", str(REPO / TIMING_HARDWARE_FILE)]),
                      "--meta", str(meta), "--out", str(REPO / RESULTS),
                      "--workers", str(self.args.workers)], "render.log", REPO / "eval",
                     scope=True)
        # The results site's data and the README's headline facts follow the new charts.
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
            wanted -= {"scan", "time-x86", "time-jetson"}
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
            scan = launch("scan", delete)
            time_x86 = launch("time-x86", scan, timed_build)
            time_jetson = launch("time-jetson", scan, jetson_build)
            render = launch("render", measure, time_x86, time_jetson)
            check = launch("check", render)
            steps = {"build": x86_build, "verify": verify, "build-jetson": jetson_build,
                     "build-timed": timed_build, "delete": delete,
                     "measure": measure, "scan": scan, "time-x86": time_x86,
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
    """One refresh at a time checks out and builds in a shared build tree."""
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


def merge_timing(stored_path: Path, records: list[Path], rescale: bool, build: str) -> None:
    """Replace the timed points' entries in a stored timing file, each naming the `build`
    it was timed on (the site's captions say which builds a machine's speeds come from).

    `rescale` (x86) puts each per-clip time on the stored session's footing: the
    point's median per-round time ratio to the anchor times the stored anchor time.
    The raw median times and the same-session ratio to the second anchor are kept.
    """
    stored = json.loads(stored_path.read_text())
    reference = stored["points"][ANCHOR]["ms_per_frame"]
    for path in records:
        record = json.loads(path.read_text())
        clips = record["clips"]
        raw = {c: v["ms_per_frame"]["point"] for c, v in clips.items()}
        entry = {"budget_fraction": {c: v["budget_fraction"] for c, v in clips.items()},
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
    """Each held-out chart refreshed for the main refresh's families (a family whose
    streams did not change keeps its points: Refresh.retime_unchanged), then rendered
    against HELDOUT_TUNED_ANCHOR too."""
    families = ",".join(f for f in (args.families, args.retime) if f)
    for clipset in HELDOUT_CLIPSETS:
        common = [sys.executable, str(Path(__file__).resolve()), args.commit,
                  "--clipset", str(clipset), "--cloud-budget", str(args.cloud_budget),
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
                   help="comma-separated chart families it changes, e.g. nanocodec-h264")
    p.add_argument("--retime", default="",
                   help="comma-separated chart families only re-timed on COMMIT's build: their "
                        "streams must equal their recorded build's (checked), so their quality "
                        "points stay")
    p.add_argument("--time-anchors", default="",
                   help="comma-separated reference encoders (cbr.preset_anchor names, e.g. "
                        "x264-veryslow) also scanned and timed on both machines")
    p.add_argument("--metrics-only", action="store_true",
                   help="no scan and no timing (nor the release and Jetson builds they need): "
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
    p.add_argument("--cloud-budget", type=float, default=8.0,
                   help="with --clipset: USD cap over all rented sessions")
    p.add_argument("--heldout", action="store_true",
                   help="after the main chart, refresh the held-out charts (HELDOUT_CLIPSETS) "
                        "for the same families, incrementally, so the README's headline "
                        "facts follow")
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
