#!/usr/bin/env python3
"""Refresh the bitrate-vs-speed charts for the families one commit changes.

    python3 tools/dev/chart_refresh.py COMMIT --families nanocodec-h264,nanocodec-h264@h264-fast \\
        [--rescore] [--from STEP] [--workers 8] [--cpus 14,15] [--dry-run]

Run it from your own worktree: it writes that tree's docs/results (the
charts, their plan, timing and build record), which you then commit and land.
Every chart family is measured on its own build, recorded in
docs/results/bd-vs-speed-builds.json (family -> build id; build id ->
description); a build lives in BUILDS_ROOT/<id>/ on the PC (`nanocodec`, the
scoring build; `nanocodec-timed-pgo`, the profile-guided release build x86
timing uses (tools/dev/pgo_build.sh); the
commit's `presets/`) and on the Jetson (`nanocodec`, release), each with a
`COMMIT` file written last (a build without it is incomplete). The steps
(`--from` resumes at one; the steps before it are skipped):

  build        COMMIT built for x86 in the tool's own worktrees and target dirs:
               the scoring build (`gate` profile, incremental: seconds to a minute)
               first, the timed release build (fat LTO and PGO, minutes) in the
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
  time-x86     anchor_chart_timing.py time-host under tools/dev/host-timed.sh, on
               the PC's reserved timing CPUs (one per physical core), merged into
               bd-vs-speed-timing-x86.json on the stored session's footing: each
               per-clip time is the median per-round time ratio to x264-medium
               (run back to back with it) times the stored x264-medium time (the
               boosting cores' clock moves with load, between and within sessions;
               the raw times are kept in raw_ms_per_frame, and the same-session
               ratio to SVT-AV1 M13 in same_session_time_ratio).
  time-jetson  the same streams timed on the Jetson (anchor_chart_timing.py time
               under jetson-timed.sh, which retries until the board is quiet),
               merged into bd-vs-speed-timing.json.
  render       nceval.chart render of both charts, every family on its build, and the
               hardware encoders (NVENC, on this PC's GPU) beside them.
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

  x86 scoring build -> delete -> measure -> time-x86 -------.
                              `-> scan ----'                |-> render
  x86 release build (background) -----'                     |
  Jetson build (background) -> (after scan) time-jetson ----'   render -> check

time-x86 waits for measure: our own measure beside it skewed even the
anchor-paired x86 times. It times a point's clips side by side on the --cpus;
time-jetson times only the point, against the stored anchor time.

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
SCORING_PROFILE = "gate"
#: A build directory's binaries: the scoring build (scan, measure, render) and
#: the timed profile-guided release build (time-x86, tools/dev/pgo_build.sh;
#: named apart from the earlier plain `nanocodec-timed`, so a build without it
#: is rebuilt and never lends its binaries). The Jetson's `nanocodec` is a
#: plain release build.
SCORING_BINARY = "nanocodec"
TIMED_BINARY = "nanocodec-timed-pgo"
#: The profile-guided release build script (PROCEDURES "Build and check").
PGO_BUILD = REPO / "tools" / "dev" / "pgo_build.sh"
WORK = DATA / "work" / "chart-refresh"
JETSON_WORK = "nanocodec-data/work/chart-refresh"
JETSON_TARGET = "nanocodec-data/targets/chart-build"
#: The Jetson's persistent build source tree (incremental builds) and its lock.
JETSON_SRC = f"{JETSON_WORK}/src"
JETSON_BUILD_JOBS = 6
RESULTS = "docs/results"
BUILDS_FILE = f"{RESULTS}/bd-vs-speed-builds.json"
#: A chart on another clip set (--clipset) keeps its own build record beside the main one.
CLIPSET_BUILDS = RESULTS + "/bd-vs-speed-{stem}-builds.json"
#: The held-out chart's rented sessions: streams (bitstreams, logs) under the data root,
#: scoring sessions side by side and each one's limit (aws_heldout_chart.py).
CLIPSET_STREAMS = DATA / "work" / "heldout-chart" / "streams-{stem}"
CLOUD_SESSIONS = 2
CLOUD_SCORE_MINUTES = 120
PLAN_FILE = f"{RESULTS}/bd-vs-speed-plan.json"
TIMING_FILE = f"{RESULTS}/bd-vs-speed-timing.json"
TIMING_X86_FILE = f"{RESULTS}/bd-vs-speed-timing-x86.json"
#: The hardware encoders' (NVENC) wall-clock fps (tools/dev/nvenc_timing.py); render
#: adds them, measured on this PC's GPU, beside the chart.
TIMING_HARDWARE_FILE = f"{RESULTS}/bd-vs-speed-timing-nvenc.json"
#: The paths whose change can change an encoder build's output.
BINARY_INPUTS = ("crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo")
BUILD_INPUTS = (*BINARY_INPUTS, "presets")
#: Written last into a finished build directory (on both machines).
COMPLETE_MARKER = "COMMIT"
STEPS = ("build", "verify", "delete", "scan", "measure", "time-x86", "time-jetson", "render",
         "check")
#: Runs a command on the PC's reserved timing cores, one timing job at a time.
HOST_TIMED = REPO / "tools/dev/host-timed.sh"
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
        self.timed = self.targets + self.retimed
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
            self.cargo_build(BUILD_TREE, BUILD_TARGET, "release", timed, pgo=True)
        elif not self.args.dry_run:
            place(self.build_dir(donor) / TIMED_BINARY, timed)

    def cargo_build(self, tree: Path, target: Path, profile: str, destination: Path,
                    pgo: bool = False) -> None:
        """COMMIT checked out in `tree` (its own worktree: unchanged files keep their
        times, so the build is incremental), built with `profile` (with `pgo`, the
        release build through PGO_BUILD), its binary placed at `destination`."""
        log = f"build-{profile}{'-pgo' if pgo else ''}.log"
        env = {"CARGO_TARGET_DIR": str(target),
               "PATH": f"{Path.home() / '.cargo/bin'}:{os.environ['PATH']}"}
        with build_tree_lock(tree):
            if not tree.exists():
                self.run(["git", "worktree", "add", "--detach", str(tree), self.commit], log)
            self.run(["git", "checkout", "-q", "--detach", self.commit], log, tree)
            built = target / profile / "nanocodec"
            if pgo:
                self.run(["bash", str(PGO_BUILD), str(tree), str(target), str(built)], log, tree,
                         env=env)
            else:
                self.run(["cargo", "build", "--profile", profile, "-j8", "-p", "nc-cli"], log,
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
        archive = subprocess.run(["git", "-C", str(REPO), "archive", self.commit, "presets"],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "--strip-components=1", "-C", str(presets)], input=archive,
                       check=True)
        (target / COMPLETE_MARKER).write_text(self.commit + "\n")

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
        """git archive of COMMIT synced into the Jetson's persistent source tree (rsync
        without -t: only the files that changed get new times, so cargo rebuilds only
        what they touch; an archive's own mtimes are the commit time, which let cargo
        keep a stale binary), built under the shared build lock and the tree's own
        lock. The tree has no git, so NC_BUILD_COMMIT stamps the binary, and the
        build is refused (no marker) unless `--version` names COMMIT."""
        stage = f"{JETSON_WORK}/stage-{self.commit}"
        build = f"{BUILDS_ROOT}/{self.commit}"
        full = git("rev-parse", f"{self.commit}^{{commit}}")
        built = f"~/{JETSON_TARGET}/release/nanocodec"
        stamp_check = (f"{{ {built} --version | grep -Eq '^nanocodec [^ ]+ {full}$' || "
                       f"{{ echo \"Jetson build is not {full}: $({built} --version)\" >&2; "
                       f"false; }}; }}")
        inner = (f"rsync -rc --delete {stage}/ {JETSON_SRC}/ && cd {JETSON_SRC} && "
                 f"nice -n 19 env PATH=$HOME/.cargo/bin:$PATH NC_BUILD_COMMIT={full} "
                 f"cargo build --release -j{JETSON_BUILD_JOBS} -p nc-cli "
                 f"--target-dir ~/{JETSON_TARGET} && {stamp_check} && "
                 f"mkdir -p ~/{build} && cp {built} ~/{build}/ && "
                 f"echo {self.commit} > ~/{build}/{COMPLETE_MARKER}")
        remote = (f"rm -rf {stage} && mkdir -p {stage} {JETSON_SRC} && tar -x -C {stage} && "
                  f"~/nanocodec-data/jetson-build.sh flock {JETSON_SRC}.lock "
                  f"bash -c {shlex.quote(inner)}; status=$?; rm -rf {stage}; exit $status")
        return (f"git -C {shlex.quote(str(REPO))} archive {self.commit} | "
                f"ssh -o BatchMode=yes {jetson()} {shlex.quote(remote)}")

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
        timing clip); then its timing build is recorded beside its quality build."""
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
        timed_on = self.record.setdefault("timed_on", {})
        timed_on.update({f: self.commit for f in self.retimed})
        self.save_record()

    def drop_unused_builds(self) -> None:
        for family in self.targets:  # measured and timed on one build again
            self.record.get("timed_on", {}).pop(family, None)
        used = (set(self.record["families"].values()) | {self.record["base"]}
                | set(self.record.get("timed_on", {}).values()) | other_records_builds(
                    self.builds_file))
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
        out = WORK / f"host-{self.commit}"
        if out.exists() and not self.args.dry_run:
            shutil.rmtree(out)  # a re-run times afresh
        self.run([str(HOST_TIMED), "python3", "../tools/dev/anchor_chart_timing.py", "time-host",
                  "--plan", str(REPO / PLAN_FILE), "--binary", str(build / TIMED_BINARY),
                  "--presets-dir", str(build / "presets"), "--points", ",".join(self.timed),
                  "--cpus", self.args.cpus, "--rounds", str(self.args.x86_rounds),
                  "--out-dir", str(out)],
                 "time-x86.log", REPO / "eval", env={"PYTHONPATH": "."})
        if not self.args.dry_run:
            merge_timing(REPO / TIMING_X86_FILE, sorted(out.glob("t-*.json")), rescale=True)
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
            merge_timing(REPO / TIMING_FILE, sorted(local.glob("t-*.json")), rescale=False)
            shutil.rmtree(local)
            ssh(f"rm -rf {JETSON_WORK}/timing")

    def render(self) -> None:
        meta = WORK / "meta.json"
        if not self.args.dry_run:
            meta.write_text(json.dumps(self.meta(), indent=1) + "\n")
        clipset = (["--clipset", str(Path(self.args.clipset).resolve()),
                    "--prebuilt", str(self.streams())] if self.args.clipset else [])
        self.run(["python3", "-m", "nceval.chart", "render", *self.chart_args(), *clipset,
                  "--timing", str(REPO / TIMING_FILE), "--timing-x86", str(REPO / TIMING_X86_FILE),
                  "--timing-hardware", str(REPO / TIMING_HARDWARE_FILE), "--meta", str(meta), "--out", str(REPO / RESULTS),
                  "--workers", str(self.args.workers)], "render.log", REPO / "eval", scope=True)

    def check(self) -> None:
        """chart_sanity.py on the refreshed families' codecs: flags printed, never fatal."""
        if not self.timed or self.args.dry_run:
            return
        codecs = {chart_sanity.codec_of(f) for f in self.timed}
        flags = chart_sanity.check(codecs, instructions=not self.args.metrics_only)
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
            # After measure: our own measure load beside it skewed even the anchor-paired
            # x86 times (2026-11-11: AV1 Faster 107 instead of 156 fps).
            time_x86 = launch("time-x86", scan, timed_build, measure)
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


def merge_timing(stored_path: Path, records: list[Path], rescale: bool) -> None:
    """Replace the timed points' entries in a stored timing file.

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
        entry = {"budget_fraction": {c: v["budget_fraction"] for c, v in clips.items()}}
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
    p.add_argument("--cpus", default="14,15",
                   help="reserved timing CPUs for x86 timing, one per physical core")
    p.add_argument("--x86-rounds", type=int, default=HOST_ROUNDS,
                   help=f"interleaved x86 timing rounds per clip (default {HOST_ROUNDS}; "
                        "more on a loaded PC)")
    p.add_argument("--clipset", help="refresh the chart on this clip set (the held-out chart) "
                   "instead of the main one; its points are measured on rented machines")
    p.add_argument("--seed", action="store_true",
                   help="with --clipset: start its build record (every family on COMMIT)")
    p.add_argument("--cloud-budget", type=float, default=8.0,
                   help="with --clipset: USD cap over all rented sessions")
    p.add_argument("--dry-run", action="store_true", help="print the steps' commands only")
    args = p.parse_args()
    if not (args.families or args.rescore or args.retime):
        p.error("give --families, --retime, --rescore or a combination")
    began = time.monotonic()
    Refresh(args).execute(args.start)
    minutes, seconds = divmod(round(time.monotonic() - began), 60)
    print(f"done in {minutes}:{seconds:02d}: commit {RESULTS} and "
          f"{builds_file(args.clipset)}", flush=True)


if __name__ == "__main__":
    main()
