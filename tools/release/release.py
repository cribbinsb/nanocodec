#!/usr/bin/env python3
"""Make one nanocodec release: version, PGO builds, charts, crates, GitHub, mirror.

    python3 tools/release/release.py VERSION [--commit SHA] [--from STAGE] [--only STAGE]
        [--pipe-clean] [--real [--yes-publish]]

Day-to-day builds stay plain release builds; only a published version gets
profile-guided (PGO) builds, and the published charts change only at a
release (docs/PROCEDURES.md "Release"). The stages, each resumable (a finished
stage is recorded in the release's state file and skipped; `--from` redoes one
and those after it) and idempotent; independent ones run side by side
(STAGE_NEEDS: the charts' quality-only chains beside the PGO builds, crates and
github beside the charts):

  commit   The release commit: the newest commit of the integration branch that
           passed both parts of `land.sh --verify` (run/verified-history, the
           verified-head files, the verify logs), or --commit, which must have.
           In its own worktree: the workspace version bumped, the lockfile and
           the crate README's dependency line with it, and a CHANGELOG section
           drafted from the commit subjects since the last release for a human
           to edit. One commit on top ("release X.Y.Z: ..."), whose diff may touch
           only those files, so the verified code is exactly what ships.
           --real: the draft must have been edited (its marker line removed)
           before the commit is made; branch release/vX.Y.Z and tag vX.Y.Z on
           it, and the same commit cherry-picked and landed on the integration
           branch by land.sh, so the main line carries the version and CHANGELOG.
  pgo      PGO builds of that commit, cached by commit: x86_64 on the remote build
           box (nc-remote), aarch64 on the Jetson (its own toolchain), each the CLI
           plus the C API's static and shared libraries, trained on public
           clips (TRAINING_CLIPS: never the charts' or the lens's) and built
           beside a plain release build of the same commit. Every shipped
           preset of every codec must code byte-identical streams in both
           (tools/dev/preset_identity.py: on the PC for x86, on the board for
           aarch64), and the Jetson's PGO-over-plain cycles on one preset are
           recorded (tools/dev/jperf.py). WASM stays a plain build: PGO needs
           the profile run on the target, and the browser module is built by
           tools/web/build.sh as before.
  charts   --real: in a worktree of the integration head, docs/results/release.json
           names the release (the site's speed captions read "release X.Y.Z,
           PGO build (commit)"), tools/dev/chart_refresh.py --pgo re-measures
           and re-times every family of ours (c8a Zen 5 and Jetson, held-out
           charts too), the CBR zero lines are re-measured with XPSNR on
           (nceval.chart measure; the held-out ones by aws_heldout_chart.py),
           the Graviton4 (c8g) and Zen 3 (c6a) CBR series by aws_timing.py and
           anchor_chart_timing.py merge, and the Fixed QP lens's encoders of
           ours re-timed on c8a, c6a, c8g and the Jetson (fixedqp_timing.py
           plan/run/record/merge, SVT-AV1 M10/M12/M13 as the same-session check);
           then committed and landed. Dry run: the full refresh is printed
           (chart_refresh --dry-run, aws_timing --dry-run) and a pipe-clean
           runs: the Fixed QP lens timing of one encoder on one sequence on
           each machine, and the Jetson's CBR timing of one family, into
           this release's directory only. The CBR chart's timing clips are
           private: they go to rented machines only in a real release, as
           chart_refresh does today.
  crates   tools/release/publish_crates.sh at the release commit: its dry run
           (every crate packaged and built, on the build box) unless
           --yes-publish. crates.io carries source, so PGO never reaches crate
           users (the CHANGELOG draft and the crate README say so).
  github   The GitHub release's files: per architecture a tarball of the PGO CLI,
           libnanocodec.a, libnanocodec.so and nanocodec.h, SHA256SUMS, the
           release notes (the CHANGELOG section) and the `gh release create`
           command. The tag goes on a public commit: the release commit's tree
           minus the internal docs (publish_public.py's filter), parented on
           the public head and reachable from the tag alone. Executed only
           with --yes-publish.
  publish  --real: the mirror (overnight ref, mirror.py, publish_public.py) and the
           results site (tools/web/publish.sh from its own worktree). Dry run:
           publish_public.py --dry-run.

Modes. The default is a dry run: every stage does its work in this release's
own directory (RELEASE_ROOT/vX.Y.Z-dry) and worktree, and nothing outside them
changes: no tag, branch, landing, docs/results write, upload or push.
`--pipe-clean` trains PGO on one codec and keeps stage 3 to its pipe-clean, to
prove the plumbing quickly. `--real` makes the release (tag, landings,
charts, mirror); `--yes-publish`, which only the owner gives, per release,
uploads (crates.io, the GitHub release and its tag). A published crate
version can never be removed.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / "nanocodec-data"
RELEASE_ROOT = DATA / "release"
#: PGO builds by release commit (BUILDS/<commit8>[-pipeclean]/<arch>/).
BUILDS = RELEASE_ROOT / "builds"
WORKTREES = DATA / "worktrees"
INTEGRATION = "variance-integration"
RUN = DATA / "run"
VERIFY_HISTORY = RUN / "verified-history"
VERIFY_HEADS = {"streams": RUN / "verified-streams-head", "tests": RUN / "verified-head"}
VERIFY_LOGS = RUN / "land-logs"
VERIFIED_LINE = re.compile(r"^== (streams|tests) verified: \S+ ([0-9a-f]{7,40})$")
NC_REMOTE = DATA / "bin" / "nc-remote"
CARGO = Path.home() / ".cargo" / "bin" / "cargo"
MEMORY_SCOPE = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=8G",
                "-p", "MemorySwapMax=0"]
#: The coordinator's public-mirror tools (outside the repo: they hold its push rules).
OVERNIGHT = DATA / "work" / "overnight"
PUBLISH_PUBLIC = OVERNIGHT / "publish_public.py"
DEMO_PUBLISH_TREE = WORKTREES / "demo-publish"
PUBLIC_REF = "refs/heads/public"
#: The demo site and results pages are published from this worktree (PROCEDURES "Publishing").

STAGES = ("commit", "pgo", "charts", "crates", "github", "publish")
#: The stages each stage waits for; the others run side by side (owner, 2026-11-18:
#: independent stages in parallel). The charts stage starts with the commit: its
#: clip-set charts' chains measure quality only, on the commit's scoring build, so they
#: run beside the PGO builds, and only its timing chains wait for them (Release.wait_for;
#: a dry run's pipe-clean times them too). crates and github need the PGO builds checked
#: (and github the binaries), never the charts.
STAGE_NEEDS = {"commit": (), "pgo": ("commit",), "charts": ("commit",), "crates": ("pgo",),
               "github": ("pgo",), "publish": ("charts", "crates", "github")}
#: A real release's expected wall time per stage, minutes, from 0.3.0 (2026-11-18): the
#: commit with its landing; both PGO builds side by side (x86 23.5 on the build box,
#: aarch64 30.7 on the Jetson) and their identity checks; the crates' dry run (2.2 cold;
#: the upload, --yes-publish, took 11.1); the GitHub files; the mirror and site. The charts
#: stage is its slowest chain (CHART_CHAIN_MINUTES), then the site, commit and landing; a
#: dry run's pipe-clean took 4.4.
STAGE_MINUTES = {"commit": 2, "pgo": 34, "crates": 3, "github": 1, "publish": 6}
CRATES_UPLOAD_MINUTES = 11
#: The charts stage's site data, commit and landing after its chains (0.3.0: 35 s).
CHARTS_LAND_MINUTES = 1
#: Each chart chain's minutes, from 0.3.0 unless said: main (zero line 0.6, refresh 16.2
#: with the c8a and Jetson CBR timing), CBR c6a 24.9 and c8g 18.8, lens c8a 18.4, c6a
#: 29.0, c8g 24.2, and the Jetson's lens at two of six points (a third of its 92 min at
#: six) after the main chain's Jetson CBR timing on the same lock (15.4). The clip-set
#: chains depend on N, the families whose streams changed since their recorded builds
#: (chart_refresh's verify; 0.3.0: 6), and are priced at N = 14 (0.4.0's commit subjects:
#: H.264 Default, HEVC and VVC (shared SAO), AV2, AV1 Faster), with the scoring sessions
#: of CLIPSET_CLOUD_REGIONS:
#: - all held out, then fresh (its clips cached): coding ~8, scoring 11 of session
#:   overhead + 1.7 c7a.metal-min per family on 1 session (0.3.0: 3,232 points of
#:   ~0.3 Mpx in 6 min), renders ~8, the fresh refresh and renders ~9: 38 + 1.7 N;
#: - VC-1K (0.92 Mpx): coding ~13, scoring 11 + 5.1 metal-min per family (437 points at
#:   85/min, 2026-11-18) over 3 sessions, renders ~8: 32 + 1.7 N.
#: Each chain as measured at 0.5.0 (2026-11-20, state.json: every command's minutes summed
#: per chain, 26 families of ours), times 35/26 for 0.6.0's 35 families (2026-11-21: the
#: renamed and new rungs), rounded up. The estimates above (from 0.3.0) were far under it:
#: 0.5.0's main chain took 85 (priced 17), held out 137 (62), VC-1K 90 (56), the Fixed QP
#: gaps 87-89 (26-36). The main chain's point cache (work/realuse/f150-ip300) was empty at
#: 0.6.0, so all its 1,554 points are coded again on this PC (Runner.price: ~51 min at 8
#: workers for the measure alone).
#: 0.6.0 (2026-11-21) moved the main chart's measure and scan and the Fixed QP gaps to
#: rented machines (measured on 3 VAL clips x 40 families: 680 encodes in 0.7 min on a
#: c7a.24xlarge, its 440 points scored at ~60/min a clip shard; a 2-clip gap 0.35 min of
#: box time; a 3-family scan 19 s on the build box): the main chain after the PGO builds
#: is the scan, both timings (the Jetson's ~25) and the render (its quality ~20 min runs
#: from the release commit, MAIN_BEFORE_PGO), and each Fixed QP gap part one box.
CHART_CHAIN_MINUTES = {"main": 45, "heldout-independent_2026-11-13": 185,
                       "heldout-vc1k_2026-11-18": 121, "cbr-zen3": 37,
                       "cbr-graviton": 30, "lens-x86": 34, "lens-zen3": 44,
                       "lens-graviton": 44, "lens-jetson": 39, "qp-vc1k": 35,
                       "qp-fresh": 30, "lens-quality": 29}
#: Our presets in a tree (the release commit's: its PGO build's own).
PRESETS = "crates/nc-encoder/presets"
#: The Fixed QP sample views' chains (qp_commands) start with the PGO builds: our 26
#: families' gap on this PC, then their c8a timing. 2026-11-18's VC-1K runs (one
#: 8-worker gap, this PC at load 50-67): 2.5-2.8 min a family (5 min for two, 2.8 for
#: one), so 26 in 3 parts ~22 min; held-out fresh at 0.42 of VC-1K's pixels ~12 in 2;
#: the c8a timing 8-12 min on 16 instances (VC-1K), a plan and a record in seconds.
QP_PARTS = {"qp-vc1k": 3}
QP_DEFAULT_PARTS = 2
QP_WORKERS = 8
#: A Fixed QP gap part's box budget, USD (aws_measure.py --budget).
QP_BOX_BUDGET = 3.0
QP_TIMING_PARALLEL = 16
#: The Fixed QP lens's quality chain (lens_quality_commands): every lens row of ours
#: (0.4.0: 49 variants of 26 presets) on the build box, LENS_BOX_JOBS slots side by side
#: with 8 workers each. 0.3.0: 110 s a variant on one slot once built (492 s with the
#: build); here 7 variants a slot: ~14 min, plus the syncs, compare and rows: ~20.
LENS_QUALITY_CHAIN = "lens-quality"
LENS_SITE_DATA = "web/results/data/fixed-qp-lens.json"
LENS_BOX_JOBS = 7
LENS_BOX_WORKERS = 8
#: The lens's public sequences on the box, under its home (PROCEDURES "Remote build box").
LENS_BOX_SOURCES = "nanocodec-data/ctc-sources"
#: Chains that need no PGO build (the clip-set charts: quality only).
QUALITY_CHAIN_PREFIX = "heldout-"
DRY_CHARTS_MINUTES = 5
#: A release refuses to start when its stages are expected to take longer (owner,
#: 2026-11-18), unless --over-budget.
MAX_RELEASE_MINUTES = 90
#: The charts stage's pre-flight: the space a whole stage needs under the work budget
#: (nceval.diskguard), with room to spare (2026-11-19: chains measured at 102 of 100 GiB).
#: A resumed stage needs the share of it of the chains with data-writing commands left
#: (Release.charts_work_gib; 2026-11-20 it refused twice with only renders left).
CHARTS_WORK_HEADROOM_GIB = 25
#: Chart commands that write next to nothing under work/ (renders from cached points,
#: merges and records of small JSON files, plans, the site's data): by name suffix.
LIGHT_CHART_COMMANDS = ("-psnr", "-merge", "-record", "-plan", "-site", "-compare", "-rows")
#: A failed chart command is retried once (unless --no-chain-retry) after this wait: its
#: rented sessions fall back through other instance types already (aws_session), and
#: spot capacity comes back within minutes as often as not (2026-11-20: none at 14:17,
#: a c7a.24xlarge at once at 16:33).
CHAIN_RETRY_WAIT_S = 120
SHORT = 8

# --- the release commit -------------------------------------------------------------------

#: The only files the release commit may change.
RELEASE_FILES = ("Cargo.toml", "Cargo.lock", "CHANGELOG.md", "crates/nanocodec/README.md")
CRATE_README = "crates/nanocodec/README.md"
DRAFT_MARKER = ("<!-- release.py draft: rewrite this section for users, then delete this "
                "line (a real release refuses to commit while it is here) -->")
RELEASE_SUBJECT = re.compile(r"^release (\d+\.\d+\.\d+)\b")
#: Commit-subject areas that are internal work, counted in the draft instead of listed.
INTERNAL_AREAS = {"docs", "eval", "tools", "ci", "record", "claude.md", "procedures",
                  "status", "readme", "land.sh", "web", "coordinator", "other"}
SPEED_NOTE = (
    "- **Speed figures.** The results site's speeds are measured on this release's "
    "profile-guided (PGO) builds of the command-line tool and C API (x86_64 and aarch64, "
    "attached to the GitHub release). crates.io carries source: a crate built by cargo is a "
    "plain release build, which took 1.4-3.2 % more CPU on x86 than PGO (AV1, HEVC and H.264 "
    "Faster, 2026-11-11). The browser (WASM) module is a plain build.")

# --- PGO ----------------------------------------------------------------------------------

#: Public 1280x720 50 fps sequences (Xiph derf; 4:2:0, 8-bit) the PGO builds train on.
#: None is a chart or lens clip (the CBR chart times private clips, the Fixed QP lens
#: the AOM CTC set), so the timed content never shaped the profile; they are kept on
#: this PC and the Jetson only, as raw cuts, never redistributed.
DERF = "https://media.xiph.org/video/derf/y4m"
TRAINING_CLIPS = ("720p50_parkrun_ter", "old_town_cross_420_720p50",
                  "in_to_tree_420_720p50", "ducks_take_off_420_720p50")
TRAIN_WIDTH, TRAIN_HEIGHT = 1280, 720
TRAIN_FRAME_BYTES = TRAIN_WIDTH * TRAIN_HEIGHT * 3 // 2
#: Frames cut from each sequence, from TRAIN_START (past any title leader), and the
#: frames pgo_build.sh encodes of each.
TRAIN_START, TRAIN_FRAMES = 50, 30
PGO_FRAMES = 20
#: The pipe-clean trains on one codec's presets (the build covers every codec still).
PIPE_CLEAN_CODECS = "av1"
TRAIN_DIR = RELEASE_ROOT / "pgo-clips"
#: The same cuts zstd-compressed for the build box (nc-remote never syncs *.yuv files).
TRAIN_ZST_DIR = RELEASE_ROOT / "pgo-clips-zst"
#: The x86 build: a job on the remote build box, its outputs fetched into the worktree.
REMOTE_JOB = "release-pgo"
REMOTE_OUT = "remote-out/release"
#: The Jetson build: sources unpacked per commit, its own persistent targets.
JETSON_ROOT = "nanocodec-data/release-build"
JETSON_TARGETS = "nanocodec-data/targets/release"
JETSON_JOBS = 4
JETSON_IDENTITY_CLIPS = "~/nanocodec-data/clips/quick"
JETSON_BUILD = "~/nanocodec-data/jetson-build.sh"
JETSON_TIMED = "~/nanocodec-data/jetson-timed.sh"
#: The sources an aarch64 build and its checks need (the presets live in crates/).
JETSON_SOURCES = ("crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo",
                  "eval/nceval", "tools/dev/pgo_build.sh", "tools/dev/preset_identity.py",
                  "tools/dev/jperf.py")
#: JETSON_SOURCES taken from this checkout, not the release commit: the release tools and
#: the evaluation library they import (both tooling; no binary is built from them).
LAUNCH_TREE_SOURCES = ("tools/", "eval/")
#: The Jetson's PGO-over-plain cycle check: one preset, one clip, min of the rounds.
CYCLES_ROUNDS = 3
CYCLES_CLIP = "crowd_mot17_09"
CYCLES_ARGS = ("encode --codec av1 --preset fast --input {} --width 1280 --height 720 "
               "--fps 30 --frames 60 --bitrate 600 --buffer-ms 1000 --intra-period 300 "
               "--output /dev/null")
ARCHES = ("x86_64", "aarch64")
ARTIFACTS = ("nanocodec", "libnanocodec.a", "libnanocodec.so")
CAPI_HEADER = "crates/nc-capi/include/nanocodec.h"

# --- charts -------------------------------------------------------------------------------

#: The charts stage's worktree, one per version: a tree left by an earlier release would
#: otherwise be reused at that release's branch and code (0.3.0's was still there).
CHART_TREE = WORKTREES / "release-charts-v{version}"
RESULTS = "docs/results"
RELEASE_RECORD = f"{RESULTS}/release.json"
BUILDS_RECORD = f"{RESULTS}/bd-vs-speed-builds.json"
CHART_BUILDS = DATA / "charts" / "builds"
JETSON_CHART_BUILDS = "nanocodec-data/charts/builds"
PGO_TIMED_BINARY = "nanocodec-timed-pgo"
JETSON_PGO_BINARY = "nanocodec-pgo"
ZERO_LINES = ("x264-medium", "x264-medium~psnr")
#: The held-out charts are also rendered against this zero line (chart_refresh.py).
HELDOUT_TUNED_ANCHOR = "x264-medium~psnr"
#: Each held-out chart's prebuilt streams (chart_refresh.CLIPSET_STREAMS).
HELDOUT_STREAMS = DATA / "work" / "heldout-chart" / "streams-{stem}"
#: A held-out zero-line session's spending cap, USD, and hard lifetime, minutes (the
#: cap prices the whole lifetime; coding or scoring two families takes ~5-15 min)
ZERO_LINE_BUDGET = 2.0
ZERO_LINE_LIFETIME = 40
#: Each clip-set chart's scoring sessions, one per region listed (chart_refresh
#: --cloud-regions). Both chains score at once, and each region's spot quota holds two
#: c7a.metal-48xl, each chain's encode session and then one scoring session at a time in
#: eu-north-1: VC-1K (0.92 Mpx, ~5.1 metal-min a family) gets three, the held-out sets
#: (~0.3 Mpx, ~1.7) one, so both chains take ~32-38 + 1.7 N min (two each: VC-1K
#: 32 + 2.6 N, the release's longest chain past N = 18).
CLIPSET_CLOUD_REGIONS = {"vc1k_2026-11-18": "eu-central-1,eu-north-1,eu-central-1"}
HELDOUT_CLOUD_REGIONS = "eu-north-1,eu-north-1"
#: c7a.metal-48xl (192 vCPU) sessions each region's spot quota holds at once (2026-11-21:
#: eu-north-1 raised to 1,200 vCPU; the others 512), with the main chart's two scoring
#: sessions (chart_refresh.CLOUD_REGIONS) beside the clip-set chains'.
METAL_SESSIONS_PER_REGION = {"eu-north-1": 6}
DEFAULT_METAL_SESSIONS = 2
#: The main chain's commands that need no PGO build (quality only, on rented machines):
#: they start with the release commit, the rest of the chain after the PGO builds.
MAIN_BEFORE_PGO = ("main-build", "main-quality")
#: The main chart's scoring sessions (chart_refresh --cloud-regions).
MAIN_CLOUD_REGIONS = "eu-north-1,eu-north-1"
#: Fixed QP lens: per machine its timing record and how our build is named there.
LENS_MACHINES = {"x86": "fixed-qp-timing-x86.json", "zen3": "fixed-qp-timing-zen3.json",
                 "graviton": "fixed-qp-timing-graviton.json",
                 "jetson": "fixed-qp-timing-jetson.json"}
#: The Jetson times the lens at two of the six quantizer points of each encoder's scale
#: (nceval.ctc.conditions.POINTS: QP 27/42, qindex 135/210, Q63 34/52): all six on its
#: one timing core took 92 min at 0.3.0 (2,016 encodes). Positions, not QP values: the
#: AV1/VP9/AV2 and SVT-AV1 scales have no QP 27 or 42 (2026-11-19: "27,42" planned none).
JETSON_LENS_POINT_POSITIONS = (1, 4)
#: SVT-AV1's lens rungs timed beside ours as the same-session check (PROCEDURES).
LENS_CHECK = ("svtav1-m10", "svtav1-m12", "svtav1-m13")
#: The rented CBR series beyond chart_refresh's c8a: instance, record, machine wording.
CBR_SERIES = {"zen3": ("c6a.large", "bd-vs-speed-timing-zen3.json",
                       "AWS c6a.large, Zen 3, one pair at a time"),
              "graviton": ("c8g.large", "bd-vs-speed-timing-graviton.json",
                           "AWS c8g.large, one pair at a time")}
AARCH64_TOOLS = DATA / "anchors" / "aarch64" / "bin"
#: The pipe-clean's single encoder and sequence (Fixed QP) and family (Jetson CBR).
PIPE_CLEAN_ENCODER = "nanocodec-av1@fast"
PIPE_CLEAN_SEQUENCE = "Johnny_1280x720_60"
PIPE_CLEAN_FAMILY = "nanocodec-av1@fast"

# --- GitHub -------------------------------------------------------------------------------

GH_REPO = "cribbinsb/nanocodec"


def say(line: str) -> None:
    print(f"[release {datetime.datetime.now():%H:%M:%S}] {line}", flush=True)


def alarm(line: str) -> None:
    """`say`, and on stderr too unless it is the same file as stdout (the run log)."""
    say(line)
    try:
        same = os.path.samestat(os.fstat(sys.stdout.fileno()), os.fstat(sys.stderr.fileno()))
    except (OSError, ValueError, AttributeError):  # a captured stream has no descriptor
        same = False
    if not same:
        print(f"[release {datetime.datetime.now():%H:%M:%S}] {line}", file=sys.stderr,
              flush=True)


def run(cmd: list[str], cwd: Path = REPO, log: Path | None = None, check: bool = True,
        capture: bool = False, env: dict[str, str] | None = None) -> str:
    """Run `cmd`; its output to `log` (appended) or captured and returned."""
    say(f"$ {shlex.join(cmd)}" + (f"  [{log}]" if log else ""))
    full_env = {**os.environ, **env} if env else None
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as out:
            status = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=full_env).returncode
        if check and status:
            raise SystemExit(f"failed ({status}): see {log}")
        return ""
    result = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True,
                            stdin=subprocess.DEVNULL, env=full_env)
    if check and result.returncode:
        raise SystemExit(f"failed ({result.returncode}): {shlex.join(cmd)}\n"
                         f"{(result.stderr or '')[-2000:]}")
    return (result.stdout or "").strip() if capture else ""


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def jetson_host() -> str:
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.localenv import require
    return require("NC_JETSON")


def jetson(command: str, log: Path | None = None) -> str:
    return run(["ssh", "-o", "BatchMode=yes", jetson_host(), command], log=log,
               capture=log is None)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- verified commits -----------------------------------------------------------------------

def verified_commits() -> dict[str, set[str]]:
    """Every commit each part of `land.sh --verify` passed (full hashes): the durable
    history, the current head files and the verify logs still kept (LOG_DAYS)."""
    found: dict[str, set[str]] = {"streams": set(), "tests": set()}
    if VERIFY_HISTORY.is_file():
        for line in VERIFY_HISTORY.read_text().splitlines():
            part, commit, *_ = line.split()
            found[part].add(commit)
    for part, path in VERIFY_HEADS.items():
        if path.is_file():
            found[part].add(path.read_text().strip())
    for log in sorted(VERIFY_LOGS.glob("*.console")):
        for line in log.read_text(errors="replace").splitlines():
            match = VERIFIED_LINE.match(line.strip())
            if match:
                found[match[1]].add(match[2])
    return {part: {full_commit(c) for c in commits if full_commit(c)}
            for part, commits in found.items()}


def full_commit(commit: str) -> str | None:
    result = subprocess.run(["git", "-C", str(REPO), "rev-parse", "-q", "--verify",
                             f"{commit}^{{commit}}"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def pick_commit(requested: str | None) -> str:
    """The release's base: `requested`, or the newest integration commit that passed
    both verify parts; refused unless both passed it."""
    verified = verified_commits()
    both = verified["streams"] & verified["tests"]
    if requested:
        commit = full_commit(requested) or sys.exit(f"no commit {requested}")
        missing = [p for p in ("streams", "tests") if commit not in verified[p]]
        if missing:
            sys.exit(f"{commit[:9]} has not passed land.sh --verify ({' and '.join(missing)}): "
                     f"run `tools/dev/land.sh --verify` while it is the head, or pick a "
                     f"verified one")
        return commit
    for commit in git("rev-list", "--first-parent", "-2000", INTEGRATION).split():
        if commit in both:
            return commit
    sys.exit("no integration commit passed both verify parts: run tools/dev/land.sh --verify")


def last_release(base: str) -> tuple[str, str]:
    """(commit, version) of the newest release before `base`: a vX.Y.Z tag on its
    history, else the newest `release X.Y.Z` commit subject."""
    tags = [t for t in git("tag", "--merged", base, "--sort=-v:refname", "v*").split()
            if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    if tags:
        return git("rev-list", "-1", tags[0]), tags[0][1:]
    for line in git("log", "--first-parent", "--format=%H %s", base).splitlines():
        commit, _, subject = line.partition(" ")
        match = RELEASE_SUBJECT.match(subject)
        if match:
            return commit, match[1]
    sys.exit("no earlier release (tag or `release X.Y.Z` commit) to draft the CHANGELOG from")


# --- version bump and CHANGELOG -------------------------------------------------------------

def bump_files(tree: Path, old: str, new: str) -> None:
    """The workspace version, the internal dependency versions, the lockfile's
    entries of our packages and the crate README's dependency line: `old` -> `new`."""
    manifest = tree / "Cargo.toml"
    text = manifest.read_text()
    text, package = re.subn(r'(\[workspace\.package\]\nversion = ")' + re.escape(old) + '"',
                            r"\g<1>" + new + '"', text)
    text, deps = re.subn(r'(package = "nanocodec-[a-z0-9-]+", version = ")' + re.escape(old)
                         + '"', r"\g<1>" + new + '"', text)
    if package != 1 or not deps:
        sys.exit(f"Cargo.toml: workspace version {old} not found as expected")
    manifest.write_text(text)
    lock = tree / "Cargo.lock"
    text, entries = re.subn(r'(name = "nanocodec[a-z0-9-]*"\nversion = ")' + re.escape(old)
                            + '"', r"\g<1>" + new + '"', lock.read_text())
    if not entries:
        sys.exit(f"Cargo.lock: no nanocodec package at {old}")
    lock.write_text(text)
    readme = tree / CRATE_README
    short_old, short_new = old.rsplit(".", 1)[0], new.rsplit(".", 1)[0]
    text = readme.read_text().replace(f'version = "{short_old}"', f'version = "{short_new}"')
    readme.write_text(text)


def check_lockfile(tree: Path) -> None:
    """The lockfile is exactly what cargo would write: a `--locked` build of the tag
    works (no network: the dependencies are unchanged). Without --no-deps, so every
    package's entry is checked, the workspace members' versions too."""
    run([str(CARGO), "metadata", "--locked", "--offline", "--format-version", "1"],
        cwd=tree, capture=True)


def changelog_draft(version: str, since: str, since_version: str, base: str) -> str:
    """A CHANGELOG section for a human to rewrite: the speed note, then the commit
    subjects since the last release grouped by their area prefix (internal areas
    counted only)."""
    subjects = git("log", "--no-merges", "--format=%s", f"{since}..{base}").splitlines()
    groups: dict[str, list[str]] = {}
    for subject in subjects:
        area, sep, rest = subject.partition(":")
        key = area.strip().lower() if sep and len(area) <= 24 and " " not in area.strip() \
            else "other"
        groups.setdefault(key, []).append(rest.strip() if key != "other" else subject)
    today = datetime.date.today().isoformat()
    lines = [f"## {version} ({today})", DRAFT_MARKER, "", SPEED_NOTE, "",
             f"### Changes since {since_version} ({len(subjects)} commits; draft, by area)"]
    internal = []
    for area in sorted(groups, key=lambda a: (-len(groups[a]), a)):
        if area in INTERNAL_AREAS:
            internal.append(f"{area} {len(groups[area])}")
            continue
        lines += ["", f"#### {area} ({len(groups[area])})", *(f"- {s}" for s in groups[area])]
    if internal:
        lines += ["", f"Internal (not listed): {', '.join(internal)}."]
    return "\n".join(lines) + "\n\n"


def insert_changelog(tree: Path, section: str) -> None:
    path = tree / "CHANGELOG.md"
    text = path.read_text()
    head = "# Changelog\n\n"
    if not text.startswith(head):
        sys.exit("CHANGELOG.md does not start with '# Changelog'")
    path.write_text(head + section + text[len(head):])


def changelog_section(text: str, version: str) -> str:
    """One version's CHANGELOG section, without its heading."""
    match = re.search(rf"^## {re.escape(version)} \(.*?\)\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match:
        sys.exit(f"CHANGELOG.md has no section {version}")
    return match[1].strip() + "\n"


# --- the release ----------------------------------------------------------------------------

class Release:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.version = args.version
        self.real = args.real
        self.publish = args.yes_publish
        self.pipe_clean = args.pipe_clean
        self.name = f"v{self.version}" + ("" if self.real else "-dry")
        self.root = RELEASE_ROOT / self.name
        self.tree = WORKTREES / f"release-{self.name}"
        self.state_file = self.root / "state.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = (json.loads(self.state_file.read_text()) if self.state_file.is_file()
                      else {"version": self.version, "stages": {}})
        #: Stages run side by side (STAGE_NEEDS): the state and its file are shared.
        self.state_lock = threading.RLock()
        #: This run's stages: each one's event is set when it ends, done or failed.
        self.running: dict[str, threading.Event] = {}
        self.failed: set[str] = set()

    def save(self) -> None:
        with self.state_lock:
            self.state_file.write_text(json.dumps(self.state, indent=1) + "\n")

    def log(self, name: str) -> Path:
        return self.root / "logs" / f"{name}.log"

    @property
    def commit(self) -> str:
        return self.state["release_commit"]

    @property
    def short(self) -> str:
        return self.commit[:SHORT]

    @property
    def binary_commit(self) -> str:
        """The commit the shipped and charted binaries are built from: the release
        commit, or an earlier one whose builds exist and from which it differs only in
        files no binary is built from (BINARY_NEUTRAL; reuse_binaries)."""
        return self.state.get("binary_commit") or self.commit

    @property
    def binary_short(self) -> str:
        return self.binary_commit[:SHORT]

    @property
    def build_key(self) -> str:
        """A build's cache key: the binaries' commit, and whether it is a pipe-clean's."""
        return self.binary_short + ("-pipeclean" if self.pipe_clean else "")

    def builds(self, arch: str) -> Path:
        return BUILDS / self.build_key / arch

    def version_line(self) -> str:
        return f"nanocodec {self.version} {self.binary_commit}"

    def reuse_binaries(self) -> None:
        """Never rebuild for a change no binary is built from: an earlier release commit
        of this version with complete builds of both architectures, differing from the
        release commit only in BINARY_NEUTRAL files, provides the binaries (0.3.0: the
        lockfile fixed after its builds were timed for the charts)."""
        if self.state.get("binary_commit") or not BUILDS.is_dir():
            return
        for build in sorted(BUILDS.iterdir()):
            commit = full_commit(build.name)
            if (not commit or commit == self.commit or build.name.endswith("-pipeclean")
                    or not all((build / arch / a).is_file() for arch in ARCHES
                               for a in ARTIFACTS)):
                continue
            changed = git("diff", "--name-only", commit, self.commit).split()
            if (git("rev-parse", f"{commit}~1") == git("rev-parse", f"{self.commit}~1")
                    and all(binary_neutral(path) for path in changed)):
                say(f"binaries of {commit[:9]} reused: it differs from {self.short} only in "
                    f"{', '.join(changed)}")
                self.state["binary_commit"] = commit
                self.save()
                return

    # --- stage 1: commit ----------------------------------------------------------------

    def stage_commit(self) -> dict:
        base = pick_commit(self.args.commit)
        current = re.search(r'\[workspace\.package\]\nversion = "([^"]+)"',
                            git("show", f"{base}:Cargo.toml"))[1]
        if version_tuple(self.version) <= version_tuple(current):
            sys.exit(f"{self.version} is not after the workspace's {current}")
        check_launch_tree(base)
        since, since_version = last_release(base)
        say(f"release {self.version}: base {base[:9]} (verified: streams and tests), "
            f"last release {since_version} at {since[:9]}")
        if self.real and git("tag", "-l", f"v{self.version}"):
            sys.exit(f"tag v{self.version} exists")
        if not (self.tree / ".git").exists():
            run(["git", "worktree", "add", "--detach", str(self.tree), base])
        head = git("rev-parse", "HEAD", cwd=self.tree)
        if head != base and git("rev-parse", "HEAD~1", cwd=self.tree) == base:
            say(f"release commit {head[:9]} exists on {base[:9]}: kept")
            release = head
        else:
            if head != base:
                run(["git", "checkout", "-q", "--detach", base], cwd=self.tree)
            release = self.make_release_commit(base, current, since, since_version)
        self.check_release_diff(base, release)
        check_lockfile(self.tree)
        if self.real:
            self.tag_and_land(release)
        return {"base_commit": base, "release_commit": release, "previous": since_version}

    def make_release_commit(self, base: str, current: str, since: str,
                            since_version: str) -> str:
        changelog = self.tree / "CHANGELOG.md"
        if f"## {self.version} (" not in changelog.read_text():
            bump_files(self.tree, current, self.version)
            insert_changelog(self.tree, changelog_draft(self.version, since, since_version, base))
            say(f"drafted CHANGELOG {self.version} in {changelog}")
        if self.real and DRAFT_MARKER in changelog.read_text():
            sys.exit(f"edit the drafted section of {changelog} (and delete its marker line), "
                     f"then run this again")
        # every workspace member's lockfile entry at the new version (bump_files moves only
        # the nanocodec-* packages; 0.3.0 shipped nc-checkasm at 0.2.0 in its first cut)
        run([str(CARGO), "update", "--workspace", "--offline"], cwd=self.tree, capture=True)
        check_lockfile(self.tree)
        run(["git", "add", "--", *RELEASE_FILES], cwd=self.tree)
        run(["git", "commit", "-q", "-m",
             f"release {self.version}: workspace version, lockfile, crate README, CHANGELOG"],
            cwd=self.tree)
        return git("rev-parse", "HEAD", cwd=self.tree)

    def check_release_diff(self, base: str, release: str) -> None:
        changed = set(git("diff", "--name-only", base, release).split())
        extra = changed - set(RELEASE_FILES)
        if extra or git("rev-parse", f"{release}~1") != base:
            sys.exit(f"the release commit must be one commit on {base[:9]} touching only "
                     f"{', '.join(RELEASE_FILES)}; it touches {sorted(extra)}")

    def tag_and_land(self, release: str) -> None:
        """--real: branch and tag the release commit; land the same change on the
        integration branch."""
        branch, tag = f"release/v{self.version}", f"v{self.version}"
        if not git("branch", "--list", branch):
            run(["git", "branch", branch, release])
        if not git("tag", "-l", tag):
            run(["git", "tag", "-a", tag, "-m", f"nanocodec {self.version}", release])
        if self.state.get("landed"):
            return
        lander = WORKTREES / f"release-land-v{self.version}"
        land_branch = f"release-land/v{self.version}"
        if not (lander / ".git").exists():
            run(["git", "worktree", "add", "-b", land_branch, str(lander), INTEGRATION])
        if git("log", "-1", "--format=%s", cwd=lander).startswith(f"release {self.version}:"):
            say("the release change is already on the landing branch")
        else:
            run(["git", "cherry-pick", release], cwd=lander)
        run(["tools/dev/land.sh"], cwd=lander, log=self.log("land-release"))
        self.state["landed"] = git("rev-parse", "HEAD", cwd=lander)
        self.save()

    # --- stage 2: PGO builds and their checks --------------------------------------------

    def stage_pgo(self) -> dict:
        prepare_training_clips()
        with ThreadPoolExecutor(2) as pool:
            jobs = {"x86_64": pool.submit(self.build_x86), "aarch64": pool.submit(self.build_aarch64)}
            results = {arch: job.result() for arch, job in jobs.items()}
        if self.real:
            self.place_chart_builds()
        return results

    def training_env(self) -> str:
        clips = " ".join(f"$HOME/nanocodec-data/release/pgo-clips/{c}.yuv" for c in TRAINING_CLIPS)
        codecs = f" PGO_CODECS={PIPE_CLEAN_CODECS}" if self.pipe_clean else ""
        return f'PGO_CAPI=1 PGO_FRAMES={PGO_FRAMES} PGO_CLIPS="{clips}"{codecs}'

    def build_x86(self) -> dict:
        """The x86_64 PGO and plain builds on the remote box (cached), then the
        identity check here."""
        out = self.builds("x86_64")
        began = time.monotonic()
        if not self.cached(out):
            # this checkout's pgo_build.sh (the release commit's may predate PGO_CAPI)
            tool = self.root / "pgo_build.sh"
            shutil.copy2(REPO / "tools/dev/pgo_build.sh", tool)
            unpack = " && ".join(
                f"zstd -d -q -f ~/{(TRAIN_ZST_DIR / c).relative_to(Path.home())}.yuv.zst "
                f"-o ~/{(TRAIN_DIR / c).relative_to(Path.home())}.yuv" for c in TRAINING_CLIPS)
            command = (
                f"mkdir -p ~/{TRAIN_DIR.relative_to(Path.home())} && {unpack} && "
                f"~/.cargo/bin/rustup component add llvm-tools >/dev/null && "
                f"cp ~/{tool.relative_to(Path.home())} tools/dev/pgo_build.sh && "
                f"mkdir -p {REMOTE_OUT}/pgo {REMOTE_OUT}/plain && "
                f"{self.training_env()} tools/dev/pgo_build.sh . target/release-pgo "
                f"{REMOTE_OUT}/pgo/nanocodec && "
                f"CARGO_TARGET_DIR=target/release-plain cargo build -q --release --bin nanocodec && "
                f"cp target/release-plain/release/nanocodec {REMOTE_OUT}/plain/nanocodec")
            run([str(NC_REMOTE), "run", "--worktree", str(self.tree), "--name", REMOTE_JOB,
                 "--sync", str(TRAIN_ZST_DIR), "--sync", str(tool), "--fetch", REMOTE_OUT,
                 "--", command],
                log=self.log("pgo-x86_64"))
            self.collect(self.tree / REMOTE_OUT / "pgo", self.tree / REMOTE_OUT / "plain", out)
        build_minutes = (time.monotonic() - began) / 60
        self.check_version(run([str(out / "nanocodec"), "--version"], capture=True))
        began = time.monotonic()
        identity = self.identity_x86(out)
        return {"build_minutes": round(build_minutes, 1), "identity": identity,
                "identity_minutes": round((time.monotonic() - began) / 60, 1),
                "sha256": {a: sha256(out / a) for a in ARTIFACTS}}

    def cached(self, out: Path) -> bool:
        """A complete build directory of this commit (its CLI names the commit)."""
        binary = out / "nanocodec"
        if not (binary.is_file() and (out / "plain" / "nanocodec").is_file()
                and all((out / a).is_file() for a in ARTIFACTS)):
            return False
        if out.parent.name.endswith("-pipeclean") != self.pipe_clean:
            return False
        try:
            self.check_version(run([str(binary), "--version"], capture=True))
        except SystemExit:
            return False
        say(f"{out}: cached")
        return True

    def collect(self, pgo: Path, plain: Path, out: Path) -> None:
        (out / "plain").mkdir(parents=True, exist_ok=True)
        for artifact in ARTIFACTS:
            shutil.copy2(pgo / artifact, out / artifact)
        shutil.copy2(plain / "nanocodec", out / "plain" / "nanocodec")

    def check_version(self, line: str) -> None:
        if line.strip() != self.version_line():
            raise SystemExit(f"build says {line.strip()!r}, not {self.version_line()!r}")

    def identity_x86(self, out: Path) -> str:
        """Every shipped preset: PGO and plain builds code identical streams (PC clips).
        The release tree's tool: it lists the presets of its own tree (a launch tree's
        would check that tree's presets, not the release's)."""
        log = self.log("identity-x86_64")
        run([*MEMORY_SCOPE, "python3", str(self.tree / "tools/dev/preset_identity.py"),
             str(out / "plain" / "nanocodec"), str(out / "nanocodec"), "--jobs", "8"], log=log)
        return log.read_text().strip().splitlines()[-1]

    def build_aarch64(self) -> dict:
        """The aarch64 PGO and plain builds on the Jetson (cached), the identity check
        and the PGO-over-plain cycle count there."""
        out = self.builds("aarch64")
        remote = f"{JETSON_ROOT}/{self.build_key}"
        host = jetson_host()
        began = time.monotonic()
        if not self.cached_aarch64(out):
            if not self.built_on_jetson(remote):
                self.build_on_jetson(remote, host)
            fetched = self.root / "fetch-aarch64"
            fetched.mkdir(exist_ok=True)
            run(["rsync", "-a", f"{host}:{remote}/pgo/", f"{fetched}/pgo/"])
            run(["rsync", "-a", f"{host}:{remote}/plain/", f"{fetched}/plain/"])
            self.collect(fetched / "pgo", fetched / "plain", out)
            shutil.rmtree(fetched)
        build_minutes = (time.monotonic() - began) / 60
        self.check_version(jetson(f"~/{remote}/pgo/nanocodec --version"))
        began = time.monotonic()
        log = self.log("identity-aarch64")
        # --local: the board runs its own identity (preset_identity goes to the build box by
        # default since 2026-11-21, which the board has no client for)
        jetson(f"cd ~/{remote}/src && {JETSON_BUILD} nice -n 19 python3 tools/dev/preset_identity.py "
               f"~/{remote}/plain/nanocodec ~/{remote}/pgo/nanocodec "
               f"--clip-dir {JETSON_IDENTITY_CLIPS} --jobs {JETSON_JOBS} --local", log=log)
        identity = log.read_text().strip().splitlines()[-1]
        identity_minutes = (time.monotonic() - began) / 60
        return {"build_minutes": round(build_minutes, 1), "identity": identity,
                "identity_minutes": round(identity_minutes, 1),
                "cycles": self.jetson_cycles(remote),
                "sha256": {a: sha256(out / a) for a in ARTIFACTS}}

    def built_on_jetson(self, remote: str) -> bool:
        """Both builds of this commit already on the board (a resumed stage)."""
        names = " ".join(f"~/{remote}/pgo/{a}" for a in ARTIFACTS)
        lines = jetson(f"ls {names} >/dev/null 2>&1 && ~/{remote}/pgo/nanocodec --version && "
                       f"~/{remote}/plain/nanocodec --version || true").splitlines()
        done = lines == [self.version_line()] * 2
        if done:
            say(f"Jetson builds of {self.short} exist: kept")
        return done

    def build_on_jetson(self, remote: str, host: str) -> None:
        self.jetson_sources(remote)
        jetson("mkdir -p ~/nanocodec-data/release/pgo-clips")
        run(["rsync", "-a", f"{TRAIN_DIR}/", f"{host}:nanocodec-data/release/pgo-clips/"])
        build = (
            f"cd ~/{remote}/src && ~/.cargo/bin/rustup component add llvm-tools >/dev/null && "
            f"mkdir -p ~/{remote}/pgo ~/{remote}/plain && "
            f"{JETSON_BUILD} nice -n 19 env JOBS={JETSON_JOBS} NC_BUILD_COMMIT={self.commit} "
            f"{self.training_env()} bash tools/dev/pgo_build.sh . ~/{JETSON_TARGETS}/pgo "
            f"~/{remote}/pgo/nanocodec && "
            f"{JETSON_BUILD} nice -n 19 env NC_BUILD_COMMIT={self.commit} "
            f"~/.cargo/bin/cargo build -q --release -j{JETSON_JOBS} --bin nanocodec "
            f"--target-dir ~/{JETSON_TARGETS}/plain && "
            f"cp ~/{JETSON_TARGETS}/plain/release/nanocodec ~/{remote}/plain/nanocodec")
        jetson(build, log=self.log("pgo-aarch64"))

    def cached_aarch64(self, out: Path) -> bool:
        return (all((out / a).is_file() for a in ARTIFACTS)
                and (out / "plain" / "nanocodec").is_file()
                and out.parent.name.endswith("-pipeclean") == self.pipe_clean)

    def jetson_sources(self, remote: str) -> None:
        """The release commit's build inputs unpacked on the board (git archive), with
        the release tools of this checkout and the evaluation library they import (the
        release commit may predate them: 2026-11-21's dry run on an older base shipped
        today's preset_identity.py beside the base's nceval, without nceval.tempdirs)."""
        sources = [p for p in JETSON_SOURCES if not p.startswith(LAUNCH_TREE_SOURCES)
                   and subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                                       f"{self.commit}:{p}"], capture_output=True).returncode == 0]
        archive = self.root / "aarch64-sources.tar"
        run(["git", "archive", "--format=tar", "-o", str(archive), self.commit, *sources])
        with tarfile.open(archive, mode="a") as tar:
            for tool in (p for p in JETSON_SOURCES if p.startswith(LAUNCH_TREE_SOURCES)):
                tar.add(REPO / tool, arcname=tool,
                        filter=lambda info: None if "__pycache__" in info.name else info)
        with archive.open("rb") as stream:
            subprocess.run(["ssh", "-o", "BatchMode=yes", jetson_host(),
                            f"mkdir -p ~/{remote}/src && tar -x -C ~/{remote}/src"],
                           stdin=stream, check=True)
        archive.unlink()

    def jetson_cycles(self, remote: str) -> str:
        """PGO over plain on the Jetson's timing core (jperf: min of rounds, one preset)."""
        jetson(f"cd ~/{remote} && ln -sf plain/nanocodec nanocodec-plain && "
               f"ln -sf pgo/nanocodec nanocodec-pgo")
        log = self.log("cycles-aarch64")
        jetson(f"cd ~/{remote} && {JETSON_TIMED} python3 src/tools/dev/jperf.py "
               f"--rounds {CYCLES_ROUNDS} --input {JETSON_IDENTITY_CLIPS}/{CYCLES_CLIP}.yuv "
               f"nanocodec-plain nanocodec-pgo -- {CYCLES_ARGS}", log=log)
        return " | ".join(line.strip() for line in log.read_text().splitlines()
                          if "nanocodec-" in line and "instr" in line)

    def place_chart_builds(self) -> None:
        """--real: the PGO builds where chart_refresh --pgo times them."""
        target = CHART_BUILDS / self.binary_short
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.builds("x86_64") / "nanocodec", target / PGO_TIMED_BINARY)
        jetson(f"mkdir -p ~/{JETSON_CHART_BUILDS}/{self.binary_short} && cp ~/{JETSON_ROOT}/{self.build_key}"
               f"/pgo/nanocodec ~/{JETSON_CHART_BUILDS}/{self.binary_short}/{JETSON_PGO_BINARY}")

    # --- stage 3: charts ------------------------------------------------------------------

    def stage_charts(self) -> dict:
        if self.pipe_clean or not self.real:
            return self.charts_dry()
        return self.charts_real()

    def our_families(self, tree: Path) -> list[str]:
        """Every shipped single-layer preset of `tree` (nceval.plotted), so a preset
        the chart record does not have yet is measured, not left out."""
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.encoders.nanocodec import presets_in
        from nceval.plotted import own_names
        return own_names(presets_in(tree))

    def chart_commands(self, tree: Path) -> list[tuple[str, list[str]]]:
        """(log name, command) of a release's whole chart refresh, run in `tree`."""
        families = ",".join(self.our_families(tree))
        ours = [f for f in self.lens_encoders() if f.startswith("nanocodec-")]
        refresh = ["python3", "tools/dev/chart_refresh.py", self.binary_short]
        main_refresh = [*refresh, *(["--local"] if self.args.local_charts else
                                    ["--cloud-regions", MAIN_CLOUD_REGIONS])]
        record = json.loads((tree / BUILDS_RECORD).read_text())
        base = CHART_BUILDS / record["base"]
        # the zero lines first, with XPSNR (a cached point lacking it is scored again), so
        # every chart renders once, with them. Main chart: every family on its recorded
        # build, as chart_refresh's measure gives them (low rungs are judged on every point)
        builds = "".join(f" --binary-for {f}={CHART_BUILDS / b}/nanocodec --presets-for "
                         f"{f}={CHART_BUILDS / b}/presets"
                         for f, b in sorted(record["families"].items()))
        if self.args.local_charts:  # the proven local commands (0.5.0), unchanged
            commands = [("zero-line-xpsnr", [
                "bash", "-c", f"cd eval && python3 -m nceval.chart measure --binary "
                f"{base}/nanocodec --presets-dir {base}/presets{builds} --points "
                f"{','.join(ZERO_LINES)} --workers 8 --over-budget"])]
        else:
            # on rented machines, from the release commit (MAIN_BEFORE_PGO): its scoring
            # build on the build box (build_cache: the plain gate build codes the PGO
            # build's streams), then every point of the chart, the references and zero
            # lines with ours (chart_refresh measures the main chart on boxes); the
            # refresh after the PGO builds finds them cached and only scans and times
            commands = [("main-build", ["bash", "tools/dev/binary_for.sh", self.binary_short]),
                        ("main-quality", [*main_refresh, "--families", families,
                                          "--metrics-only"])]
        commands.append(("chart-refresh", [*main_refresh, "--families", families, "--pgo"]))
        # each clip-set chart (every site chart view's, chart_refresh.clipset_chains): its
        # zero lines coded where missing and scored on rented machines (cached: no
        # machine), then the refresh as chart_refresh --heldout runs it (refresh_heldout),
        # one command each so that a resumed stage does not redo a finished one
        heldout_tool = ["python3", "tools/dev/aws_heldout_chart.py"]
        session = ["--budget", str(ZERO_LINE_BUDGET), "--lifetime", str(ZERO_LINE_LIFETIME)]
        for clipset in [path for chain in clip_chains() for path in chain]:
            zero = ["--clipset", str(clipset), "--families", ",".join(ZERO_LINES),
                    "--build", str(CHART_BUILDS / self.binary_short),
                    "--streams", str(HELDOUT_STREAMS).format(stem=clipset.stem), *session]
            held_out = [*refresh, "--clipset", str(clipset), "--families", families,
                        "--cloud-regions",
                        CLIPSET_CLOUD_REGIONS.get(clipset.stem, HELDOUT_CLOUD_REGIONS)]
            name = f"heldout-{clipset.stem}"
            commands += [
                (f"{name}-zero-line", [*heldout_tool, "run", *zero,
                                       "--purpose", "release-zero-line"]),
                (f"{name}-zero-line-score", [*heldout_tool, "score-cloud", *zero,
                                             "--purpose", "release-zero-line-score"]),
                (name, [*held_out, "--metrics-only"]),
                (f"{name}-psnr", [*held_out, "--from", "render",
                                  "--anchor", HELDOUT_TUNED_ANCHOR])]
        x86 = self.builds("x86_64") / "nanocodec"
        arm = self.builds("aarch64") / "nanocodec"
        for machine, (instance, record_file, wording) in CBR_SERIES.items():
            binary = ["--binary", str(arm), "--tools-dir", str(AARCH64_TOOLS),
                      "--expect-version", self.version_line()] if machine == "graviton" \
                else ["--binary", str(x86)]
            out = self.root / f"cbr-{machine}"
            commands.append((f"cbr-{machine}", [
                "python3", "tools/dev/aws_timing.py", "--instance-type", instance, *binary,
                "--repeats", "1", "--points", families, "--purpose", "release-cbr-timing",
                "--out", str(out)]))
            commands.append((f"cbr-{machine}-merge", [
                "bash", "-c", f"python3 tools/dev/anchor_chart_timing.py merge --machine "
                f"{shlex.quote(wording)} --build {self.binary_short} --update --out {RESULTS}/{record_file} "
                f"{out}/repeat-1/t-*.json"]))
        for machine, record_file in LENS_MACHINES.items():
            commands += self.lens_commands(machine, [*ours, *LENS_CHECK], ours,
                                           f"{RESULTS}/{record_file}")
        for view in qp_views():
            commands += self.qp_commands(tree, view, self.our_families(tree))
        commands += self.lens_quality_commands(tree, self.our_families(tree))
        return commands

    def lens_quality_commands(self, tree: Path, ours: list[str]) -> list[tuple[str, list[str]]]:
        """The Fixed QP lens's quality (the overview's headline chart) of every row of ours
        on the release's x86 PGO build: each (encoder, column) variant coded by `nceval.ctc
        local` on the build box, LENS_BOX_JOBS jobs side by side (one slot each, 8 workers),
        then compared here and written into the lens data and its CSV (nceval.ctc
        lens-rows). Every row of ours, not only changed ones: the rows were on five builds."""
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.ctc.lensrows import variants
        data = tree / LENS_SITE_DATA
        runs = variants(json.loads(data.read_text()), ours)
        binaries = self.builds("x86_64")
        on_box = f"$HOME/{binaries.relative_to(Path.home())}"
        results = []
        jobs = []
        for index in range(LENS_BOX_JOBS):
            mine = runs[index::LENS_BOX_JOBS]
            if not mine:
                continue
            steps, fetch = [], []
            for run_spec in mine:
                run_dir = (f"remote-out/lens-rel/{run_spec['encoder'].removeprefix('nanocodec-')}"
                           f"-{run_spec['label']}").replace("@", "-")
                results.append(self.tree / run_dir / "out/results")
                fetch += ["--fetch", f"{run_dir}/out/results"]
                # paths in the box's home: $HOME left unquoted, to expand there
                steps.append(f"(cd eval && NC_NANOCODEC_BIN={on_box} python3 -m nceval.ctc "
                             f"local --source-dir $HOME/{LENS_BOX_SOURCES} " + shlex.join([
                    "--run-dir", f"../{run_dir}", "--encoder", run_spec["encoder"],
                    "--label", run_spec["label"],
                    *(a for param in run_spec["params"] for a in ("--param", param)),
                    "--workers", str(LENS_BOX_WORKERS)]) + ")")
            jobs.append(shlex.join([str(NC_REMOTE), "run", "--worktree", str(self.tree),
                                    "--name", f"release-lens-{index + 1}",
                                    "--sync", str(binaries), *fetch, "--",
                                    " && ".join(steps)]))
        compare = self.root / "lens-compare.json"
        return [
            ("lens-quality-box", ["bash", "-c", all_of(jobs)]),
            ("lens-quality-compare", ["bash", "-c", "cd eval && " + shlex.join([
                "python3", "-m", "nceval.ctc", "compare", "--results",
                ",".join(map(str, results)), "--json", str(compare)])]),
            ("lens-quality-rows", ["bash", "-c", "cd eval && " + shlex.join([
                "python3", "-m", "nceval.ctc", "lens-rows", "--data", f"../{LENS_SITE_DATA}",
                "--compare", str(compare), "--build", self.binary_short])])]

    def qp_commands(self, tree: Path, view, ours: list[str]) -> list[tuple[str, list[str]]]:
        """A Fixed QP sample view (nceval.site.SAMPLE_QP_VIEWS) on the release's x86 PGO
        build: our families' quality by `nceval.fixedqp gap` on this PC, in parts side by
        side (the references' rows are kept: their points are cached and their code is
        unchanged), then their speed on those encodes on c8a (fixedqp_timing.py plan-pool,
        x264 medium beside them as the same-session check), merged into the view's record."""
        record = json.loads((tree / RESULTS / view.data).read_text())
        sys.path.insert(0, str(REPO / "eval"))
        from nceval.config import load_paths
        clipset = load_paths().clips / record["clipset"]
        binary, presets = self.builds("x86_64") / "nanocodec", self.tree / PRESETS
        anchor = record["anchor"]
        parts = [ours[i::QP_PARTS.get(view.id, QP_DEFAULT_PARTS)]
                 for i in range(QP_PARTS.get(view.id, QP_DEFAULT_PARTS))]
        gaps = [self.root / f"{view.id}-gap-{i + 1}.json" for i in range(len(parts))]
        tool = ["python3", "tools/dev/fixedqp_timing.py"]
        plan, out = self.root / f"{view.id}-timing-plan.json", self.root / f"{view.id}-timing"
        scratch = self.root / f"{view.id}-timing-record.json"
        # on a rented box each (tools/dev/aws_measure.py: the clip set's clips, the build,
        # its presets and x264 go up, the result JSON comes back), or here (--local-charts)
        box = ([] if self.args.local_charts else
               ["python3", "../tools/dev/aws_measure.py", "--budget", str(QP_BOX_BUDGET), "--"])
        gap = [shlex.join([*box, "python3", "-m", "nceval.fixedqp", "gap", "--binary", str(binary),
                           "--presets-dir", str(presets), "--clipset", str(clipset),
                           "--clips", str(len(record["clips"])),
                           "--families", ",".join([anchor, *part]),
                           "--pairs", ",".join(f"{anchor}:{f}" for f in part),
                           "--workers", str(QP_WORKERS), "--json", str(json_path)])
               for part, json_path in zip(parts, gaps)]
        timing = view.timing["x86"]
        return [
            (f"{view.id}-gap", ["bash", "-c", all_of(gap, cwd="eval")]),
            (f"{view.id}-site", ["bash", "-c", "cd eval && " + shlex.join([
                "python3", "-m", "nceval.fixedqp", "site",
                *(a for g in gaps for a in ("--gap", str(g))), "--build", self.binary_short,
                "--keep-references", str(tree / RESULTS / view.data),
                "--json", str(tree / RESULTS / view.data)])]),
            (f"{view.id}-timing-plan", [*tool, "plan-pool", "--binary", str(binary),
                                        "--presets-dir", str(presets),
                                        *(a for g in gaps for a in ("--gap", str(g))),
                                        "--out", str(plan)]),
            (f"{view.id}-timing-run", [*tool, "run", "--plan", str(plan), "--out", str(out),
                                       "--parallel", str(QP_TIMING_PARALLEL)]),
            (f"{view.id}-timing-record", [*tool, "record", "--plan", str(plan), "--out",
                                          str(out), "--build", self.binary_short,
                                          "--record", str(scratch)]),
            (f"{view.id}-timing-merge", [*tool, "merge", "--record", f"{RESULTS}/{timing}",
                                         "--from", str(scratch), "--encoders", ",".join(ours)])]

    def lens_env(self) -> dict[str, str]:
        """The lens planner resolves our tool by NC_NANOCODEC_BIN: the x86 PGO build (an
        aarch64 plan names its own binary, which runs only there)."""
        return {"NC_NANOCODEC_BIN": str(self.builds("x86_64"))}

    def lens_encoders(self) -> list[str]:
        sys.path.insert(0, str(REPO / "tools" / "dev"))
        sys.path.insert(0, str(REPO / "eval"))
        fixedqp = load_module("fixedqp_timing", REPO / "tools/dev/fixedqp_timing.py")
        return fixedqp.chart_encoders()

    def lens_commands(self, machine: str, encoders: list[str], merged: list[str] | None,
                      record: str, sequences: str | None = None) -> list[tuple[str, list[str]]]:
        """Plan, run and record our lens encodes on `machine` (and merge `merged` into
        `record`; None: `record` is the output itself)."""
        tool = ["python3", "tools/dev/fixedqp_timing.py"]
        plan, out = self.root / f"lens-{machine}.json", self.root / f"lens-{machine}"
        if machine == "jetson":
            binary = ["--binary", f"~/{JETSON_ROOT}/{self.build_key}/pgo/nanocodec", "--rounds", "1",
                      "--points", jetson_lens_points()]
        elif machine == "graviton":
            binary = ["--binary", str(self.builds("aarch64") / "nanocodec"),
                      "--expect-version", self.version_line()]
        else:
            binary = ["--binary", str(self.builds("x86_64") / "nanocodec")]
        only = ["--sequences", sequences, "--rounds", "1"] if sequences else []
        runner = ["run-jetson"] if machine == "jetson" else ["run"]
        scratch = str(self.root / f"lens-{machine}-record.json") if merged else record
        commands = [
            (f"lens-{machine}-plan", [*tool, "plan", "--machine", machine, *binary,
                                      "--encoders", ",".join(encoders), *only,
                                      "--out", str(plan)]),
            (f"lens-{machine}-run", [*tool, *runner, "--plan", str(plan), "--out", str(out)]),
            (f"lens-{machine}-record", [*tool, "record", "--plan", str(plan), "--out", str(out),
                                        "--build", self.binary_short, "--record", scratch])]
        if merged:
            commands.append((f"lens-{machine}-merge", [*tool, "merge", "--record", record,
                                                       "--from", scratch,
                                                       "--encoders", ",".join(merged)]))
        return commands

    def charts_dry(self) -> dict:
        """The full refresh printed; the pipe-clean run on every machine into this
        release's directory."""
        self.wait_for("pgo")  # the pipe-clean times the PGO builds
        results: dict[str, object] = {}
        began = time.monotonic()
        run(["python3", "tools/dev/chart_refresh.py", self.binary_short, "--families",
             ",".join(self.our_families(self.tree)), "--pgo", "--dry-run"],
            log=self.log("charts-refresh-dry-run"))
        for machine, (instance, _, _) in CBR_SERIES.items():
            arm = machine == "graviton"
            binary = self.builds("aarch64" if arm else "x86_64") / "nanocodec"
            extra = ["--tools-dir", str(AARCH64_TOOLS), "--expect-version",
                     self.version_line()] if arm else []
            run(["python3", "tools/dev/aws_timing.py", "--instance-type", instance,
                 "--binary", str(binary), *extra, "--repeats", "1", "--points",
                 PIPE_CLEAN_FAMILY, "--dry-run"], log=self.log(f"charts-cbr-{machine}-dry-run"))
        results["printed_minutes"] = round((time.monotonic() - began) / 60, 1)
        if not self.real:
            self.stage_jetson_pgo_for_charts()
        try:
            with ThreadPoolExecutor(len(LENS_MACHINES)) as pool:
                jobs = {m: pool.submit(self.lens_pipe_clean, m) for m in LENS_MACHINES}
                jobs["jetson-cbr"] = pool.submit(self.cbr_jetson_pipe_clean)
                for name, job in jobs.items():
                    results[name] = job.result()
        finally:
            if not self.real:  # the dry run leaves no chart build behind on the board
                placed = f"~/{JETSON_CHART_BUILDS}/{self.binary_short}"
                jetson(f"rm -f -- {placed}/{JETSON_PGO_BINARY} && rmdir --ignore-fail-on-non-empty "
                       f"-- {placed}")
        return results

    def lens_pipe_clean(self, machine: str) -> dict:
        began = time.monotonic()
        record = self.root / f"lens-{machine}-pipe-clean.json"
        for name, command in self.lens_commands(machine, [PIPE_CLEAN_ENCODER], None,
                                                str(record), PIPE_CLEAN_SEQUENCE):
            run(command, log=self.log(f"charts-{name}"), env=self.lens_env())
        point = json.loads(record.read_text())["points"][PIPE_CLEAN_ENCODER]
        return {"minutes": round((time.monotonic() - began) / 60, 1),
                "ms_per_frame": point["ms_per_frame"]}

    def stage_jetson_pgo_for_charts(self) -> None:
        """The dry run's Jetson CBR pipe-clean times the PGO build where chart_refresh
        --pgo looks for it (no COMMIT marker: no tool takes it for a chart build)."""
        jetson(f"mkdir -p ~/{JETSON_CHART_BUILDS}/{self.binary_short} && cp ~/{JETSON_ROOT}/{self.build_key}"
               f"/pgo/nanocodec ~/{JETSON_CHART_BUILDS}/{self.binary_short}/{JETSON_PGO_BINARY}")

    def cbr_jetson_pipe_clean(self) -> dict:
        """chart_refresh's own Jetson timing of one family on the PGO build (its stream
        check included), kept in this release's directory instead of merged."""
        began = time.monotonic()
        refresh_tool = load_module("chart_refresh", REPO / "tools/dev/chart_refresh.py")
        args = argparse.Namespace(commit=self.binary_short, families=PIPE_CLEAN_FAMILY, retime="",
                                  time_anchors="", clipset=None, seed=False, pgo=True,
                                  dry_run=False, metrics_only=False)
        refresh = refresh_tool.Refresh(args)
        refresh.timed = [PIPE_CLEAN_FAMILY]
        out = self.root / "cbr-jetson-pipe-clean"
        refresh.jetson_records(out)
        record = json.loads((out / f"t-{PIPE_CLEAN_FAMILY}.json").read_text())
        return {"minutes": round((time.monotonic() - began) / 60, 1),
                "ms_per_frame": {c: v["ms_per_frame"]["point"]
                                 for c, v in record["clips"].items()}}

    def charts_real(self) -> dict:
        """--real: the whole refresh in a worktree of the integration head, landed."""
        tree = Path(str(CHART_TREE).format(version=self.version))
        branch = f"release-charts/v{self.version}"
        if not (tree / ".git").exists():
            run(["git", "worktree", "add", "-b", branch, str(tree), INTEGRATION])
        elif git("branch", "--show-current", cwd=tree) != branch:
            sys.exit(f"{tree} is not on {branch}: remove it or check it out")
        (tree / RELEASE_RECORD).write_text(json.dumps(
            {"version": self.version, "commit": self.binary_short, "build": "PGO",
             # the day the charts were measured: the site's captions say it
             "date": datetime.date.today().isoformat()}, indent=1) + "\n")
        # a resumed stage skips the commands it finished (each one re-times or rents
        # machines); the list is cleared when the stage is done
        with self.state_lock:
            timings = self.state.setdefault("charts_progress", {})
        chains: dict[str, list[tuple[str, list[str]]]] = {}
        for name, command in self.chart_commands(tree):
            chains.setdefault(chart_chain(name), []).append((name, command))

        # each machine's timing is its own rented session (or the Jetson's lock), and the
        # quality refresh runs on this PC and its own sessions: the stage takes as long
        # as its slowest chain (0.3.0 ran them one after another: ~2.5 h of timing)
        with ThreadPoolExecutor(len(chains)) as pool:
            jobs = {chain: pool.submit(self.run_chart_chain, chain, commands, tree, timings)
                    for chain, commands in chains.items()}
        failed = [chain for chain, job in jobs.items() if job.exception()]
        if failed:
            sys.exit(f"chart chain(s) failed: {', '.join(failed)} (each logged when it "
                     f"failed); resume with --from charts")
        run(["python3", "-m", "nceval.site"], cwd=tree / "eval", log=self.log("charts-site"))
        run(["git", "add", "--", RESULTS, "web/results/data", "README.md"], cwd=tree)
        run(["git", "commit", "-q", "-m", f"charts: release {self.version} (PGO builds of "
             f"{self.binary_short}) on every chart machine"], cwd=tree)
        run(["tools/dev/land.sh"], cwd=tree, log=self.log("charts-land"))
        del self.state["charts_progress"]
        return {"minutes": timings}

    def run_chart_chain(self, chain: str, commands: list[tuple[str, list[str]]], tree: Path,
                        timings: dict[str, float]) -> None:
        """One chain's commands in order, each finished one recorded (a resume skips it)."""
        for name, command in commands:
            if not chain.startswith(QUALITY_CHAIN_PREFIX) and name not in MAIN_BEFORE_PGO:
                self.wait_for("pgo")  # it times the PGO builds (or codes with them)
            if name in timings:
                say(f"charts: {name} done earlier ({timings[name]} min): skipped")
                continue
            began = time.monotonic()
            self.run_chart_command(chain, name, command, tree)
            with self.state_lock:
                timings[name] = round((time.monotonic() - began) / 60, 1)
                self.save()

    def run_chart_command(self, chain: str, name: str, command: list[str], tree: Path) -> None:
        """A chart command; its failure is announced at once in the run log and on stderr
        (2026-11-20: a chain that died at 14:17 was reported at 16:28, when the last
        chain ended), then the command is retried once (unless --no-chain-retry)."""
        log = self.log(f"charts-{name}")
        try:
            run(command, cwd=tree, log=log, env=self.lens_env())
            return
        except SystemExit:
            alarm(f"== charts chain {chain} FAILED: {log}")
            if self.args.no_chain_retry:
                raise
        retry = self.log(f"charts-{name}-retry")
        say(f"== charts chain {chain}: {name} retried once in {CHAIN_RETRY_WAIT_S} s "
            f"[{retry}]")
        time.sleep(CHAIN_RETRY_WAIT_S)
        try:
            run(command, cwd=tree, log=retry, env=self.lens_env())
        except SystemExit:
            alarm(f"== charts chain {chain} FAILED (after its retry): {retry}")
            raise
        say(f"== charts chain {chain}: {name} passed on its retry")

    def charts_work_gib(self) -> float:
        """The work/ headroom the charts stage still needs: CHARTS_WORK_HEADROOM_GIB
        shared by the chains' minutes, counting only chains with a data-writing command
        left (LIGHT_CHART_COMMANDS write next to nothing). A first run needs all of it;
        a resume with only renders, the site and the landing left, none."""
        progress = self.state.get("charts_progress")
        tree = Path(str(CHART_TREE).format(version=self.version))
        if not progress or not (tree / ".git").exists():
            return CHARTS_WORK_HEADROOM_GIB
        heavy_left = {chart_chain(name) for name, _ in self.chart_commands(tree)
                      if name not in progress and not name.endswith(LIGHT_CHART_COMMANDS)}
        minutes = chain_minutes()
        return CHARTS_WORK_HEADROOM_GIB * sum(minutes.get(c, 0) for c in heavy_left) \
            / sum(minutes.values())

    # --- stage 4: crates ------------------------------------------------------------------

    def stage_crates(self) -> dict:
        """The release tree's own script: it publishes the workspace it sits in (2026-11-20:
        the launch tree's script published the launch tree, at 0.4.0)."""
        script = "tools/release/publish_crates.sh"
        check_tree_version(self.tree, self.version)
        if self.publish:
            # the real upload runs here, with this PC's `cargo login`, from the clean tag
            run(["bash", str(self.tree / script), "--yes-publish"], cwd=self.tree,
                log=self.log("crates-publish"))
            return {"published": True}
        # the dry run packages and builds every crate: on the build box (no .git there,
        # so cargo sees no dirty files)
        run([str(NC_REMOTE), "run", "--worktree", str(self.tree), "--name", REMOTE_JOB,
             "--", f"bash {script}"], log=self.log("crates-dry-run"))
        return {"published": False, "dry_run": "passed"}

    # --- stage 5: GitHub release ----------------------------------------------------------

    def stage_github(self) -> dict:
        out = self.root / "github"
        out.mkdir(exist_ok=True)
        files = []
        header = self.tree / CAPI_HEADER
        for arch in ARCHES:
            name = f"nanocodec-{self.version}-{arch}-linux"
            tarball = out / f"{name}.tar.gz"
            with tarfile.open(tarball, "w:gz") as tar:
                for artifact in ARTIFACTS:
                    tar.add(self.builds(arch) / artifact, arcname=f"{name}/{artifact}")
                tar.add(header, arcname=f"{name}/include/nanocodec.h")
            files.append(tarball)
        sums = out / "SHA256SUMS"
        sums.write_text("".join(f"{sha256(f)}  {f.name}\n" for f in files))
        notes = out / "notes.md"
        section = changelog_section((self.tree / "CHANGELOG.md").read_text(), self.version)
        notes.write_text(section + "\n" + build_notes(self.version) + self.binary_note() + "\n")
        check_tree_version(self.tree, self.version)
        public = self.public_tag_commit()
        tag = f"v{self.version}"
        commands = {"push": ["git", "push", "origin", f"{public}:refs/tags/{tag}"],
                    "release": ["gh", "release", "create", tag, "--repo", GH_REPO,
                                "--verify-tag", "--title", f"nanocodec {self.version}",
                                "--notes-file", str(notes), *map(str, files), str(sums)]}
        script = out / "upload.sh"
        script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                          + "".join(shlex.join(c) + "\n" for c in commands.values()))
        if self.publish:
            # idempotent (2026-11-20: a resume pushed a second v0.5.0 tag object, refused):
            # a tag already there on the same public tree, and its release, are kept
            remote = remote_tag_commit(tag)
            todo = github_actions(remote and git("rev-parse", f"{remote}^{{tree}}"),
                                  git("rev-parse", f"{public}^{{tree}}"),
                                  remote is not None and github_release_exists(tag))
            if not todo:
                say(f"github: {tag} and its release exist at {remote[:9]} (the same tree): "
                    f"kept")
            for action in todo:
                run(commands[action], log=self.log("github"))
            public = remote or public
        return {"files": [f.name for f in files], "public_commit": public,
                "uploaded": self.publish, "script": str(script)}

    def binary_note(self) -> str:
        """Where the binaries come from when not built from the tagged commit."""
        if self.binary_commit == self.commit:
            return ""
        changed = ", ".join(git("diff", "--name-only", self.binary_commit, self.commit).split())
        return (f" The binaries were built from {self.binary_short} (`nanocodec --version` "
                f"names it), which differs from the tagged commit {self.short} only in "
                f"{changed}: no source change, and every shipped preset codes identical "
                f"streams.")

    def public_tag_commit(self) -> str:
        """The public commit the tag names: the release tree minus the internal docs and
        held-back results, parented on the public head (reachable from the tag only)."""
        publish = load_module("publish_public", PUBLISH_PUBLIC)  # its git runs in REPO (main)
        tree = publish.public_tree(self.commit)
        publish.assert_clean(tree, publish.excluded_paths(self.commit))
        parent = git("rev-parse", PUBLIC_REF)
        # no raw video, no large new blob outside the published media (2026-11-21: rec.yuv)
        publish.assert_no_raw_or_large(tree, f"{parent}^{{tree}}")
        message = (f"nanocodec {self.version}\n\nRelease commit {self.commit[:9]}, "
                   f"published tree.\n\n{publish.TRAILER}\n")
        # dated as the release commit, so a rerun makes the same commit object
        date = git("log", "-1", "--format=%cI", self.commit)
        result = subprocess.run(["git", "-C", str(REPO), "commit-tree", tree, "-p", parent,
                                 "-m", message], capture_output=True, text=True, check=True,
                                env={**os.environ, "GIT_AUTHOR_DATE": date,
                                     "GIT_COMMITTER_DATE": date})
        return result.stdout.strip()

    # --- stage 6: mirror and site -----------------------------------------------------------

    def stage_publish(self) -> dict:
        if not self.real:
            run(["python3", str(PUBLISH_PUBLIC), "--dry-run"], cwd=REPO,
                log=self.log("publish-dry-run"))
            return {"mirrored": False}
        run(["git", "update-ref", "refs/heads/overnight", f"refs/heads/{INTEGRATION}"])
        run(["python3", str(OVERNIGHT / "mirror.py")], log=self.log("publish"))
        run(["python3", str(PUBLISH_PUBLIC), "--internal", "refs/heads/overnight"],
            log=self.log("publish"))
        run(["tools/web/publish.sh", f"release {self.version}"], cwd=DEMO_PUBLISH_TREE,
            env={"PUBLISH_BRANCH": INTEGRATION}, log=self.log("publish-site"))
        return {"mirrored": True}

    # --- driver ---------------------------------------------------------------------------

    def stage_estimate(self, stage: str) -> float:
        """Expected wall minutes of `stage` in this mode, once the stages it needs are
        done (charts: its own estimate, chart_finish)."""
        if stage == "crates" and self.publish:
            return CRATES_UPLOAD_MINUTES
        return STAGE_MINUTES[stage]

    def chart_finish(self, commit_done: float, pgo_done: float) -> float:
        """When the charts stage ends, given when the commit and pgo stages end: its
        quality chains start with the commit, its timing chains with the PGO builds;
        then the site, commit and landing."""
        if self.pipe_clean or not self.real:
            return pgo_done + DRY_CHARTS_MINUTES
        # a chain with a finished command counts as done: a conservative resume estimate
        # would need each command's minutes, and a resume is rare
        started = {chart_chain(name) for name in self.state.get("charts_progress", {})}
        ends = [(commit_done if chain.startswith(QUALITY_CHAIN_PREFIX) else pgo_done) + minutes
                for chain, minutes in chain_minutes().items() if chain not in started]
        return max(ends, default=pgo_done) + CHARTS_LAND_MINUTES

    def plan(self, stages: list[str]) -> list[str]:
        """The stages to run, printed with when each is expected to end (independent
        stages side by side, STAGE_NEEDS); refused when the whole run passes
        MAX_RELEASE_MINUTES (unless --over-budget)."""
        todo = [s for s in stages if not (self.state["stages"].get(s) and not self.args.start
                                          and not self.args.only)]
        ends: dict[str, float] = {}
        for stage in STAGES:  # in order: every stage's needs come before it
            if stage not in todo:
                ends[stage] = 0.0
            elif stage == "charts":
                ends[stage] = self.chart_finish(ends["commit"], ends["pgo"])
            else:
                ends[stage] = (max((ends[need] for need in STAGE_NEEDS[stage]), default=0.0)
                               + self.stage_estimate(stage))
        for stage in stages:
            say(f"plan: {stage:8s} " + (f"ends at ~{ends[stage]:g} min" if stage in todo
                                         else "done earlier"))
        if self.real and "charts" in todo and (
                problems := charts_preflight(headroom_gib=self.charts_work_gib())):
            sys.exit("charts pre-flight failed:\n" + "\n".join(f"  {p}" for p in problems))
        total = max(ends.values())
        say(f"plan: ~{total:g} min in all (limit {MAX_RELEASE_MINUTES})")
        if total > MAX_RELEASE_MINUTES and not self.args.over_budget:
            sys.exit(f"expected ~{total:g} min, over the {MAX_RELEASE_MINUTES}-minute limit: "
                     f"cut the work, or pass --over-budget")
        return todo

    def wait_for(self, stage: str) -> None:
        """Block until `stage` is done: in this run, or recorded by an earlier one."""
        if stage in self.running:
            self.running[stage].wait()
            if stage in self.failed:
                sys.exit(f"{stage} failed")
        elif stage not in self.state["stages"]:
            sys.exit(f"{stage} has not run: run it first")

    def run_stage(self, stage: str) -> None:
        """One stage, once the stages it needs are done; its event set however it ends."""
        try:
            for need in STAGE_NEEDS[stage]:
                self.wait_for(need)
            if stage != "commit" and "release_commit" not in self.state:
                sys.exit("no release commit yet: run the commit stage first")
            if stage != "commit":
                with self.state_lock:
                    self.reuse_binaries()
            say(f"== {stage}")
            began = time.monotonic()
            result = getattr(self, f"stage_{stage}")()
            minutes = round((time.monotonic() - began) / 60, 1)
            with self.state_lock:
                if stage == "commit":
                    self.state.update(base_commit=result["base_commit"],
                                      release_commit=result["release_commit"])
                self.state["stages"][stage] = {"minutes": minutes, **result}
                self.save()
            say(f"== {stage} done in {minutes} min")
        except BaseException:
            self.failed.add(stage)
            raise
        finally:
            self.running[stage].set()

    def execute(self) -> None:
        if "base_commit" in self.state:
            check_launch_tree(self.state["base_commit"])
        start = STAGES.index(self.args.start) if self.args.start else 0
        stages = [self.args.only] if self.args.only else list(STAGES[start:])
        todo = self.plan(stages)
        for stage in stages:
            if stage not in todo:
                say(f"== {stage}: done earlier ({self.state['stages'][stage]['minutes']} min): "
                    f"skipped")
        self.running = {stage: threading.Event() for stage in todo}
        with ThreadPoolExecutor(max(len(todo), 1)) as pool:
            jobs = {stage: pool.submit(self.run_stage, stage) for stage in todo}
        errors = {stage: job.exception() for stage, job in jobs.items() if job.exception()}
        for stage, error in errors.items():
            say(f"== {stage} failed: {error}")
        if errors:
            first = min(errors, key=STAGES.index)
            sys.exit(f"resume with --from {first} (stages done are kept)")
        say(f"state: {self.state_file}")
        print(json.dumps(self.state, indent=1))


#: Files no shipped binary is built from: a release commit differing from built ones only
#: in these reuses their binaries (Release.reuse_binaries).
BINARY_NEUTRAL_FILES = ("Cargo.lock",)


def binary_neutral(path: str) -> bool:
    """Cargo.lock (the binaries were built with the dependencies it pins; a workspace
    member's version line in it is not compiled in) or documentation."""
    return path in BINARY_NEUTRAL_FILES or path.endswith(".md")


def check_launch_tree(base: str, launch: Path = REPO) -> None:
    """The tools of the tree release.py runs from are no older than the code it releases:
    its HEAD contains the release base. (Stages read the release's own files from the
    release tree, never from this one: Release.stage_crates, identity_x86.)"""
    head = git("rev-parse", "HEAD", cwd=launch)
    if subprocess.run(["git", "-C", str(launch), "merge-base", "--is-ancestor", base, head],
                      capture_output=True).returncode:
        sys.exit(f"launch release.py from a tree that contains the release base {base[:9]}: "
                 f"{launch} is at {head[:9]}")


def check_tree_version(tree: Path, version: str) -> None:
    """`tree` is the release's: its workspace version is `version`."""
    found = re.search(r'\[workspace\.package\]\nversion = "([^"]+)"',
                      (tree / "Cargo.toml").read_text())
    if not found or found[1] != version:
        sys.exit(f"{tree} is at version {found[1] if found else '?'}, not {version}: the "
                 f"release's stages run on the release tree")


def remote_tag_commit(tag: str) -> str | None:
    """The commit `tag` names on origin, fetched here (None: no such tag)."""
    listed = git("ls-remote", "origin", f"refs/tags/{tag}")
    if not listed:
        return None
    commit = listed.split()[0]
    if not full_commit(commit):
        run(["git", "fetch", "--no-tags", "origin", f"refs/tags/{tag}"], capture=True)
    return commit


def github_release_exists(tag: str) -> bool:
    return subprocess.run(["gh", "release", "view", tag, "--repo", GH_REPO],
                          capture_output=True).returncode == 0


def github_actions(remote_tree: str | None, public_tree: str, release_exists: bool
                   ) -> list[str]:
    """What the github stage still has to do: push the tag and create the release, or
    the release alone (the tag is there), or nothing; a tag on another tree stops it."""
    if remote_tree is None:
        return ["push", "release"]
    if remote_tree != public_tree:
        sys.exit(f"the remote tag names another tree ({remote_tree[:9]}, the release's is "
                 f"{public_tree[:9]}): a published tag is never moved; check by hand")
    return [] if release_exists else ["release"]


def chart_chain(name: str) -> str:
    """The chain a chart command runs in, in order within it: one per timing machine
    and series (CBR or the Fixed QP lens), one per group of clip-set charts that share
    clips (named after its first), one per Fixed QP sample view, and the main chart."""
    if name.startswith(LENS_QUALITY_CHAIN):
        return LENS_QUALITY_CHAIN
    for view in qp_views():
        if name == view.id or name.startswith(f"{view.id}-"):
            return view.id
    for series in ("cbr", "lens"):
        for machine in (*CBR_SERIES, *LENS_MACHINES):
            if name.startswith(f"{series}-{machine}"):
                return f"{series}-{machine}"
    for chain in clip_chains():
        for clipset in chain:
            if name == f"{QUALITY_CHAIN_PREFIX}{clipset.stem}" or name.startswith(
                    f"{QUALITY_CHAIN_PREFIX}{clipset.stem}-"):
                return f"{QUALITY_CHAIN_PREFIX}{chain[0].stem}"
    return "main"


def all_of(commands: list[str], cwd: str | None = None) -> str:
    """A shell line running `commands` side by side, each in `cwd` if given: it waits for
    every one and fails if any did. The `cd` is its own statement: prefixed as
    `cd DIR && ...` it joined the first command's background job, and the others ran
    outside DIR (2026-11-19: parts 2 and 3 of the qp gaps found no `nceval`)."""
    enter = f"cd {shlex.quote(cwd)} || exit 1; " if cwd else ""
    return (enter + "".join(f"{command} & p{i}=$!; " for i, command in enumerate(commands))
            + "status=0; " + "".join(f"wait $p{i} || status=1; " for i in range(len(commands)))
            + "exit $status")


def chart_tool_paths() -> list[Path]:
    """What the chart chains read outside the repository, as their own modules name it:
    the --bitrate openh264 driver (the CBR charts' openh264-baseline), the lens sources
    and both platforms' anchor tools (fixedqp_timing.py)."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.realuse import OPENH264_HARNESS
    timing = load_module("fixedqp_timing", REPO / "tools/dev/fixedqp_timing.py")
    return [OPENH264_HARNESS, timing.DEFAULT_SOURCES, timing.ANCHOR_TOOLS, timing.AARCH64_TOOLS]


def charts_preflight(work: Path = DATA / "work", tools: list[Path] | None = None,
                     headroom_gib: float = CHARTS_WORK_HEADROOM_GIB) -> list[str]:
    """Why the charts stage would fail part-way, each named; empty when it can start.
    2026-11-19 lost over an hour to all three: the work budget, a pruned openh264 driver
    and a missing lens source directory. The work directory's size is measured afresh
    before it refuses (2026-11-20: the shared cache, up to 30 min old, still said 75 GiB
    after a prune)."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.diskguard import GIB, WORK_BUDGET_BYTES, measure_work_bytes, shared_work_bytes
    problems = [f"missing (or empty): {path}" for path in
                (tools if tools is not None else chart_tool_paths())
                if not path.exists() or (path.is_dir() and not any(path.iterdir()))]
    work.mkdir(parents=True, exist_ok=True)
    headroom = (WORK_BUDGET_BYTES - shared_work_bytes(work)) / GIB
    if headroom < headroom_gib:
        say(f"{work}: {headroom:.0f} GiB left by the shared size cache; measuring afresh")
        headroom = (WORK_BUDGET_BYTES - measure_work_bytes(work)) / GIB
    if headroom < headroom_gib:
        problems.append(f"{work} has {headroom:.0f} GiB left under its "
                        f"{WORK_BUDGET_BYTES / GIB:.0f} GiB budget (need "
                        f"{headroom_gib:.0f}): prune it first")
    return problems


def jetson_lens_points() -> str:
    """`fixedqp_timing.py plan --points` for the Jetson: JETSON_LENS_POINT_POSITIONS of
    every scale (the scales' points are disjoint, so each encoder keeps its own two)."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.ctc.conditions import POINTS
    return ",".join(str(points[i]) for points in POINTS.values()
                    for i in JETSON_LENS_POINT_POSITIONS)


def qp_views() -> tuple:
    """The Fixed QP views on clip-set samples a release re-measures: every one the
    results site has (nceval.site.SAMPLE_QP_VIEWS), published or not."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.site import SAMPLE_QP_VIEWS
    return SAMPLE_QP_VIEWS


@functools.cache
def clip_chains() -> tuple[tuple[Path, ...], ...]:
    """The clip-set charts a release refreshes, in chains (chart_refresh.clipset_chains:
    the results site's chart views, grouped by shared clips)."""
    refresh = load_module("chart_refresh", REPO / "tools/dev/chart_refresh.py")
    return tuple(tuple(chain) for chain in refresh.clipset_chains())


def chain_minutes() -> dict[str, float]:
    """Every chart chain's expected minutes; a chain without an estimate stops the plan
    (a new chart view must be priced before a release runs it)."""
    chains = ["main", *(f"{series}-{machine}" for series, machines in
                        (("cbr", CBR_SERIES), ("lens", LENS_MACHINES)) for machine in machines),
              *(f"{QUALITY_CHAIN_PREFIX}{chain[0].stem}" for chain in clip_chains()),
              *(view.id for view in qp_views()), LENS_QUALITY_CHAIN]
    unpriced = [chain for chain in chains if chain not in CHART_CHAIN_MINUTES]
    if unpriced:
        sys.exit(f"no time estimate for chart chain(s) {', '.join(unpriced)}: add them to "
                 f"CHART_CHAIN_MINUTES")
    return {chain: CHART_CHAIN_MINUTES[chain] for chain in chains}


def build_notes(version: str) -> str:
    return (f"Binaries: the `nanocodec` command-line tool and the C API (`libnanocodec.a`, "
            f"`libnanocodec.so`, `include/nanocodec.h`) for x86_64 and aarch64 Linux, "
            f"profile-guided (PGO) release builds of {version}, the builds the results site's "
            f"speed figures come from. crates.io carries the same version as source; a "
            f"cargo build of it is a plain release build. SHA256SUMS lists the files' hashes.")


def prepare_training_clips() -> None:
    """The public training cuts (TRAINING_CLIPS) as raw I420, fetched once by HTTP range:
    TRAIN_FRAMES frames from TRAIN_START of each sequence."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = TRAIN_DIR / "manifest.json"
    known = json.loads(manifest.read_text()) if manifest.is_file() else {}
    for clip in TRAINING_CLIPS:
        raw = TRAIN_DIR / f"{clip}.yuv"
        if raw.is_file() and known.get(clip) == sha256(raw):
            continue
        url = f"{DERF}/{clip}.y4m"
        header = subprocess.run(["curl", "-fsS", "-A", "nanocodec-release-script", "-r", "0-255",
                                 url], check=True, capture_output=True).stdout
        line = header.split(b"\n", 1)[0]
        fields = line.decode().split()
        size = (int(fields[1][1:]), int(fields[2][1:]))
        colour = next((f for f in fields if f.startswith("C")), "C420jpeg")
        if size != (TRAIN_WIDTH, TRAIN_HEIGHT) or not colour.startswith("C420"):
            sys.exit(f"{clip}: {line!r} is not 1280x720 8-bit 4:2:0")
        frame = len(b"FRAME\n") + TRAIN_FRAME_BYTES
        first = len(line) + 1 + TRAIN_START * frame
        last = first + TRAIN_FRAMES * frame - 1
        data = subprocess.run(["curl", "-fsS", "-A", "nanocodec-release-script", "-r",
                               f"{first}-{last}", url], check=True, capture_output=True).stdout
        with raw.open("wb") as out:
            for index in range(TRAIN_FRAMES):
                chunk = data[index * frame:(index + 1) * frame]
                if not chunk.startswith(b"FRAME"):
                    sys.exit(f"{clip}: no frame marker at frame {TRAIN_START + index}")
                out.write(chunk[chunk.index(b"\n") + 1:])
        known[clip] = sha256(raw)
        say(f"training clip {clip}: {TRAIN_FRAMES} frames")
    manifest.write_text(json.dumps(known, indent=1) + "\n")
    TRAIN_ZST_DIR.mkdir(exist_ok=True)
    for clip in TRAINING_CLIPS:
        packed = TRAIN_ZST_DIR / f"{clip}.yuv.zst"
        if not packed.is_file() or packed.stat().st_mtime < (TRAIN_DIR / f"{clip}.yuv").stat().st_mtime:
            run(["zstd", "-q", "-f", "-9", str(TRAIN_DIR / f"{clip}.yuv"), "-o", str(packed)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="X.Y.Z")
    parser.add_argument("--commit", help="the verified commit to release (default: the "
                        "newest integration commit both verify parts passed)")
    parser.add_argument("--from", dest="start", choices=STAGES,
                        help="redo this stage and every one after it")
    parser.add_argument("--only", choices=STAGES, help="redo this stage alone")
    parser.add_argument("--pipe-clean", action="store_true",
                        help="PGO trained on one codec; stage 3 only its pipe-clean")
    parser.add_argument("--real", action="store_true",
                        help="make the release: tag, landings, charts, mirror (no uploads)")
    parser.add_argument("--over-budget", action="store_true",
                        help=f"run although the plan expects over {MAX_RELEASE_MINUTES} min")
    parser.add_argument("--local-charts", action="store_true",
                        help="measure the main chart and the Fixed QP gaps on this PC, as up "
                             "to 0.5.0 (default: on rented machines)")
    parser.add_argument("--no-chain-retry", action="store_true",
                        help="a failed chart command is not retried (default: once, after "
                             f"{CHAIN_RETRY_WAIT_S} s)")
    parser.add_argument("--yes-publish", action="store_true",
                        help="with --real: upload to crates.io and GitHub (owner's word only)")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        parser.error("VERSION is X.Y.Z")
    if args.yes_publish and not args.real:
        parser.error("--yes-publish needs --real")
    os.chdir(REPO)  # the coordinator's publish_public.py runs git in the working directory
    Release(args).execute()


if __name__ == "__main__":
    main()
