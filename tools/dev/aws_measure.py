#!/usr/bin/env python3
"""One nceval quality measurement on a rented spot box: the same command, scored there.

    python3 tools/dev/aws_measure.py [--instance-type c7i.24xlarge] [--budget 4] [--dry-run] \\
        -- python3 -m nceval.screen --codec av1 --binary BIN --param a.b=1 --json OUT
    ... -- python3 -m nceval.levels final-check --old C0 --new C1 --codecs av1 --json OUT
    ... -- python3 -m nceval.fixedqp ab --codec av1 --binary BIN --param a.b=1 --json OUT
    ... -- python3 -m nceval.rc_oracle --codec av1 --binary BIN --clips quick --json OUT
    ... -- python3 -m nceval.arm_oracle --codec av1 --binary BIN --clips adopt --arm a:p=v --json OUT
    python3 tools/dev/aws_measure.py --batch COMMANDS.json   # several on one box

`--batch` takes a JSON list of such commands (argv lists, each with its own --json),
run side by side on one box: one upload of the union of their clips and builds, and
the points they share (the same anchor) measured once. nceval.train packs a round's
screens this way.

Box size (`--instance-type auto`, the default; coordinator 2026-11-21: an L1 left a
c7a.48xlarge 17 % busy while the account's 512-vCPU spot quota was full): the largest
of AUTO_SIZES that still gives every worker MIN_WAVES points, never above 96 vCPU, then
the same size of the other vendors, then one size down. aws_session waits in line for
quota room (keeping the build box's share) instead of failing over to the PC.

The PC is the bottleneck when many agents screen at once (load 40-50: every screen
2-4x slower, runs hitting the 30-minute cap). This runs the command on its own
many-core box instead, and leaves the PC only the upload:

  1. Here: the command is checked and its draw pinned (`--rotation`, today's key, so a
     session that crosses midnight UTC measures the clips listed here); final-check's
     commits are resolved through the landed-build cache (tools/dev/build_cache.py,
     never a local compile) and its estimator, no-change and preset checks run here,
     since the box has no git; the clips the command can draw are listed.
  2. Staged as symlinks: this tree at HEAD (git archive: eval/, tools/ and the presets
     must be committed), every clip set file, the media of the listed clips only, the
     libvmaf tool and model, the builds (with their presets/), and the earlier cloud
     points of those clips (CLOUD_WORK).
  3. One spot box (tools/dev/aws_session.py: the budget cap, the instance fallback
     ladder, a hard lifetime, a root volume sized to the upload). Readiness: the tools
     run, the builds' version stamps, the pinned ffmpeg release (apt; another version
     decodes variable-rate clips differently), DECODE_CHECKS clips decode there exactly
     as here,
     the CPU metrics backend imports. Then the command, unchanged but for its paths,
     with NC_METRICS_BACKEND=libvmaf-cpu, the box's cores as CPU slots, workers to
     match (NC_MAX_WORKERS) and the point cache in a tmpfs.
  4. Back: the result JSON (the box's paths replaced by these), its lever-ledger rows
     (recorded here as the command records them), the new point-cache entries (into
     CLOUD_WORK); the clips are deleted on the box and it is terminated on every exit.

Scores. The box scores on the CPU (libvmaf, nceval.msssim); the PC on the GPU (VMAFx
CUDA). They are not bit-identical (2026-11-21, 16 points, PC: MS-SSIM differed on
every point, by up to 8.4e-7 in the mean; one frame's VMAF v1 by 2e-6, the last
printed digit, on 2 points; PSNR identical), so the two are never mixed: no local
point goes up, and cloud points never enter the local point cache. They seed only
later cloud runs. After each run, the cloud points that the local cache also holds
are compared with them and the differences printed (`parity`).

Speed. Each run pays a fixed 3-4 min (launch, upload, setup), then scales with the box
(2026-11-21, an L1 of 116 new points: 4.6 min end to end on a c7a.48xlarge, $0.14);
docs/PROCEDURES.md "Measure compression" has the measured times and costs.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))
import measure_image  # noqa: E402
from aws_session import PULL_SECONDS, NoCapacity, RemoteFailure, fallbacks_after, say  # noqa: E402
from nceval import fixedqp, gpuscore, levels, levers, pool, retention  # noqa: E402
from nceval.clipframes import clip_decoder, decode_command, decoder_environment  # noqa: E402
from nceval.clips import Clip  # noqa: E402
from nceval.config import load_paths  # noqa: E402
from nceval.metrics import (LIBVMAF_CPU_BACKEND, METRICS_BACKEND_ENV,  # noqa: E402
                            VMAF_V1_MODEL, VMAFX_CUDA_BACKEND)
from nceval.cloud import LOCAL_FLAG, ON_BOX_ENV, RUN_CAP_MINUTES  # noqa: E402
from nceval.gpuserve import POOL_SIZE_ENV  # noqa: E402
from nceval.realuse import MAX_WORKERS_ENV, WORK_SUBDIR, screen_clips  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
#: Cloud-scored points, in the point cache's layout (NC_WORK/realuse/...): never read by
#: a local run (see the module doc), only uploaded to seed the next cloud run.
CLOUD_WORK = DATA / "work/cloud-measure"
STAGE_ROOT = DATA / "work/cloud-measure-stage"
#: Where the tree and the builds live on the box (its user's home).
REMOTE_HOME = "/home/ubuntu"
REMOTE_DATA = f"{REMOTE_HOME}/nanocodec-data"
REMOTE_REPO = f"{REMOTE_HOME}/repo"
REMOTE_OUT = f"{REMOTE_HOME}/out"
REMOTE_JSON = f"{REMOTE_OUT}/result.json"
REMOTE_SEED = f"{REMOTE_HOME}/seed"
#: The point cache and the run's scratch: a tmpfs (RAM), since nceval.diskguard wants
#: 200 GiB free, which a root sized to the upload never has (aws_heldout_chart's rule).
REMOTE_WORK = "/mnt/work"
REMOTE_WORK_GB = 220
REMOTE_SPOOL = f"{REMOTE_WORK}/levers.jsonl"
REPO_ARCHIVE = "repo.tar"
#: An encode round's directories in its job's home (encode_env).
ENCODE_WORK = "encode-work"
PREBUILT = "prebuilt"
#: Each encode round's session facts, one JSON line, beside the run's prebuilt directory.
ROUNDS_LOG = "rounds.jsonl"
#: A GPU-scored command's kept streams (prebuilt_home) and how long they stay.
GPU_PREBUILT_PREFIX = "gpu-prebuilt-"
#: Where a result was scored (its `scored_on`, and its ledger rows' source), and the file a
#: GPU-scored command that cannot go on leaves its reason in (gpuscore.GpuUnavailable).
SCORED_GPU = "GPU (VMAFx CUDA on the PC)"
SCORED_CPU = "CPU (libvmaf on a box)"
UNAVAILABLE_REASON = "unavailable-reason.txt"
PREBUILT_KEEP_HOURS = 24
#: The commands (module, sub-command) GPU scoring here (nceval.gpuscore) can run: their
#: points go through nceval.realuse.Runner's rounds (fixedqp ab: the A/B of two builds of
#: our encoder; `gap` and `cost` stay on boxes, gap's reference encoders run x86 quality
#: builds). These are always scored on this PC's GPU: when it cannot score them the
#: measurement is refused, never sent to a box's CPU (owner, 2026-11-27: no cloud scoring,
#: no silent fallbacks; `--score cpu` and the automatic fallback are gone).
GPU_COMMANDS = (("nceval.quick", None), ("nceval.screen", None),
                ("nceval.levels", "final-check"), ("nceval.fixedqp", "ab"))
#: Python packages nceval imports that Ubuntu does not package, installed in a venv over
#: the system's (apt's numpy, scipy, opencv, xxhash): measure_image owns the package list.
REMOTE_VENV = measure_image.REMOTE_VENV
#: The anchor decoders every box carries (tools/anchors/build_anchors.sh builds, under
#: anchors/x86_64/bin, uploaded from here by `stage`): the standard decoders of every
#: codec (nceval.conformance.DECODERS: pinned ffmpeg, JM, HM, VVdeC, aomdec, vpxdec,
#: avmdec, dav2d) and dav1d, which a screen that decodes (e.g. layered AV1 settings)
#: needs; until 2026-11-23 only fixedqp gap jobs had them, and such screens failed on a
#: warm box ("dav1d not found"). Each must run there (`job_setup`).
BOX_DECODERS = ("ffmpeg-decode", "ldecod", "TAppDecoder", "vvdecapp", "aomdec", "vpxdec",
                "avmdec", "dav2d", "dav1d")
#: The clip decoder on the box: the pinned one (nceval.clipframes.clip_decoder, Ubuntu's
#: ffmpeg 8.0.1, apt-installed there as here) behind wrappers in REMOTE_FFMPEG/bin, which
#: is first on the box's PATH. Until 2026-11-23 the PC's first ffmpeg went up with its
#: libraries: conda's 7.1.1, which decodes some encoded clips to other frames
#: (docs/PROCEDURES.md "Reference decode of encoded clips").
REMOTE_FFMPEG = f"{REMOTE_HOME}/ffmpeg"
#: The ffmpeg programs a measurement runs (decodes, XPSNR; packet sizes).
FFMPEG_TOOLS = ("ffmpeg", "ffprobe")
CHECKPOINT_MARKER = ".checkpoint"
#: Defaults: a box sized to the job (AUTO_SIZES), and the ladder after it when it has no
#: capacity (aws_session.THROUGHPUT_X86_TYPES).
REGION = "eu-north-1"
DEFAULT_INSTANCE = "auto"
#: Box sizes `auto` picks from (vCPUs: types, preferred first), smallest first.
#: The guide's packed x86 pair at each size (docs/aws-cost-guide.md: no c7a, no m family).
AUTO_SIZES = ((32, ("c7i.8xlarge", "c6i.8xlarge")),
              (48, ("c7i.12xlarge", "c6i.12xlarge")),
              (64, ("c7i.16xlarge", "c6i.16xlarge")),
              (96, ("c7i.24xlarge", "c6i.24xlarge")))
#: The box's machine. x86_64 is the packed pair above; aarch64 is Graviton4 (c8g), which coded
#: 2.6x the packed encodes per dollar of c7i.24xlarge on 2026-11-24 (434k against 165k,
#: docs/aws-cost-guide.md). `--arch auto` takes aarch64 when `arm_reason` finds nothing
#: against it (a command that codes only our encoder, the aarch64 anchors and libvmaf here),
#: and x86_64 otherwise or when no c8g box launches; an aarch64-able job whose build has no
#: clean commit is refused, not moved to x86_64 (only `--arch x86_64` takes it there).
X86, ARM = "x86_64", "aarch64"
AMI_ARCH = {X86: "amd64", ARM: "arm64"}
ARCH_CHOICES = ("auto", X86, ARM)
#: auto (owner, 2026-11-24: dev screens and trains go to Graviton4 at once): Graviton4 where
#: the job allows it (2.3x the scored points and 2.6x the encodes per dollar, scores equal to
#: x86's on the same streams, docs/aws-cost-guide.md).
DEFAULT_ARCH = "auto"
#: The commands an aarch64 box runs: those that code and score our encoder only (a reference
#: encoder's quality is the x86 build's, nceval.config.QUALITY_MACHINE).
#: (module, sub-command), None for any sub-command. `nceval.fixedqp ab` A/Bs two builds of
#: ours (2026-11-24 parity: streams and scores equal on c8g and x86); `fixedqp gap` and
#: `cost` code reference encoders on their x86 quality builds, so they stay x86.
ARM_COMMANDS = (("nceval.screen", None), ("nceval.quick", None), ("nceval.levels", None),
                ("nceval.fixedqp", "ab"))
ARM_FAMILY = "c8g"
AUTO_SIZES_ARM = ((32, ("c8g.8xlarge",)), (48, ("c8g.12xlarge",)), (64, ("c8g.16xlarge",)))
#: Points each worker should get at least (a box with fewer idles on its tail: the
#: anchor's curves, then the candidate's single points, each end with a part-empty wave).
#: A first guess, to be refined from the runs' "box CPUs busy" lines.
MIN_WAVES = 8
#: CPUs per worker: one point runs our encoder, libvmaf (2 threads) and the Python
#: MS-SSIM and pipe threads.
CPUS_PER_WORKER = 3
#: Stage limits (minutes); a run is held to nceval.cloud.RUN_CAP_MINUTES (--run-minutes).
PREPARE_MINUTES = 10
#: The command prints a line per level or part; a level of slow points on a busy box
#: can be quiet for a few minutes (aws_heldout_chart: 10 for its score rounds).
STALL_MINUTES = 10
#: The box's lifetime and budget defaults: launch + setup + the run + the downloads.
DEFAULT_LIFETIME_MINUTES = 60
#: A box lost to a spot reclaim is replaced and the run started again, once (the
#: points it had measured are lost: they come back only at the end).
SESSION_ATTEMPTS = 2
DEFAULT_BUDGET_USD = 4.0
ROOT_SYSTEM_GB = 12
ROOT_DATA_MARGIN = 1.25
#: Clips whose decode on the box is compared with a local one before the run.
DECODE_CHECKS = 2
#: The commands this runs, and the options of each that name a local file: the builds
#: (`BINARY_OPTIONS`), input files (`FILE_OPTIONS`) and the result (`--json`).
MODULES = {("nceval.screen", None), ("nceval.levels", "final-check"), ("nceval.fixedqp", "ab"),
           ("nceval.fixedqp", "gap"),
           ("nceval.quick", None), ("nceval.tune", "search"), ("nceval.tune", "sensitivity"),
           ("nceval.tune", "finalists"), ("nceval.tune", "spsa"), ("nceval.tune", "sprt"),
           ("nceval.rc_oracle", None),
           ("nceval.arm_oracle", None)}
BINARY_OPTIONS = {"nceval.screen": ("--binary", "--anchor-binary"),
                  "nceval.quick": ("--binary",),
                  "nceval.tune": ("--binary",),
                  "nceval.levels": ("--old", "--new"),
                  "nceval.fixedqp": ("--binary", "--anchor-binary"),
                  "nceval.rc_oracle": ("--binary",),
                  "nceval.arm_oracle": ("--binary",)}
FILE_OPTIONS = {"nceval.fixedqp": ("--sets",), "nceval.tune": ("--space",)}
#: Options this tool sets itself.
OWN_OPTIONS = ("--workers",)
#: The lever-ledger rows each command records (levers.*_rows on its result).
ROW_SOURCES = {"nceval.screen": "nceval.screen", "nceval.levels": "nceval.levels final-check",
               "nceval.quick": "nceval.quick", "nceval.tune": "nceval.tune",
               "nceval.fixedqp": "nceval.fixedqp ab"}


@dataclass(frozen=True)
class Layout:
    """Where one job's own files live on the box: its home on a one-off box, a job
    directory under it on a warm box (aws_warm), where several jobs run at once and
    share the clips, ffmpeg, the environment and the point cache."""
    root: str = REMOTE_HOME
    job: str = ""  # "" on a one-off box

    @property
    def repo(self) -> str:
        return f"{self.root}/repo"

    @property
    def out(self) -> str:
        return f"{self.root}/out"

    @property
    def spool(self) -> str:
        return REMOTE_SPOOL if not self.job else f"{self.root}/levers.jsonl"

    @property
    def marker(self) -> str:
        """The point-cache checkpoint whose newer files are this job's to bring back."""
        return CHECKPOINT_MARKER if not self.job else f"{CHECKPOINT_MARKER}-{self.job}"

    def log(self, name: str) -> str:
        return f"{self.root}/{name}.log"


def warm_layout(job: str) -> Layout:
    return Layout(f"{REMOTE_HOME}/jobs/{job}", job)


# --- the command ------------------------------------------------------------------------

@dataclass
class Command:
    """An nceval command line (`python3 -m MODULE [SUB] ARGS`) and its option values."""
    module: str
    sub: str | None
    args: list[str]

    @classmethod
    def parse(cls, argv: list[str]) -> "Command":
        if argv[:1] == ["--"]:
            argv = argv[1:]
        if len(argv) < 3 or Path(argv[0]).name not in ("python3", "python") or argv[1] != "-m":
            raise SystemExit("expected `-- python3 -m nceval.MODULE ...` after the options")
        module, rest = argv[2], argv[3:]
        sub = rest[0] if rest and not rest[0].startswith("-") else None
        if (module, sub) not in MODULES:
            raise SystemExit(f"{module} {sub or ''}: not supported here; one of "
                             + ", ".join(f"{m} {s or ''}".strip() for m, s in sorted(
                                 MODULES, key=str)))
        command = cls(module, sub, rest[1:] if sub else rest)
        for option in OWN_OPTIONS:
            if command.get(option) is not None:
                command.args = command.without(option)
        return command

    def get(self, option: str) -> str | None:
        """The option's last value (`--x v` or `--x=v`), else None."""
        value = None
        for index, arg in enumerate(self.args):
            if arg == option and index + 1 < len(self.args):
                value = self.args[index + 1]
            elif arg.startswith(option + "="):
                value = arg.split("=", 1)[1]
        return value

    def has(self, flag: str) -> bool:
        return flag in self.args

    def without(self, option: str) -> list[str]:
        out, skip = [], False
        for arg in self.args:
            if skip:
                skip = False
            elif arg == option:
                skip = True
            elif not arg.startswith(option + "="):
                out.append(arg)
        return out

    def set(self, option: str, value: str) -> None:
        self.args = self.without(option) + [option, value]

    def argv(self) -> list[str]:
        return ["python3", "-m", self.module, *([self.sub] if self.sub else []), *self.args]


# --- the plan: builds, clips, rotation --------------------------------------------------

@dataclass
class Arm:
    """A build the command names: its option, local binary, presets and box path."""
    option: str
    binary: Path
    presets: Path | None
    remote_dir: str
    #: The same commit's aarch64 release build (aws_cpu_ab.release_build), for an aarch64 box.
    aarch64: Path | None = None

    def binary_for(self, machine: str) -> Path:
        return self.aarch64 if machine == ARM else self.binary

    @property
    def remote_binary(self) -> str:
        return f"{self.remote_dir}/nanocodec"


@dataclass
class Plan:
    command: Command
    arms: list[Arm]
    clips: list[Clip]
    inputs: dict[str, Path] = field(default_factory=dict)  # box path -> local file
    local_json: Path | None = None
    remote_json: str = REMOTE_JSON
    layout: Layout = Layout()
    #: Fields the local preflight decided (final-check: the commits), put back into the
    #: result: the box sees only binaries.
    report_fields: dict = field(default_factory=dict)
    #: Files the run changes in place on the box, brought back here (box path -> local):
    #: a tune study's Optuna storage.
    copies: dict[str, Path] = field(default_factory=dict)
    #: The draw the clips come from (a finalists run: the study's own, not an option).
    rotation: str | None = None
    #: An encode round (nceval.gpuscore): the command runs encode-only in the job's own
    #: work and prebuilt directories, and only the streams come back.
    encode_only: bool = False


def plan(command: Command, index: int | None = None, layout: Layout = Layout()) -> Plan:
    """Resolve the command's builds, pin its draw, list its clips (see the module doc).
    `index`: its place in a batch (its own result file and build directories)."""
    paths = load_paths()
    json_out = command.get("--json")
    if not json_out:
        raise SystemExit("give the command --json OUT: the result comes back there")
    if command.module == "nceval.screen" and command.get("--mode") == "cqp":
        raise SystemExit("--mode cqp (the fixed-QP proxy) caches elsewhere: not supported here")
    if command.module == "nceval.fixedqp" and command.sub == "ab" and command.get("--clipset"):
        raise SystemExit("fixedqp ab --clipset: not supported here (the pool draw only)")
    if command.sub == "finalists":  # the study's own draw
        rotation = tune_attrs(command, paths)["rotation"]
    else:
        rotation = command.get("--rotation") or (
            fixedqp.default_rotation() if command.module == "nceval.fixedqp"
            else pool.rotation_key())
        command.set("--rotation", rotation)
    tag = "" if index is None else f"{index}-"
    result = Plan(command, [], [], rotation=rotation, local_json=Path(json_out).resolve(),
                  remote_json=f"{layout.out}/result{'' if index is None else f'-{index}'}.json",
                  layout=layout)
    command.set("--json", result.remote_json)
    if command.module == "nceval.levels":
        result.arms, result.report_fields = final_check_arms(command, paths, tag, layout.root)
    else:
        for position, option in enumerate(BINARY_OPTIONS[command.module]):
            value = command.get(option)
            if value:
                binary = Path(value).resolve()
                if not binary.is_file():
                    raise SystemExit(f"{option} {value}: no such binary")
                beside = binary.parent / "presets"
                result.arms.append(Arm(option, binary, beside if beside.is_dir() else None,
                                       f"{layout.root}/arms/{tag}{position}"))
    for arm in result.arms:
        command.set(arm.option, arm.remote_binary)
    for option in FILE_OPTIONS.get(command.module, ()):
        value = command.get(option)
        if value:
            remote = f"{layout.root}/inputs/{Path(value).name}"
            result.inputs[remote] = Path(value).resolve()
            command.set(option, remote)
    if command.module == "nceval.rc_oracle":  # its schedules and rate logs stay on the box
        command.set("--work", f"{layout.root}/oracle-work")
    if command.module == "nceval.tune":
        # The study's storage (or a sweep's results) goes up when it exists (resumed)
        # and comes back after the run.
        local = tune_state(command, paths, rotation)
        remote = f"{REMOTE_DATA}/{local.relative_to(paths.data_root)}"
        if local.exists():
            if local.suffix == ".sqlite":
                checkpoint(local)  # WAL mode: the whole study in the file that goes up
            result.inputs[remote] = local
        result.copies[remote] = local
    result.clips = command_clips(command, paths, rotation)
    if command.module == "nceval.fixedqp" and command.sub == "gap":
        gap_inputs(command, result, paths, layout)  # after the clips: it moves --clipset
    return result


def gap_inputs(command: Command, result: Plan, paths, layout: Layout) -> None:
    """`fixedqp gap` (the results site's Fixed QP views): its chart clip set where the box
    keeps the clip sets (the clip-set files go up whole), our presets directory (the
    binary's own commit's: a release build has none beside it), and the reference
    encoders it codes, where the box's anchors are."""
    clipset = command.get("--clipset")
    if clipset:
        local = Path(clipset).resolve()
        if local.parent != paths.clips.resolve():
            raise SystemExit(f"fixedqp gap --clipset {clipset}: a clip set under {paths.clips}")
        command.set("--clipset", f"{REMOTE_DATA}/clips/{local.name}")
    presets = command.get("--presets-dir")
    if presets:
        remote = f"{layout.root}/inputs/presets"
        result.inputs[remote] = Path(presets).resolve()
        command.set("--presets-dir", remote)
    from nceval.conformance import DECODERS
    binary = Path(result.arms[0].binary) if result.arms else Path("nanocodec")
    # the anchors prefix and the timed reference builds beside it (config.REFERENCE_BUILDS:
    # quality runs the timed build's program, cbr.anchor_tool), each at its place under
    # the data root on the box too, where config.quality_reference finds it
    data_root = paths.data_root.absolute()
    anchors = data_root / "anchors"
    for name in (command.get("--families") or "").split(","):
        if name and not name.startswith(fixedqp.OWN_PREFIX):
            family = fixedqp.family_of(name, binary, paths)
            # its encoder (and VVenC's configuration), and the standard decoders its points
            # are scored from (an anchor writes no reconstruction: nceval.conformance gates
            # and decodes it). A tool's place is its path under the anchors prefix, not
            # its link target's (vpxenc and vpxdec link into anchors/vp9-av2): the
            # staging uploads the target's file there.
            for tool in (*family.program_files, *(d.binary(paths) for d in DECODERS[family.codec])):
                tool = Path(tool).absolute()
                if not tool.is_relative_to(anchors):
                    raise SystemExit(f"fixedqp gap: {name} needs {tool}, not an anchor tool")
                result.inputs[f"{REMOTE_DATA}/{tool.relative_to(data_root)}"] = tool


def final_check_arms(command: Command, paths, tag: str = "",
                     root: str = REMOTE_HOME) -> tuple[list[Arm], dict]:
    """final-check's two builds as binaries with their presets beside, and its checks
    that need git (levels.final_check's preamble), run here: the box gets the decided
    estimator, and --allow-same-presets once the preset check has passed here."""
    resolved = {}
    for role, option in (("old", "--old"), ("new", "--new")):
        spec = command.get(option)
        if Path(spec).is_file():
            binary, commit = Path(spec).resolve(), ""
        else:
            commit = levels.git(REPO, "rev-parse", f"{spec}^{{commit}}")
            directory = subprocess.run([sys.executable, str(REPO / "tools/dev/build_cache.py"),
                                        "for", commit], capture_output=True, text=True,
                                       check=True).stdout.strip().splitlines()[-1]
            binary = Path(directory) / "nanocodec"
        resolved[role] = (binary, commit, levels.presets_of(binary, paths))
    (old, old_commit, old_presets), (new, new_commit, new_presets) = \
        resolved["old"], resolved["new"]
    preset = command.get("--preset")
    presets_differ = bool(preset and preset != levels.MEDIUM) and \
        not levels.same_tree(old_presets, new_presets)
    unchanged = levels.no_change_reason((old, old_commit), (new, new_commit), presets_differ)
    if unchanged:
        raise SystemExit(f"{unchanged}: nothing to measure, no box rented")
    estimator = levels.final_estimator(command.get("--estimator") or levels.AUTO,
                                       old_commit, new_commit)
    command.set("--estimator", estimator)
    if preset and preset != levels.MEDIUM and levels.check_presets(
            (old, old_presets), (new, new_presets), command.has("--allow-same-presets")):
        if not command.has("--allow-same-presets"):
            command.args.append("--allow-same-presets")
    arms = [Arm(option, binary, presets, f"{root}/arms/{tag}{role}")
            for (role, option), (binary, _, presets) in zip(
                (("old", "--old"), ("new", "--new")), (resolved["old"], resolved["new"]))]
    fields = {"old": command.get("--old"), "new": command.get("--new"),
              "old_commit": old_commit, "new_commit": new_commit}
    return arms, fields


def command_clips(command: Command, paths, rotation: str) -> list[Clip]:
    """Every clip the command can measure: the draw's levels it runs, or named clips."""
    if command.module == "nceval.fixedqp" and command.get("--clipset"):
        return fixedqp.held_out_clips(command.get("--clipset"), int(command.get("--clips")),
                                      int(command.get("--frames") or fixedqp.FRAMES))
    if command.module == "nceval.fixedqp":
        count = command.get("--clips")
        return fixedqp.draw_clips(paths, rotation, command.get("--level") or pool.QUICK,
                                  int(count) if count else None,
                                  int(command.get("--frames") or fixedqp.FRAMES))
    if command.module == "nceval.tune":
        return tune_clips(command, paths, rotation)
    if command.module in ("nceval.rc_oracle", "nceval.arm_oracle"):
        from nceval import rc_oracle
        per_class = command.get("--per-class")
        return rc_oracle.clip_list(command.get("--clips") or "val", paths,
                                   int(command.get("--frames") or levels.DEFAULT_FRAMES),
                                   int(per_class) if per_class else None, rotation,
                                   tuple(c for c in (command.get("--classes") or "").split(",")
                                         if c))
    if command.module == "nceval.quick":
        from nceval import key_shares, quick
        draw = pool.draw(pool.load_pool(paths, levels.usable), rotation, levels.DEFAULT_FRAMES)
        probes = [c for share in key_shares.parse_periods(command.get("--key-shares"))
                  for c in quick.probe_clips(draw, share.window)]
        return draw.levels[pool.QUICK] + list(dict.fromkeys(probes))
    spec = command.get("--clips") or "adopt"
    if command.module == "nceval.screen" and spec != "adopt":
        return screen_clips(spec, paths)
    frames = int(command.get("--frames") or levels.DEFAULT_FRAMES)
    draw = pool.draw(pool.load_pool(paths, levels.usable), rotation, frames)
    if command.module == "nceval.levels":
        wanted = (pool.FINAL,)
    elif command.get("--levels"):
        wanted = tuple(level.strip() for level in command.get("--levels").split(","))
    else:
        wanted = pool.LEVELS[:int(command.get("--max-level") or len(pool.LEVELS))]
    return [clip for level in wanted for clip in draw.levels[level]]


def checkpoint(database: Path) -> None:
    """Fold an SQLite database's write-ahead log into the file."""
    import sqlite3
    with sqlite3.connect(database, timeout=60) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def tune_state(command: Command, paths, rotation: str) -> Path:
    """What a tune command keeps between runs: a study's Optuna storage, or a
    sensitivity sweep's results (tune_pool.sensitivity_dir)."""
    from nceval import tune_pool, tune_spsa
    if command.sub in ("spsa", "sprt"):
        return tune_spsa.state_path(command.get("--name"))
    if command.sub == "sensitivity":
        return tune_pool.sensitivity_dir(paths.data_root / "tuning", command.get("--codec"),
                                         command.get("--preset") or tune_pool.MEDIUM,
                                         rotation) / "results.json"
    return paths.data_root / "tuning" / f"{command.get('--study')}.sqlite"


def tune_attrs(command: Command, paths) -> dict:
    """A stored study's user attributes ({} when there is none here)."""
    import optuna
    name = command.get("--study")
    storage = f"sqlite:///{paths.data_root / 'tuning' / f'{name}.sqlite'}"
    if not (paths.data_root / "tuning" / f"{name}.sqlite").exists() or \
            name not in optuna.study.get_all_study_names(storage):
        if command.sub == "finalists":
            raise SystemExit(f"tune finalists: no study {name} here")
        return {}
    return optuna.load_study(study_name=name, storage=storage).user_attrs


def tune_clips(command: Command, paths, rotation: str) -> list:
    """A tune command's clips, plus the callgrind cost clips (nceval.tune.COST_CLIPS).
    search: the study's recorded trial and re-check clips (resumed), else its draw's;
    sensitivity: the first STAGE_1_CLIPS of the draw; finalists: the study's clips, its
    draw's L3 (at the trial window and the long window) and the long-horizon guard's."""
    from nceval import tune, tune_pool
    by_name = {c.name: c for c in tune.load_clips("train")}
    cost = [by_name[n] for n in tune.COST_CLIPS]
    if command.sub in ("spsa", "sprt"):
        from nceval import tune_spsa
        return tune_spsa.staged_clips(command.get, command.sub) + cost
    if command.sub == "sensitivity":
        return tune_pool.draw_sets(rotation, "none")[0][:tune_pool.STAGE_1_CLIPS] + cost
    attrs = tune_attrs(command, paths)
    clips: list = []
    if attrs:
        clips = tune_pool.clips_from_record(attrs["clips"]) + \
            tune_pool.clips_from_record(attrs.get("recheck_clips", []))
    elif command.get("--window-frames"):
        clips = [c for s in tune_pool.long_window_sets(rotation,
                                                       int(command.get("--window-frames")))
                 for c in s]
    else:
        clips = [c for s in tune_pool.draw_sets(rotation, command.get("--half") or "none")
                 for c in s]
    if command.sub == "finalists":
        usable = pool.load_pool(paths, levels.usable)
        for frames in (attrs.get("frames") or tune_pool.FRAMES, levels.LONG_FRAMES):
            clips += pool.draw(usable, rotation, frames).levels[pool.FINAL]
        clips += tune_pool.long_guard_clips()
    return list({c.name: c for c in clips}.values()) + cost


def clip_files(clips: list[Clip]) -> list[Path]:
    """The clips' media (and clean references), each under the data root."""
    files = sorted({p for c in clips for p in (c.yuv, c.clean_reference) if p})
    outside = [p for p in files if not p.is_relative_to(DATA)]
    if outside:
        raise SystemExit(f"clips outside {DATA}: {outside[:3]}")
    return files


#: A box run's time model (`run_minutes`), from measured runs on 2026-11-24: a full box
#: spends ~60 vCPU-s per planned point (job_points; an 8-screen L3-ladder batch coded 3,995
#: points in 41 min on a 48-vCPU c8g.12xlarge, about half of its 3,856 planned points'
#: work), and one command cannot run faster than its own critical path however big the box
#: (a final-check: 26 min on 48 vCPU; a quick: 7.6 min on 32).
BOX_VCPU_S_PER_POINT = 60.0
COMMAND_FLOOR_MINUTES = {"final": 26.0, "quick": 8.0}
#: A box's planned run must stay under this share of the run cap (--run-minutes).
RUN_FILL = 0.8


def command_floor(command: Command) -> float:
    """The shortest a command runs on any box (COMMAND_FLOOR_MINUTES): a long-window final
    check (final-check, or a screen of the final level), a quick, else none."""
    final = command.module == "nceval.levels" or command.get("--final-frames") or \
        "final" in (command.get("--levels") or "")
    if final:
        return COMMAND_FLOOR_MINUTES["final"]
    return COMMAND_FLOOR_MINUTES["quick"] if command.module == "nceval.quick" else 0.0


def run_minutes(runs: list[Plan], vcpus: int) -> float:
    """A box run's planned minutes for `runs` on `vcpus`: their work spread over the box, or
    the longest one's own floor, whichever is longer."""
    work = job_points(runs) * BOX_VCPU_S_PER_POINT / max(1, vcpus) / 60
    return max(work, max((command_floor(r.command) for r in runs), default=0.0))


def split_to_fit(runs: list[Plan], vcpus: int, limit: float) -> list[list[int]]:
    """The batch's runs (by index) in the fewest groups, each planned under `limit` minutes
    on `vcpus` (first fit, largest first); SystemExit naming the runs that cannot fit alone."""
    alone = {i: run_minutes([r], vcpus) for i, r in enumerate(runs)}
    over = [i for i, minutes in alone.items() if minutes > limit]
    if over:
        raise SystemExit(
            f"{len(over)} command(s) are planned over {limit:.0f} min on {vcpus} vCPU on their "
            f"own (run cap --run-minutes x {RUN_FILL}): " + "; ".join(
                f"#{i} {runs[i].command.module} ~{alone[i]:.0f} min" for i in over)
            + ". Nothing launched: give them fewer clips (--clips), or a longer --run-minutes")
    groups: list[list[int]] = []
    for index in sorted(alone, key=alone.get, reverse=True):
        for group in groups:
            if run_minutes([runs[i] for i in group + [index]], vcpus) <= limit:
                group.append(index)
                break
        else:
            groups.append([index])
    return [sorted(g) for g in groups]


def merged(runs: list[Plan]) -> Plan:
    """One staging plan for a batch: every run's builds and inputs, the union of clips."""
    clips = list({(c.name, getattr(c, "start_frame", 0)): c for r in runs for c in r.clips}.values())
    return Plan(runs[0].command, [a for r in runs for a in r.arms], clips,
                {k: v for r in runs for k, v in r.inputs.items()}, layout=runs[0].layout,
                rotation=runs[0].rotation)


def job_points(runs: list[Plan]) -> int:
    """About how many points the runs measure: per clip the anchor's ladder and one
    candidate point (single-point levels; a ladder level measures more, so this errs
    towards the smaller box)."""
    from nceval.cbr import clip_targets
    from nceval.realuse import LADDER_KBPS
    def curves(run: Plan) -> int:
        """Ladder curves per clip: an oracle run codes its variants and arms too."""
        if run.command.module == "nceval.arm_oracle":  # each arm in CBR and fixed QP
            return 2 * (1 + run.command.args.count("--arm"))
        if run.command.module != "nceval.rc_oracle":
            return 1
        from nceval import rc_oracle
        variants = run.command.get("--variants")
        return 3 + len(variants.split(",") if variants else rc_oracle.VARIANTS) + \
            run.command.args.count("--arm")
    return sum(curves(r) * len(clip_targets(c, LADDER_KBPS)) + 1 for r in runs for c in r.clips
               if isinstance(c, Clip))  # a tune study's cost clips are not ladder clips


def auto_types(points: int, sizes=AUTO_SIZES) -> tuple[str, ...]:
    """The types `auto` tries for a job of `points`: the largest AUTO_SIZES size whose
    workers each get MIN_WAVES points (the smallest when none does), its other vendors,
    then the next size down."""
    fitting = [i for i, (vcpus, _) in enumerate(sizes)
               if points >= MIN_WAVES * vcpus // CPUS_PER_WORKER]
    chosen = fitting[-1] if fitting else 0
    return sizes[chosen][1] + (sizes[chosen - 1][1] if chosen else ())


def aws_warm_idle_minutes() -> int:
    import aws_warm
    return aws_warm.IDLE_MINUTES


def commit_of(binary: Path) -> str | None:
    """The clean commit a build's version names (`nanocodec VERSION COMMIT`), else None."""
    words = version_of(binary).split()
    commit = words[2] if len(words) == 3 else ""
    return commit if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit) else None


def arm_command(command: Command) -> bool:
    """Whether `command` may run on an aarch64 box (ARM_COMMANDS)."""
    return any(command.module == module and sub in (None, command.sub)
               for module, sub in ARM_COMMANDS)


def arm_reason(runs: list[Plan], arms: list[Arm]) -> str | None:
    """Why these runs cannot go to an aarch64 box (None: they can), the aarch64 builds
    found on the way set on the arms (made on the build box when missing). SystemExit when
    they could but a build has no clean commit (its aarch64 twin cannot be built)."""
    # an encode round codes our encoder only (nceval.gpuscore): any GPU command may go
    others = sorted(" ".join(filter(None, (r.command.module, r.command.sub))) for r in runs
                    if not getattr(r, "encode_only", False) and not arm_command(r.command))
    if others:
        return f"{', '.join(others)} runs reference encoders (x86 quality builds)"
    anchors = load_paths().anchors.parent / ARM
    missing = [t for t in ("vmaf", *BOX_DECODERS) if not (anchors / "bin" / t).is_file()]
    if missing or not (anchors / VMAF_V1_MODEL).is_file():
        return f"{anchors}: no aarch64 {', '.join(missing) or 'VMAF model'}"
    commits = {arm.binary: commit_of(arm.binary) for arm in arms}
    dirty = [str(b) for b, c in commits.items() if c is None]
    if dirty:
        # never a quiet x86 fallback (2026-11-25: a dirty build took a 48-vCPU c6i box in
        # place of a 96-vCPU Graviton and doubled the wall time); --arch x86_64 still may
        raise SystemExit(
            f"no clean commit behind {', '.join(dirty)}: an aarch64 build needs one. Rebuild "
            f"from the commit (NC_NANOCODEC_BIN=$(tools/dev/binary_for.sh COMMIT)), or pass "
            f"--arch {X86} to measure this build on an x86 box")
    import aws_cpu_ab
    for arm in arms:
        arm.aarch64 = aws_cpu_ab.release_build(commits[arm.binary], ARM)
    return None


def choose_machine(arch: str, runs: list[Plan], arms: list[Arm]) -> str:
    """The box's machine for `--arch` (see ARM)."""
    if arch == X86:
        return X86
    reason = arm_reason(runs, arms)
    if reason and arch == ARM:
        raise SystemExit(f"--arch {ARM}: {reason}")
    if reason:
        say(f"an x86 box: {reason}")
    return X86 if reason else ARM


# --- staging -----------------------------------------------------------------------------

def stage(stage_dir: Path, run: Plan, machine: str = X86, commit: str | None = None) -> None:
    """The box's home as symlinks (listed in manifest.txt, so a half-made staging is
    removed too): repo.tar, nanocodec-data/ (clip sets, the clips' media, libvmaf and
    its model), arms/, inputs/ and seed/ (earlier cloud points of these clips)."""
    stage_dir.mkdir(parents=True)
    job_home = stage_dir / Path(run.layout.root).relative_to(REMOTE_HOME)
    archive = job_home / REPO_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_tree(archive, commit)
    links = {stage_dir / "nanocodec-data" / p.relative_to(DATA): p
             for p in clip_files(run.clips)}
    paths = load_paths()
    links |= {stage_dir / "nanocodec-data/clips" / p.name: p
              for p in sorted(paths.clips.glob("*.json"))}
    anchors = stage_dir / "nanocodec-data/anchors" / machine
    local_anchors = paths.anchors.parent / machine
    links[anchors / "bin/vmaf"] = local_anchors / "bin/vmaf"
    links[anchors / "share/vmaf"] = local_anchors / "share/vmaf"
    for tool in BOX_DECODERS:
        source = local_anchors / "bin" / tool
        if not source.is_file():
            raise SystemExit(f"{source}: missing (tools/anchors/build_anchors.sh); every "
                             f"box carries the anchor decoders")
        links[anchors / "bin" / tool] = source
    for arm in run.arms:
        home = stage_dir / Path(arm.remote_dir).relative_to(REMOTE_HOME)
        links[home / "nanocodec"] = arm.binary_for(machine)
        if arm.presets:
            links[home / "presets"] = arm.presets
    links |= {stage_dir / Path(remote).relative_to(REMOTE_HOME): local
              for remote, local in run.inputs.items()}
    # A warm box keeps its own point cache: no seed.
    links |= {} if run.layout.job else seed_links(stage_dir, {c.name for c in run.clips})
    wrappers = [stage_dir / "ffmpeg/bin" / tool for tool in FFMPEG_TOOLS]
    (stage_dir / "manifest.txt").write_text("".join(f"{p}\n"
                                                    for p in [*links, *wrappers, archive]))
    make_links(stage_dir, links)
    wrappers[0].parent.mkdir(parents=True)
    for wrapper in wrappers:
        pinned = clip_decoder().with_name(wrapper.name)
        wrapper.write_text(f"#!/bin/sh\nexec {pinned} \"$@\"\n")
        wrapper.chmod(0o755)


def seed_links(stage_dir: Path, names: set[str]) -> dict[Path, Path]:
    """{staged path: CLOUD_WORK directory} of the earlier cloud points of the clips `names`
    (one directory per frame window and clip; the upload follows the links, so it sends
    what the directory holds then)."""
    cache = CLOUD_WORK / WORK_SUBDIR
    found = {stage_dir / "seed" / window.name / clip.name: clip
             for window in (sorted(cache.iterdir()) if cache.is_dir() else ())
             for clip in sorted(window.iterdir()) if clip.name in names}
    return {link: clip for link, clip in found.items() if claim_seed(clip)}


def claim_seed(clip: Path) -> bool:
    """Mark a seed directory as just used, so the cache's prune (which spares entries used
    within CLOUD_SEED_FRESH_S) leaves it while this job stages and uploads it; False when
    the prune removed it first (a seed is only a saving: the job measures those points)."""
    try:
        os.utime(clip)
        return True
    except FileNotFoundError:
        return False


def reseed(stage_dir: Path, names: set[str]) -> int:
    """Before a retry: link the clips' cloud-point directories made since staging (a lost
    box's points pulled back into windows the seed did not have), so the next box re-runs
    only the points still missing; returns how many links were added. The links already
    there follow their directories' new files by themselves."""
    added = {link: target for link, target in seed_links(stage_dir, names).items()
             if not link.is_symlink()}
    for link, target in added.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target.resolve())
    with (stage_dir / "manifest.txt").open("a") as manifest:
        manifest.writelines(f"{link}\n" for link in added)
    return len(added)


def archive_tree(archive: Path, commit: str | None = None) -> None:
    """The tree the box runs: `commit` (a job's pinned tree, `--tree-commit`), else HEAD,
    refusing uncommitted eval/, tools/ or presets (nceval.cloud.pinned_tree)."""
    from nceval.cloud import pinned_tree
    subprocess.run(["git", "-C", str(REPO), "archive", "-o", str(archive),
                    commit or pinned_tree()], check=True)


def unstage(stage_dir: Path) -> None:
    """Remove the staging tree: its symlinks, its own copies (repo.tar, ffmpeg) and
    directories. rmtree removes a symlink itself, never what it points to. (It removed
    only the manifest's entries and then rmdir'd, which failed on the copies: 27 jobs'
    staging, 5.5 GB, were left on 2026-11-21.)"""
    shutil.rmtree(stage_dir, ignore_errors=True)


#: A cloud point directory used (written, or claimed by a staging job's seed) this
#: recently is never pruned: a job's staging and upload take minutes (2026-11-24: a prune
#: removed a directory between another job's seed listing and its upload, which failed).
CLOUD_SEED_FRESH_S = 2 * 3600


def prune_cloud_points() -> int:
    """The cloud point cache within its cap (retention.toml work/cloud-measure): its
    least recently written clip directories (one per frame window and clip; a seed links
    whole ones) go first; returns bytes freed. They are re-measured on the next cloud run
    that needs them."""
    cap = retention.cap_bytes(CLOUD_WORK, DATA)
    cache = CLOUD_WORK / WORK_SUBDIR
    if cap is None or not cache.is_dir():
        return 0
    clips = [clip for window in sorted(cache.iterdir()) if window.is_dir()
             for clip in sorted(window.iterdir()) if clip.is_dir()]
    others = retention.entry_bytes(CLOUD_WORK) - sum(retention.entry_bytes(c) for c in clips)
    return retention.prune_to_cap(clips, cap, "aws_measure", fixed_bytes=others,
                                  min_idle_s=CLOUD_SEED_FRESH_S)[0]


def sweep_staging(root: Path = None) -> int:
    """Remove the staging and download directories (`stage-STAMP-PID`, `back-STAMP-PID`)
    of jobs whose process is gone (killed before its cleanup); returns bytes freed."""
    root = root or STAGE_ROOT
    freed = 0
    for entry in root.glob("*-*-*-*") if root.is_dir() else ():
        pid = entry.name.rsplit("-", 1)[-1]
        if not (entry.name.startswith(("stage-", "back-")) and pid.isdigit()) or \
                Path(f"/proc/{pid}").exists() or retention.guard(entry, "aws_measure") is not None:
            continue
        freed += sum(f.stat().st_size for f in entry.rglob("*")
                     if f.is_file() and not f.is_symlink())
        shutil.rmtree(entry, ignore_errors=True)
    return freed


def staged_bytes(stage_dir: Path) -> int:
    """What the upload sends (symlinks followed, as `tar -h` does)."""
    total = 0
    for directory, _, files in os.walk(stage_dir, followlinks=True):
        for name in files:
            with contextlib.suppress(FileNotFoundError):  # a seed file pruned meanwhile
                total += os.stat(os.path.join(directory, name)).st_size
    return total


def root_gb(upload_bytes: int) -> int:
    """The root volume: the system and the upload (the run writes to the tmpfs)."""
    return ROOT_SYSTEM_GB + int(upload_bytes * ROOT_DATA_MARGIN / (1 << 30) + 0.999)


# --- the box's commands -------------------------------------------------------------------

def remote_env(layout: Layout = Layout()) -> str:
    """The run's environment on the box: CPU scoring, its cores as slots and workers,
    and ON_BOX_ENV (an entry point that would offload itself runs here, nceval.cloud)."""
    return " ".join([f"PATH={REMOTE_FFMPEG}/bin:{REMOTE_VENV}/bin:$PATH", f"NC_DATA_ROOT={REMOTE_DATA}", f"NC_WORK={REMOTE_WORK}",
                     f"NC_METRICS_BACKEND={LIBVMAF_CPU_BACKEND}", "NC_CPU_SLOTS=$(nproc)",
                     f"{MAX_WORKERS_ENV}=$(( $(nproc) / {CPUS_PER_WORKER} ))",
                     f"NC_LEVERS_SPOOL={layout.spool}", f"{ON_BOX_ENV}=1"])


def encode_dirs(layout: Layout, index: int) -> tuple[str, str]:
    """An encode round's request `index`: its work and prebuilt directories on the box
    (every request's prebuilt directory under one PREBUILT, the round's one pull)."""
    return f"{layout.root}/{ENCODE_WORK}/{index}", f"{layout.root}/{PREBUILT}/{index}"


def encode_env(layout: Layout, index: int) -> str:
    """A request's environment on top of remote_env: encode-only, into its own prebuilt
    directory, its point cache its own (seeded with the PC's GPU results, which never
    reach the box's CPU-scored cache)."""
    work, prebuilt = encode_dirs(layout, index)
    return " ".join([f"NC_WORK={work}", f"{gpuscore.ENCODE_ONLY_ENV}=1",
                     f"{gpuscore.PREBUILT_ENV}={prebuilt}"])


def rate_probes_command(layout: Layout, index: int) -> str:
    """After a request's command: its rate probes (fixedqp's quantizer choice) into its
    prebuilt directory, at their paths in its work directory, so they come back with the
    streams (nceval.gpuscore.import_rate_probes)."""
    work, prebuilt = encode_dirs(layout, index)
    return (f"(cd {work} && find . -name 'rate-*.json' -print0 | "
            f"xargs -0 -r cp --parents -t {prebuilt}/{gpuscore.RATE_PROBES}) "
            f"2>/dev/null || true")


def make_links(stage_dir: Path, links: dict[Path, Path]) -> None:
    """The stage's symlinks. A link under another link of the stage is already served by it
    and is skipped: creating it would follow the parent link out of the stage (2026-11-25:
    a batch staging a clip set's directory and one of its files tried to write a link into
    the data root's clips and failed with FileExistsError). A link that already points at
    its target is kept; nothing is ever created outside `stage_dir`."""
    stage = stage_dir.resolve()
    for link in sorted(links, key=lambda p: len(p.parts)):
        if any(parent in links for parent in link.parents):
            continue
        target = links[link].resolve()
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.parent.resolve().is_relative_to(stage):
            raise RuntimeError(f"stage link {link} would leave the stage {stage_dir}")
        if link.is_symlink() and link.resolve() == target:
            continue
        link.symlink_to(target)


def encode_setup(layout: Layout, requests: int) -> list[str]:
    """An encode round's readiness: each request's work seeded with the PC's results
    (staged as seed/<i>), the keys the PC holds (have-<i>.txt) beside its prebuilt
    directory; the checkpoint marker the streams come back from."""
    steps = []
    for index in range(requests):
        work, prebuilt = encode_dirs(layout, index)
        steps += [f"mkdir -p {work}/{WORK_SUBDIR} {prebuilt}/{gpuscore.RATE_PROBES}",
                  f"cp -rL {layout.root}/seed/{index}/. {work}/",
                  f"cp {layout.root}/have-{index}.txt {prebuilt}/{gpuscore.HAVE_FILE}"]
    return steps + [f"touch {layout.root}/{PREBUILT}/{layout.marker}"]


def box_setup(machine: str = X86) -> list[str]:
    """A box's own readiness, once: packages (the pinned ffmpeg among them), the
    environment, the point cache's tmpfs, and that its ffmpeg is this machine's pinned
    release and libvmaf (uploaded) runs there."""
    return [
        # the packages (measure_image): skipped on a box launched from the baked image
        measure_image.install_unless_baked(),
        f"sudo mkdir -p {REMOTE_WORK}",
        f"sudo mount -t tmpfs -o size={REMOTE_WORK_GB}g,mode=1777 tmpfs {REMOTE_WORK}",
        f"mkdir -p {REMOTE_WORK}/{WORK_SUBDIR}",
        f"test \"$({REMOTE_FFMPEG}/bin/ffmpeg -version | head -1)\" = "
        f"{shlex.quote(ffmpeg_version())}",
        f"{REMOTE_FFMPEG}/bin/ffmpeg -hide_banner -filters | grep -qw xpsnr",
        f"{REMOTE_FFMPEG}/bin/ffprobe -version > /dev/null",
        f"! ldd {REMOTE_DATA}/anchors/{machine}/bin/vmaf | grep 'not found'",
        f"{REMOTE_DATA}/anchors/{machine}/bin/vmaf --version",
        f"test -f {REMOTE_DATA}/anchors/{machine}/{VMAF_V1_MODEL}",
        f"test \"$(uname -m)\" = {machine}",
        "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs",
    ]


def seed_steps(warm: bool) -> list[str]:
    """The seed into the box's point cache. A warm box keeps its own cache and takes a seed
    only when a reclaimed box's retry staged one (rent: reseed), never overwriting the
    points it has (cp -n)."""
    if warm:
        return [f"if [ -d {REMOTE_SEED} ]; then cp -rn {REMOTE_SEED}/. "
                f"{REMOTE_WORK}/{WORK_SUBDIR}/; fi"]
    return [f"mkdir -p {REMOTE_SEED}", f"cp -r {REMOTE_SEED}/. {REMOTE_WORK}/{WORK_SUBDIR}/"]


def job_setup(run: Plan, machine: str = X86) -> list[str]:
    """One job's readiness: its tree, (one-off box) the seed, the checkpoint its points
    are brought back from, each build's version stamp is the one here, the metrics
    backend imports, and the first clips decode to what they decode to here."""
    layout = run.layout
    versions = [(arm.remote_binary, version_of(arm.binary)) for arm in run.arms]
    decodes = [f"{shlex.join(remote_decode(c))} | sha256sum | sed 's/-$/{c.name}/'"
               for c in decode_checks(run.clips)]
    seed = seed_steps(warm=bool(layout.job))
    # a tune study's CPU objective is callgrind: installed with the packages since
    # 2026-11-25 (measure_image), checked here for boxes set up before
    tools = (["command -v valgrind > /dev/null || sudo DEBIAN_FRONTEND=noninteractive "
              "apt-get -qq install -y valgrind > /dev/null"]
             if run.command.module == "nceval.tune" else [])
    return [
        f"mkdir -p {layout.repo} {layout.out} && tar -C {layout.repo} -xf "
        f"{layout.root}/{REPO_ARCHIVE}",
        *seed, *tools,
        # what is in the cache already: only what the run adds comes back
        f"touch {REMOTE_WORK}/{WORK_SUBDIR}/{layout.marker}",
        *[f"test \"$({binary} --version)\" = {shlex.quote(version)}"
          for binary, version in versions],
        f"cd {layout.repo}/eval && {remote_env(layout)} python3 -c 'import "
        f"{run.command.module}, nceval.metrics as m; print(\"backend\", m.metrics_backend())'",
        *decodes,
        *decoder_checks(machine),
        "echo cpus $(nproc) workers $(( $(nproc) / %d ))" % CPUS_PER_WORKER,
    ]


def decoder_checks(machine: str = X86) -> list[str]:
    """Every anchor decoder is on the box and its libraries resolve (fails fast)."""
    checks = []
    for tool in BOX_DECODERS:
        path = f"{REMOTE_DATA}/anchors/{machine}/bin/{tool}"
        checks += [f"test -x {path} || (echo 'missing anchor decoder {tool}' >&2; false)",
                   f"! ldd {path} 2>&1 | grep 'not found'"]
    return checks


def prepare_command(run: Plan, machine: str = X86) -> str:
    """Readiness on a one-off box, failing fast: the box's, then the job's."""
    return " && ".join(box_setup(machine) + job_setup(run, machine))


def ffmpeg_version() -> str:
    return subprocess.run([str(clip_decoder()), "-version"], capture_output=True, text=True,
                          check=True, env=decoder_environment()).stdout.splitlines()[0]


def version_of(binary: Path) -> str:
    return subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                          check=True).stdout.strip()


def decode_checks(clips: list[Clip]) -> list[Clip]:
    """The encoded clips whose decode is compared, the likeliest to differ first: a
    frame rate that is not whole, from a rotated start (ffmpeg's frame timing)."""
    encoded = [c for c in clips if getattr(c, "encoded", False)]
    return sorted(encoded, key=lambda c: (c.fps == round(c.fps), c.start_frame == 0,
                                          c.name))[:DECODE_CHECKS]


def remote_decode(clip: Clip) -> list[str]:
    """The clip's decode command (nceval.clipframes) with the box's paths."""
    command = decode_command(clip, min(clip.frames, levels.DEFAULT_FRAMES))
    return [f"{REMOTE_DATA}/{Path(a).relative_to(DATA)}" if a.startswith(str(DATA)) else
            (f"{REMOTE_FFMPEG}/bin/ffmpeg" if index == 0 else a) for index, a in enumerate(map(str, command))]


def local_decodes(run: Plan) -> dict[str, str]:
    """The sha256 of the first decode-checked clips' decodes here."""
    import hashlib
    checks = decode_checks(run.clips)
    return {c.name: hashlib.sha256(subprocess.run(
        decode_command(c, min(c.frames, levels.DEFAULT_FRAMES)), capture_output=True,
        check=True, env=decoder_environment()).stdout).hexdigest() for c in checks}


def run_command(run: Plan, runs: list[Plan] | None = None) -> str:
    """The command, then how busy the box's CPUs were over it (sizing the box: a run that
    leaves most cores idle wants a smaller one). It ends on the command's status, never
    `exit`: aws_session.run_detached appends the `echo $? > LOG.status` that ends the
    watch (an `exit` here skipped it, and the finished run stalled for 10 min)."""
    busy = ("awk -v a=\"$cpu0\" '{split(a, s); t = 0; for (i = 2; i <= NF; i++) "
            "t += $i - s[i]; idle = $5 + $6 - s[5] - s[6]; "
            "printf \"box CPUs busy %.0f%% over the run\\n\", 100 * (1 - idle / t)}' "
            "<(head -1 /proc/stat)")
    workers = f"--workers $(( $(nproc) / {CPUS_PER_WORKER} ))"
    # A heartbeat while the job's own processes use CPU (its session: run_detached
    # starts it with setsid), so a long quiet level of a packed batch is not taken for
    # a stall (unused-tools and bg-ref hit the no-progress kill, 2026-11-21); a job
    # whose processes stop working still goes quiet and is stopped.
    beat = ("{ prev=; while sleep 60; do now=$(ps -o times= --sid $$ | awk '{s += $1} "
            "END {print s + 0}'); [ \"$now\" != \"$prev\" ] && echo \"job CPU ${now} s\"; "
            "prev=$now; done; } & beat=$!; ")
    stop = "pkill -P $beat 2>/dev/null; kill $beat 2>/dev/null; "  # and its sleep
    if getattr(run, "encode_only", False):
        return encode_run_command(run, runs or [run], beat, stop, busy, workers)
    if not runs:
        return (f"cpu0=$(head -1 /proc/stat); {beat}cd {run.layout.repo}/eval && "
                f"{remote_env(run.layout)} {shlex.join(run.command.argv())} {workers}; "
                f"status=$?; {stop}{busy}; (exit $status)")
    # A batch: side by side, each with the box's workers (the slots share the cores).
    started = " ".join(f"{remote_env(run.layout)} {shlex.join(r.command.argv())} {workers} & "
                       f"p{i}=$!;" for i, r in enumerate(runs))
    waits = " ".join(f"wait $p{i} || status=1;" for i in range(len(runs)))
    # Grouped: `cd D && a & b &` backgrounds the cd with `a` alone, and b ran from ~.
    # No `exit` (see above): a failed cd leaves status 1.
    return (f"cpu0=$(head -1 /proc/stat); {beat}status=1; cd {run.layout.repo}/eval && "
            f"{{ {started} status=0; {waits} }}; {stop}{busy}; (exit $status)")


def encode_run_command(run: Plan, runs: list[Plan], beat: str, stop: str, busy: str,
                       workers: str) -> str:
    """An encode round's run: its requests' commands side by side, each encode-only in
    its own work and prebuilt directories (`encode_env`), then each one's rate probes
    beside its streams. A command's decisions on missing results may fail it: its streams
    are the result, and the PC scores (and encodes) whatever is missing, so the round
    ends 0."""
    layout = run.layout
    started = " ".join(f"{remote_env(layout)} {encode_env(layout, i)} "
                       f"{shlex.join(r.command.argv())} {workers} & p{i}=$!;"
                       for i, r in enumerate(runs))
    waits = " ".join(f"wait $p{i} || echo \"encode-only: command {i} ended with status $?\";"
                     for i in range(len(runs)))
    probes = " ".join(f"{rate_probes_command(layout, i)};" for i in range(len(runs)))
    return (f"cpu0=$(head -1 /proc/stat); {beat}cd {layout.repo}/eval && "
            f"{{ {started} {waits} }}; {probes} status=0; {stop}{busy}; (exit $status)")


def cleanup_command(layout: Layout = Layout()) -> str:
    """The clips and builds leave the box before it is terminated (owner, 2026-11-20).
    A warm box's job removes only its own directory: the box keeps the clips for the
    next job, and its idle timer removes them (box_cleanup_command) before it stops."""
    if layout.job:
        return f"rm -rf -- {layout.root}; true"
    return box_cleanup_command()


def box_cleanup_command() -> str:
    return (f"rm -rf -- {REMOTE_DATA}/clips {REMOTE_HOME}/arms {REMOTE_HOME}/jobs "
            f"{REMOTE_SEED} {REMOTE_WORK}/{WORK_SUBDIR}; sudo umount {REMOTE_WORK}; true")


# --- results -------------------------------------------------------------------------------

def localise(text: str, run: Plan) -> str:
    """The box's paths in a result replaced by this machine's (longest first)."""
    mapping = {arm.remote_binary: str(arm.binary) for arm in run.arms}
    mapping |= {remote: str(local) for remote, local in run.inputs.items()}
    mapping |= {run.remote_json: str(run.local_json), REMOTE_DATA: str(DATA),
                run.layout.repo: str(REPO)}
    for remote in sorted(mapping, key=len, reverse=True):
        text = text.replace(remote, mapping[remote])
    return text


def result_status(path: Path) -> str:
    """"done", "missing" or the command's own error status (nceval.screen writes one)."""
    if not path.exists():
        return "missing"
    try:
        return json.loads(path.read_text()).get("status", "done")
    except json.JSONDecodeError:
        return "unreadable"


def result_name(run: Plan) -> str:
    return Path(run.remote_json).name


def finish(run: Plan, back: Path, cloud: dict, runs: list[Plan] | None = None) -> dict:
    """The result JSON here (localised, with the session's facts and the final-check
    fields decided here, and where it was scored); returns it. `runs`: the batch whose lever rows (one spool on the box) are recorded
    here with it, each localised by its own run."""
    result = json.loads(localise((back / result_name(run)).read_text(), run))
    result.update(run.report_fields)
    result["cloud"] = cloud
    result["scored_on"] = SCORED_CPU
    run.local_json.parent.mkdir(parents=True, exist_ok=True)
    run.local_json.write_text(json.dumps(result, indent=2, default=str) + "\n")
    spool = back / "levers.jsonl"
    if runs is not None and spool.exists():  # the rows the commands recorded there
        text = spool.read_text()
        for each in runs:
            text = localise(text, each)
        rows = [{f: stored.get(f) for f in levers.FIELDS}
                for stored in map(json.loads, text.splitlines())]
        for row in rows:
            row["source"] = f"{run.local_json} (aws {cloud['instance_type']}, CPU metrics)"
        levers.record(rows, f"{ROW_SOURCES.get(run.command.module, run.command.module)} on aws")
    say(f"result: {run.local_json}")
    return result


def merge_points(back: Path) -> int:
    """The run's new cloud points into CLOUD_WORK; returns how many."""
    source, target = back / WORK_SUBDIR, CLOUD_WORK / WORK_SUBDIR
    count = 0
    for path in sorted(source.rglob("*.json")) if source.is_dir() else ():
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        path.unlink()
        count += 1
    return count


#: The score fields compared by `parity` (per point and per frame).
STREAM_FIELDS = ("frame_bits", "kbps")
PARITY_FIELDS = ("psnr_yuv", "vmaf_v1", "ms_ssim", "frame_psnr_yuv", "frame_vmaf_v1",
                 "frame_bits", "kbps")


def parity(cloud_points: Path, local_points: Path, names: set[str] | None = None) -> dict:
    """Cloud points against the same points in the local cache (same identity files):
    per field, how many differ and the largest difference."""
    report = {"compared": 0, "identical": 0, "fields": {}}
    for path in sorted(cloud_points.rglob("*.json")) if cloud_points.is_dir() else ():
        relative = path.relative_to(cloud_points)
        if path.name.startswith("rate-") or (names and relative.parts[1] not in names):
            continue
        local = local_points / relative
        if not local.is_file():
            continue
        ours, theirs = json.loads(path.read_text()), json.loads(local.read_text())
        report["compared"] += 1
        same = True
        for name in PARITY_FIELDS:
            a, b = ours.get(name), theirs.get(name)
            if a == b:
                continue
            same = False
            pairs = zip(a, b) if isinstance(a, list) and isinstance(b, list) else [(a, b)]
            gap = max((abs(x - y) for x, y in pairs if isinstance(x, (int, float))
                       and isinstance(y, (int, float))), default=float("inf"))
            entry = report["fields"].setdefault(name, {"differ": 0, "max_abs": 0.0})
            entry["differ"] += 1
            entry["max_abs"] = max(entry["max_abs"], gap)
        report["identical"] += same
    return report


def describe_parity(report: dict) -> str:
    if not report["compared"]:
        return "parity: no cloud point is in the local cache"
    fields = ", ".join(f"{k} differs on {v['differ']} (max {v['max_abs']:.3g})"
                       for k, v in report["fields"].items())
    text = (f"parity with the local cache: {report['identical']} of {report['compared']} "
            f"points identical" + (f"; {fields}" if fields else ""))
    if any(name in report["fields"] for name in STREAM_FIELDS):
        # scores may differ by backend; the streams never may: an input or encoder problem
        text += ("\n!!! the box coded different streams from this machine's for the same "
                 "point: do not trust this run (the clip's decode? an ISA-dependent encode?)")
    return text


# --- the session ----------------------------------------------------------------------------

def measure(args: argparse.Namespace, session_factory=None) -> dict | None:
    """Plan, stage, rent, run, bring back (see the module doc). `session_factory`
    (spec -> context manager) replaces aws_session.Session in tests."""
    batch = getattr(args, "batch", None)
    if getattr(args, "encode_round", None) or getattr(args, "encode_rounds", None):
        if not getattr(args, "rounds_log", None):
            args.rounds_log = str(Path(getattr(args, "encode_round", None) or args.encode_rounds).parent
                                  / ROUNDS_LOG)
        return encode_rounds(args, round_requests(args), session_factory)
    commands = json.loads(Path(batch).read_text()) if batch else [args.command]
    # session_factory: the tests drive the box path with fake sessions; a dry run rents
    # nothing and scores nothing
    if gpu_scored(commands) and not args.dry_run and session_factory is None:
        outcomes = gpu_score(args, commands)
        done = [r for r in outcomes if isinstance(r, dict)]
        left = sorted({r for r in outcomes if isinstance(r, str)})
        if left:
            raise SystemExit(f"{len(done)} of {len(commands)} command(s) scored (their --json "
                             f"written); the rest: " + gpuscore.refusal("; ".join(left)))
        return done[0] if not batch else {"results": done}
    warm = session_factory is None and not getattr(args, "cold", False)
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    layout = warm_layout(stamp) if warm else Layout()
    runs = [plan(Command.parse(c), i if batch else None, layout)
            for i, c in enumerate(commands)]
    run = merged(runs) if batch else runs[0]
    files = clip_files(run.clips)
    points = job_points(runs)
    arch = getattr(args, "arch", DEFAULT_ARCH)
    machine = choose_machine(arch, runs, run.arms)
    types = auto_types(points, AUTO_SIZES_ARM if machine == ARM else AUTO_SIZES) \
        if args.instance_type == "auto" else (args.instance_type,)
    if getattr(args, "instance_family", None):
        import aws_warm
        types = aws_warm.family_types(args.instance_family)
    fitted, types = fit_to_run_cap(args, runs, commands, types, machine, session_factory)
    if fitted is not None:  # split across boxes
        return fitted
    say(f"{' + '.join(f'{r.command.module} {r.command.sub or ""}'.strip() for r in runs)}: "
        f"{len(run.clips)} clips ({sum(p.stat().st_size for p in files) / 1e9:.1f} GB), "
        f"{len(run.arms)} builds, rotation {run.rotation}, ~{points} points: "
        + ("a warm box" if warm else types[0]) + f" ({machine})")
    stage_dir, back = STAGE_ROOT / f"stage-{stamp}", STAGE_ROOT / f"back-{stamp}"
    try:
        stage(stage_dir, run, machine, getattr(args, "tree_commit", None))
        upload = staged_bytes(stage_dir)
        stages = [("prepare", " && ".join(job_setup(run, machine)) if warm
                   else prepare_command(run, machine),
                   PREPARE_MINUTES),
                  ("run", run_command(run, runs if batch else None), args.run_minutes)]
        if args.dry_run:
            print(f"upload {upload / 1e9:.2f} GB, root {root_gb(upload)} GB")
            for name, command, minutes in stages:
                print(f"{name} ({minutes} min): {command}")
            return None
        expected = local_decodes(run)
        copies = {k: v for r in runs for k, v in r.copies.items()}
        try:
            cloud = rent(args, warm, machine, types, upload, session_factory, stage_dir, back,
                         stages, expected, runs, layout, copies)
        except NoCapacity as error:
            if machine != ARM or arch != "auto":
                raise
            say(f"no {ARM_FAMILY} box ({error}): the job goes to an x86 box")
            unstage(stage_dir)
            return measure(argparse.Namespace(**{**vars(args), "arch": X86}), session_factory)
        cloud["over_budget"] = report_phases(
            {k: v * 60 for k, v in cloud["stage_minutes"].items()},
            phase_budgets(cloud.get("points_kept", 0) or job_points(runs)))
        kept = cloud.pop("points_kept", 0)  # merged as they came back (run_stages)
        good = [r for r in runs if result_status(back / result_name(r)) == "done"]
        results = [finish(r, back, cloud, runs if r is good[0] else None) for r in good]
        failures = [r for r in runs if r not in good]
        added = kept + merge_points(back)
        say(f"{added} cloud points kept in {CLOUD_WORK / WORK_SUBDIR}")
        say(describe_parity(parity(CLOUD_WORK / WORK_SUBDIR, load_paths().work / WORK_SUBDIR,
                                   {c.name for c in run.clips})))
        for r in failures:
            say(f"FAILED on the box: {shlex.join(r.command.argv())[:160]} "
                f"({result_status(back / result_name(r))}); log: {r.layout.log('run')}")
        if capped(cloud.get("failed")):
            raise SystemExit(
                f"stopped at the run cap ({args.run_minutes:g} min, --run-minutes): partial "
                f"results in {CLOUD_WORK / WORK_SUBDIR} ({added} points) and "
                f"{len(results)} of {len(runs)} result(s) back; the same command run again "
                f"skips those points on a one-off box (seeded) or the same warm box (its own "
                f"cache), and recomputes them on another warm box")
        if failures:
            raise SystemExit(f"{len(failures)} of {len(runs)} command(s) failed on the box "
                             f"({cloud.get('failed') or 'no result'}); the other "
                             f"{len(results)} result(s) are back and recorded")
        return results[0] if not batch else {"results": results, "cloud": cloud}
    finally:
        unstage(stage_dir)
        remove_back(back)


def fit_to_run_cap(args: argparse.Namespace, runs: list[Plan], commands: list[list[str]],
                   types, machine: str, session_factory) -> tuple[dict | None, tuple]:
    """Check the run against the run cap before anything launches (2026-11-24: an 8-screen
    L3-ladder batch on one c8g.12xlarge hit the 35-min cap with none of its 8 results back).
    Returns (None, the types to run on): it fits on them, or on the machine's largest auto
    size, which it then asks for. A batch that fits on no one box runs as sub-batches that
    each do (split_to_fit; each on its own box, side by side, each within --budget):
    (their merged results, types). A command that cannot fit alone is refused (SystemExit)."""
    if any((r.command.module, r.command.sub) not in GPU_COMMANDS for r in runs):
        return None, types  # a study's slices and the oracles bound their own time
    import aws_warm
    vcpus = min(aws_warm.type_vcpus(t) for t in types)
    limit = args.run_minutes * RUN_FILL
    minutes = run_minutes(runs, vcpus)
    say(f"planned run: ~{minutes:.0f} min on {vcpus} vCPU (limit {limit:.0f} min: "
        f"{RUN_FILL:g} x the {args.run_minutes:g}-min run cap)")
    if minutes <= limit:
        return None, types
    sizes = AUTO_SIZES_ARM if machine == ARM else AUTO_SIZES
    largest, largest_types = max(sizes)
    groups = split_to_fit(runs, max(vcpus, largest), limit)
    if len(groups) == 1:
        if args.instance_type != "auto":
            raise SystemExit(f"planned ~{minutes:.0f} min on {vcpus} vCPU, over {limit:.0f}: "
                             f"give --instance-type auto (it fits on {largest} vCPU)")
        say(f"it fits on {largest} vCPU: asking for that size")
        return None, largest_types
    say(f"the batch is planned over the run cap: {len(groups)} sub-batches on their own "
        f"boxes, " + ", ".join(f"{len(g)} command(s) "
                               f"~{run_minutes([runs[i] for i in g], largest):.0f} min"
                               for g in groups))
    return run_sub_batches(args, [[commands[i] for i in g] for g in groups],
                           session_factory), largest_types


def run_sub_batches(args: argparse.Namespace, groups: list[list[list[str]]],
                    session_factory) -> dict:
    """Each group as its own --batch, side by side; their results merged."""
    from concurrent.futures import ThreadPoolExecutor
    directory = STAGE_ROOT / f"back-split-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    directory.mkdir(parents=True)
    files = []
    for index, group in enumerate(groups):
        files.append(directory / f"batch-{index}.json")
        files[-1].write_text(json.dumps(group))

    def one(path: Path) -> dict:
        return measure(argparse.Namespace(**{**vars(args), "batch": str(path)}),
                       session_factory) or {}
    try:
        with ThreadPoolExecutor(len(files)) as pool:
            outcomes = list(pool.map(one, files))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    return {"results": [r for o in outcomes for r in o.get("results", [])],
            "sub_batches": len(files)}


def encode_rounds(args: argparse.Namespace, requests: list[dict],
                  session_factory=None) -> dict | None:
    """One encode round (nceval.gpuscore) for `requests` ({command, prebuilt, seed, have}:
    one command each, a batch's together, 2026-11-24: one round per command launched 14
    boxes for a 26-screen batch): the commands encode-only side by side on one warm box,
    each one's streams pulled into its own prebuilt directory as they finish. Returns the
    session's facts, also appended to `args.rounds_log`."""
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    layout = warm_layout(stamp)
    runs = [plan(Command.parse(r["command"]), i, layout) for i, r in enumerate(requests)]
    for run in runs:
        run.encode_only = True
    run = merged(runs)
    run.encode_only = True
    arch = getattr(args, "arch", DEFAULT_ARCH)
    machine = choose_machine(arch, runs, run.arms)
    say(f"encode round: {len(runs)} command(s), {len(run.clips)} clips, rotation "
        f"{run.rotation}, one warm box ({machine})")
    stage_dir, back = STAGE_ROOT / f"stage-{stamp}", STAGE_ROOT / f"back-{stamp}"
    try:
        stage(stage_dir, run, machine, getattr(args, "tree_commit", None))
        job_home = stage_dir / Path(layout.root).relative_to(REMOTE_HOME)
        (job_home / "seed").mkdir(parents=True, exist_ok=True)
        for index, request in enumerate(requests):
            (job_home / "seed" / str(index)).symlink_to(Path(request["seed"]).resolve())
            (job_home / f"have-{index}.txt").symlink_to(Path(request["have"]).resolve())
        stages = [("prepare", " && ".join(job_setup(run, machine)
                                          + encode_setup(layout, len(requests))),
                   PREPARE_MINUTES),
                  ("run", run_command(run, runs), args.run_minutes)]
        if args.dry_run:
            for name, command, minutes in stages:
                print(f"{name} ({minutes} min): {command}")
            return None
        pull = Pull([Path(r["prebuilt"]) for r in requests], [Path(r["have"]) for r in requests])
        try:
            cloud = rent(args, True, machine, (), 0, session_factory, stage_dir, back, stages,
                         local_decodes(run), runs, layout, {}, pull)
        except NoCapacity as error:
            if machine != ARM or arch != "auto":
                raise
            say(f"no {ARM_FAMILY} box ({error}): the round goes to an x86 box")
            unstage(stage_dir)
            return encode_rounds(argparse.Namespace(**{**vars(args), "arch": X86}), requests,
                                 session_factory)
        cloud["commands"] = len(requests)
        cloud["points_back"] = cloud.pop("points_kept", 0)
        # the budget's point count: the planned points (points pulled during the run are
        # merged as they come and not all counted in points_back)
        cloud["points_planned"] = job_points(runs)
        say(f"encode round: {len(requests)} command(s), {cloud['points_back']} points "
            f"back, ~${cloud['estimated_usd']}")
        cloud["over_budget"] = report_phases(
            {k: v * 60 for k, v in cloud["stage_minutes"].items()},
            phase_budgets(max(cloud["points_back"], cloud["points_planned"])))
        with Path(args.rounds_log).open("a") as log:
            log.write(json.dumps(cloud) + "\n")
        return cloud
    finally:
        unstage(stage_dir)
        remove_back(back)


def round_requests(args: argparse.Namespace) -> list[dict]:
    """The encode round's requests: `--encode-rounds FILE`, or the one of
    `--encode-round DIR --seed --have -- COMMAND`."""
    if args.encode_rounds:
        return json.loads(Path(args.encode_rounds).read_text())
    return [{"command": args.command, "prebuilt": args.encode_round, "seed": args.seed,
             "have": args.have}]


def gpu_scored(commands: list[list[str]]) -> bool:
    """Whether these commands are scored on the GPU (GPU_COMMANDS); refuses a batch that
    mixes them with commands only a box can run (it would score the GPU commands on a
    box's CPU)."""
    gpu = [(parsed.module, parsed.sub) in GPU_COMMANDS
           for parsed in map(Command.parse, map(list, commands))]
    if any(gpu) and not all(gpu):
        raise SystemExit("refused: the batch mixes commands scored on this PC's GPU ("
                         + ", ".join(" ".join(filter(None, c)) for c in GPU_COMMANDS)
                         + ") with commands only a box runs; split it into two batches")
    return all(gpu)


def gpu_score(args: argparse.Namespace, commands: list[list[str]]) -> list[dict | str]:
    """The commands scored here on the GPU, their points encoded on boxes (nceval.gpuscore),
    all in one bounded scope; per command its result, or why the GPU could not score it (the
    caller refuses): every command's, when the GPU may not be used now at all."""
    every = lambda reason: [reason] * len(commands)
    lock = gpuscore.GpuQueue()
    wait = getattr(args, "gpu_wait", None)
    say("waiting for a place in the GPU scoring service")
    wait_s = (gpuscore.QUEUE_WAIT_MINUTES if wait is None else wait) * 60
    deadline = time.monotonic() + wait_s
    busy = lock.acquire(wait_s=wait_s)
    if busy:
        return every(f"queue wait: the GPU scoring service was full for {wait_s / 60:g} min")
    try:
        # memory in the slice for this command or batch, within the same wait
        while (short := gpuscore.memory_reason(len(commands))) and time.monotonic() < deadline:
            time.sleep(gpuscore.QUEUE_POLL_SECONDS)
        if short:
            return every(f"service memory: {short}")
        reason = gpuscore.local_reason(load_paths().work, lambda: gpuscore.gate_reason(
            gpuscore.START_FREE_MIB))
        if reason:
            return every(reason)
        prune_prebuilt()
        entries = [scoring_entry(argv) for argv in commands]
        budget = getattr(args, "budget", None)
        planned = planned_rounds_usd([command for command, _, _ in entries])
        say(f"planned encode rounds: one warm {ARM_FAMILY} box at a time, a first round of "
            f"{len(entries)} command(s) ~${planned:.3f} (budget ${budget})")
        if budget is not None and planned > budget:
            raise SystemExit(f"the planned encode rounds (~${planned:.3f}) pass --budget "
                             f"${budget:g}: nothing launched")
        facts = run_scoring([(command, home) for command, home, _ in entries],
                            getattr(args, "tree_commit", None), budget)
        results = []
        for (command, home, fields), fact in zip(entries, facts):
            if isinstance(fact, str):  # refused (why): its streams are kept for a retry
                results.append(fact)
                continue
            shutil.rmtree(home, ignore_errors=True)
            out = Path(command.get("--json"))
            result = json.loads(out.read_text())
            result.update(fields)
            result["gpu_score"] = fact
            result["scored_on"] = SCORED_GPU
            # what train and the ledger read of a box run: this run's own box cost
            result["cloud"] = {"instance_type": f"{ARM_FAMILY} encode rounds + local GPU",
                               "estimated_usd": fact["rounds_usd"],
                               "stage_minutes": {"run": round(fact["wall_s"] / 60, 2)},
                               "metrics_backend": VMAFX_CUDA_BACKEND}
            out.write_text(json.dumps(result, indent=2, default=str) + "\n")
            say(f"result: {out}")
            results.append(result)
        return results
    finally:
        lock.release()


#: The planned cost of a GPU batch's first round (`planned_rounds_usd`): a warm c8g box's
#: price with its root volume and IPv4 (docs/aws-cost-guide.md), a new box's launch and
#: setup, and encodes per box-minute (c8g.8xlarge: 71,068 150-frame encodes an hour,
#: 2026-11-24), at the points the commands plan (job_points) and their frames.
ROUND_BOX_USD_PER_HOUR = 0.25
ROUND_SETUP_MINUTES = 6.0
ROUND_ENCODES_PER_MINUTE = 71068 / 60


#: Each phase's expected seconds (owner, 2026-11-25: inefficiencies are found by the tools,
#: not by reading narration). A phase over OVER_BUDGET x its budget prints one loud line.
#: launch+setup: a fresh box from the baked image (measure_image; ~7 min on the stock image).
PHASE_BUDGET_S = {"launch+setup": 240.0, "box": 60.0, "sync": 120.0, "upload": 240.0,
                  "prepare": 180.0, "download": 120.0}
#: The PC's GPU scoring, per point (2026-11-25: 1,041 quick points in 9.5 min, 0.55 s), and
#: at least a minute (the service's start, a small batch).
SCORE_BUDGET_S_PER_POINT = 0.6
SCORE_BUDGET_FLOOR_S = 60.0
#: An encode round's run phase: the points at ROUND_ENCODES_PER_MINUTE, plus a minute.
RUN_BUDGET_FLOOR_S = 60.0
OVER_BUDGET = 1.5


def phase_budgets(points: int) -> dict[str, float]:
    """Every phase's budget for a round or job that brings back `points` points."""
    return {**PHASE_BUDGET_S, "run": RUN_BUDGET_FLOOR_S + 60.0 * points / ROUND_ENCODES_PER_MINUTE}


def report_phases(seconds: dict[str, float], budgets: dict[str, float]) -> list[str]:
    """Print the phase timeline with each budget; one loud line per phase over
    OVER_BUDGET x its budget. Returns the phases over."""
    say("phases: " + ", ".join(
        f"{name} {secs:.0f} s" + (f" (budget {budgets[name]:.0f})" if name in budgets else "")
        for name, secs in seconds.items()))
    over = [name for name, secs in seconds.items()
            if budgets.get(name) and secs > OVER_BUDGET * budgets[name]]
    for name in over:
        say(f"OVER BUDGET: {name} took {seconds[name]:.0f} s, {seconds[name] / budgets[name]:.1f}x "
            f"its budget of {budgets[name]:.0f} s")
    return over


def planned_rounds_usd(commands: list[Command]) -> float:
    """What the first encode round of these commands should cost: one box, set up anew,
    for their planned points (an upper estimate: a warm box needs no setup)."""
    runs = [plan(Command.parse(c.argv()), i) for i, c in enumerate(commands)]
    minutes = ROUND_SETUP_MINUTES + job_points(runs) / ROUND_ENCODES_PER_MINUTE
    return ROUND_BOX_USD_PER_HOUR * minutes / 60


def scoring_entry(argv: list[str]) -> tuple[Command, Path, dict]:
    """One command as the GPU path runs it: its draw pinned, final-check's commits
    resolved here (the landed-build cache, never a local compile; its git checks run
    here, as for a box), its streams' home (a retry's kept streams reused); the fields
    put back into its result."""
    command = Command.parse(list(argv))
    command.set("--rotation", command.get("--rotation") or (
        fixedqp.default_rotation() if command.module == "nceval.fixedqp" else pool.rotation_key()))
    fields = {}
    if command.module == "nceval.levels":
        arms, fields = final_check_arms(command, load_paths())
        for arm in arms:
            command.set(arm.option, str(arm.binary))
    if not command.get("--json"):
        raise SystemExit("give the command --json OUT: the result is written there")
    home = prebuilt_home(command)
    (home / PREBUILT).mkdir(parents=True, exist_ok=True)
    kept = len(gpuscore.done_keys(home / PREBUILT))
    if kept:
        say(f"{kept} streams encoded by an earlier attempt of this command are reused")
    return command, home, fields


def prebuilt_home(command: Command) -> Path:
    """A GPU-scored command's streams, named by the command (its rotation pinned): kept
    when the run falls back or fails, so the same command's next attempt reuses them
    (2026-11-24: a quick lost its encoded streams to a local guard). The keys are the
    encodes' own identities, so a stream there is only ever used for its own point."""
    digest = hashlib.sha256(json.dumps(command.argv()).encode()).hexdigest()[:16]
    return STAGE_ROOT / f"{GPU_PREBUILT_PREFIX}{digest}"


def prune_prebuilt(root: Path | None = None, now: float | None = None) -> int:
    """Remove kept GPU streams older than PREBUILT_KEEP_HOURS (no retry comes that late);
    returns how many directories went."""
    root = root or STAGE_ROOT
    cutoff = (now or time.time()) - PREBUILT_KEEP_HOURS * 3600
    gone = 0
    for entry in sorted(root.glob(f"{GPU_PREBUILT_PREFIX}*")) if root.is_dir() else ():
        if entry.stat().st_mtime < cutoff and retention.guard(entry, "aws_measure") is None:
            shutil.rmtree(entry, ignore_errors=True)
            gone += 1
    return gone


def run_scoring(entries: list[tuple[Command, Path]], tree_commit: str | None = None,
                budget_usd: float | None = None) -> list[dict | str]:
    """The commands here, side by side in one scope in the scoring service's slice
    (gpuscore.SERVICE_SLICE: one CPU quota and memory cap with every other GPU-scored
    command), scoring with VMAFx CUDA on one server; their encode rounds go to boxes. The
    machine-wide GPU slots (LOCAL_SLOTS) queue their points for the server. Returns each command's facts (the
    scope's PC CPU and memory, which also hold the VMAFx server it starts; its rounds and
    their cost), or why the GPU could not score one (gpuscore.UNAVAILABLE_EXIT)."""
    unit = f"nanocodec-gpu-score-{os.getpid()}"
    slots = str(gpuscore.LOCAL_SLOTS)
    common = {**os.environ, METRICS_BACKEND_ENV: VMAFX_CUDA_BACKEND,
              POOL_SIZE_ENV: str(gpuscore.VMAFX_SERVERS), levels.GPU_SLOTS_ENV: slots,
              MAX_WORKERS_ENV: str(gpuscore.worker_share())}
    common.pop(ON_BOX_ENV, None)
    from nceval.cloud import pinned_tree
    common[gpuscore.TREE_COMMIT_ENV] = tree_commit or pinned_tree()
    if budget_usd is not None:
        common[gpuscore.ROUNDS_BUDGET_ENV] = str(budget_usd)
    spec = [{"argv": [sys.executable, "-m", command.module,
                      *([command.sub] if command.sub else []), *command.args, LOCAL_FLAG],
             "env": {gpuscore.PREBUILT_ENV: str(home / PREBUILT),
                     gpuscore.ROUND_COMMAND_ENV: json.dumps(command.argv()),
                     gpuscore.UNAVAILABLE_REASON_ENV: str(home / UNAVAILABLE_REASON),
                     levers.SCORED_ON_ENV: SCORED_GPU},
             "status": str(home / "status")}
            for command, home in entries]
    # the run's spec, its round broker and rounds log; swept once this process is gone
    batch = STAGE_ROOT / f"back-gpubatch-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}" \
        / "batch.json"
    batch.parent.mkdir(parents=True)
    batch.write_text(json.dumps(spec))
    gpuscore.ensure_slice()
    argv = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
            f"--slice={gpuscore.SERVICE_SLICE}",
            sys.executable, "-m", "nceval.gpuscore", "run", str(batch)]
    say(f"GPU scoring here ({unit}: {len(entries)} command(s), in the scoring service "
        f"{gpuscore.SERVICE_SLICE}: CPUQuota {gpuscore.SCOPE_CPU_QUOTA}, MemoryMax "
        f"{gpuscore.SCOPE_MEMORY_MAX}, {slots} GPU slots, one VMAFx server, "
        f"{gpuscore.worker_share()} workers a command); encodes on boxes")
    started = time.monotonic()
    service_before = scope_usage(gpuscore.SERVICE_SLICE).get("pc_cpu_s", 0.0)
    subprocess.call(argv, cwd=REPO / "eval", env=common)
    wall = time.monotonic() - started
    usage = scope_usage(f"{unit}.scope")
    # the whole service's CPU over this run (other commands in it included): a scope that
    # ends at once (the VMAFx server already up in another) has no accounting of its own
    usage["service_cpu_s"] = round(scope_usage(gpuscore.SERVICE_SLICE).get("pc_cpu_s", 0.0)
                                   - service_before, 1)
    rounds_log = batch.parent / ROUNDS_LOG
    rounds = [json.loads(line) for line in rounds_log.read_text().splitlines()] \
        if rounds_log.exists() else []
    rounds_usd = round(sum(r.get("estimated_usd") or 0.0 for r in rounds), 3)
    tails = [r for r in rounds if "idle_tail_s" in r]  # a released box's idle time
    rounds = [r for r in rounds if "idle_tail_s" not in r]
    say(f"the batch's encode rounds: {len(rounds)} round(s), "
        f"{len({r.get('warm_box') for r in rounds})} box(es), ${rounds_usd} with "
        f"{sum(r['idle_tail_s'] for r in tails)} s of idle tail")
    round_s = sum(60 * sum(r.get("stage_minutes", {}).values()) for r in rounds)
    counts = [max(r.get("points_back", 0), r.get("points_planned", 0)) for r in rounds]
    scoring = max(SCORE_BUDGET_FLOOR_S, SCORE_BUDGET_S_PER_POINT * sum(counts))
    report_phases({"encode rounds": round_s, "PC scoring and waits": max(0.0, wall - round_s),
                   "batch": wall},
                  {"PC scoring and waits": scoring,
                   "batch": sum(sum(phase_budgets(n).values()) for n in counts) + scoring})
    shutil.rmtree(batch.parent, ignore_errors=True)
    out: list[dict | None] = []
    for command, home in entries:
        status_file = home / "status"
        status = int(status_file.read_text()) if status_file.exists() else 1
        status_file.unlink(missing_ok=True)
        if status == gpuscore.UNAVAILABLE_EXIT:
            why = home / UNAVAILABLE_REASON
            out.append(why.read_text().strip() if why.exists() else "guard: during the run")
            why.unlink(missing_ok=True)
            continue
        if status:
            raise SystemExit(f"the GPU-scored run failed (status {status}): "
                             f"{shlex.join(command.argv())[:160]}")
        facts = {"wall_s": round(wall, 1), **usage, "commands_in_scope": len(entries),
                 "box_encoded_points": len(gpuscore.done_keys(home / PREBUILT)),
                 # the batch's rounds are shared: each command carries its share, and the
                 # batch's totals beside it
                 "rounds_usd": round(rounds_usd / len(entries), 4),
                 "batch_rounds": len(rounds), "batch_rounds_usd": rounds_usd,
                 "batch_boxes": len({r.get("warm_box") for r in rounds}),
                 "metrics_backend": VMAFX_CUDA_BACKEND}
        say("GPU scoring: " + ", ".join(f"{k} {v}" for k, v in facts.items()))
        out.append(facts)
    return out


def scope_usage(unit: str) -> dict:
    """A unit's (scope's or slice's) CPU seconds and peak memory so far (systemd
    accounting); empty when systemd does not report them (the scope already gone)."""
    shown = subprocess.run(["systemctl", "--user", "show", unit, "-p",
                            "CPUUsageNSec", "-p", "MemoryPeak"], capture_output=True,
                           text=True).stdout
    values = dict(line.split("=", 1) for line in shown.splitlines() if "=" in line)
    facts = {}
    if values.get("CPUUsageNSec", "").isdigit():
        facts["pc_cpu_s"] = round(int(values["CPUUsageNSec"]) / 1e9, 1)
    if values.get("MemoryPeak", "").isdigit():
        facts["pc_peak_mb"] = round(int(values["MemoryPeak"]) / 1e6)
    return facts


def rent(args, warm: bool, machine: str, types, upload: int, session_factory,
         stage_dir: Path, back: Path, stages: list, expected: dict[str, str],
         runs: list[Plan], layout: Layout, copies: dict[str, Path],
         pull: Pull | None = None) -> dict:
    """The job on a warm box (another one after a failed sync or a reclaim) or a one-off
    box; NoCapacity when no box of the machine launches."""
    if not warm:
        return cold_session(args, types, upload, session_factory, stage_dir, back, stages,
                            expected, runs, copies, machine)
    avoid: set[str] = set()
    for attempt in range(1, SESSION_ATTEMPTS + 1):
        try:
            return warm_session(args, stage_dir, back, stages, expected,
                                [r.remote_json for r in runs], layout, avoid, copies, machine,
                                pull, min_vcpus(args, types))
        except SyncFailed as error:
            if attempt == SESSION_ATTEMPTS:
                raise
            avoid.add(error.box)
            say(f"{error}: the job goes to another warm box")
        except RemoteFailure as error:
            if attempt == SESSION_ATTEMPTS or not reclaimed(error):
                raise
            # the next acquire reaps the reclaimed box from the registry; the points pulled
            # back from it (run_stages) go up as the next box's seed, so it re-runs only
            # the points still missing (the job's prepare copies them into its cache);
            # an encode round's streams pulled back join the keys the next box skips
            if pull is not None and pull.targets:
                pull.refresh_have()
            linked = reseed(stage_dir, {c.name for r in runs for c in r.clips})
            say(f"the warm box was reclaimed (spot): the job goes to another box, "
                f"attempt {attempt + 1} of {SESSION_ATTEMPTS}, seeded with the points "
                f"pulled back ({linked} point directories linked)")
    raise AssertionError("unreachable: the last attempt returns or raises")


def cold_session(args, types, upload: int, session_factory, stage_dir: Path, back: Path,
                 stages: list, expected: dict[str, str], runs: list[Plan],
                 copies: dict[str, Path] | None = None, machine: str = X86) -> dict:
    """One one-off box for the job, terminated at its end; a spot reclaim gets one more."""
    if session_factory is None:
        from aws_session import Session as session_factory  # boto3 only here
    from aws_session import SessionSpec
    spec = SessionSpec(region=args.region, instance_type=types[0],
                       arch=AMI_ARCH[machine], purpose="measure", lifetime_minutes=args.lifetime,
                       budget_usd=args.budget, root_gb=root_gb(upload),
                       fallback_types=() if args.no_fallback else (
                           types[1:] if args.instance_type == "auto" or len(types) > 1
                           else fallbacks_after(types[0]) if machine == X86 else ()))
    say(f"upload {upload / 1e9:.2f} GB, root volume {spec.root_gb} GB")
    for attempt in range(1, SESSION_ATTEMPTS + 1):
        try:
            cloud = one_session(session_factory(spec), stage_dir, back, stages, expected,
                                [r.remote_json for r in runs], copies)
            break
        except RemoteFailure as error:
            if attempt == SESSION_ATTEMPTS or not reclaimed(error):
                raise
            linked = reseed(stage_dir, {c.name for r in runs for c in r.clips})
            say(f"the box was reclaimed (spot): a new one, attempt {attempt + 1} of "
                f"{SESSION_ATTEMPTS}, seeded with the points pulled back ({linked} new "
                f"point directories linked)")
    return cloud


#: How aws_session's watch words a run that outlived its time limit (--run-minutes).
RUN_CAP_REASON = "no result in"


def run_stage(session, name: str, command: str, log: str, minutes: float) -> str | None:
    """One stage on the box. A failed run (a command of a batch, or the command) and a
    run stopped at the run cap are returned, not raised: what the commands measured
    still comes back and is recorded, and the caller fails after that. A failed
    prepare, a stall or a reclaim raises."""
    try:
        session.run_detached(command, log, minutes, stall_minutes=STALL_MINUTES,
                             max_errors=1 << 30)
        return None
    except RemoteFailure as error:
        if name != "run" or reclaimed(error) or "no progress" in str(error):
            raise
        say(f"the run {'stopped at the run cap' if capped(str(error)) else 'failed'} on "
            f"the box ({error}); bringing back what it measured")
        return str(error)


def capped(failed: str | None) -> bool:
    """Whether a run's failure (run_stage) was the run cap, not the command failing."""
    return bool(failed) and failed.startswith(RUN_CAP_REASON)


def run_stages(session, stages: list, expected: dict[str, str], back: Path,
               timings: dict[str, float], layout: Layout = Layout(),
               pull: "Pull | None" = None) -> tuple[str | None, int]:
    """The stages in order, each one's minutes into `timings`; returns the run's failure
    (run_stage) and the points kept so far. While the run runs, the points it finishes
    come back every PULL_SECONDS (Session.pulling, as aws_heldout_chart's
    rounds do) and go straight into CLOUD_WORK, with one last pull before the box is
    cleaned: a run stopped at the cap, stalled or killed here keeps every point it
    finished (2026-11-22: three jobs past the cap brought nothing back)."""
    failed, kept = None, 0
    pull = pull or Pull()

    def keep(_files: list[Path]) -> None:
        nonlocal kept
        kept += pull.merge(back)
    for name, command, minutes in stages:
        started = time.monotonic()
        say(f"remote: {name}")
        if name == "run":  # after prepare: the checkpoint marker is there
            with session.pulling(pull.remote(layout), back / pull.local, layout.marker,
                                 keep, PULL_SECONDS):
                failed = run_stage(session, name, command, layout.log(name), minutes)
        else:
            run_stage(session, name, command, layout.log(name), minutes)
        timings[name] = time.monotonic() - started
        if name == "prepare":
            check_decodes(session, expected, layout)
    return failed, kept


@dataclass(frozen=True)
class Pull:
    """What a job's run brings back as it goes: the new cloud points (into CLOUD_WORK),
    or an encode round's streams, request i's into the PC's prebuilt directory
    `targets[i]`."""
    targets: list[Path] = field(default_factory=list)
    #: Each request's list of the keys the PC holds (gpuscore.HAVE_FILE; staged as links).
    haves: list[Path] = field(default_factory=list)

    def refresh_have(self) -> None:
        """Before a retry: every stream back so far joins the keys the box skips."""
        for target, have in zip(self.targets, self.haves):
            have.write_text("".join(f"{k}\n" for k in gpuscore.done_keys(target)))

    @property
    def local(self) -> str:
        return PREBUILT if self.targets else WORK_SUBDIR

    def remote(self, layout: Layout) -> str:
        return f"{layout.root}/{PREBUILT}" if self.targets else f"{REMOTE_WORK}/{WORK_SUBDIR}"

    def merge(self, back: Path) -> int:
        if not self.targets:
            return merge_points(back)
        return sum(merge_prebuilt(back / PREBUILT / str(i), target)
                   for i, target in enumerate(self.targets))


def merge_prebuilt(source: Path, target: Path) -> int:
    """An encode round's streams into `target`, each point's `.done` last and only once the
    files it records are there whole (gpuscore.point_whole: a point is complete there only
    with all its files); returns the points completed. A `.done` whose files have not all
    come back stays in `source` for a later pull to complete; when the box is gone first,
    the point is not in `target`'s done keys, so the next box codes it again (the reclaim
    rules, docs/aws-cost-guide.md: retry only what is missing)."""
    if not source.is_dir():
        return 0
    files = sorted((f for f in source.rglob("*") if f.is_file()
                    and f.name != gpuscore.HAVE_FILE and not f.name.startswith(".")),
                   key=lambda f: (f.suffix in (".done", ".failed"), str(f)))
    done = 0
    for file in files:
        destination = target / file.relative_to(source)
        if file.suffix == ".done" and not gpuscore.point_whole(
                file, Path(str(destination)[:-len(".done")])):
            say(f"{destination.name}: held back, its files are not all here whole yet")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(file, destination)
        done += file.suffix in (".done", ".failed")
    return done


class SyncFailed(RuntimeError):
    """The staged files would not go to a warm box (twice): the job tries another box."""

    def __init__(self, box: str, reason: str):
        super().__init__(f"rsync to warm box {box} failed: {reason}")
        self.box = box


SYNC_RETRY_SECONDS = 10


def sync(session, stage_dir: Path) -> None:
    """The staged home onto a warm box: only files it lacks or that changed (rsync,
    symlinks followed, times kept, so a clip already there is not sent again), one
    job's upload at a time per box (aws_warm.sync_lock)."""
    import aws_warm
    remote_shell = " ".join(shlex.quote(a) for a in session.ssh_base()[:-1])
    with aws_warm.sync_lock(session.instance_id):
        say(f"syncing {stage_dir} to the warm box")
        subprocess.run(["rsync", "-rLt", "--partial", "--exclude", "manifest.txt", "-e",
                        remote_shell, f"{stage_dir}/",
                        f"{session.ssh_base()[-1]}:{REMOTE_HOME}/"], check=True)


def min_vcpus(args, types) -> int:
    """The smallest box an `auto`-sized job may take: the smallest of its planned types
    (auto_types), so a warm box too small to finish it inside the run cap is not used."""
    if getattr(args, "instance_type", None) != "auto" or not types:
        return 0
    import aws_warm
    return min(aws_warm.type_vcpus(t) for t in types)


def warm_session(args, stage_dir: Path, back: Path, stages: list, expected: dict[str, str],
                 results: list[str], layout: Layout, avoid: set[str] = frozenset(),
                 copies: dict[str, Path] | None = None, machine: str = X86,
                 pull: Pull | None = None, need_vcpus: int = 0) -> dict:
    """The job on a warm box (aws_warm): a running one with room, else a new one set up
    once (the box's readiness and its idle timer); the job syncs, runs and comes back,
    and the box stays up for the next job."""
    import aws_warm

    def setup(session) -> None:
        sync(session, stage_dir)
        session.run_detached(" && ".join(box_setup(machine)) + " && "
                             + aws_warm.idle_watch_command(box_cleanup_command(),
                                                           args.idle_minutes),
                             f"{REMOTE_HOME}/box-setup.log", PREPARE_MINUTES,
                             stall_minutes=STALL_MINUTES, max_errors=1 << 30)
    timings: dict[str, float] = {}
    started = time.monotonic()
    family = getattr(args, "instance_family", None) or (ARM_FAMILY if machine == ARM else None)
    box, session, fresh = aws_warm.acquire(layout.job, aws_warm.warm_spec(
        args.region, args.warm_budget, family, need_vcpus), setup,
        getattr(args, "warm_spend_cap", None) or aws_warm.SPEND_CAP_USD_PER_HOUR, avoid,
        family, need_vcpus)
    timings["box" if not fresh else "launch+setup"] = time.monotonic() - started
    try:
        session.ssh(f"mkdir -p {layout.root} && touch {layout.root}/{aws_warm.JOB_MARK} "
                    f"{aws_warm.BUSY_FILE}", quiet=True)
        started = time.monotonic()
        if not fresh:
            try:
                sync(session, stage_dir)
            except subprocess.CalledProcessError:  # once more, then another box
                time.sleep(SYNC_RETRY_SECONDS)
                try:
                    sync(session, stage_dir)
                except subprocess.CalledProcessError as error:
                    raise SyncFailed(box.instance_id, str(error)) from error
        timings["sync"] = time.monotonic() - started
        failed, kept = None, 0
        try:
            failed, kept = run_stages(session, stages, expected, back, timings, layout, pull)
        finally:
            started = time.monotonic()
            busy = session.ssh(f"grep -o 'box CPUs busy [0-9]*' {layout.log('run')}",
                               check=False, quiet=True).stdout.split()
            bring_back(session, back, results, layout, copies, pull)
            if pull is not None:
                kept += pull.merge(back)
            timings["download"] = time.monotonic() - started
    finally:
        aws_warm.release(box, layout.job)
    hours = sum(timings.values()) / 3600
    return {"instance_type": session.spec.instance_type, "region": session.spec.region,
            "warm_box": box.instance_id, "fresh_box": fresh, "failed": failed,
            "points_kept": kept,
            "busy_pct": int(busy[-1]) if busy and busy[-1].isdigit() else None,
            "zone": session.zone, "spot_usd_per_hour": session.price,
            "metrics_backend": LIBVMAF_CPU_BACKEND,
            # the box's cost over this job's wall time (an upper bound: jobs share it)
            "estimated_usd": round(session.cost(session.price or 0.0, hours), 3),
            "stage_minutes": {k: round(v / 60, 2) for k, v in timings.items()}}


def one_session(session_cm, stage_dir: Path, back: Path, stages: list,
                expected: dict[str, str], results: list[str] = (REMOTE_JSON,),
                copies: dict[str, Path] | None = None) -> dict:
    """One box: upload, readiness, the run, and back; returns the session's facts."""
    timings: dict[str, float] = {}
    with session_cm as session:
        started = time.monotonic()
        session.upload(stage_dir, REMOTE_HOME)
        timings["upload"] = time.monotonic() - started
        failed, kept = None, 0
        try:
            failed, kept = run_stages(session, stages, expected, back, timings)
        finally:  # whatever the run measured comes back before the box goes
            started = time.monotonic()
            bring_back(session, back, results, copies=copies)
            timings["download"] = time.monotonic() - started
        hours = (datetime.datetime.now(datetime.timezone.utc)
                 - session.launched).total_seconds() / 3600 if session.launched else 0.0
        busy = session.ssh(f"grep -o 'box CPUs busy [0-9]*' {REMOTE_HOME}/run.log",
                           check=False, quiet=True).stdout.split()
        return {"instance_type": session.spec.instance_type, "region": session.spec.region,
                "failed": failed, "points_kept": kept,
                "busy_pct": int(busy[-1]) if busy and busy[-1].isdigit() else None,
                "zone": session.zone, "spot_usd_per_hour": session.price,
                "metrics_backend": LIBVMAF_CPU_BACKEND,
                "estimated_usd": round(session.cost(session.price or 0.0, hours), 3),
                "stage_minutes": {k: round(v / 60, 2) for k, v in timings.items()}}


def reclaimed(error: Exception) -> bool:
    """Whether a remote failure was the box going away (aws_session's watch reports an
    instance that is no longer running so), not the command failing."""
    return "the instance is " in str(error)


def bring_back(session, back: Path, results: list[str] = (REMOTE_JSON,),
               layout: Layout = Layout(), copies: dict[str, Path] | None = None,
               pull: Pull | None = None) -> None:
    """The run's new points, its result and its lever rows into `back`; then the clips,
    builds and points are deleted on the box, whatever came back."""
    back.mkdir(parents=True, exist_ok=True)
    try:
        # The results first: a failed point checkpoint must not lose a finished result.
        for name, remote in ([(Path(r).name, r) for r in results]
                             + [("levers.jsonl", layout.spool)]):
            fetched = session.ssh(f"cat {remote} 2>/dev/null", check=False, quiet=True)
            if fetched.returncode == 0 and fetched.stdout:
                (back / name).write_text(fetched.stdout)
        for remote, local in (copies or {}).items():  # binary files, whole
            if remote.endswith(".sqlite"):  # fold its write-ahead log in first
                session.ssh(f"python3 -c \"import sqlite3; sqlite3.connect('{remote}')"
                            f".execute('PRAGMA wal_checkpoint(TRUNCATE)')\"",
                            check=False, quiet=True)
            fetched = session.ssh(f"base64 -w0 {remote} 2>/dev/null", check=False, quiet=True)
            if fetched.returncode == 0 and fetched.stdout:
                import base64
                local.parent.mkdir(parents=True, exist_ok=True)
                partial = local.with_name(local.name + ".partial")
                partial.write_bytes(base64.b64decode(fetched.stdout))
                for log in ("-wal", "-shm"):  # this machine's own log of the old file
                    local.with_name(local.name + log).unlink(missing_ok=True)
                partial.rename(local)
                say(f"brought back {local}")
        try:
            pull = pull or Pull()
            session.download_new(pull.remote(layout), back / pull.local, layout.marker)
        except RuntimeError as error:  # a box that went away: keep the run's own error
            say(f"no points came back: {error}")
    finally:
        session.ssh(cleanup_command(layout), check=False, quiet=True)
        say("this job's files deleted on the box" if layout.job else
            "clips, builds and points deleted on the box")


def check_decodes(session, expected: dict[str, str], layout: Layout = Layout()) -> None:
    """The box's decodes (in the prepare log) equal this machine's."""
    log = session.ssh(f"cat {layout.log('prepare')}", quiet=True, check=False).stdout
    found = {line.split()[1]: line.split()[0] for line in log.splitlines()
             if len(line.split()) == 2 and len(line.split()[0]) == 64}
    for name, digest in expected.items():
        if found.get(name) != digest:
            raise SystemExit(f"{name}: the box decodes it differently from this machine "
                             f"({found.get(name)} vs {digest}): refusing to measure")
    say(f"the box decodes {len(expected)} clips exactly as this machine does")


def remove_back(back: Path) -> None:
    """The download directory, once its files are merged (or the job failed)."""
    shutil.rmtree(back, ignore_errors=True)


def main(argv: list[str]) -> None:
    from nceval.cloud import check_fresh
    check_fresh()
    # A kill runs the cleanup too (staging, download, the warm box's job slot): Python
    # dies on SIGTERM without unwinding unless told to.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, _frame: sys.exit(f"aws_measure: signal {signum}"))
    freed = sweep_staging()
    if freed:
        say(f"removed the staging of finished jobs: {freed / 1e9:.1f} GB")
    freed = prune_cloud_points()
    if freed:
        say(f"cloud point cache over its cap: {freed / 1e9:.1f} GB of the least recently "
            f"written clips freed")
    split = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser(prog="aws_measure.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance-type", default=DEFAULT_INSTANCE,
                   help="a type, or auto (default): sized to the job (AUTO_SIZES)")
    p.add_argument("--batch", help="a JSON list of commands (argv lists) to run on one box")
    p.add_argument("--cold", action="store_true",
                   help="a one-off box for this job (default: a warm box, tools/dev/aws_warm.py)")
    p.add_argument("--warm", action="store_true", help=argparse.SUPPRESS)  # the default
    p.add_argument("--idle-minutes", type=int, default=aws_warm_idle_minutes(),
                   help="a warm box this job starts stops itself after this long idle")
    # $10 lets a 96-vCPU box start in eu-west-1 or eu-central-1, whose spot is ~2.5x
    # eu-north-1's ($1.43-1.55/h, 2026-11-21: at $6 no type fitted there and jobs waited
    # 25 min for eu-north-1's quota); the box still stops after 20 idle minutes.
    p.add_argument("--warm-budget", type=float, default=10.0,
                   help="a new warm box's worst case (bid cap x its lifetime) must fit this")
    p.add_argument("--warm-spend-cap", type=float,
                   help="$/h all warm boxes may cost together before jobs queue instead of "
                        "launching (default $6, NC_WARM_SPEND_CAP)")
    p.add_argument("--arch", choices=ARCH_CHOICES, default=DEFAULT_ARCH,
                   help=f"the box's machine (default {DEFAULT_ARCH}; auto: {ARM} when the job "
                        f"allows it, docs/aws-cost-guide.md)")
    p.add_argument("--instance-family",
                   help="only boxes of this family (e.g. c7a): a tune study pinned to one CPU")
    p.add_argument("--no-fallback", action="store_true",
                   help="only --instance-type (default: fall back through the throughput ladder)")
    p.add_argument("--region", default=REGION)
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                   help="the session's worst case (bid cap x lifetime) must fit this (USD)")
    p.add_argument("--lifetime", type=int, default=DEFAULT_LIFETIME_MINUTES,
                   help="the box's hard lifetime (minutes)")
    p.add_argument("--run-minutes", type=float, default=RUN_CAP_MINUTES,
                   help="the command's time limit on the box")
    p.add_argument("--dry-run", action="store_true", help="stage and print, rent nothing")
    p.add_argument("--tree-commit", help="the commit the box runs (a long job's pinned tree, "
                   "nceval.cloud.pinned_tree; default: HEAD, refused when uncommitted)")
    p.add_argument("--gpu-wait", type=float, metavar="MINUTES",
                   help="wait up to this long for a place in the GPU scoring service (default "
                        f"{gpuscore.QUEUE_WAIT_MINUTES}), then refuse (never scored on a box)")
    p.add_argument("--encode-round", metavar="DIR", help=argparse.SUPPRESS)  # nceval.gpuscore
    p.add_argument("--encode-rounds", metavar="FILE", help=argparse.SUPPRESS)  # requests
    p.add_argument("--rounds-log", help=argparse.SUPPRESS)
    p.add_argument("--seed", help=argparse.SUPPRESS)
    p.add_argument("--have", help=argparse.SUPPRESS)
    args = p.parse_args(argv[:split])
    args.command = argv[split + 1:]
    if not args.command and not args.batch and not args.encode_rounds:
        p.error("give the nceval command after --, or --batch FILE")
    measure(args)


if __name__ == "__main__":
    main(sys.argv[1:])
