#!/usr/bin/env python3
"""One nceval quality measurement on a rented spot box: the same command, scored there.

    python3 tools/dev/aws_measure.py [--instance-type c7a.48xlarge] [--budget 4] [--dry-run] \\
        -- python3 -m nceval.screen --codec av1 --binary BIN --param a.b=1 --json OUT
    ... -- python3 -m nceval.levels final-check --old C0 --new C1 --codecs av1 --json OUT
    ... -- python3 -m nceval.fixedqp ab --codec av1 --binary BIN --param a.b=1 --json OUT
    ... -- python3 -m nceval.rc_oracle --codec av1 --binary BIN --clips quick --json OUT
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
     run, the builds' version stamps, this machine's ffmpeg (uploaded: another version
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
import datetime
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
from aws_session import RemoteFailure, fallbacks_after, say  # noqa: E402
from nceval import fixedqp, levels, levers, pool, retention  # noqa: E402
from nceval.clipframes import decode_command  # noqa: E402
from nceval.clips import Clip  # noqa: E402
from nceval.config import find_tool, load_paths  # noqa: E402
from nceval.metrics import LIBVMAF_CPU_BACKEND, VMAF_V1_MODEL  # noqa: E402
from nceval.cloud import ON_BOX_ENV  # noqa: E402
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
#: Python packages nceval imports that Ubuntu does not package, installed in a venv over
#: the system's (apt's numpy, scipy, opencv, xxhash).
REMOTE_VENV = f"{REMOTE_HOME}/venv"
#: This machine's ffmpeg on the box (`ffmpeg_links`), and the library directories the box
#: has itself (glibc).
REMOTE_FFMPEG = f"{REMOTE_HOME}/ffmpeg"
#: The ffmpeg programs a measurement runs (decodes, XPSNR; packet sizes).
FFMPEG_TOOLS = ("ffmpeg", "ffprobe")
SYSTEM_LIBS = ("/lib/", "/lib64/", "/usr/lib/")
#: Libraries a linked library loads at run time (no ldd entry): conda's SDL2 is
#: sdl2-compat, which loads SDL3 and makes ffmpeg exit without it (2026-11-21).
DLOPENED = {"libSDL2-2.0.so.0": ("libSDL3.so.0",)}
PIP_PACKAGES = ("optuna",)
CHECKPOINT_MARKER = ".checkpoint"
#: Defaults: a 192-vCPU spot box (not metal: metal boots in 5-10 min), and the ladder
#: after it when it has no capacity (aws_session.THROUGHPUT_X86_TYPES).
REGION = "eu-north-1"
DEFAULT_INSTANCE = "auto"
#: Box sizes `auto` picks from (vCPUs: types, preferred first), smallest first.
AUTO_SIZES = ((32, ("c7a.8xlarge", "c7i.8xlarge", "m7a.8xlarge")),
              (48, ("c7a.12xlarge", "c7i.12xlarge", "m7a.12xlarge")),
              (64, ("c7a.16xlarge", "c7i.16xlarge", "m7a.16xlarge")),
              (96, ("c7a.24xlarge", "c7i.24xlarge", "m7a.24xlarge")))
#: Points each worker should get at least (a box with fewer idles on its tail: the
#: anchor's curves, then the candidate's single points, each end with a part-empty wave).
#: A first guess, to be refined from the runs' "box CPUs busy" lines.
MIN_WAVES = 8
#: CPUs per worker: one point runs our encoder, libvmaf (2 threads) and the Python
#: MS-SSIM and pipe threads.
CPUS_PER_WORKER = 3
#: Stage limits (minutes). A run is held to the project's 30-minute run budget
#: (nceval.budget.MAX_MINUTES) plus RUN_MARGIN_MINUTES for its last points.
PREPARE_MINUTES = 10
RUN_MARGIN_MINUTES = 5
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
           ("nceval.tune", "finalists"), ("nceval.rc_oracle", None)}
BINARY_OPTIONS = {"nceval.screen": ("--binary", "--anchor-binary"),
                  "nceval.quick": ("--binary",),
                  "nceval.tune": ("--binary",),
                  "nceval.levels": ("--old", "--new"),
                  "nceval.fixedqp": ("--binary", "--anchor-binary"),
                  "nceval.rc_oracle": ("--binary",)}
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
    anchors = paths.anchors.resolve()
    for name in (command.get("--families") or "").split(","):
        if name and not name.startswith(fixedqp.OWN_PREFIX):
            family = fixedqp.family_of(name, binary, paths)
            # its encoder, and the standard decoders its points are scored from (an
            # anchor writes no reconstruction: nceval.conformance gates and decodes it)
            for tool in (family.tool, *(d.binary(paths) for d in DECODERS[family.codec])):
                tool = Path(tool).resolve()
                if not tool.is_relative_to(anchors):
                    raise SystemExit(f"fixedqp gap: {name} needs {tool}, not an anchor tool")
                result.inputs[f"{REMOTE_DATA}/anchors/x86_64/{tool.relative_to(anchors)}"] = tool


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
    if command.module == "nceval.rc_oracle":
        from nceval import rc_oracle
        per_class = command.get("--per-class")
        return rc_oracle.clip_list(command.get("--clips") or "val", paths,
                                   int(command.get("--frames") or levels.DEFAULT_FRAMES),
                                   int(per_class) if per_class else None, rotation,
                                   tuple(c for c in (command.get("--classes") or "").split(",")
                                         if c))
    if command.module == "nceval.quick":
        from nceval import quick
        draw = pool.draw(pool.load_pool(paths, levels.usable), rotation, levels.DEFAULT_FRAMES)
        return draw.levels[pool.QUICK] + quick.probe_clips(draw)
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
    from nceval import tune_pool
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
        if run.command.module != "nceval.rc_oracle":
            return 1
        from nceval import rc_oracle
        variants = run.command.get("--variants")
        return 3 + len(variants.split(",") if variants else rc_oracle.VARIANTS) + \
            run.command.args.count("--arm")
    return sum(curves(r) * len(clip_targets(c, LADDER_KBPS)) + 1 for r in runs for c in r.clips
               if isinstance(c, Clip))  # a tune study's cost clips are not ladder clips


def auto_types(points: int) -> tuple[str, ...]:
    """The types `auto` tries for a job of `points`: the largest AUTO_SIZES size whose
    workers each get MIN_WAVES points (the smallest when none does), its other vendors,
    then the next size down."""
    fitting = [i for i, (vcpus, _) in enumerate(AUTO_SIZES)
               if points >= MIN_WAVES * vcpus // CPUS_PER_WORKER]
    chosen = fitting[-1] if fitting else 0
    return AUTO_SIZES[chosen][1] + (AUTO_SIZES[chosen - 1][1] if chosen else ())


# --- staging -----------------------------------------------------------------------------

def stage(stage_dir: Path, run: Plan) -> None:
    """The box's home as symlinks (listed in manifest.txt, so a half-made staging is
    removed too): repo.tar, nanocodec-data/ (clip sets, the clips' media, libvmaf and
    its model), arms/, inputs/ and seed/ (earlier cloud points of these clips)."""
    stage_dir.mkdir(parents=True)
    job_home = stage_dir / Path(run.layout.root).relative_to(REMOTE_HOME)
    archive = job_home / REPO_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_tree(archive)
    links = {stage_dir / "nanocodec-data" / p.relative_to(DATA): p
             for p in clip_files(run.clips)}
    paths = load_paths()
    links |= {stage_dir / "nanocodec-data/clips" / p.name: p
              for p in sorted(paths.clips.glob("*.json"))}
    anchors = stage_dir / "nanocodec-data/anchors/x86_64"
    links[anchors / "bin/vmaf"] = paths.anchors / "bin/vmaf"
    links[anchors / "share/vmaf"] = paths.anchors / "share/vmaf"
    links |= ffmpeg_links(stage_dir / "ffmpeg")
    for arm in run.arms:
        home = stage_dir / Path(arm.remote_dir).relative_to(REMOTE_HOME)
        links[home / "nanocodec"] = arm.binary
        if arm.presets:
            links[home / "presets"] = arm.presets
    links |= {stage_dir / Path(remote).relative_to(REMOTE_HOME): local
              for remote, local in run.inputs.items()}
    names = {c.name for c in run.clips}
    cache = CLOUD_WORK / WORK_SUBDIR
    # A warm box keeps its own point cache: no seed.
    for window in sorted(cache.iterdir()) if cache.is_dir() and not run.layout.job else ():
        for clip in sorted(window.iterdir()):
            if clip.name in names:
                links[stage_dir / "seed" / window.name / clip.name] = clip
    wrappers = [stage_dir / "ffmpeg/bin" / tool for tool in FFMPEG_TOOLS]
    (stage_dir / "manifest.txt").write_text("".join(f"{p}\n"
                                                    for p in [*links, *wrappers, archive]))
    for link, target in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target.resolve())
    wrappers[0].parent.mkdir(parents=True)
    for wrapper in wrappers:
        wrapper.write_text(f"#!/bin/sh\nLD_LIBRARY_PATH={REMOTE_FFMPEG}/lib exec "
                           f"{REMOTE_FFMPEG}/libexec/{wrapper.name} \"$@\"\n")
        wrapper.chmod(0o755)


def ffmpeg_links(home: Path) -> dict[Path, Path]:
    """This machine's ffmpeg and ffprobe and the shared libraries they bring (all but the system's,
    which it finds on the box): the pool's clips are decoded by it, and another ffmpeg
    decodes a variable-rate clip differently (2026-11-21: Ubuntu 26.04's apt ffmpeg gave
    a different frame 143 of a 29.917 fps clip from a rotated start, so the streams
    differed). The box runs it through a wrapper that points it at its libraries only."""
    binaries = {tool: find_tool(tool).resolve() for tool in FFMPEG_TOOLS}
    links = {home / "libexec" / tool: binary for tool, binary in binaries.items()}
    objects = list(binaries.values())
    while objects:
        listed = subprocess.run(["ldd", str(objects.pop())], capture_output=True, text=True,
                                check=True).stdout
        for line in listed.splitlines():
            name, arrow, rest = line.strip().partition(" => ")
            library = Path(rest.split(" (")[0]) if arrow else None
            if library and library.is_absolute() and \
                    not str(library).startswith(SYSTEM_LIBS) and home / "lib" / name not in links:
                links[home / "lib" / name] = library
                for loaded in DLOPENED.get(name, ()):  # loaded at run time, not linked
                    if (library.parent / loaded).exists():
                        links[home / "lib" / loaded] = library.parent / loaded
                        objects.append(library.parent / loaded)
    return links


def archive_tree(archive: Path) -> None:
    """This tree at HEAD, refusing uncommitted eval/, tools/ or presets: the box runs HEAD."""
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "eval",
                            "tools", levels.PRESETS_PATH], capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"commit eval/, tools/ and the presets first: the box runs this "
                         f"tree's HEAD ({dirty.splitlines()[0]} ...)")
    subprocess.run(["git", "-C", str(REPO), "archive", "-o", str(archive), "HEAD"], check=True)


def unstage(stage_dir: Path) -> None:
    """Remove the staging tree: its symlinks, its own copies (repo.tar, ffmpeg) and
    directories. rmtree removes a symlink itself, never what it points to. (It removed
    only the manifest's entries and then rmdir'd, which failed on the copies: 27 jobs'
    staging, 5.5 GB, were left on 2026-11-21.)"""
    shutil.rmtree(stage_dir, ignore_errors=True)


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
        total += sum(os.stat(os.path.join(directory, name)).st_size for name in files)
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


def box_setup() -> list[str]:
    """A box's own readiness, once: packages, the environment, the point cache's tmpfs,
    and that this machine's ffmpeg and libvmaf (uploaded) run there."""
    return [
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y python3-numpy "
        "python3-scipy python3-opencv python3-xxhash python3-venv rsync > /dev/null",
        # what apt does not package (tune's optuna, imported by nceval.screen), beside apt's
        f"python3 -m venv --system-site-packages {REMOTE_VENV}",
        f"{REMOTE_VENV}/bin/pip install -q {' '.join(PIP_PACKAGES)}",
        f"sudo mkdir -p {REMOTE_WORK}",
        f"sudo mount -t tmpfs -o size={REMOTE_WORK_GB}g,mode=1777 tmpfs {REMOTE_WORK}",
        f"mkdir -p {REMOTE_WORK}/{WORK_SUBDIR}",
        f"test \"$({REMOTE_FFMPEG}/bin/ffmpeg -version | head -1)\" = "
        f"{shlex.quote(ffmpeg_version())}",
        f"{REMOTE_FFMPEG}/bin/ffmpeg -hide_banner -filters | grep -qw xpsnr",
        f"{REMOTE_FFMPEG}/bin/ffprobe -version > /dev/null",
        f"! ldd {REMOTE_DATA}/anchors/x86_64/bin/vmaf | grep 'not found'",
        f"{REMOTE_DATA}/anchors/x86_64/bin/vmaf --version",
        f"test -f {REMOTE_DATA}/anchors/x86_64/{VMAF_V1_MODEL}",
        "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs",
    ]


def job_setup(run: Plan) -> list[str]:
    """One job's readiness: its tree, (one-off box) the seed, the checkpoint its points
    are brought back from, each build's version stamp is the one here, the metrics
    backend imports, and the first clips decode to what they decode to here."""
    layout = run.layout
    versions = [(arm.remote_binary, version_of(arm.binary)) for arm in run.arms]
    decodes = [f"{shlex.join(remote_decode(c))} | sha256sum | sed 's/-$/{c.name}/'"
               for c in decode_checks(run.clips)]
    seed = [] if layout.job else [f"mkdir -p {REMOTE_SEED}",
                                  f"cp -r {REMOTE_SEED}/. {REMOTE_WORK}/{WORK_SUBDIR}/"]
    # a tune study's CPU objective is callgrind (once per box)
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
        "echo cpus $(nproc) workers $(( $(nproc) / %d ))" % CPUS_PER_WORKER,
    ]


def prepare_command(run: Plan) -> str:
    """Readiness on a one-off box, failing fast: the box's, then the job's."""
    return " && ".join(box_setup() + job_setup(run))


def ffmpeg_version() -> str:
    return subprocess.run([str(find_tool("ffmpeg")), "-version"], capture_output=True,
                          text=True, check=True).stdout.splitlines()[0]


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
        check=True).stdout).hexdigest() for c in checks}


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
    fields decided here); returns it. `runs`: the batch whose lever rows (one spool on
    the box) are recorded here with it, each localised by its own run."""
    result = json.loads(localise((back / result_name(run)).read_text(), run))
    result.update(run.report_fields)
    result["cloud"] = cloud
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
    commands = json.loads(Path(batch).read_text()) if batch else [args.command]
    warm = session_factory is None and not getattr(args, "cold", False)
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    layout = warm_layout(stamp) if warm else Layout()
    runs = [plan(Command.parse(c), i if batch else None, layout)
            for i, c in enumerate(commands)]
    run = merged(runs) if batch else runs[0]
    files = clip_files(run.clips)
    points = job_points(runs)
    types = auto_types(points) if args.instance_type == "auto" else (args.instance_type,)
    if getattr(args, "instance_family", None):
        import aws_warm
        types = aws_warm.family_types(args.instance_family)
    say(f"{' + '.join(f'{r.command.module} {r.command.sub or ""}'.strip() for r in runs)}: "
        f"{len(run.clips)} clips ({sum(p.stat().st_size for p in files) / 1e9:.1f} GB), "
        f"{len(run.arms)} builds, rotation {run.rotation}, ~{points} points: "
        + ("a warm box" if warm else types[0]))
    stage_dir, back = STAGE_ROOT / f"stage-{stamp}", STAGE_ROOT / f"back-{stamp}"
    try:
        stage(stage_dir, run)
        upload = staged_bytes(stage_dir)
        stages = [("prepare", " && ".join(job_setup(run)) if warm else prepare_command(run),
                   PREPARE_MINUTES),
                  ("run", run_command(run, runs if batch else None), args.run_minutes)]
        if args.dry_run:
            print(f"upload {upload / 1e9:.2f} GB, root {root_gb(upload)} GB")
            for name, command, minutes in stages:
                print(f"{name} ({minutes} min): {command}")
            return None
        expected = local_decodes(run)
        copies = {k: v for r in runs for k, v in r.copies.items()}
        if warm:
            avoid: set[str] = set()
            for attempt in range(1, SESSION_ATTEMPTS + 1):
                try:
                    cloud = warm_session(args, stage_dir, back, stages, expected,
                                         [r.remote_json for r in runs], layout, avoid,
                                         copies)
                    break
                except SyncFailed as error:
                    if attempt == SESSION_ATTEMPTS:
                        raise
                    avoid.add(error.box)
                    say(f"{error}: the job goes to another warm box")
                    continue
                except RemoteFailure as error:
                    if attempt == SESSION_ATTEMPTS or not reclaimed(error):
                        raise
                    # the next acquire reaps the reclaimed box from the registry
                    say(f"the warm box was reclaimed (spot): the job goes to another box, "
                        f"attempt {attempt + 1} of {SESSION_ATTEMPTS}")
        else:
            cloud = cold_session(args, types, upload, session_factory, stage_dir, back, stages,
                                 expected, runs, copies)
        say("stage minutes: " + ", ".join(f"{k} {v}" for k, v in cloud["stage_minutes"].items()))
        good = [r for r in runs if result_status(back / result_name(r)) == "done"]
        results = [finish(r, back, cloud, runs if r is good[0] else None) for r in good]
        failures = [r for r in runs if r not in good]
        added = merge_points(back)
        say(f"{added} cloud points kept in {CLOUD_WORK / WORK_SUBDIR}")
        say(describe_parity(parity(CLOUD_WORK / WORK_SUBDIR, load_paths().work / WORK_SUBDIR,
                                   {c.name for c in run.clips})))
        for r in failures:
            say(f"FAILED on the box: {shlex.join(r.command.argv())[:160]} "
                f"({result_status(back / result_name(r))}); log: {r.layout.log('run')}")
        if failures:
            raise SystemExit(f"{len(failures)} of {len(runs)} command(s) failed on the box "
                             f"({cloud.get('failed') or 'no result'}); the other "
                             f"{len(results)} result(s) are back and recorded")
        return results[0] if not batch else {"results": results, "cloud": cloud}
    finally:
        unstage(stage_dir)
        remove_back(back)


def cold_session(args, types, upload: int, session_factory, stage_dir: Path, back: Path,
                 stages: list, expected: dict[str, str], runs: list[Plan],
                 copies: dict[str, Path] | None = None) -> dict:
    """One one-off box for the job, terminated at its end; a spot reclaim gets one more."""
    if session_factory is None:
        from aws_session import Session as session_factory  # boto3 only here
    from aws_session import SessionSpec
    spec = SessionSpec(region=args.region, instance_type=types[0],
                       arch="amd64", purpose="measure", lifetime_minutes=args.lifetime,
                       budget_usd=args.budget, root_gb=root_gb(upload),
                       fallback_types=() if args.no_fallback else (
                           types[1:] if args.instance_type == "auto" or len(types) > 1
                           else fallbacks_after(types[0])))
    say(f"upload {upload / 1e9:.2f} GB, root volume {spec.root_gb} GB")
    for attempt in range(1, SESSION_ATTEMPTS + 1):
        try:
            cloud = one_session(session_factory(spec), stage_dir, back, stages, expected,
                                [r.remote_json for r in runs], copies)
            break
        except RemoteFailure as error:
            if attempt == SESSION_ATTEMPTS or not reclaimed(error):
                raise
            say(f"the box was reclaimed (spot): a new one, attempt {attempt + 1} of "
                f"{SESSION_ATTEMPTS}")
    return cloud


def run_stage(session, name: str, command: str, log: str, minutes: float) -> str | None:
    """One stage on the box. A failed run (a command of a batch, or the command) is
    returned, not raised: what the other commands measured still comes back and is
    recorded, and the caller fails after that. A failed prepare, a stall or a reclaim
    raises."""
    try:
        session.run_detached(command, log, minutes, stall_minutes=STALL_MINUTES,
                             max_errors=1 << 30)
        return None
    except RemoteFailure as error:
        if name != "run" or reclaimed(error) or "no progress" in str(error) \
                or "no result in" in str(error):
            raise
        say(f"the run failed on the box ({error}); bringing back what it measured")
        return str(error)


class SyncFailed(RuntimeError):
    """The staged files would not go to a warm box (twice): the job tries another box."""

    def __init__(self, box: str, reason: str):
        super().__init__(f"rsync to warm box {box} failed: {reason}")
        self.box = box


SYNC_RETRY_SECONDS = 10


def sync(session, stage_dir: Path) -> None:
    """The staged home onto a warm box: only files it lacks or that changed (rsync,
    symlinks followed, times kept, so a clip already there is not sent again)."""
    say(f"syncing {stage_dir} to the warm box")
    remote_shell = " ".join(shlex.quote(a) for a in session.ssh_base()[:-1])
    subprocess.run(["rsync", "-rLt", "--partial", "--exclude", "manifest.txt", "-e",
                    remote_shell, f"{stage_dir}/", f"{session.ssh_base()[-1]}:{REMOTE_HOME}/"],
                   check=True)


def warm_session(args, stage_dir: Path, back: Path, stages: list, expected: dict[str, str],
                 results: list[str], layout: Layout, avoid: set[str] = frozenset(),
                 copies: dict[str, Path] | None = None) -> dict:
    """The job on a warm box (aws_warm): a running one with room, else a new one set up
    once (the box's readiness and its idle timer); the job syncs, runs and comes back,
    and the box stays up for the next job."""
    import aws_warm

    def setup(session) -> None:
        sync(session, stage_dir)
        session.run_detached(" && ".join(box_setup()) + " && "
                             + aws_warm.idle_watch_command(box_cleanup_command(),
                                                           args.idle_minutes),
                             f"{REMOTE_HOME}/box-setup.log", PREPARE_MINUTES,
                             stall_minutes=STALL_MINUTES, max_errors=1 << 30)
    timings: dict[str, float] = {}
    started = time.monotonic()
    family = getattr(args, "instance_family", None)
    box, session, fresh = aws_warm.acquire(layout.job, aws_warm.warm_spec(
        args.region, args.warm_budget, family), setup,
        getattr(args, "warm_spend_cap", None) or aws_warm.SPEND_CAP_USD_PER_HOUR, avoid,
        family)
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
        failed = None
        try:
            for name, command, minutes in stages:
                started = time.monotonic()
                say(f"remote: {name}")
                failed = run_stage(session, name, command, layout.log(name), minutes)
                timings[name] = time.monotonic() - started
                if name == "prepare":
                    check_decodes(session, expected, layout)
        finally:
            started = time.monotonic()
            busy = session.ssh(f"grep -o 'box CPUs busy [0-9]*' {layout.log('run')}",
                               check=False, quiet=True).stdout.split()
            bring_back(session, back, results, layout, copies)
            timings["download"] = time.monotonic() - started
    finally:
        aws_warm.release(box, layout.job)
    hours = sum(timings.values()) / 3600
    return {"instance_type": session.spec.instance_type, "region": session.spec.region,
            "warm_box": box.instance_id, "fresh_box": fresh, "failed": failed,
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
        failed = None
        try:
            for name, command, minutes in stages:
                started = time.monotonic()
                say(f"remote: {name}")
                failed = run_stage(session, name, command, f"{REMOTE_HOME}/{name}.log", minutes)
                timings[name] = time.monotonic() - started
                if name == "prepare":
                    check_decodes(session, expected)
        finally:  # whatever the run measured comes back before the box goes
            started = time.monotonic()
            bring_back(session, back, results, copies=copies)
            timings["download"] = time.monotonic() - started
        hours = (datetime.datetime.now(datetime.timezone.utc)
                 - session.launched).total_seconds() / 3600 if session.launched else 0.0
        busy = session.ssh(f"grep -o 'box CPUs busy [0-9]*' {REMOTE_HOME}/run.log",
                           check=False, quiet=True).stdout.split()
        return {"instance_type": session.spec.instance_type, "region": session.spec.region,
                "failed": failed,
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
               layout: Layout = Layout(), copies: dict[str, Path] | None = None) -> None:
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
            session.download_new(f"{REMOTE_WORK}/{WORK_SUBDIR}", back / WORK_SUBDIR,
                                 layout.marker)
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
    split = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser(prog="aws_measure.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance-type", default=DEFAULT_INSTANCE,
                   help="a type, or auto (default): sized to the job (AUTO_SIZES)")
    p.add_argument("--batch", help="a JSON list of commands (argv lists) to run on one box")
    p.add_argument("--cold", action="store_true",
                   help="a one-off box for this job (default: a warm box, tools/dev/aws_warm.py)")
    p.add_argument("--warm", action="store_true", help=argparse.SUPPRESS)  # the default
    p.add_argument("--idle-minutes", type=int, default=20,
                   help="a warm box this job starts stops itself after this long idle")
    # $10 lets a 96-vCPU box start in eu-west-1 or eu-central-1, whose spot is ~2.5x
    # eu-north-1's ($1.43-1.55/h, 2026-11-21: at $6 no type fitted there and jobs waited
    # 25 min for eu-north-1's quota); the box still stops after 20 idle minutes.
    p.add_argument("--warm-budget", type=float, default=10.0,
                   help="a new warm box's worst case (bid cap x its lifetime) must fit this")
    p.add_argument("--warm-spend-cap", type=float,
                   help="$/h all warm boxes may cost together before jobs queue instead of "
                        "launching (default $6, NC_WARM_SPEND_CAP)")
    p.add_argument("--instance-family",
                   help="only boxes of this family (e.g. c7a): a tune study pinned to one CPU")
    p.add_argument("--no-fallback", action="store_true",
                   help="only --instance-type (default: fall back through the throughput ladder)")
    p.add_argument("--region", default=REGION)
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                   help="the session's worst case (bid cap x lifetime) must fit this (USD)")
    p.add_argument("--lifetime", type=int, default=DEFAULT_LIFETIME_MINUTES,
                   help="the box's hard lifetime (minutes)")
    p.add_argument("--run-minutes", type=float, default=30 + RUN_MARGIN_MINUTES,
                   help="the command's time limit on the box")
    p.add_argument("--dry-run", action="store_true", help="stage and print, rent nothing")
    args = p.parse_args(argv[:split])
    args.command = argv[split + 1:]
    if not args.command and not args.batch:
        p.error("give the nceval command after --, or --batch FILE")
    measure(args)


if __name__ == "__main__":
    main(sys.argv[1:])
