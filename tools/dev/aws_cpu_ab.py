#!/usr/bin/env python3
"""CPU A/B of two encoder builds or settings on rented single-core targets, side by side.

    python3 tools/dev/aws_cpu_ab.py --base COMMIT|BINARY --cand COMMIT|BINARY \\
        [--base-param k=v ...] [--cand-param k=v ...] --rungs hevc:fast,av1:faster \\
        [--clips crowd_mot17_09,bodycam_airport,...] [--frames 150] [--bitrate 600 | --qp Q] \\
        [--rounds 5] [--targets graniterapids,graviton4[,zen5]] [--boxes 1] [--json OUT]
        [--dry-run]
        [--key-shares [150,300]]
    python3 tools/dev/aws_cpu_ab.py --plan PAIRS.json ...     # several pairs in one run
    python3 tools/dev/aws_cpu_ab.py summarize --runs DIR [--json OUT]

The CPU targets are the benchmark's timing machines (BP-3.10, docs/benchmark-procedure.md
section 2; tools/dev/cpu_ab_shard.py TARGETS): c8i.2xlarge (Granite Rapids) and c8g.2xlarge
(Graviton 4), eu-north-1 spot, with c8a.2xlarge (Zen 5, on demand) as an optional third. Each
target and shard gets its own box, all at the same time, placed as the benchmark's boxes are:
encodes on vCPU 1, the harness on another physical core (tools/bench/pinning.py). Base and
candidate are measured on that same core in interleaved rounds (tools/dev/cpu_ab_box.py:
A B, B A, ...), so the box's level (the benchmark's 1.4-1.5 % spread between c8i boxes) and its
drift cancel in each pair; the benchmark's timing references (section 6), which correct jobs
that are compared across time on a box, would cancel out of these paired ratios. A target is
a CPU, not a judge: where two targets disagree on a change, that is a fact about the change on
those CPUs (as AVX2 against AVX-512 is), and the change is judged on the whole set.

Each run counts user-space hardware events where the instance offers them (Graviton:
cycles, instructions, cache refills, backend stalls; most x86 VMs: none) and always
the encode's CPU time; the summary gives, per pair, rung and target, the change in
cycles (else CPU time) as the mean of the per-round paired log ratios over the clips,
with a 95 % interval, and the instruction change beside it.

Builds. A BINARY is used as given (it must be of the target's architecture: an
aarch64 build for Graviton, x86_64 for zen5/intel). A COMMIT is built once per
architecture on the build box (nc-remote: `cargo build --release`, aarch64 cross
compiled with the Ubuntu cross linker; the Jetson's builds are native and use the same
default target CPU, so the same code generation up to the toolchain version) and kept
in BUILD_CACHE (by commit, architecture and the release profile). Builds of different
commits run side by side in BUILD_DIRS warm target directories per architecture on the
box; only requests for the same (commit, architecture) wait for each other. A pair that
differs only in `--param` values needs one build.

Boxes. tools/dev/aws_session.py: spot (on demand with `--on-demand`; a failed shard's
retries stay on spot unless `--retry-on-demand`), a budget cap and a hard lifetime per box, quota-aware
(a session waits in line for spot quota), terminated and verified on every exit, the
ledger written. Clips are cut to `--frames`, zstd-compressed (cached in CLIP_CACHE),
uploaded and deleted with the box.

Key shares. `--key-shares` measures every pair at each key share of the acceptance
screens (eval/nceval/key_shares.py: a key every 150 frames on 600, every 300 on 900,
from frame 0), in the rate mode given (CBR by default, as the screens' quality), on
KEY_SHARE_CLIPS (the long clips with 900 frames); each pair becomes `<pair>@k<period>`
and the summary prints, per rung and target, the `--dcpu 150=X,300=Y` that
nceval.quick, nceval.train and nceval.screen take.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))
from aws_session import say  # noqa: E402
from cpu_ab_shard import (DONE_FILE, BOX_BUDGET_USD, DEFAULT_TARGETS, MAX_RUN_VCPUS, TARGET_VCPUS, TARGETS,  # noqa: E402
                          box_command, prepare_command)
from nceval import decision_cpu, key_shares, retention  # noqa: E402
from nceval.clipframes import ClipFrames  # noqa: E402
from nceval.clips import Clip  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
WORK = DATA / "work/cpu-ab"
BUILD_CACHE = WORK / "builds"
CLIP_CACHE = WORK / "clips"
CLIPSET = DATA / "clips/quick.json"
NC_REMOTE = DATA / "bin/nc-remote"
#: The box side: the runner, its counter and the benchmark's CPU placement (one definition).
BOX_FILES = (REPO / "tools/dev/cpu_ab_box.py", REPO / "tools/dev/jperf.py",
             REPO / "tools/bench/pinning.py")
SHARD_SCRIPT = Path(__file__).resolve().parent / "cpu_ab_shard.py"
COMMIT_MARKER = "COMMIT"


DEFAULT_CLIPS = ("crowd_mot17_09", "bodycam_airport", "pan_jaad_0117", "lowlight_pp22_004")
DEFAULT_FRAMES = 150
DEFAULT_KBPS = 600
DEFAULT_ROUNDS = 5
#: `--key-shares`' clips: raw 720p clips of three classes with the 1/300 share's window
#: (900 frames) in clips/long.json.
KEY_SHARE_CLIPSET = DATA / "clips/long.json"
KEY_SHARE_CLIPS = ("long_static_chirla_s024c2", "long_crowd_mot20_02", "long_pan_bdd_00c12bd0")
#: The pair-name suffix of a key share's pair (`<pair>@k150`).
SHARE_SUFFIX = "@k"
#: ELF e_machine values (offset 18) of the two architectures we build.
ELF_MACHINES = {0xB7: "aarch64", 0x3E: "x86_64"}
RUST_TRIPLES = {"aarch64": "aarch64-unknown-linux-gnu", "x86_64": "x86_64-unknown-linux-gnu"}
#: Cross linker for aarch64 on the (x86) build box, installed by the build job if missing.
CROSS_PACKAGES = "gcc-aarch64-linux-gnu libc6-dev-arm64-cross"
CROSS_LINKER_ENV = "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc"
#: Every stage says it is alive and fails loudly past its limit (2026-11-21: an agent's
#: A/B sat 30 min silent before its aarch64 build, queued behind other agents' builds on
#: the per-architecture lock). A build is ~3 min warm; the lock waits for at most a few.
BUILD_TIMEOUT_S = 20 * 60
LOCK_WAIT_S = 45 * 60
#: Build-box target directories per architecture: builds of this many commits run at
#: once (each a slot of the 96-vCPU box), each directory kept warm for the next.
BUILD_DIRS = 4
PROGRESS_SECONDS = 60
#: A shard's wall limit: its box's lifetime, plus a wait for spot quota and the download.
SHARD_WALL_MINUTES = 75 + 20
#: Tries per shard, all on spot in the target's region (aws_session tries every zone) unless
#: --on-demand, or --retry-on-demand for the retries (coordinator, 2026-11-24: the automatic
#: on-demand retry cost c8g.2xlarge $0.05 a box against $0.01 spot; docs/aws-cost-guide.md
#: rule 6). One shard
#: that fails every try loses the whole run's charge, so a third try (2026-11-23: Zen 5
#: boxes failed twice in a row on spot capacity and a dropped upload, and a 9-box
#: re-judge of slow rungs was lost).
SHARD_ATTEMPTS = 3
#: Two-sided 95 % Student t quantiles by degrees of freedom (larger: the normal's).
T975 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}
NORMAL_975 = 1.960


# --- what is measured ------------------------------------------------------------------

@dataclass(frozen=True)
class Side:
    """One side of a pair: a build (commit or binary path) and its extra arguments.

    `inputs` (a --plan side only): a local directory of per-job files the side's encodes
    read, staged on the box; `job_args` are encoder arguments in which `{inputs}`,
    `{codec}`, `{preset}` and `{clip}` name that directory on the box and the job, e.g.
    `["--partition-replay", "{inputs}/{codec}-{preset}/{clip}.plog"]`."""
    build: str
    params: tuple[str, ...] = ()
    inputs: str | None = None
    job_args: tuple[str, ...] = ()

    def argv_tail(self, codec: str = "", preset: str = "", clip: str = "") -> list[str]:
        fields = {"inputs": f"inputs/{self.inputs_key()}", "codec": codec, "preset": preset,
                  "clip": clip}
        return [a for p in self.params for a in ("--param", p)] + [
            a.format(**fields) for a in self.job_args]

    def inputs_key(self) -> str | None:
        if self.inputs is None:
            return None
        return hashlib.sha1(str(Path(self.inputs).resolve()).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Pair:
    name: str
    base: Side
    cand: Side
    rungs: tuple[str, ...]  # codec:preset
    clips: tuple[str, ...]
    frames: int
    rate: tuple[str, ...]  # ("--bitrate", "600") or ("--qp", "32")
    #: A key every this many frames (a key share); None: the encoder's default.
    intra_period: int | None = None


def pairs_from_args(args) -> list[Pair]:
    rate = ("--qp", str(args.qp)) if args.qp is not None else ("--bitrate", str(args.bitrate))
    if args.plan:
        pairs = [pair_from_json(entry, args, rate) for entry in json.loads(args.plan.read_text())]
    elif not (args.base and args.cand and args.rungs):
        raise SystemExit("give --base, --cand and --rungs, or --plan")
    else:
        pairs = [Pair("ab", Side(args.base, tuple(args.base_param)),
                      Side(args.cand, tuple(args.cand_param)), tuple(args.rungs.split(",")),
                      tuple(args.clips.split(",")), args.frames, rate)]
    return at_key_shares(pairs, args.key_shares) if getattr(args, "key_shares", None) else pairs


def at_key_shares(pairs: list[Pair], periods: str) -> list[Pair]:
    """Each pair once per key share (nceval.key_shares): its window, a key every period."""
    return [dataclasses.replace(p, name=f"{p.name}{SHARE_SUFFIX}{s.period}", frames=s.window,
                                intra_period=s.period)
            for p in pairs for s in key_shares.parse_periods(periods)]


def pair_from_json(entry: dict, args, rate: tuple[str, ...]) -> Pair:
    """A --plan entry: {"name", "base": {"build", "params", optional "inputs" and
    "job_args" (Side)}, "cand": {...}, "rungs",
    optional "clips", "frames", "bitrate" / "qp"} (missing fields: the command line's)."""
    if "qp" in entry:
        rate = ("--qp", str(entry["qp"]))
    elif "bitrate" in entry:
        rate = ("--bitrate", str(entry["bitrate"]))
    side = lambda s: Side(s["build"], tuple(s.get("params", ())), s.get("inputs"),
                          tuple(s.get("job_args", ())))
    return Pair(entry["name"], side(entry["base"]), side(entry["cand"]),
                tuple(entry["rungs"]), tuple(entry.get("clips", args.clips.split(","))),
                int(entry.get("frames", args.frames)), rate)


def load_clips(names: set[str], clipset: Path | None = None) -> dict[str, dict]:
    clipset = clipset or CLIPSET  # read at call time: tests point CLIPSET elsewhere
    known = {c["name"]: c for c in json.loads(clipset.read_text())["clips"]}
    missing = sorted(names - set(known))
    if missing:
        raise SystemExit(f"not in {clipset}: {missing}")
    return {n: known[n] for n in names}


def build_key(build: str) -> str:
    """A short, path-safe name for a build on the box."""
    path = Path(build)
    if path.is_file():
        return f"file-{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]}"
    return f"commit-{resolve_commit(build)[:12]}"


def resolve_commit(build: str) -> str:
    done = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{build}^{{commit}}"],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"{build}: neither a binary file nor a commit")
    return done.stdout.strip()


def jobs_for(pairs: list[Pair], clips: dict[str, dict]) -> list[dict]:
    """The box plan's jobs: one per pair, rung and clip, with the box's paths."""
    jobs = []
    for pair in pairs:
        for rung in pair.rungs:
            codec, preset = rung.split(":")
            for name in pair.clips:
                clip = clips[name]
                common = ["encode", "--codec", codec, "--preset", preset,
                          "--input", f"clips/{clip_file(name, pair.frames)}",
                          "--width", str(clip["width"]), "--height", str(clip["height"]),
                          "--fps", f"{clip['fps']:g}", "--frames", str(pair.frames),
                          *pair.rate, "--output", "/dev/null"]
                if pair.intra_period:
                    common += ["--intra-period", str(pair.intra_period)]
                variants = {}
                for side in ("base", "cand"):
                    this = getattr(pair, side)
                    variants[side] = [f"bin/{build_key(this.build)}/nanocodec", *common,
                                      *this.argv_tail(codec, preset, name)]
                inputs = {f"inputs/{s.inputs_key()}": str(Path(s.inputs).resolve())
                          for s in (pair.base, pair.cand) if s.inputs}
                jobs.append({"id": f"{pair.name}|{rung}|{name}", "variants": variants,
                             **({"inputs": inputs} if inputs else {})})
    return jobs


def decision_jobs(plan: dict) -> list[dict]:
    """The box jobs of a --decision-plan (nceval.decision_cpu): each decision's own
    commands, the binary, clip and stream filled in, each side's arguments appended; the
    pair is `<decision>@<group>` (decision_cpu.GROUP_MARK), one per protocol group."""
    jobs = []
    for decision in plan["decisions"]:
        rung = f"{decision['codec']}:{decision['rung']}"
        for record in decision["jobs"]:
            job = decision_cpu.CpuJob.from_record(record)
            pair = f"{decision['name']}{decision_cpu.GROUP_MARK}{job.group}"
            clip = f"clips/{decision_clip_file(job)}"
            variants = {}
            for side in ("base", "cand"):
                this = decision[side]
                fill = {decision_cpu.BIN: f"bin/{build_key(this['build'])}/nanocodec",
                        decision_cpu.INPUT: clip, decision_cpu.OUTPUT: "/dev/null"}
                variants[side] = [*(fill.get(a, a) for a in job.argv), *this["extra"]]
            jobs.append({"id": f"{pair}|{rung}|{job.clip}-{job.kbps}k", "variants": variants})
    return jobs


def decision_clip_file(job: "decision_cpu.CpuJob") -> str:
    return f"{job.clip}-s{job.start_frame}-{job.frames}f.yuv"


def decision_clips(plan: dict) -> dict[str, Path]:
    """{clip file: its compressed frames} for every job of a --decision-plan."""
    files = {}
    for decision in plan["decisions"]:
        for record in decision["jobs"]:
            job = decision_cpu.CpuJob.from_record(record)
            source = decision["clips"][job.clip]
            files[decision_clip_file(job)] = cached_frames(
                Clip(**{**source, "yuv": Path(source["yuv"])}, start_frame=job.start_frame),
                job.frames, decision_clip_file(job))
    return files


def decisions_record(plan: dict) -> dict:
    """The summary's record of what each decision measured (decision_cpu.charge reads it)."""
    return {d["name"]: {"digest": d["digest"], "rotation": d["rotation"], "codec": d["codec"],
                        "rung": d["rung"], "base": d["base"], "cand": d["cand"],
                        "groups": sorted({decision_cpu.CpuJob.from_record(j).group
                                          for j in d["jobs"]})}
            for d in plan["decisions"]}


def input_of(job: dict) -> str:
    argv = next(iter(job["variants"].values()))
    return argv[argv.index("--input") + 1]


def shards(jobs: list[dict], count: int) -> list[list[dict]]:
    """`count` near-equal slices of the jobs ordered by clip, so that each box gets only
    a few of the clips (each box's upload is its clips: ~100 MB per 150-frame 720p clip)."""
    ordered = sorted(jobs, key=input_of)
    size, extra = divmod(len(ordered), count)
    parts, start = [], 0
    for index in range(count):
        end = start + size + (index < extra)
        parts.append(ordered[start:end])
        start = end
    return [part for part in parts if part]


# --- builds and clips ------------------------------------------------------------------

def elf_machine(path: Path) -> str | None:
    head = path.read_bytes()[:20]
    if head[:4] != b"\x7fELF":
        return None
    return ELF_MACHINES.get(int.from_bytes(head[18:20], "little"))


#: What WORK keeps (owner, 2026-11-21: the PC disk is shared; hold as little as possible):
#: the KEEP_BUILDS most recently used builds per architecture (a build is ~20 MB, a
#: rebuild 3 min on the build box), and clips and run directories used in KEEP_DAYS.
KEEP_BUILDS = 12
KEEP_DAYS = 3
#: The cache's size cap (tools/dev/retention.toml `cap_gb` of work/cpu-ab): past it, the
#: least recently used cut clips go first (one is re-cut in seconds; 114 of 119 had been
#: read within a day on 2026-11-24, 16 GiB in all).
CACHE_CAP_ENTRY = "work/cpu-ab"
#: A build's temporary worktree and fetch directory (build_on_box: `tree-` / `fetch-`
#: MACHINE-COMMIT-PID) left by a process that is gone.
LEFTOVER_PREFIXES = ("tree-", "fetch-")


def prune(now: float | None = None) -> int:
    """Evict what WORK no longer needs (see KEEP_BUILDS); returns bytes freed. A build
    another run is making or using (its build.lock held) is kept."""
    now = time.time() if now is None else now
    old = lambda path: now - path.stat().st_mtime > KEEP_DAYS * 86400
    doomed: list[Path] = []
    for machine in RUST_TRIPLES:
        homes = [h for h in (BUILD_CACHE / machine).iterdir() if (h / COMMIT_MARKER).is_file()] \
            if (BUILD_CACHE / machine).is_dir() else []
        homes.sort(key=lambda h: (h / COMMIT_MARKER).stat().st_mtime, reverse=True)
        doomed += [h for h in homes[KEEP_BUILDS:] if not locked(h / "build.lock")]
    doomed += [c for c in CLIP_CACHE.glob("*.zst") if old(c)] if CLIP_CACHE.is_dir() else []
    doomed += [r for r in WORK.glob("run-*") if r.is_dir() and old(r)]
    freed = 0
    for path in doomed:
        if retention.guard(path, "aws_cpu_ab") is not None:
            continue
        files = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
        freed += sum(f.stat().st_size for f in files)
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    freed += remove_leftovers()
    return freed + cap_clips()


def remove_leftovers() -> int:
    """Builds' temporary worktrees and fetch directories whose process is gone (a build
    stopped before its cleanup); a worktree is unregistered from the repository too."""
    freed = 0
    for path in sorted(WORK.glob("*-*-*-*")) if WORK.is_dir() else ():
        pid = path.name.rsplit("-", 1)[-1]
        if not (path.name.startswith(LEFTOVER_PREFIXES) and pid.isdigit() and path.is_dir()) \
                or Path(f"/proc/{pid}").exists() or retention.guard(path, "aws_cpu_ab"):
            continue
        freed += retention.entry_bytes(path)
        if path.name.startswith("tree-"):
            subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(path)],
                           check=False, capture_output=True)
        if path.exists():
            shutil.rmtree(path)
    return freed


def cap_clips() -> int:
    """The cut clips, least recently used first, until the cache is within its cap."""
    cap = retention.cap_bytes(DATA / CACHE_CAP_ENTRY, DATA)
    if cap is None or not CLIP_CACHE.is_dir():
        return 0
    clips = sorted(CLIP_CACHE.glob("*.zst"))
    others = retention.entry_bytes(WORK) - sum(retention.entry_bytes(c) for c in clips)
    return retention.prune_to_cap(clips, cap, "aws_cpu_ab", fixed_bytes=others)[0]


def locked(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def all_builds(builds: list[str], machines: list[str]) -> dict[str, dict[str, Path]]:
    """Every (build, machine) binary, the ones still to make built side by side."""
    pairs = [(b, m) for m in machines for b in builds]
    started = time.monotonic()
    with ThreadPoolExecutor(len(pairs)) as pool:
        found = dict(zip(pairs, pool.map(lambda pair: build_for(*pair), pairs)))
    say(f"builds ready in {time.monotonic() - started:.0f} s ({len(pairs)}: "
        f"{', '.join(f'{build_key(b)} {m}' for b, m in pairs)})")
    return {m: {build_key(b): found[(b, m)] for b in builds} for m in machines}


def build_for(build: str, machine: str) -> Path:
    """The binary for `machine`: a given file (checked), else the commit's release build."""
    path = Path(build)
    if path.is_file():
        found = elf_machine(path)
        if found != machine:
            raise SystemExit(f"{build} is built for {found}, the target needs {machine}")
        return path.resolve()
    return release_build(resolve_commit(build), machine)


def release_build(commit: str, machine: str) -> Path:
    """The commit's release build for `machine`, from BUILD_CACHE (by commit and
    architecture, release profile) or built on the build box. Builds of different
    commits run side by side, each in one of BUILD_DIRS target directories per
    architecture on the box (kept warm, so incremental); only two requests for the same
    (commit, architecture) wait for each other, and the second then finds it built
    (owner, 2026-11-21: no serialisation that is not necessary; one lock per
    architecture had four agents' builds queue for 25 min)."""
    home = BUILD_CACHE / machine / commit[:12]
    binary = home / "nanocodec"
    if built(home, commit):
        (home / COMMIT_MARKER).touch()  # in use: prune keeps it
        return binary
    home.mkdir(parents=True, exist_ok=True)
    with (home / "build.lock").open("a+") as same:
        wait_for_lock(same, f"the {machine} build of {commit[:12]} another request is making")
        mark(same, f"building {commit[:12]}")
        if built(home, commit):
            return binary
        with build_dir(machine) as name:
            build_on_box(commit, machine, name, binary)
        (home / COMMIT_MARKER).write_text(commit + "\n")
    return binary


def built(home: Path, commit: str) -> bool:
    marker = home / COMMIT_MARKER
    return marker.is_file() and marker.read_text().strip() == commit


@contextmanager
def build_dir(machine: str):
    """One of the BUILD_DIRS build-box job names (each its own target directory there)
    for `machine`, held while its build runs; waits, saying so, when all are busy."""
    locks = [(BUILD_CACHE / f"{machine}-dir{k}.lock").open("a+") for k in range(BUILD_DIRS)]
    try:
        started = last = time.monotonic()
        while True:
            for k, lock in enumerate(locks):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                mark(lock, "building")
                # the first keeps the name (and the warm target) of the one directory before
                yield f"cpuab-{machine}" + (f"-{k}" if k else "")
                return
            waited = time.monotonic() - started
            if waited > LOCK_WAIT_S:
                raise SystemExit(f"!!! all {BUILD_DIRS} {machine} build directories busy for "
                                 f"{waited / 60:.0f} min: giving up")
            if time.monotonic() - last >= PROGRESS_SECONDS:
                say(f"all {BUILD_DIRS} {machine} build directories busy "
                    f"({'; '.join(holder(lock) for lock in locks)}), {waited:.0f} s")
                last = time.monotonic()
            time.sleep(2)
    finally:
        for lock in locks:
            lock.close()


def mark(lock, what: str) -> None:
    """Name the holder in a lock file (holder() reads it)."""
    lock.seek(0)
    lock.truncate(0)
    lock.write(f"pid {os.getpid()} {what} since {time.strftime('%H:%M:%S')}\n")
    lock.flush()


def build_on_box(commit: str, machine: str, name: str, binary: Path) -> None:
    """Build `commit` for `machine` as build-box job `name` (its target directory);
    the binary to `binary`."""
    tree = WORK / f"tree-{machine}-{commit[:12]}-{os.getpid()}"
    fetched = WORK / f"fetch-{machine}-{commit[:12]}-{os.getpid()}"
    subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(tree), commit],
                   check=True, capture_output=True)
    try:
        path = f"target/{RUST_TRIPLES[machine]}/release/nanocodec"
        what = f"building {commit[:12]} for {machine} on the build box ({name})"
        say(what)
        run_watched([str(NC_REMOTE), "run", "--name", name, "--fetch", path,
                     "--fetch-to", str(fetched), "--", build_command(machine)],
                    what, BUILD_TIMEOUT_S, cwd=tree)
        shutil.move(str(fetched / path), binary)
    finally:
        # --force: the build leaves untracked files (the target link) that a plain remove
        # refuses, and 30 trees (7.5 GiB) had piled up by 2026-11-24
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(tree)],
                       check=False)
        shutil.rmtree(fetched, ignore_errors=True)


def wait_for_lock(lock, what: str, limit_s: float = LOCK_WAIT_S) -> None:
    """flock `lock`, saying every PROGRESS_SECONDS who holds it; fail after `limit_s`."""
    started = last = time.monotonic()
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            pass
        waited = time.monotonic() - started
        if waited > limit_s:
            raise SystemExit(f"!!! waited {waited / 60:.0f} min for {what} "
                             f"({holder(lock)}): giving up")
        if time.monotonic() - last >= PROGRESS_SECONDS:
            say(f"waiting for {what} ({holder(lock)}), {waited:.0f} s")
            last = time.monotonic()
        time.sleep(2)


def holder(lock) -> str:
    lock.seek(0)
    return lock.read().strip() or "holder unknown"


def run_watched(argv: list[str], what: str, limit_s: float, **kwargs) -> None:
    """Run `argv`, saying every PROGRESS_SECONDS that `what` is still going; kill it and
    fail loudly past `limit_s`, or when it fails."""
    started = last = time.monotonic()
    process = subprocess.Popen(argv, **kwargs)
    while process.poll() is None:
        time.sleep(2)
        elapsed = time.monotonic() - started
        if elapsed > limit_s:
            process.kill()
            process.wait()
            raise SystemExit(f"!!! {what}: no end after {elapsed / 60:.0f} min: killed")
        if time.monotonic() - last >= PROGRESS_SECONDS:
            say(f"still {what}, {elapsed:.0f} s")
            last = time.monotonic()
    if process.returncode != 0:
        raise SystemExit(f"!!! {what}: failed (exit {process.returncode})")


def build_command(machine: str) -> str:
    """The build box's release build for `machine` (aarch64: with the cross linker,
    installed on the box by the first such build)."""
    cargo = f"cargo build -q --release --target {RUST_TRIPLES[machine]} -p nanocodec-cli"
    if machine != "aarch64":
        return f"set -e; {cargo}"
    return ("set -e; dpkg -s gcc-aarch64-linux-gnu >/dev/null 2>&1 || sudo DEBIAN_FRONTEND="
            f"noninteractive apt-get -qq install -y --no-install-recommends {CROSS_PACKAGES} "
            f">/dev/null; {CROSS_LINKER_ENV} {cargo}")


def clip_file(name: str, frames: int) -> str:
    return f"{name}-{frames}f.yuv"


def cached_clip(clip: dict, frames: int) -> Path:
    """The clip's first `frames` frames, zstd-compressed, made once."""
    out = CLIP_CACHE / f"{clip_file(clip['name'], frames)}.zst"
    if out.is_file():
        out.touch()  # in use: prune keeps it
        return out
    source = DATA / clip["yuv"]
    frame_bytes = clip["width"] * clip["height"] * 3 // 2
    if source.stat().st_size < frames * frame_bytes:
        raise SystemExit(f"{source} has fewer than {frames} frames")
    with source.open("rb") as raw:
        return compressed(raw.read(frames * frame_bytes), out)


def cached_frames(clip: Clip, frames: int, file: str) -> Path:
    """`frames` frames of a decision's clip from its start frame, as its quality encodes
    read them (nceval.clipframes: raw or decoded), zstd-compressed, made once."""
    out = CLIP_CACHE / f"{file}.zst"
    if out.is_file():
        out.touch()
        return out
    with ClipFrames(clip, frames, CLIP_CACHE / "decode") as served:
        with Path(served.reader()).open("rb") as frames_in:
            data = frames_in.read(frames * clip.frame_bytes)
    if len(data) != frames * clip.frame_bytes:
        raise SystemExit(f"{clip.name}: fewer than {frames} frames from {clip.start_frame}")
    return compressed(data, out)


def compressed(data: bytes, out: Path) -> Path:
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(f".part{os.getpid()}")
    subprocess.run(["zstd", "-q", "-3", "-T4", "-o", str(partial)], input=data, check=True)
    partial.rename(out)
    return out


def stage(stage_dir: Path, jobs: list[dict], binaries: dict[str, Path],
          clip_files: dict[str, Path]) -> None:
    """Symlinks for one box: the binaries its jobs name, their clips, the plan, the runner."""
    (stage_dir / "out").mkdir(parents=True)
    for key in sorted({j["variants"][s][0].split("/")[1] for j in jobs for s in j["variants"]}):
        (stage_dir / "bin" / key).mkdir(parents=True)
        (stage_dir / "bin" / key / "nanocodec").symlink_to(binaries[key])
    for job in jobs:
        for name, source in job.get("inputs", {}).items():
            link = stage_dir / name
            if not link.exists():
                link.parent.mkdir(exist_ok=True)
                link.symlink_to(source)
    (stage_dir / "clips").mkdir()
    for job in jobs:
        for argv in job["variants"].values():
            file = argv[argv.index("--input") + 1].removeprefix("clips/")
            link = stage_dir / "clips" / f"{file}.zst"
            if not link.exists():
                link.symlink_to(clip_files[file])
    (stage_dir / "plan.json").write_text(json.dumps({"jobs": jobs}, indent=1) + "\n")
    for path in BOX_FILES:
        (stage_dir / path.name).symlink_to(path)


def launch_shards(stage_dirs: dict[str, Path], run_dir: Path, rounds: int, on_demand: bool,
                  purpose: str, box_budget: float, retry_on_demand: bool = False,
                  start_shard=None) -> list[str]:
    """Every shard in its own process (aws_session's signal handling needs a main thread),
    all at once, each a light cpu_ab_shard.py (about 24 MB, not this module's ~110 MB with
    nceval: 13 runs of 9 shards once filled the PC's memory); a failed shard is retried on spot
    (on demand only with `retry_on_demand`). Returns the shards that failed.
    `start_shard(label, attempt, demand)` replaces the process start in tests."""
    def start(label: str, attempt: int, demand: bool) -> subprocess.Popen:
        target = label.split("-")[0]
        log = (run_dir / f"{label}.attempt{attempt}.log").open("w")
        say(f"{label}: box {TARGETS[target].instance_type} (log {log.name})")
        return subprocess.Popen(
            [sys.executable, str(SHARD_SCRIPT), "--stage", str(stage_dirs[label]),
             "--target", target, "--out", str(run_dir / label), "--rounds", str(rounds),
             "--purpose", purpose, "--budget", str(box_budget)]
            + (["--on-demand"] if demand else []),
            stdout=log, stderr=subprocess.STDOUT)

    start = start_shard or start
    retry_market = "on demand (--retry-on-demand)" if on_demand or retry_on_demand else \
        "spot in its region, every zone"
    running = {label: (start(label, 1, on_demand), 1) for label in stage_dirs}
    started = {label: time.monotonic() for label in stage_dirs}
    failed, last = [], time.monotonic()
    while running:
        time.sleep(10)
        if time.monotonic() - last >= 5 * PROGRESS_SECONDS:
            for label in running:
                say(f"{label}: {time.monotonic() - started[label]:.0f} s, "
                    f"{last_line(run_dir / f'{label}.attempt{running[label][1]}.log')}")
            last = time.monotonic()
        for label, (process, attempt) in list(running.items()):
            if process.poll() is None:
                if time.monotonic() - started[label] > 60 * SHARD_WALL_MINUTES:
                    say(f"!!! {label}: no end after {SHARD_WALL_MINUTES} min: killed "
                        f"(its box ends with its session)")
                    process.terminate()
                    try:
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                continue
            del running[label]
            if process.returncode == 0:
                say(f"{label}: done")
            elif attempt < SHARD_ATTEMPTS:
                done = keep_finished(run_dir / label, stage_dirs[label], attempt, rounds)
                say(f"{label}: failed (exit {process.returncode}), retrying on {retry_market}; "
                    f"{done} finished job(s) kept, the rest measured again")
                running[label] = (start(label, attempt + 1, on_demand or retry_on_demand),
                                  attempt + 1)
                started[label] = time.monotonic()
            else:
                say(f"{label}: failed {attempt} times on {retry_market}; see its logs in "
                    f"{run_dir}" + ("" if on_demand or retry_on_demand else
                                    "; --retry-on-demand allows on-demand retries (about 5x "
                                    "the spot price), only with a stated reason"))
                failed.append(label)
    return failed


def keep_finished(out: Path, stage_dir: Path, attempt: int, rounds: int) -> int:
    """Before a shard's retry: the lost box's finished jobs (every variant at every round,
    pulled while it ran) are kept as runs-attempt<N>.jsonl and their ids staged in
    DONE_FILE for the next box to skip; the rest of `out` goes. Returns the jobs kept."""
    runs = out / "runs.jsonl"
    records = [json.loads(line) for line in runs.read_text().splitlines()] if runs.is_file() else []
    finished = finished_jobs(records, rounds)
    kept = [r for r in records if r["job"] in finished]
    if kept:
        (out.parent / f"{out.name}.attempt{attempt}-runs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in kept))
    shutil.rmtree(out, ignore_errors=True)
    done = stage_dir / DONE_FILE
    earlier = set(json.loads(done.read_text())) if done.is_file() else set()
    done.write_text(json.dumps(sorted(earlier | finished)) + "\n")
    return len(finished)


def finished_jobs(records: list[dict], rounds: int) -> set[str]:
    """The jobs whose every variant ran every round."""
    seen: dict[str, set[tuple[str, int]]] = {}
    for r in records:
        seen.setdefault(r["job"], set()).add((r["variant"], r["round"]))
    return {job for job, runs in seen.items()
            if len({v for v, _ in runs}) * rounds == len(runs)
            and {i for _, i in runs} == set(range(rounds))}


def shard_runs(run_dir: Path) -> dict[str, list[Path]]:
    """Each shard's record files: the last box's runs.jsonl and the finished jobs kept from
    boxes lost before it (keep_finished)."""
    files: dict[str, list[Path]] = {}
    for path in sorted(run_dir.glob("*/runs.jsonl")):
        files.setdefault(path.parent.name, []).append(path)
    for path in sorted(run_dir.glob("*.attempt*-runs.jsonl")):
        files.setdefault(path.name.split(".attempt")[0], []).append(path)
    return files


def last_line(log: Path) -> str:
    try:
        lines = log.read_text(errors="replace").strip().splitlines()
    except OSError:
        return "no log yet"
    return lines[-1][:160] if lines else "no output yet"


# --- summary ---------------------------------------------------------------------------

def t975(df: int) -> float:
    if df <= 0:
        return math.inf
    fitting = [d for d in T975 if d <= df]
    return T975[max(fitting)] if df < max(T975) else NORMAL_975


def paired_logs(runs: list[dict], metric: str) -> list[float]:
    """ln(cand / base) per round in which both sides counted the metric (every round for
    times and the base events; the rounds of its turn for an extra event)."""
    by_round: dict[int, dict[str, float]] = {}
    for run in runs:
        value = run[metric] if metric in ("user_s", "wall_s") else run["counts"].get(metric)
        if value:
            by_round.setdefault(run["round"], {})[run["variant"]] = value
    return [math.log(sides["cand"] / sides["base"]) for sides in by_round.values()
            if "base" in sides and "cand" in sides]


def mean_interval(groups: list[list[float]]) -> tuple[float, float, float] | None:
    """Mean over the groups (clips) of each group's mean log ratio, with a 95 % interval
    from the rounds' spread within the groups; as percentages."""
    groups = [g for g in groups if g]
    if not groups:
        return None
    means = [sum(g) / len(g) for g in groups]
    variance = sum((sum((x - m) ** 2 for x in g) / max(len(g) - 1, 1)) / len(g)
                   for g, m in zip(groups, means)) / len(groups) ** 2
    df = sum(len(g) - 1 for g in groups)
    centre = sum(means) / len(means)
    half = t975(df) * math.sqrt(variance)

    def pct(x: float) -> float:
        return 100 * math.expm1(x)
    return pct(centre), pct(centre - half), pct(centre + half)


def headline(runs: list[dict]) -> str:
    """Cycles where every run counted them, else the encode's CPU time."""
    return "cycles" if all(r["counts"].get("cycles") is not None for r in runs) else "user_s"


def summarize(run_dir: Path) -> dict:
    """Per pair, rung and target: the headline metric's and the instructions' change with
    intervals, each clip's change, and the rounds' noise (sd of the paired log ratios)."""
    records: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for shard, files in sorted(shard_runs(run_dir).items()):
        target = shard.split("-")[0]
        for line in (line for f in files for line in f.read_text().splitlines()):
            run = json.loads(line)
            pair, rung, clip = run["job"].split("|")
            records.setdefault((pair, rung, target), {}).setdefault(clip, []).append(run)
    rows = []
    for (pair, rung, target), clips in sorted(records.items()):
        every = [r for runs in clips.values() for r in runs]
        metric = headline(every)
        logs = {clip: paired_logs(runs, metric) for clip, runs in clips.items()}
        instr = [paired_logs(runs, "instructions") for runs in clips.values()]
        extras = sorted({n for r in every for n, v in r["counts"].items()
                         if v is not None} - {"cycles", "instructions"})
        spread = [x - sum(g) / len(g) for g in logs.values() if len(g) > 1 for x in g]
        spread_df = len(spread) - sum(1 for g in logs.values() if len(g) > 1)
        rows.append({
            "pair": pair, "rung": rung, "target": target, "metric": metric,
            "change_pct": mean_interval(list(logs.values())),
            "instructions_pct": mean_interval(instr),
            # the extra PMU events, each from the rounds that counted it
            "events_pct": {n: mean_interval([paired_logs(runs, n) for runs in clips.values()])
                           for n in extras},
            "per_clip_pct": {clip: 100 * math.expm1(sum(g) / len(g)) for clip, g in logs.items()
                             if g},
            # pooled within-clip sd of one round's paired log ratio
            "round_sd_pct": 100 * math.sqrt(sum(x * x for x in spread) / spread_df)
            if spread_df > 0 else None})
    return {"rows": rows}


def print_summary(summary: dict) -> None:
    def cell(value) -> str:
        return f"{value[0]:+6.2f}% [{value[1]:+.2f}, {value[2]:+.2f}]" if value else "-"
    print(f"{'pair':16s} {'rung':16s} {'target':10s} {'metric':7s} {'change':28s} "
          f"{'instructions':28s} round sd")
    for row in summary["rows"]:
        sd = f"{row['round_sd_pct']:.2f}%" if row["round_sd_pct"] is not None else "-"
        print(f"{row['pair']:16s} {row['rung']:16s} {row['target']:10s} {row['metric']:7s} "
              f"{cell(row['change_pct']):28s} {cell(row['instructions_pct']):28s} {sd}")
    for (pair, rung, target), text in share_dcpu(summary).items():
        print(f"{pair} {rung} {target}: --dcpu {text}")


def share_dcpu(summary: dict) -> dict[tuple[str, str, str], str]:
    """Per pair, rung and target measured at key shares (`<pair>@k<period>` rows): the
    change at each share as the screens' `--dcpu 150=X,300=Y`."""
    found: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in summary["rows"]:
        pair, suffix, period = row["pair"].rpartition(SHARE_SUFFIX)
        if suffix and period.isdigit() and row["change_pct"]:
            found.setdefault((pair, row["rung"], row["target"]), {})[int(period)] = \
                row["change_pct"][0]
    return {key: ",".join(f"{p}={v:.2f}" for p, v in sorted(shares.items()))
            for key, shares in found.items()}


# --- entry points ----------------------------------------------------------------------

def main_ab(args) -> None:
    targets = args.targets.split(",")
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown targets {unknown}; known: {sorted(TARGETS)}")
    freed = prune()
    if freed:
        say(f"pruned old builds, clips, runs and leftovers from {WORK}: {freed / 1e9:.2f} GB")
    if args.key_shares and args.clips is None:
        args.clips = ",".join(KEY_SHARE_CLIPS)
        args.clipset = args.clipset or KEY_SHARE_CLIPSET
    if args.decision_plan:
        plan = json.loads(args.decision_plan.read_text())
        jobs, clip_files = decision_jobs(plan), decision_clips(plan)
        builds = sorted({d[s]["build"] for d in plan["decisions"] for s in ("base", "cand")})
        described = {"decision_plan": str(args.decision_plan), **plan}
    else:
        args.clips = args.clips or ",".join(DEFAULT_CLIPS)
        pairs = pairs_from_args(args)
        clips = load_clips({c for p in pairs for c in p.clips}, args.clipset)
        jobs = jobs_for(pairs, clips)
        clip_files = {clip_file(n, p.frames): cached_clip(clips[n], p.frames)
                      for p in pairs for n in p.clips}
        builds = sorted({side.build for p in pairs for side in (p.base, p.cand)})
        described = {"pairs": [dataclasses.asdict(p) for p in pairs]}
    spot_vcpus = TARGET_VCPUS * args.boxes * sum(not TARGETS[t].on_demand for t in targets)
    if spot_vcpus > MAX_RUN_VCPUS and not args.on_demand:
        raise SystemExit(f"{spot_vcpus} spot vCPUs asked for, over the run cap {MAX_RUN_VCPUS} "
                         f"(the region's quota is shared with the benchmark): fewer --boxes")
    machines = sorted({TARGETS[t].machine for t in targets})
    binaries = all_builds(builds, machines)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = args.out or WORK / f"run-{stamp}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pairs.json").write_text(json.dumps(described, indent=1) + "\n")
    stage_dirs = {}
    for target in targets:
        for index, part in enumerate(shards(jobs, args.boxes), 1):
            label = f"{target}-{index}"
            stage_dirs[label] = run_dir / "stage" / label
            stage(stage_dirs[label], part, binaries[TARGETS[target].machine], clip_files)
    say(f"{len(jobs)} jobs x {args.rounds} rounds on {len(stage_dirs)} boxes "
        f"({', '.join(f'{t} {TARGETS[t].instance_type}' for t in targets)}); results in {run_dir}")
    if args.dry_run:
        for label, path in stage_dirs.items():
            subprocess.run(["du", "-shL", str(path)], check=True)
        print(prepare_command(1))
        print(box_command(1, args.rounds, "out/runs.jsonl"))
        return
    try:
        failed = launch_shards(stage_dirs, run_dir, args.rounds, args.on_demand, args.purpose,
                               args.box_budget, retry_on_demand=args.retry_on_demand)
    finally:
        shutil.rmtree(run_dir / "stage", ignore_errors=True)
    summary = summarize(run_dir)
    summary["failed_shards"] = failed
    if args.decision_plan:
        summary["decisions"] = decisions_record(plan)
    (args.json or run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_summary(summary)
    if failed:
        raise SystemExit(f"shards failed: {failed} (their rows are missing)")


def main(argv: list[str]) -> None:
    if argv[:1] == ["summarize"]:
        p = argparse.ArgumentParser()
        p.add_argument("--runs", type=Path, required=True)
        p.add_argument("--json", type=Path)
        a = p.parse_args(argv[1:])
        summary = summarize(a.runs)
        if a.json:
            a.json.write_text(json.dumps(summary, indent=1) + "\n")
        print_summary(summary)
        return
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", help="commit or binary")
    p.add_argument("--cand", help="commit or binary")
    p.add_argument("--base-param", action="append", default=[], metavar="K=V")
    p.add_argument("--cand-param", action="append", default=[], metavar="K=V")
    p.add_argument("--rungs", help="comma-separated codec:preset")
    p.add_argument("--plan", type=Path, help="JSON list of pairs (see pair_from_json)")
    p.add_argument("--decision-plan", type=Path,
                   help="a decision's own CPU plan (nceval.decision_cpu; nceval.screen / quick "
                        "--cpu-plan, nceval.train run): its encodes, both sides; the summary "
                        "is what that decision's --cpu-summary takes")
    p.add_argument("--clips", help=f"names in --clipset (default {','.join(DEFAULT_CLIPS)}; "
                   f"with --key-shares {','.join(KEY_SHARE_CLIPS)} in {KEY_SHARE_CLIPSET})")
    p.add_argument("--clipset", type=Path,
                   help=f"the clip set the names are in (default {CLIPSET}; clips/main.json "
                        "holds the chart's timing clips)")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--bitrate", type=int, default=DEFAULT_KBPS, help="CBR kbit/s")
    p.add_argument("--qp", type=int, help="fixed QP instead of CBR")
    p.add_argument("--key-shares", nargs="?", const=",".join(map(str, key_shares.KEY_SHARES)),
                   help="measure at each key share (default 150,300; nceval.key_shares): "
                        "the CPU the screens charge per share")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                   help=f"comma-separated, of {', '.join(sorted(TARGETS))}")
    p.add_argument("--boxes", type=int, default=1, help="boxes per target (jobs split)")
    p.add_argument("--on-demand", action="store_true", help="on-demand boxes, not spot")
    p.add_argument("--retry-on-demand", action="store_true",
                   help="retry a failed shard on demand (default: spot again; about 5x the "
                        "price, only with a stated reason: docs/aws-cost-guide.md rule 6)")
    p.add_argument("--box-budget", type=float, default=BOX_BUDGET_USD,
                   help=f"per-box cost cap, USD (default {BOX_BUDGET_USD})")
    p.add_argument("--purpose", default="cpu-ab")
    p.add_argument("--out", type=Path, help="run directory (default under WORK)")
    p.add_argument("--json", type=Path, help="summary file (default RUN/summary.json)")
    p.add_argument("--dry-run", action="store_true", help="stage and print; launch nothing")
    main_ab(p.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
