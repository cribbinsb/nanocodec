#!/usr/bin/env python3
"""CPU A/B of two encoder builds or settings on rented single-core targets, side by side.

    python3 tools/dev/aws_cpu_ab.py --base COMMIT|BINARY --cand COMMIT|BINARY \\
        [--base-param k=v ...] [--cand-param k=v ...] --rungs hevc:fast,av1:faster \\
        [--clips crowd_mot17_09,bodycam_airport,...] [--frames 150] [--bitrate 600 | --qp Q] \\
        [--rounds 5] [--targets graviton3,graviton4,zen5] [--boxes 1] [--json OUT] [--dry-run]
    python3 tools/dev/aws_cpu_ab.py --plan PAIRS.json ...     # several pairs in one run
    python3 tools/dev/aws_cpu_ab.py summarize --runs DIR [--json OUT]

The everyday CPU targets (docs/PROCEDURES.md "CPU A/B on rented cores"): one .large
instance per target and shard, base and candidate measured on the same core of the
same box in interleaved rounds (tools/dev/cpu_ab_box.py: A B, B A, ...), every target
and shard on its own box at the same time. A target is a CPU, not a judge: where two
targets disagree on a change, that is a fact about the change on those CPUs (as AVX2
against AVX-512 is), and the change is judged on the whole set.

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

Boxes. tools/dev/aws_session.py: spot (on demand with `--on-demand`, and for a shard's
one retry after a reclaim), a budget cap and a hard lifetime per box, quota-aware
(a session waits in line for spot quota), terminated and verified on every exit, the
ledger written. Clips are cut to `--frames`, zstd-compressed (cached in CLIP_CACHE),
uploaded and deleted with the box.
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
from aws_session import Session, SessionSpec, say  # noqa: E402
from nceval import retention  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
WORK = DATA / "work/cpu-ab"
BUILD_CACHE = WORK / "builds"
CLIP_CACHE = WORK / "clips"
CLIPSET = DATA / "clips/quick.json"
NC_REMOTE = DATA / "bin/nc-remote"
REMOTE_DIR = "cpuab"
BOX_FILES = ("cpu_ab_box.py", "jperf.py")
COMMIT_MARKER = "COMMIT"


@dataclass(frozen=True)
class Target:
    """One CPU target: the instance that has it, alone on a physical core."""
    instance_type: str
    region: str
    ami_arch: str  # aws_session's AMI architecture
    machine: str  # what the binaries must be built for (uname -m)
    cpu: str
    #: The timed CPU: vCPU 1 (vCPU 0 keeps the system). On Graviton and AMD .large every
    #: vCPU is a physical core; on Intel .large the two vCPUs are one core's SMT threads.
    timed_cpu: int = 1


TARGETS = {
    "graviton2": Target("c6g.large", "eu-west-1", "arm64", "aarch64", "Graviton2, Neoverse N1"),
    "graviton3": Target("c7g.large", "eu-west-1", "arm64", "aarch64", "Graviton3, Neoverse V1"),
    "graviton4": Target("c8g.large", "eu-west-1", "arm64", "aarch64", "Graviton4, Neoverse V2"),
    "zen5": Target("c8a.large", "eu-west-1", "amd64", "x86_64", "EPYC Turin, Zen 5"),
    "intel": Target("c7i.large", "eu-west-1", "amd64", "x86_64",
                    "Xeon Sapphire Rapids (one core, its SMT sibling runs the system)"),
}
#: The everyday pair (docs/results/cpu-targets-2026-11-21.md): the two lowest-noise Arm
#: cores; add zen5 for x86 when the builds are commits (a binary serves one architecture).
DEFAULT_TARGETS = ("graviton3", "graviton4")
DEFAULT_CLIPS = ("crowd_mot17_09", "bodycam_airport", "pan_jaad_0117", "lowlight_pp22_004")
DEFAULT_FRAMES = 150
DEFAULT_KBPS = 600
DEFAULT_ROUNDS = 5
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
#: Per box: budget cap and hard lifetime (a .large is a few cents an hour).
BOX_BUDGET_USD = 1.0
BOX_LIFETIME_MINUTES = 75
PREPARE_MINUTES = 8
#: The full run's limit: this many times the pipe-cleaner's estimate, plus a margin.
RUN_TIMEOUT_FACTOR = 2.5
RUN_TIMEOUT_MARGIN_MINUTES = 10
SHARD_ATTEMPTS = 2
#: Two-sided 95 % Student t quantiles by degrees of freedom (larger: the normal's).
T975 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}
NORMAL_975 = 1.960


# --- what is measured ------------------------------------------------------------------

@dataclass(frozen=True)
class Side:
    """One side of a pair: a build (commit or binary path) and its extra arguments."""
    build: str
    params: tuple[str, ...] = ()

    def argv_tail(self) -> list[str]:
        return [a for p in self.params for a in ("--param", p)]


@dataclass(frozen=True)
class Pair:
    name: str
    base: Side
    cand: Side
    rungs: tuple[str, ...]  # codec:preset
    clips: tuple[str, ...]
    frames: int
    rate: tuple[str, ...]  # ("--bitrate", "600") or ("--qp", "32")


def pairs_from_args(args) -> list[Pair]:
    rate = ("--qp", str(args.qp)) if args.qp is not None else ("--bitrate", str(args.bitrate))
    if args.plan:
        return [pair_from_json(entry, args, rate) for entry in json.loads(args.plan.read_text())]
    if not (args.base and args.cand and args.rungs):
        raise SystemExit("give --base, --cand and --rungs, or --plan")
    return [Pair("ab", Side(args.base, tuple(args.base_param)), Side(args.cand, tuple(args.cand_param)),
                 tuple(args.rungs.split(",")), tuple(args.clips.split(",")), args.frames, rate)]


def pair_from_json(entry: dict, args, rate: tuple[str, ...]) -> Pair:
    """A --plan entry: {"name", "base": {"build", "params"}, "cand": {...}, "rungs",
    optional "clips", "frames", "bitrate" / "qp"} (missing fields: the command line's)."""
    if "qp" in entry:
        rate = ("--qp", str(entry["qp"]))
    elif "bitrate" in entry:
        rate = ("--bitrate", str(entry["bitrate"]))
    return Pair(entry["name"], Side(entry["base"]["build"], tuple(entry["base"].get("params", ()))),
                Side(entry["cand"]["build"], tuple(entry["cand"].get("params", ()))),
                tuple(entry["rungs"]), tuple(entry.get("clips", args.clips.split(","))),
                int(entry.get("frames", args.frames)), rate)


def load_clips(names: set[str]) -> dict[str, dict]:
    known = {c["name"]: c for c in json.loads(CLIPSET.read_text())["clips"]}
    missing = sorted(names - set(known))
    if missing:
        raise SystemExit(f"not in {CLIPSET}: {missing}")
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
                jobs.append({"id": f"{pair.name}|{rung}|{name}", "variants": {
                    side: [f"bin/{build_key(getattr(pair, side).build)}/nanocodec", *common,
                           *getattr(pair, side).argv_tail()] for side in ("base", "cand")}})
    return jobs


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
    return freed


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
    subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(tree), commit],
                   check=True, capture_output=True)
    try:
        path = f"target/{RUST_TRIPLES[machine]}/release/nanocodec"
        fetched = WORK / f"fetch-{machine}-{commit[:12]}-{os.getpid()}"
        what = f"building {commit[:12]} for {machine} on the build box ({name})"
        say(what)
        run_watched([str(NC_REMOTE), "run", "--name", name, "--fetch", path,
                     "--fetch-to", str(fetched), "--", build_command(machine)],
                    what, BUILD_TIMEOUT_S, cwd=tree)
        shutil.move(str(fetched / path), binary)
        shutil.rmtree(fetched)
    finally:
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", str(tree)],
                       check=False)


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
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(f".part{os.getpid()}")
    with source.open("rb") as raw:
        subprocess.run(["zstd", "-q", "-3", "-T4", "-o", str(partial)], input=raw.read(
            frames * frame_bytes), check=True)
    partial.rename(out)
    return out


def stage(stage_dir: Path, jobs: list[dict], binaries: dict[str, Path],
          clips: dict[str, dict]) -> None:
    """Symlinks for one box: the binaries its jobs name, their clips, the plan, the runner."""
    (stage_dir / "out").mkdir(parents=True)
    for key in sorted({j["variants"][s][0].split("/")[1] for j in jobs for s in j["variants"]}):
        (stage_dir / "bin" / key).mkdir(parents=True)
        (stage_dir / "bin" / key / "nanocodec").symlink_to(binaries[key])
    (stage_dir / "clips").mkdir()
    for job in jobs:
        for argv in job["variants"].values():
            file = argv[argv.index("--input") + 1].removeprefix("clips/")
            link = stage_dir / "clips" / f"{file}.zst"
            if not link.exists():
                name, frames = file.removesuffix(".yuv").rsplit("-", 1)
                link.symlink_to(cached_clip(clips[name], int(frames.removesuffix("f"))))
    (stage_dir / "plan.json").write_text(json.dumps({"jobs": jobs}, indent=1) + "\n")
    for name in BOX_FILES:
        (stage_dir / name).symlink_to(Path(__file__).resolve().parent / name)


# --- one box ---------------------------------------------------------------------------

def prepare_command(cpu: int) -> str:
    """Readiness: clips unpacked and read once (page cache), counters allowed, every
    binary runs, the CPU and the counted events recorded."""
    return (f"set -o pipefail; cd ~/{REMOTE_DIR} && for c in clips/*.zst; do zstd -d -q --rm \"$c\"; done && "
            # counters for a user process (a box without sudo just counts nothing)
            "{ sudo -n sysctl -q kernel.perf_event_paranoid=1 || true; } && "
            "for b in bin/*/nanocodec; do echo \"$b $($b --version)\"; done | tee out/versions.txt && "
            "lscpu > out/lscpu.txt && grep -m1 'Model name' out/lscpu.txt && "
            "cat clips/* > /dev/null && "
            f"python3 cpu_ab_box.py --plan plan.json --probe out/events.json --core {cpu}")


def box_command(cpu: int, rounds: int, out: str, limit: int | None = None) -> str:
    return (f"cd ~/{REMOTE_DIR} && python3 cpu_ab_box.py --plan plan.json --out {out} "
            f"--events out/events.json --core {cpu} --rounds {rounds}" + (f" --limit {limit}" if limit else ""))


def run_timeout_minutes(pipe_runs: Path, jobs: int, rounds: int) -> float:
    """The full run's limit from the pipe-cleaner (one round of the first job)."""
    seconds = sum(json.loads(line)["wall_s"] for line in pipe_runs.read_text().splitlines())
    return RUN_TIMEOUT_FACTOR * seconds * jobs * rounds / 60 + RUN_TIMEOUT_MARGIN_MINUTES


def run_shard(stage_dir: Path, target: Target, out: Path, rounds: int, on_demand: bool,
              purpose: str, session_for=Session) -> None:
    """One box: upload, readiness, pipe-cleaner, the rounds, results back."""
    jobs = len(json.loads((stage_dir / "plan.json").read_text())["jobs"])
    spec = SessionSpec(region=target.region, instance_type=target.instance_type,
                       arch=target.ami_arch, purpose=purpose, on_demand=on_demand,
                       lifetime_minutes=BOX_LIFETIME_MINUTES, budget_usd=BOX_BUDGET_USD,
                       root_gb=16)
    remote_out = f"~/{REMOTE_DIR}/out"
    started = time.monotonic()
    with session_for(spec) as session:
        session.upload(stage_dir, f"~/{REMOTE_DIR}")
        try:
            session.run_detached(prepare_command(target.timed_cpu), f"{remote_out}/prepare.log",
                                 PREPARE_MINUTES)
            session.run_detached(box_command(target.timed_cpu, 1, "out/pipe.jsonl", 1),
                                 f"{remote_out}/pipe.log", PREPARE_MINUTES)
            session.download(remote_out, out)
            timeout = min(run_timeout_minutes(out / "pipe.jsonl", jobs, rounds),
                          BOX_LIFETIME_MINUTES - PREPARE_MINUTES)
            say(f"{jobs} jobs x {rounds} rounds, limit {timeout:.0f} min")
            session.run_detached(box_command(target.timed_cpu, rounds, "out/runs.jsonl"),
                                 f"{remote_out}/runs.log", timeout)
        finally:
            session.download(remote_out, out)
    (out / "session.json").write_text(json.dumps({
        "instance_type": target.instance_type, "region": target.region,
        "minutes": round((time.monotonic() - started) / 60, 2), "on_demand": on_demand,
        "jobs": jobs, "rounds": rounds}) + "\n")


def launch_shards(stage_dirs: dict[str, Path], run_dir: Path, rounds: int, on_demand: bool,
                  purpose: str) -> list[str]:
    """Every shard in its own process (aws_session's signal handling needs a main thread),
    all at once; a failed shard is retried once on demand. Returns the shards that failed."""
    def start(label: str, attempt: int, demand: bool) -> subprocess.Popen:
        target = label.split("-")[0]
        log = (run_dir / f"{label}.attempt{attempt}.log").open("w")
        say(f"{label}: box {TARGETS[target].instance_type} (log {log.name})")
        return subprocess.Popen(
            [sys.executable, __file__, "shard", "--stage", str(stage_dirs[label]),
             "--target", target, "--out", str(run_dir / label), "--rounds", str(rounds),
             "--purpose", purpose] + (["--on-demand"] if demand else []),
            stdout=log, stderr=subprocess.STDOUT)

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
                say(f"{label}: failed (exit {process.returncode}), retrying on demand")
                shutil.rmtree(run_dir / label, ignore_errors=True)
                running[label] = (start(label, attempt + 1, True), attempt + 1)
                started[label] = time.monotonic()
            else:
                say(f"{label}: failed twice; see its logs in {run_dir}")
                failed.append(label)
    return failed


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
    for shard in sorted(p for p in run_dir.iterdir() if (p / "runs.jsonl").is_file()):
        target = shard.name.split("-")[0]
        for line in (shard / "runs.jsonl").read_text().splitlines():
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


# --- entry points ----------------------------------------------------------------------

def main_ab(args) -> None:
    targets = args.targets.split(",")
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown targets {unknown}; known: {sorted(TARGETS)}")
    freed = prune()
    if freed:
        say(f"pruned old builds, clips and runs from {WORK}: {freed / 1e9:.2f} GB")
    pairs = pairs_from_args(args)
    clips = load_clips({c for p in pairs for c in p.clips})
    jobs = jobs_for(pairs, clips)
    builds = sorted({side.build for p in pairs for side in (p.base, p.cand)})
    machines = sorted({TARGETS[t].machine for t in targets})
    binaries = all_builds(builds, machines)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = args.out or WORK / f"run-{stamp}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pairs.json").write_text(json.dumps([dataclasses.asdict(p) for p in pairs],
                                                   indent=1) + "\n")
    stage_dirs = {}
    for target in targets:
        for index, part in enumerate(shards(jobs, args.boxes), 1):
            label = f"{target}-{index}"
            stage_dirs[label] = run_dir / "stage" / label
            stage(stage_dirs[label], part, binaries[TARGETS[target].machine], clips)
    say(f"{len(jobs)} jobs x {args.rounds} rounds on {len(stage_dirs)} boxes "
        f"({', '.join(f'{t} {TARGETS[t].instance_type}' for t in targets)}); results in {run_dir}")
    if args.dry_run:
        for label, path in stage_dirs.items():
            subprocess.run(["du", "-shL", str(path)], check=True)
        print(prepare_command(1))
        print(box_command(1, args.rounds, "out/runs.jsonl"))
        return
    try:
        failed = launch_shards(stage_dirs, run_dir, args.rounds, args.on_demand, args.purpose)
    finally:
        shutil.rmtree(run_dir / "stage", ignore_errors=True)
    summary = summarize(run_dir)
    summary["failed_shards"] = failed
    (args.json or run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_summary(summary)
    if failed:
        raise SystemExit(f"shards failed: {failed} (their rows are missing)")


def main(argv: list[str]) -> None:
    if argv[:1] == ["shard"]:  # one box, started by launch_shards
        p = argparse.ArgumentParser()
        p.add_argument("--stage", type=Path, required=True)
        p.add_argument("--target", required=True, choices=sorted(TARGETS))
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--rounds", type=int, required=True)
        p.add_argument("--purpose", required=True)
        p.add_argument("--on-demand", action="store_true")
        a = p.parse_args(argv[1:])
        run_shard(a.stage, TARGETS[a.target], a.out, a.rounds, a.on_demand, a.purpose)
        return
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
    p.add_argument("--clips", default=",".join(DEFAULT_CLIPS), help=f"names in {CLIPSET}")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--bitrate", type=int, default=DEFAULT_KBPS, help="CBR kbit/s")
    p.add_argument("--qp", type=int, help="fixed QP instead of CBR")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                   help=f"comma-separated, of {', '.join(sorted(TARGETS))}")
    p.add_argument("--boxes", type=int, default=1, help="boxes per target (jobs split)")
    p.add_argument("--on-demand", action="store_true", help="on-demand boxes, not spot")
    p.add_argument("--purpose", default="cpu-ab")
    p.add_argument("--out", type=Path, help="run directory (default under WORK)")
    p.add_argument("--json", type=Path, help="summary file (default RUN/summary.json)")
    p.add_argument("--dry-run", action="store_true", help="stage and print; launch nothing")
    main_ab(p.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
