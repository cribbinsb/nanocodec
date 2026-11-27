#!/usr/bin/env python3
"""The chart's paired x86 (or Graviton) timing on a rented bare-metal spot instance.

    python3 tools/dev/aws_timing.py [--instance-type c7i.metal-24xl] [--points a,b]
        [--repeats 2] [--rounds 11] [--out DIR] [--dry-run]

The PC's timing cores share a boosting package with everything else on the
machine; a bare-metal instance gives one clean session for every chart point
at once. This runs anchor_chart_timing.py `time-cloud` there: every point
back to back with x264 medium and SVT-AV1 M13 on the same core and clip, in
rotated rounds, the median per-round time ratio kept (the same pairing as
`time-host`), each (point, clip) pair on its own physical core.

Uploaded (one tar stream, a few hundred MB): our chart timing build (the x86
release build, chart_refresh's `nanocodec-timed-release`, of the build the chart's
build record names), the reference encoders the points name (the timed reference
build, nceval.config.reference_bin: X86_TOOLS, GRAVITON_TOOLS), anchor_chart_timing.py, the chart's
timing plan (CBR: a subset of the chart's own quality encodes, nceval.chart_timing),
and its timing clips cut to the plan's frames (decoded where stored encoded) and
zstd-compressed (cached under WORK_DIR/cache). The plan gains one derived
point on x86, `svtav1-m13@asm-avx2` (M13 with `--asm avx2`): against M13's
default (`--asm max`, AVX-512 where the CPU has it) it is the speed of SVT's
AVX-512 paths.

The whole set is timed `--repeats` times in a row (the spread between
repeats is the session's own noise). Results (the per-point records and a
summary.json with fps and ratios beside the stored chart's x86 ratios) land
in --out. Instances, costs and rules: tools/dev/aws_session.py; for Graviton
(GRAVITON_MACHINE, the site's server Arm series, or c8g.metal-24xl) pass our aarch64
build with --binary and --expect-version; the references default to GRAVITON_TOOLS,
their native Graviton build (x265 with SVE/SVE2), --tools-dir to time others.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))  # nceval (clip paths)
import anchor_chart_timing as act  # noqa: E402
from aws_session import STALL_MINUTES, WORK_DIR, Session, SessionSpec, say  # noqa: E402
from nceval.config import reference_bin  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / "nanocodec-data"
PLAN_FILE = REPO / "docs/results/bd-vs-speed-plan.json"
BUILDS_FILE = REPO / "docs/results/bd-vs-speed-builds.json"
LOCAL_TIMING_FILE = REPO / "docs/results/bd-vs-speed-timing-x86.json"
BUILDS_ROOT = DATA / "charts/builds"
TIMED_BINARY = "nanocodec-timed-release"
#: The timed reference encoders (nceval.config.REFERENCE_BUILDS switches them). Their
#: openh264-harness has --bitrate (the plain anchors' 2026-11-06 copy had none, which
#: X86_OWN_TOOLS worked around until 2026-11-23).
X86_TOOLS = reference_bin("x86_64")
#: The timing sessions' clip archives (clip_archive): a capped cache of its own
#: (tools/dev/retention.toml work/aws-clip-cache, pruned by prune_clip_cache), outside
#: work/aws, which holds the durable ledger (until 2026-11-24 they sat in work/aws/cache:
#: 21 GiB that no tool could prune).
CLIP_CACHE = WORK_DIR.parent / "aws-clip-cache"
#: Known timing boxes: (region, AMI architecture); no c7a (owner: double-pumped AVX-512,
#: docs/aws-cost-guide.md). The AMD 8th-generation sizes
#: have no SMT (a vCPU is a physical core), so even a .large is alone on its cores.
INSTANCES = {"c7i.metal-24xl": ("eu-north-1", "amd64"),  # Sapphire Rapids, AVX-512 full width
             "c8a.large": ("eu-west-1", "amd64"),  # Zen 5, AVX-512 full width, 2 cores
             "c8a.4xlarge": ("eu-west-1", "amd64"),  # Zen 5, 16 cores: the x86 timing machine
             "c8a.metal-24xl": ("eu-west-1", "amd64"),  # Zen 5, bare metal
             "c8g.large": ("eu-west-1", "arm64"),  # Graviton4, Neoverse V2, 2 cores (no SMT)
             "c6a.large": ("eu-west-1", "amd64"),  # Zen 3 (Milan), AVX2 only: 1 core, 2 SMT threads
             "c8g.metal-24xl": ("eu-north-1", "arm64")}  # Graviton4, Neoverse V2, bare metal
#: The x86 timing machine (2026-11-15 probe, 5 repeats of H.264 against x264 medium: the
#: time ratio's largest deviation from the median was 0.2% on c8a.large, 0.7% on
#: c7a.large, 0.9% on c7a.metal-48xl, 0.3% on c7i.metal-24xl): Zen 5, the newest x86
#: core offered, full-width AVX-512, alone on its two cores (no SMT), one pair at a time.
#: Not a bigger VM with pairs side by side: on c8a.4xlarge (14 at once) two repeats of
#: the chart differed by 1.5% at the median and up to 8.6%.
TIMING_MACHINE = "c8a.large"
#: The server Arm timing machine beside it (the site's third speed series, 2026-11-17):
#: Graviton4 (Neoverse V2), the same 2-core size and method as TIMING_MACHINE.
GRAVITON_MACHINE = "c8g.large"
#: The x86 timing machine without AVX-512 (the site's fourth speed series, 2026-11-17): Zen 3
#: (EPYC Milan), where our encoder and the references run their AVX2 paths. The same size and
#: method as TIMING_MACHINE, except that its two vCPUs are the SMT threads of one core: the
#: timed thread's sibling is the system's (the platform note says so, `pinning_command`).
ZEN3_MACHINE = "c6a.large"
#: Reference encoders for an arm64 instance: their native Graviton build
#: (tools/dev/aws_reference_builds.py graviton; until 2026-11-23 the Jetson's builds,
#: with x265's SVE/SVE2 kernels left out).
GRAVITON_TOOLS = reference_bin("graviton")
#: Cores left to the system before the timed ones: two on a big box (time-cloud's
#: default), one on a 2-core .large (its other core times the pairs one after another).
#: The first timed core is also the quiet core of the kernel benchmarks and profiles.
SYSTEM_CORES = {"large": 1}
DEFAULT_SYSTEM_CORES = 2
#: The timed CPU where one physical core per pair cannot be had: ZEN3_MACHINE's two vCPUs
#: are one core's SMT threads, so its pairs run on vCPU 1 and the system keeps vCPU 0.
TIMED_CPUS = {ZEN3_MACHINE: "1"}
#: The chart's reference points timed beside ours by default.
DEFAULT_REFERENCES = ("svtav1-m13", "svtav1-m12", "x264-ultrafast", "x264-veryfast",
                      "x264-medium", "openh264-baseline")
AVX2_VARIANT = f"{act.SECOND_ANCHOR}@asm-avx2"
#: Our encoder forced to its AVX2 DSP tier (docs/handoff/avx512.md "Dispatch and
#: forcing"): the same binary and stream, against its default (AVX-512) tier.
ISA_VARIANT_SUFFIX = "@isa-avx2"
ISA_VARIANT_ENV = {"NC_DSP_ISA": "avx2"}
#: Chart clips each profiled point is recorded on.
PERF_CLIPS = 2
REMOTE_DIR = "timing"
ZSTD_LEVEL = 12
#: Minutes the remote timing may take, per repeat (boot and upload come on top), and
#: the readiness and pipe-cleaner stages' limit. 0.6.0's 35 families took about 37 min a
#: repeat on c8g.large (105 pairs; 30 stopped it at pair 86 of 105), inside the 90-min lifetime.
REPEAT_TIMEOUT_MINUTES = 50
PREPARE_MINUTES = 5
#: Plan entries the pipe-cleaner times within one PREPARE_MINUTES (a 3-clip plan's).
PIPE_CLEANER_ENTRIES = 3
CHECKASM_MINUTES = 20
PERF_MINUTES = 10
#: A repeat prints once per (point, clip) pair, after all its rounds: one pair of a slow
#: reference takes longer than aws_session's default stall (x265 slow on c6a.large, Zen 3:
#: over 5 min for one pair, 2026-11-17).
REPEAT_STALL_MINUTES = 15
#: Failed (point, clip) pairs a repeat may record before the session is stopped.
MAX_FAILED_PAIRS = 2


@dataclass
class RemoteJob:
    log: str
    command: str
    timeout_minutes: float
    max_errors: int = 0
    #: Silence this long is a stall (a stage that prints only at its end sets its own).
    stall_minutes: float = STALL_MINUTES


def chart_points() -> list[str]:
    from_builds = json.loads(BUILDS_FILE.read_text())["families"]
    return [*from_builds, *DEFAULT_REFERENCES]


#: Each of our points run by an earlier build (`--baseline-binary`), as `<point>@base`,
#: paired with the point in the same rounds on the same core: a before/after speed
#: comparison of two builds. Only the variants are timed (the point is their partner).
BASELINE_SUFFIX = "@base"
BASELINE_BINARY = "nanocodec-base"


def cloud_plan(points: list[str], arch: str, plan_file: Path, isa_variants: bool,
               baseline: bool = False, inputs: set[str] | None = None) -> dict:
    """The plan's entries for `points` (and both anchors), plus the AVX2-only M13 and,
    with `isa_variants`, each of our points again on our AVX2 tier (ISA_VARIANT); with
    `baseline`, each of our points again on the baseline build (BASELINE_SUFFIX). With
    `inputs`, only the entries that encode those clips (`input_subset`)."""
    from nceval.chart_timing import load_plan

    plan = load_plan(plan_file)
    if inputs is not None:
        plan = input_subset(plan, inputs)
    keep = {*points, plan["anchor"], act.SECOND_ANCHOR}
    entries = {n: v for n, v in plan["points"].items() if n in keep}
    if arch == "amd64":
        entries[AVX2_VARIANT] = {
            clip: {**entry, "argv": [*entry["argv"], "--asm", "avx2"]}
            for clip, entry in plan["points"][act.SECOND_ANCHOR].items()}
    if isa_variants:
        for name in [n for n in entries if n.startswith("nanocodec")]:
            entries[f"{name}{ISA_VARIANT_SUFFIX}"] = {
                clip: {**entry, "env": ISA_VARIANT_ENV, "pair_with": name}
                for clip, entry in entries[name].items()}
    if baseline:
        for name in [n for n in entries if n.startswith("nanocodec")]:
            entries[f"{name}{BASELINE_SUFFIX}"] = {
                clip: {**entry, "argv": [f"{{BIN}}/{BASELINE_BINARY}", *entry["argv"][1:]],
                       "pair_with": name}
                for clip, entry in entries[name].items()}
    return {**{k: v for k, v in plan.items() if k != "points"}, "points": entries}


def isolated(function, *args):
    """`function(*args)` in a child process: the plan and the clip staging import the eval
    package (~100 MB, nceval.chart), which this process, waiting on its session for hours
    and run tens at a time, should not hold (the PC's memory, 2026-11-23)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as child:
        return child.submit(function, *args).result()


def input_subset(plan: dict, inputs: set[str]) -> dict:
    """`plan` restricted to the entries that encode the clips `inputs`: an every-encode
    plan split by clip over many sessions, so each uploads only its own clips and every
    entry's points are paired with the anchor in one session."""
    keys = [k for k in act.plan_clips(plan) if act.plan_input(plan, k) in inputs]
    missing = inputs - {act.plan_input(plan, k) for k in keys}
    if missing:
        raise SystemExit(f"--inputs not in the plan: {sorted(missing)}")
    return {**plan, "clips": {k: plan["clips"][k] for k in keys},
            "inputs": {k: plan["inputs"][k] for k in keys},
            "points": {n: {k: e[k] for k in keys if k in e} for n, e in plan["points"].items()}}


def slowest_first(points: list[str]) -> list[str]:
    """The points in descending order of their stored x86 time, so that the long pairs
    start first and the cores finish together (unknown points first)."""
    stored = json.loads(LOCAL_TIMING_FILE.read_text())["points"]
    base = AVX2_VARIANT.split("@")[0]

    def load(name: str) -> float:
        entry = stored.get(name) or stored.get(base if name == AVX2_VARIANT else "")
        return sum(entry["ms_per_frame"].values()) if entry else float("inf")
    return sorted(points, key=load, reverse=True)


def chart_binaries(plan: dict) -> dict[str, Path]:
    """Each of our families' timing binary as the chart's build record has it (its
    `timed_on` build, else its quality build), staged as `nanocodec-<build>`: the
    plan's commands are pointed at it (a variant runs its family's binary)."""
    record = json.loads(BUILDS_FILE.read_text())
    timed_on = record.get("timed_on", {})
    staged = {}
    for name, clips in plan["points"].items():
        if not name.startswith("nanocodec"):
            continue
        family = name.split(ISA_VARIANT_SUFFIX)[0]
        build = timed_on.get(family, record["families"][family])
        staged[f"nanocodec-{build}"] = BUILDS_ROOT / build / TIMED_BINARY
        for entry in clips.values():
            entry["argv"] = [f"{{BIN}}/nanocodec-{build}", *entry["argv"][1:]]
    return staged


def prune_clip_cache() -> int:
    """The archives, least recently used first, until the cache is within its cap
    (retention.toml), after the partials of sessions that died writing them; returns
    bytes freed. A live session's partial is left alone."""
    from nceval import retention
    cap = retention.cap_bytes(CLIP_CACHE, DATA)
    if cap is None or not CLIP_CACHE.is_dir():
        return 0
    freed = 0
    for partial in sorted(CLIP_CACHE.glob("*.partial-*")):  # a session that died writing it
        pid = partial.name.rsplit("-", 1)[-1]
        if pid.isdigit() and not Path(f"/proc/{pid}").exists() and \
                retention.guard(partial, "aws_timing") is None:
            freed += partial.lstat().st_size
            partial.unlink()
    return freed + retention.prune_to_cap(sorted(CLIP_CACHE.glob("*.yuv.zst")), cap,
                                          "aws_timing")[0]


def clip_archive(clip, frames: int) -> Path:
    """The clip's measured frames (from its start frame, decoded where it is stored
    encoded: nceval.clipframes, as its quality points read it), zstd-compressed, cached by
    clip name, start and length."""
    from nceval.clipframes import ClipFrames

    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    archive = CLIP_CACHE / f"{clip.name}-s{clip.start_frame}-{frames}f.yuv.zst"
    if archive.is_file():
        os.utime(archive)  # in use: the cache's eviction is least recently used first
    else:
        # one partial per process: sessions started side by side stage the same clips
        partial = archive.with_suffix(f".partial-{os.getpid()}")
        with ClipFrames(clip, frames, CLIP_CACHE) as source, open(source.reader(), "rb") as raw:
            subprocess.run(["zstd", "-q", "-T8", f"-{ZSTD_LEVEL}", "-f", "-o", str(partial)],
                           stdin=raw, check=True)
        partial.rename(archive)
    return archive


def stage(stage_dir: Path, plan: dict, ours: dict[str, Path], tools_dir: Path) -> None:
    from nceval.chart import chart_clips

    (stage_dir / "bin").mkdir(parents=True)
    (stage_dir / "clips").mkdir()
    names = {e["argv"][0].removeprefix("{BIN}/") for p in plan["points"].values()
             for e in p.values()}
    for name in names:
        source = ours.get(name, tools_dir / name)
        (stage_dir / "bin" / name).symlink_to(source.resolve())
    clips = {c.name: c for c in chart_clips(plan["clipset"])}
    for name in sorted({act.plan_input(plan, key) for key in act.plan_clips(plan)}):
        (stage_dir / "clips" / f"{name}.yuv.zst").symlink_to(
            clip_archive(clips[name], act.input_frames(plan, name)))
    (stage_dir / "anchor_chart_timing.py").symlink_to(Path(act.__file__).resolve())
    (stage_dir / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")


def unstage(stage_dir: Path) -> None:
    """Remove the staging directory entry by entry: symlinks, the plan, empty dirs."""
    for sub in ("bin", "clips", "out"):
        if (stage_dir / sub).is_dir():
            for entry in (stage_dir / sub).iterdir():
                entry.unlink()
            (stage_dir / sub).rmdir()
    for name in ("anchor_chart_timing.py", "plan.json"):
        (stage_dir / name).unlink(missing_ok=True)
    stage_dir.rmdir()


#: The SIMD extensions the site records with an x86 timing machine (/proc/cpuinfo names).
ISA_FLAGS = ("avx2", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512ifma",
             "avx512vbmi", "avx512_vbmi2", "avx512_bf16")
#: nc-dsp's AVX-512 tier needs these (nc_dsp::cpu, avx512_supported); else AVX2.
OUR_AVX512_FLAGS = ("avx512f", "avx512bw", "avx512vl")


#: The SIMD extensions the site records with an Arm timing machine (/proc/cpuinfo
#: `Features` names); nc-dsp has one aarch64 tier, NEON (nc_dsp::cpu).
ARM_ISA_FLAGS = ("asimd", "asimddp", "i8mm", "sve", "sve2", "svei8mm", "bf16")


def isa_command(arch: str = "amd64") -> str:
    """Remote shell writing the platform note: the CPU model, its SIMD extensions
    (ISA_FLAGS, ARM_ISA_FLAGS on Arm) and the tier our encoder dispatches to; isa.txt
    keeps x264's and x265's and SVT-AV1's own capability lines (the references choose
    for themselves; libaom, libvpx and openh264 print none)."""
    if arch == "arm64":
        flags, flag_line, model = "|".join(ARM_ISA_FLAGS), "^Features", "lscpu | grep -m1 'Model name'"
        tier = "tier=neon; "
    else:
        flags, flag_line, model = "|".join(ISA_FLAGS), "^flags", "grep -m1 'model name' /proc/cpuinfo"
        ours = " && ".join(f"grep -qw {f} out/flags.txt" for f in OUR_AVX512_FLAGS)
        tier = f"tier=$({ours} && echo avx512 || echo avx2); "
    return (f"tr ' ' '\\n' < <(grep -m1 '{flag_line}' /proc/cpuinfo) | "
            f"grep -xE '{flags}' > out/flags.txt; " + tier +
            "head -c 6144 /dev/zero > out/blank.yuv; "  # one 64x64 I420 frame
            "bin/x264 --input-res 64x64 --frames 1 -o /dev/null out/blank.yuv 2>&1 "
            "| grep -i 'cpu capabilities' > out/isa.txt; "
            "[ ! -x bin/x265 ] || bin/x265 --input out/blank.yuv --input-res 64x64 --fps 30 "
            "--frames 1 -o /dev/null 2>&1 | grep -i 'cpu capabilities' >> out/isa.txt; "
            "timeout 60 bin/SvtAv1EncApp -w 64 -h 64 -n 1 -i out/blank.yuv -b /dev/null 2>&1 "
            "| grep -iE 'asm|cpu' >> out/isa.txt; " + (avx512_probe() if arch == "amd64" else "") +
            f"echo \"$({model} | cut -d: -f2 | xargs)"
            " (SIMD: $(paste -sd, out/flags.txt | sed 's/,/, /g'); nanocodec dispatches to "
            "its $tier tier)\" | tee out/platform.txt; rm -f out/blank.yuv")


def avx512_probe() -> str:
    """Remote shell asking each of our binaries for its AVX-512 tier (NC_DSP_ISA=avx512) on
    one blank frame and recording the answer in isa.txt: a CPU without the tier refuses it
    ("not supported by this CPU/build"), so the binary itself confirms it runs AVX2."""
    encode = ("encode --codec h264 --input out/blank.yuv --width 64 --height 64 --fps 30 "
              "--frames 1 --qp 30 --output /dev/null")
    return ("for b in bin/nanocodec*; do echo \"$b NC_DSP_ISA=avx512: $(NC_DSP_ISA=avx512 "
            f"\"$b\" {encode} 2>&1 >/dev/null | grep -o 'not supported by this CPU/build' "
            "|| echo accepted)\" >> out/isa.txt; done; ")


def pinning_command() -> str:
    """Remote shell writing how the timed thread sits (out/pinning.txt), from lscpu: on a
    2-vCPU box whose vCPUs are one core's two SMT threads (c6a.large), the timed thread
    shares its core with the system's; elsewhere each timed thread has a physical core."""
    return ("{ if [ \"$(nproc)$(lscpu | sed -n 's/^Thread(s) per core: *//p')\" = 22 ]; then "
            "echo 'one thread pinned to vCPU 1, the SMT sibling of vCPU 0 (left to the system)'; "
            "else echo 'one thread pinned to one physical core'; fi; } > out/pinning.txt")


#: Shared libraries a reference build needs that a fresh Ubuntu image may lack, by package
#: (the Jetson's aarch64 x265 links libnuma).
RUNTIME_PACKAGES = {"libnuma.so.1": "libnuma1"}


def runtime_libraries() -> str:
    """Remote shell installing RUNTIME_PACKAGES whose library the image lacks."""
    return "".join(f"{{ ldconfig -p | grep -q {lib} || {{ sudo apt-get -qq update && "
                   f"sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y {package} "
                   f"> /dev/null; }}; }} && " for lib, package in RUNTIME_PACKAGES.items())


def remote_commands(points: list[str], repeats: int, rounds: int, versions: dict[str, str],
                    plan: dict, checkasm: bool, perf_points: list[str],
                    system_cores: int = DEFAULT_SYSTEM_CORES,
                    arch: str = "amd64", timed_cpus: str | None = None,
                    repeat_minutes: float = REPEAT_TIMEOUT_MINUTES) -> list[RemoteJob]:
    """The remote stages, fail-fast first: readiness checks (minutes), a pipe-cleaner
    (one of our points, its AVX2-tier variant and the AVX2 M13, one round), the kernel
    benchmarks when asked for (before the timing, on a quiet machine), one job per
    repeat, and the profiles last (each stage runs alone on the machine)."""
    cd = f"cd ~/{REMOTE_DIR} && "
    prepare = (cd + "command -v zstd >/dev/null && "
               "for c in clips/*.zst; do zstd -d -q --rm \"$c\"; done && "
               "ls -l clips && " + runtime_libraries() + "! ldd bin/* | grep 'not found' && "
               + "".join(f"test \"$(bin/{name} --version)\" = {shlex.quote(version)} && "
                         for name, version in versions.items()) +
               "bin/x264 --version | head -1 && bin/SvtAv1EncApp --version | head -1 && "
               "lscpu > out/lscpu.txt && " + pinning_command() + " && "
               + isa_command(arch))
    platform = '"$(cat out/platform.txt), $(cat out/pinning.txt)"'

    def timing(label: str, names: list[str], timed_rounds: int) -> str:
        return (cd + f"python3 anchor_chart_timing.py time-cloud --plan plan.json --bin-dir bin "
                f"--clips-dir clips --points {','.join(names)} --rounds {timed_rounds} "
                + (f"--cpus {timed_cpus} " if timed_cpus else f"--skip-cores {system_cores} ") +
                f"--platform {platform} --work-dir scratch --out-dir out/{label} && "
                f"test ! -e out/{label}/errors.json")

    base = [p for p in points if p.startswith("nanocodec") and
            not p.endswith(ISA_VARIANT_SUFFIX)][:1]
    cleaner = base + [p for p in points if p == AVX2_VARIANT or
                      (base and p == f"{base[0]}{ISA_VARIANT_SUFFIX}")]
    cleaner = cleaner or points[:1]  # a session of references only
    cleaner_minutes = PREPARE_MINUTES * math.ceil(len(plan["clips"]) / PIPE_CLEANER_ENTRIES)
    jobs = [RemoteJob("prepare", prepare, PREPARE_MINUTES),
            # one round of each cleaner point on every entry, after the anchors' ratio on
            # each: an every-encode plan's 50 entries took ~4 min of a 5-min limit, so the
            # limit grows with the entries (PIPE_CLEANER_ENTRIES a PREPARE_MINUTES)
            # (it prints nothing before the anchors' ratio on every entry is done, so its
            # stall limit is its time limit: 60 entries passed 5 silent minutes, 2026-11-23)
            RemoteJob("pipe-cleaner", timing("pipe-cleaner", cleaner, 1), cleaner_minutes,
                      stall_minutes=cleaner_minutes)]
    if checkasm:
        jobs.append(RemoteJob("checkasm", cd + " && ".join(
            f"taskset -c {timed_cpus or system_cores} bin/nc-checkasm --bench --isa {isa} "
            f"| tee out/checkasm-{isa}.txt" for isa in ("avx2", "avx512")), CHECKASM_MINUTES,
            stall_minutes=CHECKASM_MINUTES))
    for repeat in range(1, repeats + 1):
        # the pipe-cleaner proved every stage: a few failed pairs are recorded and
        # reported, more stop the session
        jobs.append(RemoteJob(f"repeat-{repeat}", timing(f"repeat-{repeat}", points, rounds),
                              repeat_minutes, MAX_FAILED_PAIRS,
                              stall_minutes=REPEAT_STALL_MINUTES))
    if perf_points:  # last: a profiling problem never costs the timing
        jobs.append(RemoteJob("perf", perf_command(plan, perf_points,
                                                   int(timed_cpus or system_cores)),
                              PERF_MINUTES))
    return jobs


def perf_command(plan: dict, names: list[str], cpu: int) -> str:
    """perf (installed for the running kernel) recording each point on the first
    PERF_CLIPS chart clips with call graphs, on the default tier; per recording the
    self-cycles table by symbol (the encoder's own symbols resolve there) and the
    whole recording's cycles, as text. The raw recordings stay on the instance."""
    steps = [f"cd ~/{REMOTE_DIR}",
             "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update",
             "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y "
             "linux-tools-$(uname -r) > /dev/null"]
    for name in names:
        for clip in act.plan_clips(plan)[:PERF_CLIPS]:
            label = f"perf-{name.replace('@', '_')}-{clip}"
            argv = [f"bin/{a.removeprefix('{BIN}/')}" if i == 0 else
                    f"clips/{act.plan_input(plan, clip)}.yuv" if a == "{INPUT}" else
                    "scratch/perf.bin" if a == "{OUTPUT}" else a
                    for i, a in enumerate(plan["points"][name][clip]["argv"])]
            steps += [f"mkdir -p scratch && sudo perf record -q -g -o scratch/{label}.data -- "
                      f"taskset -c {cpu} {shlex.join(argv)}",
                      f"sudo perf report -i scratch/{label}.data --no-children --sort symbol "
                      f"--stdio --percent-limit 0.1 > out/{label}.txt",
                      f"echo {label}: $(grep -m1 'Event count' out/{label}.txt)"]
    return " && ".join(steps)


def geomean(values) -> float:
    return statistics.geometric_mean(list(values))


def summarize(out: Path, points: list[str], repeats: int, plan: dict) -> dict:
    """Per point and repeat: fps (geometric mean over the clips) and the per-round time
    ratios to x264 medium and M13; beside them, the stored chart's x86 ratios."""
    stored_record = json.loads(LOCAL_TIMING_FILE.read_text())
    # a stored record of other clips or another rate mode is no comparison
    comparable = (stored_record.get("rate_mode") == plan["rate_mode"]
                  and stored_record.get("clips") == plan["clips"])
    local = stored_record["points"] if comparable else {}
    anchor = plan["anchor"]
    summary = {}
    for name in points:
        row: dict = {}
        for repeat in range(1, repeats + 1):
            path = out / f"repeat-{repeat}" / f"t-{name}.json"
            if not path.is_file():
                continue
            clips = json.loads(path.read_text())["clips"].values()
            partner = name.removesuffix(BASELINE_SUFFIX) if name.endswith(BASELINE_SUFFIX) \
                else None
            if partner:  # this build's time over the baseline's: < 1 is faster
                row[f"repeat-{repeat}-over-base"] = 1 / geomean(
                    c["time_ratio"][partner] for c in clips)
            row[f"repeat-{repeat}"] = {
                "fps": 1000 / geomean(c["ms_per_frame"]["point"] for c in clips),
                "ratio_x264_medium": geomean(c["time_ratio"][anchor] for c in clips),
                "ratio_m13": geomean(c["time_ratio"][act.SECOND_ANCHOR] for c in clips),
                "stream_mismatch": sorted({r for c in clips for r, s in c["streams"].items()
                                           if "planned_bytes" in s})}
        stored = local.get(name)
        if stored:
            row["local"] = {"ratio_x264_medium": geomean(
                stored["ms_per_frame"][c] / local[anchor]["ms_per_frame"][c]
                for c in act.plan_clips(plan))}
            if "same_session_time_ratio" in stored:
                row["local"]["ratio_m13"] = geomean(
                    stored["same_session_time_ratio"][act.SECOND_ANCHOR].values())
        summary[name] = row
    return summary


def print_summary(summary: dict, repeats: int) -> None:
    print(f"{'point':28s} " + " ".join(f"{'fps r' + str(r):>8s} {'/M13 r' + str(r):>8s}"
                                       for r in range(1, repeats + 1))
          + f" {'/M13 PC':>8s} {'/x264m':>7s} {'PC':>7s}")
    for name, row in summary.items():
        over_base = [f"{row[k]:.4f}" for k in sorted(row) if k.endswith("-over-base")]
        if over_base:
            print(f"{name:28s} new/base time: {' '.join(over_base)}")
        cells = []
        for repeat in range(1, repeats + 1):
            r = row.get(f"repeat-{repeat}")
            cells.append(f"{r['fps']:8.1f} {r['ratio_m13']:8.3f}" if r else f"{'-':>8s} {'-':>8s}")
        last = row.get(f"repeat-{repeats}") or {}
        pc = row.get("local", {})
        print(f"{name:28s} " + " ".join(cells)
              + f" {pc.get('ratio_m13', float('nan')):8.3f}"
              + f" {last.get('ratio_x264_medium', float('nan')):7.3f}"
              + f" {pc.get('ratio_x264_medium', float('nan')):7.3f}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance-type", default=TIMING_MACHINE, choices=sorted(INSTANCES))
    parser.add_argument("--region", help="another region than the instance type's default "
                        "(spot capacity and prices move; stay within check.sh's regions)")
    parser.add_argument("--points", help="comma-separated plan points (default: our chart "
                        "families and " + ", ".join(DEFAULT_REFERENCES) + ")")
    parser.add_argument("--inputs", help="comma-separated clips: only the plan's entries that "
                        "encode them (an every-encode plan split by clip over sessions)")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=act.HOST_ROUNDS)
    parser.add_argument("--binary", type=Path, help="our build for the instance's architecture "
                        "(default: each family's x86 release build from the chart's record)")
    parser.add_argument("--binaries", action="append", default=[], metavar="NAME=PATH",
                        help="several builds of ours staged as bin/NAME, which the plan's "
                             "commands name ({BIN}/NAME): one session times them side by "
                             "side (each checked by its own --version line)")
    parser.add_argument("--tools-dir", type=Path,
                        help=f"reference encoders (default: x86 {X86_TOOLS}, "
                             f"arm64 {GRAVITON_TOOLS})")
    parser.add_argument("--expect-version", help="our binary's `--version` line (default: "
                        "run it here; needed for a binary of another architecture)")
    parser.add_argument("--plan", type=Path, default=PLAN_FILE,
                        help="timing plan (anchor_chart_timing.py plan; default: the chart's)")
    parser.add_argument("--isa-variants", action="store_true",
                        help=f"also time each of our points with {ISA_VARIANT_ENV}")
    parser.add_argument("--checkasm", type=Path,
                        help="an nc-checkasm build: run its benchmarks per x86 tier")
    parser.add_argument("--baseline-binary", type=Path,
                        help="an earlier build of ours (with --binary): each of our points "
                        "is timed against it in the same rounds on the same core")
    parser.add_argument("--repeat-minutes", type=float, default=REPEAT_TIMEOUT_MINUTES,
                        help="a repeat's time limit (an every-encode plan's session times "
                             "more pairs; raise --lifetime with it)")
    parser.add_argument("--perf", help="comma-separated plan points to profile with perf")
    parser.add_argument("--purpose", default="x86-timing", help="the session's Purpose tag")
    parser.add_argument("--budget", type=float, default=5.0, help="session budget, USD")
    parser.add_argument("--lifetime", type=int, default=90, help="hard instance lifetime, min")
    parser.add_argument("--out", type=Path, help="local results directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="stage the upload and print the plan; launch nothing")
    args = parser.parse_args(argv)
    freed = prune_clip_cache()
    if freed:
        say(f"clip cache over its cap: {freed / 1e9:.1f} GB of the least recently used freed")

    region, arch = INSTANCES[args.instance_type]
    region = args.region or region
    if arch == "arm64" and not (args.binary and args.expect_version):
        raise SystemExit("an arm64 instance needs our aarch64 build: --binary and "
                         "--expect-version (the binary's `--version` line)")
    if args.baseline_binary and not args.binary:
        raise SystemExit("--baseline-binary compares with --binary: give both")
    points = args.points.split(",") if args.points else chart_points()
    plan = isolated(cloud_plan, points, arch, args.plan, args.isa_variants,
                    bool(args.baseline_binary), set(args.inputs.split(",")) if args.inputs else None)
    if arch == "amd64":
        points = [*points, AVX2_VARIANT]
    if args.isa_variants:
        points += [f"{p}{ISA_VARIANT_SUFFIX}" for p in points if p.startswith("nanocodec")]
    if args.baseline_binary:  # the variants' partners are the points themselves
        points = [f"{p}{BASELINE_SUFFIX}" if p.startswith("nanocodec") else p
                  for p in points]
    points = slowest_first(points)
    missing = [p for p in points if p not in plan["points"]]
    if missing:
        raise SystemExit(f"not in the plan: {missing}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.out or WORK_DIR / f"timing-{args.instance_type}-{stamp}"
    # sessions run side by side: one staging directory each
    stage_dir = WORK_DIR / f"stage-{args.instance_type}-{stamp}-{os.getpid()}"
    named = dict(pair.split("=", 1) for pair in args.binaries)
    ours = ({name: Path(path) for name, path in named.items()} if named
            else {"nanocodec": args.binary} if args.binary else chart_binaries(plan))
    if args.baseline_binary:
        ours[BASELINE_BINARY] = args.baseline_binary
    # the remote readiness check compares each binary's version stamp with these
    # an explicit empty --expect-version: a build whose --version prints nothing (before 2026-11-11)
    versions = {name: args.expect_version if args.expect_version is not None else subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, check=not named).stdout.strip()
        for name, path in ours.items()}
    isolated(stage, stage_dir, plan, ours,
             args.tools_dir or (GRAVITON_TOOLS if arch == "arm64" else X86_TOOLS))
    if args.checkasm:
        (stage_dir / "bin/nc-checkasm").symlink_to(args.checkasm.resolve())
    try:
        (stage_dir / "out").mkdir()
        jobs = remote_commands(points, args.repeats, args.rounds, versions, plan,
                               bool(args.checkasm),
                               args.perf.split(",") if args.perf else [],
                               SYSTEM_CORES.get(args.instance_type.split(".")[1],
                                                DEFAULT_SYSTEM_CORES), arch,
                               TIMED_CPUS.get(args.instance_type), args.repeat_minutes)
        if args.dry_run:
            subprocess.run(["du", "-shL", str(stage_dir)], check=True)
            for job in jobs:
                print(f"{job.log} ({job.timeout_minutes} min): {job.command}")
            return
        spec = SessionSpec(region=region, instance_type=args.instance_type, arch=arch,
                           purpose=args.purpose,
                           lifetime_minutes=args.lifetime, budget_usd=args.budget,
                           # on demand: a reclaimed spot box loses the timing (four c8a.large
                           # spot sessions were reclaimed within 6 min, 2026-11-15)
                           on_demand=True)
        started = time.monotonic()
        with Session(spec) as session:
            session.upload(stage_dir, f"~/{REMOTE_DIR}")
            try:
                for job in jobs:
                    say(f"remote: {job.log}")
                    session.run_detached(job.command, f"~/{REMOTE_DIR}/out/{job.log}.log",
                                         job.timeout_minutes, stall_minutes=job.stall_minutes,
                                         max_errors=job.max_errors)
            finally:  # whatever finished is fetched before the instance goes
                session.download(f"~/{REMOTE_DIR}/out", out)
        say(f"session wall time {(time.monotonic() - started) / 60:.1f} min")
    finally:
        unstage(stage_dir)
    summary = summarize(out, points, args.repeats, plan)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_summary(summary, args.repeats)
    say(f"results in {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
