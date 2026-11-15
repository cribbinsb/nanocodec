"""Jetson CPU cost of encoder settings: one build per commit, hardware counters per setting.

The tuner's finalists (nceval.tune_pool) are timed on the Jetson, the
primary CPU platform (weighted 2:1 over x86). Settings differ only in
`--param` values, so one release build of the study's commit serves them
all. Builds share chart refresh's layout (tools/dev/chart_refresh.py:
BUILDS_ROOT/<commit>/nanocodec, finished when the COMMIT marker holds the
commit; the persistent source tree and target dir under their lock), so a
commit built for either is built once.

Each setting runs as a tiny wrapper script (`exec nanocodec "$@" --params
FILE`) so tools/dev/jperf.py can interleave them; jperf counts user-space
instructions and cycles (min over rounds) with the board's timing lock held
(jetson-timed.sh). Inputs are the x86 cost clips (nceval.tune.COST_CLIPS),
150-frame native copies already on the board.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from statistics import geometric_mean

from .codecs import Codec
from .localenv import require

REPO = Path(__file__).resolve().parents[2]
BUILDS_ROOT = "nanocodec-data/charts/builds"
COMPLETE_MARKER = "COMMIT"
SHORT_COMMIT = 8
WORK = "nanocodec-data/work/chart-refresh"
SOURCE = f"{WORK}/src"
TARGET = "nanocodec-data/targets/chart-build"
BUILD_JOBS = 6
SCRATCH = "nanocodec-data/work/tune-jetson"
#: What the build reads: the workspace manifests, the crates (with the test data
#: they include) and the embedded presets. The whole repository (docs, web, eval
#: data) was 231 MB an archive against 15 MB, sent, unpacked and checksummed on
#: the board for every build.
BUILD_INPUTS = ("Cargo.toml", "Cargo.lock", "rustfmt.toml", "crates", "presets")
#: 720p, 150-frame copies of nceval.tune.COST_CLIPS (tools/dev/offpath_ab.sh uses them too).
CLIP_DIR = "nanocodec-data/work/adopted-fast-reference-native"
CLIPS = {"crowd_mot17_09": 30.0, "bodycam_bc1": 29.97, "static_cam22": 25.0}
WIDTH, HEIGHT = 1280, 720
FRAMES = 31
ROUNDS = 3
INTRA_PERIOD = 300
LINE = re.compile(r"^\s+(\S+)\s+instr\s+([\d.]+)G.*cycles\s+([\d.]+)G")


def ssh(command: str, capture: bool = True) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", require("NC_JETSON"), command],
                          check=True, capture_output=capture, text=True,
                          stdin=subprocess.DEVNULL).stdout or ""


def build_command(commit: str) -> str:
    """The shell pipeline that builds `commit` (8-character abbreviation) on the board.

    `git archive` of BUILD_INPUTS is synced into the board's persistent source tree
    (rsync without -t: only the files that changed get new times, so cargo rebuilds
    only what they touch; an archive's own mtimes are the commit time, which let
    cargo keep a stale binary) and built under the shared build lock and the
    tree's own lock. The tree has no git, so NC_BUILD_COMMIT stamps the binary,
    and no marker is written unless `--version` names the commit. One command
    for the tuner's builds and chart refresh's (tools/dev/chart_refresh.py)."""
    full = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{commit}^{{commit}}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    stage = f"{WORK}/stage-{commit}"
    build = f"{BUILDS_ROOT}/{commit}"
    built = f"~/{TARGET}/release/nanocodec"
    stamp_check = (f"{{ {built} --version | grep -Eq '^nanocodec [^ ]+ {full}$' || "
                   f"{{ echo \"Jetson build is not {full}: $({built} --version)\" >&2; "
                   f"false; }}; }}")
    inner = (f"rsync -rc --delete {stage}/ {SOURCE}/ && cd {SOURCE} && "
             f"nice -n 19 env PATH=$HOME/.cargo/bin:$PATH NC_BUILD_COMMIT={full} "
             f"cargo build --release -j{BUILD_JOBS} -p nc-cli --target-dir ~/{TARGET} && "
             f"{stamp_check} && mkdir -p ~/{build} && cp {built} ~/{build}/ && "
             f"echo {commit} > ~/{build}/{COMPLETE_MARKER}")
    remote = (f"rm -rf {stage} && mkdir -p {stage} {SOURCE} && tar -x -C {stage} && "
              f"~/nanocodec-data/jetson-build.sh flock {SOURCE}.lock "
              f"bash -c {shlex.quote(inner)}; status=$?; rm -rf {stage}; exit $status")
    return (f"git -C {shlex.quote(str(REPO))} archive {commit} {' '.join(BUILD_INPUTS)} | "
            f"ssh -o BatchMode=yes {require('NC_JETSON')} {shlex.quote(remote)}")


def ensure_build(commit: str) -> str:
    """The board's release binary of `commit` (built now unless it exists), named by
    its 8-character abbreviation as chart refresh names its builds."""
    commit = commit[:SHORT_COMMIT]
    build = f"{BUILDS_ROOT}/{commit}"
    built = ssh(f"cd ~/{BUILDS_ROOT} 2>/dev/null && cat */{COMPLETE_MARKER} || true").split()
    if commit in built:
        return f"~/{build}/nanocodec"
    donor = next((other for other in built if same_build_inputs(other, commit)), None)
    if donor is not None:  # nothing it reads changed: its binary, without a build
        ssh(f"mkdir -p ~/{build} && cp ~/{BUILDS_ROOT}/{donor}/nanocodec ~/{build}/ && "
            f"echo {commit} > ~/{build}/{COMPLETE_MARKER}")
    else:
        subprocess.run(["bash", "-o", "pipefail", "-c", build_command(commit)], check=True,
                       capture_output=True)
    return f"~/{build}/nanocodec"


def same_build_inputs(a: str, b: str) -> bool:
    """Whether commits `a` and `b` have identical BUILD_INPUTS (a commit unknown here: no)."""
    return subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", a, b, "--", *BUILD_INPUTS],
                          capture_output=True).returncode == 0


def encode_args(codec: Codec, clip: str, quantizer: int) -> str:
    return (f"encode --codec {codec.value} --input ~/{CLIP_DIR}/{clip}-150f.yuv "
            f"--width {WIDTH} --height {HEIGHT} --fps {CLIPS[clip]:g} --frames {FRAMES} "
            f"--intra-period {INTRA_PERIOD} --qp {quantizer} --output /dev/null")


def cost(commit: str, codec: Codec, quantizer: int,
         settings: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """{setting: {"instructions", "cycles"}} as ratios to the first setting
    (geometric mean over CLIPS); setting names must be plain words."""
    binary = ensure_build(commit)
    with tempfile.TemporaryDirectory() as local:
        for name, values in settings.items():
            Path(local, f"{name}.params").write_text(
                "".join(f"{k} = {v!r}\n" for k, v in sorted(values.items())))
            Path(local, name).write_text(
                f'#!/bin/sh\nexec {binary} "$@" --params ~/{SCRATCH}/{name}.params\n')
            Path(local, name).chmod(0o755)
        Path(local, "jperf.py").write_bytes((REPO / "tools/dev/jperf.py").read_bytes())
        ssh(f"rm -rf ~/{SCRATCH} && mkdir -p ~/{SCRATCH}")
        subprocess.run(["scp", "-q", "-p", *sorted(str(p) for p in Path(local).iterdir()),
                        f"{require('NC_JETSON')}:{SCRATCH}/"], check=True)
    wrappers = " ".join(f"~/{SCRATCH}/{name}" for name in settings)
    runs = "; ".join(f"echo '== {clip}'; python3 ~/{SCRATCH}/jperf.py --rounds {ROUNDS} "
                     f"{wrappers} -- {encode_args(codec, clip, quantizer)}" for clip in CLIPS)
    try:
        output = ssh(f"~/nanocodec-data/jetson-timed.sh sh -c {shlex.quote(runs)} "
                     f"> ~/{SCRATCH}/out.txt 2>&1 < /dev/null; cat ~/{SCRATCH}/out.txt")
    finally:
        ssh(f"rm -rf ~/{SCRATCH}")
    per_clip: dict[str, list[tuple[float, float]]] = {name: [] for name in settings}
    for line in output.splitlines():
        if match := LINE.match(line):
            per_clip[match[1]].append((float(match[2]), float(match[3])))
    first = next(iter(settings))
    if any(len(v) != len(CLIPS) for v in per_clip.values()):
        raise RuntimeError(f"jperf output incomplete:\n{output[-800:]}")
    return {name: {"instructions": geometric_mean(
                       [a[0] / b[0] for a, b in zip(values, per_clip[first])]),
                   "cycles": geometric_mean(
                       [a[1] / b[1] for a, b in zip(values, per_clip[first])])}
            for name, values in per_clip.items()}
