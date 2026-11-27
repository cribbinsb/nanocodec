#!/usr/bin/env python3
"""The nightly verifier's Arm part: one Graviton4 spot box runs `nightly_checks.sh arm`.

    python3 tools/dev/verify_arm.py TREE OUT_DIR

TREE is a checkout of the commit to verify (tools/dev/verify_schedule.py's nightly
worktree). Its aarch64 release build of the CLI and nc-checkasm is cross-compiled on the
x86 build box (nc-remote, as aws_cpu_ab builds Graviton binaries), then one c8g spot box
(docs/aws-cost-guide.md: `aws_choose.py build-arm`; every Graviton4 size has SVE2,
dotprod and I8MM) launched through aws_session (budget, hard lifetime, ledger, cleanup on
every exit) from the baked arm64 measurement image (numpy) gets:

  - the two binaries, the tree's tools/, eval/ and presets (git archive, ~25 MB);
  - the pinned aarch64 standard decoders (anchors/aarch64/bin, DECODERS);
  - the first CLIP_FRAMES frames of the landing conformance clips (zstd), deleted with
    the box;

and runs checkasm (NEON with its dotprod and I8MM paths, SVE2; a missing tier fails), the
landing conformance streams coded there, and ISA identity (scalar = NEON = SVE2). The
summary and per-check logs come back into OUT_DIR (nightly_checks.sh's layout). Exit
status: 0 all passed, 1 a check failed, 2 the run could not complete (build, launch,
reclaim): the scheduler records that as an infrastructure failure and the next night
reruns the whole part (unit: one ~30-minute run; no partial resume).
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_image  # noqa: E402
from aws_cpu_ab import CROSS_LINKER_ENV, CROSS_PACKAGES, RUST_TRIPLES  # noqa: E402
from aws_session import RemoteFailure, Session, SessionSpec, say  # noqa: E402

DATA = Path.home() / "nanocodec-data"
NC_REMOTE = DATA / "bin/nc-remote"
#: aws_choose.py build-arm's pick, then the next size (no capacity: same work, ~same cost).
BOX_TYPES = ("c8g.8xlarge", "c8g.16xlarge")
REGION = "eu-north-1"
#: The run takes ~30 min; the hard lifetime and the budget bound a stuck box.
LIFETIME_MINUTES = 90
BUDGET_USD = 0.75
RUN_TIMEOUT_MINUTES = 75
#: nightly_checks.sh prints only a line per check, and the landing check alone takes
#: tens of minutes.
STALL_MINUTES = 60
ROOT_GB = 24
#: The standard decoders landing_conformance.py runs (nceval.conformance).
DECODERS = ("ldecod", "TAppDecoder", "aomdec", "dav1d", "ffmpeg-decode", "vpxdec",
            "vvdecapp", "avmdec", "dav2d")
#: Source frames kept of each landing clip: the longest stepped shape reads 41 frames
#: every second frame (landing_conformance.py SWITCH, SOURCE_STEP 2), so 82 suffice.
CLIP_FRAMES = 90
#: The landing clips' frame sizes (landing_conformance.py shapes' source_size).
CLIP_SIZES = {"crowd_mot20_06": (1884, 720)}
DEFAULT_CLIP_SIZE = (1280, 720)
#: Cut clips are kept here between nights (regenerated when missing).
CLIP_CACHE = DATA / "scratch/verify-arm/clips"
#: What the tree contributes (the checks need no Rust sources on the box).
TREE_PATHS = ("tools", "eval", "crates/nc-encoder/presets")
BOX_ROOT = "nightly"


def cross_build(tree: Path, stage: Path) -> None:
    """The aarch64 release CLI and nc-checkasm, built on the x86 build box, into stage/bin."""
    triple = RUST_TRIPLES["aarch64"]
    cargo = f"cargo build -q --release --target {triple} -p nanocodec-cli -p nc-checkasm"
    command = ("set -e; dpkg -s gcc-aarch64-linux-gnu >/dev/null 2>&1 || sudo DEBIAN_FRONTEND="
               f"noninteractive apt-get -qq install -y --no-install-recommends {CROSS_PACKAGES} "
               f">/dev/null; {CROSS_LINKER_ENV} {cargo}")
    fetched = stage / "build"
    subprocess.run([str(NC_REMOTE), "run", "--worktree", str(tree), "--name", "verifier-arm",
                    "--fetch", f"target/{triple}/release/nanocodec",
                    "--fetch", f"target/{triple}/release/nc-checkasm",
                    "--fetch-to", str(fetched), "--", command], check=True)
    (stage / "bin").mkdir(parents=True)
    for name in ("nanocodec", "nc-checkasm"):
        shutil.copy2(fetched / f"target/{triple}/release/{name}", stage / "bin" / name)


def landing_clips(tree: Path) -> list[Path]:
    listed = subprocess.run([sys.executable, "tools/ci/landing_conformance.py", "--list-clips"],
                            cwd=tree, capture_output=True, text=True, check=True).stdout
    return [Path(line) for line in listed.split()]


def cut_clips(clips: list[Path], into: Path) -> None:
    """The first CLIP_FRAMES frames of each clip, zstd-compressed, into `into` (cached)."""
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    into.mkdir(parents=True)
    for clip in clips:
        width, height = CLIP_SIZES.get(clip.stem, DEFAULT_CLIP_SIZE)
        cached = CLIP_CACHE / f"{clip.stem}-{CLIP_FRAMES}f.yuv.zst"
        if not cached.is_file():
            partial = cached.with_suffix(".partial")
            with clip.open("rb") as source:
                data = source.read(width * height * 3 // 2 * CLIP_FRAMES)
            subprocess.run(["zstd", "-q", "-3", "-f", "-o", str(partial)], input=data, check=True)
            partial.replace(cached)
        (into / cached.name).symlink_to(cached)


def stage_inputs(tree: Path, stage: Path) -> None:
    """Everything the box needs, laid out as under its home (`tar -h` follows the links)."""
    root = stage / "home"
    src = root / BOX_ROOT / "src"
    src.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(tree), "archive", "HEAD", *TREE_PATHS],
                             capture_output=True, check=True).stdout
    subprocess.run(["tar", "-C", str(src), "-xf", "-"], input=archive, check=True)
    shutil.move(stage / "bin", root / BOX_ROOT / "bin")
    decoders = root / "nanocodec-data/anchors/aarch64/bin"
    decoders.mkdir(parents=True)
    for name in DECODERS:
        (decoders / name).symlink_to(DATA / "anchors/aarch64/bin" / name)
    cut_clips(landing_clips(tree), root / "nanocodec-data/clips/zst")


def run_on_box(stage: Path, out: Path) -> int:
    spec = SessionSpec(region=REGION, instance_type=BOX_TYPES[0], arch="arm64",
                       purpose="nightly-verify-arm", lifetime_minutes=LIFETIME_MINUTES,
                       budget_usd=BUDGET_USD, root_gb=ROOT_GB, fallback_types=BOX_TYPES[1:],
                       image_tags=measure_image.image_tags("arm64"), launch_wait_minutes=20)
    unpack = ("(command -v zstd && command -v file) >/dev/null || sudo DEBIAN_FRONTEND="
              "noninteractive apt-get -qq install -y zstd file >/dev/null; "
              "mkdir -p ~/nanocodec-data/clips/main && for f in ~/nanocodec-data/clips/zst/*.zst; "
              "do n=$(basename \"$f\" .zst); zstd -q -d -f \"$f\" -o "
              "~/nanocodec-data/clips/main/${n%-*}.yuv; done")
    checks = (f"cd ~/{BOX_ROOT}/src && NC_ARM_BIN=$HOME/{BOX_ROOT}/bin OUT=$HOME/{BOX_ROOT}/out "
              "bash tools/ci/nightly_checks.sh arm")
    status = 0
    with Session(spec) as session:
        session.upload(stage / "home", "~")
        try:
            session.run_detached(f"{measure_image.install_unless_baked()} && {unpack} && {checks}",
                                 "nightly-arm.log", RUN_TIMEOUT_MINUTES,
                                 stall_minutes=STALL_MINUTES)
        except RemoteFailure as failure:
            say(f"arm checks: {failure}")
            status = 1
        session.download(f"~/{BOX_ROOT}/out", out)
    if not (out / "summary").is_file():
        return 2
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tree", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="nanocodec-verify-arm-") as scratch:
        stage = Path(scratch)
        try:
            cross_build(a.tree.resolve(), stage)
            stage_inputs(a.tree.resolve(), stage)
            return run_on_box(stage, a.out)
        except (subprocess.CalledProcessError, RuntimeError, OSError) as error:
            say(f"arm verify could not complete: {error}")
            return 2


if __name__ == "__main__":
    sys.exit(main())
