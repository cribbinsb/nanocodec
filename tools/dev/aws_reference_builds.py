#!/usr/bin/env python3
"""Build the timed reference encoders on a rented box and A/B them against the old ones.

    python3 tools/dev/aws_reference_builds.py graviton [--instance-type c8g.4xlarge]
        [--install anchors/NAME] [--ours COMMIT] [--no-ab] [--jobs N]
    python3 tools/dev/aws_reference_builds.py ab --new DIR --old DIR
        [--instance-type c8a.large]
    common: [--clip handheld_mot17_10] [--kbps 600] [--frames 60] [--repeats 3]
        [--only F,...] [--variant FAMILY+SUFFIX=OPTIONS] [--budget 8] [--dry-run]

`graviton` builds tools/anchors/build_anchors.sh --opt natively on a Graviton box (the
x86_64 build runs on the build box through nc-remote): x265 with its SVE/SVE2 kernels,
every encoder with its own LTO and PGO (docs/PROCEDURES.md "Reference builds"). It
installs the result in nceval.config.REFERENCE_BUILDS["graviton"] and then, on the same
box, runs tools/anchors/reference_ab.py against the old arm64 references (the Jetson's
builds, reference_bin("jetson")); --install names another kept prefix (a benchmark's own
store), --no-ab leaves the A/B out. --ours COMMIT also builds our PGO release binary of
COMMIT on the same box, beside the references (tools/dev/pgo_build.sh, the same training
clips), installed as PREFIX/nanocodec with BUILD.json. Every installed encoder then codes
5 frames with each mode's record command (pgo_train.py --smoke), and PREFIX/SHA256SUMS
lists every binary. `ab` runs only reference_ab.py (default: a Zen 5
timing box, the x86_64 build). Both A/Bs use one chart timing clip (aws_timing's cached cut) and the
timing plan's own command lines (docs/results/bd-vs-speed-plan.json); results land in
--out (default WORK_DIR/refbuild-<kind>-<stamp>). Instances, budget and clean-up:
tools/dev/aws_session.py.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))
import aws_timing  # noqa: E402
from aws_session import WORK_DIR, Session, SessionSpec, say  # noqa: E402
from nceval.config import REFERENCE_BUILDS, reference_bin  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / "nanocodec-data"
PGO_CLIPS_ZST = DATA / "release/pgo-clips-2026-11-23-zst"
#: Build packages on the stock Ubuntu image (nasm only matters on x86).
PACKAGES = "build-essential cmake nasm git zstd python3"
REMOTE = "refbuild"
BUILD_MINUTES = 75
AB_MINUTES = 30
#: Parallel compile jobs on the build box size (c8g.4xlarge: 16 vCPUs).
BUILD_JOBS = 16
#: The Rust toolchain our build uses on the box (the PC's and the build box's,
#: docs/PROCEDURES.md "Remote build box"), with llvm-profdata for PGO.
RUST_TOOLCHAIN = "1.99.1"
#: Our build's inputs at its commit (release.py JETSON_SOURCES, the subset an aarch64
#: PGO build reads).
OURS_SOURCES = ("crates", "Cargo.toml", "Cargo.lock", ".cargo", "eval/nceval",
                "tools/dev/pgo_build.sh")
#: pgo_build.sh's training length per clip (release.py PGO_FRAMES).
OURS_PGO_FRAMES = 20
#: Frames each smoke encode codes.
SMOKE_FRAMES = 5
#: The scoring decoders' smoke test: each decodes a smoke stream of its codec
#: (pgo_train.py names them FAMILY-MODE-CLIP.bin under its --work; AV2 is ours).
SMOKE_DECODES = {"aomdec": "aomenc/libaom-av1-cpu8-qp-0.bin",
                 "vpxdec": "vpxenc/libvpx-vp9-cpu7-qp-0.bin",
                 "avmdec": "nanocodec/nanocodec-av2@faster-qp-0.bin"}
#: The A/B's timed core: the second (aws_timing's one-system-core rule on a .large).
TIMED_CPU = "1"
#: The references timed before the opt builds, per AMI architecture: the plain x86_64
#: build and the Jetson's arm64 build.
PLAIN_TOOLS = {"amd64": DATA / "anchors/x86_64/bin", "arm64": reference_bin("jetson")}
#: The encoders the opt build installs (build_anchors.sh tool names, in build order).
TOOLS = ("openh264", "harness", "x264", "x265", "kvazaar", "svtav1", "libaom", "libvpx", "vvenc",
         "avm")
#: The record's tool names (pgo_train.py --tool) of the encoders TOOLS installs (libaom:
#: aomenc, the main rows; svc_encoder_rtc runs only the +drop rows, which are not trained).
RECORD_TOOLS = ("openh264-harness", "x264", "x265", "kvazaar", "SvtAv1EncApp", "aomenc",
                "vpxenc", "vvencFFapp")


def stage(stage_dir: Path, clip: str | None, old: Path, new: Path | None, build: bool,
          ours: str | None) -> None:
    """Everything the box needs, as symlinks (Session.upload follows them); with `ours`,
    our sources at that commit as a tar. Without `clip` (no A/B) no timing clip is staged."""
    stage_dir.mkdir(parents=True)
    if clip is not None:
        stage_clip(stage_dir, clip, old, new)
    stage_build(stage_dir, build, ours)


def stage_clip(stage_dir: Path, clip: str, old: Path, new: Path | None) -> None:
    """The A/B's inputs: the timing clip, the plan, the A/B script and both builds."""
    zst = aws_timing.CLIP_CACHE / f"{clip}-s0-150f.yuv.zst"
    if not zst.is_file():
        raise SystemExit(f"no cached timing cut {zst} (aws_timing caches it)")
    (stage_dir / "clip.yuv.zst").symlink_to(zst)
    from nceval.chart_timing import load_plan
    (stage_dir / "plan.json").write_text(json.dumps(load_plan(aws_timing.PLAN_FILE)) + "\n")
    (stage_dir / "reference_ab.py").symlink_to(REPO / "tools/anchors/reference_ab.py")
    (stage_dir / "old").symlink_to(old)
    if new:
        (stage_dir / "new").symlink_to(new)


def stage_build(stage_dir: Path, build: bool, ours: str | None) -> None:
    """The build's inputs: the anchors' scripts, the PGO clips, the record, our sources."""
    if build:
        (stage_dir / "anchors").symlink_to(REPO / "tools/anchors")
        (stage_dir / "openh264-harness").symlink_to(REPO / "tools/openh264-harness")
        (stage_dir / "pgo-clips-zst").symlink_to(PGO_CLIPS_ZST)
        # pgo_train.py takes the record's commands from nceval.fairness
        (stage_dir / "nceval").symlink_to(REPO / "eval/nceval")
        (stage_dir / "benchmark-fairness.toml").symlink_to(REPO / "docs/benchmark-fairness.toml")
    if ours:
        subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar",
                        "-o", str(stage_dir / "ours-src.tar"), ours, *OURS_SOURCES], check=True)


def ours_command(commit: str, jobs: int) -> str:
    """Our PGO release build of `commit` (pgo_build.sh, the references' training clips) into
    out/install/nanocodec, with the toolchain's version beside it."""
    clips = " ".join(f"$HOME/{REMOTE}/pgo-clips/{c.name.removesuffix('.zst')}"
                     for c in sorted(PGO_CLIPS_ZST.glob("*.yuv.zst")))
    return (f"curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal "
            f"--default-toolchain {RUST_TOOLCHAIN} -c llvm-tools && "
            f"mkdir -p ours out/install && tar -C ours -xf ours-src.tar && cd ours && "
            f"~/.cargo/bin/rustc --version > ~/{REMOTE}/out/install/nanocodec.rustc && "
            f"env JOBS={jobs} NC_BUILD_COMMIT={commit} PGO_FRAMES={OURS_PGO_FRAMES} "
            f"PGO_CLIPS=\"{clips}\" bash tools/dev/pgo_build.sh . ~/{REMOTE}/ours-target "
            f"~/{REMOTE}/out/install/nanocodec")


def smoke_command(prefix: str, ours: bool) -> str:
    """Each installed encoder codes SMOKE_FRAMES frames of one training clip with each mode's
    record command (pgo_train.py --smoke; ours: one family per codec), and each scoring
    decoder decodes one of those streams (SMOKE_DECODES)."""
    tools = [*RECORD_TOOLS, *(["nanocodec"] if ours else [])]
    steps = [f"python3 -I tools/anchors/pgo_train.py --smoke --tool {tool} "
             f"--bin {'out/install' if tool == 'nanocodec' else prefix + '/bin'} "
             f"--cfg {prefix}/share/vvenc/cfg --work smoke/{tool} --frames {SMOKE_FRAMES} "
             f"pgo-clips/*.yuv && echo smoke ok: {tool}" for tool in tools]
    steps += [f"{prefix}/bin/{decoder} --md5 smoke/{stream} && echo smoke ok: {decoder}"
              for decoder, stream in SMOKE_DECODES.items() if ours or decoder != "avmdec"]
    if ours:
        steps.append("out/install/nanocodec --version > out/install/nanocodec.version")
    return f"set -eo pipefail; cd ~/{REMOTE}; " + " && ".join(steps)


def build_command(prefix: str, jobs: int, ours: str | None) -> str:
    """Unpack the PGO clips, then build_anchors.sh --opt into `prefix` on the box; with
    `ours`, our build runs beside it (ours.log) on a third of the jobs."""
    return (f"set -eo pipefail; cd ~/{REMOTE}; "
            f"sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update && "
            f"sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y {PACKAGES} >/dev/null; "
            f"mkdir -p tools pgo-clips eval docs && rm -rf tools/anchors tools/openh264-harness "
            f"eval/nceval && cp -rL anchors tools/anchors && "
            f"cp -rL openh264-harness tools/openh264-harness && cp -rL nceval eval/nceval && "
            f"cp benchmark-fairness.toml docs/; "
            f"for c in pgo-clips-zst/*.yuv.zst; do "
            f"zstd -d -q -f $c -o pgo-clips/$(basename $c .zst); done; "
            f"gcc --version | head -1; "
            + (f"({ours_command(ours, max(1, jobs // 3))}) > ours.log 2>&1 & ours=$!; "
               if ours else "")
            + f"bash tools/anchors/build_anchors.sh --opt --jobs {jobs - jobs // 3 if ours else jobs} "
            f"--prefix {prefix} --pgo-clips ~/{REMOTE}/pgo-clips {' '.join(TOOLS)}; "
            f"mkdir -p out/install && cp -r {prefix}/bin {prefix}/share {prefix}/build-flags.json "
            f"out/install/ && cp -r {prefix}/build/logs out/install/logs"
            + ("; wait $ours || { tail -n 30 ours.log; exit 1; }" if ours else ""))


def ab_command(clip: str, kbps: int, new: str, frames: int, repeats: int,
               extra: str = "") -> str:
    return (f"set -eo pipefail; cd ~/{REMOTE}; "
            f"command -v zstd >/dev/null || {{ sudo DEBIAN_FRONTEND=noninteractive apt-get -qq "
            f"update && sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y zstd "
            f">/dev/null; }}; zstd -d -q -f clip.yuv.zst -o clip.yuv; "
            f"grep -m1 -E '^(flags|Features)' /proc/cpuinfo; "
            f"mkdir -p out; "
            f"python3 -I reference_ab.py --plan plan.json --clip {clip}@{kbps} --yuv clip.yuv "
            f"--old ~/{REMOTE}/old --new {new} --frames {frames} --repeats {repeats} "
            f"--cpu {TIMED_CPU} --out out/ab.json {extra}")


def ab_options(args: argparse.Namespace) -> str:
    """reference_ab.py's selection options, shell-quoted."""
    words = [*(["--only", args.only] if args.only else []),
             *(w for v in args.variant for w in ("--variant", v))]
    return shlex.join(words)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("graviton", "ab"))
    parser.add_argument("--instance-type")
    parser.add_argument("--old", type=Path, help="the old reference binaries (default PLAIN_TOOLS)")
    parser.add_argument("--new", type=Path, help="ab: the new reference binaries")
    parser.add_argument("--clip", default="handheld_mot17_10")
    parser.add_argument("--kbps", type=int, default=600, help="the plan's rung to code")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--region", help="default: aws_timing.INSTANCES' region, else eu-west-1")
    parser.add_argument("--budget", type=float, default=8.0)
    parser.add_argument("--lifetime", type=int, default=120)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", help="reference_ab.py --only")
    parser.add_argument("--variant", action="append", default=[], help="reference_ab.py --variant")
    parser.add_argument("--install", type=Path,
                        help="graviton: the kept prefix, relative to ~/nanocodec-data "
                             "(default nceval.config.REFERENCE_BUILDS['graviton'])")
    parser.add_argument("--ours", help="graviton: also our PGO build of this commit")
    parser.add_argument("--no-ab", action="store_true", help="graviton: build only")
    parser.add_argument("--jobs", type=int, default=BUILD_JOBS, help="compile jobs on the box")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    build = args.kind == "graviton"
    instance = args.instance_type or ("c8g.4xlarge" if build else aws_timing.TIMING_MACHINE)
    arch = "arm64" if instance.startswith(("c8g", "c7g", "m8g", "c6g")) else "amd64"
    old = args.old or PLAIN_TOOLS[arch]
    if not build and not args.new:
        raise SystemExit("ab needs --new")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.out or WORK_DIR / f"refbuild-{args.kind}-{stamp}"
    stage_dir = WORK_DIR / f"stage-refbuild-{stamp}-{os.getpid()}"
    ours = None
    if args.ours:
        ours = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify",
                               f"{args.ours}^{{commit}}"], check=True, capture_output=True,
                              text=True).stdout.strip()
    stage(stage_dir, None if args.no_ab else args.clip, old, None if build else args.new, build, ours)
    prefix = f"~/{REMOTE}/prefix"
    jobs = ([("build", build_command(prefix, args.jobs, ours), BUILD_MINUTES),
             ("smoke", smoke_command(prefix, bool(ours)), AB_MINUTES)] if build else [])
    if not args.no_ab:
        jobs.append(("ab", ab_command(args.clip, args.kbps,
                                      f"{prefix}/bin" if build else f"~/{REMOTE}/new",
                                      args.frames, args.repeats, ab_options(args)), AB_MINUTES))
    try:
        if args.dry_run:
            for name, command, minutes in jobs:
                print(f"{name} ({minutes} min): {command}")
            return
        region = args.region or aws_timing.INSTANCES.get(instance, ("eu-west-1", arch))[0]
        spec = SessionSpec(region=region, instance_type=instance, arch=arch,
                           purpose=f"reference builds {args.kind}", root_gb=30,
                           lifetime_minutes=args.lifetime, budget_usd=args.budget,
                           on_demand=not build)
        with Session(spec) as session:
            session.upload(stage_dir, f"~/{REMOTE}")
            try:
                for name, command, minutes in jobs:
                    say(f"remote: {name}")
                    session.run_detached(command, f"~/{REMOTE}/{name}.log", minutes,
                                         stall_minutes=minutes)
            finally:
                session.ssh(f"mkdir -p ~/{REMOTE}/out && cp ~/{REMOTE}/*.log ~/{REMOTE}/out/",
                            check=False)
                session.download(f"~/{REMOTE}/out", out)
    finally:
        shutil.rmtree(stage_dir)
    # a failed build or smoke test raised above: nothing untested is installed
    if build and (out / "install/bin").is_dir():
        install(out / "install", DATA / (args.install or REFERENCE_BUILDS["graviton"]), ours)
    say(f"results in {out}")
    if not args.no_ab:
        subprocess.run(["cat", str(out / "ab.log")], check=False)


def install(built: Path, prefix: Path, ours: str | None) -> None:
    """The downloaded build into its kept prefix (files replaced, nothing removed), our
    binary and its BUILD.json with `ours`, then SHA256SUMS of every binary there."""
    for name in ("bin", "share"):
        shutil.copytree(built / name, prefix / name, dirs_exist_ok=True)
    shutil.copytree(built / "logs", prefix / "build/logs", dirs_exist_ok=True)
    shutil.copy2(built / "build-flags.json", prefix / "build-flags.json")
    if ours:
        shutil.copy2(built / "nanocodec", prefix / "nanocodec")
        record = {"nanocodec": {
            "commit": ours,
            "version": (built / "nanocodec.version").read_text().strip(),
            "sha256": sha256(prefix / "nanocodec"),
            "rustc": (built / "nanocodec.rustc").read_text().strip(),
            "build": "tools/dev/pgo_build.sh (release fat LTO, -Cprofile-generate / "
                     "-Cprofile-use, no target-cpu) natively on a Graviton box "
                     "(tools/dev/aws_reference_builds.py graviton --ours)",
            "training": f"PGO_FRAMES={OURS_PGO_FRAMES} PGO_CLIPS = the four cuts of "
                        "eval/clipset/pgo_training_2026-11-23.json (release/pgo-clips-2026-11-23); "
                        "every shipped preset of every codec, CBR 600 kbps (pgo_build.sh)"}}
        (prefix / "BUILD.json").write_text(json.dumps(record, indent=1) + "\n")
    binaries = sorted(p for p in [prefix / "nanocodec", *(prefix / "bin").iterdir()]
                      if p.is_file() and os.access(p, os.X_OK))
    (prefix / "SHA256SUMS").write_text("".join(
        f"{sha256(p)}  {p.relative_to(prefix)}\n" for p in binaries))
    say(f"installed in {prefix}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main(sys.argv[1:])
