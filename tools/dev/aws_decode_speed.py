#!/usr/bin/env python3
"""The results site's decoder and browser speed tables, timed on one quiet rented x86 core.

    python3 tools/dev/aws_decode_speed.py --pgo BIN --plain BIN --pkg PKG \\
        --streams DIR [DIR ...] --ffmpeg-core DIR --out DIR [--compare-pkg NAME=PKG ...] \\
        [--decoder-rounds 7] [--yardstick-rounds 7] [--table-rounds 3] [--dry-run]

One on-demand c8a.large (aws_timing.TIMING_MACHINE: AMD EPYC 9R45, Zen 5; one vCPU per
physical core), launched through tools/dev/aws_session.py (budget, hard lifetime, ledger,
cleanup on every exit). Every timed process is pinned to CPU 1 and the jobs run one after
another, so each table's ours and references share one quiet session:

  prepare      node (the official v22.21.1 build), readiness: every binary loads, version
               stamps, the CPU model, its SIMD flags and the pinning;
  pipe-clean   decoder_speed.py one round on two streams;
  decoders     tools/dev/decoder_speed.py (our --pgo build against FFmpeg's decoders and
               dav1d through libavcodec, both timed in process), --decoder-rounds rounds;
  yardstick    tools/web/ffmpeg_wasm_yardstick.py (our browser module against
               ffmpeg.wasm), --yardstick-rounds rounds; then, with --compare-pkg, the
               same module encodes of each compared module interleaved with --pkg's;
  table        tools/web/bench.py, the browser table's documented runs (decode; encode
               at 360p; encode at 720p), the native column on the plain build,
               --table-rounds rounds.

The clips bench.py and the yardstick read go up as 150-frame zstd cuts; the outputs come
back into --out (decoders.json, yardstick.json, compare-NAME.json, table-*.json/csv and the
platform notes).
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aws_session import Session, SessionSpec, say  # noqa: E402
from aws_timing import (INSTANCES, TIMING_MACHINE, isa_command,  # noqa: E402
                        pinning_command)

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / "nanocodec-data"
ANCHORS = DATA / "anchors/x86_64/bin"
#: The x86 anchors the jobs run: decoder_speed.py's libavcodec-bench, bench.py's decoders.
DECODERS = ("libavcodec-bench", "vpx-bench", "vvdec-bench", "avm-bench", "dav2d-bench",
            "ffmpeg-decode", "dav1d", "vpxdec", "vvdecapp", "avmdec")
#: bench.py finds dav1d on PATH; its `ffmpeg ... -f null -` goes to the pinned decode-only
#: FFmpeg (no null muxer) through FFMPEG_WRAPPER, writing rawvideo to /dev/null instead.
PATH_NAMES = {"dav1d": "dav1d"}
FFMPEG_WRAPPER = """#!/bin/bash
args=("$@")
if [ "${args[-3]}" = -f ] && [ "${args[-2]}" = null ] && [ "${args[-1]}" = - ]; then
  args=("${args[@]:0:${#args[@]}-3}" -f rawvideo -y /dev/null)
fi
exec "$(dirname "$0")/../bin/ffmpeg-decode" "${args[@]}"
"""
#: The repository files the browser runs need, at their repository paths.
REPO_FILES = ("tools/dev/decoder_speed.py", "tools/web/bench.py", "tools/web/bench-runner.mjs",
              "tools/web/packets.mjs", "tools/web/ffmpeg_wasm_yardstick.py",
              "tools/web/ffmpeg-wasm-runner.cjs", "web/codecs.js")
#: bench.py's clip sets and clips (bench.RESOLUTIONS), and the yardstick's clip.
CLIP_SETS = {"quick": ("crowd_mot17_09", "pan_jaad_0117", "static_chirla_s000c3"),
             "quick360": ("crowd_mot17_09_360p", "pan_jaad_0117_360p", "static_chirla_s000c3_360p")}
YARDSTICK_CLIP = ("quick", "crowd_mot17_09")
FRAMES = 150
NODE_VERSION = "v22.21.1"
NODE_URL = f"https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.xz"
CPU = 1
REMOTE_DIR = "decode-speed"
#: A job's own timeout (minutes); the whole session's lifetime is --lifetime.
JOB_MINUTES = {"prepare": 10, "pipe-clean": 5, "decoders": 20, "yardstick": 20, "compare": 10,
               "table-decode": 20, "table-encode-360p": 25, "table-encode-720p": 30}


def cut(source: Path, frames: int, frame_bytes: int, target: Path) -> None:
    """The first `frames` frames of an I420 file, zstd-compressed."""
    with source.open("rb") as f, subprocess.Popen(["zstd", "-q", "-3", "-o", str(target)],
                                                  stdin=subprocess.PIPE) as z:
        z.stdin.write(f.read(frames * frame_bytes))
        z.stdin.close()
        if z.wait():
            raise SystemExit(f"zstd failed on {source}")


def stage(stage_dir: Path, args, clip_sets: dict[str, tuple[str, ...]]) -> None:
    """The upload: bin/ (ours and the references), repo/ (the scripts at their repository
    paths), data/ (bench.py's NC_DATA_ROOT: clip sets and 150-frame cuts), pkg/ (each
    module), core/ (@ffmpeg/core's dist/umd) and streams/."""
    bin_dir = stage_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "nanocodec-pgo").symlink_to(args.pgo.resolve())
    (bin_dir / "nanocodec-plain").symlink_to(args.plain.resolve())
    for name in DECODERS:
        (bin_dir / name).symlink_to((ANCHORS / name).resolve())
    path_dir = stage_dir / "path"
    path_dir.mkdir()
    for name, anchor in PATH_NAMES.items():
        (path_dir / name).symlink_to((ANCHORS / anchor).resolve())
    (path_dir / "ffmpeg").write_text(FFMPEG_WRAPPER)
    (path_dir / "ffmpeg").chmod(0o755)
    (path_dir / "nanocodec").symlink_to(args.plain.resolve())  # bench.py --bin dir
    (stage_dir / "data/anchors/x86_64/bin").mkdir(parents=True)  # bench.py's avmdec
    (stage_dir / "data/anchors/x86_64/bin/avmdec").symlink_to((ANCHORS / "avmdec").resolve())
    for rel in REPO_FILES:
        (stage_dir / "repo" / rel).parent.mkdir(parents=True, exist_ok=True)
        (stage_dir / "repo" / rel).symlink_to(REPO / rel)
    for name, pkg in {"main": args.pkg, **dict(args.compare_pkg)}.items():
        (stage_dir / "pkg").mkdir(exist_ok=True)
        (stage_dir / "pkg" / name).symlink_to(Path(pkg).resolve())
    (stage_dir / "core").symlink_to(args.ffmpeg_core.resolve())  # the package (its version)
    (stage_dir / "streams").mkdir()
    for directory in args.streams:
        (stage_dir / "streams" / directory.name).symlink_to(directory.resolve())
    for setname, names in clip_sets.items():
        index = {c["name"]: c for c in json.loads((DATA / f"clips/{setname}.json").read_text())["clips"]}
        clips = [{**index[n], "yuv": f"clips/{setname}/{n}.yuv"} for n in names]
        (stage_dir / "data/clips" / setname).mkdir(parents=True)
        (stage_dir / "data/clips" / f"{setname}.json").write_text(json.dumps({"clips": clips}))
        for c in clips:
            frame = c["width"] * c["height"] * 3 // 2
            cut(DATA / index[c["name"]]["yuv"], FRAMES, frame,
                stage_dir / "data" / f"{c['yuv']}.zst")
    (stage_dir / "out").mkdir()


def jobs(args) -> list[tuple[str, str]]:
    """(name, remote shell) for each stage, in order."""
    cd = f"cd ~/{REMOTE_DIR} && export PATH=$HOME/{REMOTE_DIR}/node/bin:$HOME/{REMOTE_DIR}/path:$PATH " \
         f"NC_DATA_ROOT=$HOME/{REMOTE_DIR}/data && "
    pin = f"taskset -c {CPU}"
    streams = " ".join(f"streams/{d.name}" for d in args.streams)
    decoder = (f"python3 repo/tools/dev/decoder_speed.py {{rounds}} {{out}} --nanocodec bin/nanocodec-pgo "
               f"--anchors bin --repeats {args.decoder_repeats} --streams {{streams}}")
    yuv = "data/clips/{0}/{1}.yuv".format(*YARDSTICK_CLIP)
    yardstick = (f"{pin} python3 repo/tools/web/ffmpeg_wasm_yardstick.py --pkg {{pkg}}/relaxed "
                 f"--core core/dist/umd/ffmpeg-core.js --streams streams/all --yuv {yuv} --cpu {CPU} "
                 f"--rounds {{rounds}} --json {{out}}")
    table = (f"python3 repo/tools/web/bench.py --bin path --pkg pkg/main --modules simd128,relaxed "
             f"--targets 720p=600,360p=300 --presets medium,fast --cpu {CPU} --workers 1 "
             f"--work scratch --rounds {args.table_rounds}")
    out = [
        ("prepare", cd + f"curl -fsSL {NODE_URL} | tar -xJ && mv node-{NODE_VERSION}-linux-x64 node && "
         "for c in data/clips/*/*.zst; do zstd -d -q --rm \"$c\"; done && "
         "mkdir -p streams/all && for d in " + streams + "; do ln -f $d/* streams/all/; "
         "done && ln -f streams/all/cam150.axis-m5074-720p.hevc streams/all/axis720.cam.hevc && "
         "! ldd bin/* | grep 'not found' && bin/nanocodec-pgo --version && "
         "bin/nanocodec-plain --version && node --version && path/ffmpeg -version | head -1 && lscpu > out/lscpu.txt && "
         + pinning_command() + " && " + isa_command("amd64")),
        ("pipe-clean", cd + f"mkdir -p pc && ln -sf ../streams/all/crowd_mot17_09.600k.av1.ivf pc/ && "
         "ln -sf ../streams/all/crowd_mot17_09.600k.hevc.hevc pc/ && "
         + pin + " " + decoder.format(rounds=1, out="out/pipe-clean.json", streams="pc")),
        ("decoders", cd + pin + " " + decoder.format(rounds=args.decoder_rounds, out="out/decoders.json",
                                                  streams=streams)),
        ("yardstick", cd + yardstick.format(pkg="pkg/main", rounds=args.yardstick_rounds,
                                            out="out/yardstick.json")),
    ]
    for name, _ in args.compare_pkg:
        # interleaved: the yardstick's H.264 Medium encode on each module in turn
        encode = (f"{pin} node repo/tools/web/bench-runner.mjs encode --pkg pkg/{{pkg}}/relaxed "
                  f"--codec h264 --yuv {yuv} --width 1280 --height 720 --frames {FRAMES} --fps 30 "
                  "--kbps 600 --peak-kbps 1200 --buffer-ms 1000 --key-seconds 30 --repeat 1 "
                  "| tail -1 >> out/compare-{pkg}.jsonl")
        out.append((f"compare-{name}", cd + " && ".join(
            encode.format(pkg=pkg) for _ in range(args.yardstick_rounds) for pkg in ("main", name))))
    out += [
        ("table-decode", cd + f"{table} --skip-encode --repeat 2 --json out/table-decode.json "
                              "--csv out/table-decode.csv"),
        ("table-encode-360p", cd + f"{table} --skip-decode --encode-res 360p --repeat 2 "
                                   "--json out/table-360p.json --csv out/table-360p.csv"),
        ("table-encode-720p", cd + f"{table} --skip-decode --encode-res 720p --repeat 1 "
                                   "--json out/table-720p.json --csv out/table-720p.csv"),
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pgo", type=Path, required=True, help="our x86 CLI timed in the decoder table")
    parser.add_argument("--plain", type=Path, required=True, help="our x86 plain CLI (browser table)")
    parser.add_argument("--pkg", type=Path, required=True, help="tools/web/build.sh output dir")
    parser.add_argument("--compare-pkg", type=lambda s: tuple(s.split("=", 1)), action="append",
                        default=[], metavar="NAME=PKG", help="another module, timed against --pkg")
    parser.add_argument("--streams", type=Path, nargs="+", required=True)
    parser.add_argument("--ffmpeg-core", type=Path, required=True, help="the @ffmpeg/core package directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoder-rounds", type=int, default=5)
    parser.add_argument("--decoder-repeats", type=int, default=3, help="timed passes per process")
    parser.add_argument("--yardstick-rounds", type=int, default=7)
    parser.add_argument("--table-rounds", type=int, default=3)
    parser.add_argument("--skip", default="", help="comma list of jobs to leave out")
    parser.add_argument("--lifetime", type=int, default=90, help="hard instance lifetime, minutes")
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--stage-dir", type=Path, required=True,
                        help="an empty local staging directory (clip cuts: about 1 GB)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    todo = [(name, command) for name, command in jobs(args) if name not in skip]
    if args.dry_run:
        for name, command in todo:
            print(f"{name}: {command}")
        return
    if args.stage_dir.exists() and any(args.stage_dir.iterdir()):
        sys.exit(f"{args.stage_dir} is not empty")
    # the browser table reads every clip set, the yardstick only its clip
    tables = any(name.startswith("table") for name, _ in todo)
    stage(args.stage_dir, args, CLIP_SETS if tables else {YARDSTICK_CLIP[0]: YARDSTICK_CLIP[1:]})
    region, arch = INSTANCES[TIMING_MACHINE]
    spec = SessionSpec(region=region, instance_type=TIMING_MACHINE, arch=arch,
                       purpose="decoder and browser speed tables (results site)",
                       lifetime_minutes=args.lifetime, budget_usd=args.budget, on_demand=True,
                       root_gb=20)
    started = time.monotonic()
    with Session(spec) as session:
        session.upload(args.stage_dir, f"~/{REMOTE_DIR}")
        try:
            for name, command in todo:
                say(f"remote: {name}")
                session.run_detached(command, f"~/{REMOTE_DIR}/out/{name}.log", JOB_MINUTES.get(
                    name.split("-")[0] if name.startswith("compare") else name, 20),
                    stall_minutes=JOB_MINUTES.get(name, 20))
        finally:
            session.download(f"~/{REMOTE_DIR}/out", args.out)
    say(f"session wall time {(time.monotonic() - started) / 60:.1f} min; results in {args.out}")
    shutil.rmtree(args.stage_dir)
    print(shlex.join(["results", str(args.out)]))


if __name__ == "__main__":
    main()
