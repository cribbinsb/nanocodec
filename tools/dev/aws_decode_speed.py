#!/usr/bin/env python3
"""The results site's decoder and browser speed tables, timed on quiet rented cores.

    python3 tools/dev/aws_decode_speed.py --machine c8i.2xlarge --nanocodec BIN --pkg PKG \\
        --streams DIR [DIR ...] --ffmpeg-core DIR --out DIR --stage-dir DIR \\
        [--cpus 1,2,3] [--skip table-decode,...] [--plain BIN] [--compare-pkg NAME=PKG ...] \\
        [--decoder-rounds 5] [--yardstick-rounds 7] [--table-rounds 3] [--dry-run]

One box of --machine (MACHINES: the benchmark's two timing families, c8i.2xlarge and
c8g.2xlarge, spot in eu-north-1; or the earlier records' c8a.large, on demand), launched
through tools/dev/aws_session.py (budget, hard lifetime, ledger, cleanup on every exit).
Each timed process is pinned to one physical core of --cpus (the first also runs the browser
jobs) and the jobs run one after another; the decoder job splits the streams over the
--cpus, one decoder_speed.py process per core, so ours and the references of a stream run in
the same process's interleaved rounds. Each round's records leave the box as written
(Session.pulling); a lost spot box is rerun whole (a session is minutes). The jobs:

  prepare      node (the official v22.21.1 build), readiness: every binary loads, version
               stamps, the CPU model, its SIMD flags and the pinning;
  pipe-clean   decoder_speed.py one round on two streams;
  decoders     tools/dev/decoder_speed.py (our --nanocodec build against the in-process
               harnesses of FFmpeg, dav1d, libvpx, VVdeC, AVM and dav2d), --decoder-rounds
               rounds, out/decoders-cpuN.json per core; each --also build of ours
               is timed beside it in the same rounds;
  yardstick    tools/web/ffmpeg_wasm_yardstick.py (our browser module against
               ffmpeg.wasm), --yardstick-rounds rounds; then, with --compare-pkg, the
               same module encodes of each compared module interleaved with --pkg's;
  table        tools/web/bench.py, the browser table's documented runs (decode; encode
               at 360p; encode at 720p), the native column on the plain build,
               --table-rounds rounds (needs --plain);
  wasm-yardsticks  tools/web/wasm_yardsticks.py (our browser module against the WebAssembly
               builds of FFmpeg, dav1d and libvpx, --wasm-bin, and ffmpeg.wasm), every stream,
               split over the --cpus like the decoders, --yardstick-rounds rounds,
               out/wasm-yardsticks-cpuN.json;
  webcodecs    tools/web/webcodecs_yardstick.py (our module against Chrome's own software
               decoders through WebCodecs, in the same headless browser: Google Chrome on
               x86, Playwright's Chromium on Arm, where Google ships no Linux build), the
               same split, out/webcodecs-cpuN.json.

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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aws_session import Session, SessionSpec, say  # noqa: E402
from aws_timing import isa_command, pinning_command  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / "nanocodec-data"
#: The anchors the jobs run: decoder_speed.py's in-process harnesses, then bench.py's
#: decoders (the browser table's; staged where the architecture's anchors have them).
HARNESSES = ("libavcodec-bench", "vpx-bench", "vvdec-bench", "avm-bench", "dav2d-bench")
TABLE_DECODERS = ("ffmpeg-decode", "dav1d", "vpxdec", "vvdecapp", "avmdec")


@dataclass(frozen=True)
class Target:
    """A timing box: its region, AMI architecture, market and the anchors built for it."""
    region: str
    arch: str
    on_demand: bool
    anchors: Path
    node: str  # the official node build's platform suffix


#: The boxes: the benchmark's two timing families (docs/aws-cost-guide.md row 1: quiet,
#: 0.3-0.4 % per-encode sd, spot), and the earlier records' c8a.large (on demand: four c8a
#: spot sessions were reclaimed within 6 min, aws_timing.py).
MACHINES = {
    "c8i.2xlarge": Target("eu-north-1", "amd64", False, DATA / "anchors/x86_64/bin", "linux-x64"),
    "c8g.2xlarge": Target("eu-north-1", "arm64", False, DATA / "anchors/aarch64/bin", "linux-arm64"),
    "c8a.large": Target("eu-west-1", "amd64", True, DATA / "anchors/x86_64/bin", "linux-x64"),
}
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
              "tools/web/ffmpeg-wasm-runner.cjs", "web/codecs.js", "tools/web/wasm_yardsticks.py",
              "tools/web/webcodecs_yardstick.py", "tools/web/webcodecs-bench.html")
#: The Playwright release the webcodecs job installs (the PC's), and its browser per box:
#: Google Chrome where Google ships it (x86), else Playwright's Chromium build.
PLAYWRIGHT_VERSION = "1.60.0"
BROWSER_CHANNELS = {"amd64": "chrome", "arm64": "chromium"}
PLAYWRIGHT_PLATFORMS = {"amd64": "ubuntu24.04-x64", "arm64": "ubuntu24.04-arm64"}
#: bench.py's clip sets and clips (bench.RESOLUTIONS), and the yardstick's clip.
CLIP_SETS = {"quick": ("crowd_mot17_09", "pan_jaad_0117", "static_chirla_s000c3"),
             "quick360": ("crowd_mot17_09_360p", "pan_jaad_0117_360p", "static_chirla_s000c3_360p")}
YARDSTICK_CLIP = ("quick", "crowd_mot17_09")
FRAMES = 150
NODE_VERSION = "v22.21.1"
NODE_URL = "https://nodejs.org/dist/{0}/node-{0}-{1}.tar.xz"
REMOTE_DIR = "decode-speed"
#: A job's own timeout (minutes); the whole session's lifetime is --lifetime.
JOB_MINUTES = {"prepare": 10, "pipe-clean": 5, "decoders": 45, "yardstick": 25, "compare": 10,
               "wasm-yardsticks": 45, "webcodecs": 45,
               "table-decode": 20, "table-encode-360p": 25, "table-encode-720p": 30}


def cut(source: Path, frames: int, frame_bytes: int, target: Path) -> None:
    """The first `frames` frames of an I420 file, zstd-compressed."""
    with source.open("rb") as f, subprocess.Popen(["zstd", "-q", "-3", "-o", str(target)],
                                                  stdin=subprocess.PIPE) as z:
        z.stdin.write(f.read(frames * frame_bytes))
        z.stdin.close()
        if z.wait():
            raise SystemExit(f"zstd failed on {source}")


def shards(directories: list[Path], count: int) -> list[list[Path]]:
    """The streams split over `count` cores, largest first onto the lightest core (bytes
    stand in for decode time), each list in name order."""
    streams = sorted((p for d in directories for p in d.iterdir() if p.is_file()),
                     key=lambda p: (-p.stat().st_size, p.name))
    out: list[list[Path]] = [[] for _ in range(count)]
    load = [0] * count
    for stream in streams:
        lightest = load.index(min(load))
        out[lightest].append(stream)
        load[lightest] += stream.stat().st_size
    return [sorted(s, key=lambda p: p.name) for s in out]


def stage(stage_dir: Path, args, clip_sets: dict[str, tuple[str, ...]], tables: bool) -> None:
    """The upload: bin/ (ours and the references), repo/ (the scripts at their repository
    paths), data/ (bench.py's NC_DATA_ROOT: clip sets and 150-frame cuts), pkg/ (each
    module), core/ (@ffmpeg/core's dist/umd) and streams/shard-N/ (one per timed core)."""
    anchors = MACHINES[args.machine].anchors
    bin_dir = stage_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "nanocodec").symlink_to(args.nanocodec.resolve())
    for name, binary in args.also:
        (bin_dir / f"nanocodec-{name}").symlink_to(Path(binary).resolve())
    for name in HARNESSES + (TABLE_DECODERS if tables else ()):
        if (anchors / name).exists():
            (bin_dir / name).symlink_to((anchors / name).resolve())
    path_dir = stage_dir / "path"
    path_dir.mkdir()
    if tables:
        (bin_dir / "nanocodec-plain").symlink_to(args.plain.resolve())
        for name, anchor in PATH_NAMES.items():
            (path_dir / name).symlink_to((anchors / anchor).resolve())
        (path_dir / "ffmpeg").write_text(FFMPEG_WRAPPER)
        (path_dir / "ffmpeg").chmod(0o755)
        (path_dir / "nanocodec").symlink_to(args.plain.resolve())  # bench.py --bin dir
        (stage_dir / "data/anchors/x86_64/bin").mkdir(parents=True)  # bench.py's avmdec
        (stage_dir / "data/anchors/x86_64/bin/avmdec").symlink_to((anchors / "avmdec").resolve())
    for rel in REPO_FILES:
        (stage_dir / "repo" / rel).parent.mkdir(parents=True, exist_ok=True)
        (stage_dir / "repo" / rel).symlink_to(REPO / rel)
    for name, pkg in {"main": args.pkg, **dict(args.compare_pkg)}.items():
        (stage_dir / "pkg").mkdir(exist_ok=True)
        (stage_dir / "pkg" / name).symlink_to(Path(pkg).resolve())
    (stage_dir / "core").symlink_to(args.ffmpeg_core.resolve())  # the package (its version)
    if args.wasm_bin:
        (stage_dir / "wasm-bin").symlink_to(args.wasm_bin.resolve())
    for index, streams in enumerate(shards(args.streams, len(args.cpus))):
        (stage_dir / f"streams/shard-{index}").mkdir(parents=True)
        for stream in streams:
            (stage_dir / f"streams/shard-{index}" / stream.name).symlink_to(stream.resolve())
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


def jobs(args, tables: bool) -> list[tuple[str, str]]:
    """(name, remote shell) for each stage, in order."""
    target = MACHINES[args.machine]
    cd = f"cd ~/{REMOTE_DIR} && export PATH=$HOME/{REMOTE_DIR}/node/bin:$HOME/{REMOTE_DIR}/path:$PATH " \
         f"NC_DATA_ROOT=$HOME/{REMOTE_DIR}/data && "
    cpu = args.cpus[0]
    pin = f"taskset -c {cpu}"
    decoder = (f"python3 repo/tools/dev/decoder_speed.py {{rounds}} {{out}} --nanocodec bin/nanocodec "
               f"--anchors bin --repeats {args.decoder_repeats} --streams {{streams}}"
               + "".join(f" --also {name}=bin/nanocodec-{name}" for name, _ in args.also))

    def per_core(command: str, rounds: int, out: str) -> str:
        """One process per core, each on its shard; the job fails when any of them does."""
        return "{ " + " ".join(
            f"taskset -c {c} " + command.format(rounds=rounds, out=f"out/{out}-cpu{c}.json",
                                                streams=f"streams/shard-{i}") + " &"
            for i, c in enumerate(args.cpus)) + " fail=0; for p in $(jobs -p); do wait $p || fail=1; done; exit $fail; }"
    decoders = per_core(decoder, args.decoder_rounds, "decoders")
    wasm = per_core("python3 repo/tools/web/wasm_yardsticks.py {rounds} {out} --pkg pkg/main/relaxed "
                    "--bin wasm-bin --core core/dist/umd/ffmpeg-core.js --streams {streams}",
                    args.yardstick_rounds, "wasm-yardsticks")
    channel = BROWSER_CHANNELS[target.arch]
    webcodecs = ("export PATH=$HOME/.local/bin:$PATH && "
                 # the image has no pip: the official bootstrap, into the user's site
                 "{ python3 -m pip --version || curl -fsSL https://bootstrap.pypa.io/get-pip.py | "
                 "python3 - -q --user --break-system-packages; } > out/pip-install.log 2>&1 && "
                 f"python3 -m pip install -q --user --break-system-packages playwright=={PLAYWRIGHT_VERSION} && "
                 # Playwright 1.60 knows Ubuntu up to 24.04; the image is newer: its 24.04 builds
                 f"export PLAYWRIGHT_HOST_PLATFORM_OVERRIDE={PLAYWRIGHT_PLATFORMS[target.arch]} && "
                 f"python3 -m playwright install --with-deps {channel} > out/playwright-install.log 2>&1 && "
                 + per_core(f"python3 repo/tools/web/webcodecs_yardstick.py {{rounds}} {{out}} --pkg pkg/main/relaxed "
                            f"--channel {channel} --streams {{streams}}", args.yardstick_rounds, "webcodecs"))
    node_dir = f"node-{NODE_VERSION}-{target.node}"
    yuv = "data/clips/{0}/{1}.yuv".format(*YARDSTICK_CLIP)
    yardstick = (f"{pin} python3 repo/tools/web/ffmpeg_wasm_yardstick.py --pkg {{pkg}}/relaxed "
                 f"--core core/dist/umd/ffmpeg-core.js --streams streams/all --yuv {yuv} --cpu {cpu} "
                 f"--rounds {{rounds}} --json {{out}}")
    table = (f"python3 repo/tools/web/bench.py --bin path --pkg pkg/main --modules simd128,relaxed "
             f"--targets 720p=600,360p=300 --presets medium,fast --cpu {cpu} --workers 1 "
             f"--work scratch --rounds {args.table_rounds}")
    out = [
        ("prepare", cd + f"curl -fsSL {NODE_URL.format(NODE_VERSION, target.node)} | tar -xJ && "
         f"mv {node_dir} node && "
         "for c in data/clips/*/*.zst; do zstd -d -q --rm \"$c\"; done && "
         "mkdir -p streams/all && ln -f streams/shard-*/* streams/all/ && "
         "ln -f streams/all/cam150.axis-m5074-720p.hevc streams/all/axis720.cam.hevc && "
         "! ldd bin/* | grep 'not found' && bin/nanocodec --version | tee out/nanocodec-version.txt && "
         "node --version && lscpu > out/lscpu.txt && lscpu -e > out/lscpu-e.txt && "
         + pinning_command() + (" && " + isa_command(target.arch) if tables else "")),
        ("pipe-clean", cd + "mkdir -p pc && ln -sf ../streams/all/crowd_mot17_09.600k.av1.ivf pc/ && "
         "ln -sf ../streams/all/crowd_mot17_09.600k.vvc.vvc pc/ && "
         "ln -sf ../streams/all/crowd_mot17_09.600k.av2.ivf pc/ && "
         "ln -sf ../streams/all/crowd_mot17_09.600k.vp9.ivf pc/ && "
         + pin + " " + decoder.format(rounds=1, out="out/pipe-clean.json", streams="pc")),
        ("decoders", cd + decoders),
        ("yardstick", cd + yardstick.format(pkg="pkg/main", rounds=args.yardstick_rounds,
                                            out="out/yardstick.json")),
        ("wasm-yardsticks", cd + wasm),
        ("webcodecs", cd + webcodecs),
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
    parser.add_argument("--machine", choices=sorted(MACHINES), default="c8i.2xlarge")
    parser.add_argument("--nanocodec", type=Path, required=True,
                        help="our CLI for --machine's architecture, timed in the decoder table")
    parser.add_argument("--plain", type=Path, help="our x86 plain CLI (the browser table jobs)")
    parser.add_argument("--also", type=lambda s: tuple(s.split("=", 1)), action="append", default=[],
                        metavar="NAME=BIN", help="another build of ours, timed beside --nanocodec as ours-NAME")
    parser.add_argument("--cpus", type=lambda s: [int(c) for c in s.split(",")], default=[1],
                        help="the timed cores (one physical core each), comma list")
    parser.add_argument("--pkg", type=Path, required=True, help="tools/web/build.sh output dir")
    parser.add_argument("--compare-pkg", type=lambda s: tuple(s.split("=", 1)), action="append",
                        default=[], metavar="NAME=PKG", help="another module, timed against --pkg")
    parser.add_argument("--streams", type=Path, nargs="+", required=True)
    parser.add_argument("--ffmpeg-core", type=Path, required=True, help="the @ffmpeg/core package directory")
    parser.add_argument("--wasm-bin", type=Path,
                        help="tools/web/build_wasm_yardsticks.sh's PREFIX/bin (the wasm-yardsticks job)")
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
    names = [name for name, _ in jobs(args, True) if name not in skip]
    tables = any(name.startswith("table") for name in names)
    if tables and not args.plain:
        sys.exit("the table jobs need --plain (or --skip them)")
    todo = [(name, command) for name, command in jobs(args, tables) if name not in skip]
    if args.dry_run:
        for name, command in todo:
            print(f"{name}: {command}")
        return
    if args.stage_dir.exists() and any(args.stage_dir.iterdir()):
        sys.exit(f"{args.stage_dir} is not empty")
    # the browser table reads every clip set, the yardstick only its clip
    stage(args.stage_dir, args, CLIP_SETS if tables else {YARDSTICK_CLIP[0]: YARDSTICK_CLIP[1:]}, tables)
    target = MACHINES[args.machine]
    spec = SessionSpec(region=target.region, instance_type=args.machine, arch=target.arch,
                       purpose="decoder and browser speed tables (results site)",
                       lifetime_minutes=args.lifetime, budget_usd=args.budget,
                       on_demand=target.on_demand, root_gb=20)
    started = time.monotonic()
    with Session(spec) as session:
        session.upload(args.stage_dir, f"~/{REMOTE_DIR}")
        # each round's records leave the box as written (and at once on a spot notice)
        with session.pulling(f"~/{REMOTE_DIR}/out", args.out, "decode-speed-out"):
            for name, command in todo:
                say(f"remote: {name}")
                session.run_detached(command, f"~/{REMOTE_DIR}/out/{name}.log", JOB_MINUTES.get(
                    name.split("-")[0] if name.startswith("compare") else name, 20),
                    stall_minutes=JOB_MINUTES.get(name, 20))
    say(f"session wall time {(time.monotonic() - started) / 60:.1f} min; results in {args.out}")
    shutil.rmtree(args.stage_dir)
    print(shlex.join(["results", str(args.out)]))


if __name__ == "__main__":
    main()
