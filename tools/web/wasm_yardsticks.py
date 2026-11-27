#!/usr/bin/env python3
"""Our browser module against WebAssembly reference decoders, every stream, under node.

    taskset -c 1 python3 tools/web/wasm_yardsticks.py 5 out.json --pkg PKG/relaxed \\
        --bin WASM_PREFIX/bin --core CORE/dist/umd/ffmpeg-core.js --streams DIR [DIR ...]

The browser counterpart of tools/dev/decoder_speed.py: the same streams, record format
({"stream|codec|decoder": {"seconds": [...], "frames": n}}) and interleaved rounds (the order
reversed every other round), one process per run. Each process decodes the stream twice, a
fresh decoder each pass, and the faster pass counts (the first also warms V8's optimising
compiler: WebAssembly starts in its baseline tier); ffmpeg.wasm, whose package runs one
command per module, has one pass. Neither side is charged module start-up, probing, demuxing
or output:

- ours-wasm: tools/web/bench-runner.mjs decode (the BrowserDecoder.decode calls; the
  display hand-off it also times is left out);
- ffmpeg-wasm-simd (H.264, HEVC, VP9, VVC) and dav1d-wasm-simd (AV1, through libavcodec's
  libdav1d wrapper): FFmpeg n8.0.1 and dav1d 1.5.4 compiled by us with emscripten -O3
  -msimd128 (tools/web/build_wasm_yardsticks.sh), timed by the native comparison's own
  in-process harness, tools/anchors/libavcodec_bench.c;
- libvpx-wasm-simd (VP9): libvpx v1.16.0 the same way, tools/anchors/vpx_bench.c;
- ffmpeg.wasm (H.264, HEVC, VP9): the official @ffmpeg/core single-thread build
  (tools/web/ffmpeg-wasm-runner.cjs, the sum of its decode_video calls).

A run that fails (a reference without the stream's features, or out of memory) is recorded
under "errors" and not repeated; the others go on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dev"))
from decoder_speed import codec_of  # noqa: E402

RUNNER = HERE / "bench-runner.mjs"
FFMPEG_RUNNER = HERE / "ffmpeg-wasm-runner.cjs"
OURS = "ours-wasm"
#: The references each codec is timed against: record name -> (harness, extra arguments).
REFERENCES = {
    "h264": {"ffmpeg-wasm-simd": ("libavcodec-bench.js", []), "ffmpeg.wasm": None},
    "hevc": {"ffmpeg-wasm-simd": ("libavcodec-bench.js", []), "ffmpeg.wasm": None},
    "vp9": {"ffmpeg-wasm-simd": ("libavcodec-bench.js", []), "libvpx-wasm-simd": ("vpx-bench.js", []),
            "ffmpeg.wasm": None},
    "vvc": {"ffmpeg-wasm-simd": ("libavcodec-bench.js", [])},
    "av1": {"dav1d-wasm-simd": ("libavcodec-bench.js", ["--decoder", "libdav1d"])},
    "av2": {},
}
#: Process limit per run (seconds): far above the slowest stream, so a hang cannot stall a box.
RUN_TIMEOUT = 600
#: Timed passes per process; the faster counts.
PASSES = 2


def command(decoder: str, codec: str, stream: Path, args) -> list[str]:
    if decoder == OURS:
        return ["node", str(RUNNER), "decode", "--pkg", str(args.pkg), "--codec", codec,
                "--stream", str(stream), "--repeat", str(PASSES)]
    if decoder == "ffmpeg.wasm":
        return ["node", str(FFMPEG_RUNNER), str(args.core), str(stream), "none", "--",
                "-i", "in", "-f", "null", "-"]
    harness, extra = REFERENCES[codec][decoder]
    return ["node", str(args.bin / harness), *extra, "--repeat", str(PASSES), str(stream)]


def seconds_and_frames(decoder: str, lines: list[dict]) -> tuple[float, int]:
    """One run's decode time (its faster pass) and frame count from its JSON lines."""
    if decoder not in (OURS, "ffmpeg.wasm"):
        if len({p["frames"] for p in lines}) != 1:
            raise RuntimeError("frame counts differ between passes")
        return min(p["seconds"] for p in lines), lines[0]["frames"]
    out = lines[-1]
    if decoder == OURS:  # bench-runner reports its faster pass
        return out["decode_ms_per_frame"] * out["frames"] / 1000, out["frames"]
    if decoder == "ffmpeg.wasm":
        if out["rc"] != 0:
            raise RuntimeError(f"ffmpeg.wasm exited {out['rc']}")
        return out["decode_ms"] / 1000, out["frames"]


def run(cmd: list[str]) -> list[dict]:
    """The JSON lines a run prints."""
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    if done.returncode:
        raise RuntimeError(f"exit {done.returncode}: {done.stderr.strip()[-300:]}")
    return [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rounds", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pkg", type=Path, required=True, help="our browser module (tools/web/build.sh)")
    parser.add_argument("--bin", type=Path, required=True, help="build_wasm_yardsticks.sh's PREFIX/bin")
    parser.add_argument("--core", type=Path, required=True, help="@ffmpeg/core's dist/umd/ffmpeg-core.js")
    parser.add_argument("--streams", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    for name in ("pkg", "bin", "core"):  # the node runners resolve paths against their own directory
        setattr(args, name, getattr(args, name).resolve())
    streams = sorted((p.absolute() for d in args.streams for p in d.iterdir()
                      if p.is_file() and not p.name.startswith("pan_")),
                     key=lambda p: p.name)
    jobs = [(s, codec_of(s.name), decoder) for s in streams
            for decoder in (OURS, *REFERENCES[codec_of(s.name)])]
    record: dict = {"res": {}, "errors": {}}
    for index in range(args.rounds):
        for stream, codec, decoder in jobs if index % 2 == 0 else jobs[::-1]:
            key = f"{stream.name}|{codec}|{decoder}"
            if key in record["errors"]:
                continue
            try:
                seconds, frames = seconds_and_frames(decoder, run(command(decoder, codec, stream, args)))
            except (RuntimeError, subprocess.TimeoutExpired, KeyError, ValueError) as error:
                record["errors"][key] = str(error)
                record["res"].pop(key, None)
                print(f"{key}: {error}", file=sys.stderr, flush=True)
                continue
            entry = record["res"].setdefault(key, {"seconds": [], "frames": frames})
            entry["seconds"].append(seconds)
        # after every round, so a lost box keeps the rounds it finished
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"round {index} done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
