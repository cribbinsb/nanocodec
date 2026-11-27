#!/usr/bin/env python3
"""Single-thread decode speed: our decoder against FFmpeg's, both timed in process.

    # x86, on a quiet rented core (tools/dev/aws_decode_speed.py runs this pinned to CPU 1)
    taskset -c 1 python3 tools/dev/decoder_speed.py 5 x86.json --nanocodec BIN \\
        --anchors ~/nanocodec-data/anchors/x86_64/bin --streams DIR [DIR ...]

Neither side is charged process start-up, probing, demuxing or output: each process
reads its stream into memory once, then decodes it --repeats times, each pass with a
fresh one-thread decoder, frames discarded, and reports each pass's time:

- ours: `nanocodec decode --bench N` (crates/nc-cli/src/decode.rs `bench`; a pass also
  covers the decoder's construction and the container framing over memory);
- FFmpeg n8.0.1: `libavcodec-bench --repeat N` (tools/anchors/libavcodec_bench.c, built
  with the pinned decode-only FFmpeg; packets demuxed by libavformat before timing, a pass
  is avcodec_send_packet / avcodec_receive_frame over every packet plus the flush). It
  decodes H.264, HEVC, VP9 and VVC with FFmpeg's own decoders, and AV1 with the pinned
  dav1d 1.5.4 through libavcodec's libdav1d wrapper (recorded as "dav1d");
- libvpx (vpxdec's library, VP9), VVdeC, AVM (avmdec's library, AV2) and dav2d (AV2):
  `vpx-bench`, `vvdec-bench`, `avm-bench`, `dav2d-bench` (tools/anchors/*_bench.c,
  decode_bench.h): the stream split
  into IVF frames or Annex-B NAL units in memory before timing, a pass is the library's
  decode call over every unit plus its flush.

A reference whose harness is not in --anchors is skipped with a note. Files named `pan_*`
are skipped.

Each round runs every process once, the order reversed every other round. The record
keeps every pass: {"stream|codec|decoder": {"seconds": [...], "frames": n, "t": median
seconds, "rss": peak KiB of the process}}. Exactness is checked separately (the
decoders' conformance suites and the landing conformance check), not here.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

#: Container or file suffix -> codec, for streams not named `NAME.CODEC.ivf`.
EXTENSION_CODECS = {"264": "h264", "hevc": "hevc", "vvc": "vvc"}
#: Camera recordings (`cam150.*`) are raw HEVC.
CAMERA_PREFIX = "cam150"
#: The references each codec is timed against, by record name.
REFERENCES = {"h264": ("ffmpeg",), "hevc": ("ffmpeg",), "vp9": ("ffmpeg", "vpxdec"),
              "vvc": ("ffmpeg", "vvdec"), "av1": ("dav1d",), "av2": ("avmdec", "dav2d")}
#: Each reference's in-process harness (tools/anchors/build_anchors.sh avbench, decbench).
HARNESSES = {"ffmpeg": "libavcodec-bench", "dav1d": "libavcodec-bench", "vpxdec": "vpx-bench",
             "vvdec": "vvdec-bench", "avmdec": "avm-bench", "dav2d": "dav2d-bench"}


def codec_of(name: str) -> str:
    parts = name.split(".")
    if parts[0] == CAMERA_PREFIX:
        return "hevc"
    if parts[-1] == "ivf":
        return parts[-2]
    return EXTENSION_CODECS[parts[-1]]


def run(command: list[str]) -> tuple[list[dict], int]:
    """The passes a bench process prints, and its peak RSS (KiB)."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, error = process.stdout.read(), process.stderr.read()  # the passes print little
    _, status, usage = os.wait4(process.pid, 0)
    if os.waitstatus_to_exitcode(status):
        sys.exit(f"FAIL {' '.join(command)}: {error[-300:]}")
    return [json.loads(line) for line in out.splitlines()], usage.ru_maxrss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rounds", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--nanocodec", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True, help="the directory holding the harnesses")
    parser.add_argument("--streams", type=Path, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3, help="timed passes per process")
    args = parser.parse_args()
    harnesses = {name: args.anchors / binary for name, binary in HARNESSES.items()}
    for name, harness in harnesses.items():
        if not harness.exists():
            print(f"note: {harness} is missing: {name} is not timed (tools/anchors/build_anchors.sh "
                  "avbench, decbench)", file=sys.stderr)
    streams = sorted((p for d in args.streams for p in d.iterdir()
                      if not p.name.startswith("pan_") and codec_of(p.name) in REFERENCES),
                     key=lambda p: p.name)
    jobs = []
    for stream in streams:
        codec = codec_of(stream.name)
        jobs.append((stream.name, codec, "ours", [
            str(args.nanocodec), "decode", "--codec", codec, "--input", str(stream),
            "--bench", str(args.repeats)]))
        jobs += [(stream.name, codec, name, [str(harnesses[name]), "--repeat", str(args.repeats), str(stream)])
                 for name in REFERENCES[codec] if harnesses[name].exists()]
    results: dict[str, dict] = {}
    for round_ in range(args.rounds):
        for name, codec, decoder, command in jobs if round_ % 2 == 0 else jobs[::-1]:
            passes, rss = run(command)
            entry = results.setdefault(f"{name}|{codec}|{decoder}",
                                       {"seconds": [], "frames": passes[0]["frames"], "rss": 0})
            if any(p["frames"] != entry["frames"] for p in passes):
                sys.exit(f"{name} {decoder}: frame counts differ between passes")
            entry["seconds"] += [p["seconds"] for p in passes]
            entry["rss"] = max(entry["rss"], rss)
        # after every round, so a lost box keeps the rounds it finished
        for entry in results.values():
            entry["t"] = statistics.median(entry["seconds"])
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        print(f"round {round_} done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
