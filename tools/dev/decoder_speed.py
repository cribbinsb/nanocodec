#!/usr/bin/env python3
"""Single-thread decode speed: our decoder against the reference decoders, interleaved rounds.

The results site's decoder table (eval/nceval/site.py `decoder_data`,
docs/results/decoder-speed-*/) comes from this, run once per platform:

    # Jetson (anchors from ~/nanocodec-data/anchors/aarch64/bin, core 3 under the board lock)
    ~/nanocodec-data/jetson-timed.sh taskset -c 3 python3 decoder_speed.py 5 jetson.json \\
        --nanocodec BIN --anchors DIR --avmdec AVMDEC --streams DIR [DIR ...]
    # x86 (this PC's timing core 14 under the host lock)
    tools/dev/host-timed.sh taskset -c 14 python3 tools/dev/decoder_speed.py 7 x86.json \\
        --nanocodec BIN --anchors ~/nanocodec-data/anchors/x86_64/bin --streams DIR [DIR ...]

Every decoder runs with one thread and writes nothing (`/dev/null`, `--noblit`): ours
`nanocodec decode`, FFmpeg n8.0.1 for H.264, HEVC, VP9 and VVC, dav1d (AV1), vpxdec
(VP9), VVdeC (VVC) and avmdec (AV2). Each round runs every command once, the order
reversed every other round; the record keeps each command's best process wall time
(seconds) and peak RSS (KiB, from wait4): {"stream|codec|decoder": {"t": s, "rss": KiB}}.
Stream files named `pan_*` are skipped. Exactness is checked separately (the decoders'
conformance suites and the landing conformance check), not here.

This is the board harness of the 2026-11-13 decoder close-out (`~/speed-audit/jdec2.py`),
with its paths as arguments.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

#: Container or file suffix -> codec, for streams not named `NAME.CODEC.ivf`.
EXTENSION_CODECS = {"264": "h264", "hevc": "hevc", "vvc": "vvc"}
#: Camera recordings (`cam150.*`) are raw HEVC.
CAMERA_PREFIX = "cam150"


def codec_of(name: str) -> str:
    parts = name.split(".")
    if parts[0] == CAMERA_PREFIX:
        return "hevc"
    if parts[-1] == "ivf":
        return parts[-2]
    return EXTENSION_CODECS[parts[-1]]


def ffmpeg(binary: Path, fmt: str | None, stream: str) -> list[str]:
    return [str(binary), "-v", "error", "-threads", "1", *(["-f", fmt] if fmt else []), "-i", stream,
            "-f", "rawvideo", "-y", "/dev/null"]


def references(codec: str, stream: str, anchors: Path, avmdec: Path | None) -> dict[str, list[str]]:
    ff = anchors / "ffmpeg-decode"
    commands = {
        "h264": {"ffmpeg": ffmpeg(ff, "h264", stream)},
        "hevc": {"ffmpeg": ffmpeg(ff, "hevc", stream)},
        "av1": {"dav1d": [str(anchors / "dav1d"), "--threads", "1", "--framedelay", "1", "--muxer",
                          "null", "-o", "/dev/null", "-i", stream]},
        "vp9": {"ffmpeg": ffmpeg(ff, None, stream),
                "vpxdec": [str(anchors / "vpxdec"), "--codec=vp9", "--threads=1", "--noblit", stream]},
        "vvc": {"vvdec": [str(anchors / "vvdecapp"), "-v", "0", "-t", "1", "-b", stream],
                "ffmpeg": ffmpeg(ff, "vvc", stream)},
        "av2": {},
    }[codec]
    if codec == "av2" and avmdec and avmdec.exists():
        commands["avmdec"] = [str(avmdec), "--threads=1", "--noblit", stream]
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rounds", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--nanocodec", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True, help="the reference decoders' bin dir")
    parser.add_argument("--avmdec", type=Path, help="default: avmdec in --anchors")
    parser.add_argument("--streams", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    avmdec = args.avmdec or args.anchors / "avmdec"
    streams = sorted((p for d in args.streams for p in d.iterdir() if not p.name.startswith("pan_")),
                     key=lambda p: p.name)
    jobs = []
    for stream in streams:
        codec = codec_of(stream.name)
        jobs.append((stream.name, codec, "ours", [str(args.nanocodec), "decode", "--codec", codec,
                                                   "--input", str(stream), "--output", "/dev/null"]))
        jobs += [(stream.name, codec, name, command)
                 for name, command in references(codec, str(stream), args.anchors, avmdec).items()]
    results: dict[str, dict] = {}
    for round_ in range(args.rounds):
        for name, codec, decoder, command in jobs if round_ % 2 == 0 else jobs[::-1]:
            start = time.perf_counter()
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            error = process.stderr.read()
            _, status, usage = os.wait4(process.pid, 0)
            elapsed = time.perf_counter() - start
            if os.waitstatus_to_exitcode(status):
                print(f"FAIL {name} {decoder}: {error[-200:]!r}", file=sys.stderr)
                continue
            entry = results.setdefault(f"{name}|{codec}|{decoder}", {"t": float("inf"), "rss": 0})
            entry["t"] = min(entry["t"], elapsed)
            entry["rss"] = max(entry["rss"], usage.ru_maxrss)
        print(f"round {round_} done", file=sys.stderr, flush=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
