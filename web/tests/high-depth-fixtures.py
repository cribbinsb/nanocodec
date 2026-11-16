#!/usr/bin/env python3
"""Make public FourPeople 720p deep fixtures and FFmpeg hashes, keeping no YUV.

SOURCE is the public CTC FourPeople_1280x720_60_E.yuv. FFmpeg converts its
8-bit samples to 10 bits before encoding: this tests deep syntax/storage,
not HDR. Only compressed bitstreams and reference hashes are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ENCODERS = {
    "av1": (
        "ivf",
        [
            "-c:v",
            "libsvtav1",
            "-preset",
            "11",
            "-crf",
            "35",
            "-g",
            "30",
            "-svtav1-params",
            "lp=4",
        ],
    ),
    "hevc": (
        "hevc",
        [
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-crf",
            "28",
            "-x265-params",
            "pools=2:frame-threads=1:bframes=0:keyint=30:min-keyint=30:scenecut=0:repeat-headers=1:aud=1:log-level=error",
        ],
    ),
    "h264": (
        "h264",
        [
            "-c:v",
            "libx264",
            "-profile:v",
            "high10",
            "-preset",
            "medium",
            "-crf",
            "28",
            "-threads",
            "1",
            "-x264-params",
            "bframes=0:keyint=30:min-keyint=30:scenecut=0:repeat-headers=1:aud=1",
        ],
    ),
    "vp9": (
        "ivf",
        [
            "-c:v",
            "libvpx-vp9",
            "-profile:v",
            "2",
            "-deadline",
            "realtime",
            "-cpu-used",
            "7",
            "-crf",
            "35",
            "-b:v",
            "0",
            "-g",
            "30",
            "-lag-in-frames",
            "0",
            "-threads",
            "2",
        ],
    ),
}
WIDTH, HEIGHT, DEPTH = 1280, 720, 10


def golden(stream, codec):
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(stream),
        "-pix_fmt",
        "yuv420p10le",
        "-f",
        "rawvideo",
        "-",
    ]
    frames = []
    frame_bytes = WIDTH * HEIGHT * 3
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        while data := process.stdout.read(frame_bytes):
            if len(data) != frame_bytes:
                raise RuntimeError("partial reference frame")
            display = (
                (np.frombuffer(data, dtype="<u2") >> (DEPTH - 8)).astype("u1").tobytes()
            )
            frames.append(
                {
                    "native_md5": hashlib.md5(data).hexdigest(),
                    "display_md5": hashlib.md5(display).hexdigest(),
                }
            )
        if process.wait():
            raise RuntimeError("reference decode failed")
    return {
        "codec": codec,
        "file": stream.name,
        "width": WIDTH,
        "height": HEIGHT,
        "depth": DEPTH,
        "stream_sha256": hashlib.sha256(stream.read_bytes()).hexdigest(),
        "reference_command": command,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--codecs", nargs="+", choices=ENCODERS, default=["av1", "hevc"]
    )
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for codec in args.codecs:
        container, options = ENCODERS[codec]
        stream = args.output / f"fourpeople-{codec}-10bit.{container}"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-threads",
            "2",
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuv420p",
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-framerate",
            "30",
            "-i",
            str(args.source),
            "-frames:v",
            str(args.frames),
            "-vf",
            "format=yuv420p10le",
            *options,
            "-an",
            "-f",
            container,
            str(stream),
        ]
        subprocess.run(command, check=True)
        entry = golden(stream, codec)
        if len(entry["frames"]) != args.frames:
            raise RuntimeError("reference frame count")
        entry["encode_command"] = command
        entries.append(entry)
    (args.output / "goldens.json").write_text(
        json.dumps({"streams": entries}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
