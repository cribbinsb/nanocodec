#!/usr/bin/env python3
"""Our browser module against ffmpeg.wasm, the official single-thread build of FFmpeg for the web.

The results site's "browser against ffmpeg.wasm" chart and decode rows (nceval.site
wasm_yardstick_data) read the record this writes. Method (docs/results/competitor-yardsticks.md
"Method"): one PC core, the same node, interleaved rounds (the order reversed every other round),
the best round counts.

- Encode: crowd_mot17_09 at 720p, 150 frames, 600 kbps with the same rate limits on both
  sides (ENCODE_SETTINGS, kept in the record): a 1 s buffer (600 kbit) and a peak of twice the
  target (the demo's), x264's -maxrate / -bufsize. Ours through tools/web/bench-runner.mjs
  (the demo's BrowserEncoder.encode, H.264 Medium and Fast, HEVC Fast, VP9 Medium, AV1 Fast);
  ffmpeg.wasm's x264 at veryfast and ultrafast at real-time settings (zero latency, no
  B-frames, no lookahead).
- Decode: DECODE_STREAMS, ours through bench-runner.mjs (`decode_ms_per_frame`: the
  BrowserDecoder.decode calls; the display hand-off is reported beside it), ffmpeg.wasm with
  `-f null -`; and OURS_ONLY_DECODE_STREAMS, the formats ffmpeg.wasm 0.12.10 has no decoder
  for (AV1, VVC, AV2), ours alone (compared with our native decoder by the results page).

Both compare the codec calls alone on both sides. Encode: ours the encode() calls
(`encode_ms_per_frame`; loading the frame into a SourceFrame is reported beside it, in
`ms_per_frame`), ffmpeg.wasm the sum of its encode_video calls (`encode_ms`). Decode: ours the decode() calls, ffmpeg.wasm the
sum of its decode_video calls as FFmpeg's -benchmark_all logs them (`decode_ms`), so neither
is charged module start-up, probing, demuxing or command parsing; site.py divides by the
frames. ffmpeg.wasm's start-up (`startup_ms`) and whole run (`ms`) are recorded beside it
(tools/web/ffmpeg-wasm-runner.cjs).

    tools/dev/host-timed.sh python3 tools/web/ffmpeg_wasm_yardstick.py --pkg PKG/relaxed \\
        --core ~/nanocodec-data/work/ffmpeg-wasm-yardstick/node_modules/@ffmpeg/core/dist/umd/ffmpeg-core.js \\
        --streams DIR --yuv crowd_mot17_09.yuv --cpu 15 --rounds 2 --json wasm.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "bench-runner.mjs"
FFMPEG_RUNNER = HERE / "ffmpeg-wasm-runner.cjs"
OURS, FFMPEG = "ours-wasm", "ffmpeg.wasm"
WIDTH, HEIGHT, FPS, FRAMES = 1280, 720, 30, 150
KBPS, BUFFER_MS, KEY_SECONDS = 600, 1000, 30
#: The demo's peak rate over its target (tools/web/bench.py PEAK_FACTOR), given to both.
PEAK_FACTOR = 2
#: Both encoders' rate limits, written into the record for the page to state.
ENCODE_SETTINGS = {
    "fps": FPS, "frames": FRAMES, "target_kbps": KBPS, "peak_kbps": PEAK_FACTOR * KBPS, "buffer_kbit": KBPS * BUFFER_MS // 1000,
    "ours": f"set_rate_control({KBPS}, {PEAK_FACTOR * KBPS}, {BUFFER_MS}): target kbps, peak kbps, "
            "buffer ms",
    "x264": f"-b:v {KBPS}k -maxrate {PEAK_FACTOR * KBPS}k -bufsize {KBPS * BUFFER_MS // 1000}k "
            "-tune zerolatency -bf 0 -rc-lookahead 0",
}
#: Our encode runs: name -> (codec, Fast preset).
OUR_ENCODES = {"h264": ("h264", False), "h264-fast": ("h264", True), "hevc-fast": ("hevc", True),
               "vp9": ("vp9", False), "av1-fast": ("av1", True)}
#: ffmpeg.wasm's x264 runs: name -> x264 preset.
X264_ENCODES = {"x264 veryfast": "veryfast", "x264 ultrafast": "ultrafast"}
#: Decode streams (file names in --streams, as site.py WASM_DECODE_STREAMS keys them) by codec.
DECODE_STREAMS = (("hevc", "axis720.cam.hevc"), ("h264", "crowd_mot17_09.600k.h264.264"),
                  ("hevc", "crowd_mot17_09.600k.hevc.hevc"), ("h264", "crowd1080.3000k.x264.h264.264"),
                  ("hevc", "crowd1080.3000k.ours.hevc.hevc"), ("vp9", "crowd1080.3000k.vpxenc.vp9.ivf"),
                  ("vp9", "crowd_mot17_09.600k.vp9.ivf"))
#: Streams of the formats @ffmpeg/core 0.12.10 cannot decode (FFmpeg 5.1.4 has no VVC or AV2
#: decoder, and the package carries no AV1 decoder library): ours alone.
OURS_ONLY_DECODE_STREAMS = (("av1", "crowd_mot17_09.600k.av1.ivf"), ("av1", "crowd1080.3000k.ours.av1.ivf"),
                            ("vvc", "crowd_mot17_09.600k.vvc.vvc"), ("vvc", "crowd1080.3000k.ours.vvc.vvc"),
                            ("av2", "crowd_mot17_09.600k.av2.ivf"), ("av2", "crowd1080.3000k.ours.av2.ivf"))


def run_json(cmd: list[str], cpu: int) -> dict:
    out = subprocess.run(["taskset", "-c", str(cpu), *cmd], check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def x264_args(preset: str) -> list[str]:
    rate = f"{KBPS}k"
    return ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "in",
            "-frames:v", str(FRAMES), "-c:v", "libx264", "-preset", preset, "-tune", "zerolatency",
            "-bf", "0", "-rc-lookahead", "0", "-g", str(KEY_SECONDS * FPS),
            "-b:v", rate, "-maxrate", f"{PEAK_FACTOR * KBPS}k", "-bufsize", f"{KBPS * BUFFER_MS // 1000}k",
            "-f", "h264", "out.264"]


def jobs(args) -> list[tuple[str, list[str]]]:
    """(record key, command) for every run of one round."""
    out = []
    for name, (codec, fast) in OUR_ENCODES.items():
        out.append((f"enc|{name}|{OURS}", [
            "node", str(RUNNER), "encode", "--pkg", str(args.pkg), "--codec", codec, "--yuv", str(args.yuv),
            "--width", str(WIDTH), "--height", str(HEIGHT), "--frames", str(FRAMES), "--fps", str(FPS),
            "--kbps", str(KBPS), "--peak-kbps", str(PEAK_FACTOR * KBPS), "--buffer-ms", str(BUFFER_MS),
            "--key-seconds", str(KEY_SECONDS), "--repeat", "1", *(["--fast"] if fast else [])]))
    for name, preset in X264_ENCODES.items():
        out.append((f"enc|{name}|{FFMPEG}", ["node", str(FFMPEG_RUNNER), str(args.core), str(args.yuv),
                                              "out.264", "--", *x264_args(preset)]))
    for codec, name in DECODE_STREAMS + OURS_ONLY_DECODE_STREAMS:
        stream = args.streams / name
        out.append((f"dec|{codec}|{name}|{OURS}", ["node", str(RUNNER), "decode", "--pkg", str(args.pkg),
                                                    "--codec", codec, "--stream", str(stream), "--repeat", "1"]))
        if (codec, name) in OURS_ONLY_DECODE_STREAMS:
            continue
        out.append((f"dec|{codec}|{name}|{FFMPEG}", ["node", str(FFMPEG_RUNNER), str(args.core), str(stream),
                                                      "none", "--", "-i", "in", "-f", "null", "-"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pkg", type=Path, required=True, help="our browser module (tools/web/build.sh)")
    parser.add_argument("--core", type=Path, required=True, help="@ffmpeg/core's dist/umd/ffmpeg-core.js")
    parser.add_argument("--streams", type=Path, required=True, help="directory with DECODE_STREAMS")
    parser.add_argument("--yuv", type=Path, required=True, help="crowd_mot17_09, 1280x720 I420")
    parser.add_argument("--cpu", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    # absolute: the node runners resolve paths (require) against their own directory
    for name in ("pkg", "core", "streams", "yuv"):
        setattr(args, name, getattr(args, name).resolve())
    core_version = json.loads((args.core.parents[2] / "package.json").read_text())["version"]
    todo = jobs(args)
    record: dict = {"date": datetime.date.today().isoformat(), "ffmpeg_core": core_version,
                    "node": subprocess.run(["node", "--version"], check=True, capture_output=True,
                                           text=True).stdout.strip(),
                    "cpu": args.cpu, "rounds": args.rounds, "encode_settings": ENCODE_SETTINGS, "load_start": list(os.getloadavg()), "res": {}}
    for index in range(args.rounds):
        for key, cmd in (todo if index % 2 == 0 else todo[::-1]):
            result = run_json(cmd, args.cpu)
            if result.get("rc", 0) != 0:
                sys.exit(f"{key}: ffmpeg.wasm exited {result['rc']}")
            record["res"].setdefault(key, []).append(result)
        # after every round, so a lost box keeps the rounds it finished
        record["load_end"] = list(os.getloadavg())
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1) + "\n")
        print(f"round {index + 1}/{args.rounds} done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
