#!/usr/bin/env python3
"""Retime the 24 and 25 fps sample clips (web/clips/) to 30 fps, without re-encoding.

The page samples every source on a 30 Hz timeline (web/timeline.js). A 25 fps
clip sampled there repeats every fifth frame, which judders and is nearly free
to code, so it would flatter every encoder's bitrate and quality. Instead each
such clip plays every frame once at 30 fps (25 fps content runs 1.2x fast, 24 fps
1.25x). Only timestamps change: the H.264 stream is copied (`-c copy`, its VUI
timing set to 30 fps with the h264_metadata filter), so no picture changes.
Already-30 fps files are left alone, so the script can be rerun.

  tools/web/retime-clips.py
"""
import json
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

CLIP_DIR = Path(__file__).resolve().parents[2] / "web/clips"
TARGET_FPS = 30
# Native rates that are retimed (60 fps clips sample every second frame: no repeats).
RETIMED_RATES = (24, 25)
# MP4 track timescale: a multiple of 30 makes every frame time exact.
TIMESCALE = 30000
# H.264 VUI ticks: two per frame (one per field), H.264 Annex E.
TICKS_PER_FRAME = 2


def frame_rate(path: Path) -> Fraction:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return Fraction(json.loads(out)["streams"][0]["r_frame_rate"])


def retime(path: Path, rate: Fraction) -> None:
    with tempfile.TemporaryDirectory(dir=CLIP_DIR) as scratch:
        out = Path(scratch) / path.name
        subprocess.run(["ffmpeg", "-v", "error", "-itsscale", str(float(rate / TARGET_FPS)), "-i", str(path),
                        "-c", "copy", "-bsf:v", f"h264_metadata=tick_rate={TARGET_FPS * TICKS_PER_FRAME}",
                        "-video_track_timescale", str(TIMESCALE), "-movflags", "+faststart", "-an", str(out)], check=True)
        if frame_rate(out) != TARGET_FPS:
            raise RuntimeError(f"{path.name}: retimed file reports {frame_rate(out)} fps")
        out.replace(path)


def main() -> None:
    for path in sorted(CLIP_DIR.glob("*.mp4")):
        rate = frame_rate(path)
        if rate in RETIMED_RATES:
            retime(path, rate)
            print(f"{path.name}: {rate} -> {TARGET_FPS} fps ({TARGET_FPS / rate:.2f}x)")


if __name__ == "__main__":
    main()
