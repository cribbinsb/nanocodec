#!/usr/bin/env python3
"""Make the demo's MOTChallenge sample clips (web/clips/mot*-{360p,720p}.mp4).

Sources are the original JPEG frames of the MOT17 / MOT20 training sequences
(read only, under /mldata; CC BY-NC-SA 3.0, https://motchallenge.net). Each
clip is the hardest excerpt of its sequence (`--pick` ranks every window by
mean frame-to-frame change times mean spatial detail on 160-pixel-wide grey
frames) and is encoded like the other sample clips (web/clips/ATTRIBUTION.md):
Lanczos scaling, ffmpeg/libx264 `-preset veryslow`, the CRF in CLIPS,
`aq-mode=3`, a key frame every 2 s, H.264 High, yuv420p, BT.709 limited range
(the JPEGs are full-range BT.601), MP4 with faststart, no audio, every source
frame once at 30 fps (a 25 fps sequence plays 1.2x fast). A source that
is not 16:9 keeps its whole picture (MOT20-05: 1654x1080 -> 1102x720 and
552x360); the page letterboxes it. Nothing is written but the two mp4 files
per clip.

  tools/web/make-mot-clips.py            # make every clip in CLIPS
  tools/web/make-mot-clips.py --pick ID  # rank windows of that clip's sequence
"""
from __future__ import annotations

import argparse
import configparser
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CLIP_DIR = REPO / "web/clips"
MOT_ROOT = Path("/mldata/downloaded_datasets/other")
# The two variants every clip ships in (web/clips.js VARIANT_HEIGHTS).
VARIANT_HEIGHTS = (360, 720)
# Every clip plays each source frame once at the page's 30 Hz timeline
# (web/timeline.js SOURCE_FPS): a 25 fps sequence is retimed, 1.2x fast.
PLAYBACK_FPS = 30
KEY_INTERVAL_SECONDS = 2
# Grey thumbnails used to rank excerpts.
PICK_WIDTH, PICK_HEIGHT = 160, 90


@dataclass(frozen=True)
class MotClip:
    id: str  # web/clips.js id
    sequence: str  # e.g. MOT17/train/MOT17-11
    start: int  # first frame (1-based, as the JPEG names)
    frames: int
    crf: tuple[int, int]  # 360p, 720p


CLIPS = [
    MotClip("mot17-mall-walk", "MOT17/train/MOT17-11", 601, 300, (25, 28)),
    MotClip("mot17-night-crowd", "MOT17/train/MOT17-04", 211, 300, (25, 28)),
    MotClip("mot20-dense-crowd", "MOT20/train/MOT20-05", 2301, 300, (25, 28)),
]


def sequence_info(clip: MotClip) -> tuple[Path, float, int, int, int]:
    """Frame directory, frame rate, frame count and size of the clip's sequence."""
    ini = configparser.ConfigParser()
    ini.read(MOT_ROOT / clip.sequence / "seqinfo.ini")
    s = ini["Sequence"]
    return (MOT_ROOT / clip.sequence / s["imDir"], float(s["frameRate"]), int(s["seqLength"]),
            int(s["imWidth"]), int(s["imHeight"]))


def frames_input(directory: Path, fps: float, start: int) -> list[str]:
    return ["-framerate", f"{fps:g}", "-start_number", str(start), "-i", str(directory / "%06d.jpg")]


def pick(clip: MotClip) -> None:
    """Print the windows of `clip.frames` frames, hardest first."""
    directory, fps, length, _, _ = sequence_info(clip)
    raw = subprocess.run(["ffmpeg", "-v", "error", *frames_input(directory, fps, 1), "-vf",
                          f"scale={PICK_WIDTH}:{PICK_HEIGHT},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    grey = np.frombuffer(raw, np.uint8).reshape(-1, PICK_HEIGHT, PICK_WIDTH).astype(np.float32)
    motion = np.abs(np.diff(grey, axis=0)).mean(axis=(1, 2))
    detail = np.abs(np.diff(grey, axis=2)).mean(axis=(1, 2))[1:]
    score = motion * detail
    windows = [(score[s:s + clip.frames - 1].mean(), s + 1) for s in range(0, length - clip.frames + 1, int(fps))]
    for value, start in sorted(windows, reverse=True)[:5]:
        print(f"{clip.id}: start frame {start} ({(start - 1) / fps:.1f} s), score {value:.1f}")


def make(clip: MotClip) -> None:
    directory, _, _, width, height = sequence_info(clip)
    for variant, crf in zip(VARIANT_HEIGHTS, clip.crf):
        # Whole picture at the variant's height, even width (16:9 sources give 640/1280).
        out_width = round(width * variant / height / 2) * 2
        out = CLIP_DIR / f"{clip.id}-{variant}p.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", *frames_input(directory, PLAYBACK_FPS, clip.start), "-frames:v", str(clip.frames),
            "-vf", f"scale={out_width}:{variant}:flags=lanczos:in_range=pc:in_color_matrix=bt601:"
                   "out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf), "-x264-params", "aq-mode=3",
            "-g", str(KEY_INTERVAL_SECONDS * PLAYBACK_FPS), "-profile:v", "high", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-movflags", "+faststart", "-an", str(out)], check=True)
        print(f"{out.name}: {out.stat().st_size / 2**20:.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pick", metavar="ID", help="rank excerpt windows of this clip's sequence")
    args = parser.parse_args()
    if args.pick:
        pick(next(c for c in CLIPS if c.id == args.pick))
        return
    for clip in CLIPS:
        make(clip)


if __name__ == "__main__":
    main()
