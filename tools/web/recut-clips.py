#!/usr/bin/env python3
"""Re-cut sample clips (web/clips/<id>-{360p,720p}.mp4) from their masters, heavy enough
for the rates the demo codes them at.

The demo codes a clip at most at a third of its file's bitrate (CLAUDE.md "Testing": a
compressed source is coded at most at half its rate; a third leaves room), so a file
must reach MIN_SOURCE_RATIO times the highest pre-encoded stream rate on it
(web/streams-ref/manifest.json; a view of height h reads the smallest file at least h
tall) and LIVE_DEFAULT_RATIO times a live view's default target (web/panel.js
DEFAULT_TARGET_KBPS). Each clip in CLIPS is made as web/clips/ATTRIBUTION.md describes
the set's files: cut to the segment, every `step`th master frame, the source's crop,
Lanczos scaling, 8-bit 4:2:0 BT.709 limited range, ffmpeg/libx264 `-preset veryslow`,
`aq-mode=3`, a key frame every `key_frames` frames as the set's other files of that
source, H.264 High, MP4 with faststart, no audio, every kept frame once at 30 fps (as
tools/web/retime-clips.py retimed the set's 24 and 25 fps files). The CRF is the highest
from START_CRF down whose file reaches the rate (a lighter file than needed would hide
less; a heavier one costs page weight).

Masters are read from their public servers (only the segment's bytes for FTP and y4m;
an archive is downloaded whole into a temporary directory). Run it where downloads are
fast (the build box); the segment is decoded once per height to a temporary y4m, deleted
at the end. Prints each file's CRF, size and bitrate (ATTRIBUTION.md records the CRFs).

  tools/web/recut-clips.py [ID ...] [--height 360|720]     # default: every clip in CLIPS
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLIP_DIR = REPO / "web/clips"
MANIFEST = REPO / "web/streams-ref/manifest.json"
PLAYBACK_FPS = 30
VARIANT_HEIGHTS = (360, 720)
# A file's bitrate against the rates coded from it (owner, 2026-11-22).
MIN_SOURCE_RATIO = 3
LIVE_DEFAULT_RATIO = 2
LIVE_DEFAULT_KBPS = int(re.search(r"const DEFAULT_TARGET_KBPS = (\d+);", (REPO / "web/panel.js").read_text())[1])
# The set's lightest CRF (ATTRIBUTION.md): the search starts there.
START_CRF = 28
MIN_CRF = 10
SVT_2022 = "ftp://svtopencontent.svt.se/pub/svt_videotestsuite_natural_complexity/REC709/"
XIPH_Y4M = "https://media.xiph.org/video/derf/y4m/"
BLENDER_TOS = "https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip"


@dataclass(frozen=True)
class Recut:
    id: str  # web/clips.js id
    url: str  # the master (a .zip: `member` inside it)
    start_s: float  # segment start in the master
    frames: int  # frames kept
    step: int  # keep every step-th master frame (50 / 60 fps masters: 2)
    key_frames: int  # key interval of the set's files from this source
    crop: str = ""  # ffmpeg crop before scaling
    member: str = ""


XIPH = dict(start_s=0.0, frames=300, step=2, key_frames=60)
CLIPS = [
    Recut("svt-smoke-sauna", SVT_2022 + "smoke_sauna_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov", 10.0, 250, 2, 50),
    Recut("tos-canal", BLENDER_TOS, 258.0, 288, 1, 50, crop="1422:800", member="tears_of_steel_1080p.mov"),
    *(Recut(clip, XIPH_Y4M + master, **XIPH) for clip, master in [
        ("xiph-johnny", "Johnny_1280x720_60.y4m"), ("xiph-kristen-sara", "KristenAndSara_1280x720_60.y4m"),
        ("xiph-four-people", "FourPeople_1280x720_60.y4m"), ("xiph-vidyo1", "vidyo1_720p_60fps.y4m"),
        ("xiph-vidyo3", "vidyo3_720p_60fps.y4m"), ("xiph-vidyo4", "vidyo4_720p_60fps.y4m")]),
]


def variant(height: int) -> int:
    return next((h for h in VARIANT_HEIGHTS if h >= height), VARIANT_HEIGHTS[-1])


def min_kbps(clip: str, height: int) -> float:
    streams = json.loads(MANIFEST.read_text())["streams"]
    rates = [s["targetKbps"] for s in streams if s["clip"] == clip and variant(s["height"]) == height]
    return max(MIN_SOURCE_RATIO * max(rates, default=0), LIVE_DEFAULT_RATIO * LIVE_DEFAULT_KBPS)


def segment(clip: Recut, master: str, height: int, out: Path) -> None:
    """The segment's kept frames at `height`, as 8-bit 4:2:0 BT.709 limited range y4m."""
    filters = [f"select='not(mod(n,{clip.step}))'", f"setpts=N/({PLAYBACK_FPS}*TB)"]
    if clip.crop:
        filters.append(f"crop={clip.crop}")
    filters.append(f"scale=-2:{height}:flags=lanczos:out_color_matrix=bt709:out_range=tv,format=yuv420p")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{clip.start_s:g}", "-i", master, "-vf", ",".join(filters),
                    "-frames:v", str(clip.frames), "-r", str(PLAYBACK_FPS), "-strict", "-1", str(out)], check=True)


def encode(clip: Recut, source: Path, crf: int, out: Path) -> float:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source),
                    "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf), "-x264-params", "aq-mode=3",
                    "-g", str(clip.key_frames), "-profile:v", "high", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                    "-movflags", "+faststart", "-an", str(out)], check=True)
    return out.stat().st_size * 8 / (clip.frames / PLAYBACK_FPS) / 1000


def needs(clip: Recut, height: int) -> bool:
    out = CLIP_DIR / f"{clip.id}-{height}p.mp4"
    return out.stat().st_size * 8 / (clip.frames / PLAYBACK_FPS) / 1000 < min_kbps(clip.id, height)


def make(clip: Recut, heights: tuple[int, ...], scratch: Path) -> None:
    master = clip.url
    if clip.member and any(needs(clip, h) for h in heights):
        archive = scratch / Path(clip.url).name
        subprocess.run(["curl", "-sSfL", "-o", str(archive), clip.url], check=True)
        with zipfile.ZipFile(archive) as z:
            master = z.extract(clip.member, scratch)
        archive.unlink()
    for height in heights:
        wanted, out = min_kbps(clip.id, height), CLIP_DIR / f"{clip.id}-{height}p.mp4"
        if not needs(clip, height):
            print(f"{out.name}: kept (needs {wanted:.0f} kbps)", flush=True)
            continue
        source = scratch / f"{clip.id}-{height}p.y4m"
        segment(clip, master, height, source)
        for crf in range(START_CRF, MIN_CRF - 1, -1):
            kbps = encode(clip, source, crf, out)
            if kbps >= wanted:
                break
        else:
            raise RuntimeError(f"{out.name}: {kbps:.0f} kbps at CRF {MIN_CRF}, under {wanted:.0f}")
        source.unlink()
        print(f"{out.name}: CRF {crf}, {out.stat().st_size / 2**20:.2f} MiB, {kbps:.0f} kbps (needs {wanted:.0f})", flush=True)
    if master != clip.url:
        Path(master).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="clips to make (default: all in CLIPS)")
    parser.add_argument("--height", type=int, choices=VARIANT_HEIGHTS, help="only this variant")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nanocodec-recut-") as scratch:
        for clip in CLIPS:
            if not args.ids or clip.id in args.ids:
                make(clip, (args.height,) if args.height else VARIANT_HEIGHTS, Path(scratch))


if __name__ == "__main__":
    main()
