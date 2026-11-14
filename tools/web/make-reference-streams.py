#!/usr/bin/env python3
"""Regenerate the browser demo's pre-encoded reference streams (web/streams-ref/).

Each stream is a sample clip coded offline by a third-party encoder in the
exact real-time configuration the bitrate-vs-speed chart uses (the families of
eval/nceval/cbr.py: the same command builder, so the flags cannot drift),
for a visitor to compare live against nanocodec. The grid (GRID) is a sparse,
deliberately varied set: content, encoder and preset, resolution, frame rate and
bitrate all vary, and every entry is one the other view can be set to live
(a resolution, frame rate and target the page offers; web/tests/reference.test.js).

Source frames are the page's own (web/capture-worker.js): the hosted mp4 the
page would read at that height (clips.js `clipUrl`) is decoded by ffmpeg to its
yuv420p planes (limited-range BT.709, as the browser's decoder gives them: H.264
decoding is exact), passed through at the file's size, or scaled and
letterboxed into other sizes exactly as nc-web `SourceFrame` does
(source_frames.py). The clip is sampled on the page's 30 Hz tick grid
(web/timeline.js): tick c shows source frame floor(c * F / 30) of an F fps clip
(the hosted clips are 30 fps, tools/web/retime-clips.py, or 60 fps, every second
frame), exactly as the live view sees them; a stream at 30/N fps codes every Nth
tick. A live view and a pre-encoded one therefore code the same pixels
(web/tests/capture-identity.py checks the browser's against this).

Every stream is decoded by the codec's reference decoder (ffmpeg n8.0.1, dav1d,
vpxdec) and by our native decoder; the run fails unless both agree on every
frame. The per-frame MD5s go to hashes.json, which the WASM decoder test
(web/tests/reference-streams-wasm.mjs) checks. H.264/HEVC access units are
stored as IVF frames (FourCC H264/HEVC) so the page reads one container.

Streams made elsewhere (SUPPLIED: the owner's bodycam SoC encodes) cannot be
regenerated: their files are pinned by hash in SUPPLIED, checked by the same
decoders, and described from their provenance (no command line).

Usage: tools/web/make-reference-streams.py --nanocodec BIN [--only NAME...]
A run remakes only entries that are new or out of date (source file, command or
stream file changed); the rest keep their manifest records and checks.
       tools/web/make-reference-streams.py --table
Writes web/streams-ref/{*.ivf,manifest.json,hashes.json} and the table of
streams in docs/results/reference-streams.md (`--table` rewrites only that,
from the manifest). The I420 input of each
encode is a temporary file under the work directory, deleted as soon as the
encode ends; no decoded YUV is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from nceval.cbr import BUFFER_MS, Family, preset_anchor  # noqa: E402
from nceval.clips import Clip  # noqa: E402
from nceval.codecs import Codec  # noqa: E402
from nceval.config import load_paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_frames import split_i420, view_frame  # noqa: E402

WEB = REPO / "web"
OUT = WEB / "streams-ref"
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
ANCHORS = DATA / "anchors/x86_64/bin"
# The openh264 driver with bitrate control (eval/nceval/cbr.py --openh264-harness).
OPENH264_RC_HARNESS = DATA / "work/ratecontrol/bin/openh264-harness"
VERSIONS_ENV = REPO / "tools/anchors/versions.env"
TABLE = REPO / "docs/results/reference-streams.md"

# web/timeline.js SOURCE_FPS: the page samples every source on a 30 Hz tick grid.
SOURCE_FPS = 30
# The hosted renditions, read from web/clips.js VARIANT_HEIGHTS (one list for both):
# a view of height h reads the smallest at least h tall (clips.js clipUrl, `clip_file`).
VARIANT_HEIGHTS = tuple(int(h) for h in re.search(r"const VARIANT_HEIGHTS = \[([\d, ]+)\]",
                                                  (WEB / "clips.js").read_text()).group(1).split(","))
# The chart's key interval (eval/nceval/cbr.py DEFAULT_INTRA_PERIOD): 10 s.
KEY_INTERVAL_SECONDS = 10
# The supplied bodycam SoC streams' key interval (SUPPLIED): frames 0 and 150 of 300 at 30 fps.
SUPPLIED_KEY_SECONDS = 5
IVF_HEADER = struct.Struct("<4sHH4sHHIII4x")
IVF_FRAME = struct.Struct("<IQ")
IVF_FOURCC = {Codec.H264: b"H264", Codec.HEVC: b"HEVC", Codec.AV1: b"AV01", Codec.VP9: b"VP90"}
# Pixel formats the page can compare (web/view-settings.js): 8-bit 4:2:0 only.
PIXEL_FORMATS = {"yuv420p": {"bitDepth": 8, "chroma": "4:2:0"}}
PAGE_CODEC = {Codec.H264: "h264", Codec.HEVC: "hevc", Codec.AV1: "av1", Codec.VP9: "vp9"}

# How source frames are made (`write_source`): a record made otherwise is out of date.
SOURCE_PIXELS = "decoded planes, nc-web SourceFrame placement (tools/web/source_frames.py)"


@dataclass(frozen=True)
class Encoder:
    """A reference encoder configuration: a chart family, or a named variant of one."""
    key: str
    label: str
    family: str  # eval/nceval/cbr.py family name
    tool: str  # binary whose version is recorded
    pinned: str = ""  # versions.env prefix of the pinned source
    chart: bool = True  # False: not one of the chart's configurations
    rewrite: tuple[tuple[str, str], ...] = ()  # flag replacements for a variant


ENCODERS = {e.key: e for e in [
    Encoder("x264-medium", "x264 medium", "x264-medium", "x264", "X264"),
    Encoder("x264-veryfast", "x264 veryfast", "x264-veryfast", "x264", "X264"),
    Encoder("openh264", "openh264 (real-time, medium complexity)", "openh264-baseline", "openh264", "OPENH264"),
    Encoder("x265-medium", "x265 medium", "x265-medium", "x265", "X265"),
    Encoder("x265-superfast", "x265 superfast", "x265-superfast", "x265", "X265"),
    Encoder("svtav1-m12", "SVT-AV1 M12 --rtc", "svtav1-m12", "SvtAv1EncApp", "SVTAV1"),
    Encoder("svtav1-m8", "SVT-AV1 M8 --rtc", "svtav1-m8", "SvtAv1EncApp", "SVTAV1"),
    Encoder("aom-rt8", "libaom AV1 realtime cpu-used 8", "libaom-av1-cpu8", "aomenc", "LIBAOM"),
    Encoder("vpx-rt7", "libvpx VP9 realtime cpu-used 7", "libvpx-vp9-cpu7", "vpxenc", "LIBVPX"),
    Encoder("vpx-rt9", "libvpx VP9 realtime cpu-used 9", "libvpx-vp9-cpu9", "vpxenc", "LIBVPX"),
    # libvpx's good-quality mode, one pass, no lag: not a chart configuration.
    Encoder("vpx-good4", "libvpx VP9 good cpu-used 4 (one pass, no lag)", "libvpx-vp9-cpu4", "vpxenc", "LIBVPX",
            chart=False, rewrite=(("--rt", "--good"),)),
    Encoder("nvenc-h264", "NVENC H.264 p4 low latency (RTX 5090)", "nvenc-h264", "ffmpeg"),
    Encoder("nvenc-hevc", "NVENC HEVC p4 low latency (RTX 5090)", "nvenc-hevc", "ffmpeg"),
    Encoder("nvenc-av1", "NVENC AV1 p4 low latency (RTX 5090)", "nvenc-av1", "ffmpeg"),
]}


@dataclass(frozen=True)
class Entry:
    clip: str  # web/clips.js id
    encoder: str  # ENCODERS key
    width: int
    height: int
    fps: int
    kbps: int
    key_seconds: int = KEY_INTERVAL_SECONDS  # another interval only to match a supplied stream's

    @property
    def key_interval(self) -> int:
        return self.key_seconds * self.fps

    @property
    def name(self) -> str:
        return f"{self.clip}-{self.encoder}-{self.width}x{self.height}-{self.fps}fps-{self.kbps}k"


# Public-domain (Xiph), CC BY (Blender, SVT 2022) and CC BY-NC-SA (MOTChallenge,
# whose notice travels in the manifest: NOTICES) clips, and Netflix FoodMarket only
# (CC BY-NC-ND 4.0, by the owner's decision, with its notice). No SVT Multi Format.
# Rates span a starved, a mid and a comfortable rung.
GRID = [
    # Johnny 720p30 300 kbps: H.264 (x264, openh264), AV1 and VP9 at one rate.
    Entry("xiph-johnny", "x264-medium", 1280, 720, 30, 300),
    Entry("xiph-johnny", "openh264", 1280, 720, 30, 300),
    Entry("xiph-johnny", "svtav1-m12", 1280, 720, 30, 300),
    Entry("xiph-johnny", "vpx-rt7", 1280, 720, 30, 300),
    Entry("xiph-johnny", "openh264", 640, 360, 15, 150),
    # FourPeople 480p30 400 kbps: H.264, HEVC, VP9 and AV1 at one rate.
    Entry("xiph-four-people", "x264-medium", 854, 480, 30, 400),
    Entry("xiph-four-people", "x265-medium", 854, 480, 30, 400),
    Entry("xiph-four-people", "vpx-rt7", 854, 480, 30, 400),
    Entry("xiph-four-people", "svtav1-m12", 854, 480, 30, 400),
    # KristenAndSara 720p30 600 kbps across H.264 (NVENC, x264) and VP9; openh264 starved.
    Entry("xiph-kristen-sara", "nvenc-h264", 1280, 720, 30, 600),
    Entry("xiph-kristen-sara", "x264-veryfast", 1280, 720, 30, 600),
    Entry("xiph-kristen-sara", "vpx-rt9", 1280, 720, 30, 600),
    Entry("xiph-kristen-sara", "openh264", 640, 360, 30, 100),
    # vidyo4 360p30 starved at 100 kbps: x264, openh264, VP9.
    Entry("xiph-vidyo4", "x264-veryfast", 640, 360, 30, 100),
    Entry("xiph-vidyo4", "openh264", 640, 360, 30, 100),
    Entry("xiph-vidyo4", "vpx-rt7", 640, 360, 30, 100),
    # vidyo3 720p15 250 kbps: VP9, x264, openh264, libaom.
    Entry("xiph-vidyo3", "vpx-rt9", 1280, 720, 15, 250),
    Entry("xiph-vidyo3", "x264-medium", 1280, 720, 15, 250),
    Entry("xiph-vidyo3", "openh264", 1280, 720, 15, 250),
    Entry("xiph-vidyo3", "aom-rt8", 1280, 720, 15, 250),
    # Tears of Steel canal: 720p30 1200 kbps (AV1, HEVC, VP9 good), 360p10 150 kbps (AV1, H.264).
    Entry("tos-canal", "svtav1-m8", 1280, 720, 30, 1200),
    Entry("tos-canal", "x265-superfast", 1280, 720, 30, 1200),
    Entry("tos-canal", "vpx-good4", 1280, 720, 30, 1200),
    Entry("tos-canal", "aom-rt8", 640, 360, 10, 150),
    Entry("tos-canal", "openh264", 640, 360, 10, 150),
    # Waterfall 720p30 2000 kbps: H.264, AV1 (NVENC), VP9.
    Entry("svt-waterfall", "x264-medium", 1280, 720, 30, 2000),
    Entry("svt-waterfall", "nvenc-av1", 1280, 720, 30, 2000),
    Entry("svt-waterfall", "vpx-rt7", 1280, 720, 30, 2000),
    # Smoke sauna 480p30 600 kbps: VP9 good, x264, SVT-AV1; VP9 360p15 200 kbps.
    Entry("svt-smoke-sauna", "vpx-good4", 854, 480, 30, 600),
    Entry("svt-smoke-sauna", "x264-medium", 854, 480, 30, 600),
    Entry("svt-smoke-sauna", "svtav1-m12", 854, 480, 30, 600),
    Entry("svt-smoke-sauna", "vpx-rt9", 640, 360, 15, 200),
    # Midnight sun 360p30 400 kbps: HEVC (NVENC, x265), VP9; openh264 216p15 100 kbps.
    Entry("svt-midnight-sun", "nvenc-hevc", 640, 360, 30, 400),
    Entry("svt-midnight-sun", "x265-medium", 640, 360, 30, 400),
    Entry("svt-midnight-sun", "vpx-rt7", 640, 360, 30, 400),
    Entry("svt-midnight-sun", "openh264", 384, 216, 15, 100),
    # Big Buck Bunny tree: 216p15 100 kbps (AV1), 480p30 500 kbps (openh264, x264, VP9 good).
    Entry("bbb-tree", "svtav1-m12", 384, 216, 15, 100),
    Entry("bbb-tree", "openh264", 854, 480, 30, 500),
    Entry("bbb-tree", "x264-veryfast", 854, 480, 30, 500),
    Entry("bbb-tree", "vpx-good4", 854, 480, 30, 500),
    # MOT20-05: ~200 people at night, 720p30 1500 kbps across AV1, H.264 (x264, openh264), VP9; starved 360p30.
    Entry("mot20-dense-crowd", "svtav1-m12", 1280, 720, 30, 1500),
    Entry("mot20-dense-crowd", "x264-medium", 1280, 720, 30, 1500),
    Entry("mot20-dense-crowd", "vpx-rt7", 1280, 720, 30, 1500),
    Entry("mot20-dense-crowd", "openh264", 1280, 720, 30, 1500),
    Entry("mot20-dense-crowd", "x264-medium", 640, 360, 30, 150),
    # MOT17-11: walking through a mall, 720p30 1000 kbps; openh264 480p15 400 kbps.
    Entry("mot17-mall-walk", "svtav1-m12", 1280, 720, 30, 1000),
    Entry("mot17-mall-walk", "x264-medium", 1280, 720, 30, 1000),
    Entry("mot17-mall-walk", "vpx-rt7", 1280, 720, 30, 1000),
    Entry("mot17-mall-walk", "openh264", 854, 480, 15, 400),
    # MOT17-04: night street crowd, static camera, 720p30 800 kbps; VP9 360p15 250 kbps.
    Entry("mot17-night-crowd", "svtav1-m12", 1280, 720, 30, 800),
    Entry("mot17-night-crowd", "x264-medium", 1280, 720, 30, 800),
    Entry("mot17-night-crowd", "openh264", 1280, 720, 30, 800),
    Entry("mot17-night-crowd", "vpx-rt9", 640, 360, 15, 250),
    # Netflix FoodMarket (owner's decision for this clip only): software encoders at the
    # supplied bodycam SoC streams' size, rate, bitrates and key interval (SUPPLIED: a key
    # every 5 s, frames 0 and 150), so the comparison is like for like.
    *(Entry("netflix-food-market", encoder, 1280, 720, 30, kbps, key_seconds=SUPPLIED_KEY_SECONDS)
      for encoder in ("x264-medium", "openh264", "x265-medium") for kbps in (1000, 1500, 2000)),
]

# Licence notices that travel with every stream of a clip (manifest `notice`, shown in
# the demo's panel; web/clips/ATTRIBUTION.md). MOTChallenge is CC BY-NC-SA 3.0.
MOT_NOTICE = ("MOTChallenge {set} ({authors}), CC BY-NC-SA 3.0, https://motchallenge.net. This stream re-encodes "
              "the demo's hosted excerpt (cut, scaled, retimed to 30 fps, re-encoded) for non-commercial codec "
              "comparison, and is offered under the same CC BY-NC-SA 3.0 licence.")
NETFLIX_FOOD_MARKET_NOTICE = (
    "Copyright (C) NETFLIX INC. 2015 · Netflix El Fuente \"Food Market\" · CC BY-NC-ND 4.0, "
    "https://creativecommons.org/licenses/by-nc-nd/4.0/. This stream is an encode of the demo's hosted excerpt, "
    "made for non-commercial codec comparison; the picture is otherwise unchanged.")
NOTICES = {
    "netflix-food-market": NETFLIX_FOOD_MARKET_NOTICE,
    "mot17-mall-walk": MOT_NOTICE.format(set="MOT17", authors="Milan et al."),
    "mot17-night-crowd": MOT_NOTICE.format(set="MOT17", authors="Milan et al."),
    "mot20-dense-crowd": MOT_NOTICE.format(set="MOT20", authors="Dendorfer et al."),
}

# How each kind of encoder controls its rate (manifest `rateControl`, shown in the
# panel): the chart's real-time CBR settings (eval/nceval/cbr.py `Family.command`).
RATE_CONTROL = {
    "x264": "ABR with a 1 s VBV at the target (the chart's CBR)",
    "x265": "ABR with a 1 s VBV at the target (the chart's CBR)",
    "openh264": "openh264 real-time bitrate control at the target (the chart's)",
    "svt": "CBR (--rc 2), 1 s buffer",
    "vpx": "CBR (--end-usage=cbr), 1 s buffer",
    "aom": "CBR (--end-usage=cbr), 1 s buffer",
    "nvenc": "CBR, peak at the target, 1 s buffer",
}


@dataclass(frozen=True)
class Supplied:
    """A stream made outside this repository and supplied as an IVF file in
    web/streams-ref/: nothing here can regenerate it, so the file is pinned by its
    hash, decoded and checked like the others, and described from its provenance."""
    clip: str
    codec: Codec
    width: int
    height: int
    fps: int
    kbps: int  # the supplier's target
    sha256: str
    # PSNR-YUV (6:1:1, mean over frames) against the frames the encoder was given,
    # minus the same against the demo's current source frames, in dB.
    psnr_gap_db: float

    @property
    def name(self) -> str:
        return f"{self.clip}-bwc-soc-{PAGE_CODEC[self.codec]}-{self.width}x{self.height}-{self.fps}fps-{self.kbps}k"


# The owner's bodycam SoC streams of FoodMarket (v2, 2026-11-14; README supplied with them):
# the SoC's hardware encoder at the camera's recording settings with a fix to how its rate
# control picks the first key frame's quantiser (v1 had a fixed QP 26 key at every rate),
# EVBR rate control, a key frame every 5 s, P frames only, H.264 High / HEVC Main. Their
# PSNR-Y against the demo's former source frames (BT.601 via RGB, make-reference-streams.py
# before e41b52fa) equals the README's to 0.01 dB, so they were fed those frames again,
# not the current ones; the gap was measured against both (docs/handoff/browser-demo.md).
BWC_SOC = {
    "label": "BWC SOC",
    "provenance": "Supplied by the project owner: the hardware encoder of a production body-worn-camera SoC at the "
                  "camera's recording settings, with a fix to how its rate control picks the first key frame's "
                  "quantiser (P frames only, a key frame every 5 s). Not made by any tool in this repository; vendor "
                  "and model are not named.",
    "version": "hardware encoder of a production body-worn-camera SoC (vendor and model not named)",
    "rateControl": "the camera's EVBR recording rate control (not CBR)",
    "command": "not available (hardware encoder)",
}
SUPPLIED_SOURCE_NOTE = ("Encoded from an earlier RGB-converted copy of the demo's frames (limited-range BT.601 via RGB); "
                        "its PSNR against the frames the demo shows now is about {gap:.1f} dB lower than against its "
                        "own input.")
SUPPLIED = [
    Supplied("netflix-food-market", Codec.H264, 1280, 720, 30, 1000,
             "9d0ce5b075f00bd771ac0b23b108579e6658f75bd702ad61193caef6e9cf621a", 0.37),
    Supplied("netflix-food-market", Codec.H264, 1280, 720, 30, 1500,
             "190157e0d217b916b2eec1c29f64d7f7c94af2383165956c9e0ab508f5ce2b49", 0.50),
    Supplied("netflix-food-market", Codec.H264, 1280, 720, 30, 2000,
             "dca953101d27dccee2031f28c0e7d790de1fd9e74668f69b29dfe8788e548408", 0.62),
    Supplied("netflix-food-market", Codec.HEVC, 1280, 720, 30, 1000,
             "11b015dc548c370f4152bc478cdb3e4f263e44e88abae3aff044cb0b2575b299", 0.36),
    Supplied("netflix-food-market", Codec.HEVC, 1280, 720, 30, 1500,
             "ec0cd6f874131a220e52cb19c20d0596a07e8e1fa231dd5c97ea4512902ddda0", 0.46),
    Supplied("netflix-food-market", Codec.HEVC, 1280, 720, 30, 2000,
             "aef5bc5d184de34da5b307f3de51065b2809cdd82c0e3abd6e03be789f229c61", 0.55),
]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, **kwargs)


def clip_file(clip: str, height: int) -> Path:
    variant = next((h for h in VARIANT_HEIGHTS if h >= height), VARIANT_HEIGHTS[-1])
    return WEB / "clips" / f"{clip}-{variant}p.mp4"


def probe(path: Path) -> tuple[int, int, float, int]:
    """Width, height, frame rate and frame count of the clip file."""
    out = run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries",
               "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)], text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return s["width"], s["height"], num / den, int(s["nb_read_frames"])


def write_source(entry: Entry, path: Path) -> tuple[int, int]:
    """Write the entry's I420 frames to `path`; returns (frames, loop ticks)."""
    source = clip_file(entry.clip, entry.height)
    source_width, source_height, clip_fps, clip_frames = probe(source)
    loop_ticks = round(clip_frames * SOURCE_FPS / clip_fps)
    step = SOURCE_FPS // entry.fps
    if SOURCE_FPS % entry.fps or loop_ticks % step:
        raise ValueError(f"{entry.name}: {entry.fps} fps does not divide the {loop_ticks}-tick loop")
    wanted = [int(c * clip_fps / SOURCE_FPS + 1e-9) for c in range(0, loop_ticks, step)]
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(source), "-frames:v", str(wanted[-1] + 1),
                                "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    size = source_width * source_height + 2 * ((source_width + 1) // 2) * ((source_height + 1) // 2)
    view = (entry.width, entry.height)
    with open(path, "wb") as out:
        frame, index = None, -1
        for wanted_index in wanted:
            while index < wanted_index:
                data = decoder.stdout.read(size)
                if len(data) != size:
                    raise RuntimeError(f"{source}: ended at frame {index}")
                frame, index = data, index + 1
            same_size = (source_width, source_height) == view
            out.write(frame if same_size else view_frame(split_i420(frame, source_width, source_height), view))
    decoder.stdout.close()
    decoder.wait()
    return len(wanted), loop_ticks


def family(encoder: Encoder) -> Family:
    if encoder.key == "openh264":
        return Family("openh264-baseline", Codec.H264, OPENH264_RC_HARNESS, kind="openh264")
    return preset_anchor(encoder.family, load_paths())


def command(encoder: Encoder, entry: Entry, frames: int, yuv: Path, out: Path) -> list[str]:
    clip = Clip(name=entry.clip, yuv=yuv, width=entry.width, height=entry.height, fps=entry.fps,
                frames=frames, source_kbps=0.0, content_class="", split="")
    cmd = family(encoder).command(clip, frames, entry.key_interval, out, None, kbps=entry.kbps)
    rewrites = dict(encoder.rewrite)
    return [rewrites.get(arg, arg) for arg in cmd]


def pinned_source(prefix: str) -> dict:
    if not prefix:
        return {}
    env = dict(re.findall(r"^(\w+)=(\S+)$", VERSIONS_ENV.read_text(), re.M))
    return {"url": env[f"{prefix}_URL"], "ref": env[f"{prefix}_REF"], "commit": env[f"{prefix}_COMMIT"]}


def version(encoder: Encoder, binary: str) -> dict:
    """The binary's own version string, the pinned source and (NVENC) the driver."""
    probes = {"x264": ["--version"], "x265": ["--version"], "SvtAv1EncApp": ["--version"],
              "ffmpeg": ["-hide_banner", "-version"], "aomenc": ["--help"], "vpxenc": ["--help"]}
    info = {"binary": binary.replace(str(Path.home()), "~"), "source": pinned_source(encoder.pinned)}
    if encoder.tool in probes:
        r = subprocess.run([binary, *probes[encoder.tool]], capture_output=True, text=True)
        text = (r.stdout + r.stderr).splitlines()
        pattern = {"aomenc": "AOMedia Project AV1 Encoder", "vpxenc": "WebM Project VP9 Encoder"}.get(encoder.tool)
        lines = [line for line in text if pattern in line] if pattern else text[:2]
        tidy = (" ".join(line.removeprefix("x265 [info]: ").split()) for line in lines)
        info["version"] = " / ".join(line.removeprefix("vp9 - ").removeprefix("av1 - ") for line in tidy if line)
    else:
        info["version"] = f"openh264 {pinned_source(encoder.pinned)['ref']} (eval openh264-harness driver)"
    if encoder.tool == "ffmpeg":
        driver = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True)
        info["gpu"] = driver.stdout.strip()
    return info


def access_units(stream: bytes, codec: Codec) -> list[bytes]:
    """Split an Annex-B stream into access units (H.264 §7.4.1.2.3, HEVC §7.4.2.4.4)."""
    starts = [m.start() for m in re.finditer(b"\x00\x00\x01", stream)]
    nals = []
    for i, at in enumerate(starts):
        begin = at - 1 if at and stream[at - 1] == 0 else at
        end = starts[i + 1] if i + 1 < len(starts) else len(stream)
        if i + 1 < len(starts) and stream[end - 1] == 0:
            end -= 1
        nals.append((begin, at + 3, end))
    units, current, has_picture = [], [], False
    for begin, header, end in nals:
        if codec == Codec.H264:
            kind = stream[header] & 0x1F
            vcl, opens = kind in (1, 5), kind in (6, 7, 8, 9)
            first = vcl and stream[header + 1] & 0x80  # first_mb_in_slice == 0
        else:
            kind = (stream[header] >> 1) & 0x3F
            vcl, opens = kind < 32, kind in (32, 33, 34, 35, 39)
            first = vcl and stream[header + 2] & 0x80  # first_slice_segment_in_pic_flag
        if has_picture and (opens or first):
            units.append(b"".join(current))
            current, has_picture = [], False
        current.append(stream[begin:end])
        has_picture |= vcl
    if current:
        units.append(b"".join(current))
    return units


def ivf_packets(data: bytes) -> list[bytes]:
    packets, at = [], IVF_HEADER.size
    while at < len(data):
        size, _ = IVF_FRAME.unpack_from(data, at)
        packets.append(data[at + IVF_FRAME.size:at + IVF_FRAME.size + size])
        at += IVF_FRAME.size + size
    return packets


def write_ivf(path: Path, codec: Codec, entry: Entry, packets: list[bytes]) -> None:
    with open(path, "wb") as f:
        f.write(IVF_HEADER.pack(b"DKIF", 0, IVF_HEADER.size, IVF_FOURCC[codec], entry.width, entry.height,
                                entry.fps, 1, len(packets)))
        for index, packet in enumerate(packets):
            f.write(IVF_FRAME.pack(len(packet), index))
            f.write(packet)


def frame_hashes(cmd: list[str], output: Path, frame_bytes: int) -> list[str]:
    """Per-frame MD5s of a decoder's I420 output; the output file is removed."""
    try:
        run(cmd)
        hashes = []
        with open(output, "rb") as f:
            while chunk := f.read(frame_bytes):
                hashes.append(hashlib.md5(chunk).hexdigest())
        return hashes
    finally:
        output.unlink(missing_ok=True)


def verify(stream: Path, codec: Codec, entry: Entry, nanocodec: Path, scratch: Path) -> list[str]:
    """Reference and native decodes must agree on every frame; returns the hashes."""
    frame_bytes = entry.width * entry.height * 3 // 2
    out = scratch / "decoded.yuv"
    reference = {Codec.AV1: [str(ANCHORS / "dav1d"), "-q", "-i", str(stream), "-o", str(out)],
                 Codec.VP9: [str(ANCHORS / "vpxdec"), "--i420", "-o", str(out), str(stream)]}.get(
        codec, [str(ANCHORS / "ffmpeg-decode"), "-v", "error", "-y", "-i", str(stream), "-f", "rawvideo",
                "-pix_fmt", "yuv420p", str(out)])
    expected = frame_hashes(reference, out, frame_bytes)
    ours = frame_hashes([str(nanocodec), "decode", "--codec", codec.value, "--input", str(stream),
                         "--output", str(out)], out, frame_bytes)
    if ours != expected:
        first = next((i for i, (a, b) in enumerate(zip(ours, expected)) if a != b), min(len(ours), len(expected)))
        raise RuntimeError(f"{stream}: our decoder differs from the reference decoder at frame {first}"
                           f" ({len(ours)} vs {len(expected)} frames)")
    return expected


def make(entry: Entry, nanocodec: Path, scratch: Path) -> tuple[dict, list[str]]:
    encoder = ENCODERS[entry.encoder]
    fam = family(encoder)
    raw = scratch / f"{entry.name}.raw"
    yuv = scratch / "source.yuv"
    try:
        frames, loop_ticks = write_source(entry, yuv)
        cmd = command(encoder, entry, frames, yuv, raw)
        run(cmd)
    finally:
        yuv.unlink(missing_ok=True)
    data = raw.read_bytes()
    packets = ivf_packets(data) if data[:4] == b"DKIF" else access_units(data, fam.codec)
    if len(packets) != frames:
        raise RuntimeError(f"{entry.name}: {len(packets)} coded frames, expected {frames}")
    target = OUT / f"{entry.name}.ivf"
    write_ivf(target, fam.codec, entry, packets)
    # The native decoder reads Annex-B for H.264/HEVC: verify the encoder's own file.
    checked = raw if fam.codec in (Codec.H264, Codec.HEVC) else target
    hashes = verify(checked, fam.codec, entry, nanocodec, scratch)
    pixel_format = run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=pix_fmt", "-of",
                        "csv=p=0", str(checked)], text=True).stdout.strip()
    if pixel_format not in PIXEL_FORMATS:
        raise RuntimeError(f"{entry.name}: {pixel_format}, the page codes 8-bit 4:2:0 only")
    coded = sum(map(len, packets))
    seconds = frames / entry.fps
    record = {
        "id": entry.name, "file": target.name, "clip": entry.clip, "codec": PAGE_CODEC[fam.codec],
        "encoder": encoder.key, "label": encoder.label, "chartConfiguration": encoder.chart,
        "width": entry.width, "height": entry.height, "fps": entry.fps, "frames": frames,
        "loopTicks": loop_ticks, "keyInterval": entry.key_interval,
        "targetKbps": entry.kbps, "actualKbps": round(coded * 8 / seconds / 1000, 1), "bufferMs": BUFFER_MS,
        "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "sourceFile": f"clips/{clip_file(entry.clip, entry.height).name}",
        "sourceSha256": sha256(clip_file(entry.clip, entry.height)),
        "sourcePixels": SOURCE_PIXELS,
        "command": " ".join(display_arg(arg, yuv, raw) for arg in cmd),
        "version": version(encoder, cmd[0]),
        **PIXEL_FORMATS[pixel_format],
        **({"notice": NOTICES[entry.clip]} if entry.clip in NOTICES else {}),
    }
    raw.unlink()
    return record, hashes


def key_frames(packets: list[bytes], codec: Codec) -> list[int]:
    """The access units that hold an IDR (H.264 NAL type 5) or IRAP picture (HEVC 16-23)."""
    def keyed(unit: bytes) -> bool:
        for m in re.finditer(b"\x00\x00\x01", unit):
            header = unit[m.end()]
            if (codec == Codec.H264 and header & 0x1F == 5) or (codec == Codec.HEVC and 16 <= (header >> 1) & 0x3F <= 23):
                return True
        return False
    return [index for index, unit in enumerate(packets) if keyed(unit)]


def key_interval(item: Supplied, keys: list[int], frames: int) -> int:
    """A supplied stream's key interval: its key frames must start the stream and be
    evenly spaced, as the page's matching key interval assumes."""
    interval = keys[1] - keys[0] if len(keys) > 1 else frames
    if keys != list(range(0, frames, interval)):
        raise RuntimeError(f"{item.name}: key frames at {keys} are not evenly spaced from frame 0")
    return interval


def check_supplied(item: Supplied, nanocodec: Path, scratch: Path) -> tuple[dict, list[str]]:
    """Decode-check a supplied stream (reference and native decoders agree on every
    frame) and describe it."""
    target = OUT / f"{item.name}.ivf"
    if not target.exists() or sha256(target) != item.sha256:
        raise RuntimeError(f"{target}: missing or not the supplied file (sha256 {item.sha256})")
    packets = ivf_packets(target.read_bytes())
    annex_b = scratch / f"{item.name}.raw"
    annex_b.write_bytes(b"".join(packets))
    try:
        hashes = verify(annex_b, item.codec, item, nanocodec, scratch)
    finally:
        annex_b.unlink()
    source = clip_file(item.clip, item.height)
    _, _, clip_fps, clip_frames = probe(source)
    frames, keys = len(packets), key_frames(packets, item.codec)
    interval = key_interval(item, keys, frames)
    record = {
        "id": item.name, "file": target.name, "clip": item.clip, "codec": PAGE_CODEC[item.codec],
        "encoder": "bwc-soc", "label": BWC_SOC["label"], "chartConfiguration": False, "supplied": True,
        "width": item.width, "height": item.height, "fps": item.fps, "frames": frames,
        "loopTicks": round(clip_frames * SOURCE_FPS / clip_fps), "keyFrames": len(keys),
        "keyInterval": interval,
        "targetKbps": item.kbps, "actualKbps": round(sum(map(len, packets)) * 8 / (frames / item.fps) / 1000, 1),
        "bufferMs": BUFFER_MS, "bytes": target.stat().st_size, "sha256": item.sha256,
        "sourceFile": f"clips/{source.name}", "sourceSha256": sha256(source),
        "sourcePixels": "an earlier RGB-converted copy (limited-range BT.601 via RGB)",
        "sourceNote": SUPPLIED_SOURCE_NOTE.format(gap=item.psnr_gap_db),
        "provenance": BWC_SOC["provenance"], "rateControl": BWC_SOC["rateControl"],
        "command": BWC_SOC["command"], "version": {"version": BWC_SOC["version"], "source": {}},
        **PIXEL_FORMATS["yuv420p"],
        **({"notice": NOTICES[item.clip]} if item.clip in NOTICES else {}),
    }
    return record, hashes


def described(record: dict) -> dict:
    """A generated record with the descriptive fields that follow from its encoder
    (refreshed on every run, as they need no re-encode)."""
    if record.get("supplied"):
        return record
    encoder = ENCODERS[record["encoder"]]
    return {**record, "label": encoder.label, "rateControl": RATE_CONTROL[family(encoder).kind],
            "keyFrames": -(-record["frames"] // record["keyInterval"])}


def display_arg(arg: str, yuv: Path, raw: Path) -> str:
    """The command as recorded: scratch paths replaced by readable names."""
    return arg.replace(str(yuv), "source.yuv").replace(str(raw), "stream.out").replace(str(Path.home()), "~")


def write_table(manifest: dict) -> None:
    """The streams, their encoders' versions and exact commands, as a results page."""
    def note(r: dict) -> str:
        return " (supplied; hardware encoder)" if r.get("supplied") else "" if r["chartConfiguration"] else " (not a chart configuration)"
    rows = [f"| {r['clip']} | {r['label']}{note(r)} | {r['codec']} | {r.get('rateControl', '')} | "
            f"{r['width']}x{r['height']} | {r['fps']} | {r['targetKbps']} | {r['actualKbps']} | {r['bytes'] / 1024:.0f} |"
            for r in manifest["streams"]]
    commands = [f"- `{r['id']}`: {r['version']['version']}"
                f"{' (source ' + r['version']['source']['ref'] + ' ' + r['version']['source']['commit'][:12] + ')' if r['version']['source'] else ''}"
                f"{' on ' + r['version']['gpu'] if 'gpu' in r['version'] else ''}\n\n  ```\n  {r['command']}\n  ```"
                for r in manifest["streams"]]
    total = sum(r["bytes"] for r in manifest["streams"]) / 2**20
    TABLE.write_text("\n".join([
        "# Pre-encoded reference streams in the browser demo", "",
        "Generated by `tools/web/make-reference-streams.py` from `web/streams-ref/manifest.json`; do not edit.",
        f"{len(rows)} streams, {total:.1f} MiB. Each is decoded identically, frame by frame, by the codec's reference",
        "decoder, our native decoder and both WASM builds (`web/tests/reference-streams-wasm.mjs`).", "",
        "| Clip | Encoder | Codec | Rate control | Size | fps | Target kbps | Actual kbps | KiB |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
        *rows, "", "## Versions and commands", "", "`source.yuv` is the clip as the page samples it (see the generator).", "",
        *commands, ""]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def up_to_date(entry: Entry, record: dict, scratch: Path) -> bool:
    """Whether a manifest record still stands: same source file, source frames made
    the same way, same command, and its stream file unchanged. Such streams are neither re-encoded nor re-verified."""
    stream = OUT / record["file"]
    if record.get("sourceSha256") != sha256(clip_file(entry.clip, entry.height)) or not stream.exists():
        return False
    if record.get("sourcePixels") != SOURCE_PIXELS:
        return False
    yuv, raw = scratch / "source.yuv", scratch / f"{entry.name}.raw"
    cmd = command(ENCODERS[entry.encoder], entry, record["frames"], yuv, raw)
    return record["command"] == " ".join(display_arg(arg, yuv, raw) for arg in cmd) and record["sha256"] == sha256(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="remake only these entries (by default every entry that is not up to date)")
    parser.add_argument("--table", action="store_true", help="only rewrite the results table from the manifest")
    parser.add_argument("--write-frames", nargs=4, metavar=("CLIP", "WxH", "FPS", "OUT"),
                        help="only write the frames the demo codes for CLIP at that size and rate, as raw I420, to OUT "
                             "(for an outside encoder; NV12: ffmpeg -f rawvideo -pix_fmt yuv420p -s WxH -i OUT "
                             "-f rawvideo -pix_fmt nv12 OUT.nv12)")
    parser.add_argument("--nanocodec", type=Path,
                        help="native CLI whose decoder must match the reference decoders")
    args = parser.parse_args()
    manifest_path, hashes_path = OUT / "manifest.json", OUT / "hashes.json"
    if args.table:
        write_table(json.loads(manifest_path.read_text()))
        return
    if args.write_frames:
        clip, size, fps, out = args.write_frames
        width, height = map(int, size.split("x"))
        frames, _ = write_source(Entry(clip, "", width, height, int(fps), 0), Path(out))
        print(f"{out}: {frames} frames of {clip}, {width}x{height} I420, {fps} fps")
        return
    if not args.nanocodec:
        parser.error("--nanocodec is required to make streams")
    OUT.mkdir(exist_ok=True)
    old = {r["id"]: r for r in json.loads(manifest_path.read_text())["streams"]} if manifest_path.exists() else {}
    old_hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    records, hashes = [], {}
    with tempfile.TemporaryDirectory(dir=DATA / "work") as scratch:
        for entry in GRID:
            kept = old.get(entry.name)
            if kept and (entry.name not in args.only if args.only else up_to_date(entry, kept, Path(scratch))):
                records.append(kept)
                hashes[entry.name] = old_hashes[entry.name]
                continue
            if args.only and entry.name not in args.only:
                continue  # never made yet: made by a run that names it, or a full run
            record, frame_md5 = make(entry, args.nanocodec, Path(scratch))
            print(f"{entry.name}: {record['actualKbps']} kbps, {record['bytes']} bytes", flush=True)
            records.append(record)
            hashes[entry.name] = frame_md5
        for item in SUPPLIED:
            kept = old.get(item.name)
            current = (kept and kept["sha256"] == item.sha256 and (OUT / kept["file"]).exists()
                       and sha256(OUT / kept["file"]) == item.sha256
                       and kept.get("sourceSha256") == sha256(clip_file(item.clip, item.height)))
            if kept and (item.name not in args.only if args.only else current):
                records.append(kept)
                hashes[item.name] = old_hashes[item.name]
                continue
            record, frame_md5 = check_supplied(item, args.nanocodec, Path(scratch))
            print(f"{item.name}: supplied, {record['actualKbps']} kbps, {record['keyFrames']} key frame(s)", flush=True)
            records.append(record)
            hashes[item.name] = frame_md5
    records = [described(r) for r in records]
    manifest = {"generator": "tools/web/make-reference-streams.py", "sourceFps": SOURCE_FPS, "streams": records}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    hashes_path.write_text(json.dumps(hashes) + "\n")
    write_table(manifest)


if __name__ == "__main__":
    main()
