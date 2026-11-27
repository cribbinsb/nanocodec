#!/usr/bin/env python3
"""Foreign-encoder check: do our decoders take streams made by real encoders?

The official suites (tools/ci/conformance.py) and our own streams
(tools/ci/landing_conformance.py) do not cover what other encoders emit in
their default and common settings: x264 and openh264 (H.264), x265 and
kvazaar (HEVC), VVenC (VVC), aomenc, SVT-AV1 and rav1e (AV1), vpxenc and
libvpx's SVC example (VP9), avmenc (AV2), and FFmpeg's wrappers of them.
This check codes a few short clips with each encoder setting in `CASES`,
decodes every stream with the codec's two reference decoders and with ours,
and compares the decoded pictures bit-exactly (per-frame MD5).

Every decoder writes into a pipe that is hashed as it arrives, so no decoded
picture is written to disk; the streams live in a temporary directory and
only failing ones are kept, with `--keep-failing DIR` (never publish them).

A stream's verdict:
    pass            our output equals the reference decoders' (which agree)
    mismatch        decoded, but some picture differs (first one recorded)
    rejected        our decoder refused it as unsupported (the tool it names)
    error           our decoder failed with another error
    crash           our decoder panicked or died on a signal
    timeout         our decoder ran past --decode-timeout
    encoder-error   the encoder refused the setting (nothing to judge)
    reference-error a reference decoder failed, or the two disagree

Sources: `--clips DIR` holds raw 4:2:0 clips named
`{name}_{W}x{H}_{depth}.yuv` (depth 8: bytes; 10: 16-bit little-endian),
one per `SOURCES` entry, e.g. scaled from project clips (CLAUDE.md test
clips) with ffmpeg. Tools are looked up in each `--tools DIR`, then the
anchors (`~/nanocodec-data/anchors/<arch>/bin`, `anchors/vp9-av2/bin`), then
$PATH; the FFmpeg *encoder* is the one on $PATH (it needs libx264, libx265,
libvpx, libaom, libsvtav1 and librav1e), the FFmpeg *decoder* the pinned
anchors build `ffmpeg-decode`.

Usage:
    python3 -I tools/ci/foreign_encoder_check.py --clips DIR [--bin NANOCODEC]
        [--tools DIR ...] [--codecs h264,hevc,...] [--only SUBSTRING]
        [--jobs 16] [--json OUT] [--markdown OUT] [--keep-failing DIR]

Exit status 0 when every judged stream passes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
ANCHOR_DIRS = (DATA / "anchors" / platform.machine() / "bin", DATA / "anchors/vp9-av2/bin")

FPS = 30
FRAMES = 30
CHUNK = 1 << 20
#: Bytes of a failing decoder's error output kept for the report.
ERROR_TAIL = 400
ENCODE_TIMEOUT_SECONDS = 600
DECODE_TIMEOUT_SECONDS = 300
#: Rust's exit status for a panic in the main thread.
RUST_PANIC_STATUS = 101
UNSUPPORTED = re.compile(r"unsupported bitstream: (.*)")


# --------------------------------------------------------------------------
# Sources and cases


@dataclass(frozen=True)
class Source:
    """A raw 4:2:0 clip: `{name}_{width}x{height}_{depth}.yuv` in --clips."""

    name: str
    width: int
    height: int

    def path(self, clips: Path, depth: int) -> Path:
        return clips / f"{self.name}_{self.width}x{self.height}_{depth}.yuv"


#: A: block-aligned width, 240 rows (not a multiple of 64-sample CTUs or
#: superblocks); B: neither dimension a multiple of 8; T: wide enough for two
#: VP9 tile columns (each at least 256 samples); O: odd dimensions, for the
#: codecs whose 4:2:0 allows them (AV1, VP9, AV2).
SOURCES = {
    "A": Source("market", 416, 240),
    "B": Source("box", 202, 98),
    "T": Source("market", 640, 360),
    "O": Source("box", 201, 99),
}


@dataclass(frozen=True)
class Case:
    codec: str          # our CLI's --codec
    encoder: str        # a key of ENCODERS
    label: str          # the setting, as shown in the table
    args: tuple = ()    # the encoder's extra arguments
    source: str = "A"   # a key of SOURCES
    depth: int = 8      # source and stream bit depth
    frames: int = FRAMES
    #: Decoded bit depth when it differs from the source's (VVenC codes
    #: 8-bit input at 10 bits by default).
    out_depth: int = 0
    expect: str = ""    # known gap, shown beside the verdict
    #: Reference decoders to use when not all of the codec's apply.
    references: tuple = ()

    @property
    def name(self) -> str:
        size = SOURCES[self.source]
        return (f"{self.codec}/{self.encoder}/{self.label}/"
                f"{size.width}x{size.height}/{self.depth}bit")

    @property
    def decoded_depth(self) -> int:
        return self.out_depth or self.depth


def matrix(codec, encoder, settings, sources=("A", "B"), **common):
    """One case per (label, args) setting and source."""
    return [Case(codec, encoder, label, tuple(args), source, **common)
            for label, args in settings for source in sources]


def presets(flag, names):
    return [(f"{flag.lstrip('-')} {name}", [flag, name]) for name in names]


def h264_cases():
    x264 = [
        *presets("--preset", ["ultrafast", "superfast", "veryfast", "faster", "fast",
                              "medium", "slow", "slower", "veryslow"]),
        ("tune film", ["--tune", "film"]),
        ("tune zerolatency", ["--tune", "zerolatency"]),
        ("tune animation", ["--tune", "animation"]),
        ("bframes 0", ["--bframes", "0"]),
        ("bframes 3 b-pyramid none", ["--bframes", "3", "--b-pyramid", "none"]),
        ("bframes 8 b-pyramid strict", ["--bframes", "8", "--b-pyramid", "strict"]),
        ("weightp 1", ["--weightp", "1"]),
        ("weightp 0 no-weightb", ["--weightp", "0", "--no-weightb"]),
        ("cqm jvt", ["--cqm", "jvt"]),
        ("no-8x8dct", ["--no-8x8dct"]),
        ("cavlc", ["--no-cabac"]),
        ("profile baseline", ["--profile", "baseline"]),
        ("profile main", ["--profile", "main"]),
        ("slices 4", ["--slices", "4"]),
        ("ref 16", ["--ref", "16"]),
        ("keyint 10 open-gop", ["--keyint", "10", "--open-gop"]),
        ("intra only", ["--keyint", "1"]),
        ("aq-mode 3", ["--aq-mode", "3"]),
        ("no-deblock", ["--no-deblock"]),
        ("nal-hrd cbr", ["--bitrate", "200", "--vbv-maxrate", "200", "--vbv-bufsize", "200",
                         "--nal-hrd", "cbr"]),
        ("lossless qp 0", ["--qp", "0"]),
    ]
    return [
        *matrix("h264", "x264", x264),
        Case("h264", "x264", "preset placebo", ("--preset", "placebo"), "B"),
        *matrix("h264", "x264", [("preset medium", ["--preset", "medium"]),
                                 ("preset ultrafast", ["--preset", "ultrafast"]),
                                 ("preset veryslow", ["--preset", "veryslow"])], depth=10),
        *matrix("h264", "ffmpeg-libx264", [("defaults", [])]),
        *matrix("h264", "ffmpeg-libx264", [("defaults", [])], depth=10),
        # Interlaced coding needs a height in whole field macroblock pairs.
        *matrix("h264", "ffmpeg-libx264", [
            ("interlaced tff (MBAFF)", ["-flags", "+ildct+ilme", "-x264-params", "interlaced=1"]),
            ("fake-interlaced", ["-x264-params", "fake-interlaced=1:pic-struct=1"]),
        ], ("A",), expect="interlaced is not decoded yet"),
        *matrix("h264", "openh264", [("defaults", []), ("rc off", ["-rc", "-1"]),
                                     ("temporal layers 3", ["-numtl", "3"]),
                                     ("slices 4", ["-slcmd", "0", "1", "-slcnum", "0", "4"])]),
    ]


def hevc_cases():
    x265 = [
        *presets("--preset", ["ultrafast", "superfast", "veryfast", "faster", "fast",
                              "medium", "slow", "slower", "veryslow"]),
        *presets("--tune", ["psnr", "ssim", "grain", "zerolatency", "fastdecode"]),
        ("scaling-list default", ["--scaling-list", "default"]),
        ("no-wpp", ["--no-wpp"]),
        ("slices 4", ["--slices", "4"]),
        ("no-sao", ["--no-sao"]),
        ("sao-non-deblock limit-sao", ["--sao-non-deblock", "--limit-sao"]),
        ("deblock -2:-2", ["--deblock", "-2:-2"]),
        ("no-deblock", ["--no-deblock"]),
        ("amp rect", ["--amp", "--rect"]),
        ("ctu 32", ["--ctu", "32"]),
        ("ctu 16", ["--ctu", "16"]),
        ("bframes 8 b-pyramid", ["--bframes", "8", "--b-pyramid", "--b-intra"]),
        ("ref 5 weightb", ["--ref", "5", "--weightb"]),
        ("keyint 10 (CRA, RASL)", ["--keyint", "10", "--open-gop"]),
        ("closed gop keyint 10", ["--keyint", "10", "--no-open-gop"]),
        ("intra only", ["--keyint", "1"]),
        ("tskip", ["--tskip"]),
        ("constrained-intra", ["--constrained-intra"]),
        ("no-strong-intra-smoothing", ["--no-strong-intra-smoothing"]),
        ("rd 6 rdoq 2", ["--rd", "6", "--rdoq-level", "2"]),
        ("lossless", ["--lossless"]),
        ("hrd vbv", ["--bitrate", "200", "--vbv-maxrate", "200", "--vbv-bufsize", "200", "--hrd"]),
        ("no-sign-hide", ["--no-signhide"]),
        ("tu depth 4", ["--tu-intra-depth", "4", "--tu-inter-depth", "4"]),
    ]
    return [
        *matrix("hevc", "x265", x265),
        Case("hevc", "x265", "preset placebo", ("--preset", "placebo"), "B"),
        *matrix("hevc", "ffmpeg-libx265", [("defaults", [])]),
        *matrix("hevc", "ffmpeg-libx265", [
            ("defaults", []),
            ("preset ultrafast", ["-preset", "ultrafast"]),
            ("preset slow", ["-preset", "slow"]),
            ("scaling-list default", ["-x265-params", "scaling-list=default"]),
        ], depth=10),
        *matrix("hevc", "kvazaar", [
            ("defaults", []),
            *presets("--preset", ["ultrafast", "medium", "veryslow"]),
            ("no-wpp", ["--no-wpp"]),
            ("gop 8", ["--gop", "8"]),
            ("gop lp-g4d3t1", ["--gop", "lp-g4d3t1"]),
            ("smp amp", ["--smp", "--amp"]),
            ("tr-skip", ["--transform-skip"]),
            ("intra only", ["-p", "1"]),
        ]),
        # Main profile tile columns are at least 256 samples wide (HEVC A.3.2).
        *matrix("hevc", "kvazaar", [("tiles 2x2", ["--tiles", "2x2"])], ("T",)),
    ]


def vvc_cases():
    vvenc = [
        *presets("--preset", ["faster", "fast", "medium", "slow", "slower"]),
        ("internal 8-bit", ["--preset", "faster", "--InternalBitDepth", "8"]),
        ("wpp", ["--preset", "faster", "--WaveFrontSynchro", "1"]),
        ("no alf/ccalf", ["--preset", "faster", "--ALF", "0", "--CCALF", "0"]),
        ("lmcs", ["--preset", "faster", "--LMCSEnable", "1"]),
        ("intra period 8", ["--preset", "faster", "--IntraPeriod", "8"]),
        ("rpr on", ["--preset", "faster", "--RPR", "1"]),
        ("no dep-quant", ["--preset", "faster", "--DepQuant", "0"]),
        ("ibc on", ["--preset", "faster", "--IBC", "1"]),
    ]
    return [
        *matrix("vvc", "vvenc", vvenc, out_depth=10),
        # Tiles: VVenC picks the level from the picture size, and small
        # pictures' levels allow one tile column; level 4.1 allows two.
        *matrix("vvc", "vvenc", [("tiles 2x2", ["--preset", "faster", "--Level", "4.1",
                                                "--Tiles", "2x2"])], ("T",), out_depth=10),
        *matrix("vvc", "vvenc", [("preset faster", ["--preset", "faster"]),
                                 ("preset medium", ["--preset", "medium"])], depth=10),
    ]


def av1_cases():
    aom = [
        *[(f"good cpu-used {n}", ["--good", f"--cpu-used={n}"]) for n in range(7)],
        *[(f"rt cpu-used {n}", ["--rt", f"--cpu-used={n}"]) for n in range(5, 11)],
        ("enable-qm", ["--good", "--cpu-used=4", "--enable-qm=1"]),
        ("deltaq-mode 1", ["--good", "--cpu-used=4", "--deltaq-mode=1"]),
        ("deltaq-mode 1 delta-lf-mode 1",
         ["--good", "--cpu-used=4", "--deltaq-mode=1", "--delta-lf-mode=1"]),
        *[(f"aq-mode {n}", ["--good", "--cpu-used=4", f"--aq-mode={n}"]) for n in (1, 2, 3)],
        ("rt aq-mode 3", ["--rt", "--cpu-used=8", "--aq-mode=3"]),
        ("lossless", ["--good", "--cpu-used=4", "--lossless=1"]),
        ("error-resilient", ["--rt", "--cpu-used=8", "--error-resilient=1"]),
        ("kf-max-dist 10", ["--good", "--cpu-used=4", "--kf-max-dist=10"]),
        ("lag 0", ["--good", "--cpu-used=4", "--lag-in-frames=0"]),
        ("superres", ["--good", "--cpu-used=4", "--superres-mode=1"]),
        ("resize", ["--good", "--cpu-used=4", "--resize-mode=1"]),
        ("screen content", ["--good", "--cpu-used=4", "--tune-content=screen"]),
        ("film grain", ["--good", "--cpu-used=4", "--denoise-noise-level=10"]),
        ("end-usage cbr", ["--rt", "--cpu-used=8", "--end-usage=cbr", "--target-bitrate=200"]),
    ]
    svt = [
        *[(f"preset {n}", ["--preset", str(n)]) for n in range(13)],
        *[(f"rtc preset {n}", ["--rtc", "1", "--preset", str(n)]) for n in (8, 10, 12)],
        ("enable-qm", ["--preset", "8", "--enable-qm", "1"]),
        ("variance boost", ["--preset", "8", "--enable-variance-boost", "1"]),
        ("aq-mode 1", ["--preset", "8", "--aq-mode", "1"]),
        ("film grain 8", ["--preset", "8", "--film-grain", "8"]),
        ("keyint 10", ["--preset", "8", "--keyint", "10"]),
        ("low delay", ["--preset", "8", "--pred-struct", "1"]),
        ("screen content", ["--preset", "8", "--scm", "1"]),
        ("fast-decode 1", ["--preset", "8", "--fast-decode", "1"]),
    ]
    tiles = [("aom tiles 2x2", "aomenc",
              ["--good", "--cpu-used=4", "--tile-columns=1", "--tile-rows=1"]),
             ("aom rt tiles 4 cols", "aomenc", ["--rt", "--cpu-used=8", "--tile-columns=2"]),
             ("svt tiles 2x2", "svtav1", ["--preset", "8", "--tile-columns", "1",
                                          "--tile-rows", "1"])]
    return [
        *matrix("av1", "aomenc", aom),
        *matrix("av1", "aomenc", [("good cpu-used 4", ["--good", "--cpu-used=4"]),
                                  ("rt cpu-used 8", ["--rt", "--cpu-used=8"])], depth=10),
        *matrix("av1", "aomenc", [("good cpu-used 4", ["--good", "--cpu-used=4"])], ("O",)),
        *matrix("av1", "svtav1", svt),
        *matrix("av1", "svtav1", [("preset 8", ["--preset", "8"]),
                                  ("preset 4", ["--preset", "4"])], depth=10),
        *[Case("av1", encoder, label, tuple(args), "T") for label, encoder, args in tiles],
        *matrix("av1", "ffmpeg-libaom", [("defaults", [])]),
        *matrix("av1", "ffmpeg-libsvtav1", [("defaults", [])]),
        *matrix("av1", "ffmpeg-librav1e", [("defaults", []),
                                           ("speed 10", ["-speed", "10"])]),
        *matrix("av1", "ffmpeg-librav1e", [("defaults", [])], depth=10),
    ]


def vp9_cases():
    vpx = [
        *[(f"good cpu-used {n}", ["--good", f"--cpu-used={n}"]) for n in range(6)],
        *[(f"rt cpu-used {n}", ["--rt", f"--cpu-used={n}"]) for n in range(5, 10)],
        *[(f"aq-mode {n}", ["--good", "--cpu-used=2", f"--aq-mode={n}"]) for n in (1, 2, 3)],
        ("rt aq-mode 3 cbr", ["--rt", "--cpu-used=8", "--aq-mode=3", "--end-usage=cbr",
                              "--target-bitrate=200", "--lag-in-frames=0"]),
        ("lossless", ["--good", "--cpu-used=2", "--lossless=1"]),
        ("error-resilient", ["--rt", "--cpu-used=8", "--error-resilient=1"]),
        ("frame-parallel", ["--good", "--cpu-used=2", "--frame-parallel=1"]),
        ("no alt-ref", ["--good", "--cpu-used=2", "--auto-alt-ref=0"]),
        ("kf-max-dist 10", ["--good", "--cpu-used=2", "--kf-max-dist=10"]),
        ("rt resize", ["--rt", "--cpu-used=8", "--end-usage=cbr", "--target-bitrate=30",
                       "--resize-allowed=1", "--lag-in-frames=0"]),
        ("screen content", ["--rt", "--cpu-used=8", "--tune-content=screen"]),
    ]
    return [
        *matrix("vp9", "vpxenc", vpx),
        *matrix("vp9", "vpxenc", [("good cpu-used 2", ["--good", "--cpu-used=2"])], ("O",)),
        *matrix("vp9", "vpxenc", [("tiles 2 cols", ["--good", "--cpu-used=2", "--tile-columns=1"]),
                                  ("tiles 2 cols 2 rows", ["--good", "--cpu-used=2",
                                                           "--tile-columns=1", "--tile-rows=1"]),
                                  ("rt tiles 2 cols", ["--rt", "--cpu-used=8",
                                                       "--tile-columns=1"])], ("T",)),
        # FFmpeg shows every spatial layer's picture, vpxdec (and a player)
        # only the top layer's: vpxdec alone is the reference.
        *matrix("vp9", "vp9svc", [("2 spatial 1 temporal", ["-sl", "2", "-tl", "1"]),
                                  ("2 spatial 3 temporal", ["-sl", "2", "-tl", "3"]),
                                  ("3 spatial 1 temporal", ["-sl", "3", "-tl", "1"]),
                                  ("1 spatial 3 temporal", ["-sl", "1", "-tl", "3"])], ("A", "T"),
                references=("vpxdec",)),
        *matrix("vp9", "ffmpeg-libvpx-vp9", [("defaults", [])]),
        # Profile 2: the pinned vpxenc is an 8-bit build, FFmpeg's libvpx is not.
        *matrix("vp9", "ffmpeg-libvpx-vp9", [
            ("defaults", []),
            ("good cpu-used 2", ["-deadline", "good", "-cpu-used", "2"]),
            ("realtime cpu-used 8", ["-deadline", "realtime", "-cpu-used", "8"]),
        ], depth=10),
    ]


def av2_cases():
    # AVM is slow (its defaults take minutes per CIF frame): short clips.
    short = {"frames": 10}
    return [
        # Its defaults take about a minute per 202x98 frame on a loaded box.
        *matrix("av2", "avmenc", [("defaults", [])], ("B",), frames=5),
        *matrix("av2", "avmenc", [
            ("cpu-used 5", ["--cpu-used=5"]),
            ("cpu-used 9", ["--cpu-used=9"]),
            ("cpu-used 9 lag 0", ["--cpu-used=9", "--lag-in-frames=0"]),
            ("cpu-used 9 kf-max-dist 4", ["--cpu-used=9", "--kf-max-dist=4"]),
        ], **short),
        *matrix("av2", "avmenc", [("cpu-used 9", ["--cpu-used=9"])], ("O",), **short),
        # The pinned dav2d is built without 10-bit decoding: avmdec alone.
        *matrix("av2", "avmenc", [("cpu-used 9", ["--cpu-used=9"])], depth=10,
                references=("avmdec",), **short),
        *matrix("av2", "avmenc", [("tiles 2x2", ["--cpu-used=9", "--tile-columns=1",
                                                 "--tile-rows=1"])], ("T",), **short),
    ]


CASES = [*h264_cases(), *hevc_cases(), *vvc_cases(), *av1_cases(), *vp9_cases(), *av2_cases()]


# --------------------------------------------------------------------------
# Tools


class Tools:
    """Finds executables: --tools directories, the anchors, then $PATH."""

    def __init__(self, extra_dirs):
        self.dirs = [Path(d) for d in extra_dirs] + list(ANCHOR_DIRS)
        self.found = {}

    def __call__(self, *names) -> str:
        """The first of `names` found (alternatives in order of preference)."""
        for name in names:
            if name in self.found:
                return self.found[name]
            for directory in self.dirs:
                if (directory / name).is_file() and os.access(directory / name, os.X_OK):
                    self.found[name] = str(directory / name)
                    return self.found[name]
            if shutil.which(name):
                self.found[name] = shutil.which(name)
                return self.found[name]
        raise FileNotFoundError(f"none of {names} found")

    def record(self) -> dict:
        """Path and SHA-256 of every tool used, for the JSON record."""
        out = {}
        for name, path in sorted(self.found.items()):
            out[name] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        return out


# --------------------------------------------------------------------------
# Encoders: (case, tools, source path, size, output) -> argv; output extension


def pix_fmt(depth):
    return "yuv420p" if depth == 8 else f"yuv420p{depth}le"


def enc_x264(case, tools, src, size, out):
    return [tools("x264"), "--input-res", f"{size.width}x{size.height}",
            "--input-depth", str(case.depth), "--output-depth", str(case.depth),
            "--fps", str(FPS), "--frames", str(case.frames), "--quiet", *case.args,
            "-o", str(out), str(src)]


def enc_openh264(case, tools, src, size, out):
    # One spatial layer at the source size: the console encoder's built-in
    # layer is 160x90 unless -dw/-dh set it.
    return [tools("h264enc"), "-org", str(src), "-sw", str(size.width), "-sh", str(size.height),
            "-frms", str(case.frames), "-frin", str(FPS), "-numl", "1",
            "-dw", "0", str(size.width), "-dh", "0", str(size.height), "-frout", "0", str(FPS),
            *case.args, "-bf", str(out)]


def enc_x265(case, tools, src, size, out):
    return [tools("x265"), "--input", str(src), "--input-res", f"{size.width}x{size.height}",
            "--input-depth", str(case.depth), "--fps", str(FPS), "--frames", str(case.frames),
            "--log-level", "error", *case.args, "-o", str(out)]


def enc_kvazaar(case, tools, src, size, out):
    return [tools("kvazaar"), "-i", str(src), "--input-res", f"{size.width}x{size.height}",
            "--input-fps", str(FPS), "-n", str(case.frames), *case.args, "-o", str(out)]


def enc_vvenc(case, tools, src, size, out):
    return [tools("vvencFFapp"), "-i", str(src), "-s", f"{size.width}x{size.height}",
            "--InputBitDepth", str(case.depth), "-fr", str(FPS), "-f", str(case.frames),
            "--Verbosity", "1", *case.args, "-b", str(out)]


def enc_aomenc(case, tools, src, size, out):
    return [tools("aomenc"), f"--width={size.width}", f"--height={size.height}",
            f"--fps={FPS}/1", f"--limit={case.frames}", f"--bit-depth={case.depth}",
            f"--input-bit-depth={case.depth}", "--quiet", *case.args, "--ivf",
            "-o", str(out), str(src)]


def enc_svtav1(case, tools, src, size, out):
    return [tools("SvtAv1EncApp"), "-i", str(src), "-w", str(size.width), "-h", str(size.height),
            "--input-depth", str(case.depth), "--fps-num", str(FPS), "--fps-denom", "1",
            "-n", str(case.frames),
            "--progress", "0", *case.args, "-b", str(out)]


def enc_vpxenc(case, tools, src, size, out):
    # Profile 2 (10-bit) flags come with the case.
    return [tools("vpxenc"), "--codec=vp9", f"--width={size.width}",
            f"--height={size.height}", f"--fps={FPS}/1", f"--limit={case.frames}",
            "--quiet", *case.args, "--ivf", "-o", str(out), str(src)]


def enc_vp9svc(case, tools, src, size, out):
    # libvpx's examples/vp9_spatial_svc_encoder: real-time SVC, as WebRTC codes it.
    return [tools("vp9_spatial_svc_encoder"), "-w", str(size.width), "-h", str(size.height),
            "-f", str(case.frames), "-t", f"1/{FPS}", "-b", "300", "-sp", "7", "--rc-end-usage=1",
            *case.args, "-o", str(out), str(src)]


def enc_avmenc(case, tools, src, size, out):
    return [tools("avmenc"), f"--width={size.width}", f"--height={size.height}",
            f"--fps={FPS}/1", f"--limit={case.frames}", f"--bit-depth={case.depth}",
            f"--input-bit-depth={case.depth}", "--quiet", *case.args, "--ivf",
            "-o", str(out), str(src)]


def ffmpeg_encoder(library, muxer):
    """An FFmpeg wrapper encoder at its defaults plus the case's arguments."""
    def command(case, tools, src, size, out):
        return [tools("ffmpeg"), "-nostdin", "-v", "error", "-f", "rawvideo",
                "-pix_fmt", pix_fmt(case.depth), "-s", f"{size.width}x{size.height}",
                "-r", str(FPS), "-i", str(src), "-frames:v", str(case.frames),
                "-c:v", library, *case.args, "-f", muxer, "-y", str(out)]
    return command


#: encoder -> (argv builder, stream file extension)
ENCODERS = {
    "x264": (enc_x264, "264"),
    "openh264": (enc_openh264, "264"),
    "ffmpeg-libx264": (ffmpeg_encoder("libx264", "h264"), "264"),
    "x265": (enc_x265, "265"),
    "kvazaar": (enc_kvazaar, "265"),
    "ffmpeg-libx265": (ffmpeg_encoder("libx265", "hevc"), "265"),
    "vvenc": (enc_vvenc, "266"),
    "aomenc": (enc_aomenc, "ivf"),
    "svtav1": (enc_svtav1, "ivf"),
    "ffmpeg-libaom": (ffmpeg_encoder("libaom-av1", "ivf"), "ivf"),
    "ffmpeg-libsvtav1": (ffmpeg_encoder("libsvtav1", "ivf"), "ivf"),
    "ffmpeg-librav1e": (ffmpeg_encoder("librav1e", "ivf"), "ivf"),
    "vpxenc": (enc_vpxenc, "ivf"),
    "vp9svc": (enc_vp9svc, "ivf"),
    "ffmpeg-libvpx-vp9": (ffmpeg_encoder("libvpx-vp9", "ivf"), "ivf"),
    "avmenc": (enc_avmenc, "ivf"),
}


# --------------------------------------------------------------------------
# Decoders: (tools, stream, output path) -> argv. Outputs are raw 4:2:0,
# cropped, at the stream's depth (deeper than 8 bits: 16-bit little-endian).


def dec_ffmpeg(demuxer, decoder=()):
    def command(tools, stream, out):
        return [tools("ffmpeg-decode"), "-nostdin", "-v", "error", "-xerror",
                "-err_detect", "explode", "-threads", "1", "-f", demuxer, *decoder,
                "-i", str(stream), "-fps_mode", "passthrough", "-f", "rawvideo", "-y", str(out)]
    return command


REFERENCE_DECODERS = {
    "h264": {"jm": lambda t, s, o: [t("ldecod"), "-i", str(s), "-o", str(o)],
             "ffmpeg": dec_ffmpeg("h264")},
    "hevc": {"hm": lambda t, s, o: [t("TAppDecoder"), "-b", str(s), "-o", str(o), "-d", "0"],
             "ffmpeg": dec_ffmpeg("hevc")},
    "vvc": {"vvdec": lambda t, s, o: [t("vvdecapp"), "-v", "0", "-t", "1", "-b", str(s),
                                      "-o", str(o)],
            "ffmpeg": dec_ffmpeg("vvc", ("-c:v", "vvc"))},
    "av1": {"aomdec": lambda t, s, o: [t("aomdec"), "--rawvideo", "--skip-film-grain",
                                       "-o", str(o), str(s)],
            "dav1d": lambda t, s, o: [t("dav1d"), "--threads", "1", "--filmgrain", "0",
                                      "--muxer", "yuv", "-i", str(s), "-o", str(o)]},
    "vp9": {"vpxdec": lambda t, s, o: [t("vpxdec-hbd", "vpxdec"), "--codec=vp9", "--rawvideo",
                                       "--threads=1", "-o", str(o), str(s)],
            "ffmpeg": dec_ffmpeg("ivf", ("-c:v", "vp9"))},
    "av2": {"avmdec": lambda t, s, o: [t("avmdec"), "--rawvideo", "--threads=1",
                                       "-o", str(o), str(s)],
            "dav2d": lambda t, s, o: [t("dav2d"), "--threads", "1", "--filmgrain", "0",
                                      "--muxer", "yuv", "--input", str(s), "--output", str(o)]},
}

def own_decoder(binary, codec):
    grain_free = ["--film-grain", "0"] if codec == "av1" else []
    return lambda tools, stream, out: [str(binary), "decode", "--codec", codec,
                                       "--input", str(stream), "--output", str(out), *grain_free]


# --------------------------------------------------------------------------
# Decoding into hashes


class FrameHasher:
    """Hashes 4:2:0 pictures streamed from a decoder, per picture and whole."""

    def __init__(self, width, height, depth):
        sample = 1 if depth == 8 else 2
        chroma = ((width + 1) // 2) * ((height + 1) // 2)
        self.frame_bytes = (width * height + 2 * chroma) * sample
        self.buffer = bytearray()
        self.whole = hashlib.md5()
        self.frames = []

    def update(self, chunk):
        self.whole.update(chunk)
        self.buffer.extend(chunk)
        while len(self.buffer) >= self.frame_bytes:
            self.frames.append(hashlib.md5(self.buffer[:self.frame_bytes]).hexdigest())
            del self.buffer[:self.frame_bytes]

    @property
    def leftover(self):
        """Trailing bytes that do not make a whole picture."""
        return len(self.buffer)


@dataclass
class Decode:
    status: str               # ok, error, crash, timeout
    frames: list = field(default_factory=list)
    md5: str = ""
    leftover: int = 0         # trailing bytes that were not a whole picture
    message: str = ""
    seconds: float = 0.0


def run_decoder(command, hasher, cwd, timeout):
    """Run a decoder writing into a pipe (its output path /dev/fd/N)."""
    read_fd, write_fd = os.pipe()
    argv = command(f"/dev/fd/{write_fd}")
    start = time.monotonic()
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, pass_fds=(write_fd,), cwd=cwd)
    except OSError as err:
        os.close(read_fd)
        os.close(write_fd)
        return Decode("error", message=str(err))
    os.close(write_fd)
    errors = bytearray()
    stderr_reader = threading.Thread(target=lambda: errors.extend(process.stderr.read()))
    stderr_reader.start()
    timed_out = False
    with os.fdopen(read_fd, "rb", buffering=0) as pipe:
        killer = threading.Timer(timeout, process.kill)
        killer.start()
        while chunk := pipe.read(CHUNK):
            hasher.update(chunk)
        process.wait()
        timed_out = not killer.is_alive() and process.returncode < 0
        killer.cancel()
    stderr_reader.join()
    text = errors.decode(errors="replace").strip()
    result = Decode("ok", hasher.frames, hasher.whole.hexdigest(), hasher.leftover,
                    seconds=round(time.monotonic() - start, 2))
    if timed_out:
        result.status = "timeout"
    elif process.returncode < 0 or process.returncode == RUST_PANIC_STATUS or "panicked" in text:
        result.status = "crash"
    elif process.returncode != 0:
        result.status = "error"
    if result.status != "ok":
        result.message = f"exit {process.returncode}: " + text[-ERROR_TAIL:].replace("\n", " | ")
    return result


# --------------------------------------------------------------------------
# One case


@dataclass
class Outcome:
    name: str
    codec: str
    encoder: str
    label: str
    size: str
    depth: int
    frames: int
    args: list
    expect: str
    verdict: str = ""
    detail: str = ""
    stream_bytes: int = 0
    encode_seconds: float = 0.0
    reference_frames: int = 0
    own_frames: int = 0
    first_wrong_frame: int | None = None
    reference_md5: str = ""
    own_md5: str = ""
    decoders: dict = field(default_factory=dict)


def first_difference(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def run_case(case, args, tools):
    size = SOURCES[case.source]
    outcome = Outcome(case.name, case.codec, case.encoder, case.label,
                      f"{size.width}x{size.height}", case.depth, case.frames, list(case.args),
                      case.expect)
    builder, extension = ENCODERS[case.encoder]
    with tempfile.TemporaryDirectory(prefix="nanocodec-foreign-") as work:
        stream = Path(work) / f"stream.{extension}"
        start = time.monotonic()
        try:
            argv = builder(case, tools, size.path(args.clips, case.depth), size, stream)
            encoded = subprocess.run(argv, cwd=work, stdin=subprocess.DEVNULL,
                                     capture_output=True, timeout=ENCODE_TIMEOUT_SECONDS)
            failed = encoded.returncode != 0 or not stream.is_file() or not stream.stat().st_size
            message = encoded.stderr.decode(errors="replace").strip()[-ERROR_TAIL:]
        except (subprocess.TimeoutExpired, FileNotFoundError) as err:
            failed, message = True, str(err)
        outcome.encode_seconds = round(time.monotonic() - start, 2)
        if failed:
            outcome.verdict, outcome.detail = "encoder-error", message.replace("\n", " | ")
            return outcome
        outcome.stream_bytes = stream.stat().st_size

        decoders = {name: command for name, command in REFERENCE_DECODERS[case.codec].items()
                    if name in (case.references or (name,))}
        decoders["nanocodec"] = own_decoder(args.bin, case.codec)
        results = {}
        for name, command in decoders.items():
            hasher = FrameHasher(size.width, size.height, case.decoded_depth)
            try:
                bound = lambda out, command=command: command(tools, stream, out)
                results[name] = run_decoder(bound, hasher, work, args.decode_timeout)
            except FileNotFoundError as err:
                results[name] = Decode("error", message=str(err))
        outcome.decoders = {name: {"status": r.status, "frames": len(r.frames), "md5": r.md5,
                                   "leftover_bytes": r.leftover, "seconds": r.seconds,
                                   **({"message": r.message} if r.message else {})}
                            for name, r in results.items()}
        judge(outcome, results)
        if outcome.verdict != "pass" and args.keep_failing:
            target = args.keep_failing / f"{case.name.replace('/', '__').replace(' ', '_')}.{extension}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stream, target)
    return outcome


def judge(outcome, results):
    """Sets the verdict from the decoders' results (module docstring).

    Streams are compared over the whole output; the per-picture hashes only
    locate the first difference. A stream whose pictures change size (VP9
    spatial layers, resizing) is cut into source-sized pieces there, so its
    first wrong "picture" is approximate.
    """
    own = results.pop("nanocodec")
    references = list(results.items())
    for name, ref in references:
        if ref.status != "ok":
            outcome.verdict, outcome.detail = "reference-error", f"{name}: {ref.message}"
            return
    (first_name, first), *others = references
    for name, ref in others:
        if ref.md5 != first.md5:
            wrong = first_difference(first.frames, ref.frames)
            outcome.verdict = "reference-error"
            outcome.detail = (f"{first_name} and {name} differ at picture {wrong} "
                              f"({len(first.frames)} / {len(ref.frames)} pictures)")
            return
    outcome.reference_frames, outcome.reference_md5 = len(first.frames), first.md5
    outcome.own_frames, outcome.own_md5 = len(own.frames), own.md5
    if own.status == "ok" and own.md5 == first.md5:
        outcome.verdict = "pass"
        return
    outcome.first_wrong_frame = first_difference(first.frames, own.frames)
    unsupported = UNSUPPORTED.search(own.message)
    if own.status == "ok":
        outcome.verdict = "mismatch"
        outcome.detail = (f"first wrong picture {outcome.first_wrong_frame} "
                          f"({len(own.frames)}/{len(first.frames)} pictures)")
    elif own.status == "error" and unsupported:
        outcome.verdict, outcome.detail = "rejected", unsupported.group(1).split(" | ")[0]
    else:
        outcome.verdict = own.status
        outcome.detail = (f"after {len(own.frames)}/{len(first.frames)} pictures: "
                          f"{own.message}")


# --------------------------------------------------------------------------
# Report

CODEC_TITLES = {"h264": "H.264", "hevc": "HEVC", "vvc": "VVC", "av1": "AV1", "vp9": "VP9",
                "av2": "AV2"}
JUDGED = ("pass", "mismatch", "rejected", "error", "crash", "timeout")


def markdown(outcomes, header):
    lines = [header, ""]
    for codec, title in CODEC_TITLES.items():
        rows = [o for o in outcomes if o.codec == codec]
        if not rows:
            continue
        counts = {v: sum(o.verdict == v for o in rows) for v in dict.fromkeys(
            [*JUDGED, "encoder-error", "reference-error"])}
        summary = ", ".join(f"{n} {v}" for v, n in counts.items() if n)
        lines += [f"## {title}", "", summary, "",
                  "| encoder | setting | size | bits | verdict | detail |",
                  "|---|---|---|---|---|---|"]
        for o in rows:
            detail = (o.detail + (f" (known: {o.expect})" if o.expect else "")).replace("|", "/")
            lines.append(f"| {o.encoder} | {o.label} | {o.size} | {o.depth} | {o.verdict} "
                         f"| {detail[:160]} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, required=True, help="directory of source clips")
    parser.add_argument("--bin", type=Path, default=Path(os.environ.get(
        "NC_NANOCODEC_BIN_FILE", ROOT / "target/release/nanocodec")))
    parser.add_argument("--tools", action="append", default=[], help="extra tool directory")
    parser.add_argument("--codecs", default=",".join(CODEC_TITLES))
    parser.add_argument("--only", default="", help="keep cases whose name contains this")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--decode-timeout", type=int, default=DECODE_TIMEOUT_SECONDS)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--keep-failing", type=Path, help="copy non-passing streams here")
    parser.add_argument("--list", action="store_true", help="print the cases and stop")
    args = parser.parse_args()
    args.bin = args.bin.resolve()
    codecs = set(args.codecs.split(","))
    cases = [c for c in CASES if c.codec in codecs and args.only in c.name]
    if args.list:
        print("\n".join(c.name for c in cases))
        return 0
    tools = Tools(args.tools)
    started = time.monotonic()
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case, c, args, tools) for c in cases]
        for future in concurrent.futures.as_completed(futures):
            o = future.result()
            outcomes.append(o)
            print(f"{o.verdict:15} {o.name}  {o.detail[:140]}", flush=True)
    order = {c.name: i for i, c in enumerate(cases)}
    outcomes.sort(key=lambda o: order[o.name])
    commit = os.environ.get("NC_COMMIT", "")
    header = (f"# Foreign-encoder check ({datetime.date.today()})\n\n"
              f"nanocodec {commit or args.bin}; {len(outcomes)} streams; per-frame MD5 "
              f"against the reference decoders (tools/ci/foreign_encoder_check.py).")
    report = markdown(outcomes, header)
    print("\n" + report)
    if args.markdown:
        args.markdown.write_text(report + "\n")
    if args.json:
        record = {
            "check": "foreign-encoders", "date": str(datetime.date.today()),
            "nanocodec": {"commit": commit, "binary": str(args.bin),
                          "sha256": hashlib.sha256(args.bin.read_bytes()).hexdigest()},
            "host": platform.node(), "seconds": round(time.monotonic() - started),
            "tools": tools.record(),
            "summary": {v: sum(o.verdict == v for o in outcomes)
                        for v in sorted({o.verdict for o in outcomes})},
            "streams": [asdict(o) for o in outcomes],
        }
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if all(o.verdict in ("pass", "encoder-error") for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
