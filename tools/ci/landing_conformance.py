#!/usr/bin/env python3
"""Landing conformance check: our streams decode bit-exactly everywhere.

Correctness and evaluation are separate tasks (CLAUDE.md "Testing").
Evaluation (screens, boards, sweeps, charts, tune) only encodes and scores
the encoder's reconstruction; this check is where conformance is proved, on
every change that can alter a stream (tools/dev/land.sh runs it when the
candidate reaches nc-cli, i.e. any encoder, codec, DSP or embedded-preset
change) and on demand:

    tools/ci/landing_conformance.py [--codecs h264,av1] [--workers 8] [--param a.b=1 ...]

`--param` adds encoder parameters to every case: a lever that lands off by
default is checked this way before it lands (the shipped presets alone never
switch it on).

For every shipped preset of every codec (`default` plus the codec's
presets/<codec>-*.toml) it encodes a few short streams of real content that
exercise the risky tools, then requires each to decode bit-exactly, frame
by frame, against the encoder's reconstruction of the pictures shown, with
both standard decoders and our own (nceval.conformance.gate_stream):

    H.264 JM + ffmpeg    HEVC HM + ffmpeg    VVC VVdeC + ffmpeg
    AV1 aomdec + dav1d (ffmpeg)    VP9 vpxdec + ffmpeg    AV2 avmdec + dav2d

    main     CBR at a comfortable rate, one key: the preset's coding tools
    fixedqp  `--qp` (the codec's fixed-QP profile) with a key every KEY_PERIOD frames
    starved  CBR far below the content's rate with a key every KEY_PERIOD
             frames (keys under a starved budget) over a source that
             freezes for FREEZE_FRAMES frames (frame holds)
    layers   three temporal layers in CBR, keys interrupting the mini-GOPs
    refresh  intra refresh instead of periodic keys, in CBR (not VP9/AV2)
    large    `main` on a few 720-line frames (largest blocks, long vectors)
    timed    capture timestamps with skipped captures (a live sender that cannot
             keep up, `--timestamps`), three temporal layers, keys every
             TIMED_KEY_PERIOD nominal frames of capture time (default preset)
    switch   adaptive resolution (codecs with scaled references): one key, a
             starved rate that switches to half size, then a rate rise so the
             first high-quality frame after the switch codes detail and
             predicts from the one before it, still held at full size (its
             second reference, scaled MC)

A spatial preset (two layers: presets/av1-svc*.toml, presets/vp9-svc*.toml)
is checked as SVC instead: L2T1 and L2T3 at two rates on three clips, every
operating point of each stream decoded by dav1d, aomdec and ours
(gate_av1_operating_points), or for VP9 the full stream and the base-only
stream after the superframe strip by vpxdec, ffmpeg and ours
(gate_vp9_operating_points).

No picture width is a multiple of 8 (cropping in every codec). A stream
whose bytes already passed both standard decoders (the verified-stream
cache, nceval.gatepolicy; for spatial streams every operating point,
nceval.conformance.gate_operating_points) is only decoded by ours, so an
unchanged encoder costs little. Each run codes into its own directory under the work
directory, deleted when every stream passes; a failure keeps it and prints
the command that reproduces each failing stream. The encoder binary comes
from NC_NANOCODEC_BIN (default target/release).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from nceval.codecs import Codec  # noqa: E402
from nceval.config import Paths, find_tool, load_paths  # noqa: E402
from nceval.conformance import gate_operating_points, gate_stream  # noqa: E402
from nceval.display import switched_format  # noqa: E402
from nceval.framepipe import ProcessFailed  # noqa: E402
from nceval.framestats import UNCODED_FRAME_TYPES  # noqa: E402
from nceval.metrics import StreamSpec  # noqa: E402
from nceval.point import run_point  # noqa: E402
from nceval.yuv import SwitchedFormat, YuvFormat  # noqa: E402

CODECS = (Codec.H264, Codec.HEVC, Codec.VVC, Codec.AV1, Codec.VP9, Codec.AV2)
PRESETS_DIR = REPO_ROOT / "presets"
MAX_WORKERS = 8
DEFAULT_WORKERS = 4
FPS = 30
SOURCE_STEP = 2  # every second source frame: more motion per coded frame


@dataclass(frozen=True)
class Shape:
    """A crop of a crowd scene (real motion and texture), frozen on one frame
    from `freeze_start` for `freeze_frames` frames: nothing changes there, so
    the encoder holds frames."""
    width: int
    height: int
    frames: int
    source: str
    source_size: tuple[int, int]
    origin: tuple[int, int]
    freeze_start: int = 0
    freeze_frames: int = 0

    @property
    def fmt(self) -> YuvFormat:
        return YuvFormat(self.width, self.height)


# Not a multiple of 8 in either dimension.
SMALL = Shape(330, 186, 30, "clips/main/crowd_mot17_09.yuv", (1280, 720), (400, 260), 12, 8)
# 720 lines of a 1884-wide clip, 1282 wide.
LARGE = Shape(1282, 720, 6, "clips/main/crowd_mot20_06.yuv", (1884, 720), (300, 0))
# Unaligned in both dimensions, and so is its half size (235x131 rounds up to 236x132).
SWITCH = Shape(470, 262, 41, "clips/main/crowd_mot20_06.yuv", (1884, 720), (300, 200))
# Direct 1280x720 slices read in place from clips/main/ (no extra YUV files on disk).
SVC_PAN = Shape(1280, 720, 16, "clips/main/pan_jaad_0101.yuv", (1280, 720), (0, 0))
SVC_LOWLIGHT = Shape(1280, 720, 16, "clips/main/lowlight_city_night.yuv", (1280, 720), (0, 0))
SVC_720P5_CROWD = Shape(1280, 720, 19, "clips/main/crowd_mot17_09.yuv", (1280, 720), (0, 0))
SVC_720P5_PAN = Shape(1280, 720, 19, "clips/main/pan_jaad_0101.yuv", (1280, 720), (0, 0))
SVC_720P5_LOWLIGHT = Shape(
    1280, 720, 19, "clips/main/lowlight_city_night.yuv", (1280, 720), (0, 0)
)
SPATIAL_SHAPES = {
    "crowd": SMALL,
    "pan": SVC_PAN,
    "lowlight": SVC_LOWLIGHT,
    "crowd720p5": SVC_720P5_CROWD,
    "pan720p5": SVC_720P5_PAN,
    "lowlight720p5": SVC_720P5_LOWLIGHT,
}
KEY_PERIOD = 8  # starved case: three keys inside the freeze-and-motion clip
LAYER_KEY_PERIOD = 10  # not a multiple of the 4-frame mini-GOP
SVC_KEY_PERIOD = 10
SVC_720P5_KEY_PERIOD = 12  # hits TUs 0 and 12; TUs 6 and 18 are inter 720p5 frames
REFRESH_PERIOD = 5
# Timed case: capture intervals, in nominal frames, repeated over the clip (skips of
# one and two frames between delivered ones), and the key period in nominal frames.
TIMED_GAPS = (1, 2, 1, 1, 3, 1, 2)
TIMED_KEY_PERIOD = 8
NOMINAL_MS = 1000 / FPS
# CBR rates in bits per pixel per frame: comfortable, starved, and between.
MAIN_BPP, STARVED_BPP, MID_BPP = 0.3, 0.02, 0.12
#: The fixed-QP case's quantizer per codec scale (`--qp`: the fixed-QP profile,
#: `CodecBackend::FIXED_QP_PARAMS`): mid-range, a key every KEY_PERIOD frames.
FIXED_QP = {"qp": 32, "qindex": 160}
QINDEX_CODECS = (Codec.AV1, Codec.VP9, Codec.AV2)
# Spatial SVC with --frame-drop far below the maximum-quantizer floor (about
# 30 kbps at 720p30), so rate control drops S1 and then whole TUs.
SVC_DROP_BPP = 0.0011
SVC_RATE_BPP = {
    "mid": MID_BPP,
    "starved": STARVED_BPP,
    "drop": SVC_DROP_BPP,
    "s1loss": MID_BPP,
    "s2loss": MID_BPP,
}
# Codecs without intra refresh (docs/handoff/vp9-av2.md).
NO_INTRA_REFRESH = (Codec.VP9, Codec.AV2)
# Codecs with adaptive resolution (scaled references; docs/handoff/adaptive-resolution.md).
ADAPTIVE_RESOLUTION_CODECS = (Codec.AV1,)
# Switch case: the starved rate, and the frame from which a rate rise lets the
# high-quality frame after the switch (the cadence's base frame SWITCH_RISE_FRAME
# + 7 at the start-up period of 8) code detail.
SWITCH_START_BPP, SWITCH_RISE_BPP = 0.005, 0.026
SWITCH_RISE_FRAME = 33
# Policy and cadence settings that make the short case reach that frame: decide
# on a short window, stay at half size, accept the half size on any evidence, and
# give every high-quality frame a clearly finer quantizer and the second reference.
SWITCH_PARAMS = (
    "resolution.window_seconds=0.1",
    "resolution.dwell_seconds=1",
    "resolution.down_floor_ratio=2",
    "structure.secondary_min_depth=0",
    "rate.cadence_startup_base_boost=12",
)


@dataclass(frozen=True)
class Case:
    codec: Codec
    preset: str
    kind: str

    @property
    def shape(self) -> Shape:
        if self.kind.startswith("svc-"):
            clip_tag = self.kind.split("-")[2]
            return SPATIAL_SHAPES[clip_tag]
        return {"large": LARGE, "switch": SWITCH}.get(self.kind, SMALL)

    @property
    def rate_controlled(self) -> bool:
        """Every case codes in CBR but the fixed-QP one (`--qp`, no rate log)."""
        return self.kind != "fixedqp"

    @property
    def is_spatial(self) -> bool:
        return self.kind.startswith("svc-")

    @property
    def name(self) -> str:
        return f"{self.codec.value}-{self.preset}-{self.kind}"

    def flags(self) -> list[str]:
        shape = self.shape

        def kbps(bits_per_pixel: float) -> str:
            return str(max(1, round(bits_per_pixel * shape.width * shape.height * FPS / 1000)))
        if self.kind.startswith("svc-"):
            _, mode, _, rate_tag = self.kind.split("-")
            bpp = SVC_RATE_BPP[rate_tag]
            # Drop cases gate enhancement-first frame dropping
            # (`decide_unit_drop_layers`): top layers first, then skipped TUs.
            drop = ["--frame-drop"] if rate_tag == "drop" else []
            s1_loss_tu = "6:1" if mode == "480p30_720p5" else "4:1"
            loss = (["--simulate-loss", s1_loss_tu] if rate_tag == "s1loss"
                    else ["--simulate-loss", "4:2"] if rate_tag == "s2loss"
                    else [])
            if mode == "480p30_720p5":
                return ["--preset", self.preset, "--bitrate", kbps(bpp), *drop, *loss,
                        "--intra-period", str(SVC_720P5_KEY_PERIOD),
                        "--scalability", "L2T1",
                        "--param", "spatial.base_scale=0.6666667",
                        "--param", "spatial.enhancement_interval=6"]
            if mode == "l3custom":
                return ["--preset", self.preset, "--bitrate", kbps(bpp), *drop, *loss,
                        "--intra-period", str(SVC_KEY_PERIOD),
                        "--scalability", "L3T2",
                        "--param", "spatial.s0_scale=0.3333333",
                        "--param", "spatial.s1_scale=0.6666667",
                        "--param", "layers.adaptive=0"]
            return ["--preset", self.preset, "--bitrate", kbps(bpp), *drop, *loss,
                    "--intra-period", str(SVC_KEY_PERIOD),
                    "--scalability", mode.upper(),
                    "--param", "layers.adaptive=0"]
        one_key = ["--intra-period", str(shape.frames * 10)]
        flags = {
            "main": ["--bitrate", kbps(MAIN_BPP), *one_key],
            "large": ["--bitrate", kbps(MAIN_BPP), *one_key],
            "starved": ["--bitrate", kbps(STARVED_BPP), "--intra-period", str(KEY_PERIOD)],
            "layers": ["--bitrate", kbps(MID_BPP), "--intra-period", str(LAYER_KEY_PERIOD),
                       "--temporal-layers", "3"],
            "l1t2": ["--bitrate", kbps(MID_BPP), "--intra-period", str(LAYER_KEY_PERIOD),
                     "--scalability", "L1T2", "--param", "layers.adaptive=0"],
            "l1t3": ["--bitrate", kbps(MID_BPP), "--intra-period", str(LAYER_KEY_PERIOD),
                     "--scalability", "L1T3", "--param", "layers.adaptive=0"],
            "refresh": ["--bitrate", kbps(MID_BPP), "--intra-refresh", str(REFRESH_PERIOD)],
            "timed": ["--bitrate", kbps(MID_BPP), "--intra-period", str(TIMED_KEY_PERIOD),
                      "--temporal-layers", "3"],
            "switch": ["--bitrate", kbps(SWITCH_START_BPP), *one_key, "--rate-change",
                       f"{SWITCH_RISE_FRAME}:{kbps(SWITCH_RISE_BPP)}",
                       *(arg for pair in SWITCH_PARAMS for arg in ("--param", pair))],
            "fixedqp": ["--qp", str(FIXED_QP["qindex" if self.codec in QINDEX_CODECS else "qp"]),
                        "--intra-period", str(KEY_PERIOD)],
        }[self.kind]
        return ["--preset", self.preset, *flags]


def shipped_presets(codec: Codec) -> list[str]:
    """`default` and the codec's own presets (presets/<codec>-*.toml)."""
    return ["default", *sorted(p.stem for p in PRESETS_DIR.glob(f"{codec.value}-*.toml"))]


def spatial_layers(preset: str) -> int:
    """The spatial layer count a preset codes (presets/<codec>-svc*.toml: 2 or 3; 1 for
    every other preset). A spatial preset's streams are checked at every operating
    point (the svc- cases), not as single-layer streams."""
    path = PRESETS_DIR / f"{preset}.toml"
    if not path.is_file():
        return 1
    for line in path.read_text().splitlines():
        setting = line.split("#")[0].replace(" ", "")
        if setting.startswith("spatial.layers="):
            return int(float(setting.split("=")[1]))
    return 1


def cases(codecs: list[Codec]) -> list[Case]:
    kinds = ("main", "starved", "layers", "refresh", "large", "fixedqp")
    base = [Case(codec, preset, kind) for codec in codecs for preset in shipped_presets(codec)
            if spatial_layers(preset) == 1
            for kind in kinds if not (kind == "refresh" and codec in NO_INTRA_REFRESH)]
    base += [Case(codec, "default", "timed") for codec in codecs]
    base += [Case(codec, preset, "switch") for codec in codecs
             if codec in ADAPTIVE_RESOLUTION_CODECS
             for preset in shipped_presets(codec) if spatial_layers(preset) == 1]
    # Spatial presets (av1-svc*, vp9-svc*): LnT1, LnT3, LnT1_KEY, and LnT3_KEY
    # at their own layer count n, every operating point.
    base += [Case(codec, preset, f"svc-l{layers}{mode}-{clip}-{rate}")
             for codec in codecs for preset in shipped_presets(codec)
             if (layers := spatial_layers(preset)) > 1
             for mode in ("t1", "t3", "t1_key", "t3_key")
             for clip in ("crowd", "pan", "lowlight")
             for rate in ("mid", "starved")]
    for svc_codec in (Codec.AV1, Codec.VP9):
        if svc_codec in codecs:
            base += [Case(svc_codec, "default", "l1t2"),
                     Case(svc_codec, "default", "l1t3")]
            # (L2T1..L2T3, L3T1..L3T3, L2T1_KEY..L3T3_KEY) x 3 clips x 2 rates +
            # (480p30 + 720p5) x 2 clips x 2 rates, custom 3-layer scales,
            # frame-dropping, and S1 / S2 loss repair cases.
            base += [Case(svc_codec, "default", f"svc-{mode}-{clip}-{rate}")
                     for mode in ("l2t1", "l2t2", "l2t3",
                                  "l3t1", "l3t2", "l3t3",
                                  "l2t1_key", "l2t2_key", "l2t3_key",
                                  "l3t1_key", "l3t2_key", "l3t3_key")
                     for clip in ("crowd", "pan", "lowlight")
                     for rate in ("mid", "starved")]
            base += [Case(svc_codec, "default", f"svc-480p30_720p5-{clip}-{rate}")
                     for clip in ("crowd720p5", "pan720p5")
                     for rate in ("mid", "starved")]
            base += [Case(svc_codec, "default", "svc-l2t2-pan-drop"),
                     Case(svc_codec, "default", "svc-480p30_720p5-pan720p5-drop"),
                     Case(svc_codec, "default", "svc-l2t2-crowd-s1loss"),
                     Case(svc_codec, "default", "svc-480p30_720p5-crowd720p5-s1loss"),
                     Case(svc_codec, "default", "svc-l3t2-pan-drop"),
                     Case(svc_codec, "default", "svc-l3t2-crowd-s1loss"),
                     Case(svc_codec, "default", "svc-l3t2-crowd-s2loss"),
                     Case(svc_codec, "default", "svc-l2t2_key-pan-drop"),
                     Case(svc_codec, "default", "svc-l2t2_key-crowd-s1loss"),
                     Case(svc_codec, "default", "svc-l3t2_key-pan-drop"),
                     Case(svc_codec, "default", "svc-l3t2_key-crowd-s1loss"),
                     Case(svc_codec, "default", "svc-l3t2_key-crowd-s2loss"),
                     Case(svc_codec, "default", "svc-l3custom-pan-mid")]
    return base


def source_clip(shape: Shape, paths: Paths, directory: Path) -> Path:
    """The shape's crop as a small YUV file (cached; a few MB), or the source
    file itself when read uncropped at full frame size."""
    if (shape.width, shape.height) == shape.source_size and shape.freeze_frames == 0:
        return paths.data_root / shape.source
    path = directory / f"landing_{shape.width}x{shape.height}_{shape.frames}f.yuv"
    if path.is_file():
        return path
    sw, sh = shape.source_size
    frame_bytes = sw * sh * 3 // 2
    x0, y0 = shape.origin
    out = bytearray()
    with (paths.data_root / shape.source).open("rb") as source:
        for i in range(shape.frames):
            moving = (i if i < shape.freeze_start else
                      max(shape.freeze_start - 1, i - shape.freeze_frames))
            source.seek(moving * SOURCE_STEP * frame_bytes)
            data = np.frombuffer(source.read(frame_bytes), dtype=np.uint8)
            luma = data[:sw * sh].reshape(sh, sw)
            chroma = data[sw * sh:].reshape(2, sh // 2, sw // 2)
            out += luma[y0:y0 + shape.height, x0:x0 + shape.width].tobytes()
            for plane in chroma:
                out += plane[y0 // 2:(y0 + shape.height) // 2,
                             x0 // 2:(x0 + shape.width) // 2].tobytes()
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(bytes(out))
    temporary.replace(path)
    return path


def timestamps_file(frames: int, directory: Path) -> Path:
    """A capture timestamp file (`--timestamps`, mkvmerge v2) for `frames` delivered
    frames TIMED_GAPS nominal intervals apart."""
    path = directory / f"timed_{frames}f.txt"
    if not path.is_file():
        times, at = [], 0
        for i in range(frames):
            times.append(f"{at * NOMINAL_MS:.3f}")
            at += TIMED_GAPS[i % len(TIMED_GAPS)]
        path.write_text("# timestamp format v2\n" + "\n".join(times) + "\n")
    return path


@dataclass
class Outcome:
    case: Case
    passed: bool
    reason: str = ""
    repro: str = ""
    uncoded: int = 0  # input frames held or dropped (rate log)


def run_case(case: Case, clips: dict[Shape, Path], binary: Path, paths: Paths,
             work: Path, params: tuple[str, ...] = ()) -> Outcome:
    shape, clip = case.shape, clips[case.shape]
    stream = work / f"{case.name}{case.codec.bitstream_suffix}"
    rate_log = work / f"{case.name}.rate.csv" if case.rate_controlled else None
    layer_md5 = work / f"{case.name}.layers.csv" if case.is_spatial else None
    extra = ["--layer-md5", str(layer_md5)] if layer_md5 is not None else []
    extra += [arg for pair in params for arg in ("--param", pair)]
    if case.kind == "timed":
        extra += ["--timestamps", str(timestamps_file(shape.frames, work))]

    def command(recon: str) -> list[str]:
        return [str(binary), "encode", "--codec", case.codec.value, "--input", str(clip),
                "--width", str(shape.width), "--height", str(shape.height), "--fps", str(FPS),
                "--frames", str(shape.frames), *case.flags(),
                *(["--rate-log", str(rate_log)] if rate_log else []), *extra, "--output", str(stream), "--display-recon", recon]
    repro = " ".join(command("recon.yuv"))
    spec = StreamSpec(reference=clip, fmt=shape.fmt, frames=shape.frames, fps=FPS, paths=None)
    try:
        result = run_point(command, stream, case.codec, spec, paths, own_decoder=binary,
                           rate_log=rate_log, layer_md5=layer_md5)
    except ProcessFailed as err:
        return Outcome(case, False, f"encode: {err}", repro)
    uncoded = 0
    if rate_log is not None:
        with rate_log.open(newline="") as log:
            uncoded = sum(row["type"] in UNCODED_FRAME_TYPES for row in csv.DictReader(log))
    if not result.gate.passed:
        return Outcome(case, False, result.gate.reason, repro)
    if layer_md5 is not None:
        op_gate = gate_operating_points(case.codec, stream, layer_md5, paths, own_decoder=binary)
        if not op_gate.passed:
            return Outcome(case, False, op_gate.reason, repro)
    if case.kind in ("l1t2", "l1t3"):
        t_layers = 2 if case.kind == "l1t2" else 3
        t_gate = _gate_temporal_substreams(case, stream, rate_log, result.gate.frame_hashes,
                                           t_layers, binary, paths, work)
        if t_gate:
            return Outcome(case, False, t_gate, repro)
    return Outcome(case, True, uncoded=uncoded)


def _gate_temporal_substreams(case: Case, stream: Path, rate_log: Path,
                              full_hashes: list[str], t_layers: int,
                              binary: Path, paths: Paths, work: Path) -> str:
    """Verify `nanocodec extract --temporal-layers` sub-streams for L1T2/L1T3."""
    listed = subprocess.run(
        [str(binary), "extract", "--codec", case.codec.value,
         "--temporal-layers", str(t_layers), "--list", "--input", str(stream)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if listed.returncode != 0:
        return f"extract --list failed: {listed.stderr.strip()}"
    pkt_layers = [int(line) for line in listed.stdout.splitlines() if line.strip()]
    if len(pkt_layers) != len(full_hashes):
        return f"extract --list reported {len(pkt_layers)} packets, expected {len(full_hashes)}"
    with rate_log.open(newline="") as log:
        coded_rows = [row for row in csv.DictReader(log) if row["type"] not in UNCODED_FRAME_TYPES]
    fmt = case.shape.fmt
    for max_tl in range(t_layers - 1):
        sub = work / f"{case.name}.max_t{max_tl}{case.codec.bitstream_suffix}"
        extracted = subprocess.run(
            [str(binary), "extract", "--codec", case.codec.value,
             "--temporal-layers", str(t_layers), "--max-layer", str(max_tl),
             "--input", str(stream), "--output", str(sub)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if extracted.returncode != 0:
            return f"extract --max-layer {max_tl} failed: {extracted.stderr.strip()}"
        kept_hashes = [h for h, tl in zip(full_hashes, pkt_layers) if tl <= max_tl]
        kept_sizes = [(int(r["width"]), int(r["height"]))
                      for r, tl in zip(coded_rows, pkt_layers) if tl <= max_tl]
        sub_sizes: SwitchedFormat | None = None
        if kept_sizes and any(sz != (fmt.width, fmt.height) for sz in kept_sizes):
            sw = switched_format(kept_sizes)
            sub_sizes = sw if isinstance(sw, SwitchedFormat) else SwitchedFormat(sw, ())
        sub_gate = gate_stream(sub, case.codec, fmt, len(kept_hashes), paths,
                               own_decoder=binary, coded_sizes=sub_sizes)
        if not sub_gate.passed:
            return f"max_t{max_tl}: {sub_gate.reason}"
        if sub_gate.frame_hashes != kept_hashes:
            return f"max_t{max_tl}: extracted decode differs from full stream's kept frames"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--codecs", default=",".join(c.value for c in CODECS))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--param", action="append", default=[],
                        help="name=value added to every case (a lever that lands off)")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1..{MAX_WORKERS}")
    started = time.monotonic()
    paths = load_paths()
    binary = find_tool("nanocodec", paths)
    shared = paths.work / "landing-conformance"  # the source crops, shared by runs
    shared.mkdir(parents=True, exist_ok=True)
    todo = cases([Codec(c) for c in args.codecs.split(",")])
    shapes_needed = dict.fromkeys([SMALL, LARGE, *(c.shape for c in todo)])
    clips = {shape: source_clip(shape, paths, shared) for shape in shapes_needed}
    work = Path(tempfile.mkdtemp(prefix="run-", dir=shared))
    # The large and 720p SVC cases first: the longest jobs, so the pool ends together.
    todo.sort(key=lambda case: case.shape.width < 1280)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(lambda case: run_case(case, clips, binary, paths, work,
                                                        tuple(args.param)), todo))
    failures = [o for o in outcomes if not o.passed]
    for failure in failures:
        print(f"FAIL {failure.case.name}: {failure.reason}\n  repro: {failure.repro}")
    if failures:
        print(f"failing streams kept in {work}")
    else:
        shutil.rmtree(work)
    for codec in dict.fromkeys(o.case.codec for o in outcomes):
        mine = [o for o in outcomes if o.case.codec is codec]
        presets = sorted({o.case.preset for o in mine})
        print(f"  {codec.value:<5} {sum(o.passed for o in mine)}/{len(mine)} streams, "
              f"presets {', '.join(presets)}; {sum(o.uncoded for o in mine)} held/dropped frames")
    extra = f" with {' '.join(args.param)}" if args.param else ""
    print(f"landing conformance{extra}: {len(outcomes) - len(failures)}/{len(outcomes)} streams "
          f"bit-exact in every decoder ({binary}, {time.monotonic() - started:.0f} s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
