"""Spatial scalability (SVC) chart: two- and three-layer streams on the main chart's scale.

Spatial SVC is its own optimisation target (docs/STATUS.md): one AV1 or VP9
stream carries a half-resolution base layer (operating point 1, e.g. 360p30)
and the full resolution (operating point 0, 720p30); a three-layer AV1 stream
also a quarter-resolution one under them (operating point 2, 180p30). Every point here is measured on the
main chart's clips, ladder, length and key interval (nceval.chart: natural VAL
clips, nceval.realuse.LADDER_KBPS scaled per clip and capped at half the
source rate, 150 frames, one key frame, 1 s buffer), with:

  - rate: the TOTAL stream rate, both layers (IVF framing excluded);
  - quality: J_real (PSNR-YUV, VMAF v1, MS-SSIM on every frame) of operating
    point 0 against the source; BD-J against single-layer x264 medium at the
    same total rate (nceval.chart.ANCHOR), so SVC points sit on the main
    chart's scale;
  - base layer: operating point 1 (360p) against the source downscaled to half
    size (ffmpeg Lanczos), at its own rate (both lower layers' in a three-layer
    stream), as BD-J against the same codec's Default coded at that size (its
    two-layer simulcast baseline's low stream); a three-layer stream's
    operating point 2 (180p) the same way at quarter size;
  - speed: encode CPU for both layers on one core, relative to x264 medium on
    the main chart's timing clips at the 600 kbps rung (`time`).

Points (`POINTS`):

  - ours: the `av1-svc` and `vp9-svc` presets (presets/<codec>-svc.toml) in L2T1
    and in L2T3, and their speed points `av1-svc-fast` / `vp9-svc-fast` in L2T1;
    the three-layer `av1-svc3` in L3T1 and L3T3 and its `av1-svc3-fast` in L3T1;
  - libaom `svc_encoder_rtc` (the real-time SVC example WebRTC's AV1 SVC is
    modelled on; tools/anchors `aomsvc`) in L2T1, L2T3, L3T1 and L3T3 at two
    speeds, CBR
    with its 1 s buffer, no lookahead, cyclic-refresh AQ as WebRTC sets it, no
    frame drops (every frame is scored), its encode-time test decode off;
  - libvpx `vp9_spatial_svc_encoder` (tools/anchors `vpxsvc`) in L2T1 at two
    real-time speeds, CBR, no lookahead, cyclic-refresh AQ (WebRTC's VP9
    setting), no frame drops, inter-layer prediction on;
  - simulcast: our AV1 or VP9 Default at 720p and at 360p as two independent
    streams, the rate summed (operating point 0 is the 720p stream, the base
    the 360p); and AV1 at 720p, 360p and 180p as three;
  - and, through nceval.realuse (cached like the main chart's points), the
    anchor and single-layer AV1 and VP9 Default at 720p as reference lines.

Every two-layer stream splits its total target the same way: BASE_RATE_SHARE
to the base layer (our `spatial.base_rate_share` default, which our encoder
then adapts per frame), the rest to the full-resolution layer; every
three-layer stream THREE_LAYER_RATE_SHARES to the 180p and 360p layers (our
`spatial.s0_rate_share` / `spatial.s1_rate_share` defaults).

Scoring: our SVC streams score operating point 0 from the encoder's own
reconstruction (`--display-recon`); their base layer is decoded at operating
point 1 (AV1: dav1d; VP9: vpxdec on the base-only stream, `nanocodec extract
--base-only`) and must equal the encoder's base reconstruction
(`--layer-md5`). The references' streams are decoded at each operating point
(AV1: dav1d scored, aomdec hash-compared; VP9: vpxdec scored, ffmpeg
hash-compared; libvpx codes every layer shown, so its base-only stream is each
superframe's first frame). Nothing is written but the
bitstreams, deleted once read; results are cached per point by identity.

    python3 -m nceval.svc_chart measure --binary BIN [--points a,b] [--part I/N]
    python3 -m nceval.svc_chart time --binary BIN
    python3 -m nceval.svc_chart render --binary BIN [--out DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

from . import buildinfo
from .cachefile import entry_lock, read_json, write_atomic
from .cbr import (BUFFER_MS, DEFAULT_INTRA_PERIOD, Family, Point, add_real_series,
                  anchor_families, clip_targets, summarize)
from .chart import (ANCHOR, JETSON, LOW_LADDER_KBPS, X86, ChartPoint, anchor_points,
                    binary_overrides, chart_point, fmt, low_rung_clips, pareto, per_clip_rows,
                    svg_chart)
from .clips import Clip
from .codecs import Codec
from .config import Paths, find_anchor_tool, find_tool, load_paths
from .diskguard import check_disk
from .encoders.base import fps_fraction
from .encoders.nanocodec import PRESETS_DIR
from .fingerprint import measurement_fingerprint
from .framepipe import ProcessFailed
from .framestats import coded_at_times, probe_frames, with_held_frames
from .gatepolicy import EVALUATION_GATE
from .metrics import REAL_METRICS_REVISION, StreamSpec, real_metrics
from .point import encode_streamed, run_point
from .realuse import LADDER_KBPS, Runner, main_clips, natural_val, own_family, point_terms
from .slots import cpu_slots
from .stages import balanced_parts
from .svcbase import (BASE_RATE_SHARE, THREE_LAYER_RATE_SHARES, Md5Sink, ScaledFrames,
                      base_size, decode_scored, ivf_frames, ksvc_receiver_bits, layer_size,
                      operating_point, operating_point_bits, own_base_layer, per_input,
                      split_rate, split_rate3, stream_layer_bits, vp9_extract_view,
                      vp9_superframe_sizes, write_ivf)
from .sweep import IVF_FILE_HEADER_BYTES
from .yuv import YuvFormat

#: libaom three-temporal-layer split (L2T3, L3T3): cumulative share of a spatial
#: layer's rate by temporal layer 0, 1, 2. libaom's own allocation, as its tests
#: set it (test/svc_datarate_test.cc: 60/80/100 when switching to three layers;
#: 50/70/100 for one spatial layer). WebRTC's 40/60/100, used before
#: 2026-11-13, understated libaom: on crowd_mot20_06 (150 frames, speed 10,
#: 1400 kbps total) its L2T3 720p PSNR-Y was 28.9 dB at 40/60/100, 29.6 dB at
#: 50/70/100 and 30.0 dB at 60/80/100, against 30.9 dB for L2T1.
TEMPORAL_SHARES_L2T3 = (0.6, 0.8, 1.0)
#: svc_encoder_rtc layering modes by (spatial, temporal) layers
#: (examples/svc_encoder_rtc.cc set_layer_pattern: 5 L2T1, 7 L2T3, 6 L3T1, 9 L3T3, 10 L3T3_KEY).
AOM_LAYERING_MODES = {(2, 1): 5, (2, 3): 7, (3, 1): 6, (3, 3): 9}
AOM_KEY_LAYERING_MODES = {(3, 3): 10}
#: svc_encoder_rtc's quantizer ceiling: the full range, as aomenc's default (the
#: main chart's libaom points). The example's own 52 cannot reach the low rungs:
#: pan_jaad_0101 at 600 kbps L2T1 came out at 2.4x the target (1.0x with 63).
AOM_MAX_Q = 63
#: Our preset for spatial SVC (presets/av1-svc.toml): L2T1 at 720p + 360p.
SVC_PRESET = "av1-svc"
#: Its speed point (presets/av1-svc-fast.toml): AV1 Fast's levers plus SVC-only ones.
SVC_FAST_PRESET = "av1-svc-fast"
#: The three-layer presets (presets/av1-svc3.toml, presets/av1-svc3-fast.toml):
#: L3T1 at 720p + 360p + 180p, and its speed point.
SVC3_PRESET = "av1-svc3"
SVC3_FAST_PRESET = "av1-svc3-fast"
#: The VP9 presets (presets/vp9-svc*.toml): 2-layer and 3-layer SVC and their speed points.
VP9_SVC_PRESET = "vp9-svc"
VP9_SVC_FAST_PRESET = "vp9-svc-fast"
VP9_SVC3_PRESET = "vp9-svc3"
VP9_SVC3_FAST_PRESET = "vp9-svc3-fast"
#: vp9_spatial_svc_encoder: cyclic-refresh AQ (aq-mode 3, as WebRTC's VP9 sets
#: it) and the full quantizer range, as svc_encoder_rtc (AOM_MAX_Q); its minimum
#: is WebRTC's VP9 one.
VPX_AQ_MODE = 3
VPX_MIN_Q = 2
VPX_MAX_Q = 63
#: Spatial scale factors, lowest layer first (`vp9_spatial_svc_encoder -r`), and
#: 3-temporal-layer scheme (`VP9E_TEMPORAL_LAYERING_MODE_0212`, `-tlm 3`).
VPX_SCALE_FACTORS = "1/2,1/1"
VPX_SCALE_FACTORS_3 = "1/4,1/2,1/1"
VPX_TEMPORAL_LAYERING_MODE_T3 = 3
#: The reference encoders' point kinds (every other kind is ours).
REFERENCE_KINDS = ("aom", "vpx")
#: Result layout revision: part of every SVC point's identity.
PROTOCOL = 1
WORK_SUBDIR = "svc-chart"
FRAMES = 150
DEFAULT_WORKERS = 4
OUTPUT_STEM = "svc-chart"
#: The main chart's single-layer AV1 and VP9 Default (reference lines on this chart).
SINGLE_LAYER = "nanocodec-av1"
SINGLE_LAYERS = {Codec.AV1: SINGLE_LAYER, Codec.VP9: "nanocodec-vp9"}
#: The base-layer table's anchors: each codec's Default at the base size (its
#: simulcast point's low stream).
BASE_ANCHORS = {Codec.AV1: ("nanocodec-av1-360p", "AV1 Default, 360p alone"),
                Codec.VP9: ("nanocodec-vp9-360p", "VP9 Default, 360p alone")}
#: The 180p table's anchors: each codec's Default at quarter size (the three-layer
#: simulcast point's lowest stream).
LOWEST_ANCHOR = ("nanocodec-av1-180p", "AV1 Default, 180p alone")
LOWEST_ANCHORS = {Codec.AV1: LOWEST_ANCHOR,
                  Codec.VP9: ("nanocodec-vp9-180p", "VP9 Default, 180p alone")}


@dataclass(frozen=True)
class SvcPoint:
    """One SVC chart point: `kind` ours / aom / vpx / simulcast, its temporal
    layers, the reference program's speed (aom, vpx), our preset and extra
    --param pairs, the codec, its spatial layers (2 or 3), and whether it uses
    K-SVC (`key_mode`: inter-layer prediction on key temporal units only)."""
    name: str
    label: str
    kind: str
    temporal_layers: int = 1
    speed: int = 0
    params: tuple[str, ...] = ()
    preset: str = SVC_PRESET
    codec: Codec = Codec.AV1
    layers: int = 2
    key_mode: bool = False

    @property
    def ours(self) -> bool:
        return self.kind not in REFERENCE_KINDS

    def layer_rates(self, kbps: int) -> tuple[int, ...]:
        """Each spatial layer's share of the total `kbps`, lowest first."""
        return split_rate3(kbps) if self.layers == 3 else split_rate(kbps)


POINTS = (
    SvcPoint(f"nanocodec-av1@{SVC_PRESET}", "AV1 SVC L2T1", "ours", 1),
    # A fixed L2T3 pattern, like libaom's: AV1 Default's adaptive layering
    # (`layers.adaptive`) codes moving mini-GOPs flat, which a receiver that
    # drops temporal layers cannot use.
    SvcPoint(f"nanocodec-av1@{SVC_PRESET}+L2T3", "AV1 SVC L2T3", "ours", 3,
             params=("--param", "layers.adaptive=0")),
    SvcPoint(f"nanocodec-av1@{SVC_FAST_PRESET}", "AV1 SVC Fast L2T1", "ours", 1,
             preset=SVC_FAST_PRESET),
    SvcPoint("simulcast-av1", "AV1 simulcast 720p+360p", "simulcast"),
    SvcPoint("libaom-svc-cpu8-L2T1", "libaom SVC L2T1 rt 8", "aom", 1, 8),
    SvcPoint("libaom-svc-cpu10-L2T1", "libaom SVC L2T1 rt 10", "aom", 1, 10),
    SvcPoint("libaom-svc-cpu8-L2T3", "libaom SVC L2T3 rt 8", "aom", 3, 8),
    SvcPoint("libaom-svc-cpu10-L2T3", "libaom SVC L2T3 rt 10", "aom", 3, 10),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC_PRESET}", "VP9 SVC L2T1", "ours", 1,
             preset=VP9_SVC_PRESET, codec=Codec.VP9),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC_PRESET}+L2T3", "VP9 SVC L2T3", "ours", 3,
             params=("--param", "layers.adaptive=0"), preset=VP9_SVC_PRESET, codec=Codec.VP9),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC_FAST_PRESET}", "VP9 SVC Fast L2T1", "ours", 1,
             preset=VP9_SVC_FAST_PRESET, codec=Codec.VP9),
    SvcPoint("simulcast-vp9", "VP9 simulcast 720p+360p", "simulcast", codec=Codec.VP9),
    # libvpx's real-time speeds of the main chart (libvpx-vp9-cpu7 / cpu9).
    SvcPoint("libvpx-svc-cpu7-L2T1", "libvpx SVC L2T1 rt 7", "vpx", 1, 7, codec=Codec.VP9),
    SvcPoint("libvpx-svc-cpu9-L2T1", "libvpx SVC L2T1 rt 9", "vpx", 1, 9, codec=Codec.VP9),
    SvcPoint(f"nanocodec-av1@{SVC3_PRESET}", "AV1 SVC L3T1", "ours", 1, preset=SVC3_PRESET,
             layers=3),
    SvcPoint(f"nanocodec-av1@{SVC3_PRESET}+L3T3", "AV1 SVC L3T3", "ours", 3,
             params=("--param", "layers.adaptive=0"), preset=SVC3_PRESET, layers=3),
    SvcPoint(f"nanocodec-av1@{SVC3_FAST_PRESET}", "AV1 SVC Fast L3T1", "ours", 1,
             preset=SVC3_FAST_PRESET, layers=3),
    SvcPoint("simulcast3-av1", "AV1 simulcast 720p+360p+180p", "simulcast", layers=3),
    SvcPoint("libaom-svc-cpu8-L3T1", "libaom SVC L3T1 rt 8", "aom", 1, 8, layers=3),
    SvcPoint("libaom-svc-cpu10-L3T1", "libaom SVC L3T1 rt 10", "aom", 1, 10, layers=3),
    SvcPoint("libaom-svc-cpu8-L3T3", "libaom SVC L3T3 rt 8", "aom", 3, 8, layers=3),
    SvcPoint("libaom-svc-cpu10-L3T3", "libaom SVC L3T3 rt 10", "aom", 3, 10, layers=3),
    SvcPoint(f"nanocodec-av1@{SVC_PRESET}+L2T1_KEY", "AV1 SVC L2T1_KEY", "ours", 1,
             params=("--param", "spatial.inter_layer=2"), key_mode=True),
    SvcPoint(f"nanocodec-av1@{SVC_PRESET}+L2T3_KEY", "AV1 SVC L2T3_KEY", "ours", 3,
             params=("--param", "spatial.inter_layer=2", "--param", "layers.adaptive=0"),
             key_mode=True),
    SvcPoint(f"nanocodec-av1@{SVC_FAST_PRESET}+L2T1_KEY", "AV1 SVC Fast L2T1_KEY", "ours", 1,
             params=("--param", "spatial.inter_layer=2"), preset=SVC_FAST_PRESET,
             key_mode=True),
    SvcPoint(f"nanocodec-av1@{SVC3_PRESET}+L3T1_KEY", "AV1 SVC L3T1_KEY", "ours", 1,
             params=("--param", "spatial.inter_layer=2"), preset=SVC3_PRESET, layers=3,
             key_mode=True),
    SvcPoint(f"nanocodec-av1@{SVC3_PRESET}+L3T3_KEY", "AV1 SVC L3T3_KEY", "ours", 3,
             params=("--param", "spatial.inter_layer=2", "--param", "layers.adaptive=0"),
             preset=SVC3_PRESET, layers=3, key_mode=True),
    SvcPoint(f"nanocodec-av1@{SVC3_FAST_PRESET}+L3T1_KEY", "AV1 SVC Fast L3T1_KEY", "ours", 1,
             params=("--param", "spatial.inter_layer=2"), preset=SVC3_FAST_PRESET, layers=3,
             key_mode=True),
    SvcPoint("libaom-svc-cpu8-L3T3_KEY", "libaom SVC L3T3_KEY rt 8", "aom", 3, 8, layers=3,
             key_mode=True),
    SvcPoint("libaom-svc-cpu10-L3T3_KEY", "libaom SVC L3T3_KEY rt 10", "aom", 3, 10, layers=3,
             key_mode=True),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC_PRESET}+L2T1_KEY", "VP9 SVC L2T1_KEY", "ours", 1,
             params=("--param", "spatial.inter_layer=2"), preset=VP9_SVC_PRESET,
             codec=Codec.VP9, key_mode=True),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC_PRESET}+L2T3_KEY", "VP9 SVC L2T3_KEY", "ours", 3,
             params=("--param", "spatial.inter_layer=2", "--param", "layers.adaptive=0"),
             preset=VP9_SVC_PRESET, codec=Codec.VP9, key_mode=True),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC_FAST_PRESET}+L2T1_KEY", "VP9 SVC Fast L2T1_KEY", "ours", 1,
             params=("--param", "spatial.inter_layer=2"), preset=VP9_SVC_FAST_PRESET,
             codec=Codec.VP9, key_mode=True),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC3_PRESET}", "VP9 SVC L3T1", "ours", 1,
             preset=VP9_SVC3_PRESET, codec=Codec.VP9, layers=3),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC3_PRESET}+L3T3", "VP9 SVC L3T3", "ours", 3,
             params=("--param", "layers.adaptive=0"), preset=VP9_SVC3_PRESET,
             codec=Codec.VP9, layers=3),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC3_FAST_PRESET}", "VP9 SVC Fast L3T1", "ours", 1,
             preset=VP9_SVC3_FAST_PRESET, codec=Codec.VP9, layers=3),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC3_PRESET}+L3T1_KEY", "VP9 SVC L3T1_KEY", "ours", 1,
             params=("--param", "spatial.inter_layer=2"), preset=VP9_SVC3_PRESET,
             codec=Codec.VP9, layers=3, key_mode=True),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC3_PRESET}+L3T3_KEY", "VP9 SVC L3T3_KEY", "ours", 3,
             params=("--param", "spatial.inter_layer=2", "--param", "layers.adaptive=0"),
             preset=VP9_SVC3_PRESET, codec=Codec.VP9, layers=3, key_mode=True),
    SvcPoint(f"nanocodec-vp9@{VP9_SVC3_FAST_PRESET}+L3T1_KEY", "VP9 SVC Fast L3T1_KEY", "ours", 1,
             params=("--param", "spatial.inter_layer=2"), preset=VP9_SVC3_FAST_PRESET,
             codec=Codec.VP9, layers=3, key_mode=True),
    SvcPoint("simulcast3-vp9", "VP9 simulcast 720p+360p+180p", "simulcast", codec=Codec.VP9,
             layers=3),
    # libvpx L2T1_KEY rt 7 and L3T3_KEY rt 7/9 lack x264 overlap on pan_jaad_0101 /
    # crowd_mot20_06 / lowlight_bdd_00ce95b0 (like libvpx L2T3), which would drop the
    # shared chart from 8/8 clips to 5/5; reported in spatial-svc.md §14.5 instead.
    SvcPoint("libvpx-svc-cpu9-L2T1_KEY", "libvpx SVC L2T1_KEY rt 9", "vpx", 1, 9,
             codec=Codec.VP9, key_mode=True),
    SvcPoint("libvpx-svc-cpu7-L3T1", "libvpx SVC L3T1 rt 7", "vpx", 1, 7,
             codec=Codec.VP9, layers=3),
    SvcPoint("libvpx-svc-cpu9-L3T1", "libvpx SVC L3T1 rt 9", "vpx", 1, 9,
             codec=Codec.VP9, layers=3),
    SvcPoint("libvpx-svc-cpu7-L3T1_KEY", "libvpx SVC L3T1_KEY rt 7", "vpx", 1, 7,
             codec=Codec.VP9, layers=3, key_mode=True),
    SvcPoint("libvpx-svc-cpu9-L3T1_KEY", "libvpx SVC L3T1_KEY rt 9", "vpx", 1, 9,
             codec=Codec.VP9, layers=3, key_mode=True),
)
#: The realuse-measured points: (family, label).
LINES = ((ANCHOR, "x264 medium (720p)"), (SINGLE_LAYER, "AV1 Default 720p (single layer)"),
         (SINGLE_LAYERS[Codec.VP9], "VP9 Default 720p (single layer)"))


# --- AV1 streams: libaom's framing ---------------------------------------------------

def merge_temporal_units(path: Path) -> None:
    """Rewrite an IVF file whose layers are separate frames (svc_encoder_rtc writes
    one IVF frame per layer, with the temporal unit's pts) as one IVF frame per
    temporal unit, the framing our streams and the decoders' operating-point
    selection expect (AV1 section 7.5)."""
    data = path.read_bytes()
    units: list[tuple[int, bytes]] = []
    for pts, payload in ivf_frames(data):
        if units and units[-1][0] == pts:
            units[-1] = (pts, units[-1][1] + payload)
        else:
            units.append((pts, payload))
    write_ivf(path, data[:IVF_FILE_HEADER_BYTES], units)


def _vpx_temporal_bitrates(rate: int, shares: tuple[float, ...]) -> list[int]:
    """Cumulative temporal bitrates for one spatial layer (`vp9_spatial_svc_encoder`
    requires strictly increasing positive integer targets across temporal layers)."""
    out = [max(1, round(rate * s)) for s in shares]
    out[-1] = max(rate, len(shares))
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] - 1)
    for i in range(len(out)):
        out[i] = max(out[i], (out[i - 1] + 1) if i else 1)
    return out


# --- measurement ------------------------------------------------------------------------

@dataclass
class SvcBench:
    """Measures and caches SVC chart points (one clip, one total target each)."""
    paths: Paths
    binary: Path
    presets: Path
    frames: int = FRAMES
    intra: int = DEFAULT_INTRA_PERIOD
    #: Points of ours measured on another build than `binary` (a preset whose
    #: levers landed later): point name -> binary (--binary-for).
    point_binaries: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.work = self.paths.work / WORK_SUBDIR / f"f{self.frames}-ip{self.intra}"
        self.measurement = measurement_fingerprint()
        self._hashes: dict[Path, str] = {}
        self._lock = threading.Lock()

    def binary_of(self, name: str) -> Path:
        return self.point_binaries.get(name, self.binary)

    def tool_hash(self, tool: Path) -> str:
        with self._lock:
            if tool not in self._hashes:
                self._hashes[tool] = hashlib.sha256(tool.read_bytes()).hexdigest()
            return self._hashes[tool]

    def aom_tool(self) -> Path:
        return find_anchor_tool("svc_encoder_rtc", self.paths)

    def vpx_tool(self) -> Path:
        return find_anchor_tool("vp9_spatial_svc_encoder", self.paths)

    def encoder_tool(self, point: SvcPoint) -> Path:
        """The program that codes `point`."""
        return {"aom": self.aom_tool, "vpx": self.vpx_tool}.get(
            point.kind, lambda: self.binary_of(point.name))()

    def own_params(self, preset: str | None, codec: Codec = Codec.AV1) -> tuple[str, ...]:
        """A preset's values as --param pairs (as realuse.own_family)."""
        single = SINGLE_LAYERS[codec]
        return own_family(f"{single}@{preset}" if preset else single,
                          self.binary, self.presets).extra

    def identity(self, point: SvcPoint, clip: Clip, kbps: int) -> str:
        decoder = "dav1d" if point.codec is Codec.AV1 else "vpxdec"
        parts = [PROTOCOL, self.tool_hash(self.encoder_tool(point)),
                 self.tool_hash(find_anchor_tool(decoder, self.paths)),
                 point.kind, point.temporal_layers, point.speed, point.params,
                 self.own_params(point.preset if point.kind == "ours" else None, point.codec),
                 clip.name, clip.yuv.stat().st_size, self.frames, self.intra, kbps, BUFFER_MS,
                 BASE_RATE_SHARE, self.measurement, REAL_METRICS_REVISION]
        if point.layers == 3:  # two-layer identities (and their caches) stay as they were
            parts += [point.layers, THREE_LAYER_RATE_SHARES]
        if point.kind == "aom" and point.temporal_layers == 3:
            parts += [list(TEMPORAL_SHARES_L2T3)]
        if point.key_mode:  # full-SVC identities (and their caches) stay as they were
            parts.append("key")
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]

    def result_path(self, point: SvcPoint, clip: Clip, kbps: int) -> Path:
        directory = self.work / clip.name / point.name
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"cbr{kbps}-{self.identity(point, clip, kbps)}.json"

    def cached(self, point: SvcPoint, clip: Clip, kbps: int) -> bool:
        return read_json(self.result_path(point, clip, kbps)) is not None

    def measure(self, point: SvcPoint, clip: Clip, kbps: int) -> dict:
        path = self.result_path(point, clip, kbps)
        cached = read_json(path)
        if cached is not None:
            return cached
        with entry_lock(path):
            cached = read_json(path)
            if cached is not None:
                return cached
            check_disk(self.paths.work)
            measure = {"ours": self._ours, "aom": self._aom, "vpx": self._vpx,
                       "simulcast": self._simulcast}
            with tempfile.TemporaryDirectory(prefix="nc-svc-", dir=path.parent) as scratch:
                result = measure[point.kind](point, clip, kbps, Path(scratch))
            write_atomic(path, json.dumps(result, indent=1))
            return result

    # -- the three kinds

    def own_command(self, clip: Clip, kbps: int, out: Path, preset: str | None, layers: int,
                    source: Path, size: tuple[int, int], extra: list[str],
                    binary: Path | None = None, codec: Codec = Codec.AV1) -> list[str]:
        num, den = fps_fraction(clip.fps)
        return [str(binary or self.binary), "encode", "--codec", codec.value,
                "--input", str(source),
                "--width", str(size[0]), "--height", str(size[1]), "--fps", f"{num}/{den}",
                "--frames", str(self.frames), "--intra-period", str(self.intra),
                "--temporal-layers", str(layers), "--output", str(out),
                "--bitrate", str(kbps), "--buffer-ms", str(BUFFER_MS),
                *self.own_params(preset, codec), *extra]

    def _ours(self, point: SvcPoint, clip: Clip, kbps: int, scratch: Path) -> dict:
        stream, md5, times = scratch / "s.ivf", scratch / "layers.csv", scratch / "times.txt"
        full = YuvFormat(clip.width, clip.height)
        spec = StreamSpec(clip.yuv, full, self.frames, clip.fps, self.paths)
        outcome = run_point(lambda recon: self.own_command(
            clip, kbps, stream, point.preset, point.temporal_layers, clip.yuv,
            (clip.width, clip.height),
            [*point.params, "--display-recon", recon, "--layer-md5", str(md5),
             "--timestamps-out", str(times)], self.binary_of(point.name), point.codec),
            stream, point.codec, spec, self.paths, real_metrics(spec), standard=False)
        coded = coded_at_times(times, clip.fps, self.frames)
        binary = self.binary_of(point.name)
        # The 360p operating point (op 1): the base of two layers, S1 of three.
        base = own_base_layer(stream, point.codec, binary, md5, clip, self.frames, coded,
                              scratch, self.paths, spatial_id=point.layers - 2,
                              key_mode=point.key_mode)
        lowest = own_base_layer(stream, point.codec, binary, md5, clip, self.frames, coded,
                                scratch, self.paths, steps=2,
                                key_mode=point.key_mode) if point.layers == 3 else None
        return self.result(point, clip, kbps, stream, outcome.measurement, base, coded, lowest)

    def _aom(self, point: SvcPoint, clip: Clip, kbps: int, scratch: Path) -> dict:
        stream = scratch / "s.ivf"
        with cpu_slots(1):
            returncode, stderr = run_fed(self.aom_command(point, clip, kbps, stream), clip,
                                         self.frames)
        if returncode:
            raise RuntimeError(f"{point.name} {clip.name} {kbps}: {stderr[-300:]}")
        merge_temporal_units(stream)
        top = point.layers - 1
        full, base, *lowest = (self.aom_decode_scored(stream, clip, sid, top, scratch)
                               for sid in range(top, -1, -1))
        return self.result(point, clip, kbps, stream, full, base,
                           lowest=lowest[0] if lowest else None)

    def _vpx(self, point: SvcPoint, clip: Clip, kbps: int, scratch: Path) -> dict:
        stream = scratch / "s.ivf"
        with cpu_slots(1):
            returncode, stderr = run_fed(self.vpx_command(point, clip, kbps, stream), clip,
                                         self.frames)
        if returncode:
            raise RuntimeError(f"{point.name} {clip.name} {kbps}: {stderr[-300:]}")
        top = point.layers - 1
        views = []
        for sid in range(top, -1, -1):
            if sid == top and not point.key_mode:
                view_stream = stream
            else:
                view_stream = scratch / f"view_s{sid}.ivf"
                vp9_extract_view(stream, view_stream, spatial_id=sid, key_mode=point.key_mode)
            views.append(self.vpx_decode_scored(view_stream, clip, spatial_id=sid, top=top,
                                                scratch=scratch))
        full, base, *lowest = views
        return self.result(point, clip, kbps, stream, full, base,
                           lowest=lowest[0] if lowest else None)

    def _simulcast(self, point: SvcPoint, clip: Clip, kbps: int, scratch: Path) -> dict:
        """Independent single-layer streams, one per layer size (lowest first), at the
        layers' shares of `kbps`; operating point 0 is the full-size stream."""
        binary, codec = self.binary_of(point.name), point.codec
        measured, bits = [], []
        for steps, layer_kbps in zip(range(point.layers - 1, -1, -1), point.layer_rates(kbps)):
            stream, times = scratch / f"layer{steps}.ivf", scratch / f"layer{steps}.times"
            size = layer_size(clip, steps)
            with contextlib.ExitStack() as stack:
                if steps:
                    scaled = stack.enter_context(
                        ScaledFrames(clip, self.frames, scratch, self.paths, size=size))
                    reference, source = scaled, scaled.reader
                else:
                    reference, source = clip.yuv, lambda: clip.yuv
                spec = StreamSpec(reference, YuvFormat(*size), self.frames, clip.fps,
                                  self.paths)
                measured.append(run_point(lambda recon: self.own_command(
                    clip, layer_kbps, stream, None, 1, source(), size,
                    ["--display-recon", recon, "--timestamps-out", str(times)], binary, codec),
                    stream, codec, spec, self.paths, real_metrics(spec),
                    standard=False).measurement)
            coded = coded_at_times(times, clip.fps, self.frames)
            bits.append(per_input(stream_layer_bits(stream, codec)[0], coded))
            if not steps:
                keys = [f.frame_type == "I" for f in
                        with_held_frames(probe_frames(stream, self.paths), coded)]
        totals = [sum(unit) for unit in zip(*bits)]
        # The 360p operating point is the 360p stream; a receiver of it in a
        # three-way simulcast takes that stream alone.
        lowest = (bits[0], measured[0]) if point.layers == 3 else None
        return self.summary(point, clip, kbps, totals, bits[-2], keys, measured[-1],
                            measured[-2], lowest)

    # -- decoding and summarising

    def aom_command(self, point: SvcPoint, clip: Clip, kbps: int, out: Path) -> list[str]:
        """svc_encoder_rtc in CBR (its defaults: 1 s buffer, 600 ms start and optimal
        levels, no lag), WebRTC's cyclic-refresh AQ, no frame drops, one thread, the
        full quantizer range (AOM_MAX_Q). It reads the clip's frames on stdin
        (run_fed): it has no frame limit (its --frames is listed but not parsed).
        It also writes one IVF file per layer beside `out`, removed with it."""
        num, den = fps_fraction(clip.fps)
        rates = point.layer_rates(kbps)
        layers = [round(rate * share) for rate in rates
                  for share in (TEMPORAL_SHARES_L2T3 if point.temporal_layers == 3 else (1.0,))]
        lm_table = AOM_KEY_LAYERING_MODES if point.key_mode else AOM_LAYERING_MODES
        return [str(self.aom_tool()), "-w", str(clip.width), "-h", str(clip.height),
                "-t", f"{den}/{num}", "-k", str(self.intra),
                "-sl", str(point.layers), "-tl", str(point.temporal_layers),
                "-lm", str(lm_table[(point.layers, point.temporal_layers)]),
                "-sp", str(point.speed), "-aq", "1", "-th", "1", "--drop-frame=0",
                "--error-resilient=0", "--test-decode=0", f"--max-q={AOM_MAX_Q}",
                "-b", str(sum(rates)), "-bl", ",".join(map(str, layers)),
                "-o", str(out), "-"]

    def vpx_command(self, point: SvcPoint, clip: Clip, kbps: int, out: Path) -> list[str]:
        """vp9_spatial_svc_encoder in one-pass CBR at a real-time speed: 2- or
        3-layer 2:1 downscales, Full SVC (`--inter-layer-pred=0`) or K-SVC
        (`--inter-layer-pred=2`), cyclic-refresh AQ (VPX_AQ_MODE), no lag, no
        frame drops, one thread, the full quantizer range (`-f` frames)."""
        num, den = fps_fraction(clip.fps)
        rates = point.layer_rates(kbps)
        shares = TEMPORAL_SHARES_L2T3 if point.temporal_layers == 3 else (1.0,)
        bitrates = [b for rate in rates for b in _vpx_temporal_bitrates(rate, shares)]
        total_kbps = sum(bitrates[point.temporal_layers - 1 :: point.temporal_layers])
        scale_factors = VPX_SCALE_FACTORS_3 if point.layers == 3 else VPX_SCALE_FACTORS
        tlm = (["-tlm", str(VPX_TEMPORAL_LAYERING_MODE_T3)]
               if point.temporal_layers == 3 else [])
        ilp = 2 if point.key_mode else 0
        bounds = lambda q: ",".join([str(q)] * point.layers)  # noqa: E731
        return [str(self.vpx_tool()), "-w", str(clip.width), "-h", str(clip.height),
                "-t", f"{den}/{num}", "-k", str(self.intra), "-f", str(self.frames),
                "-sl", str(point.layers), "-tl", str(point.temporal_layers), *tlm,
                "-r", scale_factors, "-sp", str(point.speed), "-aq", str(VPX_AQ_MODE),
                "-th", "1", "--rc-end-usage=1", "--lag-in-frames=0", "--drop-frame=0",
                f"--inter-layer-pred={ilp}", f"--min-q={bounds(VPX_MIN_Q)}",
                f"--max-q={bounds(VPX_MAX_Q)}", "-b", str(total_kbps),
                "-bl", ",".join(map(str, bitrates)), "-o", str(out), str(clip.yuv)]

    def layer_reference(self, clip: Clip, spatial_id: int, top: int, scratch: Path,
                        stack: contextlib.ExitStack) -> tuple[Path | ScaledFrames, YuvFormat]:
        """What a reference encoder's layer `spatial_id` (of `top + 1`) is scored
        against: the source, or its downscale to that layer's size."""
        if spatial_id == top:
            return clip.yuv, YuvFormat(clip.width, clip.height)
        size = layer_size(clip, top - spatial_id)
        return (stack.enter_context(ScaledFrames(clip, self.frames, scratch, self.paths,
                                                 size=size)), YuvFormat(*size))

    def vpx_decode_scored(self, stream: Path, clip: Clip, spatial_id: int, scratch: Path,
                          top: int = 1):
        """libvpx's layer `spatial_id` (of `top + 1`, `stream`: that receiver's
        view stream) decoded by vpxdec into the J_real metrics, checked against
        ffmpeg's decode (`ffmpeg_vp9_shown_md5s`)."""
        vpxdec = find_anchor_tool("vpxdec", self.paths)
        with contextlib.ExitStack() as stack:
            reference, fmt = self.layer_reference(clip, spatial_id, top, scratch, stack)
            measured, md5s = decode_scored(lambda out: [
                str(vpxdec), "--codec=vp9", "--rawvideo", "--i420", "--threads=1",
                "-o", out, str(stream)], reference, fmt, self.frames, clip.fps,
                [True] * self.frames, self.paths)
        all_layers = [layer_size(clip, top - sid) for sid in range(spatial_id + 1)]
        if md5s != self.ffmpeg_vp9_shown_md5s(stream, all_layers):
            raise RuntimeError(f"{stream}: spatial layer {spatial_id} decodes differ")
        return measured

    def ffmpeg_vp9_shown_md5s(self, stream: Path,
                              all_layers: list[tuple[int, int]]) -> list[str]:
        """MD5 of each temporal unit's top layer as ffmpeg decodes `stream`. ffmpeg
        outputs every frame coded with `show_frame = 1`, and libvpx codes every
        spatial layer shown (`1..=len(all_layers)` frames in a temporal unit,
        ending at `all_layers[-1]`), where vpxdec outputs the unit's last."""
        byte_sizes = [w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2) for w, h in all_layers]
        tu_counts = [len(vp9_superframe_sizes(payload) or [1])
                     for _, payload in ivf_frames(stream.read_bytes())]
        digests = []
        with cpu_slots(1):
            process = subprocess.Popen(
                [str(find_anchor_tool("ffmpeg-decode", self.paths)), "-nostdin", "-v", "error",
                 "-threads", "1", "-f", "ivf", "-c:v", "vp9", "-i", str(stream),
                 "-fps_mode", "passthrough", "-noautoscale", "-f", "rawvideo",
                 "-pix_fmt", "yuv420p", "-"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for count in tu_counts:
                sizes = byte_sizes[-count:]
                unit = [process.stdout.read(n) for n in sizes]
                if any(len(picture) != n for picture, n in zip(unit, sizes)):
                    raise RuntimeError(f"{stream}: ffmpeg ended inside a temporal unit")
                digests.append(hashlib.md5(unit[-1]).hexdigest())
            if process.stdout.read(1):
                raise RuntimeError(f"{stream}: ffmpeg produced extra pictures")
            if process.wait():
                raise RuntimeError(f"{stream}: ffmpeg failed ({process.returncode})")
        return digests

    def aom_decode_scored(self, stream: Path, clip: Clip, spatial_id: int, top: int,
                          scratch: Path):
        """libaom's operating point of spatial layer `spatial_id` (all temporal layers;
        `top`: the stream's highest) decoded by dav1d into the J_real metrics,
        checked against aomdec's decode of the same operating point."""
        operating = operating_point(stream, spatial_id)
        dav1d = find_anchor_tool("dav1d", self.paths)
        with contextlib.ExitStack() as stack:
            reference, fmt = self.layer_reference(clip, spatial_id, top, scratch, stack)
            measured, md5s = decode_scored(lambda out: [
                str(dav1d), "-q", "--threads", "1", "--filmgrain", "0",
                "--oppoint", str(operating), "--alllayers", "0", "--muxer", "yuv",
                "-i", str(stream), "-o", out], reference, fmt, self.frames, clip.fps,
                [True] * self.frames, self.paths)
        aom = Md5Sink()
        with cpu_slots(1):
            encode_streamed(lambda out: [
                str(find_tool("aomdec", self.paths)), "--rawvideo", "--i420",
                "--skip-film-grain", f"--oppoint={operating}", "-o", out,
                str(stream)], fmt, [aom])
        if md5s != aom.digests:
            raise RuntimeError(f"{stream}: operating point {operating} decodes differ")
        return measured

    def result(self, point: SvcPoint, clip: Clip, kbps: int, stream: Path, full, base,
               coded: list[bool] | None = None, lowest=None) -> dict:
        """The summary of one SVC stream; `coded`: the input frames our encoder coded
        (it holds repeats); `lowest`: a three-layer stream's 180p measurement."""
        coded = coded or [True] * self.frames
        totals, bases = stream_layer_bits(stream, point.codec)
        lowest_bits = None
        if point.layers == 3:  # full SVC carries S0+S1; K-SVC carries S0+S1 on key, S1 on inter
            lowest_bits = bases
            bases = (ksvc_receiver_bits(stream, point.codec, 1) if point.key_mode
                     else operating_point_bits(stream, point.codec, 1))
        top_bits = (per_input(ksvc_receiver_bits(stream, point.codec, point.layers - 1), coded)
                    if point.key_mode else None)
        keys = [f.frame_type == "I" for f in
                with_held_frames(probe_frames(stream, self.paths), coded)]
        return self.summary(point, clip, kbps, per_input(totals, coded),
                            per_input(bases, coded), keys, full, base,
                            (per_input(lowest_bits, coded), lowest) if lowest else None,
                            top_bits=top_bits)

    def summary(self, point: SvcPoint, clip: Clip, kbps: int, totals: list[int],
                bases: list[int], keys: list[bool], full, base, lowest=None,
                top_bits: list[int] | None = None) -> dict:
        """The realuse result layout (cbr.summarize) for operating point 0 at the total
        rate, with the 360p operating point's own result (at its own rate) under
        "base", a three-layer stream's 180p result under "lowest", and (for K-SVC)
        the 720p receiver's own downloaded stream rate under "top_view"."""
        if not len(totals) == len(bases) == len(keys) == self.frames:
            raise RuntimeError(f"{point.name} {clip.name} {kbps}: {len(totals)} temporal "
                               f"units, expected {self.frames}")
        family = Family(point.name, point.codec, self.binary_of(point.name), own=point.ours)

        def one(target: int, sizes: list[int], measured) -> dict:
            achieved = sum(sizes) * clip.fps / self.frames / 1000.0
            result = summarize(Point(family, clip, kbps=target), achieved, sizes,
                               measured.summary["psnr_yuv"], measured.per_frame,
                               measured.summary["vmaf_v1"],
                               [f["vmaf_v1"] for f in measured.per_frame])
            add_real_series(result, measured, keys)
            return result
        rates = point.layer_rates(kbps)
        result = one(kbps, totals, full)
        base_target = rates[-2] if point.key_mode else sum(rates[:-1])
        result["base"] = one(base_target, bases, base)
        if lowest:
            result["lowest"] = one(rates[0], *lowest)
        if top_bits is not None:
            result["top_view"] = one(rates[-1], top_bits, full)
        if point.ours:
            result["binary_commit"] = buildinfo.info_of(self.binary_of(point.name))["commit"]
        return result


def run_fed(cmd: list[str], clip: Clip, frames: int) -> tuple[int, str]:
    """Run `cmd` with the clip's first `frames` frames on its stdin, streamed;
    (exit status, stderr)."""
    returncode, stderr, _ = run_measured(cmd, clip, frames)
    return returncode, stderr


def run_measured(cmd: list[str], clip: Clip, frames: int) -> tuple[int, str, float]:
    """Run `cmd` (fed the clip's first `frames` frames on stdin when its input is
    "-"); (exit status, stderr, the process's own CPU seconds from its rusage:
    the feeding thread and any other process are not counted)."""
    fed = cmd[-1] == "-"
    frame_size = clip.width * clip.height * 3 // 2
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE if fed else subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def feed() -> None:
        try:
            with clip.yuv.open("rb") as source:
                for _ in range(frames):
                    process.stdin.write(source.read(frame_size))
        except BrokenPipeError:
            pass  # the encoder failed; its stderr says why
        finally:
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
    writer = threading.Thread(target=feed, daemon=True) if fed else None
    if writer:
        writer.start()
    stderr = process.stderr.read().decode(errors="replace")
    _, status, usage = os.wait4(process.pid, 0)
    process.returncode = os.waitstatus_to_exitcode(status)
    if writer:
        writer.join()
    return process.returncode, stderr, usage.ru_utime + usage.ru_stime


# --- curves -------------------------------------------------------------------------------

def chart_clips() -> list[Clip]:
    return natural_val(load_paths())


def low_ladder(clip: Clip) -> list[int]:
    return clip_targets(clip, LOW_LADDER_KBPS)


def jobs_for(points: list[SvcPoint], clips: list[Clip], ladder=None) -> list[tuple]:
    return [(p, c, kbps) for c in clips
            for kbps in (ladder(c) if ladder else clip_targets(c, LADDER_KBPS)) for p in points]


def run_jobs(bench: SvcBench, jobs: list[tuple], workers: int) -> dict:
    """{(point, clip, kbps): result}; a job our encoder refused is left out."""
    results = {}

    def one(job):
        point, clip, kbps = job
        try:
            results[(point.name, clip.name, kbps)] = bench.measure(point, clip, kbps)
        except ProcessFailed as refusal:
            if not point.ours:
                raise
            print(f"refused: {point.name} {clip.name} {kbps}: {str(refusal)[:160]}", flush=True)
        print(f"  {point.name} {clip.name} {kbps}", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, jobs))
    return results


def svc_curves(results: dict, points: list[SvcPoint], clips: list[Clip], frames: int
               ) -> tuple[dict, dict, dict, dict]:
    """(op 0 total-rate curves, 720p receiver-rate curves, 360p operating point
    curves, three-layer 180p operating point curves) as {(family, clip): [point terms]},
    rungs in rate order."""
    full: dict = {}
    top_view: dict = {}
    base: dict = {}
    lowest: dict = {}
    for point in points:
        for clip in clips:
            rows = sorted((k, r) for (n, c, k), r in results.items()
                          if n == point.name and c == clip.name)
            window = frames / clip.fps
            full[(point.name, clip.name)] = [point_terms(r, window) for _, r in rows]
            top_view[(point.name, clip.name)] = [
                point_terms(r.get("top_view", r), window) for _, r in rows]
            base[(point.name, clip.name)] = [point_terms(r["base"], window) for _, r in rows]
            if point.layers == 3:
                lowest[(point.name, clip.name)] = [point_terms(r["lowest"], window)
                                                   for _, r in rows]
    return full, top_view, base, lowest


def line_families(bench: SvcBench, paths: Paths) -> list[Family]:
    anchor = next(f for f in anchor_families(paths, Path(), extra=[ANCHOR]) if f.name == ANCHOR)
    return [anchor, *(own_family(name, bench.binary_of(name), bench.presets)
                      for name in SINGLE_LAYERS.values())]


@dataclass
class Measured:
    """Everything the chart is drawn from."""
    full: dict
    top_view: dict
    base: dict
    lowest: dict
    families: list[Family]
    labels: dict[str, str]
    low: list[Clip]
    results: dict


def measure_all(args: argparse.Namespace, clips: list[Clip], points: list[SvcPoint],
                lines: bool = True) -> Measured:
    """Measure (or read from the cache) every point on `clips`, with the main chart's
    low rungs rule: on the clips where some point lacks a reliable overlap with the
    anchor, every point and line also gets the LOW_LADDER_KBPS rungs."""
    paths = load_paths()
    binary, presets = Path(args.binary).resolve(), Path(args.presets_dir)
    bench = SvcBench(paths, binary, presets, point_binaries=binary_overrides(args.binary_for))
    runner = Runner(paths, EVALUATION_GATE, args.workers, over_budget=True,
                    stored_ms_ssim=True)
    lines_f = line_families(bench, paths) if lines else []
    curves = runner.measure("svc chart lines", lines_f, clips, FRAMES, budgeted=False) \
        if lines_f else {}
    results = run_jobs(bench, jobs_for(points, clips), args.workers)
    full, top_view, base, lowest = svc_curves(results, points, clips, FRAMES)
    curves.update(full)
    families = lines_f + [Family(p.name, p.codec, bench.binary_of(p.name), own=p.ours)
                          for p in points]
    low = low_rung_clips(curves, families, clips) if lines_f else []
    if low:
        print(f"low rungs on {', '.join(c.name for c in low)}", flush=True)
        extra = runner.measure("svc chart lines low rungs", lines_f, low, FRAMES,
                               budgeted=False, ladder=low_ladder)
        for key, terms in extra.items():
            curves[key] = curves[key] + terms
        results.update(run_jobs(bench, jobs_for(points, low, low_ladder), args.workers))
        full, top_view, base, lowest = svc_curves(results, points, clips, FRAMES)
        curves.update(full)
    labels = dict(LINES) | {p.name: p.label for p in points}
    for key in list(curves):  # rungs in rate order for the BD curves
        curves[key] = sorted(curves[key], key=lambda t: t["kbps"])
    for lower in (top_view, base, lowest):
        for key in list(lower):
            lower[key] = sorted(lower[key], key=lambda t: t["kbps"])
    return Measured(curves, top_view, base, lowest, families, labels, low, results)


def selected_points(spec: str | None) -> list[SvcPoint]:
    if not spec:
        return list(POINTS)
    by_name = {p.name: p for p in POINTS}
    unknown = [n for n in spec.split(",") if n not in by_name]
    if unknown:
        raise SystemExit(f"unknown points: {', '.join(unknown)}")
    return [by_name[n] for n in spec.split(",")]


def measure(args: argparse.Namespace) -> None:
    clips = chart_clips()
    if args.clips:
        clips = [c for c in clips if c.name in args.clips.split(",")]
    if args.part:
        index, count = (int(x) for x in args.part.split("/"))
        clips = balanced_parts(clips, count)[index - 1]
    print(f"svc chart measure: {len(clips)} clips: {', '.join(c.name for c in clips)}",
          flush=True)
    measure_all(args, clips, selected_points(args.points), lines=not args.no_lines)


# --- speed (instruction counts) -------------------------------------------------------------

#: The main chart's timing clips and rung (tools/dev/anchor_chart_timing.py).
TIMING_CLIPS = ("lowlight_city_night", "crowd_mot17_09", "bodycam_bc1")
TIMING_RUNG_KBPS = 600
TIMING_FILE = "svc-chart-timing.json"
#: The main chart's timing plan (its fixed-QP, rate-matched commands).
PLAN = Path(__file__).resolve().parents[2] / "docs/results/bd-vs-speed-plan.json"
#: Main-chart families whose instructions are counted with their timed command: the
#: anchor and each point's bridge (`bridge`).
TIMED_FAMILIES = (ANCHOR, SINGLE_LAYER, f"{SINGLE_LAYER}@av1-fast", "libaom-av1-cpu8",
                  "libaom-av1-cpu10", SINGLE_LAYERS[Codec.VP9],
                  f"{SINGLE_LAYERS[Codec.VP9]}@vp9-fast", "libvpx-vp9-cpu7", "libvpx-vp9-cpu9")
#: Each SVC speed preset's single-layer counterpart (its timed bridge).
FAST_BRIDGES = {SVC_FAST_PRESET: f"{SINGLE_LAYER}@av1-fast",
                SVC3_FAST_PRESET: f"{SINGLE_LAYER}@av1-fast",
                VP9_SVC_FAST_PRESET: f"{SINGLE_LAYERS[Codec.VP9]}@vp9-fast",
                VP9_SVC3_FAST_PRESET: f"{SINGLE_LAYERS[Codec.VP9]}@vp9-fast"}


def bridge(name: str) -> str:
    """The timed main-chart family a point's speed is estimated from: the same program
    single-layer (instructions per cycle differ far more between programs than
    between modes of one: x264 medium runs 90-145 M instructions per 720p frame at
    67 fps on the PC, our AV1 Default 100-125 M at 105 fps)."""
    if name in TIMED_FAMILIES:
        return name
    point = next(p for p in POINTS if p.name == name)
    if point.kind == "aom":
        return f"libaom-av1-cpu{point.speed}"
    if point.kind == "vpx":
        return f"libvpx-vp9-cpu{point.speed}"
    return FAST_BRIDGES.get(point.preset, SINGLE_LAYERS[point.codec]) if point.ours else \
        SINGLE_LAYERS[point.codec]
_COLLECTED = re.compile(r"Collected\s*:\s*(\d+)")


def callgrind_instructions(cmd: list[str], clip: Clip) -> int:
    """Instructions the command executes (callgrind; children not traced). A
    command reading "-" gets the clip's frames on stdin."""
    wrapped = ["valgrind", "--tool=callgrind", "--callgrind-out-file=/dev/null", *cmd]
    with cpu_slots(1):
        if cmd[-1] == "-":
            returncode, stderr = run_fed(wrapped, clip, FRAMES)
        else:
            proc = subprocess.run(wrapped, stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True)
            returncode, stderr = proc.returncode, proc.stderr
    found = _COLLECTED.search(stderr)
    if returncode or not found:
        raise RuntimeError(f"callgrind {cmd[0]}: {stderr.strip()[-300:]}")
    return int(found.group(1))


def timing_commands(bench: SvcBench, point: str, clip: Clip, kbps: int, scratch: Path,
                    scaled: dict[int, ScaledFrames]) -> list[list[str]]:
    """The encodes one timed point runs (simulcast: every stream; `scaled`: the
    clip's downscales by 2:1 steps below its size)."""
    out = scratch / f"{point}-{clip.name}.bin"
    size = (clip.width, clip.height)
    if point in TIMED_FAMILIES:  # the main chart's timed command, unchanged
        argv = json.loads(PLAN.read_text())["points"][point][clip.name]["argv"]
        tool = argv[0].removeprefix("{BIN}/")
        binary = bench.binary if tool == "nanocodec" else find_tool(tool, bench.paths)
        return [[str(binary) if i == 0 else str(out) if a == "{OUTPUT}" else
                 str(clip.yuv) if a == "{INPUT}" else a for i, a in enumerate(argv)]]
    svc = next(p for p in POINTS if p.name == point)
    if svc.kind == "aom":
        return [bench.aom_command(svc, clip, kbps, out)]
    if svc.kind == "vpx":
        return [bench.vpx_command(svc, clip, kbps, out)]
    binary = bench.binary_of(svc.name)
    if svc.kind == "ours":
        return [bench.own_command(clip, kbps, out, svc.preset, svc.temporal_layers, clip.yuv,
                                  size, list(svc.params), binary, svc.codec)]
    return [bench.own_command(clip, layer_kbps, out, None, 1,
                              scaled[steps].reader() if steps else clip.yuv,
                              layer_size(clip, steps), [], binary, svc.codec)
            for steps, layer_kbps in zip(range(svc.layers - 1, -1, -1), svc.layer_rates(kbps))]


@contextlib.contextmanager
def scaled_sources(clip: Clip, scratch: Path, paths: Paths):
    """The clip's downscales a timed simulcast point reads: {2:1 steps: frames}."""
    with contextlib.ExitStack() as stack:
        yield {steps: stack.enter_context(ScaledFrames(clip, FRAMES, scratch, paths,
                                                       size=layer_size(clip, steps)))
               for steps in (1, 2)}


#: Paired x86 timing (`time-host`): rounds per (point, clip), each the point and
#: x264 medium back to back on one reserved CPU, in alternating order (as
#: tools/dev/anchor_chart_timing.py time-host: the per-round time ratio cancels
#: the boost clock's drift with the machine's load).
HOST_ROUNDS = 7
HOST_CPUS = (14, 15)


def pinned_cpu_ms_per_frame(cmds: list[list[str]], cpu: int, clip: Clip) -> float:
    """CPU time per frame of `cmds` (one point: simulcast runs two) pinned to `cpu`."""
    total = 0.0
    for cmd in cmds:
        returncode, stderr, seconds = run_measured(["taskset", "-c", str(cpu), *cmd], clip,
                                                   FRAMES)
        if returncode:
            raise RuntimeError(f"{cmd[0]}: {stderr.strip()[-300:]}")
        total += seconds
    return total * 1000 / FRAMES


def time_host(args: argparse.Namespace) -> None:
    """Paired x86 timing of every point on the timing clips (run it under
    tools/dev/host-timed.sh): the median per-round time ratio to x264 medium."""
    paths = load_paths()
    bench = SvcBench(paths, Path(args.binary).resolve(), Path(args.presets_dir),
                     point_binaries=binary_overrides(args.binary_for))
    clips = main_clips(paths)
    names = args.points.split(",") if args.points else \
        [SINGLE_LAYER, *(p.name for p in POINTS)]
    jobs = [(n, clips[c]) for n in names for c in TIMING_CLIPS]
    out = Path(args.out) / TIMING_FILE
    record = json.loads(out.read_text()) if out.exists() else {}
    paired = record.setdefault("x86_paired", {"method": (
        f"CPU time (rusage) on one reserved core of the PC, {HOST_ROUNDS} rounds of the point "
        "and x264 medium back to back in alternating order; median per-round time ratio"),
        "points": {}})
    lock = threading.Lock()
    free = list(map(int, args.cpus.split(",")))

    def one(job):
        name, clip = job
        with lock:
            cpu = free.pop()
        scratch = Path(tempfile.mkdtemp(prefix="nc-svc-host-", dir=paths.work))
        try:
            with scaled_sources(clip, scratch, paths) as scaled:
                kbps = clip_targets(clip, (TIMING_RUNG_KBPS,))[0]
                anchor = timing_commands(bench, ANCHOR, clip, kbps, scratch, scaled)
                runs = {"anchor": anchor, "point": None}
                samples: dict[str, list[float]] = {"anchor": [], "point": []}
                for index in range(HOST_ROUNDS + 1):  # round 0 warms up
                    order = ["point", "anchor"] if index % 2 else ["anchor", "point"]
                    for role in order:
                        # a fresh FIFO per run: simulcast reads the 360p source through one
                        cmds = runs[role] or timing_commands(bench, name, clip, kbps, scratch,
                                                             scaled)
                        ms = pinned_cpu_ms_per_frame(cmds, cpu, clip)
                        if index:
                            samples[role].append(ms)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            with lock:
                free.append(cpu)
        ratio = statistics.median(p / a for p, a in zip(samples["point"], samples["anchor"]))
        with lock:
            entry = paired["points"].setdefault(name, {"time_ratio": {}, "ms_per_frame": {}})
            entry["time_ratio"][clip.name] = ratio
            entry["ms_per_frame"][clip.name] = statistics.median(samples["point"])
        print(f"  {name} {clip.name}: {ratio:.3f}x x264 medium's time", flush=True)
    with ThreadPoolExecutor(len(free)) as pool:
        list(pool.map(one, jobs))
    paired["binary_commit"] = buildinfo.info_of(bench.binary)["commit"]
    paired["date"] = datetime.date.today().isoformat()
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}")


#: The SVC timing plan for the Jetson (tools/dev/anchor_chart_timing.py `time`,
#: which reads 150-frame copies of the timing clips there): x264 medium's
#: command from the main plan, and every two-layer point but simulcast (two
#: encodes, one of a 360p source the Jetson does not store).
PLAN_FILE = "svc-chart-plan.json"


def plan_argv(bench: SvcBench, point: SvcPoint, clip: Clip, kbps: int) -> list[str]:
    """A point's command with the plan's {BIN}/{INPUT}/{OUTPUT} placeholders: our
    encoder by --preset (each build applies its own), libaom and libvpx reading a file."""
    if point.kind == "aom":
        argv = bench.aom_command(point, clip, kbps, Path("{OUTPUT}"))
        return ["{BIN}/svc_encoder_rtc", *argv[1:-1], "{INPUT}"]
    if point.kind == "vpx":
        argv = bench.vpx_command(point, clip, kbps, Path("{OUTPUT}"))
        return ["{BIN}/vp9_spatial_svc_encoder", *argv[1:-1], "{INPUT}"]
    num, den = fps_fraction(clip.fps)
    return ["{BIN}/nanocodec", "encode", "--codec", point.codec.value, "--input", "{INPUT}",
            "--width", str(clip.width), "--height", str(clip.height), "--fps", f"{num}/{den}",
            "--frames", str(FRAMES), "--intra-period", str(bench.intra),
            "--temporal-layers", str(point.temporal_layers), "--output", "{OUTPUT}",
            "--bitrate", str(kbps), "--buffer-ms", str(BUFFER_MS), "--preset", point.preset,
            *point.params]


def write_plan(args: argparse.Namespace) -> None:
    """svc-chart-plan.json: the Jetson timing plan, our streams' sizes from this
    machine (bit-exact across platforms: `time` checks them)."""
    paths = load_paths()
    bench = SvcBench(paths, Path(args.binary).resolve(), Path(args.presets_dir),
                     point_binaries=binary_overrides(args.binary_for))
    clips = main_clips(paths)
    main_plan = json.loads(PLAN.read_text())["points"]
    plan = {"anchor": ANCHOR, "points": {ANCHOR: main_plan[ANCHOR]}}
    for point in (p for p in POINTS if p.kind != "simulcast"):
        entry = plan["points"].setdefault(point.name, {})
        for name in TIMING_CLIPS:
            clip = clips[name]
            kbps = clip_targets(clip, (TIMING_RUNG_KBPS,))[0]
            argv = plan_argv(bench, point, clip, kbps)
            with tempfile.TemporaryDirectory(dir=paths.work) as tmp:
                out = Path(tmp) / "s.ivf"
                cmd = [str(bench.binary_of(point.name)) if a == "{BIN}/nanocodec" else
                       "-" if a == "{INPUT}" and point.kind == "aom" else
                       str(clip.yuv) if a == "{INPUT}" else str(out) if a == "{OUTPUT}" else
                       a for a in argv]
                if not point.ours:
                    cmd[0] = str(bench.encoder_tool(point))
                else:  # the preset file itself: this build may predate its registration
                    at = cmd.index("--preset")
                    cmd[at:at + 2] = ["--params", str(bench.presets / f"{point.preset}.toml")]
                returncode, stderr = run_fed(cmd, clip, FRAMES)
                if returncode:
                    raise RuntimeError(f"{point.name} {name}: {stderr[-300:]}")
                size = out.stat().st_size
            entry[name] = {"argv": argv, "bytes": size, "fps": clip.fps,
                           "target_bytes": round(kbps * 1000 / 8 * FRAMES / clip.fps)}
    out = Path(args.out) / PLAN_FILE
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(f"wrote {out}")


def merge_jetson(args: argparse.Namespace) -> None:
    """Merge the Jetson's per-point timing files (anchor_chart_timing.py `time` with
    the SVC plan: medians of the point and of x264 medium per clip) into the
    timing file as `jetson_paired`."""
    out = Path(args.out) / TIMING_FILE
    record = json.loads(out.read_text())
    paired = record.setdefault("jetson_paired", {"method": (
        "CPU time on the Jetson's core 3 at 1.728 GHz (jetson-timed.sh), the point and "
        "x264 medium in rounds (tools/dev/anchor_chart_timing.py time, svc-chart-plan.json); "
        "time ratio of the medians"), "points": {}})
    for path in args.files:
        timed = json.loads(Path(path).read_text())
        clips = timed["clips"]
        paired["points"][timed["point"]] = {
            "time_ratio": {c: v["ms_per_frame"]["point"] / v["ms_per_frame"]["anchor"]
                           for c, v in clips.items()},
            "ms_per_frame": {c: v["ms_per_frame"]["point"] for c, v in clips.items()},
            "nanocodec_sha256": timed["nanocodec_sha256"]}
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"merged {len(args.files)} Jetson timings into {out}")


def time_points(args: argparse.Namespace) -> None:
    """Instructions per frame of every point (both layers) and of x264 medium on the
    timing clips: the speed axis while the machines are not quiet enough to time."""
    paths = load_paths()
    bench = SvcBench(paths, Path(args.binary).resolve(), Path(args.presets_dir),
                     point_binaries=binary_overrides(args.binary_for))
    clips = main_clips(paths)
    names = args.points.split(",") if args.points else \
        [*TIMED_FAMILIES, *(p.name for p in POINTS)]
    jobs = [(n, clips[c]) for c in TIMING_CLIPS for n in names]
    out = Path(args.out) / TIMING_FILE
    # Points not re-counted keep their stored counts (their binary did not change).
    stored = json.loads(out.read_text()) if out.exists() else {}
    counts: dict[str, dict[str, float]] = {
        n: c for n, c in stored.get("instructions_per_frame", {}).items() if n not in names}
    lock = threading.Lock()

    def one(job):
        name, clip = job
        scratch = Path(tempfile.mkdtemp(prefix="nc-svc-time-", dir=paths.work))
        try:
            with scaled_sources(clip, scratch, paths) as scaled:
                kbps = clip_targets(clip, (TIMING_RUNG_KBPS,))[0]
                total = sum(callgrind_instructions(cmd, clip) for cmd in
                            timing_commands(bench, name, clip, kbps, scratch, scaled))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        with lock:
            counts.setdefault(name, {})[clip.name] = total / FRAMES
        print(f"  {name} {clip.name}: {total / FRAMES / 1e6:.1f} M instr/frame", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, jobs))
    record = {**stored, "method": "callgrind instructions per frame (x86-64, AVX2 build), both layers",
              "rung_kbps": TIMING_RUNG_KBPS, "frames": FRAMES, "clips": list(TIMING_CLIPS),
              "binary_commit": buildinfo.info_of(bench.binary)["commit"],
              "date": datetime.date.today().isoformat(),
              "instructions_per_frame": dict(sorted(counts.items()))}
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}")


# --- render -------------------------------------------------------------------------------

def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


TIMING_FILES = {JETSON.column: "bd-vs-speed-timing.json", X86.column: "bd-vs-speed-timing-x86.json"}


def timed_fps(results: Path, column: str) -> dict[str, float]:
    """Each main-chart family's timed fps on one platform (geomean over its clips)."""
    points = json.loads((results / TIMING_FILES[column]).read_text())["points"]
    return {name: 1000.0 / geomean(list(entry["ms_per_frame"].values()))
            for name, entry in points.items()}


def estimated_fps(timing: dict, timed: dict[str, float], name: str) -> float | None:
    """A point's fps on one platform: its bridge's timed fps times the bridge's
    instructions over the point's (geomean over the timing clips)."""
    counts = timing["instructions_per_frame"]
    reference = bridge(name)
    if name not in counts or reference not in counts or reference not in timed:
        return None
    ratio = geomean([counts[reference][c] / counts[name][c] for c in counts[name]])
    return timed[reference] * ratio


#: Paired timings in the timing file, by platform (`time-host`; the Jetson's from
#: tools/dev/anchor_chart_timing.py `time` with the SVC plan, merged in).
PAIRED_KEYS = {X86.column: "x86_paired", JETSON.column: "jetson_paired"}


def with_speed(points: list[ChartPoint], timing: dict | None, timed: dict[str, float],
               column: str) -> tuple[list[ChartPoint], set[str]]:
    """Points with fps and speed vs x264 medium on one platform: paired-timed where
    the timing file has it (x264 medium's timed fps over the median per-round time
    ratio, the main chart's footing), else estimated from instructions
    (`estimated_fps`); the main chart's own families keep their timed fps. Returns the
    points and the families estimated."""
    if timing is None:
        return points, set()
    paired = timing.get(PAIRED_KEYS[column], {}).get("points", {})
    out, estimated = [], set()
    for p in points:
        if p.family in paired:
            speed = 1 / geomean(list(paired[p.family]["time_ratio"].values()))
            fps = timed[ANCHOR] * speed
        elif p.family in timed:  # a main-chart family (the anchor, a single-layer line)
            fps, speed = timed[p.family], timed[p.family] / timed[ANCHOR]
        else:
            fps = estimated_fps(timing, timed, p.family)
            speed = fps / timed[ANCHOR] if fps else None
            estimated.add(p.family)
        out.append(replace(p, fps=fps, speed=speed))
    return out, estimated


def top_view_points(measured: Measured, clips: list[Clip], points: list[SvcPoint]
                    ) -> list[ChartPoint]:
    """Each SVC point's 720p receiver view (operating point 0 at the wire rate its
    receiver downloads: all layers in full SVC, `{S0..=S_top}` on key units and
    `{S_top}` alone on non-key units in K-SVC) vs its codec's single-layer Default
    at 720p."""
    curves = dict(measured.top_view)
    for codec, single in SINGLE_LAYERS.items():
        for clip in clips:
            curves[(single, clip.name)] = measured.full[(single, clip.name)]
    return lower_points(curves, clips, [p for p in points if p.kind != "simulcast"],
                        lambda p: SINGLE_LAYERS[p.codec])


def base_points(measured: Measured, clips: list[Clip], points: list[SvcPoint]
                ) -> list[ChartPoint]:
    """Each point's 360p operating point (operating point 1, at its own rate) vs its
    codec's Default coded alone at that size (the two-layer simulcast's low stream)."""
    curves = dict(measured.base)
    for simulcast in (p for p in points if p.kind == "simulcast" and p.layers == 2):
        for clip in clips:
            curves[(BASE_ANCHORS[simulcast.codec][0], clip.name)] = \
                measured.base[(simulcast.name, clip.name)]
    return lower_points(curves, clips, [p for p in points if p.kind != "simulcast"],
                        lambda p: BASE_ANCHORS[p.codec][0])


def lowest_points(measured: Measured, clips: list[Clip], points: list[SvcPoint]
                  ) -> list[ChartPoint]:
    """Each three-layer point's 180p operating point (operating point 2, at its own
    rate) vs its codec's Default coded alone at that size (the three-way simulcast's
    lowest stream)."""
    curves = dict(measured.lowest)
    for simulcast in (p for p in points if p.kind == "simulcast" and p.layers == 3):
        for clip in clips:
            curves[(LOWEST_ANCHORS[simulcast.codec][0], clip.name)] = \
                measured.lowest[(simulcast.name, clip.name)]
    return lower_points(curves, clips,
                        [p for p in points if p.kind != "simulcast" and p.layers == 3],
                        lambda p: LOWEST_ANCHORS[p.codec][0])


def lower_points(curves: dict, clips: list[Clip], points: list[SvcPoint], anchor
                 ) -> list[ChartPoint]:
    """BD rows of `points`' lower-layer `curves` against `anchor(point)`'s curve."""
    runner = SimpleNamespace(unreachable={}, refused={})
    out = []
    for p in points:
        family = Family(p.name, p.codec, Path(), own=p.ours)
        rows = per_clip_rows(curves, anchor(p), p.name, clips)
        out.append(chart_point(family, p.label, rows, clips, runner))
    return out


def base_summary(measured: Measured, name: str, key: str = "base"
                 ) -> tuple[float, float, float]:
    """(mean share of the total rate, mean PSNR-YUV, mean VMAF v1) of a point's
    operating point `key` ("base": 360p; "lowest": a three-layer 180p;
    "top_view": 720p receiver's downloaded view)."""
    rows = [r for (n, _, _), r in measured.results.items() if n == name]
    sub = [r.get(key, r) if key == "top_view" else r[key] for r in rows]
    share = sum(s["kbps"] / r["kbps"] for s, r in zip(sub, rows)) / len(rows)
    return (share, sum(s["psnr_yuv"] for s in sub) / len(sub),
            sum(s["vmaf_v1"] for s in sub) / len(sub))


def markdown(points: list[ChartPoint], x86: list[ChartPoint], top_views: list[ChartPoint],
             bases: list[ChartPoint], lowest: list[ChartPoint], measured: Measured,
             timing: dict | None, excluded, meta: str,
             estimated: dict[str, set[str]]) -> str:
    on_x86 = {p.family: p for p in x86}

    def speed_cell(value: float | None, family: str, column: str, spec: str, unit: str) -> str:
        mark = "*" if family in estimated[column] else ""
        return f"{fmt(value, spec, unit)}{mark}"
    by_base = {p.family: p for p in bases}
    lines = ["# Spatial scalability (SVC): 720p30 + 360p30 (+ 180p30)\n",
             f"<sub>{meta}</sub>\n",
             "![SVC chart](svc-chart.svg)\n",
             "Two-layer AV1 and VP9 streams (360p base, operating point 1; 720p, operating "
             "point 0) and three-layer AV1 streams (180p, operating point 2; 360p, 1; 720p, "
             "0) on the main chart's clips, rungs, length and key interval. **Rate is "
             "the total stream rate (every layer); quality is J_real of the 720p output**, "
             "so BD-J is against single-layer x264 medium at 720p, on the same scale as "
             "[the main chart](bd-vs-speed.md). The single-layer AV1 and VP9 Default lines "
             "show what the lower layers cost at 720p.\n",
             "| Encoder | BD-J op 0 | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | "
             "360p op BD-J | 360p op share of rate | fps (Jetson, 1 core) | fps (x86, 1 core) "
             "| x86 speed vs x264 medium | clips |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for p in sorted(points, key=lambda p: (p.bd_j is None, p.bd_j or 0)):
        name = f"**{p.label}**" if p.ours else p.label
        iqr = f"{p.iqr[0]:+.1f} … {p.iqr[1]:+.1f}" if p.iqr else "n/a"
        base = by_base.get(p.family)
        share = f"{base_summary(measured, p.family)[0] * 100:.0f}%" \
            if p.family in {n for n, _, _ in measured.results} else ""
        lines.append(f"| {name} | {fmt(p.bd_j)} | {fmt(p.bd_psnr)} | {fmt(p.bd_vmaf)} | {iqr} | "
                     f"{fmt(base.bd_j) if base else ''} | {share} | "
                     f"{speed_cell(p.fps, p.family, JETSON.column, '.1f', '')} | "
                     f"{speed_cell(on_x86[p.family].fps, p.family, X86.column, '.1f', '')} | "
                     f"{speed_cell(on_x86[p.family].speed, p.family, X86.column, '.2f', '×')} | "
                     f"{p.scored}/{p.clips} |")
    method = timing["method"] if timing else "not measured"
    lines += ["",
              "Speed: x264 medium's timed fps (the main chart's) over the point's median "
              "per-round time ratio to x264 medium, timed in pairs on one reserved core "
              f"(`time-host`, the {TIMING_RUNG_KBPS} kbps rung, {', '.join(TIMING_CLIPS)}; "
              "the Jetson's with the same pairing on its core 3).\n",
              f"\\* Estimated, not timed (no paired timing on that platform yet). Each point's fps "
              f"is the main chart's timed fps of the same program coded single-layer (ours: "
              f"the codec's Default, or its Fast for an SVC Fast point; libaom: libaom rt at "
              f"the same speed, aomenc; libvpx: vpxenc rt at the same speed) times that "
              f"command's instructions over the point's ({method}; ours and the references at the "
              f"{TIMING_RUNG_KBPS} kbps rung in CBR, the single-layer commands the main "
              f"chart's rate-matched fixed-QP ones; {', '.join(TIMING_CLIPS)}). Ratios within "
              f"one program carry across machines far better than between programs; the "
              f"Jetson column applies the x86 ratio to the Jetson's timed fps.\n",
              "## 360p operating point (operating point 1)\n",
              "360p output against the source downscaled to 360p (ffmpeg Lanczos), at the "
              "operating point's own rate (what a 360p receiver gets: the base layer of two; "
              "the 180p and 360p layers of three in full SVC, or `{S0, S1}` on key units and "
              "`{S1}` alone on non-key units in KEY modes), as BD-J against the same codec's Default "
              "coded alone at 360p (the two-layer simulcast's low stream): AV1 points against "
              "AV1 Default, VP9 points against VP9 Default.\n",
              "| Encoder | base BD-J | base BD-PSNR-YUV | base BD-VMAF v1 | base share of total "
              "rate | mean base PSNR-YUV | mean base VMAF v1 | clips |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    codec_of = {p.name: p.codec for p in POINTS}
    for codec in BASE_ANCHORS:
        for b in sorted((b for b in bases if codec_of[b.family] is codec),
                        key=lambda p: (p.bd_j is None, p.bd_j or 0)):
            share, psnr, vmaf = base_summary(measured, b.family)
            lines.append(f"| {b.label} | {fmt(b.bd_j)} | {fmt(b.bd_psnr)} | {fmt(b.bd_vmaf)} | "
                         f"{share * 100:.0f}% | {psnr:.2f} dB | {vmaf:.1f} | "
                         f"{b.scored}/{b.clips} |")
        simulcast = next(p for p in POINTS
                         if p.kind == "simulcast" and p.codec is codec and p.layers == 2)
        share, psnr, vmaf = base_summary(measured, simulcast.name)
        lines.append(f"| {BASE_ANCHORS[codec][1]} | +0.0% | +0.0% | +0.0% | "
                     f"{share * 100:.0f}% | {psnr:.2f} dB | {vmaf:.1f} | |")
    lines += ["", "## 180p operating point of three layers (operating point 2)\n",
              "180p output against the source downscaled to 180p (ffmpeg Lanczos), at the "
              "180p layer's own rate (what a thumbnail receiver gets), as BD-J against the "
              "same codec's Default coded alone at 180p (the three-way simulcast's lowest "
              "stream): AV1 points against AV1 Default, VP9 points against VP9 Default.\n",
              "| Encoder | 180p BD-J | 180p BD-PSNR-YUV | 180p BD-VMAF v1 | 180p share of "
              "total rate | mean 180p PSNR-YUV | mean 180p VMAF v1 | clips |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for codec in LOWEST_ANCHORS:
        for b in sorted((b for b in lowest if codec_of[b.family] is codec),
                        key=lambda p: (p.bd_j is None, p.bd_j or 0)):
            share, psnr, vmaf = base_summary(measured, b.family, "lowest")
            lines.append(f"| {b.label} | {fmt(b.bd_j)} | {fmt(b.bd_psnr)} | {fmt(b.bd_vmaf)} | "
                         f"{share * 100:.0f}% | {psnr:.2f} dB | {vmaf:.1f} | "
                         f"{b.scored}/{b.clips} |")
        simulcast3 = next(p for p in POINTS
                          if p.kind == "simulcast" and p.codec is codec and p.layers == 3)
        share, psnr, vmaf = base_summary(measured, simulcast3.name, "lowest")
        lines.append(f"| {LOWEST_ANCHORS[codec][1]} | +0.0% | +0.0% | +0.0% | "
                     f"{share * 100:.0f}% | {psnr:.2f} dB | {vmaf:.1f} | |")
    lines += ["", "## 720p receiver view (operating point 0 at its own downloaded rate)\n",
              "720p output against the source at the wire rate a 720p receiver downloads "
              "(all layers in full SVC; `{S0..=S_top}` on key units and `{S_top}` alone on "
              "non-key units in KEY modes), as BD-J against the same codec's single-layer "
              "Default at 720p.\n",
              "| Encoder | 720p view BD-J | 720p view BD-PSNR-YUV | 720p view BD-VMAF v1 | "
              "720p share of total rate | mean 720p PSNR-YUV | mean 720p VMAF v1 | clips |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for codec, single in SINGLE_LAYERS.items():
        for t in sorted((t for t in top_views if codec_of[t.family] is codec),
                        key=lambda p: (p.bd_j is None, p.bd_j or 0)):
            share, psnr, vmaf = base_summary(measured, t.family, "top_view")
            lines.append(f"| {t.label} | {fmt(t.bd_j)} | {fmt(t.bd_psnr)} | {fmt(t.bd_vmaf)} | "
                         f"{share * 100:.0f}% | {psnr:.2f} dB | {vmaf:.1f} | "
                         f"{t.scored}/{t.clips} |")
        single_label = dict(LINES)[single]
        lines.append(f"| {single_label} | +0.0% | +0.0% | +0.0% | 100% | | | |")
    lines += ["", "## Method\n",
              f"- Every two-layer point targets the total rung with {BASE_RATE_SHARE:.0%} "
              "for the base layer (our `spatial.base_rate_share` default; our encoder adapts "
              "it per frame, libaom and simulcast keep it); every three-layer point with "
              f"{THREE_LAYER_RATE_SHARES[0]:.0%} for 180p and {THREE_LAYER_RATE_SHARES[1]:.0%} "
              "for 360p (our `spatial.s0_rate_share` / `spatial.s1_rate_share` defaults: "
              "caps our encoder passes unspent bits up from).",
              "- Three layers (AV1): `presets/av1-svc3.toml` (AV1 Default plus "
              "`spatial.layers = 3`, 2:1 steps) in L3T1 and L3T3 (`layers.adaptive=0`), "
              "`presets/av1-svc3-fast.toml` (av1-svc-fast's levers) in L3T1; libaom layering "
              "modes 6 (L3T1) and 9 (L3T3), its own 4:2:1 downscales; simulcast three "
              "independent AV1 Default streams. The 360p operating point is decoded at "
              "operating point 1 (dav1d), the 180p at operating point 2, each equal to the "
              "encoder's reconstruction of that layer (aomdec's for libaom).",
              "- Ours: `presets/av1-svc.toml` / `presets/vp9-svc.toml` (the codec's Default "
              "plus `spatial.layers = 2`), `--temporal-layers 1` (L2T1) or `3` (L2T3, with "
              "`layers.adaptive=0`), and `presets/av1-svc-fast.toml` / "
              "`presets/vp9-svc-fast.toml` (the codec's Fast levers, a 2x2-mean base downscale "
              "and a bilinear search proxy) in L2T1; CBR, 1 s buffer. Our temporal layers "
              "have no fixed rate split: the rate controller sets each temporal "
              "layer's quantizer step relative to the base from a tuned ladder (`layers.cbr_base_step_ratio`, "
              "`layers.cbr_middle_step_ratio`). Operating point 0 "
              "is scored from the encoder's reconstruction; the base layer is decoded at "
              "operating point 1 (AV1: dav1d; VP9: vpxdec on `nanocodec extract --base-only`'s "
              "stream) and must equal the encoder's base reconstruction.",
              "- libaom: `examples/svc_encoder_rtc` of the pinned libaom "
              "(tools/anchors `aomsvc`), layering mode 5 (L2T1) or 7 (L2T3), CBR with its "
              "1 s buffer (600 ms start/optimal), no lag, cyclic-refresh AQ (`-aq 1`, as "
              "WebRTC), no frame drops, one thread, its own 2:1 downscale; L2T3 and L3T3 "
              "temporal split "
              f"{'/'.join(f'{s * 100:.0f}' for s in TEMPORAL_SHARES_L2T3)} (cumulative, "
              "libaom's own test allocation). Decoded by dav1d (scored) and aomdec (equal hashes) at each "
              "operating point.",
              "- libvpx: `examples/vp9_spatial_svc_encoder` of the pinned libvpx v1.16.0 "
              f"(tools/anchors `vpxsvc`), L2T1 only (`-sl 2 -r {VPX_SCALE_FACTORS}`; no temporal "
              "split), one-pass CBR "
              f"(`--rc-end-usage=1`), no lag, cyclic-refresh AQ (`-aq {VPX_AQ_MODE}`, as "
              "WebRTC's VP9), no frame drops, inter-layer prediction on, one thread, "
              f"quantizer {VPX_MIN_Q}-{VPX_MAX_Q}. Its layers are all shown frames, so the "
              "base-only stream is each superframe's first frame. Decoded by vpxdec "
              "(scored) and ffmpeg (equal hashes) at each operating point.",
              "- Simulcast: the codec's Default at 720p and, on the ffmpeg-Lanczos 360p "
              "downscale, at 360p, two independent streams with the rate summed per frame.",
              "- Rate validity flags (realuse.rate_valid) are not shown: our points carry no "
              "key-loan adjustment here."]
    if excluded:
        lines.append(f"- Clips left out for every point (no reliable overlap): "
                     f"{', '.join(e.clip for e in excluded)}.")
    if measured.low:
        lines.append(f"- Low rungs ({'/'.join(map(str, LOW_LADDER_KBPS))} kbps at 720p30, "
                     f"every point) on: {', '.join(c.name for c in measured.low)}.")
    return "\n".join(lines) + "\n"


def csv_rows(points: list[ChartPoint], x86: list[ChartPoint], bases: list[ChartPoint],
             lowest: list[ChartPoint], measured: Measured) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["family", "label", "ours", "bd_j_op0_pct", "bd_psnr_yuv_pct",
                     "bd_vmaf_v1_pct", "iqr_low_pct", "iqr_high_pct", "base_bd_j_pct",
                     "base_share", "op180_bd_j_pct", "fps_jetson_est", "fps_x86_est", "speed_vs_anchor_instr",
                     "clips_scored", "clips"])
    on_x86 = {p.family: p for p in x86}
    by_base = {p.family: p for p in bases}
    by_lowest = {p.family: p for p in lowest}
    names = {n for n, _, _ in measured.results}
    num = (lambda v, d: "" if v is None else f"{v:.{d}f}")
    for p in points:
        base = by_base.get(p.family)
        low = by_lowest.get(p.family)
        share = base_summary(measured, p.family)[0] if p.family in names else None
        writer.writerow([p.family, p.label, int(p.ours), num(p.bd_j, 2), num(p.bd_psnr, 2),
                         num(p.bd_vmaf, 2), num(p.iqr and p.iqr[0], 2),
                         num(p.iqr and p.iqr[1], 2), num(base and base.bd_j, 2),
                         num(share, 3), num(low and low.bd_j, 2), num(p.fps, 2), num(on_x86[p.family].fps, 2),
                         num(p.speed, 3), p.scored, p.clips])
    return out.getvalue()


def render(args: argparse.Namespace) -> None:
    clips = chart_clips()
    measured = measure_all(args, clips, list(POINTS))
    runner = SimpleNamespace(unreachable={}, refused={})
    points, excluded = anchor_points(measured.full, measured.families, clips, runner,
                                     measured.labels)
    out = Path(args.out)
    timing_path = out / TIMING_FILE
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    jetson, jetson_estimated = with_speed(points, timing, timed_fps(out, JETSON.column),
                                          JETSON.column)
    x86, x86_estimated = with_speed(points, timing, timed_fps(out, X86.column), X86.column)
    estimated = {JETSON.column: jetson_estimated, X86.column: x86_estimated}
    used = [c for c in clips if c.name not in {e.clip for e in excluded}]
    top_views = top_view_points(measured, used, list(POINTS))
    bases = base_points(measured, used, list(POINTS))
    lowest = lowest_points(measured, used, list(POINTS))
    commit = buildinfo.info_of(Path(args.binary).resolve())["commit"]
    others = "".join(
        f", {name.partition('@')[2] or name} at {buildinfo.info_of(path)['commit'][:8]}"
        for name, path in binary_overrides(args.binary_for).items())
    timed_x86 = "paired-timed x86 speed" if not x86_estimated - {SINGLE_LAYER} else \
        "speed partly estimated from instruction counts"
    meta = (f"Updated {datetime.date.today().isoformat()}, build {commit[:8]}{others}; libaom "
            f"svc_encoder_rtc v3.15.3, libvpx vp9_spatial_svc_encoder v1.16.0; {timed_x86} "
            "(see the notes)")
    (out / f"{OUTPUT_STEM}.md").write_text(markdown(jetson, x86, top_views, bases, lowest,
                                                    measured, timing, excluded, meta, estimated))
    (out / f"{OUTPUT_STEM}.csv").write_text(csv_rows(jetson, x86, bases, lowest, measured))
    if timing:
        (out / f"{OUTPUT_STEM}.svg").write_text(svg_chart(
            x86, pareto(x86), "720p30 + 360p30 (+ 180p30) streams: BD-J of the 720p output "
            f"at the total rate vs single-layer x264 medium; {timed_x86}",
            X86, heading="Spatial SVC (L2T1 / L2T3 / L3T1 / L3T3): bitrate vs encode speed",
            note="Speed: one core, paired with x264 medium in rotated rounds; points without "
                 "a paired timing are estimated from instruction counts (svc-chart.md)."))
    print(f"wrote {out}/{OUTPUT_STEM}.{{md,csv{',svg' if timing else ''}}}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nceval.svc_chart", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("measure", "time", "time-host", "plan", "merge-jetson", "render"):
        p = sub.add_parser(name)
        p.add_argument("--binary", required=True, help="our encoder (x86 release build)")
        p.add_argument("--presets-dir", default=str(PRESETS_DIR))
        p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        p.add_argument("--binary-for", action="append", default=[], metavar="POINT=PATH",
                       help="measure one of our points on another build")
        p.add_argument("--out", default=str(Path(__file__).resolve().parents[2]
                                            / "docs/results"))
        if name == "measure":
            p.add_argument("--points", help="comma-separated SVC points (default: all)")
            p.add_argument("--clips", help="comma-separated chart clips (default: all)")
            p.add_argument("--part", help="I/N: the I-th of N class-balanced clip parts")
            p.add_argument("--no-lines", action="store_true",
                           help="skip x264 medium and single-layer AV1 / VP9 (and the low "
                                "rungs)")
        if name == "merge-jetson":
            p.add_argument("files", nargs="+", help="t-POINT.json files from the Jetson")
        if name == "time-host":
            p.add_argument("--cpus", default=",".join(map(str, HOST_CPUS)),
                           help="reserved timing CPUs, one run on each at a time")
        if name in ("time", "time-host"):
            p.add_argument("--points", help="comma-separated points to count (default: all; "
                           "the others keep their stored counts)")
    args = parser.parse_args(argv)
    {"measure": measure, "time": time_points, "time-host": time_host, "plan": write_plan,
     "merge-jetson": merge_jetson, "render": render}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
