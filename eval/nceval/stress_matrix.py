"""The feature matrix of the syntax-coverage stress test (tools/ci/syntax_stress.py).

Random decisions (`debug.random_decisions`) cover block-level syntax; the
real-use features change frame-level syntax and reference handling, so every
combination has to be decoded by the standard decoders too:

  structure   flat, 2 layers, 3 layers adaptive, 3 layers fixed,
              long-term reference with a reported loss (the lost frame is left
              out of the stream, the encoder recovers from the LTR), flat and
              with 2 or 3 fixed layers (a lost top-layer frame needs no
              recovery),
              intra refresh at periods 7 and 4 (no periodic key frames),
              resolution switches down to half size at RESIZE_DOWN and back
              at RESIZE_UP (--resize: key frames with new headers)
  rate        fixed QP, CBR at a low or a high bitrate, or CBR stepping from
              the high to the low bitrate mid-clip (--rate-change)
  loop filter AV1 only: av1.loop_filter 0 or 1
  filter intra AV1 only: av1.keyframe_filter_intra 0 or 1 (the sequence
              header tool; random decisions then code filter intra on DC
              blocks up to 32x32, in key and inter frames)
  entropy     H.264 only: CABAC or CAVLC slice data (h264.cavlc 0 or 1)

VP9 runs every structure but intra refresh (every rate mode; fixed QP from
0, lossless, to 255).

The structures are mutually exclusive in the encoder (intra refresh is
flat-only and excludes LTR). Seed n takes combination
`(n * STRIDE) mod count`, a stride coprime with the count, so any `count`
consecutive seeds cover every combination exactly once and a short run
(--quick) still spreads over the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd

from .codecs import Codec
from .yuv import SwitchedFormat, YuvFormat

FRAMES = 12
INTRA_PERIOD = 8  # a second key frame inside the clip (flat and layered structures)
LTR_INTERVAL = 3
# Frames whose loss the LTR case may report: not the first frame, a key frame
# or the last frame (whose loss would leave nothing to recover).
LOSS_CANDIDATES = (2, 3, 4, 5, 6, 7, 9, 10)
STRUCTURES = ("flat", "layers2", "layers3", "layers3-fixed", "ltr-loss", "layers2-fixed-ltr-loss",
              "layers3-fixed-ltr-loss", "refresh7", "refresh4", "resize")
# Resolution switches of the "resize" structure: to half size (even
# dimensions) and back, away from the key frames at 0 and INTRA_PERIOD.
RESIZE_DOWN, RESIZE_UP = 5, 9
# CBR bitrates in bits per pixel per frame: starved (rate control at its QP
# ceiling) and generous (near its floor); a step goes from the generous to the
# starved one at RATE_STEP_FRAME.
CBR_LOW, CBR_HIGH = 0.04, 0.4
RATES = ("cqp", "cbr-low", "cbr-high", "cbr-step")
RATE_STEP_FRAME = FRAMES // 2
LOOP_FILTER = {Codec.AV1: (0, 1)}
FILTER_INTRA = {Codec.AV1: (0, 1)}
CAVLC = {Codec.H264: (0, 1)}
QUANTIZERS = {Codec.H264: (18, 46), Codec.HEVC: (18, 46), Codec.VVC: (18, 58), Codec.AV1: (40, 230), Codec.VP9: (0, 255)}
# Two-layer spatial scalability modes (shared by AV1 and VP9) and three-layer
# AV1 SVC modes (L3T1/L3T2/L3T3, independent layers, and custom non-2:1 scales).
TWO_LAYER_SPATIAL_STRUCTURES = (
    "spatial-l2t1", "spatial-l2t1-indep", "spatial-l2t2", "spatial-l2t3", "spatial-sparse")
AV1_SPATIAL_STRUCTURES = TWO_LAYER_SPATIAL_STRUCTURES + (
    "spatial-l3t1", "spatial-l3t1-indep", "spatial-l3t2", "spatial-l3t3", "spatial-l3-custom")
SPATIAL_SCALES = (0.8, 0.6666667, 0.5)
THREE_LAYER_SCALE_PAIRS = ((0.25, 0.5), (0.3333333, 0.6666667), (0.4, 0.75))
# Decimated enhancement intervals 2..8 (SPATIAL_MIN_INTERVAL + 0..SPATIAL_INTERVALS-1).
SPATIAL_MIN_INTERVAL, SPATIAL_INTERVALS = 2, 7
# VP9 codes no intra refresh yet (layers and LTR recovery use
# error-resilient frames) and keeps two-layer SVC only.
CODEC_STRUCTURES = {
    Codec.VP9: tuple(s for s in STRUCTURES if not s.startswith("refresh")) + TWO_LAYER_SPATIAL_STRUCTURES,
    Codec.AV1: STRUCTURES + AV1_SPATIAL_STRUCTURES,
}
STRIDE = 37


@dataclass(frozen=True)
class Features:
    structure: str
    rate: str
    loop_filter: int
    filter_intra: int = 0
    cavlc: int = 0

    def label(self) -> str:
        return (f"{self.structure}-{self.rate}" + ("-lf" if self.loop_filter else "")
                + ("-fi" if self.filter_intra else "") + ("-cavlc" if self.cavlc else ""))


def combinations(codec: Codec) -> list[Features]:
    return [Features(s, r, lf, fi, vlc) for s, r, lf, fi, vlc in
            product(CODEC_STRUCTURES.get(codec, STRUCTURES), RATES, LOOP_FILTER.get(codec, (0,)),
                    FILTER_INTRA.get(codec, (0,)), CAVLC.get(codec, (0,)))]


def features_for(codec: Codec, seed: int, spatial_only: bool = False) -> Features:
    if spatial_only:
        return spatial_features_for(codec, seed)
    combos = combinations(codec)
    stride = STRIDE if gcd(STRIDE, len(combos)) == 1 else 1
    return combos[(seed * stride) % len(combos)]


def spatial_features_for(codec: Codec, seed: int) -> Features:
    """Spatial-SVC draw (`syntax_stress.py --spatial`): on VP9 even seeds code
    `spatial-sparse` and odd seeds cycle the four 2-layer fixed modes; on AV1
    every third seed codes `spatial-sparse` while the remaining two-thirds
    cycle all 2-layer and 3-layer modes (`L2T*`, `L3T*`, custom 3-layer scales)."""
    if codec not in (Codec.AV1, Codec.VP9):
        raise ValueError("spatial stress draws are AV1 and VP9 only")
    if codec == Codec.VP9:
        fixed = tuple(s for s in TWO_LAYER_SPATIAL_STRUCTURES if s != "spatial-sparse")
        structure = "spatial-sparse" if seed % 2 == 0 else fixed[(seed // 2) % len(fixed)]
    else:
        fixed = tuple(s for s in AV1_SPATIAL_STRUCTURES if s != "spatial-sparse")
        if seed % 3 == 0:
            structure = "spatial-sparse"
        else:
            idx = seed - (seed // 3) - 1
            structure = fixed[idx % len(fixed)]
    rest = list(product(RATES, LOOP_FILTER.get(codec, (0,)), FILTER_INTRA.get(codec, (0,))))
    rate, loop_filter, filter_intra = rest[(seed * STRIDE) % len(rest)]
    return Features(structure, rate, loop_filter, filter_intra)


@dataclass(frozen=True)
class StressCase:
    codec: Codec
    width: int
    height: int
    fps: float
    seed: int
    spatial_only: bool = False

    @property
    def features(self) -> Features:
        return features_for(self.codec, self.seed, self.spatial_only)

    @property
    def quantizer(self) -> int:
        lo, hi = QUANTIZERS[self.codec]
        return lo + (self.seed * 7919) % (hi - lo + 1)

    @property
    def lost_frame(self) -> int | None:
        if not self.features.structure.endswith("ltr-loss"):
            return None
        return LOSS_CANDIDATES[self.seed % len(LOSS_CANDIDATES)]

    @property
    def coded_frames(self) -> int:
        """Frames in the stream and the reconstruction (a lost one is left out)."""
        return FRAMES - (self.lost_frame is not None)

    @property
    def resized(self) -> tuple[int, int]:
        """The half size the "resize" structure switches to."""
        return (self.width // 4 * 2, self.height // 4 * 2)

    @property
    def frame_format(self) -> YuvFormat | SwitchedFormat:
        """The size of every frame of the source and the reconstruction."""
        full = YuvFormat(self.width, self.height)
        if self.features.structure != "resize":
            return full
        return SwitchedFormat(full, ((RESIZE_DOWN, YuvFormat(*self.resized)), (RESIZE_UP, full)))

    @property
    def is_spatial(self) -> bool:
        return self.features.structure.startswith("spatial-")

    @property
    def spatial_sparse_spec(self) -> tuple[int, float, int, bool]:
        """(temporal_layers, base_scale, enhancement_interval, inter_layer) for `spatial-sparse`.

        Each dimension reads different digits of the seed, so consecutive
        sparse seeds (the even ones under `--spatial`, `k = seed // 2`) cover
        every scale x interval pair within 21 of them: scale `k % 3`,
        interval `2 + (k // 3) % 7`."""
        k = self.seed // 2
        t_layers = 2 if (self.seed // 4) % 2 == 1 else 1
        scale = SPATIAL_SCALES[k % len(SPATIAL_SCALES)]
        interval = SPATIAL_MIN_INTERVAL + (k // len(SPATIAL_SCALES)) % SPATIAL_INTERVALS
        inter_layer = (self.seed % 5) != 0
        return t_layers, scale, interval, inter_layer

    @property
    def spatial_l3_custom_spec(self) -> tuple[int, float, float, bool]:
        """(temporal_layers, s0_scale, s1_scale, inter_layer) for `spatial-l3-custom`."""
        t_layers = 1 + (self.seed % 3)
        s0_scale, s1_scale = THREE_LAYER_SCALE_PAIRS[(self.seed // 3) % len(THREE_LAYER_SCALE_PAIRS)]
        inter_layer = (self.seed % 5) != 0
        return t_layers, s0_scale, s1_scale, inter_layer

    @property
    def name(self) -> str:
        extra = ""
        if self.features.structure == "spatial-sparse":
            t_layers, scale, interval, inter_layer = self.spatial_sparse_spec
            extra = f"-t{t_layers}-s{scale:.2f}-i{interval}-il{int(inter_layer)}"
        elif self.features.structure == "spatial-l3-custom":
            t_layers, s0, s1, inter_layer = self.spatial_l3_custom_spec
            extra = f"-t{t_layers}-s{s0:.2f}_{s1:.2f}-il{int(inter_layer)}"
        return (f"{self.codec.value}-{self.width}x{self.height}-seed{self.seed}-"
                f"q{self.quantizer}-{self.features.label()}{extra}")

    def encoder_flags(self) -> list[str]:
        """Command-line flags for the case's features (after the common ones)."""
        f = self.features
        flags = ["--param", f"debug.random_decisions={self.seed}"]
        if f.structure.startswith("layers"):
            flags += ["--temporal-layers", f.structure[len("layers")]]
            flags += ["--param", f"layers.adaptive={0 if '-fixed' in f.structure else 1}"]
        if f.structure == "spatial-l2t1":
            flags += ["--scalability", "L2T1"]
        elif f.structure == "spatial-l2t1-indep":
            flags += ["--scalability", "L2T1", "--param", "spatial.inter_layer=0"]
        elif f.structure == "spatial-l2t2":
            flags += ["--scalability", "L2T2", "--param", "layers.adaptive=0"]
        elif f.structure == "spatial-l2t3":
            flags += ["--scalability", "L2T3", "--param", "layers.adaptive=0"]
        elif f.structure == "spatial-l3t1":
            flags += ["--scalability", "L3T1"]
        elif f.structure == "spatial-l3t1-indep":
            flags += ["--scalability", "L3T1", "--param", "spatial.inter_layer=0"]
        elif f.structure == "spatial-l3t2":
            flags += ["--scalability", "L3T2", "--param", "layers.adaptive=0"]
        elif f.structure == "spatial-l3t3":
            flags += ["--scalability", "L3T3", "--param", "layers.adaptive=0"]
        elif f.structure == "spatial-l3-custom":
            t_layers, s0_scale, s1_scale, inter_layer = self.spatial_l3_custom_spec
            flags += ["--scalability", f"L3T{t_layers}",
                      "--param", "layers.adaptive=0",
                      "--param", f"spatial.s0_scale={s0_scale}",
                      "--param", f"spatial.s1_scale={s1_scale}",
                      "--param", f"spatial.inter_layer={int(inter_layer)}"]
        elif f.structure == "spatial-sparse":
            t_layers, scale, interval, inter_layer = self.spatial_sparse_spec
            flags += ["--scalability", "L2T2" if t_layers == 2 else "L2T1",
                      "--param", "layers.adaptive=0",
                      "--param", f"spatial.base_scale={scale}",
                      "--param", f"spatial.enhancement_interval={interval}",
                      "--param", f"spatial.inter_layer={int(inter_layer)}"]
        if f.structure.endswith("ltr-loss"):
            flags += ["--ltr-interval", str(LTR_INTERVAL), "--simulate-loss", str(self.lost_frame)]
        if f.structure.startswith("refresh"):
            flags += ["--intra-refresh", f.structure[len("refresh"):]]
        if f.structure == "resize":
            small = "x".join(map(str, self.resized))
            flags += ["--resize", f"{RESIZE_DOWN}:{small}",
                      "--resize", f"{RESIZE_UP}:{self.width}x{self.height}"]
        def kbps(bits_per_pixel: float) -> str:
            return f"{max(1, round(bits_per_pixel * self.width * self.height * self.fps / 1000))}"

        if f.rate == "cqp":
            flags += ["--qp", str(self.quantizer)]
        elif f.rate == "cbr-step":
            flags += ["--bitrate", kbps(CBR_HIGH), "--rate-change", f"{RATE_STEP_FRAME}:{kbps(CBR_LOW)}"]
        else:
            flags += ["--bitrate", kbps(CBR_LOW if f.rate == "cbr-low" else CBR_HIGH)]
        if self.codec == Codec.AV1:
            # Explicit both ways: the AV1 Default filters (av1.loop_filter = 1).
            flags += ["--param", f"av1.loop_filter={int(f.loop_filter)}"]
            flags += ["--param", f"av1.keyframe_filter_intra={int(f.filter_intra)}"]
        if self.codec == Codec.H264:
            flags += ["--param", f"h264.cavlc={int(f.cavlc)}"]
        return flags
