"""openh264 adapters: the baselines every target is measured against.

Two configurations (eval/README.md "Methodology: Anchors"):

  openh264-baseline  Constrained Baseline, CAVLC, MEDIUM_COMPLEXITY,
                     minimal extras. The bar for the headline targets.
  openh264-high      CABAC (labelled High), HIGH_COMPLEXITY, AQ /
                     scene-change / background detection on. openh264 has no
                     8x8 transform, so this is the best openh264 can do.

Both: CAMERA_VIDEO_REAL_TIME, 1 thread, frame skip off, denoise and LTR off,
fixed QP, and dyadic temporal layers matched to the comparison.

The encoder is driven by tools/openh264-harness (a small C program over the
openh264 API), because the stock h264enc CLI cannot set every parameter we
need. Its flags mirror the SEncParamExt fields one to one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..codecs import Codec
from ..config import Paths, find_tool
from .base import EncodeJob, EncoderAdapter


@dataclass(frozen=True)
class Openh264Settings:
    cabac: bool
    complexity: str  # "low" | "medium" | "high"
    adaptive_quant: bool
    scene_change_detection: bool
    background_detection: bool


BASELINE = Openh264Settings(cabac=False, complexity="medium", adaptive_quant=False,
                            scene_change_detection=False, background_detection=True)
HIGH = Openh264Settings(cabac=True, complexity="high", adaptive_quant=True,
                        scene_change_detection=True, background_detection=True)


class Openh264(EncoderAdapter):
    codec = Codec.H264
    quantizer_range = (10, 51)  # below 10 is far outside our bitrate range

    def __init__(self, name: str, settings: Openh264Settings):
        self.name = name
        self.settings = settings

    def supports_temporal_layers(self, layers: int) -> bool:
        return layers in (1, 2, 3)

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        s, clip = self.settings, job.clip
        on_off = {True: "1", False: "0"}
        # Fixed-QP sweeps need a fixed intra period, and scene-change detection
        # would insert extra IDRs, so it is forced off here even for "high".
        # (It applies only to rate-control runs.)
        scene_change = on_off[False]
        return [str(find_tool("openh264-harness", paths)),
                "--input", str(clip.yuv), "--output", str(job.bitstream),
                "--width", str(clip.width), "--height", str(clip.height),
                "--fps", f"{clip.fps:g}", "--frames", str(job.frames),
                "--usage", "camera-realtime", "--complexity", s.complexity,
                "--cabac", on_off[s.cabac], "--qp", str(job.quantizer),
                "--intra-period", str(job.structure.intra_period_frames),
                "--temporal-layers", str(job.structure.temporal_layers),
                "--adaptive-quant", on_off[s.adaptive_quant],
                "--scene-change", scene_change,
                "--background-detection", on_off[s.background_detection],
                "--denoise", "0", "--ltr", "0", "--frame-skip", "0", "--threads", "1"]


def openh264_anchors() -> list[Openh264]:
    return [Openh264("openh264-baseline", BASELINE), Openh264("openh264-high", HIGH)]
