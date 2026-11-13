"""Encoder adapter interface.

An adapter turns an EncodeJob (clip, quantiser, coding structure) into one
command line for a specific encoder configuration. Every adapter must produce
a *low-delay* configuration comparable with ours (eval/README.md
"Methodology: Anchors"):

  - P frames only: no B-frames, no reordering, no lookahead;
  - a single thread;
  - fixed intra period, no scene-cut keyframes;
  - fixed quantiser (rate control off) for BD-rate sweeps.

Quantisers are encoder-native (H.264/HEVC QP 0-51, AV1 0-63 / 0-255); the
runner picks per-clip values that land in the target bitrate range, so the
scales never need to be compared directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..clips import Clip
from ..codecs import Codec
from ..config import Paths


@dataclass(frozen=True)
class CodingStructure:
    """The low-delay GOP shape shared by every encoder in a comparison."""
    intra_period_frames: int
    temporal_layers: int = 1  # 1 = flat P; 2/3 = dyadic hierarchical P


@dataclass(frozen=True)
class EncodeJob:
    clip: Clip
    quantizer: int
    structure: CodingStructure
    frames: int
    bitstream: Path
    recon: Path | None = None  # ask for the encoder's reconstruction, if supported
    #: Ask for the times of the coded frames (which input frames were coded), if supported.
    coded_times: Path | None = None


class EncoderAdapter(ABC):
    #: Unique, stable name used in results (e.g. "x265-ultrafast").
    name: str
    codec: Codec
    #: Inclusive native quantiser range the QP search may use.
    quantizer_range: tuple[int, int]

    @abstractmethod
    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        """The full command line for this job."""

    def supports_temporal_layers(self, layers: int) -> bool:
        return layers == 1

    def supports_recon(self) -> bool:
        return False

    def own_decoder(self, paths: Paths) -> Path | None:
        """Our native decoder for this encoder's streams (sample gate), if any."""
        return None

    def verify_recon(self) -> bool:
        """Whether the conformance gate also checks the encoder's own
        reconstruction against the decoders (always for nanocodec)."""
        return False

    def check_job(self, job: EncodeJob) -> None:
        if not self.supports_temporal_layers(job.structure.temporal_layers):
            raise ValueError(f"{self.name} cannot code {job.structure.temporal_layers} "
                             f"temporal layers")
        lo, hi = self.quantizer_range
        if not lo <= job.quantizer <= hi:
            raise ValueError(f"{self.name}: quantizer {job.quantizer} outside {lo}..{hi}")


def fps_fraction(fps: float) -> tuple[int, int]:
    """Express a frame rate as num/den, keeping 29.97-style rates exact."""
    for den in (1, 1001):
        num = fps * den
        if abs(num - round(num)) < 1e-3:
            return round(num), den
    return round(fps * 1000), 1000
