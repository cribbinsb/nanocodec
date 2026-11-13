"""A stand-in for nceval.tune.encode_point: no encoder, decoders or metrics.

Tests patch `tune.encode_point` with a FakeEncodePoint to drive the curve
logic (calibration, caps, retention) without running any tool.
"""
from __future__ import annotations

from pathlib import Path

from nceval.conformance import GateResult
from nceval.metrics import Measurement
from nceval.point import PointResult


class FakeEncodePoint:
    """Records every encoded point; writes `stream_bytes(q)` as the bitstream."""

    def __init__(self, stream_bytes=lambda q: str(q).encode(), psnr=lambda q: 40.0,
                 vmaf_v1=80.0, ms_ssim=0.95):
        self.stream_bytes, self.psnr = stream_bytes, psnr
        self.vmaf_v1, self.ms_ssim = vmaf_v1, ms_ssim
        self.calls: list[dict] = []

    def __call__(self, binary, codec, clip, quantizer, frames, overrides, stream: Path,
                 metrics=None, own_decoder=None, standard=True) -> PointResult:
        self.calls.append({"quantizer": quantizer, "metrics": metrics,
                           "own_decoder": own_decoder, "clip": clip, "standard": standard})
        stream.write_bytes(self.stream_bytes(quantizer))
        measurement = (Measurement({"psnr_yuv": self.psnr(quantizer), "vmaf_v1": self.vmaf_v1,
                                    "ms_ssim": self.ms_ssim}) if metrics else None)
        level = "full" if standard else "own-decoder"
        return PointResult(GateResult(True, level=level), f"recon-{quantizer}", measurement)

    @property
    def quantizers(self) -> list[int]:
        return [call["quantizer"] for call in self.calls]
