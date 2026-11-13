"""Codec identities and the standard decoders used to check them."""

from __future__ import annotations

from enum import Enum


class Codec(str, Enum):
    H264 = "h264"
    HEVC = "hevc"
    VVC = "vvc"
    AV1 = "av1"
    VP9 = "vp9"
    AV2 = "av2"

    @property
    def bitstream_suffix(self) -> str:
        # Annex-B elementary streams for H.264/HEVC; IVF-wrapped OBUs for AV1,
        # which every AV1 tool (aomdec, dav1d, ffmpeg) reads.
        return {Codec.H264: ".264", Codec.HEVC: ".hevc", Codec.VVC: ".vvc", Codec.AV1: ".ivf", Codec.VP9: ".ivf", Codec.AV2: ".ivf"}[self]
