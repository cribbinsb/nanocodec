"""The scoreboard rows (codec, preset, our encoder, its bar): leaf module.

Shared by nceval.scoreboard (fixed-QP board) and nceval.realuse (J_real
board); kept apart so neither board imports the other (tests/test_imports.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .codecs import Codec


@dataclass(frozen=True)
class Row:
    codec: Codec
    preset: str
    encoder: str  # sweep adapter name
    anchor: str


ROWS = (Row(Codec.H264, "Medium", "nanocodec-h264", "openh264-baseline"),
        Row(Codec.H264, "Fast", "nanocodec-h264@fast", "openh264-baseline"),
        Row(Codec.H264, "Faster", "nanocodec-h264@faster", "openh264-baseline"),
        Row(Codec.H264, "Veryfast", "nanocodec-h264@veryfast", "openh264-baseline"),
        Row(Codec.HEVC, "Medium", "nanocodec-hevc", "svtav1-m12"),
        Row(Codec.HEVC, "Fast", "nanocodec-hevc@fast", "svtav1-m12"),
        Row(Codec.HEVC, "Faster", "nanocodec-hevc@faster", "svtav1-m12"),
        Row(Codec.AV1, "Medium", "nanocodec-av1", "svtav1-m12"),
        Row(Codec.AV1, "Fast", "nanocodec-av1@fast", "svtav1-m12"),
        Row(Codec.AV1, "Faster", "nanocodec-av1@faster", "svtav1-m12"),
        Row(Codec.VP9, "Medium", "nanocodec-vp9", "nanocodec-hevc"),
        Row(Codec.VP9, "Fast", "nanocodec-vp9@fast", "nanocodec-hevc@fast"),
        Row(Codec.VP9, "Faster", "nanocodec-vp9@faster", "nanocodec-hevc@faster"),
        Row(Codec.VP9, "Veryfast", "nanocodec-vp9@veryfast", "nanocodec-hevc@faster"),
        Row(Codec.AV2, "Medium", "nanocodec-av2", "nanocodec-av1"),
        Row(Codec.AV2, "Fast", "nanocodec-av2@fast", "nanocodec-av1@fast"),
        Row(Codec.VVC, "Medium", "nanocodec-vvc", "nanocodec-hevc"),
        Row(Codec.VVC, "Fast", "nanocodec-vvc@fast", "nanocodec-hevc@fast"))
