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


ROWS = (Row(Codec.H264, "Default", "nanocodec-h264", "openh264-baseline"),
        Row(Codec.H264, "Fast", "nanocodec-h264@h264-fast", "openh264-baseline"),
        Row(Codec.H264, "Faster", "nanocodec-h264@h264-faster", "openh264-baseline"),
        Row(Codec.H264, "Fastest", "nanocodec-h264@h264-fastest", "openh264-baseline"),
        Row(Codec.HEVC, "Default", "nanocodec-hevc", "svtav1-m12"),
        Row(Codec.HEVC, "Fast", "nanocodec-hevc@hevc-fast", "svtav1-m12"),
        Row(Codec.HEVC, "Faster", "nanocodec-hevc@hevc-faster", "svtav1-m12"),
        Row(Codec.AV1, "Default", "nanocodec-av1", "svtav1-m12"),
        Row(Codec.AV1, "Fast", "nanocodec-av1@av1-fast", "svtav1-m12"),
        Row(Codec.AV1, "Faster", "nanocodec-av1@av1-faster", "svtav1-m12"),
        Row(Codec.VP9, "Default", "nanocodec-vp9", "nanocodec-hevc"),
        Row(Codec.VP9, "Fast", "nanocodec-vp9@vp9-fast", "nanocodec-hevc@hevc-fast"),
        Row(Codec.VP9, "Faster", "nanocodec-vp9@vp9-faster", "nanocodec-hevc@hevc-faster"),
        Row(Codec.VP9, "Fastest", "nanocodec-vp9@vp9-fastest", "nanocodec-hevc@hevc-faster"),
        Row(Codec.AV2, "Default", "nanocodec-av2", "nanocodec-av1"),
        Row(Codec.AV2, "Fast", "nanocodec-av2@av2-fast", "nanocodec-av1@av1-fast"),
        Row(Codec.VVC, "Default", "nanocodec-vvc", "nanocodec-hevc"),
        Row(Codec.VVC, "Fast", "nanocodec-vvc@vvc-fast", "nanocodec-hevc@hevc-fast"))
