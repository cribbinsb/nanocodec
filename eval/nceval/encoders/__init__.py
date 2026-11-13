"""Encoder adapters and the registry of encoder configurations.

ANCHORS are the third-party reference encoders; ENCODERS adds our own
(nanocodec-*). Names are stable: they are the keys stored in the results
database, so renaming one orphans its history.
"""

from __future__ import annotations

from .anchors import AvmLowDelay, X264, X265, Kvazaar, LibaomRealtime, LibvpxRealtime, SvtAv1
from .base import CodingStructure, EncodeJob, EncoderAdapter
from .nanocodec import nanocodec_encoders, nanocodec_with_preset
from .openh264 import openh264_anchors


def _build_registry() -> dict[str, EncoderAdapter]:
    adapters: list[EncoderAdapter] = [
        *openh264_anchors(),
        X264("ultrafast"), X264("superfast"), X264("veryfast"),
        X265("ultrafast"), X265("superfast"),
        Kvazaar("ultrafast"), Kvazaar("superfast"),
        *(SvtAv1(preset) for preset in (10, 11, 12, 13)),
        *(LibaomRealtime(speed) for speed in (8, 10, 11)),
        *(LibvpxRealtime(speed) for speed in (6, 8, 9)),
        AvmLowDelay(9),
        # Adversarial-review variants (encoders/anchors.py *_VARIANTS).
        X264("veryfast", "nopsy"), X264("veryfast", "crf"),
        X265("superfast", "nopsy"), X265("superfast", "crf"),
        SvtAv1(12, "qm"),
    ]
    return {adapter.name: adapter for adapter in adapters}


ANCHORS: dict[str, EncoderAdapter] = _build_registry()
ENCODERS: dict[str, EncoderAdapter] = {**ANCHORS, **{e.name: e for e in nanocodec_encoders()}}



def resolve_encoder(name: str) -> EncoderAdapter | None:
    """A registered encoder, or `nanocodec-<codec>@<preset>` (presets/<preset>.toml)."""
    return ENCODERS.get(name) or nanocodec_with_preset(name)


__all__ = ["ANCHORS", "ENCODERS", "resolve_encoder", "CodingStructure", "EncodeJob", "EncoderAdapter"]
