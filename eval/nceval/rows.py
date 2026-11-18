"""The scoreboard rows (codec, preset, our encoder, its bar): leaf module.

Shared by nceval.scoreboard (fixed-QP board) and nceval.realuse (J_real
board); kept apart so neither board imports the other (tests/test_imports.py).
The rows follow from the shipped presets (nceval.plotted), never a list here.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import plotted
from .codecs import Codec
from .encoders.nanocodec import LADDER


@dataclass(frozen=True)
class Row:
    codec: Codec
    preset: str
    encoder: str  # sweep adapter name
    anchor: str


#: One row per shipped speed-ladder preset (nceval.plotted: the preset registry),
#: barred by its codec's bar (plotted.BARS). Slow is a quality rung, not a
#: real-time one: the boards leave it out.
ROWS = tuple(Row(f.codec, f.rung.capitalize(), f.name, plotted.bar(f))
             for f in plotted.own_families() if f.rung in LADDER)
