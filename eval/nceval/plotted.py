"""What the results views plot: the one list of chart families (leaf module).

Every chart, lens, held-out, fixed-QP and release tool takes its default
families from here; `--families` on any of them is only an explicit subset.

- Our families come from the shipped preset registry
  (crates/nc-encoder/presets, `encoders.nanocodec.shipped_presets`): every
  speed rung of every codec, Slower to Veryfast. Layered presets (`l2-*`,
  `l3-*`) are not points of these charts; the SVC chart (nceval.svc_chart)
  has its own. A preset file added to the registry is plotted without any
  other edit: its family, label, head-to-head pairs and scoreboard row
  follow from its codec and rung.
- The reference encoders are the explicit REFERENCES list below.

Hand-kept copies of either list are what lost new presets from some views
(AV1 Veryfast, AV2 Faster, 2026-11-18); eval/tests/test_plotted.py and
`nceval.site --check` fail when a shipped preset is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .codecs import Codec
from .encoders.nanocodec import LADDER, MEDIUM, PRESETS_DIR, SCALABILITY_STEM_MODES, shipped_presets

OWN_PREFIX = "nanocodec-"
#: The charts' codec order (legend, tables).
CODEC_ORDER = (Codec.H264, Codec.HEVC, Codec.AV1, Codec.VP9, Codec.AV2, Codec.VVC)
CODEC_LABELS = {Codec.H264: "H.264", Codec.HEVC: "HEVC", Codec.VVC: "VVC", Codec.AV1: "AV1",
                Codec.VP9: "VP9", Codec.AV2: "AV2"}
#: The quality rungs above the speed ladder (encoders.nanocodec.LADDER, fastest first),
#: slowest first. `slower` was named `slow` until 2026-11-21: a frozen build's `@slow`
#: family is the rung its own presets/ directory holds.
QUALITY_RUNGS = ("veryslow", "slower", "slow")
#: Every rung a chart can plot, slowest first (a codec ships a subset).
RUNG_ORDER = (*QUALITY_RUNGS, *reversed(LADDER))


@dataclass(frozen=True)
class Reference:
    """A reference encoder of the charts: its family (cbr.anchor_families), its
    label, and the rungs of ours it is the direct competitor of (`rivals`:
    codec -> rungs), which become head-to-head pairs."""
    family: str
    label: str
    rivals: tuple[tuple[Codec, tuple[str, ...]], ...] = ()


#: The reference encoders every view plots (the explicit half of "what we plot").
REFERENCES = (
    Reference("openh264-baseline", "openh264", ((Codec.AV1, (MEDIUM, "fast")),)),
    Reference("x264-ultrafast", "x264 ultrafast", ((Codec.H264, ("faster", "veryfast")),)),
    Reference("x264-veryfast", "x264 veryfast", ((Codec.H264, (MEDIUM, "fast")),)),
    Reference("x264-medium", "x264 medium", ((Codec.H264, (MEDIUM,)),)),
    Reference("x265-ultrafast", "x265 ultrafast"),
    Reference("x265-superfast", "x265 superfast", ((Codec.HEVC, ("fast",)),)),
    Reference("x265-medium", "x265 medium", ((Codec.HEVC, (MEDIUM,)), (Codec.VVC, (MEDIUM,)))),
    Reference("svtav1-m8", "SVT-AV1 M8", ((Codec.AV1, (MEDIUM,)), (Codec.AV2, (MEDIUM,)))),
    Reference("svtav1-m10", "SVT-AV1 M10"),
    Reference("svtav1-m12", "SVT-AV1 M12", ((Codec.AV2, ("faster",)),)),
    Reference("svtav1-m13", "SVT-AV1 M13", ((Codec.AV1, ("veryfast",)), (Codec.HEVC, ("faster",)))),
    Reference("libvpx-vp9-cpu7", "libvpx VP9 rt 7", ((Codec.VP9, (MEDIUM,)),)),
    Reference("libvpx-vp9-cpu9", "libvpx VP9 rt 9", ((Codec.VP9, ("fast", "faster", "veryfast")),)),
    Reference("libaom-av1-cpu8", "libaom rt 8", ((Codec.AV1, (MEDIUM,)),)),
    Reference("libaom-av1-cpu10", "libaom rt 10"),
)
#: Each codec's bar (the scoreboards' row anchor and a head-to-head pair of every
#: rung): a reference family, or another codec of ours at the same rung (the
#: nearest slower one it ships) for the formats measured against their predecessor.
BARS: dict[Codec, str | Codec] = {Codec.H264: "openh264-baseline", Codec.HEVC: "svtav1-m12",
                                  Codec.AV1: "svtav1-m12", Codec.VP9: Codec.HEVC,
                                  Codec.AV2: Codec.AV1, Codec.VVC: Codec.HEVC}


@dataclass(frozen=True)
class OwnFamily:
    codec: Codec
    rung: str

    @property
    def name(self) -> str:
        return f"{OWN_PREFIX}{self.codec.value}" + ("" if self.rung == MEDIUM else f"@{self.rung}")

    @property
    def label(self) -> str:
        """The chart label: 'AV1' for the Default (medium), 'AV1 Veryfast' otherwise."""
        codec = CODEC_LABELS[self.codec]
        return codec if self.rung == MEDIUM else f"{codec} {self.rung.capitalize()}"


def is_layered(stem: str) -> bool:
    """A scalability preset (`l2-fast`): not a point of the single-layer charts."""
    layers, _, speed = stem.partition("-")
    return bool(speed) and layers in SCALABILITY_STEM_MODES


def own_rungs(codec: Codec, presets: Path = PRESETS_DIR) -> list[str]:
    """`codec`'s shipped single-layer rungs, slowest first."""
    stems = [s for s in shipped_presets(codec, presets) if not is_layered(s)]
    unknown = [s for s in stems if s not in RUNG_ORDER]
    if unknown:
        raise ValueError(f"{codec.value}: preset(s) {unknown} have no place in RUNG_ORDER")
    return sorted(stems, key=RUNG_ORDER.index)


def own_families(presets: Path = PRESETS_DIR) -> list[OwnFamily]:
    """Every shipped single-layer preset of every codec, in chart order."""
    return [OwnFamily(c, r) for c in CODEC_ORDER for r in own_rungs(c, presets)]


def own_names(presets: Path = PRESETS_DIR) -> list[str]:
    return [f.name for f in own_families(presets)]


def nearest_rung(rungs: list[str], rung: str) -> str | None:
    """`rung` if shipped, else the nearest slower shipped one (None: none slower)."""
    slower = [r for r in rungs if RUNG_ORDER.index(r) <= RUNG_ORDER.index(rung)]
    return slower[-1] if slower else None


def bar(family: OwnFamily, presets: Path = PRESETS_DIR) -> str | None:
    """The family's bar (BARS) as a family name."""
    target = BARS[family.codec]
    if not isinstance(target, Codec):  # Codec is a str enum: test it first
        return target
    rung = nearest_rung(own_rungs(target, presets), family.rung)
    return OwnFamily(target, rung).name if rung else None


def next_slower(family: OwnFamily, presets: Path = PRESETS_DIR) -> str | None:
    """The next slower speed-ladder rung of the same codec (fast -> medium); None for
    medium, slow and slower (the quality rungs are compared with the references only)."""
    if family.rung not in LADDER or family.rung == MEDIUM:
        return None
    rungs = [r for r in own_rungs(family.codec, presets) if r in LADDER]
    slower = rungs[:rungs.index(family.rung)]
    return OwnFamily(family.codec, slower[-1]).name if slower else None


def pairs(presets: Path = PRESETS_DIR) -> list[tuple[str, str]]:
    """The chart's head-to-head pairs (ours, other): per family its bar, the
    references that list it as a rival, and the next slower rung of its codec."""
    out: list[tuple[str, str]] = []
    for family in own_families(presets):
        rivals = [r.family for r in REFERENCES
                  if family.rung in dict(r.rivals).get(family.codec, ())]
        for other in (bar(family, presets), *rivals, next_slower(family, presets)):
            if other and (family.name, other) not in out:
                out.append((family.name, other))
    return out


def labels(presets: Path = PRESETS_DIR) -> dict[str, str]:
    """Every plotted family's chart label."""
    return {**{f.name: f.label for f in own_families(presets)},
            **{r.family: r.label for r in REFERENCES}}


def plotted_names(presets: Path = PRESETS_DIR) -> list[str]:
    """Every family the views plot: ours in chart order, then the references."""
    return [*own_names(presets), *(r.family for r in REFERENCES)]
