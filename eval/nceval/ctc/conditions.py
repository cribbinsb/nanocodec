"""The two test conditions of the standards lens: sequences, frames, quantiser points, encoders.

`AOM_LD` follows the AOM AV2 CTC v8 low-delay configuration (CWG-F384o §4.3:
130 frames, one key frame, six fixed quantiser points). `JVET_E` follows the
JVET low-delay P condition on class E (JCTVC-L1100 / JVET-T2010: the whole
600-frame sequence, QP 22/27/32/37), with QP 42/47 as a labelled low-rate
extension; only its first 150 frames are coded (`JVET_E_FRAMES`). See docs/analysis/standards-evaluation-methodology-2026-11-14.md.

Each encoder runs the quantiser points of its own scale:

- QP (H.264, HEVC, VVC): JVET's 22, 27, 32, 37, then 42, 47.
- QINDEX (our AV1/VP9/AV2, which take `base_q_idx` directly): the AOM CTC's
  110, 135, 160, 185, 210, 235.
- Q63 (libaom, SVT-AV1, libvpx: 0-63 user scales): the same qindex points
  through the libaom/libvpx `quantizer_to_qindex` table (qindex = 4q below 63),
  rounded to the nearest: 28, 34, 40, 46, 52, 59 (qindex 112 ... 236).

BD-rate compares curves only over their overlapping quality range
(`nceval.bdrate`), so the scales never need to match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..codecs import Codec
from ..encoders.anchors import X264, X265, Kvazaar, LibaomRealtime, LibvpxRealtime, SvtAv1
from ..encoders.base import EncoderAdapter
from ..encoders.nanocodec import nanocodec_encoders, nanocodec_with_preset
from ..encoders.openh264 import BASELINE, Openh264
from . import sequences
from .reference_encoders import HmLowDelayP, LibaomGoodLowDelay, VtmLowDelayP, VvencLowDelay


class Scale(str, Enum):
    QP = "qp"
    QINDEX = "qindex"
    Q63 = "q63"


POINTS = {
    Scale.QP: (22, 27, 32, 37, 42, 47),
    Scale.QINDEX: (110, 135, 160, 185, 210, 235),
    Scale.Q63: (28, 34, 40, 46, 52, 59),
}
#: JVET's own points: the first four QPs. The rest are the low-rate extension.
JVET_CTC_QPS = POINTS[Scale.QP][:4]


class Group(str, Enum):
    OURS = "ours"
    STANDARDS = "standards"   # reference software at CTC settings (exhaustive by design)
    PRACTICAL = "practical"   # real-time encoders, low delay, fixed QP, one thread


@dataclass(frozen=True)
class CtcEncoder:
    adapter: EncoderAdapter
    group: Group
    scale: Scale

    @property
    def name(self) -> str:
        return self.adapter.name

    @property
    def codec(self) -> Codec:
        return self.adapter.codec

    @property
    def own(self) -> bool:
        return self.group == Group.OURS


_QINDEX_CODECS = (Codec.AV1, Codec.VP9, Codec.AV2)
#: Our presets per codec: Default (the engine defaults) and Fast.
_FAST_PRESET = {Codec.H264: "h264-fast", Codec.HEVC: "hevc-fast", Codec.VVC: "vvc-fast",
                Codec.AV1: "av1-fast", Codec.VP9: "vp9-fast", Codec.AV2: "av2-fast"}


def _ours() -> list[CtcEncoder]:
    encoders = []
    for default in nanocodec_encoders():
        scale = Scale.QINDEX if default.codec in _QINDEX_CODECS else Scale.QP
        fast = nanocodec_with_preset(f"{default.name}@{_FAST_PRESET[default.codec]}")
        if fast is None:
            raise RuntimeError(f"preset {_FAST_PRESET[default.codec]} missing")
        encoders += [CtcEncoder(default, Group.OURS, scale), CtcEncoder(fast, Group.OURS, scale)]
    return encoders


STANDARDS = (CtcEncoder(HmLowDelayP(), Group.STANDARDS, Scale.QP),
             CtcEncoder(VtmLowDelayP(), Group.STANDARDS, Scale.QP),
             CtcEncoder(LibaomGoodLowDelay(), Group.STANDARDS, Scale.Q63))
PRACTICAL = (CtcEncoder(X264("medium"), Group.PRACTICAL, Scale.QP),
             CtcEncoder(X265("medium"), Group.PRACTICAL, Scale.QP),
             CtcEncoder(Openh264("openh264-baseline", BASELINE), Group.PRACTICAL, Scale.QP),
             CtcEncoder(Kvazaar("medium"), Group.PRACTICAL, Scale.QP),
             CtcEncoder(VvencLowDelay(), Group.PRACTICAL, Scale.QP),
             CtcEncoder(SvtAv1(10), Group.PRACTICAL, Scale.Q63),
             CtcEncoder(LibaomRealtime(8), Group.PRACTICAL, Scale.Q63),
             CtcEncoder(LibvpxRealtime(7), Group.PRACTICAL, Scale.Q63))
OURS = tuple(_ours())
ALL_ENCODERS = OURS + STANDARDS + PRACTICAL


@dataclass(frozen=True)
class Condition:
    name: str
    title: str
    sequences: tuple[sequences.Sequence, ...]
    frames: int
    encoders: tuple[CtcEncoder, ...]
    #: Anchors of the report's BD tables, the first is the condition's own.
    anchors: tuple[str, ...]
    #: Also run our encoders in CBR at the rates their fixed-QP points reached.
    cbr_extra: bool
    #: Encoders run on some classes only: name -> classes.
    limited: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def encoder(self, name: str) -> CtcEncoder:
        return next(e for e in self.encoders if e.name == name)

    def runs(self, encoder: CtcEncoder, sequence: sequences.Sequence) -> bool:
        classes = dict(self.limited).get(encoder.name)
        return classes is None or sequence.sequence_class in classes


#: VTM and libaom speed 0 are far from real time: on the cloud pipe-clean
#: (c7a, 2026-11-14) one 130-frame 270p point took VTM 12.6 min (QP 22) and
#: libaom speed 0 7.7 min (q 28). Every launch must end within 25 minutes of
#: running (owner rule: 30 min per run, ExpectedEnd within 40 min), so VTM runs
#: on A5 only and libaom speed 0 on A4/A5 (~14 min at 360p); A3 keeps HM as its
#: standards anchor. JVET_E (720p) is against HM alone.
_SLOW_FOR_LONG = ("vtm-ldp", "aom-good-s0-ld")
_LIMITED = (("vtm-ldp", ("A5",)), ("aom-good-s0-ld", ("A4", "A5")))

AOM_LD = Condition("aom-ld", "AOM CTC v8 low delay, classes A3-A5 (8-bit)", sequences.AOM_LD,
                   130, ALL_ENCODERS, ("aom-good-s0-ld", "hm-ldp", "vtm-ldp", "x264-medium"),
                   cbr_extra=True, limited=_LIMITED)
#: JVET codes class E in full (600 frames); the first JVET_E_FRAMES are coded
#: here so that every run stays within the owner's 30-minute limit (HM: ~18 min
#: per QP-22 point at 150 frames). A stated deviation from the JVET CTC.
JVET_E_FRAMES = 150
JVET_E = Condition("jvet-e", "JVET low-delay P, class E (first 150 frames)", sequences.JVET_E,
                   JVET_E_FRAMES,
                   tuple(e for e in ALL_ENCODERS if e.name not in _SLOW_FOR_LONG),
                   ("hm-ldp", "x264-medium"), cbr_extra=False)
CONDITIONS = {c.name: c for c in (AOM_LD, JVET_E)}
