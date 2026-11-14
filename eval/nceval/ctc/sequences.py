"""The standards lens's test sequences: where each comes from, its checksum and its licence.

Two sets (docs/analysis/standards-evaluation-methodology-2026-11-14.md):

- `AOM_LD`: the 8-bit sequences of classes A3, A4 and A5 of the AOM AV2 CTC
  v8 (CWG-F384o, Tables 3-5), from https://media.xiph.org/video/aomctc/test_set/,
  each with the MD5 the CTC lists.
- `JVET_E`: JVET class E (the conferencing class of JCTVC-L1100 / JVET-T2010)
  from https://media.xiph.org/video/derf/y4m/. derf publishes no checksums; the
  instance records the SHA-256 it downloaded.

Every sequence has a licence tier, so each table can be produced with and
without the research-use sequences (coordinator, 2026-11-14). Dropping a tier
is a change to `HEADLINE_TIERS` alone: results are keyed by sequence, nothing
is re-run.

- FREE: public domain, CC BY-SA, or "no restrictions" in the file's notice.
- RESEARCH: "may be used for research purposes only" (NTIA/ITS).

Sequences whose notice limits use to developing standards (Vidyo3, Vidyo4,
ParkJoy) are left out entirely: this is product evaluation, not standards work.
10-bit sources (DrivingPOV, RollerCoaster, SparksElevator) are left out because
the project codes 8-bit only.

The sequences are downloaded on the rented instance only and never
redistributed. They are an evaluation lens: sealed against tuning
(`nceval.pool.SEALED_SETS`, set `standards_ctc`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

AOM_BASE = "https://media.xiph.org/video/aomctc/test_set"
DERF_INDEX = "https://media.xiph.org/video/derf/"
DERF_BASE = f"{DERF_INDEX}y4m"


class Tier(str, Enum):
    FREE = "free"
    RESEARCH = "research"


#: Tiers in the headline tables; the others are reported as a labelled addition.
#: Both, by the owner's decision of 2026-11-14 (the report still gives the
#: FREE-only figure beside each table).
HEADLINE_TIERS = (Tier.FREE, Tier.RESEARCH)


@dataclass(frozen=True)
class Licence:
    tier: Tier
    summary: str
    notice_url: str


@dataclass(frozen=True)
class Sequence:
    name: str
    sequence_class: str  # the CTC's class label (A3, A4, A5, E)
    url: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    licence: Licence
    md5: str | None = None  # the CTC's published MD5 of the .y4m (None: not published)
    #: The source's family key (`nceval.pool.family_key`), when a copy of the same
    #: source is in our clip sets under another name; else the sequence's own.
    source_family: str | None = None

    @property
    def family(self) -> str:
        return self.source_family or self.name.lower()

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3 // 2


def _aom_notice(folder: str, stem: str, suffix: str = "_Copyright.txt") -> str:
    return f"{AOM_BASE}/{folder}/{stem}{suffix}"


def _vidyo(folder: str, stem: str) -> Licence:
    return Licence(Tier.FREE, "public domain (author: Vidyo)", _aom_notice(folder, stem))


def _ntia(folder: str, stem: str) -> Licence:
    return Licence(Tier.RESEARCH, "NTIA/ITS: research use only, acknowledgement kept",
                   _aom_notice(folder, stem))


def _aom(folder: str, file: str, cls: str, size: tuple[int, int], fps: tuple[int, int],
         licence: Licence, md5: str, family: str | None = None) -> Sequence:
    return Sequence(file.removesuffix(".y4m"), cls, f"{AOM_BASE}/{folder}/{file}", *size, *fps,
                    licence, md5, family)


_A3, _A4, _A5 = "a3_720p", "a4_360p", "a5_270p"
#: KristenAndSara is also in the screening pool (an excerpt of the same Xiph file,
#: family "kristen-sara"): it was screened on before this lens existed, and sealing
#: the family takes it out of the pool from now on.
KRISTEN_AND_SARA_FAMILY = "kristen-sara"
_P720, _P360, _P270 = (1280, 720), (640, 360), (480, 270)
_FPS30, _FPS60, _FPS2997, _FPS5994, _FPS25 = (30, 1), (60, 1), (30000, 1001), (60000, 1001), (25, 1)

AOM_LD: tuple[Sequence, ...] = (
    _aom(_A3, "Johnny_1280x720_60.y4m", "A3", _P720, _FPS60,
         _vidyo(_A3, "Johnny_1280x720_60"), "1d6aab4003385c255262a88f4df7cccc"),
    _aom(_A3, "KristenAndSara_1280x720_60.y4m", "A3", _P720, _FPS60,
         _vidyo(_A3, "KristenAndSara_1280x720_60"), "43bb17b78086d0183642c7b74cc1c903",
         KRISTEN_AND_SARA_FAMILY),
    _aom(_A3, "ControlledBurn_1280x720p30_420.y4m", "A3", _P720, _FPS30,
         _ntia(_A3, "ControlledBurn_1280x720p30_420"), "875d857d8ef1b7f1b653c4c530669005"),
    _aom(_A3, "WestWindEasy_1280x720p30_420.y4m", "A3", _P720, _FPS30,
         _ntia(_A3, "WestWindEasy_1280x720p30_420"), "da8afe9c91a260c835b39f58b13c99e7"),
    _aom(_A4, "BlueSky_360p25_v2.y4m", "A4", _P360, _FPS25,
         Licence(Tier.FREE, "TU München / Taurus Media: \"Restrictions of use: No restrictions\"",
                 _aom_notice(_A4, "BlueSky_360p25")), "722261f9e84578203ea9486673ffb274"),
    _aom(_A4, "Stockholm_640x360_5994.y4m", "A4", _P360, _FPS5994,
         Licence(Tier.FREE, "SVT (2001 set): \"Restrictions of use: No restrictions\"",
                 _aom_notice(_A4, "Stockholm_640x360_5994")), "62d126930a6c8f61f5267f0b21746cde"),
    _aom(_A4, "RedKayak_360_2997.y4m", "A4", _P360, _FPS2997,
         _ntia(_A4, "RedKayak_360_2997"), "4205131788b94668c4feef73ce44aec9"),
    _aom(_A4, "SnowMountain_640x360_2997.y4m", "A4", _P360, _FPS2997,
         _ntia(_A4, "SnowMountain_640x360_2997"), "63ae6b7fb786883c2144552e98d0c4dd"),
    _aom(_A4, "SpeedBag_640x360_2997.y4m", "A4", _P360, _FPS2997,
         _ntia(_A4, "SpeedBag_640x360_2997"), "8bfe53f0215b749744b3947e16ab09a4"),
    _aom(_A4, "TouchdownPass_640x360_2997.y4m", "A4", _P360, _FPS2997,
         _ntia(_A4, "TouchdownPass_640x360_2997"), "dd98868ad8286eb0fb0b8c00827098a1"),
    _aom(_A5, "FourPeople_480x270_60.y4m", "A5", _P270, _FPS60,
         _vidyo(_A5, "FourPeople_480x270_60"), "4d35db57a2377e4211134db345dea225"),
    _aom(_A5, "Vertical_Bayshore_270x480_2997.y4m", "A5", (270, 480), _FPS2997,
         Licence(Tier.FREE, "CC BY-SA 4.0, (c) James Leslie Skinner",
                 _aom_notice(_A5, "Vertical_Bayshore_270x480_2997")),
         "6ec078264f2ae2c99e2fbe108b774847"),
)


def _derf(file: str, family: str | None = None) -> Sequence:
    stem = file.removesuffix(".y4m")
    return Sequence(f"{stem}_E", "E", f"{DERF_BASE}/{file}", *_P720, *_FPS60,
                    Licence(Tier.FREE, "public domain (derf listing; author Vidyo, as in the "
                            "AOM CTC copies' notices)", DERF_INDEX), None,
                    family or stem.lower())


JVET_E: tuple[Sequence, ...] = (
    _derf("FourPeople_1280x720_60.y4m", "fourpeople_480x270_60"),
    _derf("Johnny_1280x720_60.y4m", "johnny_1280x720_60"),
    _derf("KristenAndSara_1280x720_60.y4m", KRISTEN_AND_SARA_FAMILY))

ALL: tuple[Sequence, ...] = AOM_LD + JVET_E
#: Families no screen or tuning study may draw (`nceval.pool.load_pool`).
SEALED_FAMILIES = frozenset(s.family for s in ALL)


def by_name(name: str) -> Sequence:
    for sequence in ALL:
        if sequence.name == name:
            return sequence
    raise KeyError(f"no CTC sequence {name!r}")


def headline(sequences: tuple[Sequence, ...] | list[Sequence]) -> list[Sequence]:
    """The sequences whose licence tier is in the headline tables."""
    return [s for s in sequences if s.licence.tier in HEADLINE_TIERS]
