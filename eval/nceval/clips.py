"""Clip sets: the raw YUV test material and its metadata.

A clip set is a JSON file listing clips stored either as raw 8-bit 4:2:0 YUV
(see eval/clipset/extract.py) or as a high-bitrate encode (the rotating clip
pool, eval/clipset/build_pool.py) that the harness decodes on the fly
(nceval.clipframes). Either way the file is the clip's source: its bitrate is
the source bitrate the 50% rule applies to. Each clip carries the metadata the
harness needs to enforce the test rules, in particular the source bitrate:
no test point may exceed 50% of it (eval/README.md "Methodology: Clips").

Example clip set file:

    {
      "name": "quick",
      "clips": [
        {"name": "office_static", "yuv": "clips/office_static_720p30.yuv",
         "width": 1280, "height": 720, "fps": 30, "frames": 300,
         "source_kbps": 4200, "content_class": "static", "split": "train"}
      ]
    }

Relative "yuv" paths are resolved against the data root. "family" names the
source a clip was cut from (default: its "source", else its name): the pool
draws never put two clips of one family in one draw, nor one family in two
levels of a day.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Never test a clip above this fraction of its source bitrate (eval/README.md "Methodology").
MAX_FRACTION_OF_SOURCE_BITRATE = 0.5

CONTENT_CLASSES = ("static", "pan", "bodycam", "drone", "crowd", "lowlight", "texture",
                   "synthetic_noise")
SPLITS = ("train", "val", "test")


#: File suffix of a clip stored as raw I420; anything else is an encoded file.
RAW_SUFFIX = ".yuv"


@dataclass(frozen=True)
class Clip:
    name: str
    yuv: Path  # the stored clip: raw I420 (.yuv) or an encoded file (nceval.clipframes)
    width: int
    height: int
    fps: float
    frames: int
    source_kbps: float
    content_class: str
    split: str
    clean_reference: Path | None = None  # synthetic-noise clips: score against this
    #: First frame a measurement uses (the pool rotates it daily, nceval.pool).
    start_frame: int = 0
    family: str = ""
    #: Source quality (eval/clipset/grade.py; None: ungraded): grade A clean,
    #: B mediocre (usable downscaled), C poor; source height over clip height;
    #: source bits per stored pixel per frame; the source's dataset.
    grade: str | None = None
    downscale: float = 1.0
    source_bpp: float | None = None
    dataset: str = ""

    @property
    def encoded(self) -> bool:
        """Stored as an encoded file, decoded on the fly (nceval.clipframes)."""
        return self.yuv.suffix != RAW_SUFFIX

    @property
    def direct(self) -> bool:
        """Its frames can be read from the file itself (raw, from frame 0)."""
        return not self.encoded and self.start_frame == 0

    @property
    def frame_bytes(self) -> int:
        """Size of one 8-bit 4:2:0 frame."""
        return self.width * self.height * 3 // 2

    @property
    def max_test_kbps(self) -> float:
        return self.source_kbps * MAX_FRACTION_OF_SOURCE_BITRATE

    @property
    def metric_reference(self) -> Path:
        """The picture quality is measured against (clean original if we have one)."""
        return self.clean_reference or self.yuv

    def allows_kbps(self, kbps: float) -> bool:
        return kbps <= self.max_test_kbps


@dataclass(frozen=True)
class ClipSet:
    name: str
    clips: tuple[Clip, ...]

    def select(self, *, split: str | None = None, content_class: str | None = None) -> ClipSet:
        chosen = tuple(
            c for c in self.clips
            if (split is None or c.split == split)
            and (content_class is None or c.content_class == content_class)
        )
        return ClipSet(name=self.name, clips=chosen)


def load_clipset(path: Path, data_root: Path) -> ClipSet:
    raw = json.loads(Path(path).read_text())
    clips = tuple(_parse_clip(entry, data_root) for entry in raw["clips"])
    names = [c.name for c in clips]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate clip names")
    return ClipSet(name=raw["name"], clips=clips)


def _parse_clip(entry: dict, data_root: Path) -> Clip:
    def resolve(p: str | None) -> Path | None:
        if p is None:
            return None
        path = Path(p)
        return path if path.is_absolute() else data_root / path

    clip = Clip(
        name=entry["name"],
        yuv=resolve(entry["yuv"]),
        width=int(entry["width"]),
        height=int(entry["height"]),
        fps=float(entry["fps"]),
        frames=int(entry["frames"]),
        source_kbps=float(entry["source_kbps"]),
        content_class=entry["content_class"],
        split=entry["split"],
        clean_reference=resolve(entry.get("clean_reference")),
        family=entry.get("family") or entry.get("source") or entry["name"],
        grade=entry.get("grade"),
        downscale=float(entry.get("downscale", 1.0)),
        source_bpp=entry.get("source_bpp"),
        dataset=entry.get("dataset", ""),
    )
    if clip.content_class not in CONTENT_CLASSES:
        raise ValueError(f"clip {clip.name}: unknown content class {clip.content_class!r}")
    if clip.split not in SPLITS:
        raise ValueError(f"clip {clip.name}: unknown split {clip.split!r}")
    if clip.width % 2 or clip.height % 2:
        raise ValueError(f"clip {clip.name}: 4:2:0 needs even dimensions")
    return clip
