"""Read measured curves, per-frame data and kept bitstreams for the checks.

A curve is one (encoder, structure, clip): its passing points (rate and
sequence metrics), any conformance failures, and where each point's kept
bitstream lies. Points above the clip's 50%-of-source cap are stored without
metrics by the sweep and are ignored here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from pathlib import Path

from ..codecs import Codec
from ..encoders import resolve_encoder
from ..results import ResultsDb


@dataclass(frozen=True)
class Point:
    encode_id: int
    quantizer: int
    frames: int
    kbps: float
    metrics: dict[str, float]


@dataclass
class Curve:
    encoder: str
    clip: str
    points: list[Point] = field(default_factory=list)       # scored, by rate
    unscored: list[Point] = field(default_factory=list)     # passed, over the cap
    failures: list[tuple[int, str]] = field(default_factory=list)  # (quantizer, reason)

    def series(self, metric: str) -> tuple[list[float], list[float]]:
        """(kbps, metric) of the points that have this metric."""
        chosen = [p for p in self.points if metric in p.metrics]
        return [p.kbps for p in chosen], [p.metrics[metric] for p in chosen]

    def nearest(self, kbps: float) -> Point | None:
        """The scored point closest to a rate (in log terms)."""
        if not self.points:
            return None
        return min(self.points, key=lambda p: abs(log(p.kbps / kbps)))


@dataclass(frozen=True)
class FrameRow:
    frame: int
    bytes: int
    frame_type: str
    psnr_y: float | None
    vmaf_v1: float | None


class CurveStore:
    """Curves of one structure from a results database, plus kept streams."""

    def __init__(self, db: ResultsDb, structure: str, platform: str, work: Path):
        self.db = db
        self.structure = structure
        self.platform = platform
        self.work = work

    def curve(self, encoder: str, clip: str) -> Curve:
        curve = Curve(encoder=encoder, clip=clip)
        for encode_id, q, frames, kbps, status, reason in self.db.curve_encodes(
                encoder, self.structure, clip, self.platform):
            if status != "pass":
                curve.failures.append((int(q), reason))
                continue
            point = Point(int(encode_id), int(q), int(frames), float(kbps),
                          self.db.encode_metrics(encode_id))
            (curve.points if point.metrics else curve.unscored).append(point)
        return curve

    def frames(self, point: Point) -> list[FrameRow]:
        return [FrameRow(*row) for row in self.db.frame_rows(point.encode_id)]

    def bitstream(self, encoder: str, clip: str, point: Point) -> Path | None:
        """The kept stream of a point (the sweep keeps one fingerprint directory per curve)."""
        adapter = resolve_encoder(encoder)
        if adapter is None:
            return None
        name = f"q{point.quantizer:03d}-{point.frames}f{adapter.codec.bitstream_suffix}"
        found = sorted((self.work / encoder / self.structure / clip).glob(f"*/{name}"))
        return found[0] if found else None

    def codec(self, encoder: str) -> Codec | None:
        adapter = resolve_encoder(encoder)
        return adapter.codec if adapter else None

