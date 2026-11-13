"""Re-measure stored curves under a new measurement profile, without re-deciding rates.

A metric change (e.g. METRICS_REVISION 5, native VMAF v1) changes no
bitstream, so a full `nceval sweep` (rate probes, a full-length calibration
encode, then every point) is wasted work. This re-measures in the current
profile's namespace from curves already stored elsewhere:

  anchors (no reconstruction: SVT, openh264, ...)
      their kept bitstreams are not re-encoded: each is decoded once more
      with the new metrics attached to the decode (under an explicit gate,
      by both standard decoders);
  our encoders (nanocodec-*)
      the binary may have changed, so each point is re-encoded, but on the
      clip's stored QP grid: no rate probes and no calibration encode.
      Points above the source-rate cap are excluded, as in a sweep.

Sources are (results DB, structure) pairs, searched in order per encoder and
clip; a preset (`nanocodec-av1@x`) with no stored grid of its own uses its
base encoder's. Kept streams are found under the source DB's data root
(`<root>/results/X.sqlite` -> `<root>/work/<encoder>/<structure>/<clip>/<fp>/`).

    python3 -m nceval remeasure --clipset main.json --split val \\
        --encoders svtav1-m12,nanocodec-av1 --source DB:ip10s-tl1 [--source ...] \\
        [--gate off|sample|full] [--workers 2]
"""

from __future__ import annotations

import platform
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .cachefile import entry_lock
from .clips import Clip
from .encoders.base import EncoderAdapter
from .gatepolicy import anchor_check, combination_key, point_check
import math
from concurrent.futures import ThreadPoolExecutor

from .budget import MAX_MINUTES, check as check_budget, estimate_minutes, point_cpu_seconds
from .sweep import BUDGET_PART_FILL, Sweep, SweepConfig, curve_work, payload_bytes


@dataclass(frozen=True)
class Source:
    database: Path
    structure: str

    @property
    def work(self) -> Path:
        return self.database.parent.parent / "work"

    def points(self, encoder: str, clip: str,
               frames: int | None = None) -> list[tuple[int, int, str]]:
        """(quantizer, frames, fingerprint) of the passed encodes of one curve.

        One curve: `frames` long if given, and its most recently measured
        fingerprint (several builds' curves of one encoder name coexist).
        """
        with sqlite3.connect(f"file:{self.database}?mode=ro", uri=True) as conn:
            latest = conn.execute(
                "SELECT fingerprint FROM encodes WHERE encoder=? AND clip=? AND structure=? "
                "AND platform=? AND conformance='pass' AND (? IS NULL OR frames=?) "
                "ORDER BY created DESC LIMIT 1",
                (encoder, clip, self.structure, platform.machine(), frames, frames)).fetchone()
            if latest is None:
                return []
            return conn.execute(
                "SELECT quantizer, frames, fingerprint FROM encodes WHERE encoder=? AND clip=? "
                "AND structure=? AND platform=? AND conformance='pass' AND fingerprint=? "
                "AND (? IS NULL OR frames=?) ORDER BY quantizer",
                (encoder, clip, self.structure, platform.machine(), latest[0],
                 frames, frames)).fetchall()


def stored_points(sources: list[Source], encoder: str, clip: str,
                  frames: int | None = None) -> tuple[Source, list] | None:
    """The first source holding this curve (a preset falls back to its base encoder).

    `frames` asks for a curve of that length (a namespace may also hold
    full-length curves of the same clip).
    """
    for name in dict.fromkeys([encoder, encoder.split("@")[0]]):
        for source in sources:
            points = source.points(name, clip, frames)
            if points:
                return source, points
    return None


class Remeasure:
    def __init__(self, sweep: Sweep, cfg: SweepConfig, sources: list[Source]):
        self.sweep, self.cfg, self.sources = sweep, cfg, sources

    def frames(self, clip: Clip) -> int | None:
        """The curve length this run re-measures (None: whatever is stored)."""
        return min(clip.frames, self.cfg.max_frames) if self.cfg.max_frames else None

    def curve(self, encoder: EncoderAdapter, clip: Clip) -> str:
        found = stored_points(self.sources, encoder.name, clip.name, self.frames(clip))
        if found is None:
            return f"skip {encoder.name}/{clip.name}: no stored curve"
        source, points = found
        frames = points[0][1]
        fingerprint = self.sweep.fingerprint(encoder, self.cfg, clip)
        work = curve_work(self.sweep.paths, encoder, self.cfg, clip, fingerprint)
        with entry_lock(work):  # the same curve lock a sweep takes
            self.sweep.db.clear_curve(encoder.name, self.cfg.structure_tag, clip.name, frames,
                                      self.sweep.platform, fingerprint)
            if encoder.verify_recon():
                passed = self._reencode(encoder, clip, [q for q, _, _ in points], frames,
                                        fingerprint, work)
            else:
                passed = self._rescore(encoder, clip, source, points, fingerprint)
        return f"done {encoder.name}/{clip.name}: {len(points)} points" + (
            "" if passed else " (CONFORMANCE FAILURES)")

    def job_cpu(self, encoder: EncoderAdapter, clip: Clip) -> float:
        """CPU-seconds for one stored curve: rescoring for anchors, full points for ours."""
        found = stored_points(self.sources, encoder.name, clip.name, self.frames(clip))
        return sum(point_cpu_seconds(clip.width, clip.height, frames,
                                     rescore=not encoder.verify_recon())
                   for _, frames, _ in found[1]) if found else 0.0

    def run(self, jobs: list, force: bool, workers: int = 1) -> list[str]:
        """Remeasure the jobs in balanced parts that each fit the wall-time limit.

        Each part is priced when it starts (and split again if free slots
        dropped); only a single curve that cannot fit is refused.
        """
        cost = sum(self.job_cpu(e, c) for e, c in jobs)
        minutes = estimate_minutes(cost)
        if len(jobs) > 1 and minutes > MAX_MINUTES:
            count = min(len(jobs), math.ceil(minutes / (MAX_MINUTES * BUDGET_PART_FILL)))
            ordered = sorted(jobs, key=lambda j: (j[1].content_class, j[1].name, j[0].name))
            lines = []
            for index in range(count):
                part = ordered[index::count]
                print(f"remeasure part {index + 1}/{count} ({len(part)} curves)", flush=True)
                lines += self.run(part, force, workers)
            return lines
        check_budget(f"remeasure of {len(jobs)} curves", cost,
                     "fewer clips or encoders, or split the run", force)
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(lambda job: self.curve(*job), jobs))

    def _rescore(self, encoder: EncoderAdapter, clip: Clip, source: Source, points: list,
                 fingerprint: str) -> bool:
        passed = True
        for quantizer, frames, stored_fingerprint in points:
            matches = list((source.work / encoder.name / source.structure / clip.name /
                            stored_fingerprint).glob(
                f"q{quantizer:03d}-{frames}f{encoder.codec.bitstream_suffix}"))
            if not matches:
                print(f"missing kept stream {encoder.name}/{clip.name}/q{quantizer}: point dropped")
                continue
            bitstream = matches[0]
            kbps = payload_bytes(bitstream, encoder.codec, frames) * 8 / 1000 / (frames / clip.fps)
            passed &= self.sweep._check_and_measure(encoder, clip, self.cfg, quantizer, frames,
                                                   bitstream.parent, fingerprint, bitstream,
                                                   kbps, anchor_check(self.cfg.gate))
        return passed

    def _reencode(self, encoder: EncoderAdapter, clip: Clip, quantizers: list[int], frames: int,
                  fingerprint: str, work: Path) -> bool:
        combination = combination_key(fingerprint, {}, clip.name)
        return all([self.sweep._measure_point(
            encoder, clip, self.cfg, q, frames, work, fingerprint,
            point_check(self.cfg.gate, combination, index, len(quantizers)))
            for index, q in enumerate(quantizers)])
