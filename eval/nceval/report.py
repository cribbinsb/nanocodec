"""BD-rate reports from the results database (eval/README.md "Methodology").

For every encoder in the database (same coding structure and platform), the
BD-rate against the anchor is computed per clip, then averaged per content
class and overall (mean of per-clip values; every clip weighs the same).
A clip whose curves overlap too little is shown but left out of the means.

Synthetic-noise clips are scored against their clean originals. There,
quality is not monotonic in bitrate (more bits reproduce more noise and move
the picture further from the clean original), so BD-rate is meaningless for
them: they get no BD-rate at all. noise_summary() reports instead the best
clean-reference quality an encoder reaches and the quality at fixed bitrates.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import numpy as np

from .bdrate import BdResult, bd_rate
from .clips import ClipSet
from .results import ResultsDb
from .metrics import structure_for_metric

# Classes with no BD-rate; reported by noise_summary() instead.
SEPARATE_CLASSES = ("synthetic_noise",)
# Fixed bitrates (at 720p30, scaled by pixel rate per clip) for noise_summary.
NOISE_REPORT_KBPS_720P30 = (200.0, 400.0)


@dataclass(frozen=True)
class NoiseRow:
    encoder: str
    clip: str
    metric: str
    best_quality: float
    best_kbps: float
    at_fixed_kbps: dict[float, float | None]  # nominal 720p30 kbps -> quality


def noise_summary(db: ResultsDb, clipset: ClipSet, metric: str, structure: str,
                  platform: str, encoders: list[str], rate_scale) -> list[NoiseRow]:
    """Clean-reference quality of each encoder on the synthetic-noise clips.

    `rate_scale(clip)` converts the nominal 720p30 bitrates to the clip's
    pixel rate. Quality at a fixed rate is interpolated linearly in
    log(bitrate) between measured points; outside the measured range it is None.
    """
    structure = structure_for_metric(structure, metric)
    rows = []
    for clip in clipset.clips:
        if clip.content_class not in SEPARATE_CLASSES:
            continue
        for encoder in encoders:
            points = [(r, q) for r, q in db.rd_points(encoder, structure, clip.name, platform, metric)
                      if clip.allows_kbps(r)]
            if not points:
                continue
            best_kbps, best_quality = max(points, key=lambda p: p[1])
            fixed = {}
            for nominal in NOISE_REPORT_KBPS_720P30:
                fixed[nominal] = _interpolate_log_rate(points, nominal * rate_scale(clip))
            rows.append(NoiseRow(encoder, clip.name, metric, best_quality, best_kbps, fixed))
    return rows


def _interpolate_log_rate(points: list[tuple[float, float]], kbps: float) -> float | None:
    rates = [math.log(r) for r, _ in points]
    if not rates or not rates[0] <= math.log(kbps) <= rates[-1]:
        return None
    return float(np.interp(math.log(kbps), rates, [q for _, q in points]))


@dataclass(frozen=True)
class ClipBd:
    encoder: str
    clip: str
    content_class: str
    result: BdResult | None  # None: missing curve(s)
    note: str = ""


def compute(db: ResultsDb, clipset: ClipSet, anchor: str, metric: str, structure: str,
            platform: str, encoders: list[str] | None = None,
            max_frames: int | None = None, fingerprint_of=None) -> list[ClipBd]:
    """BD-rate of every encoder (default: all in the database) vs the anchor, per clip.

    `max_frames` selects the curves encoded at min(clip length, max_frames)
    frames; without it every stored length is used (one length per namespace).
    `fingerprint_of(encoder, clip)` names the configuration whose curve to
    use (default, or None from it: the most recently measured one).
    """
    def fingerprint(encoder: str, clip) -> str | None:
        return fingerprint_of(encoder, clip) if fingerprint_of else None
    structure = structure_for_metric(structure, metric)
    rows: list[ClipBd] = []
    for encoder in encoders or db.encoders(structure, platform):
        if encoder == anchor:
            continue
        for clip in clipset.clips:
            if clip.content_class in SEPARATE_CLASSES:
                continue  # no BD-rate for these (see module docstring)
            frames = min(clip.frames, max_frames) if max_frames else None
            a = [(r, q) for r, q in db.rd_points(anchor, structure, clip.name, platform, metric,
                                                 frames, fingerprint(anchor, clip))
                 if clip.allows_kbps(r)]
            t = [(r, q) for r, q in db.rd_points(encoder, structure, clip.name, platform, metric,
                                                 frames, fingerprint(encoder, clip))
                 if clip.allows_kbps(r)]
            if len(a) < 4 or len(t) < 4:
                rows.append(ClipBd(encoder, clip.name, clip.content_class, None,
                                   f"points: anchor {len(a)}, test {len(t)}"))
                continue
            try:
                result = bd_rate([r for r, _ in a], [q for _, q in a],
                                 [r for r, _ in t], [q for _, q in t])
                rows.append(ClipBd(encoder, clip.name, clip.content_class, result))
            except ValueError as err:
                rows.append(ClipBd(encoder, clip.name, clip.content_class, None, str(err)))
    return rows


@dataclass(frozen=True)
class EncoderSummary:
    by_class: dict[str, float]   # mean BD-rate per content class (reliable clips only)
    overall: float | None        # mean over all non-separate classes
    clips_used: int
    clips_excluded: int          # unreliable overlap or missing curves


def summarise(rows: list[ClipBd]) -> dict[str, EncoderSummary]:
    """Per encoder: mean BD-rate per class and overall (every clip weighs the same)."""
    by_encoder: dict[str, list[ClipBd]] = defaultdict(list)
    for row in rows:
        by_encoder[row.encoder].append(row)
    out = {}
    for encoder, enc_rows in by_encoder.items():
        usable = [r for r in enc_rows if r.result and r.result.reliable]
        by_class: dict[str, list[float]] = defaultdict(list)
        for r in usable:
            by_class[r.content_class].append(r.result.bd_rate_percent)
        overall = [v for c, vals in by_class.items() if c not in SEPARATE_CLASSES for v in vals]
        out[encoder] = EncoderSummary(
            by_class={c: mean(v) for c, v in by_class.items()},
            overall=mean(overall) if overall else None,
            clips_used=len(usable), clips_excluded=len(enc_rows) - len(usable))
    return out


def ordered_classes(rows: list[ClipBd]) -> list[str]:
    return sorted({row.content_class for row in rows}, key=lambda c: (c in SEPARATE_CLASSES, c))


def format_report(rows: list[ClipBd], anchor: str, metric: str, per_clip: bool) -> str:
    label = {"vmaf_v1": "vmaf_v1 (v1.0.16_3d0h,1080 display10bit)",
             "vmaf_neg": "vmaf_neg (legacy v0.6.1neg)",
             "vmaf": "vmaf (legacy v0.6.1)",
             "ms_ssim": "ms_ssim (native8bit float)"}.get(metric, metric)
    lines = [f"BD-rate vs {anchor} on {label} (negative = fewer bits than the anchor)", ""]
    classes = ordered_classes(rows)
    header = f"{'encoder':<20}" + "".join(f"{c:>12}" for c in classes) + f"{'overall':>12}"
    lines += [header, "-" * len(header)]
    for encoder, summary in sorted(summarise(rows).items()):
        cells = [f"{summary.by_class[c]:>11.1f}%" if c in summary.by_class else f"{'-':>12}"
                 for c in classes]
        cells.append(f"{summary.overall:>11.1f}%" if summary.overall is not None
                     else f"{'-':>12}")
        note = (f"  ({summary.clips_excluded} clip(s) unreliable/missing)"
                if summary.clips_excluded else "")
        lines.append(f"{encoder:<20}" + "".join(cells) + note)

    if per_clip:
        lines += ["", f"{'encoder':<20}{'clip':<28}{'BD-rate':>10}{'overlap':>9}  note"]
        for row in sorted(rows, key=lambda r: (r.encoder, r.clip)):
            if row.result and row.result.bd_rate_percent is not None:
                flag = "" if row.result.reliable else "UNRELIABLE (small overlap)"
                lines.append(f"{row.encoder:<20}{row.clip:<28}"
                             f"{row.result.bd_rate_percent:>9.1f}%"
                             f"{row.result.overlap_fraction:>9.2f}  {flag or row.note}")
            else:
                lines.append(f"{row.encoder:<20}{row.clip:<28}{'-':>10}{'-':>9}  {row.note}")
    return "\n".join(lines)


def format_gate_levels(db: ResultsDb, clips: list[str] | None = None) -> str:
    """How strongly each encoder's passed streams were checked (nceval.gatepolicy)."""
    lines = ["Conformance gate levels:"]
    for encoder, counts in sorted(db.gate_levels(clips).items()):
        lines.append(f"  {encoder}: " + ", ".join(
            f"{level or 'unrecorded'} {count}" for level, count in sorted(counts.items())))
    return "\n".join(lines)


def format_failures(db: ResultsDb, clips: list[str] | None = None) -> str:
    failures = db.failures(clips)
    if not failures:
        return "Conformance: no stream failed (unchecked streams are checked when a change lands)."
    lines = [f"Conformance FAILURES ({len(failures)}):"]
    lines += [f"  {enc} / {structure} / {clip} / q{q}: {reason}"
              for enc, structure, clip, q, reason in failures]
    return "\n".join(lines)
