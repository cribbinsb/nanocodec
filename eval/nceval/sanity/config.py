"""The sanity-report configuration (eval/configs/sanity.json).

One file names what is measured (clip sets, coding structures, encoders) and
the thresholds every check uses, so a report states exactly what it tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..clips import ClipSet, load_clipset
from ..sweep import SweepConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "eval" / "configs" / "sanity.json"
# A sanity run is measurement, not timing; the project caps one agent's job
# at 8 workers and memory-scoped runs at 4.
DEFAULT_WORKERS = 4


@dataclass(frozen=True)
class MeasuredSet:
    """Clips x encoders measured with one coding structure."""
    name: str
    clipset: Path
    clips: tuple[str, ...]  # empty: every clip of the clip set
    max_frames: int | None
    sweep: SweepConfig
    encoders: tuple[str, ...]
    checks: tuple[str, ...]

    def load_clips(self, data_root: Path) -> ClipSet:
        clipset = load_clipset(self.clipset, data_root)
        if not self.clips:
            return clipset
        by_name = {c.name: c for c in clipset.clips}
        missing = [n for n in self.clips if n not in by_name]
        if missing:
            raise ValueError(f"set {self.name}: clips {missing} not in {self.clipset}")
        return ClipSet(name=clipset.name, clips=tuple(by_name[n] for n in self.clips))


@dataclass(frozen=True)
class SanityConfig:
    sets: tuple[MeasuredSet, ...]
    ordering_tolerance: float
    ordering_metrics: tuple[str, ...]
    ordering_chains: tuple[tuple[str, ...], ...]
    outlier_limit_percent: float
    outlier_metrics: tuple[str, ...]
    outlier_references: dict[str, str]
    decay_slope_limit: float
    decay_margin: float
    tools: dict[str, float]
    cbr_miss_fraction: float
    cbr_results: Path
    documented_commands: dict[str, dict]
    source: Path = field(default=DEFAULT_CONFIG)


def load_config(path: Path, data_root: Path, workers: int = DEFAULT_WORKERS) -> SanityConfig:
    raw = json.loads(path.read_text())
    sets = tuple(_parse_set(s, data_root, workers) for s in raw["sets"])
    ordering, outliers, decay = raw["ordering"], raw["outliers"], raw["decay"]
    return SanityConfig(
        sets=sets,
        ordering_tolerance=float(ordering["tolerance"]),
        ordering_metrics=tuple(ordering["metrics"]),
        ordering_chains=tuple(tuple(chain) for chain in ordering["chains"]),
        outlier_limit_percent=float(outliers["limit_percent"]),
        outlier_metrics=tuple(outliers["metrics"]),
        outlier_references=dict(outliers["references"]),
        decay_slope_limit=float(decay["slope_limit_db_per_frame"]),
        decay_margin=float(decay["margin_vs_reference_db_per_frame"]),
        tools={k: float(v) for k, v in raw["tools"].items()},
        cbr_miss_fraction=float(raw["harness"]["cbr_miss_fraction"]),
        cbr_results=data_root / raw["harness"]["cbr_results"],
        documented_commands=dict(raw["harness"]["documented_commands"]),
        source=path,
    )


def _parse_set(raw: dict, data_root: Path, workers: int) -> MeasuredSet:
    structure = raw["structure"]
    single = bool(structure.get("single_keyframe", False))
    sweep = SweepConfig(intra_period_seconds=float(structure.get("intra_period_seconds", 0)),
                        temporal_layers=int(structure.get("temporal_layers", 1)),
                        single_keyframe=single, max_frames=raw.get("max_frames"),
                        workers=workers,
                        points_per_curve=int(raw.get("points_per_curve",
                                                     SweepConfig.points_per_curve)),
                        include_xpsnr=raw.get("include_xpsnr"))
    clipset = Path(raw["clipset"])
    return MeasuredSet(name=raw["name"],
                       clipset=clipset if clipset.is_absolute() else data_root / clipset,
                       clips=tuple(raw.get("clips", [])), max_frames=raw.get("max_frames"),
                       sweep=sweep, encoders=tuple(raw["encoders"]),
                       checks=tuple(raw["checks"]))
