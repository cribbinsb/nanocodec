"""Run every configured check over the stored results of each measured set."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime

from ..config import Paths, find_tool
from ..results import ResultsDb
from . import harness
from .activity import ToolLimits
from .config import MeasuredSet, SanityConfig
from .curves import CurveStore
from .decay import check_decay
from .flags import Flag, Severity, sort_flags
from .ordering import check_ordering
from .outliers import check_outliers
from .tools import check_tools


@dataclass
class SanityReport:
    created: str
    identity: dict
    flags: list[Flag] = field(default_factory=list)
    tables: dict = field(default_factory=dict)

    def has_flags_at_or_above(self, severity: str) -> bool:
        level = Severity.parse(severity)
        return any(f.severity >= level for f in self.flags)


def run_checks(config: SanityConfig, sets: list[MeasuredSet], paths: Paths,
               db: ResultsDb) -> SanityReport:
    report = SanityReport(created=datetime.now().strftime("%Y-%m-%d %H:%M"),
                          identity=_identity(config, sets, paths))
    flags: list[Flag] = []
    for measured in sets:
        flags += _check_set(config, measured, paths, db, report.tables)
    flags += harness.check_cbr(config.cbr_results, config.cbr_miss_fraction)
    flags += harness.check_commands(config.documented_commands, paths)
    flags += harness.check_clip_caps(paths.clips)
    report.flags = sort_flags(unique_flags(flags))
    return report


def unique_flags(flags: list[Flag]) -> list[Flag]:
    """Drop repeats of a flag that overlapping sets (same curves) raise again."""
    seen, out = set(), []
    for flag in flags:
        key = (flag.check, flag.subject.split("/", 1)[-1], flag.finding)
        if key not in seen:
            seen.add(key)
            out.append(flag)
    return out


def _check_set(config: SanityConfig, measured: MeasuredSet, paths: Paths, db: ResultsDb,
               tables: dict) -> list[Flag]:
    store = CurveStore(db, measured.sweep.structure_tag, platform.machine(), paths.work)
    clips = [c.name for c in measured.load_clips(paths.data_root).clips]
    label, flags, own = measured.name, [], tables.setdefault(measured.name, {})
    references = {t: r for t, r in config.outlier_references.items()
                  if t in measured.encoders}
    if "harness" in measured.checks:
        flags += harness.check_curves(store, clips, measured.encoders,
                                      config.ordering_metrics, label)
    if "ordering" in measured.checks:
        chains = tuple(c for c in config.ordering_chains if all(e in measured.encoders for e in c))
        flags += check_ordering(store, clips, chains, config.ordering_metrics,
                                config.ordering_tolerance, label)
    if "outliers" in measured.checks:
        found, own["bd_rate"] = check_outliers(store, clips, references, config.outlier_metrics,
                                               config.outlier_limit_percent, measured.encoders,
                                               label)
        flags += found
    if "tools" in measured.checks:
        limits = ToolLimits(config.tools["skip_share_ratio"], config.tools["share_gap"],
                            config.tools["never_active_share"],
                            config.tools["keyframe_size_ratio"])
        found, own["tools"] = check_tools(store, clips, references, measured.encoders, limits,
                                          paths, label)
        flags += found
    if "decay" in measured.checks:
        found, own["decay"] = check_decay(store, clips, references, measured.encoders,
                                          config.decay_slope_limit, config.decay_margin, label)
        flags += found
    return flags


def _identity(config: SanityConfig, sets: list[MeasuredSet], paths: Paths) -> dict:
    try:
        binary = find_tool("nanocodec", paths)
        binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    except FileNotFoundError:
        binary, binary_sha = None, None
    return {"config": str(config.source), "results_db": str(paths.results_db),
            "work": str(paths.work), "nanocodec": str(binary), "nanocodec_sha256": binary_sha,
            "sets": {s.name: {"structure": s.sweep.structure_tag, "frames": s.max_frames,
                              "encoders": list(s.encoders), "clips": list(s.clips)}
                     for s in sets}}
