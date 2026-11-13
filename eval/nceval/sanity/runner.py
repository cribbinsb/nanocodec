"""`python3 -m nceval sanity`: measure what is missing, run the checks, write the report."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..config import load_paths
from ..results import ResultsDb
from .config import MeasuredSet, load_config
from .measure import measure_set

STEPS = ("measure", "checks")


def clip_range(measured: MeasuredSet, text: str, data_root: Path) -> MeasuredSet:
    """The set restricted to its clips FIRST-LAST (1-based, inclusive)."""
    first, last = (int(v) for v in text.split("-"))
    names = [c.name for c in measured.load_clips(data_root).clips][first - 1:last]
    return replace(measured, clips=tuple(names))


def only_clips(measured: MeasuredSet, names: list[str], data_root: Path) -> MeasuredSet:
    """The set restricted to the named clips (in the set's order)."""
    chosen = [c.name for c in measured.load_clips(data_root).clips if c.name in names]
    missing = set(names) - set(chosen)
    if missing:
        raise SystemExit(f"set {measured.name} has no clips {sorted(missing)}")
    return replace(measured, clips=tuple(chosen))


def run_sanity(args: argparse.Namespace, harness_commit: str) -> int:
    paths = load_paths()
    config = load_config(Path(args.config), paths.data_root, args.workers)
    steps = tuple(args.only.split(",")) if args.only else STEPS
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; steps are {', '.join(STEPS)}")
    chosen = set(args.sets.split(",")) if args.sets else {s.name for s in config.sets}
    sets = [s for s in config.sets if s.name in chosen]
    if args.clip_range:
        sets = [clip_range(s, args.clip_range, paths.data_root) for s in sets]
    if args.clips:
        sets = [only_clips(s, args.clips.split(","), paths.data_root) for s in sets]
    db = ResultsDb(paths.results_db)
    if "measure" in steps:
        for measured in sets:
            print(f"== measure {measured.name}: {measure_set(measured, paths, db, harness_commit)}",
                  flush=True)
    if "checks" in steps:
        from .checks import run_checks
        from .markdown import write_report
        report = run_checks(config, sets, paths, db)
        out = Path(args.out) if args.out else paths.data_root / "reports"
        markdown, data = write_report(report, out)
        print(f"wrote {markdown} and {data}")
        return 1 if report.has_flags_at_or_above("high") else 0
    return 0
