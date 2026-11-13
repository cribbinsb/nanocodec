"""Fill in whatever the sanity checks need and the database lacks.

Each measured set is an ordinary fingerprinted sweep (nceval.sweep): curves
that are already current are skipped, so a rerun encodes only what changed
(a new encoder binary, a new clip). Every stream passes the two-decoder
conformance gate before any metric is stored, and only bitstreams are kept.
"""

from __future__ import annotations

from ..config import Paths
from ..encoders import EncoderAdapter, resolve_encoder
from ..results import ResultsDb
from ..sweep import Sweep
from .config import MeasuredSet


def encoders_of(measured: MeasuredSet) -> list[EncoderAdapter]:
    resolved = {name: resolve_encoder(name) for name in measured.encoders}
    unknown = [name for name, encoder in resolved.items() if encoder is None]
    if unknown:
        raise ValueError(f"set {measured.name}: unknown encoder(s) {unknown}")
    return list(resolved.values())


def measure_set(measured: MeasuredSet, paths: Paths, db: ResultsDb, harness_commit: str) -> str:
    """Sweep one set clip by clip; returns the per-clip outcome summaries.

    Clip-major order: every encoder of a clip finishes before the next clip
    starts, so a run cut short by the time budget still leaves complete
    cross-codec comparisons for the clips it reached.
    """
    sweep, encoders, outcomes = Sweep(paths, db, harness_commit), encoders_of(measured), []
    for clip in measured.load_clips(paths.data_root).clips:
        summary = sweep.run(encoders, [clip], measured.sweep)
        print(f"   {measured.name}/{clip.name}: {summary}", flush=True)
        outcomes.append(f"{clip.name}: {summary}")
    return "; ".join(outcomes)
