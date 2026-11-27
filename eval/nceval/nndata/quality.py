"""Label quality between two codings of the same job: determinism and timing noise.

- `determinism`: every column except the wall times must be identical (bit for bit,
  NaN in the same places) between two runs of one job; labels that move between
  identical runs cannot be learned.
- `timing_noise`: the spread of wall times between the runs, per coded candidate and
  per superblock: the median and 90th percentile of |log2(ns_a / ns_b)|.
- `label_shift`: for a job run on two builds (one lever toggled), how far the labels move:
  the share of nodes whose chosen slot changes, and the median relative change of the
  costs both builds coded (the label-stability set).
"""

from __future__ import annotations

import numpy as np

from .shard import Shard

TIME_COLUMNS = {("sb", "ns"), ("sb", "tool_ns"), ("node", "ns")}
#: The deterministic work columns, which a repeat must reproduce exactly (checked with the
#: other label columns).
WORK_COLUMNS = {("sb", "tool_work"), ("node", "cand_work")}
#: Quantiles reported for timing noise.
QUANTILES = (0.5, 0.9)


def determinism(a: Shard, b: Shard) -> list[str]:
    problems = []
    for table, columns in a.tables.items():
        for name, array in columns.items():
            if (table, name) in TIME_COLUMNS:
                continue
            other = b[table][name]
            if array.shape != other.shape or \
                    not np.array_equal(array, other, equal_nan=array.dtype.kind == "f"):
                problems.append(f"{table}.{name} differs between runs")
    return problems


def _log_ratio(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    both = (x > 0) & (y > 0)
    return np.abs(np.log2(x[both].astype(float) / y[both].astype(float)))


def timing_noise(a: Shard, b: Shard) -> dict:
    """|log2| time ratios between the runs at QUANTILES, per candidate and superblock."""
    out = {}
    for table in ("node", "sb"):
        ratios = _log_ratio(a[table]["ns"].ravel(), b[table]["ns"].ravel())
        out[table] = {f"q{int(q * 100)}": float(np.quantile(ratios, q)) if ratios.size else None
                      for q in QUANTILES}
    return out


def label_shift(a: Shard, b: Shard) -> dict:
    """Nodes matched by (frame, x, y, log2, parent position): chosen-slot changes and
    relative cost changes of candidates both coded."""
    def keyed(shard: Shard) -> dict:
        node = shard["node"]
        keys = zip(node["frame"].tolist(), node["x"].tolist(), node["y"].tolist(),
                   node["log2"].tolist())
        return {k: i for i, k in enumerate(keys)}
    ka, kb = keyed(a), keyed(b)
    common = sorted(set(ka) & set(kb))
    ia = np.array([ka[k] for k in common], int)
    ib = np.array([kb[k] for k in common], int)
    if not common:
        return {"common_nodes": 0}
    ca, cb = a["node"]["cost"][ia], b["node"]["cost"][ib]
    both = np.isfinite(ca) & np.isfinite(cb)
    relative = np.abs(ca[both] - cb[both]) / np.maximum(np.abs(ca[both]), 1e-9)
    return {"common_nodes": len(common), "nodes_a": len(ka), "nodes_b": len(kb),
            "chosen_changed": float((a["node"]["chosen"][ia] != b["node"]["chosen"][ib]).mean()),
            "cost_rel_change_median": float(np.median(relative)) if relative.size else None,
            "cost_rel_change_q90": float(np.quantile(relative, 0.9)) if relative.size else None}


def summarise(report: dict) -> dict:
    """The quality report in a few numbers: repeats with any column differing, the median
    of the per-job timing-noise quantiles, and the label shift's medians."""
    def median(values: list) -> float | None:
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None
    repeats, shifts = report["repeats"], report["shifts"]
    return {
        "repeats": len(repeats),
        "repeats_not_identical": sum(bool(r["determinism"]) for r in repeats),
        "timing_noise_log2": {table: {q: median([r["timing_noise"][table][q] for r in repeats])
                                      for q in (f"q{int(x * 100)}" for x in QUANTILES)}
                              for table in ("node", "sb")},
        "shifted_jobs": len(shifts),
        "chosen_changed_median": median([s.get("chosen_changed") for s in shifts]),
        "cost_rel_change_median": median([s.get("cost_rel_change_median") for s in shifts]),
        "cost_rel_change_q90_median": median([s.get("cost_rel_change_q90") for s in shifts]),
    }
