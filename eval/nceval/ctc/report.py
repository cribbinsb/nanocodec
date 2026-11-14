"""BD-rate tables for the standards lens, in the CTCs' own form, from the worker's results.

    python3 -m nceval.ctc report --results DIR[,DIR...] --out DIR

Per (condition, anchor, test encoder, sequence) and metric, a BD-rate by
`nceval.bdrate` (log rate, PCHIP of log rate over quality, overlap only:
the AOM CTC v8 §5.5 method). Metrics:

- AOM form: overall PSNR Y, U, V (MSE averaged over the frames, libvmaf's
  `apsnr_*`), BD-RateWeighted = 0.92 Y + 0.04 U + 0.04 V of those BD-rates,
  APSNR-YUV (MSE weighted 2/3, 1/6, 1/6), SSIM, MS-SSIM (dB), VMAF and
  VMAF-NEG (v0.6.1), CIEDE2000, PSNR-HVS. CAMBI is not BD-rated (§2.2.10).
- JVET form: frame-averaged PSNR Y, U, V (what HM prints) and EncT, the
  geometric mean over sequences of the encode user-time ratio (times summed
  over a sequence's points). The CTC figure uses QP 22-37 for QP-scale
  encoders; the extension figure all six points.
- Our J for cross-reference (docs/PROCEDURES.md): per sequence the mean of
  the full-overlap and lower-half BD-rates, each the mean over PSNR-YUV
  (6:1:1 of the frame-averaged planes), VMAF v1 and MS-SSIM (dB).

A class figure is the equal-weight mean over its sequences, the overall
figure the mean over all sequences (§5.1), both over sequences with a
defined BD-rate; `n` says how many. A BD-rate whose overlap covers less than
half of either curve (`nceval.bdrate.MIN_OVERLAP_FRACTION`) is kept in the
CSV with reliable=0 and left out of the means.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from ..bdrate import bd_rate
from .conditions import CONDITIONS, JVET_CTC_QPS, Condition, Scale
from .sequences import HEADLINE_TIERS, Tier, by_name

PEAK_SQUARED = 255.0 ** 2
#: AOM CTC v8 §5.5 and §2.2.3 weights.
BD_WEIGHTS = {"Y": 0.92, "U": 0.04, "V": 0.04}
APSNR_WEIGHTS = (2 / 3, 1 / 6, 1 / 6)
#: The project's PSNR-YUV weights (eval/nceval/metrics.py weighted_yuv).
PSNR_YUV_WEIGHTS = (6 / 8, 1 / 8, 1 / 8)
AOM_METRICS = ("Y", "U", "V", "APSNR-YUV", "SSIM", "MS-SSIM", "VMAF", "VMAF-NEG", "CIEDE2000",
               "PSNR-HVS")
JVET_METRICS = ("Y-fa", "U-fa", "V-fa")
J_METRICS = ("PSNR-YUV", "VMAF-v1", "MS-SSIM")
MIN_CURVE_POINTS = 4


@dataclass(frozen=True)
class Point:
    condition: str
    sequence: str
    encoder: str
    kind: str      # "qp" or "cbr"
    q: int         # the quantiser point (the CBR twin's, for kind "cbr")
    kbps: float
    user_s: float
    quality: dict[str, float]


def _mse(psnr: float) -> float:
    return PEAK_SQUARED / 10.0 ** (psnr / 10.0)


def _psnr(mse: float) -> float:
    return 10.0 * math.log10(PEAK_SQUARED / mse)


def quality_of(record: dict) -> dict[str, float]:
    means, overall = record["means"], record["overall_psnr"]
    y, u, v = overall["apsnr_y"], overall["apsnr_cb"], overall["apsnr_cr"]
    yf, uf, vf = means["psnr_y"], means["psnr_cb"], means["psnr_cr"]
    return {"Y": y, "U": u, "V": v,
            "APSNR-YUV": _psnr(sum(w * _mse(p) for w, p in zip(APSNR_WEIGHTS, (y, u, v)))),
            "SSIM": means["float_ssim"], "MS-SSIM": means["float_ms_ssim"],
            "VMAF": means["vmaf"], "VMAF-NEG": means["vmaf_neg"],
            "CIEDE2000": means["ciede2000"], "PSNR-HVS": means["psnr_hvs"],
            "CAMBI": means["cambi"],
            "Y-fa": yf, "U-fa": uf, "V-fa": vf,
            "PSNR-YUV": sum(w * p for w, p in zip(PSNR_YUV_WEIGHTS, (yf, uf, vf))),
            "VMAF-v1": means["vmaf_v1"]}


def load_points(results: list[Path]) -> list[Point]:
    """Every result record under the given directories (one per launch)."""
    points = []
    for path in sorted(p for directory in results for p in directory.rglob("*.json")):
        if path.name.endswith(".vmaf.json"):
            continue
        record = json.loads(path.read_text())
        condition, sequence, encoder, label = record["id"].split("/")
        kind = "cbr" if label.startswith("cbr") else "qp"
        points.append(Point(condition, sequence, encoder, kind, int(label.lstrip("cbrq")),
                            record["kbps"], record["user_s"], quality_of(record)))
    return points


def _curve(points: list[Point], metric: str) -> tuple[list[float], list[float]]:
    ordered = sorted(points, key=lambda p: p.kbps)
    return [p.kbps for p in ordered], [p.quality[metric] for p in ordered]


@dataclass(frozen=True)
class Bd:
    value: float | None
    overlap: float
    reliable: bool


def bd(anchor: list[Point], test: list[Point], metric: str, low_half: bool = False) -> Bd:
    if len(anchor) < MIN_CURVE_POINTS or len(test) < MIN_CURVE_POINTS:
        return Bd(None, 0.0, False)
    try:
        result = bd_rate(*_curve(anchor, metric), *_curve(test, metric), low_half=low_half)
    except ValueError:  # a non-monotone curve (saturated metric): no BD-rate
        return Bd(None, 0.0, False)
    return Bd(result.bd_rate_percent, result.overlap_fraction, result.reliable)


def bd_j(anchor: list[Point], test: list[Point]) -> Bd:
    """The project's BD-J on one sequence (module doc)."""
    parts = [bd(anchor, test, m, low) for m in J_METRICS for low in (False, True)]
    if any(p.value is None for p in parts):
        return Bd(None, 0.0, False)
    full = mean(p.value for p in parts[0::2])
    low = mean(p.value for p in parts[1::2])
    return Bd(0.5 * (full + low), min(p.overlap for p in parts), all(p.reliable for p in parts))


class Curves:
    """Points grouped by (condition, sequence, encoder, kind)."""

    def __init__(self, points: list[Point]):
        self.groups: dict[tuple, list[Point]] = defaultdict(list)
        for p in points:
            self.groups[(p.condition, p.sequence, p.encoder, p.kind)].append(p)

    def get(self, condition: str, sequence: str, encoder: str, kind: str = "qp",
            ctc_only: bool = False) -> list[Point]:
        points = self.groups.get((condition, sequence, encoder, kind), [])
        if ctc_only and CONDITIONS[condition].encoder(encoder).scale == Scale.QP:
            points = [p for p in points if p.q in JVET_CTC_QPS]
        return points


def bd_rows(curves: Curves, condition: Condition) -> list[dict]:
    """Every BD-rate of the condition: per anchor, test (and kind), sequence, metric."""
    rows = []
    for anchor in condition.anchors:
        for test in condition.encoders:
            if test.name == anchor:
                continue
            for kind in ("qp", "cbr") if test.own and condition.cbr_extra else ("qp",):
                for sequence in condition.sequences:
                    rows += _sequence_rows(curves, condition, anchor, test.name, kind, sequence.name)
    return rows


def _sequence_rows(curves: Curves, condition: Condition, anchor: str, test: str, kind: str,
                   sequence: str) -> list[dict]:
    base = {"condition": condition.name, "anchor": anchor, "test": test, "kind": kind,
            "sequence": sequence, "class": by_name(sequence).sequence_class,
            "tier": by_name(sequence).licence.tier.value}
    a = curves.get(condition.name, sequence, anchor)
    t = curves.get(condition.name, sequence, test, kind)
    if not a or not t:  # not run (a narrowed or partial run)
        return []
    results = {m: bd(a, t, m) for m in AOM_METRICS}
    results.update({f"{m}@ctc": bd(curves.get(condition.name, sequence, anchor, ctc_only=True),
                                   curves.get(condition.name, sequence, test, kind,
                                              ctc_only=True), m)
                    for m in JVET_METRICS})
    results.update({m: bd(a, t, m) for m in JVET_METRICS})
    results["J"] = bd_j(a, t)
    rows = [{**base, "metric": m, "bd": r.value, "overlap": r.overlap, "reliable": int(r.reliable)}
            for m, r in results.items()]
    planes = [results[p] for p in BD_WEIGHTS]
    weighted = (sum(w * r.value for w, r in zip(BD_WEIGHTS.values(), planes))
                if all(r.value is not None for r in planes) else None)
    rows.append({**base, "metric": "Weighted", "bd": weighted,
                 "overlap": min(r.overlap for r in planes),
                 "reliable": int(all(r.reliable for r in planes))})
    return rows


def enc_time_ratio(curves: Curves, condition: Condition, anchor: str, test: str,
                   sequence_names: list[str], ctc_only: bool = False) -> float | None:
    """Geometric mean over sequences of the summed encode user-time ratio."""
    ratios = []
    for name in sequence_names:
        a = curves.get(condition.name, name, anchor, ctc_only=ctc_only)
        t = curves.get(condition.name, name, test, ctc_only=ctc_only)
        if a and t:
            ratios.append(sum(p.user_s for p in t) / sum(p.user_s for p in a))
    return math.exp(mean(math.log(r) for r in ratios)) if ratios else None


def summarise(rows: list[dict], tiers: tuple[Tier, ...] = HEADLINE_TIERS) -> dict:
    """Mean reliable BD-rate per (condition, anchor, test, kind, metric) and class / overall."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["bd"] is None or not r["reliable"] or Tier(r["tier"]) not in tiers:
            continue
        key = (r["condition"], r["anchor"], r["test"], r["kind"], r["metric"])
        buckets[(*key, r["class"])].append(r["bd"])
        buckets[(*key, "overall")].append(r["bd"])
    return {k: (mean(v), min(v), max(v), len(v)) for k, v in buckets.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_rows(points: list[Point]) -> list[dict]:
    return [{"condition": p.condition, "sequence": p.sequence, "encoder": p.encoder,
             "kind": p.kind, "q": p.q, "kbps": round(p.kbps, 3), "user_s": round(p.user_s, 3),
             **{m: round(v, 4) for m, v in p.quality.items()}} for p in points]


def report(results: list[Path], out: Path) -> dict:
    """Write points.csv and bd.csv under `out`; return the summaries for the page."""
    points = load_points(results)
    curves = Curves(points)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "points.csv", point_rows(points))
    rows = [r for c in CONDITIONS.values() for r in bd_rows(curves, c)]
    write_csv(out / "bd.csv", [{**r, "bd": None if r["bd"] is None else round(r["bd"], 3),
                                "overlap": round(r["overlap"], 3)} for r in rows])
    return {"points": points, "curves": curves, "rows": rows,
            "all": summarise(rows), "free": summarise(rows, (Tier.FREE,))}
