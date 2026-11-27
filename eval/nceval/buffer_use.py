"""How full each encoder's 1 s buffer gets on the published CBR charts, from stored points.

The results site's buffer page (web/results/buffer.html, nceval.site.buffer_data) shows
this report. Every CBR chart point the site's chart views read (nceval.site.CHART_VIEWS,
each family on its recorded build) is read from the point cache through the chart's own
curve loader; nothing is encoded or scored. Per encoder (our presets pooled per codec):

- the achieved rate over the target;
- the bucket's peak in buffers measured AT THE TARGET (cbr_stats.replay_buffer, the
  harness's buffer), over the points within RATE_FLOOR of their target: a point further
  over is a rate floor (the encoder cannot go that low), which a target-rate bucket reads
  as many buffers and which says nothing about buffer control; their share is reported;
- the peak AT THE STREAM'S OWN RATE (cbr_stats.own_rate_fullness), over every point: the
  rate the point is plotted at, so an encoder that undershoots gets no larger buffer;
- the share of points (in percent) over each of LIMITS buffers, both ways;
- the first key frame's size in buffers at the target and the worst one-second window
  over the target (realuse.key_buffer_terms);
- `intervals`: each of those figures (FACTS) with its 95 % bootstrap interval, the
  clips resampled with replacement (a clip's points stay together: they share a source
  and are not independent), so every encoder's figures carry their uncertainty
  (results audit 2026-11-23, finding 11).

Every bit is counted (our key loan included). A point missing from the store is left out
and counted.

    python3 -m nceval.buffer_use --out ../docs/results/cbr-buffer-use-2026-11-22.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from . import chart, site
from .config import load_paths
from .gatepolicy import EVALUATION_GATE
from .realuse import DEFAULT_WORKERS, Runner

#: Buffer levels whose excess is counted: one buffer, the 0.6.1 re-code ceiling, and the
#: harness's non-compliance level (jreal.NONCOMPLIANT_FULLNESS).
LIMITS = (1.0, 1.15, 1.5)
#: Achieved / target above which a point is a rate floor (realuse.RATE_TOLERANCE).
RATE_FLOOR = 1.10
PERCENTILES = (5, 95)
#: Bootstrap resamples of the clips for `intervals`, and its seed (reproducible).
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20261123
INTERVAL = (2.5, 97.5)


def within_floor(points: list[dict]) -> list[dict]:
    """The points a target-rate bucket judges: within RATE_FLOOR of their target."""
    return [p for p in points if p["kbps"] / p["target_kbps"] <= RATE_FLOOR]


def values_of(field: str, points: list[dict]) -> np.ndarray:
    chosen = within_floor(points) if field == "max_fullness" else points
    return np.array([p[field] for p in chosen])


def share_over(field: str, limit: float):
    return lambda points: percent(values_of(field, points) > limit)


def percentile_of(field: str, q: float):
    return lambda points: float(np.percentile(values_of(field, points), q))


#: Each encoder's buffer facts: (name, statistic over a list of points), as `row`
#: computes them: at each stream's own rate over every point (own_*), at the target over
#: the points within RATE_FLOOR (target_*), and the key and one-second-window sizes.
FACTS = (
    *((f"own_over_{limit:g}", share_over("own_rate_fullness", limit)) for limit in LIMITS),
    *((f"target_over_{limit:g}", share_over("max_fullness", limit)) for limit in LIMITS),
    ("own_p95", percentile_of("own_rate_fullness", 95)),
    ("target_p95", percentile_of("max_fullness", 95)),
    ("key_buffers_median", percentile_of("key_buffers", 50)),
    ("key_buffers_p90", percentile_of("key_buffers", 90)),
    ("window_ratio_median", percentile_of("max_window_ratio", 50)),
    ("window_ratio_p90", percentile_of("max_window_ratio", 90)),
)


def values_of_any(fact: str, points: list[dict]) -> bool:
    return bool(within_floor(points)) if fact.startswith("target_") else bool(points)


def intervals(by_clip: dict[str, list[dict]]) -> dict[str, list[float]]:
    """{fact: [value, low, high]}: FACTS over every point, with the 95 % interval of a
    clip bootstrap (module doc)."""
    clips = sorted(by_clip)
    pooled = [p for c in clips for p in by_clip[c]]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [[p for i in rng.integers(0, len(clips), len(clips)) for p in by_clip[clips[i]]]
             for _ in range(BOOTSTRAP_ROUNDS)]
    result = {}
    for name, statistic in FACTS:
        # a draw of only rate-floor points has no target figure: left out of the interval
        low, high = np.nanpercentile([statistic(d) if values_of_any(name, d) else np.nan
                                      for d in draws], INTERVAL)
        result[name] = [statistic(pooled), float(low), float(high)]
    return result
BUILDS_ROOT = Path.home() / "nanocodec-data" / "charts" / "builds"
SCORING_BINARY = "nanocodec"


def encoder_group(family: str) -> str:
    """The report's row: our families pooled per codec, each reference on its own."""
    if family.startswith("nanocodec-"):
        return family.split("@")[0]
    return family


def view_families(view: site.ChartView) -> tuple[list, list]:
    """(families with the view's zero line first, the default chart's families), each of
    ours on its recorded build, as chart_refresh renders the view."""
    record = json.loads((site.RESULTS / view.builds).read_text())
    base = BUILDS_ROOT / record["base"]
    binaries = {f: BUILDS_ROOT / b / SCORING_BINARY for f, b in record["families"].items()}
    presets = {f: BUILDS_ROOT / b / "presets" for f, b in record["families"].items()}
    judge = chart.chart_families(base / SCORING_BINARY, base / "presets", binaries=binaries,
                                 presets_for=presets)
    zero = chart.chart_families(base / SCORING_BINARY, base / "presets", [view.zero],
                                binaries, presets)[1:]
    return [*zero, *judge], judge


def percent(flags: np.ndarray) -> float:
    """The share of true flags, in percent (as the page shows it)."""
    return float(100 * np.mean(flags))


def shares(values: np.ndarray) -> dict[str, float]:
    return {f"{limit:g}": percent(values > limit) for limit in LIMITS}


def row(by_clip: dict[str, list[dict]]) -> dict:
    """One encoder's figures (module doc), from its points by clip."""
    points = [p for clip_points in by_clip.values() for p in clip_points]
    ratio = np.array([p["kbps"] / p["target_kbps"] for p in points])
    at_target = np.array([p["max_fullness"] for p, r in zip(points, ratio) if r <= RATE_FLOOR])
    own = np.array([p["own_rate_fullness"] for p in points])
    return {"points": len(points), "rate_floor_pct": percent(ratio > RATE_FLOOR),
            "ratio_median": float(np.median(ratio)),
            **{f"ratio_p{q}": float(np.percentile(ratio, q)) for q in PERCENTILES},
            "target_p95": float(np.percentile(at_target, 95)), "target_max": float(at_target.max()),
            "target_over": shares(at_target),
            "own_p95": float(np.percentile(own, 95)), "own_max": float(own.max()),
            "own_over": shares(own), "intervals": intervals(by_clip)}


def view_report(view: site.ChartView, workers: int) -> dict:
    families, judge = view_families(view)
    clips = chart.chart_clips(str(load_paths().clips / view.clips) if view.clips else None)
    runner = Runner(load_paths(), EVALUATION_GATE, workers, over_budget=True,
                    stored_ms_ssim=True, xpsnr=True, stored_xpsnr=True, cached_only=True,
                    skip_missing=True)
    curves, _ = chart.chart_curves(runner, families, clips, chart.DEFAULT_FRAMES,
                                   f"buffer use {view.id}", judge=judge)
    groups: dict[str, dict[str, list[dict]]] = {}
    for (family, clip), points in curves.items():
        if points:
            groups.setdefault(encoder_group(family), {}).setdefault(clip, []).extend(points)
    record = json.loads((site.RESULTS / view.builds).read_text())
    return {"view": view.id, "clipset": view.clipset, "clips": len(clips),
            "points": sum(len(p) for g in groups.values() for p in g.values()),
            "missing": runner.skipped,
            "builds": sorted(set(record["families"].values())),
            "rows": {name: row(points) for name, points in sorted(groups.items())}}


def report(workers: int) -> dict:
    views = [v for v in site.CHART_VIEWS if site.shown(v.clipset)]
    return {"date": datetime.date.today().isoformat(), "limits": list(LIMITS),
            "rate_floor_pct": round(100 * (RATE_FLOOR - 1)), "frames": chart.DEFAULT_FRAMES,
            "views": [view_report(v, workers) for v in views]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="the report's JSON")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    args.out.write_text(json.dumps(report(args.workers), indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
