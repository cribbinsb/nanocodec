"""The independent test's analysis: per-metric BD with bootstrap intervals, and the audit.

Pure functions over scored real-use points (nceval.cbr result dicts, one per
CBR encode). tools/dev/aws_heldout_chart.py gathers the points of a run
(`independent`, `audit-chart`) and writes the tables; docs/results/
independent-*.md and rate-control-audit-*.md are their pages.

BD, as the chart (nceval.chart) defines it: per clip, BD-rate of a test curve
against the anchor's (x264 medium) on ACHIEVED rate (the stream's bytes as
ffprobe's packets report them), over the full overlap and over its lower-rate
half, averaged; per metric (PSNR-Y, PSNR-YUV 6:1:1, VMAF v1, MS-SSIM in dB)
and BD-J (the mean of PSNR-YUV, VMAF v1 and MS-SSIM). A family's value is
the class-balanced mean over the clips where the BD exists (a clip without
enough curve overlap is left out of that family, not chained as the chart
does), with a percentile bootstrap interval that resamples clips within each
class (BOOTSTRAP_DRAWS draws, fixed seed).

The audit's per-point measures (rate, buffer, frames, tails) read the same
result dicts; nothing here encodes or scores.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from . import jreal
from .cbr import BUFFER_MS
from .cbr_stats import replay_buffer
from .realuse import RATE_TOLERANCE, point_terms, rate_valid

#: The metrics reported separately, and those BD-J averages (jreal.HEADLINE_METRICS).
METRICS = ("psnr_y", "psnr_yuv", "vmaf_v1", "ms_ssim")
J = "j"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261113
INTERVAL = (2.5, 97.5)
#: The strict buffer rule applied to every encoder alike: the 1 s leaky bucket
#: (drained at the target, starting empty) never above its capacity, and the
#: raw average rate (no loan excluded) at most STRICT_OVERSHOOT over target.
STRICT_FULLNESS = 1.0
STRICT_OVERSHOOT = 0.05


def terms(result: dict) -> dict:
    """One point as the BD and audit see it: point_terms plus PSNR-Y (the mean of
    per-frame PSNR-Y, as PSNR-YUV is the mean of per-frame planes) and the raw
    per-frame series."""
    out = point_terms(result, len(result["frame_bits"]) / fps_of(result))
    out["psnr_y"] = statistics.fmean(result["frame_psnr_y"])
    out["result"] = result
    return out


def fps_of(result: dict) -> float:
    """The clip's frame rate, from the point's own bits and rate (cbr.Bench:
    kbps = bits x fps / frames / 1000)."""
    return result["kbps"] * 1000 * len(result["frame_bits"]) / sum(result["frame_bits"])


def clip_bd(anchor: list[dict], test: list[dict]) -> dict[str, float | None]:
    """Per metric and J: (full-overlap BD + lower-half BD) / 2, None when either is."""
    row: dict[str, float | None] = {}
    for metric in METRICS:
        full = jreal.curve_bd(anchor, test, metric)
        low = jreal.curve_bd(anchor, test, metric, low_half=True)
        row[metric] = None if full is None or low is None else (full + low) / 2
    headline = [row[m] for m in jreal.HEADLINE_METRICS]
    row[J] = None if None in headline else sum(headline) / len(headline)
    return row


def balanced_mean(values: dict[str, float], classes: dict[str, str]) -> float | None:
    """Mean over classes of each class's mean (jreal.aggregate's balance)."""
    by_class: dict[str, list[float]] = {}
    for clip, value in values.items():
        by_class.setdefault(classes[clip], []).append(value)
    if not by_class:
        return None
    return statistics.fmean(statistics.fmean(v) for v in by_class.values())


@dataclass(frozen=True)
class Estimate:
    """A class-balanced mean with its bootstrap interval and how many clips it has."""
    value: float | None
    low: float | None
    high: float | None
    clips: int

    def text(self) -> str:
        if self.value is None:
            return "n/a"
        return f"{self.value:+.1f} [{self.low:+.1f}, {self.high:+.1f}]"


def bootstrap(values: dict[str, float], classes: dict[str, str],
              draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> Estimate:
    """The class-balanced mean and its percentile interval, clips resampled within
    their class (the class weights stay fixed, as in the estimate)."""
    value = balanced_mean(values, classes)
    if value is None:
        return Estimate(None, None, None, 0)
    by_class: dict[str, list[float]] = {}
    for clip, v in sorted(values.items()):
        by_class.setdefault(classes[clip], []).append(v)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(statistics.fmean(rng.choices(v, k=len(v)))
                                    for v in by_class.values()) for _ in range(draws))
    pick = lambda q: means[min(len(means) - 1, round(q / 100 * (len(means) - 1)))]  # noqa
    return Estimate(value, pick(INTERVAL[0]), pick(INTERVAL[1]), len(values))


def family_bd(curves: dict, anchor: str, family: str, clips: Iterable[str],
              keep: Callable[[dict], bool] = lambda p: True) -> dict[str, dict]:
    """Per clip, `family` vs `anchor` (clip_bd) on the points `keep` admits."""
    rows = {}
    for clip in clips:
        a = [p for p in curves.get((anchor, clip), []) if keep(p)]
        t = [p for p in curves.get((family, clip), []) if keep(p)]
        rows[clip] = clip_bd(a, t) if a and t else {m: None for m in (*METRICS, J)}
    return rows


def summarise(rows: dict[str, dict], classes: dict[str, str]) -> dict[str, Estimate]:
    """Per metric and J, the bootstrap estimate over the clips that have it."""
    return {m: bootstrap({c: r[m] for c, r in rows.items() if r[m] is not None}, classes)
            for m in (*METRICS, J)}


# --- the audit's per-point measures -------------------------------------------------------

def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("mean", "median", "p10", "p90", "min", "max")}
    ordered = sorted(values)
    at = lambda q: ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]  # noqa
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p10": at(0.1), "p90": at(0.9), "min": ordered[0], "max": ordered[-1]}


def rate_audit(points: list[dict]) -> dict:
    """Actual / target rate and buffer behaviour of CBR points (one family).

    `ratio` is achieved bytes over the target; `loan_ratio` the same with our
    controller's outstanding key loan excluded (ours only; the chart's
    rate-valid rule uses it). The buffer is cbr_stats.replay_buffer: 1 s at the
    target, starting empty."""
    ratio = [p["kbps"] / p["target_kbps"] for p in points]
    loan = [1 + p["result"].get("loan_adjusted_rate_error", p["kbps"] / p["target_kbps"] - 1)
            for p in points]
    fullness = [p["max_fullness"] for p in points]
    fps = [fps_of(p["result"]) for p in points]
    window = [len(p["result"]["frame_bits"]) / f for p, f in zip(points, fps)]
    valid = [rate_valid(p["result"], w) for p, w in zip(points, window)]
    allowance = [RATE_TOLERANCE + BUFFER_MS / 1000 / w for w in window]
    over = [max(0.0, r - 1) for r in loan]
    relies = [v and (f > STRICT_FULLNESS or o > RATE_TOLERANCE)
              for v, f, o in zip(valid, fullness, over)]
    return {"points": len(points), "ratio": quantiles(ratio), "loan_ratio": quantiles(loan),
            "over_10pct": sum(r > 1 + RATE_TOLERANCE for r in ratio),
            "under_10pct": sum(r < 1 - RATE_TOLERANCE for r in ratio),
            "fullness": quantiles(fullness),
            "over_buffer": sum(f > STRICT_FULLNESS for f in fullness),
            "over_1_5": sum(f > jreal.NONCOMPLIANT_FULLNESS for f in fullness),
            "violation_seconds": quantiles([p["overruns"] / f for p, f in zip(points, fps)]),
            "first_frame_buffers": quantiles([p["result"]["frame_bits"][0] / (
                p["target_kbps"] * BUFFER_MS) for p in points]),
            "rate_valid": sum(valid), "relies_on_allowance": sum(relies),
            "allowance": quantiles(allowance), "strict_valid": sum(strict_ok(p) for p in points)}


def strict_ok(point: dict) -> bool:
    """The strict rule for every encoder alike (STRICT_FULLNESS, STRICT_OVERSHOOT)."""
    return point["max_fullness"] <= STRICT_FULLNESS and \
        point["kbps"] <= point["target_kbps"] * (1 + STRICT_OVERSHOOT)


def frames_audit(points: list[dict]) -> dict:
    """Input frames, coded frames (non-empty packets; ours: held frames carry 0 bits)
    and key frames per stream."""
    frames = [len(p["result"]["frame_bits"]) for p in points]
    coded = [sum(1 for b in p["result"]["frame_bits"] if b > 0) for p in points]
    keys = [sum(p["result"]["frame_key"]) for p in points]
    held = [f - c for f, c in zip(frames, coded)]
    return {"streams": len(points), "input_frames": sum(frames), "coded_frames": sum(coded),
            "held_frames": sum(held), "streams_with_held": sum(h > 0 for h in held),
            "max_held": max(held, default=0), "keys": quantiles(keys)}


def tails_audit(points: list[dict]) -> dict:
    """Worst-frame quality, the first key's pulse and its share of the bits, and the
    spread of per-frame quality, per stream."""
    def per(p: dict, key: str) -> list[float]:
        return p["result"][key]
    return {
        "worst_frame_psnr_yuv": quantiles([min(per(p, "frame_psnr_yuv")) for p in points]),
        "worst_1pct_psnr_yuv": quantiles([p["result"]["worst_psnr_yuv"] for p in points]),
        "worst_frame_vmaf": quantiles([min(per(p, "frame_vmaf_v1")) for p in points]),
        "worst_1pct_vmaf": quantiles([p["result"]["worst_vmaf_v1"] for p in points]),
        "first_key_pulse_psnr_y": quantiles([p["first_key_pulse_psnr_y"] for p in points]),
        "first_frame_bit_share": quantiles([per(p, "frame_bits")[0] / sum(per(p, "frame_bits"))
                                            for p in points]),
        "psnr_y_stdev": quantiles([statistics.pstdev(per(p, "frame_psnr_y")) for p in points]),
    }


def matched_tails(anchor: list[dict], test: list[dict]) -> dict[str, float | None]:
    """Test minus anchor, at matched achieved rate (jreal's interpolation over the
    overlap): BD-rate on the worst-window curves, and the first-key pulse and
    per-frame PSNR-Y spread deltas (dB)."""
    terms_ = jreal.compare_clip(anchor, test)
    spread = jreal.matched_delta(anchor, test,
                                 lambda p: statistics.pstdev(p["result"]["frame_psnr_y"]))
    return {"bd_worst_psnr_y": terms_["bd_worst_psnr_y"],
            "bd_worst_vmaf_v1": terms_["bd_worst_vmaf_v1"],
            "d_first_pulse_psnr_y": terms_["d_first_pulse_psnr_y"],
            "d_psnr_y_stdev": spread}


def replay(points: list[dict], buffer_ms: float) -> list:
    """The leaky-bucket replay of each point at another buffer size."""
    return [replay_buffer(p["result"]["frame_bits"], p["target_kbps"] * 1000.0,
                          fps_of(p["result"]), buffer_ms) for p in points]
