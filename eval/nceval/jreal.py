"""J_real: the real-use objective, its temporal terms and robust aggregation.

Pure functions over rate-controlled curves (nceval.realuse measures them).
The definitions and thresholds are the ones in docs/PROCEDURES.md
("Adoption rule: J_real"); change both together.

Per point (one CBR encode), from per-frame quality and the key-frame flags:

  worst_psnr_y, worst_vmaf_v1   the lowest WORST_FRACTION of WORST_WINDOW-frame
                                window means, windows counted from each key as
                                for flicker (the guarded form; coordinator,
                                2026-11-10): the designed coarse frames of a
                                2- or 4-frame layer period (L1T2, L1T3) or
                                cadence are not visible bad frames at 30 fps,
                                a run of bad frames is (149 P frames: the
                                worst of 37 windows)
  worst_frame_psnr_y, worst_frame_vmaf_v1    the previous per-frame form,
                                reported alongside: mean of the lowest
                                WORST_FRACTION of P frames (the worst one)
  flicker_psnr_y                pumping: mean |difference| of the mean PSNR-Y of
                                consecutive FLICKER_WINDOW-frame windows (the
                                guarded flicker; coordinator, 2026-11-10). The
                                windows start after each key and end at a
                                period-8 cadence base frame, so the designed
                                2-8 frame QP sawtooth of a pseudo-layer cadence
                                (SVT rtc, libvpx use the same) averages out and
                                only slower quality swings count
  flicker_frame_psnr_y          the previous definition, reported alongside:
                                mean |PSNR-Y(i) - PSNR-Y(i-1)| over consecutive
                                frame pairs that contain no key frame
  first_key_pulse_psnr_y        the first key frame's PSNR-Y minus the median
                                P-frame PSNR-Y. Report only: with a 10 s key
                                period a 150-frame clip has only this key, and
                                a session's first key legitimately stands out
  periodic_key_pulse_psnr_y     mean over keys after the first of PSNR-Y(key)
                                minus the median of the PULSE_NEIGHBOURS P
                                frames before it (the jump a viewer sees);
                                None when the clip has no such key (the board
                                and screens measure it in a short-key-period
                                pass, realuse PERIODIC_KEY_SECONDS)

Per clip (test curve vs anchor curve, achieved kbps on the rate axis; every
ladder point counts, including targets an encoder overshot (its rate floor),
but a point with no more quality than a cheaper one of the same curve is
dropped from that metric's curve):

  bd_<m>, low_<m>      BD-rate on metric m over the full overlap and over its
                       lower-rate half (bdrate.bd_rate low_half), for PSNR-YUV,
                       VMAF v1 and MS-SSIM (as -10 log10(1 - x))
  bd_worst_psnr_y, bd_worst_vmaf_v1    BD-rate on the worst-1% curves
                       (bd_worst_frame_* on the per-frame form, report only)
  d_flicker_psnr_y     test minus anchor flicker (dB; d_flicker_frame_psnr_y
                       the frame-to-frame form, report only); the guard judges
                       d_flicker_guard_psnr_y, the same minus the clip's
                       headroom below the calibrated real-time reference
                       (nceval.flicker_reference; coordinator, 2026-11-10), and
  d_first_pulse_psnr_y, d_periodic_pulse_psnr_y   test minus anchor |first-key|
                       and |periodic-key| pulse (dB); all three deltas are
                       interpolated in log rate and averaged over the overlap
  quality, quality_low the mean BD-rate over PSNR-YUV, VMAF v1 and MS-SSIM
                       (HEADLINE_METRICS; full / low)
  base_quality         spatial SVC only (two-layer settings, nceval.svcbase):
                       `quality` of the base layer (operating point 1) at its
                       own rate. The headline terms of a two-layer stream are
                       its full-size output (operating point 0) at the total
                       rate; the base layer is a guard, so a change cannot buy
                       the full layer's quality with the base's

Buffer compliance is reported next to the BD terms, never inside them
(coordinator, 2026-11-08: openh264 wins static clips' first key by ignoring
the buffer). Each point carries `overruns` (frames after which the strict CBR
leaky bucket of cbr_stats.replay_buffer, drained at the target with the
shared 1 s buffer, held more than its capacity) and `max_fullness` (peak
occupancy / capacity). A point is NON-COMPLIANT when max_fullness exceeds
NONCOMPLIANT_FULLNESS. Per clip: `test_overruns` / `anchor_overruns` (frames,
summed over the curve) and `test_noncompliant` / `anchor_noncompliant`
(points); `compliance` totals them over clips and lists the clips whose
anchor curve is non-compliant, so losses there are flagged, not acted on.

Rate compliance in A/B screens (ours vs ours; coordinator, 2026-11-09): a
point that breaks the strict leaky bucket or overshoots its target by more
than 10% + buffer/window (realuse.rate_valid; our key loan excluded) is not a
valid real-use point, and on achieved-rate curves it would let that arm buy
quality with bits it was not allowed. `pair_rungs` drops every ladder rung
(target) where EITHER arm's point is invalid
from both arms' curves before any term is computed, and counts it as a
compliance win (only the anchor overshot) or loss (only the test overshot):
`rate_wins` / `rate_losses` per clip, totalled in `compliance`. A test arm
with more losses than wins is rejected (the rate guard); one with more wins,
J_real <= 0 and no guard is BETTER even when its BD change is within noise. Against external
anchors the rungs are not paired: an anchor's rate floor is a real point.

Across clips every term is reported as its class-balanced mean (each content
class weighted equally: the headline), plain mean, median clip, per-class
means, the count of clips worse than the anchor and the clips it is missing
for (listed, never silently dropped).

J_real = (quality + quality_low) / 2 (class-balanced) + K_CPU * dCPU.
Guards reject a change whatever J_real says: a class worse than
CLASS_GUARD_PCT on quality or quality_low, a worst-1% BD worse than
WORST_GUARD_PCT, flicker up more than FLICKER_GUARD_DB, |periodic-key pulse|
up more than PULSE_GUARD_DB (all class-balanced; the first-key pulse is
reported, never guarded), a two-layer stream's base layer worse than
SVC_BASE_GUARD_PCT, or more non-compliant points than the anchor (the
buffer guard).
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

import numpy as np

from .bdrate import BdResult, bd_rate

#: The metrics J_real averages, in equal thirds (Mark, 2026-11-11). VMAF v1
#: (3d0h) needs no NEG variant: its ADM already caps the enhancement gain
#: (adm_enhn_gain_limit 1.0) and it has no VIF.
HEADLINE_METRICS = ("psnr_yuv", "vmaf_v1", "ms_ssim")
WORST_METRICS = ("worst_psnr_y", "worst_vmaf_v1")
#: The per-frame worst-1% metrics, reported only.
WORST_FRAME_METRICS = ("worst_frame_psnr_y", "worst_frame_vmaf_v1")
WORST_FRACTION = 0.01
#: Quality per 1% CPU, as in the fixed-QP rule (Mark, 2026-11-08).
K_CPU = 0.6
CLASS_GUARD_PCT = 3.0
WORST_GUARD_PCT = 2.0
FLICKER_GUARD_DB = 0.05
#: Frames per flicker window: the deepest flat cadence period (8), so a
#: window holds whole cadence cycles (two of period 4).
FLICKER_WINDOW = 8
#: Frames per worst-1% window: the deepest temporal-layer period (L1T3, 4),
#: which holds whole L1T2 periods and period-4 cadence cycles too. One
#: window for every encoder, so A/B and anchor comparisons stay like for like.
WORST_WINDOW = 4
PULSE_GUARD_DB = 0.25
#: Spatial SVC's base-layer guard (BD-rate, percent): as the worst-1% guard,
#: the most a change may cost a secondary output it does not optimise.
SVC_BASE_GUARD_PCT = 2.0
#: Rate-controlled curves have 4 ladder points; a staged proxy keeps 3.
BD_MIN_POINTS = 3
#: A point whose leaky bucket peaks above this multiple of the buffer
#: violates it outright (a receiver with the shared 1 s buffer would stall
#: for over half a second), as opposed to brushing its edge.
NONCOMPLIANT_FULLNESS = 1.5
#: Log-spaced rates at which flicker/pulse deltas are compared.
MATCHED_RATE_SAMPLES = 5
MARGIN_SIGMAS = 2.0
#: P frames before a periodic key that define its neighbours' level.
PULSE_NEIGHBOURS = 8


def ms_ssim_db(value: float) -> float:
    """MS-SSIM linearised for BD-rate interpolation (it saturates near 1)."""
    return -10.0 * math.log10(max(1e-10, 1.0 - value))


BD_TRANSFORMS = {"ms_ssim": ms_ssim_db}


def worst_mean(values: Sequence[float], fraction: float = WORST_FRACTION) -> float:
    """Mean of the lowest `fraction` of values (at least one)."""
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * fraction))
    return sum(ordered[:count]) / count


def temporal_stats(psnr_y: Sequence[float], vmaf_v1: Sequence[float],
                   keys: Sequence[bool]) -> dict[str, float]:
    """Worst-1%, flicker and key pulse of one point (see the module docstring)."""
    if not len(psnr_y) == len(vmaf_v1) == len(keys):
        raise ValueError("per-frame series differ in length")
    steady = [i for i, key in enumerate(keys) if not key]
    if not steady:
        raise ValueError("no non-key frames")
    key_frames = [i for i, key in enumerate(keys) if key]
    pairs = [i for i in range(1, len(keys)) if not keys[i] and not keys[i - 1]]
    steady_psnr = statistics.median(psnr_y[i] for i in steady)
    return {
        "worst_psnr_y": worst_mean(window_means(psnr_y, keys, WORST_WINDOW)
                                   or [psnr_y[i] for i in steady]),
        "worst_vmaf_v1": worst_mean(window_means(vmaf_v1, keys, WORST_WINDOW)
                                    or [vmaf_v1[i] for i in steady]),
        "worst_frame_psnr_y": worst_mean([psnr_y[i] for i in steady]),
        "worst_frame_vmaf_v1": worst_mean([vmaf_v1[i] for i in steady]),
        "flicker_psnr_y": window_flicker(psnr_y, keys),
        "flicker_frame_psnr_y": (sum(abs(psnr_y[i] - psnr_y[i - 1]) for i in pairs) / len(pairs)
                                 if pairs else 0.0),
        "first_key_pulse_psnr_y": psnr_y[key_frames[0]] - steady_psnr if key_frames else 0.0,
        "periodic_key_pulse_psnr_y": periodic_pulse(psnr_y, keys, key_frames[1:]),
    }


def flicker_over_headroom(delta: float | None, anchor: list[dict],
                          reference: float | None) -> float | None:
    """The flicker delta the guard judges: `delta` minus the headroom between
    the anchor's mean window flicker and the reference's (never negative
    headroom; no reference, no headroom)."""
    if delta is None or reference is None or not anchor:
        return delta
    anchor_level = sum(p["flicker_psnr_y"] for p in anchor) / len(anchor)
    return delta - max(0.0, reference - anchor_level)


def key_runs(values: Sequence[float], keys: Sequence[bool], size: int) -> list[list[float]]:
    """Per run of P frames between keys, the means of its whole `size`-frame
    windows counted from the key (key frames and a trailing partial window
    left out)."""
    starts = ([] if keys[0] else [-1]) + [i for i, key in enumerate(keys) if key]
    runs = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(keys)
        runs.append([statistics.fmean(values[i:i + size])
                     for i in range(start + 1, end - size + 1, size)])
    return runs


def window_means(values: Sequence[float], keys: Sequence[bool], size: int) -> list[float]:
    """Every whole `size`-frame window mean of [`key_runs`]."""
    return [mean for run in key_runs(values, keys, size) for mean in run]


def window_flicker(psnr_y: Sequence[float], keys: Sequence[bool]) -> float:
    """Mean |change| of the mean PSNR-Y between consecutive whole
    FLICKER_WINDOW-frame windows, counted from each key; 0 when no two
    windows exist."""
    steps = [abs(b - a) for means in key_runs(psnr_y, keys, FLICKER_WINDOW)
             for a, b in zip(means, means[1:])]
    return sum(steps) / len(steps) if steps else 0.0


def periodic_pulse(psnr_y: Sequence[float], keys: Sequence[bool],
                   later_keys: list[int]) -> float | None:
    """Mean jump of each later key over the P frames just before it."""
    jumps = []
    for key in later_keys:
        before = [psnr_y[i] for i in range(max(0, key - PULSE_NEIGHBOURS), key) if not keys[i]]
        if before:
            jumps.append(psnr_y[key] - statistics.median(before))
    return sum(jumps) / len(jumps) if jumps else None


def curve_bd_result(anchor: list[dict], test: list[dict], metric: str, low_half: bool = False,
                    min_points: int = BD_MIN_POINTS) -> BdResult | None:
    """BD-rate of `metric` on achieved-rate curves with its overlap; None when unusable."""
    transform = BD_TRANSFORMS.get(metric, lambda x: x)

    def axes(points: list[dict]) -> tuple[list[float], list[float]]:
        usable = sorted((p for p in points if p.get(metric) is not None), key=lambda p: p["kbps"])
        rates, qualities = [], []
        for point in usable:  # dominated points (more bits, no more quality) say nothing
            quality = transform(point[metric])
            if not qualities or quality > qualities[-1]:
                rates.append(point["kbps"])
                qualities.append(quality)
        return rates, qualities
    a_rate, a_q = axes(anchor)
    t_rate, t_q = axes(test)
    try:
        return bd_rate(a_rate, a_q, t_rate, t_q, min_points=min_points, low_half=low_half)
    except ValueError:  # too few points, or quality not increasing with rate
        return None


def point_rate_delta(anchor: list[dict], point: dict, metric: str) -> float | None:
    """BD-rate's idea at one point: percent more rate `point` spends than the anchor's
    curve needs for the same quality (owner, 2026-11-11).

    The anchor's achieved-rate curve (dominated points dropped, as in BD) is
    interpolated linearly in log rate against quality, and extended along
    its end segments, to the rate R_a at which it reaches the point's
    quality; the delta is R_c / R_a - 1. A rate-control miss of the point
    is therefore priced, not dropped. None with fewer than two anchor points.
    """
    transform = BD_TRANSFORMS.get(metric, lambda x: x)
    rates, qualities = [], []
    for p in sorted(anchor, key=lambda p: p["kbps"]):
        quality = transform(p[metric])
        if not qualities or quality > qualities[-1]:
            rates.append(math.log(p["kbps"]))
            qualities.append(quality)
    if len(rates) < 2:
        return None
    target = transform(point[metric])
    segment = min(max(int(np.searchsorted(qualities, target)) - 1, 0), len(rates) - 2)
    q0, q1 = qualities[segment], qualities[segment + 1]
    log_rate = rates[segment] + (target - q0) * (rates[segment + 1] - rates[segment]) / (q1 - q0)
    return 100.0 * (point["kbps"] / math.exp(log_rate) - 1.0)


def point_quality(anchor: list[dict], point: dict) -> float | None:
    """The single-point J quality of one clip: the mean of the headline metrics'
    point_rate_delta, None if any is missing."""
    deltas = [point_rate_delta(anchor, point, metric) for metric in HEADLINE_METRICS]
    return None if None in deltas else sum(deltas) / len(deltas)


def curve_bd(anchor: list[dict], test: list[dict], metric: str, low_half: bool = False,
             min_points: int = BD_MIN_POINTS) -> float | None:
    """BD-rate of `metric` on achieved-rate curves; None when unusable or unreliable."""
    result = curve_bd_result(anchor, test, metric, low_half, min_points)
    return result.bd_rate_percent if result and result.reliable else None


def curve_overlap(anchor: list[dict], test: list[dict]) -> float:
    """The smallest overlap fraction (bdrate.BdResult) of the headline metrics' curves, 0
    when one is unusable: how well two encoders' curves can be compared on a clip."""
    results = [curve_bd_result(anchor, test, metric) for metric in HEADLINE_METRICS]
    return min((r.overlap_fraction if r else 0.0) for r in results)


def matched_delta(anchor: list[dict], test: list[dict], key, samples: int = MATCHED_RATE_SAMPLES
                  ) -> float | None:
    """Mean of test minus anchor `key(point)`, interpolated in log rate over the overlap."""
    def series(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        ordered = sorted(points, key=lambda p: p["kbps"])
        return (np.log([p["kbps"] for p in ordered]), np.array([key(p) for p in ordered]))
    if len(anchor) < 2 or len(test) < 2:
        return None
    a_rate, a_value = series(anchor)
    t_rate, t_value = series(test)
    lo, hi = max(a_rate[0], t_rate[0]), min(a_rate[-1], t_rate[-1])
    if hi <= lo:
        return None
    rates = np.linspace(lo, hi, samples)
    return float(np.mean(np.interp(rates, t_rate, t_value) - np.interp(rates, a_rate, a_value)))


def compare_clip(anchor: list[dict], test: list[dict], min_points: int = BD_MIN_POINTS,
                 reference_flicker: float | None = None) -> dict[str, float | None]:
    """Every J_real term of one clip (see the module docstring).

    `reference_flicker`: the clip's calibrated real-time reference window
    flicker (nceval.flicker_reference); the guarded flicker delta is reduced
    by how far the anchor lies below it."""
    terms: dict[str, float | None] = {}
    for metric in HEADLINE_METRICS:
        terms[f"bd_{metric}"] = curve_bd(anchor, test, metric, min_points=min_points)
        terms[f"low_{metric}"] = curve_bd(anchor, test, metric, True, min_points)
    for metric in WORST_METRICS:
        terms[f"bd_{metric}"] = curve_bd(anchor, test, metric, min_points=min_points)
    for metric in WORST_FRAME_METRICS:
        terms[f"bd_{metric}"] = (curve_bd(anchor, test, metric, min_points=min_points)
                                 if all(metric in p for p in anchor + test) else None)
    terms["d_flicker_psnr_y"] = matched_delta(anchor, test, lambda p: p["flicker_psnr_y"])
    terms["d_flicker_guard_psnr_y"] = flicker_over_headroom(
        terms["d_flicker_psnr_y"], anchor, reference_flicker)
    terms["d_flicker_frame_psnr_y"] = (
        matched_delta(anchor, test, lambda p: p["flicker_frame_psnr_y"])
        if all("flicker_frame_psnr_y" in p for p in anchor + test) else None)
    terms["d_first_pulse_psnr_y"] = matched_delta(anchor, test,
                                                  lambda p: abs(p["first_key_pulse_psnr_y"]))
    periodic = [p for p in anchor + test if p.get("periodic_key_pulse_psnr_y") is not None]
    terms["d_periodic_pulse_psnr_y"] = (
        matched_delta(anchor, test, lambda p: abs(p["periodic_key_pulse_psnr_y"]))
        if len(periodic) == len(anchor) + len(test) else None)
    terms.update(headline_quality(terms))
    terms["base_quality"] = base_quality(anchor, test, min_points)
    for side, points in (("test", test), ("anchor", anchor)):
        terms[f"{side}_overruns"] = sum(p.get("overruns", 0) for p in points)
        terms[f"{side}_noncompliant"] = sum(1 for p in points if noncompliant(p))
    return terms


def base_quality(anchor: list[dict], test: list[dict],
                 min_points: int = BD_MIN_POINTS) -> float | None:
    """Two-layer streams: the headline BD-rate of the base layers (operating point 1,
    each at its own rate; realuse.point_terms "base"), None unless both curves
    carry them."""
    if not anchor or not test or not all("base" in p for p in anchor + test):
        return None
    base_a, base_t = [p["base"] for p in anchor], [p["base"] for p in test]
    values = [curve_bd(base_a, base_t, m, min_points=min_points) for m in HEADLINE_METRICS]
    return None if None in values else sum(values) / len(values)


def point_base_quality(anchor: list[dict], point: dict) -> float | None:
    """The single-point form of `base_quality` (point_quality on the base layers)."""
    if not anchor or "base" not in point or not all("base" in p for p in anchor):
        return None
    return point_quality([p["base"] for p in anchor], point["base"])


def headline_quality(terms: dict[str, float | None]) -> dict[str, float | None]:
    """`quality` and `quality_low`: the mean of the headline metrics' full-overlap and
    low-rate BD terms (`bd_<metric>`, `low_<metric>`), None unless all are present."""
    quality = {}
    for name, prefix in (("quality", "bd_"), ("quality_low", "low_")):
        values = [terms[f"{prefix}{m}"] for m in HEADLINE_METRICS]
        quality[name] = None if None in values else sum(values) / len(values)
    return quality


def noncompliant(point: dict) -> bool:
    return point.get("max_fullness", 0.0) > NONCOMPLIANT_FULLNESS


def pair_rungs(anchor: list[dict], test: list[dict]) -> tuple[list[dict], list[dict], int, int]:
    """Both curves without the rungs either arm overshot, plus (wins, losses).

    Points carry `target_kbps` and `rate_valid`; a rung is kept only when
    every point of both arms at that target is rate-valid.
    """
    def invalid(points: list[dict]) -> set:
        return {p["target_kbps"] for p in points if not p["rate_valid"]}
    bad_anchor, bad_test = invalid(anchor), invalid(test)
    dropped = bad_anchor | bad_test
    keep = lambda points: [p for p in points if p["target_kbps"] not in dropped]
    return (keep(anchor), keep(test), len(bad_anchor - bad_test), len(bad_test - bad_anchor))


def compliance(per_clip: dict[str, dict]) -> dict:
    """Buffer and rate compliance totals of both sides; clips with a non-compliant anchor."""
    total = lambda key: sum(row.get(key, 0) for row in per_clip.values())
    return {"rate_wins": total("rate_wins"), "rate_losses": total("rate_losses"),
            "test_overruns": total("test_overruns"), "anchor_overruns": total("anchor_overruns"),
            "test_noncompliant": total("test_noncompliant"),
            "anchor_noncompliant": total("anchor_noncompliant"),
            "anchor_noncompliant_clips": sorted(c for c, row in per_clip.items()
                                                if row.get("anchor_noncompliant"))}


TERMS = ("quality", "quality_low",
         *(f"bd_{m}" for m in HEADLINE_METRICS), *(f"low_{m}" for m in HEADLINE_METRICS),
         *(f"bd_{m}" for m in WORST_METRICS + WORST_FRAME_METRICS),
         "d_flicker_psnr_y", "d_flicker_guard_psnr_y", "d_flicker_frame_psnr_y",
         "d_first_pulse_psnr_y", "d_periodic_pulse_psnr_y", "base_quality")


def aggregate(per_clip: dict[str, dict], term: str) -> dict:
    """Class-balanced mean, mean, median, per-class means, worse count, missing clips.

    `per_clip` maps clip -> {"class": ..., term: value}. The class-balanced
    mean is over the classes with at least one value; `missing` lists the
    clips without one, so a reader sees what each mean is over.
    """
    classes: dict[str, list[float]] = {}
    missing = []
    for clip, row in sorted(per_clip.items()):
        if row.get(term) is None:
            missing.append(clip)
        else:
            classes.setdefault(row["class"], []).append(row[term])
    values = [v for vs in classes.values() for v in vs]
    class_means = {name: sum(vs) / len(vs) for name, vs in sorted(classes.items())}
    return {
        "balanced": sum(class_means.values()) / len(class_means) if class_means else None,
        "mean": sum(values) / len(values) if values else None,
        "median": statistics.median(values) if values else None,
        "classes": class_means,
        "worse": sum(1 for v in values if v > 0),
        "scored": len(values),
        "missing": missing,
    }


def summarise(per_clip: dict[str, dict]) -> dict[str, dict]:
    summary = {term: aggregate(per_clip, term) for term in TERMS}
    summary["compliance"] = compliance(per_clip)
    return summary


def objective(summary: dict[str, dict], dcpu_pct: float | None = None,
              sigma: float = 0.0, k_cpu: float = K_CPU) -> dict:
    """J_real, its guards and the staged decision (better/worse/ambiguous/neutral).

    `k_cpu` weighs the CPU change: `K_CPU` (0.6) for Fast-preset and engine
    changes, 0.1 for Default quality arms (PROCEDURES "Preset roles").

    `sigma` is the stage's estimation error (percent; 0 for the whole set).
    A BD-like guard decides "worse" at a proxy stage only when it exceeds its
    threshold by the stage margin; flicker/pulse guards, in dB, decide only
    on the whole set (a proxy stage that trips one moves on).
    """
    full, low = summary["quality"]["balanced"], summary["quality_low"]["balanced"]
    quality = None if full is None or low is None else (full + low) / 2
    j = None if quality is None else quality + k_cpu * (dcpu_pct or 0.0)
    margin = MARGIN_SIGMAS * sigma
    guards, hard = [], []

    def check(name: str, value: float | None, limit: float, scaled: bool) -> None:
        if value is not None and value > limit:
            guards.append(f"{name} {value:+.2f} > {limit:+.2f}")
            if sigma == 0 or (scaled and value > limit + margin):
                hard.append(name)
    for term in ("quality", "quality_low"):
        for name, value in summary[term]["classes"].items():
            check(f"class {name} {term}", value, CLASS_GUARD_PCT, True)
    for metric in WORST_METRICS:
        check(f"bd_{metric}", summary[f"bd_{metric}"]["balanced"], WORST_GUARD_PCT, True)
    flicker = summary.get("d_flicker_guard_psnr_y", summary["d_flicker_psnr_y"])
    check("d_flicker_guard_psnr_y", flicker["balanced"], FLICKER_GUARD_DB, False)
    check("d_periodic_pulse_psnr_y", summary["d_periodic_pulse_psnr_y"]["balanced"],
          PULSE_GUARD_DB, False)
    if "base_quality" in summary:  # None (unchecked) unless both arms are two-layer
        check("svc base_quality", summary["base_quality"]["balanced"], SVC_BASE_GUARD_PCT, True)
    buffer = summary.get("compliance")
    if buffer and buffer.get("rate_losses", 0) > buffer.get("rate_wins", 0):
        guards.append(f"rate: overshoots {buffer['rate_losses']} rungs the anchor meets "
                      f"(> {buffer['rate_wins']} the other way)")
        hard.append("rate")  # a count, not an estimate: no stage margin
    if buffer and buffer["test_noncompliant"] > buffer["anchor_noncompliant"]:
        guards.append(f"buffer: {buffer['test_noncompliant']} non-compliant points > "
                      f"anchor's {buffer['anchor_noncompliant']}")
        hard.append("buffer")  # a count, not an estimate: no stage margin
    if j is None:
        decision = "ambiguous"
    elif hard or j - margin > 0:
        decision = "worse"
    elif j + margin < 0 and not guards:
        decision = "better"
    elif j <= 0 and not guards and buffer and buffer.get("rate_wins", 0) > buffer.get(
            "rate_losses", 0):
        # Meets targets the anchor overshot at no measured quality cost: a
        # strict real-use improvement even when the BD change is within noise.
        decision = "better"
    else:
        decision = "ambiguous" if sigma > 0 else "neutral"
    return {"j_real": j, "quality": quality, "quality_full": full, "quality_low": low,
            "dcpu_pct": dcpu_pct, "k_cpu": k_cpu, "guards": guards, "decision": decision}


def weaknesses(summary: dict[str, dict], per_clip: dict[str, dict],
               limit: int = 8) -> list[tuple[str, str, float, str]]:
    """The worst (class, term, value, note) cells against the anchor.

    BD terms count when positive (more bits than the anchor); flicker and
    pulse deltas in dB are scaled by their guards so they rank alongside.
    A class where the anchor's curve violates the buffer on some clip
    (`noncompliant`) gets a note and ranks after every compliant cell: the
    anchor bought that quality with bits a real receiver could not take, so
    it must not drive our decisions.
    """
    scale = {"d_flicker_psnr_y": CLASS_GUARD_PCT / FLICKER_GUARD_DB,
             "d_periodic_pulse_psnr_y": CLASS_GUARD_PCT / PULSE_GUARD_DB}
    clips: dict[str, list[str]] = {}
    flagged: dict[str, int] = {}
    for clip, row in per_clip.items():
        clips.setdefault(row["class"], []).append(clip)
        flagged[row["class"]] = flagged.get(row["class"], 0) + bool(row.get("anchor_noncompliant"))
    cells = []
    for term in ("quality", "quality_low", *(f"bd_{m}" for m in WORST_METRICS),
                 "d_flicker_psnr_y", "d_periodic_pulse_psnr_y"):
        for name, value in summary[term]["classes"].items():
            if value > 0:
                note = (f"anchor non-compliant on {flagged[name]}/{len(clips[name])} clips"
                        if flagged.get(name) else "")
                cells.append((not note, value * scale.get(term, 1.0), name, term, value, note))
    cells.sort(reverse=True)
    return [(name, term, value, note) for _, _, name, term, value, note in cells[:limit]]
