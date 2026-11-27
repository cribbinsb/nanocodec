"""Real-use evaluation: rate-controlled curves judged by J_real (the default screen).

Our encoders are used in CBR, so that is how they are judged. Each clip is
coded at a target ladder (LADDER_KBPS at 720p30, scaled per clip and capped at
half the source bitrate exactly as the benchmark's BP-3.10 rungs: nceval.rate_ladder,
cbr.clip_targets); the anchors run their
own best real-time rate control at the same targets (cbr.anchor_families:
openh264 RC, SVT M12 --rtc CBR, x264 veryfast / x265 superfast ABR+VBV with
--tune zerolatency, no B-frames, their default AQ). BD-rate is over the
achieved-rate curves, so fixed-QP cadence tricks earn nothing. A point that
breaks the strict leaky bucket, or overshoots its target (ours net of the key
loan) by more than 10% + buffer/window (rate_valid), is a rate-control
failure and is reported as such, but against external anchors its point stays
on the curve: it is a real (rate, quality) point, usually the encoder's rate
floor (openh264 RC and our H.264 cannot reach the lowest rungs on fast 720p pans).

Every point is streamed through cbr.Bench (no YUV written; bitstream deleted
after its packet sizes are read) with the real-use metric set
(metrics.real_metrics: PSNR, VMAF v1 and MS-SSIM on every frame)
and cached by binary, flags, clip, target and metric protocol, so anchor
curves are measured once per anchor build. The terms and the objective are
nceval.jreal; docs/PROCEDURES.md ("Adoption rule: J_real") is the spec.

    # candidate vs baseline (nceval.screen's default mode calls this)
    python3 -m nceval.screen --codec h264 --binary BIN --param a.b=1 [--cpu-summary S] \
        [--clips adopt|train|val|a,b]
    # the J_real scoreboard: one row per run keeps each run under 30 min
    python3 -m nceval.realuse board --binary BIN --rows nanocodec-h264 [--phase anchors]

The default screen, `--clips adopt`, runs the pool levels on the day's
rotating draw (nceval.levels: L1 quick, L2 adopt, L3 final; nceval.pool,
nceval.interval). `--clips train` / `val` keep the old fixed staging
(proxy-4 at 60 frames on the outer three targets, proxy-6, the 18-clip
TRAIN proxy; val-proxy-5, then all natural VAL clips), whose stages stop as
soon as WORSE is certain (nceval.earlystop): ranking only. Any run over
budget splits into class-balanced parts; only uncached points are priced.

Screens, boards and charts decode nothing by default (nceval.gatepolicy:
conformance is checked when a change lands, tools/ci/landing_conformance.py).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import sys
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from . import fairness, gpuscore, jreal, rate_ladder, retention
from .budget import MAX_MINUTES, OverBudget, check as check_budget, estimate_minutes, \
    point_cpu_seconds
from .cbr import (BUFFER_MS, DEFAULT_INTRA_PERIOD, Bench, Family, Point, anchor_families,
                  clip_targets, curve_points, overshoot, rate_scale)
from .cachefile import get_or_create, read_json, sibling_temp, write_atomic
from .cbr_stats import own_rate_fullness, replay_buffer
from .diskguard import DiskBudgetExceeded
from .earlystop import StageScope, certain_worse
from .framepipe import ProcessFailed
from .framestats import coded_frames, probe_frames, with_held_frames
from .clips import Clip, load_clipset
from .flicker_reference import REFERENCE_WINDOW_FLICKER_DB
from .codecs import Codec
from .config import Paths, find_tool, load_paths, openh264_harness
from .encoders.nanocodec import PRESETS_DIR, param_text, preset_values, recorded_values
from .gatepolicy import EVALUATION_GATE, GatePolicy
from .rows import ROWS, Row
from .stages import (ALL_FOUR, PART_FILL, STAGES, TRAIN_PROXY, VAL_PROXY, VAL_STAGES, Stage,
                     balanced_parts, calibrated_plan, cache_root)

#: Target ladder at 720p30 (kbps): the benchmark's BP-3.10 rungs, one definition
#: (nceval.rate_ladder).
LADDER_KBPS = rate_ladder.LADDER_720P30_KBPS
#: CPU of a real-use point relative to budget.point_cpu_seconds (a VMAF v1
#: point): MS-SSIM on every 4th frame adds 4.7 CPU-s to the ~10 of a 720p
#: 150-frame point (measured 2026-11-11: 13-15 CPU-s in all, ~23 priced
#: with the budget's load margin). The earlier 3.0 was fitted while one
#: process ran one point at a time (nceval.slots), so it priced waiting.
REAL_COST_FACTOR = 1.5
WORK_SUBDIR = "realuse"
DEFAULT_FRAMES = 150
DEFAULT_WORKERS = 8
#: Raises the cap on a run's workers (default DEFAULT_WORKERS, the shared PC's share): a
#: rented measurement box (tools/dev/aws_measure.py) sets it to its own cores' worth.
MAX_WORKERS_ENV = "NC_MAX_WORKERS"
OPENH264_HARNESS = openh264_harness()
#: Context anchors per codec on the board (the bars are the scoreboard rows').
CONTEXT_ANCHORS = {Codec.H264: ("x264-veryfast",), Codec.HEVC: ("x265-superfast",)}
STAGE_TERMS = ("j_real",)


def history_path() -> Path:
    return cache_root() / "history-real.jsonl"


def max_workers() -> int:
    """The most workers one run may use (MAX_WORKERS_ENV, else DEFAULT_WORKERS)."""
    count = int(os.environ.get(MAX_WORKERS_ENV, DEFAULT_WORKERS))
    if count < 1:
        raise ValueError(f"{MAX_WORKERS_ENV} must be positive")
    return count


class MissingPoints(SystemExit):
    """Points a cached-only run (Runner.cached_only) would have to measure: it fails
    loudly with them instead, so a render never codes or scores on this PC."""

    #: The points listed by name in the message (the rest are counted).
    SHOWN = 40

    def __init__(self, jobs: list):
        self.points = [p for _, p in jobs]
        lines = [f"  {p.family.name} {p.clip.name} {p.label}" for p in self.points[:self.SHOWN]]
        more = len(self.points) - len(lines)
        super().__init__(
            f"{len(self.points)} points are not in the store (render measures nothing; "
            "measure them on rented machines: chart_refresh.py --from measure, or render "
            "--measure-missing for a local chart):\n" + "\n".join(lines)
            + (f"\n  ... and {more} more" if more else ""))


class Runner:
    """Benches per frame count, budgeted measurement, curves and comparisons."""

    def __init__(self, paths: Paths, policy: GatePolicy, workers: int, over_budget: bool,
                 stored_ms_ssim: bool = False, prebuilt: Path | None = None,
                 xpsnr: bool = False, stored_xpsnr: bool = False, cached_only: bool = False,
                 may_measure: frozenset[str] = frozenset(), skip_missing: bool = False):
        """`stored_ms_ssim`: reuse stored results of an older MS-SSIM protocol
        (cbr.Bench.stored_ms_ssim); `prebuilt`: score streams encoded elsewhere
        (cbr.Bench.prebuilt); `xpsnr`: also score XPSNR (cbr.Bench.xpsnr), and with
        `stored_xpsnr` reuse stored results without it (cbr.Bench.stored_xpsnr);
        `cached_only`: measure nothing but the families in `may_measure`, raising
        MissingPoints with every other point not in the store (a chart render), or with
        `skip_missing` leaving those points out of the curves (counted in `skipped`; a
        report over stored points, nceval.buffer_use)."""
        self.cached_only, self.may_measure = cached_only, may_measure
        self.skip_missing, self.skipped = skip_missing, 0
        self.paths, self.policy = paths, policy
        self.stored_ms_ssim, self.prebuilt, self.xpsnr = stored_ms_ssim, prebuilt, xpsnr
        self.stored_xpsnr = stored_xpsnr
        self.workers, self.over_budget = min(workers, max_workers()), over_budget
        if fairness.conditions() == fairness.SHIPPED_DEFAULTS:
            print(f"!!! {fairness.CONDITIONS_ENV}={fairness.SHIPPED_DEFAULTS}: our encoder runs "
                  f"at its shipped defaults, NOT the benchmark's conditions (a study of the "
                  f"defaults; its points never mix with benchmark-condition ones)",
                  file=sys.stderr, flush=True)
        # Encode on boxes, score on the PC's GPU (nceval.gpuscore): set by aws_measure's
        # environment, here (the rounds) and on the box (encode-only).
        self.encode_only = gpuscore.encode_only()
        self.prebuilt = self.prebuilt or gpuscore.prebuilt_dir()
        self.rounds = gpuscore.EncodeRounds.from_env(paths.work)
        self._benches: dict[int, Bench] = {}
        #: Points that are not rate-valid (rate_valid: buffer broken, or average
        #: rate too far over the target, e.g. openh264 RC's floor on fast pans).
        self.unreachable: dict[str, set[str]] = {}
        #: Points our encoder refused to code (it exited non-zero, e.g. VP9 on
        #: a width that is not a multiple of 8), per family: listed on the
        #: board, never silently dropped. Conformance failures still stop the run.
        self.refused: dict[str, set[str]] = {}

    def bench(self, frames: int, intra: int = DEFAULT_INTRA_PERIOD) -> Bench:
        if (frames, intra) not in self._benches:
            work = self.paths.work / WORK_SUBDIR / f"f{frames}-ip{intra}"
            # a kept store (the charts' 150-frame cache) is marked as it is made
            retention.ensure_entry_marker(work)
            self._benches[(frames, intra)] = Bench(self.paths, frames, intra, work,
                                                   metric_set="real",
                                                   stored_ms_ssim=self.stored_ms_ssim,
                                                   prebuilt=self.prebuilt, xpsnr=self.xpsnr,
                                                   stored_xpsnr=self.stored_xpsnr,
                                                   encode_only=self.encode_only)
        return self._benches[(frames, intra)]

    def points(self, families: list[Family], clips: list[Clip], frames: int,
               indices: tuple[int, ...] | None,
               key_seconds: float | None = None,
               ladder: Callable[[Clip], list[int]] | None = None,
               key_frames: int | None = None) -> list[tuple[Bench, Point]]:
        """Every point to measure; `ladder` gives other targets per clip (clip kbps)
        than the real-use LADDER_KBPS (e.g. the chart's anchor extension). The key
        period is `key_frames` frames (a key share, nceval.key_shares), else
        `key_seconds`, else one key (DEFAULT_INTRA_PERIOD)."""
        jobs = []
        for clip in clips:
            intra = key_frames or (round(key_seconds * clip.fps) if key_seconds
                                   else DEFAULT_INTRA_PERIOD)
            bench = self.bench(min(frames, clip.frames), intra)
            targets = ladder(clip) if ladder else clip_targets(clip, LADDER_KBPS, indices)
            for family in families:
                jobs += [(bench, p) for p in curve_points(bench, family, clip, self.policy,
                                                          targets=targets)]
        return jobs

    def price(self, jobs: list[tuple[Bench, Point]]) -> float:
        """CPU-seconds of the points not yet cached."""
        return sum(point_cpu_seconds(p.clip.width, p.clip.height, bench.frames)
                   * REAL_COST_FACTOR for bench, p in jobs if not bench.cached(p))

    def measure(self, label: str, families: list[Family], clips: list[Clip], frames: int,
                indices: tuple[int, ...] | None = None,
                key_seconds: float | None = None, budgeted: bool = True,
                ladder: Callable[[Clip], list[int]] | None = None,
                on_clip: ClipCallback | None = None, key_frames: int | None = None) -> dict:
        """{(family, clip): [point terms]}, run in class-balanced parts if over budget.

        `key_frames` (frames) or `key_seconds` sets a key period (default: one key,
        DEFAULT_INTRA_PERIOD).
        `budgeted` False skips the wall-time check and split: a tune trial is
        one unit of a study whose slices are bounded by `tune --timeout`.
        `on_clip(clip, curves)` is called as each clip's last point arrives
        (clips are measured in the order given); a reason it returns cancels
        the points not yet started, which are then missing from the curves.
        """
        jobs = self.points(families, clips, frames, indices, key_seconds, ladder, key_frames)
        minutes = estimate_minutes(self.price(jobs)) if budgeted else 0.0
        if len(clips) > 1 and minutes > MAX_MINUTES:
            count = min(len(clips), math.ceil(minutes / (MAX_MINUTES * PART_FILL)))
            curves: dict = {}
            stopped = None

            def note_stop(clip: str, clip_curves: dict) -> str | None:
                nonlocal stopped
                stopped = on_clip(clip, clip_curves) if on_clip else None
                return stopped
            for index, part in enumerate(balanced_parts(clips, count), 1):
                if stopped:
                    break
                print(f"{label}: part {index}/{count} ({len(part)} clips)", flush=True)
                curves.update(self.measure(label, families, part, frames, indices,
                                           key_seconds, ladder=ladder, on_clip=note_stop,
                                           key_frames=key_frames))
            return curves
        if budgeted:
            check_budget(f"{label} ({len(clips)} clips, {len(jobs)} points)", self.price(jobs),
                         "fewer --clips, --frames 60, or the staged screen", self.over_budget)
        return self.curves(jobs, self.run_jobs(jobs, [on_clip] if on_clip else None))

    def measure_together(self, families: list[Family],
                         requests: list[StageRequest]) -> list[dict]:
        """Several stages' curves (as `measure`), measured concurrently in one pool.

        A screen's TRAIN and VAL stages are independent: measured side by side
        the workers stay busy across both, rather than idling at the tail of
        the first. A reason from any request's `on_clip` cancels every
        request's points not yet started. If all of them together do not fit
        the budget, they run one after another (each splitting into parts).
        """
        jobs = [self.points(families, r.clips, r.frames, r.indices) for r in requests]
        flat = [job for request_jobs in jobs for job in request_jobs]
        cost = self.price(flat)
        if estimate_minutes(cost) > MAX_MINUTES:
            return [self.measure(r.label, families, r.clips, r.frames, r.indices,
                                 on_clip=r.on_clip) for r in requests]
        labels = " + ".join(r.label for r in requests)
        check_budget(f"{labels} ({len(flat)} points)", cost,
                     "fewer --clips, --frames 60, or the staged screen", self.over_budget)
        offsets = [sum(len(j) for j in jobs[:i]) for i in range(len(jobs) + 1)]
        results = self.run_jobs(flat, [r.on_clip for r in requests], offsets)
        return [self.curves(jobs[i], results[offsets[i]:offsets[i + 1]])
                for i in range(len(requests))]

    def run_jobs(self, jobs: list[tuple[Bench, Point]],
                 callbacks: list[ClipCallback | None] | None = None,
                 offsets: list[int] | None = None) -> list[dict | None]:
        """Measure `jobs` in the pool; results in job order (None: refused or cancelled).

        `offsets` cut `jobs` into groups (one per callback; default one
        group). As a group's clip gets its last point, the group's callback
        sees that clip's curves; a returned reason cancels every point not
        yet started.

        The bookkeeping and the callback run in the worker that measured the
        point, before it takes the next job, and every job checks the stop
        flag as it starts: a stop therefore lets only the points already in
        flight on other workers finish, however late the caller's thread is
        scheduled (with the callback on the caller's thread, a free worker
        kept starting new points until that thread ran).
        """
        skip: set[int] = set()
        if self.cached_only:
            skip = {i for i, (b, p) in enumerate(jobs)
                    if p.family.name not in self.may_measure and not b.cached(p)}
            if skip and not self.skip_missing:
                raise MissingPoints([jobs[i] for i in sorted(skip)])
            self.skipped += len(skip)
        if self.rounds is not None:
            self.rounds.ensure(jobs, lambda batch: [
                (b, p) for b, p in batch if not b.cached(p) and not b.has_prebuilt(p)])
        offsets = offsets or [0, len(jobs)]
        group = lambda index: next(g for g in range(len(offsets) - 1) if index < offsets[g + 1])
        key = lambda index: (group(index), jobs[index][1].clip.name)
        open_points = Counter(key(i) for i in range(len(jobs)))
        results: list[dict | None] = [None] * len(jobs)
        lock = threading.Lock()
        stop = threading.Event()

        def measure(index: int) -> None:
            # encode-only (on a box): every point of the batch, whatever the early
            # stops on its missing results say (the PC decides; gpuscore)
            if stop.is_set() and not self.encode_only:
                return
            try:
                result = None if index in skip else self._measure_one(jobs[index])
            except BaseException:
                stop.set()  # the run fails: start nothing more
                raise
            with lock:
                results[index] = result
                done = key(index)
                open_points[done] -= 1
                callback = callbacks[done[0]] if callbacks else None
                if stop.is_set() or open_points[done] or callback is None:
                    return
                members = [i for i in range(len(jobs)) if key(i) == done]
                if callback(done[1], self.curves([jobs[i] for i in members],
                                                 [results[i] for i in members])):
                    stop.set()

        try:
            with ThreadPoolExecutor(self.workers) as pool:
                list(pool.map(measure, range(len(jobs))))
        except DiskBudgetExceeded as error:
            if self.rounds is None:
                raise
            # GPU-scored (nceval.gpuscore): a local guard, so the measurement is refused
            # with how to fix it (its encoded streams are kept for a retry)
            raise gpuscore.GpuUnavailable(f"disk floor: {error}") from error
        return results

    def curves(self, jobs: list[tuple[Bench, Point]], results: list[dict | None]) -> dict:
        """{(family, clip): [point terms in job order]}; missing points are left out."""
        curves: dict = {(p.family.name, p.clip.name): [] for _, p in jobs}
        for (_, point), result in zip(jobs, results):
            if result is None:
                continue
            terms = point_terms(result, len(result["frame_key"]) / point.clip.fps)
            curves[(point.family.name, point.clip.name)].append(terms)
            if not terms["rate_valid"]:
                self.unreachable.setdefault(point.family.name, set()).add(
                    f"{point.clip.name}@{point.kbps}")
        return curves

    def _measure_one(self, job: tuple[Bench, Point]) -> dict | None:
        bench, point = job
        try:
            result = bench.measure(point)
            if result is not None and self.rounds is not None:
                self.rounds.note(bench.result_path(point))
        except ProcessFailed as refusal:
            if not point.family.own:
                raise
            reason = str(refusal).splitlines()[0][:120]
            self.refused.setdefault(point.family.name, set()).add(
                f"{point.clip.name}@{point.kbps}: {reason}")
            return None
        return result


#: Called as a clip's last point arrives: (clip name, its curves) -> a reason
#: to stop measuring, or None.
ClipCallback = Callable[[str, dict], "str | None"]


@dataclass(frozen=True)
class StageRequest:
    """One stage to measure alongside others (Runner.measure_together)."""
    label: str
    clips: list[Clip]  # measured in this order
    frames: int
    indices: tuple[int, ...] | None
    on_clip: ClipCallback | None = None


#: Average-rate overshoot every window may show, before the buffer allowance.
RATE_TOLERANCE = 0.10


def rate_valid(result: dict, window_seconds: float) -> bool:
    """Whether a CBR point is a valid real-use point (coordinator, 2026-11-09).

    Invalid when it breaks the strict leaky bucket (jreal.noncompliant: peak
    above 1.5x the buffer) or when its loan-adjusted average rate overshoots
    the target by more than RATE_TOLERANCE + buffer / window. The allowance is
    the debt a 1 s buffer may legitimately carry at the end of the window: a
    controller funding a static refinement burst from the buffer can read
    +8..+16% over 150 frames (5 s) and converge to +0.7% over 300 frames.
    """
    allowance = RATE_TOLERANCE + BUFFER_MS / 1000.0 / window_seconds
    return not jreal.noncompliant(result) and overshoot(result) <= allowance


def key_buffer_terms(result: dict) -> dict:
    """The first key frame's size in buffers at the target (the 1 s buffer, BUFFER_MS) and
    the worst one-second window over the target (cbr_stats.replay_buffer), as every
    encoder's buffer facts report them (nceval.buffer_use); empty without a target."""
    if not result.get("target_kbps"):
        return {}
    key = result["frame_key"].index(True) if True in result.get("frame_key", []) else 0
    capacity_bits = result["target_kbps"] * BUFFER_MS
    return {"key_buffers": result["frame_bits"][key] / capacity_bits,
            "max_window_ratio": result["max_window_ratio"]}


def point_terms(result: dict, window_seconds: float) -> dict:
    """One CBR point as J_real sees it: achieved rate, sequence and temporal quality."""
    terms = {key: result[key] for key in ("kbps", "psnr_yuv", "vmaf_v1", "ms_ssim",
                                          "overruns", "max_fullness", "target_kbps")}
    terms.update({m: result[m] for m in jreal.REPORTED_METRICS if m in result})
    if result.get("frame_psnr_y"):  # sequence PSNR-Y, the per-frame mean (metrics.psnr_measurement)
        terms["psnr_y"] = statistics.fmean(result["frame_psnr_y"])
    terms["rate_valid"] = rate_valid(result, window_seconds)
    if result.get("frame_bits"):  # the buffer at the point's own rate (nceval.buffer_use)
        terms["own_rate_fullness"] = own_rate_fullness(
            result["frame_bits"], len(result["frame_bits"]) / window_seconds, BUFFER_MS)
        terms.update(key_buffer_terms(result))
    terms["window_seconds"] = window_seconds  # the ceiling guard judges long windows absolutely
    terms.update(jreal.temporal_stats(result["frame_psnr_y"], result["frame_vmaf_v1"],
                                      result["frame_key"]))
    if "base" in result:  # a two-layer stream's base layer at its own rate (nceval.svcbase)
        terms["base"] = point_terms(result["base"], window_seconds)
    return terms


def compare(curves: dict, anchor: str, test: str, clips: list[Clip],
            min_points: int = jreal.BD_MIN_POINTS, paired: bool = False,
            regimes: dict[str, tuple[int, int]] | None = None) -> dict[str, dict]:
    """Per-clip J_real terms of `test` vs `anchor`.

    `paired` (ours vs ours: screens, and board rows whose bar is our own
    encoder) drops the rungs either arm overshot (jreal.pair_rungs).
    `regimes` ({name: (low, high) 720p30 kbps}) adds each rate regime's terms
    (jreal.regime_terms), its band scaled to each clip like the ladder (cbr.rate_scale).
    """
    rows = {}
    for clip in clips:
        a, t = curves[(anchor, clip.name)], curves[(test, clip.name)]
        wins = losses = 0
        if paired:
            a, t, wins, losses = jreal.pair_rungs(a, t)
        scale = rate_scale(clip)
        bands = {name: (low * scale, high * scale) for name, (low, high) in (regimes or {}).items()}
        terms = jreal.compare_clip(a, t, min_points, REFERENCE_WINDOW_FLICKER_DB.get(clip.name),
                                   bands)
        names = jreal.pulse_companions(jreal.PULSE_TERMS[0])
        # a curve with periodic keys (a long-window level): the guarded pulse is the
        # key-attributed one, the raw change kept beside it (as add_periodic_pulse)
        terms[names["raw"]] = terms["d_periodic_pulse_psnr_y"]
        terms["d_periodic_pulse_psnr_y"] = jreal.key_attributed_pulse(
            terms["d_periodic_pulse_psnr_y"], terms["d_periodic_key_psnr_y"])
        rows[clip.name] = {"class": clip.content_class, "points": [len(a), len(t)], **terms,
                           "rate_wins": wins, "rate_losses": losses}
    return rows


#: The periodic-key pass: keys every PERIODIC_KEY_SECONDS (150 frames at
#: 30 fps hold keys at 0, 60 and 120), on one clip per class, at the
#: PERIODIC_RUNGS lowest ladder targets (the region we target). Only its
#: periodic-key pulse is used; the main pass keeps one key per clip.
PERIODIC_KEY_SECONDS = 2.0
PERIODIC_RUNGS = (0, 1)
#: The long periodic-key pass beside it: keys every LONG_PULSE_KEY_SECONDS on
#: LONG_PULSE_FRAMES frames (keys at 0 and 300 at 30 fps), one clip per class long
#: enough. A pulse that builds up over a long key period does not show at 2 s keys:
#: 2026-11-15, rate.key_inherit passed the 2 s guard and pulsed +1.61 dB (HEVC) and
#: +0.68 dB (AV2) at 10 s keys.
LONG_PULSE_KEY_SECONDS = 10.0
LONG_PULSE_FRAMES = 600


def one_per_class(clips: list[Clip], preferred: tuple[str, ...] = ()) -> list[Clip]:
    """One clip per content class, `preferred` names first, then by name."""
    chosen: dict[str, Clip] = {}
    for clip in sorted(clips, key=lambda c: (c.name not in preferred, c.name)):
        chosen.setdefault(clip.content_class, clip)
    return sorted(chosen.values(), key=lambda c: c.content_class)


def add_periodic_pulse(runner: "Runner", anchor: Family, test: Family, clips: list[Clip],
                       frames: int, per_clip: dict[str, dict], label: str,
                       pool: list[Clip] | None = None) -> None:
    """Measure the periodic-key passes and fill, in the rows of `per_clip`,
    d_periodic_pulse_psnr_y (keys every PERIODIC_KEY_SECONDS on `clips`) and
    d_periodic_pulse_long_psnr_y (every LONG_PULSE_KEY_SECONDS, LONG_PULSE_FRAMES
    frames from frame 0, on one clip per class of `pool` (default: the rows'
    clips in `clips`) long enough)."""
    passes = [(clips, frames, PERIODIC_KEY_SECONDS, "d_periodic_pulse_psnr_y")]
    long_pool = [replace(c, start_frame=0) for c in (pool or clips)
                 if c.frames >= LONG_PULSE_FRAMES and c.name in per_clip]
    if long_pool:
        passes.append((one_per_class(long_pool), LONG_PULSE_FRAMES, LONG_PULSE_KEY_SECONDS,
                       "d_periodic_pulse_long_psnr_y"))
    for pass_clips, pass_frames, seconds, term in passes:
        curves = runner.measure(f"{label} periodic keys ({seconds:g} s)", [anchor, test],
                                pass_clips, pass_frames, PERIODIC_RUNGS, seconds)
        for clip in pass_clips:
            row = per_clip.get(clip.name)
            if row is None:
                continue  # no row at the level (our encoder refused the clip there)
            a, t = curves[(anchor.name, clip.name)], curves[(test.name, clip.name)]
            if anchor.own:
                a, t, _, _ = jreal.pair_rungs(a, t)
            terms = jreal.compare_clip(a, t, min_points=2)
            names = jreal.pulse_companions(term)
            row[names["raw"]] = terms["d_periodic_pulse_psnr_y"]
            row[names["key"]] = terms["d_periodic_key_psnr_y"]
            row[names["pre_key"]] = terms["d_periodic_pre_key_psnr_y"]
            # the guarded term: only as far as the key itself got worse
            row[term] = jreal.key_attributed_pulse(row[names["raw"]], row[names["key"]])


def param_args(pairs: list[str]) -> tuple[str, ...]:
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param needs name=value, got {pair!r}")
    return tuple(x for pair in pairs for x in ("--param", pair))


def own_family(encoder: str, binary: Path, presets: Path = PRESETS_DIR,
               recorded: bool = False) -> Family:
    """`nanocodec-<codec>[@stem]` in CBR; presets/<codec>/<stem>.toml's values
    become --param pairs (no stem: medium, the engine defaults).

    `presets` is the preset directory: a frozen build is measured with the
    presets of its own commit (board --presets-dir), not the tree's newer ones.
    `recorded`: the name as a build record wrote it, in that build's own preset names
    (nanocodec.recorded_values), not today's rung renamed onto the build.
    """
    base, _, preset = encoder.partition("@")
    codec = Codec(base.removeprefix("nanocodec-"))
    overrides = (recorded_values if recorded else preset_values)(presets, codec, preset)
    extra = param_args([f"{k}={param_text(v)}" for k, v in sorted(overrides.items())])
    return Family(encoder, codec, binary, extra, own=True, cqp=False)


def main_clips(paths: Paths) -> dict[str, Clip]:
    return {c.name: c for c in load_clipset(paths.clips / "main.json", paths.data_root).clips}


def natural_val(paths: Paths) -> list[Clip]:
    return [c for c in main_clips(paths).values() if c.split == "val" and c.clean_reference is None]


def write_natural_val_clipset(paths: Paths, out: Path) -> Path:
    """The main chart's clips (`natural_val`) as a clip-set file at `out`: main.json's own
    entries of those clips, unchanged, so `clipset_clips(out)` gives the same clips and
    every point the same identity (cbr.Bench.identity) as `natural_val`. Rented machines
    measure the main chart through it (tools/dev/aws_heldout_chart.py)."""
    wanted = {c.name for c in natural_val(paths)}
    main = json.loads((paths.clips / "main.json").read_text())
    entries = [entry for entry in main["clips"] if entry["name"] in wanted]
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"name": out.stem, "clips": entries}, indent=1) + "\n"
    if not out.is_file() or out.read_text() != text:
        out.write_text(text)
    return out


def clipset_path(spec: str, paths: Paths) -> Path:
    """A clip set by name (`adv12` -> $NC_DATA_ROOT/clips/adv12.json) or by path."""
    return Path(spec) if spec.endswith(".json") or "/" in spec else paths.clips / f"{spec}.json"


def clipset_clips(spec: str, paths: Paths) -> list[Clip]:
    """Every natural, non-TEST clip of another clip set (e.g. the held-out
    pristine robustness set `adv12`). Its board rows and page are kept apart
    from the VAL board's (board_dir)."""
    return [c for c in load_clipset(clipset_path(spec, paths), paths.data_root).clips
            if c.split != "test" and c.clean_reference is None]


#: Clip sets whose clips a screen may also name in --clips (the pristine
#: robustness set: conference clips such as adv_johnny_720p; the held-out
#: broad set every preset or Default change is confirmed on, owner
#: 2026-11-11).
EXTRA_SCREEN_CLIPSETS = ("adv12", "broad_independent_v1")


def screen_clips(spec: str, paths: Paths) -> list[Clip]:
    by_name = main_clips(paths)
    if spec == "train":
        return [by_name[n] for n in TRAIN_PROXY]
    if spec == "val":
        return natural_val(paths)
    names = spec.split(",")
    for clipset in EXTRA_SCREEN_CLIPSETS:
        if any(n not in by_name for n in names) and clipset_path(clipset, paths).exists():
            by_name.update({c.name: c for c in clipset_clips(clipset, paths)})
    unknown = [n for n in names if n not in by_name or by_name[n].split == "test"]
    if unknown:
        raise SystemExit(f"unknown or sealed clips: {', '.join(unknown)}")
    return [by_name[n] for n in names]


# --- the screen (candidate vs baseline) -----------------------------------------------

#: `--clips adopt` (the default): the pool levels L1 quick, L2 adopt, L3
#: final on the day's rotating draw (nceval.levels, nceval.pool).
#: `--clips train` / `val` run the old fixed tracks (`train` from the 60-frame
#: proxy-4): ranking only, never adoption evidence (they are overfit).
TRACK_PLANS = {"train": STAGES, "val": VAL_STAGES}
ADOPT = "adopt"
UNSTAGED = Stage("all", None, ALL_FOUR, 0.0)


@dataclass
class Track:
    """One clip set of a screen and its stages (None: one unstaged stage)."""
    name: str
    clips: list[Clip]
    plan: tuple[Stage, ...] | None
    stages: list[dict]

    @property
    def decision(self) -> str | None:
        return self.stages[-1]["decision"] if self.stages else None


def screen_tracks(spec: str, paths: Paths, staged: bool) -> list[Track]:
    plan = TRACK_PLANS.get(spec) if staged else None
    return [Track(spec, screen_clips(spec, paths), plan, [])]


class StageWatch:
    """A stage's per-clip comparison, filled as each clip completes, and its
    early stop (nceval.earlystop: a WORSE verdict already certain)."""

    def __init__(self, stage: Stage, clips: list[Clip], indices: tuple[int, ...] | None,
                 families: tuple[Family, Family], cpu: tuple[float | None, float],
                 early_stop: bool):
        self.stage, self.families, self.cpu = stage, families, cpu
        self.clips = {c.name: c for c in clips}
        self.scope = StageScope({c.name: c.content_class for c in clips},
                                {c.name: len(clip_targets(c, LADDER_KBPS, indices))
                                 for c in clips})
        self.early_stop = early_stop
        self.rows: dict[str, dict] = {}
        self.reason: str | None = None

    def ordered_clips(self) -> list[Clip]:
        return [self.clips[name] for name in self.scope.order()]

    def on_clip(self, clip: str, curves: dict) -> str | None:
        anchor, test = self.families
        self.rows.update(compare(curves, anchor.name, test.name, [self.clips[clip]],
                                 paired=True))
        if self.early_stop and self.reason is None and len(self.rows) < len(self.clips):
            dcpu, k_cpu = self.cpu
            self.reason = certain_worse(self.rows, self.scope, self.stage.sigma, dcpu, k_cpu)
        return self.reason


def run_step(runner: Runner, families: tuple[Family, Family], steps: list[tuple[Track, Stage]],
             frames_arg: int, staged: bool, args: argparse.Namespace,
             early_stop: bool = True, announce: bool = True) -> None:
    """Measure one stage of each track side by side and decide each.

    `early_stop` False: no point-estimate stop inside the stage; `announce`
    False: the caller prints the stage (the pool levels decide on their
    bootstrap interval, nceval.interval, not on the stage verdict)."""
    watches, requests = [], []
    for track, stage in steps:
        chosen = [c for c in track.clips if stage.clips is None or c.name in stage.clips]
        frames = min(frames_arg, stage.frames or frames_arg)
        indices = stage.qp_indices if staged else None
        watch = StageWatch(stage, chosen, indices, families, (args.dcpu, args.k),
                           early_stop=early_stop and not args.all_stages)
        watches.append((track, watch, frames))
        requests.append(StageRequest(f"real-use {track.name} stage {stage.name}",
                                     watch.ordered_clips(), frames, indices, watch.on_clip))
    runner.measure_together(list(families), requests)
    stopper = next((f"{track.name} {w.stage.name}: {w.reason}"
                    for track, w, _ in watches if w.reason), None)
    for track, watch, frames in watches:
        summary = jreal.summarise(watch.rows)
        verdict = jreal.objective(summary, args.dcpu, watch.stage.sigma, args.k)
        complete = len(watch.rows) == len(watch.clips)
        if watch.reason:
            verdict["decision"] = "worse"
        elif not complete:
            verdict["decision"] = "cancelled"  # another track's stage was decided WORSE
        stage = {"stage": watch.stage.name, "track": track.name, "sigma": watch.stage.sigma,
                 "frames": frames, "decision": verdict["decision"], "verdict": verdict,
                 "per_clip": watch.rows, "summary": summary}
        if not complete:
            stage["stopped_early"] = watch.reason or f"cancelled ({stopper})"
        track.stages.append(stage)
        if announce:
            print(format_stage(stage), flush=True)


def combined_decision(decisions: list[str]) -> str:
    """One verdict over the tracks: any WORSE rejects; BETTER needs every track."""
    if "worse" in decisions:
        return "worse"
    if all(d == "better" for d in decisions):
        return "better"
    return "ambiguous" if "ambiguous" in decisions else "neutral"


def screen_families(args: argparse.Namespace) -> tuple[Family, Family]:
    """A screen's (anchor, candidate): our encoder in CBR, each with its arguments."""
    codec = Codec(args.codec)
    binary = Path(args.binary).resolve()
    anchor_binary = Path(args.anchor_binary).resolve() if args.anchor_binary else binary
    test = Family(f"nanocodec-{codec.value}@candidate", codec, binary, param_args(args.param),
                  own=True, cqp=False)
    anchor = Family(f"nanocodec-{codec.value}", codec, anchor_binary,
                    param_args(args.anchor_param), own=True, cqp=False)
    if (test.tool, test.extra) == (anchor.tool, anchor.extra):
        raise SystemExit("candidate and anchor are identical")
    return anchor, test


def set_final_estimator(args: argparse.Namespace) -> None:
    """L3 per clip: single-point for a parameter-only change (one binary), the full
    ladder for a code change, unless --final-estimator says."""
    from . import levels
    anchor, test = screen_families(args)
    args.final_estimator = getattr(args, "final_estimator", None) or (
        levels.SINGLE if test.tool == anchor.tool else levels.LADDER)


def screen(args: argparse.Namespace) -> dict:
    """Candidate vs anchor build, both in CBR on the ladder, staged, decided by J_real.

    Tracks (`--clips adopt`: TRAIN and VAL) run their stages side by side; a
    stage stops as soon as its WORSE verdict is certain (nceval.earlystop),
    and a WORSE track stops the others. Points are not decoded unless
    --gate asks (nceval.gatepolicy).
    """
    paths = load_paths()
    codec = Codec(args.codec)
    anchor, test = screen_families(args)
    binary, anchor_binary = test.tool, anchor.tool
    if args.frames == "full":
        raise SystemExit("real-use screens take a frame count (e.g. --frames 150)")
    gate = GatePolicy(args.gate or EVALUATION_GATE.value)
    runner = Runner(paths, gate, args.workers, args.over_budget)
    if args.clips == ADOPT:
        from . import levels  # the pool levels build on this module's runner
        set_final_estimator(args)
        return {"mode": "real", "codec": codec.value, "binary": str(binary),
                "params": args.param, "anchor_binary": str(anchor_binary),
                "anchor_params": args.anchor_param, "frames": args.frames,
                "gate": gate.value, "ladder_kbps": LADDER_KBPS,
                **levels.run(runner, (anchor, test), paths, args)}
    tracks = screen_tracks(args.clips, paths, args.staged)
    for track in tracks:
        if track.plan:
            track.plan, note = calibrated_plan(track.name, track.plan, history_path(),
                                               STAGE_TERMS)
            print(f"{track.name} stages (real-use CBR ladder, J_real): {note}", flush=True)
    step = 0
    while True:
        # A track goes on while its proxy is ambiguous (with --all-stages: always).
        open_tracks = [t for t in tracks if step < len(t.plan or (None,)) and
                       (args.all_stages or t.decision in (None, "ambiguous"))]
        if not open_tracks:
            break
        run_step(runner, (anchor, test),
                 [(t, (t.plan or (UNSTAGED,))[step]) for t in open_tracks],
                 int(args.frames), args.staged, args)
        step += 1
        if any(t.decision == "worse" for t in tracks) and not args.all_stages:
            break
    for track in tracks:
        if track.decision in ("better", "neutral"):
            recheck_with_periodic_keys(runner, anchor, test, track.clips, int(args.frames),
                                       track.stages[-1], (args.dcpu, args.k))
    decision = combined_decision([t.decision for t in tracks])
    for track in tracks:
        complete = [s for s in track.stages if "stopped_early" not in s]
        if track.plan and complete:
            record_history(track.name, codec, complete)
    if len(tracks) > 1:
        print("tracks: " + ", ".join(f"{t.name} {t.decision.upper()}" for t in tracks)
              + f" -> decision {decision.upper()}", flush=True)
    return {"mode": "real", "codec": codec.value, "binary": str(binary), "params": args.param,
            "anchor_binary": str(anchor_binary), "anchor_params": args.anchor_param,
            "clips": args.clips, "frames": args.frames, "gate": gate.value,
            "ladder_kbps": LADDER_KBPS,
            "staged": any(t.plan for t in tracks), "decision": decision,
            "tracks": {t.name: t.decision for t in tracks},
            "stages": [s for t in tracks for s in t.stages]}


def recheck_with_periodic_keys(runner: Runner, anchor: Family, test: Family,
                               clips: list[Clip], frames: int, stage: dict,
                               cpu: tuple[float | None, float]) -> None:
    """Before a candidate passes, measure its periodic-key pulse and re-decide.

    One clip per class of the stage's clips.
    """
    rows = stage["per_clip"]
    subset = one_per_class([c for c in clips if c.name in rows])
    add_periodic_pulse(runner, anchor, test, subset, frames, rows, "real-use")
    stage["summary"] = jreal.summarise(rows)
    dcpu, k_cpu = cpu
    stage["verdict"] = jreal.objective(stage["summary"], dcpu, stage["sigma"], k_cpu)
    stage["decision"] = stage["verdict"]["decision"]
    print(f"periodic-key pulse: {pct(stage['summary']['d_periodic_pulse_psnr_y']['balanced'], 'dB')}"
          f" -> decision {stage['decision'].upper()}", flush=True)


def record_history(clip_set: str, codec: Codec, stages: list[dict]) -> None:
    entry = {"set": clip_set, "codec": codec.value,
             "stages": {s["stage"]: {"j_real": s["verdict"]["j_real"]} for s in stages}}
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(entry) + "\n")


def pct(value: float | None, unit: str = "%") -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}{unit}" if unit == "dB" else f"{value:+.1f}%"


CLIP_COLUMNS = (("quality", "J-full"), ("quality_low", "J-low"), ("bd_psnr_yuv", "PSNR"),
                ("bd_vmaf_v1", "VMAFv1"), ("bd_ms_ssim", "MSSSIM"),
                ("bd_worst_psnr_y", "w1%Y"), ("bd_worst_vmaf_v1", "w1%VMAF"),
                ("bd_worst_frame_psnr_y", "w1%Y1"))


def format_stage(stage: dict) -> str:
    verdict = stage["verdict"]
    lines = [f"== stage {stage['stage']} (sigma {stage['sigma']:.1f}%, {stage['frames']} frames;"
             " real-use CBR ladder)",
             f"{'clip':26s} {'class':9s} " + " ".join(f"{h:>7s}" for _, h in CLIP_COLUMNS)
             + f" {'dFlk':>6s} {'dFlk1':>6s} {'dKey1':>6s} {'dKeyN':>6s}"]
    for name, row in sorted(stage["per_clip"].items(), key=lambda kv: (kv[1]["class"], kv[0])):
        lines.append(f"{name:26s} {row['class'][:9]:9s} "
                     + " ".join(f"{pct(row.get(t)):>7s}" for t, _ in CLIP_COLUMNS)
                     + f" {pct(row['d_flicker_psnr_y'], 'dB'):>6s}"
                     f" {pct(row.get('d_flicker_frame_psnr_y'), 'dB'):>6s}"
                     f" {pct(row['d_first_pulse_psnr_y'], 'dB'):>6s}"
                     f" {pct(row.get('d_periodic_pulse_psnr_y'), 'dB'):>6s}")
    lines.append(format_summary(stage["summary"]))
    buffer = stage["summary"]["compliance"]
    lines.append(f"rate: rungs dropped as overshot (>10%), compliance wins "
                 f"{buffer['rate_wins']} / losses {buffer['rate_losses']}; buffer: overrun frames "
                 f"{buffer['test_overruns']} (anchor {buffer['anchor_overruns']}), "
                 f"non-compliant points {buffer['test_noncompliant']} "
                 f"(anchor {buffer['anchor_noncompliant']})")
    lines.append(f"J_real {pct(verdict['j_real'])} (full {pct(verdict['quality_full'])}, "
                 f"low {pct(verdict['quality_low'])}, dCPU "
                 f"{'not given' if verdict['dcpu_pct'] is None else pct(verdict['dcpu_pct'])}"
                 f" at k {verdict['k_cpu']:g})"
                 f"; guards: {'; '.join(verdict['guards']) or 'none'}")
    lines.append(f"decision: {stage['decision'].upper()}")
    return "\n".join(lines)


def format_summary(summary: dict) -> str:
    rows = []
    for term, label in CLIP_COLUMNS + (("d_flicker_psnr_y", "dFlicker"),
                                       ("d_flicker_guard_psnr_y", "dFlkGrd"),
                                       ("d_flicker_frame_psnr_y", "dFlick1"),
                                       ("d_first_pulse_psnr_y", "dKey1"),
                                       ("d_periodic_pulse_psnr_y", "dKeyN")):
        agg = summary[term]
        unit = "dB" if term.startswith("d_") else "%"
        classes = ", ".join(f"{k} {pct(v, unit)}" for k, v in agg["classes"].items())
        missing = f"; missing {','.join(agg['missing'])}" if agg["missing"] else ""
        rows.append(f"  {label:8s} balanced {pct(agg['balanced'], unit):>8s}  median "
                    f"{pct(agg['median'], unit):>8s}  worse {agg['worse']}/{agg['scored']}  "
                    f"[{classes}]{missing}")
    return "\n".join(rows)


# --- the J_real scoreboard -------------------------------------------------------------

def row_families(row: Row, binary: Path, paths: Paths,
                 presets: Path = PRESETS_DIR) -> tuple[Family, Family, list[Family]]:
    """(ours, the bar, context anchors) for one scoreboard row."""
    anchors = {f.name: f for f in anchor_families(paths, OPENH264_HARNESS,
                                                  extra=["x264-veryfast", "x265-superfast"])}
    ours = own_family(row.encoder, binary, presets)
    bar = anchors.get(row.anchor) or own_family(row.anchor, binary, presets)
    context = [anchors[n] for n in CONTEXT_ANCHORS.get(row.codec, ())]
    return ours, bar, context


def board_dir(paths: Paths, clipset: str | None = None) -> Path:
    """Stored rows: the VAL board's, or one clip set's (`board-<name>`)."""
    return paths.work / WORK_SUBDIR / (f"board-{clipset}" if clipset else "board")


def board_row(args: argparse.Namespace, row: Row, paths: Paths, runner: Runner,
              clips: list[Clip], clipset: str | None = None) -> dict | None:
    ours, bar, context = row_families(row, Path(args.binary).resolve(), paths,
                                      Path(args.presets_dir))
    label = f"{row.codec.value} {row.preset}"
    runner.measure(f"board {label} anchors", [bar, *context], clips, args.frames)
    if args.phase == "anchors":
        return None
    part = getattr(args, "part", None)
    if part:
        measure_part(runner, row, ours, [bar, *context], clips, args.frames, part)
        return None
    curves = runner.measure(f"board {label} ours", [ours, bar, *context], clips, args.frames)
    record = {"codec": row.codec.value, "preset": row.preset, "encoder": row.encoder,
              "binary": str(args.binary),
              "binary_commit": getattr(args, "binary_commit", None), "frames": args.frames, "ladder_kbps": LADDER_KBPS,
              "date": datetime.datetime.now().isoformat(timespec="minutes"),
              "headline_metrics": list(jreal.HEADLINE_METRICS), "anchors": {},
              "unreachable": {f.name: sorted(runner.unreachable.get(f.name, ()))
                              for f in (ours, bar, *context)},
              "refused": {f.name: sorted(runner.refused.get(f.name, ()))
                          for f in (ours, bar, *context) if runner.refused.get(f.name)}}
    for anchor in (bar, *context):
        per_clip = compare(curves, anchor.name, ours.name, clips, paired=anchor.own)
        add_periodic_pulse(runner, anchor, ours, one_per_class(clips, VAL_PROXY), args.frames,
                           per_clip, f"board {label} vs {anchor.name}")
        summary = jreal.summarise(per_clip)
        record["anchors"][anchor.name] = {
            "bar": anchor is bar, "per_clip": per_clip, "summary": summary,
            "verdict": jreal.objective(summary),
            "weaknesses": jreal.weaknesses(summary, per_clip)}
        print(f"== {label} vs {anchor.name}\n{format_summary(summary)}", flush=True)
    out = board_dir(paths, clipset) / f"{row.encoder}.json"
    write_atomic(out, json.dumps(record, indent=1))
    return record


def measure_part(runner: Runner, row: Row, ours: Family, anchors: list[Family],
                 clips: list[Clip], frames: int, part: str) -> None:
    """Measure (cache) part I of N of a row's clips, main and periodic-key passes.

    Each invocation stays under the 30-minute rule; a final invocation without
    --part then finds every point cached and writes the row in seconds.
    """
    index, count = (int(x) for x in part.split("/"))
    if not 1 <= index <= count:
        raise SystemExit(f"--part {part}: need 1 <= I <= N")
    chosen = balanced_parts(clips, count)[index - 1]
    label = f"board {row.codec.value} {row.preset} part {part}"
    runner.measure(f"{label} ours", [ours], chosen, frames)
    periodic = [c for c in one_per_class(clips, VAL_PROXY) if c in chosen]
    if periodic:
        runner.measure(f"{label} periodic keys", [ours, *anchors], periodic, frames,
                       PERIODIC_RUNGS, PERIODIC_KEY_SECONDS)


def board(args: argparse.Namespace) -> Path | None:
    paths = load_paths()
    wanted = set(args.rows.split(",")) if args.rows else None
    rows = [r for r in ROWS if wanted is None or r.encoder in wanted]
    runner = Runner(paths, EVALUATION_GATE, args.workers, args.over_budget)
    clipset = getattr(args, "clipset", None)
    clips = clipset_clips(clipset, paths) if clipset else natural_val(paths)
    name = clipset_path(clipset, paths).stem if clipset else None
    for row in rows:
        board_row(args, row, paths, runner, clips, name)
    if args.phase == "anchors" or getattr(args, "part", None):
        return None
    page = render_board(paths, name)
    print(f"page: {page}")
    return page


def buffer_cells(summary: dict) -> str:
    buffer = summary["compliance"]
    return (f"{buffer['test_overruns']} / {buffer['anchor_overruns']} | "
            f"{buffer['test_noncompliant']} / {buffer['anchor_noncompliant']}")


BOARD_COLUMNS = (("quality", "J full", "%"), ("quality_low", "J low", "%"),
                 ("bd_psnr_yuv", "PSNR-YUV", "%"), ("bd_vmaf_v1", "VMAF v1", "%"),
                 ("bd_ms_ssim", "MS-SSIM", "%"),
                 ("bd_worst_psnr_y", "worst-1% Y", "%"), ("bd_worst_vmaf_v1", "worst-1% VMAF v1", "%"),
                 ("d_flicker_psnr_y", "Δflicker", "dB"), ("d_first_pulse_psnr_y", "Δ|first key| (report)", "dB"),
                 ("d_periodic_pulse_psnr_y", "Δ|periodic key|", "dB"))


def render_board(paths: Paths, clipset: str | None = None) -> Path:
    """reports/realboard-<date>.md from every stored row (one-row runs accumulate)."""
    records = [json.loads(p.read_text())
               for p in sorted(board_dir(paths, clipset).glob("*.json"))]
    # Rows stored before J_real's metrics changed (2026-11-11: VMAF NEG ->
    # VMAF v1) are not comparable; they are listed for a re-run, not shown.
    stale = [r["encoder"] for r in records
             if r.get("headline_metrics") != list(jreal.HEADLINE_METRICS)]
    records = [r for r in records if r["encoder"] not in stale]
    order = {r.encoder: i for i, r in enumerate(ROWS)}
    records.sort(key=lambda r: order.get(r["encoder"], len(order)))
    now = datetime.datetime.now()
    clips_label = f"clip set {clipset}" if clipset else "VAL natural"
    lines = [f"## J_real scoreboard ({now:%Y-%m-%d %H:%M}; {clips_label}, 150 frames, CBR ladder "
             f"{'/'.join(map(str, LADDER_KBPS))} kbps at 720p30)", "",
             "Class-balanced BD-rate on achieved-rate curves (negative = fewer bits than the "
             "anchor), median clip in brackets. J = mean of PSNR-YUV, VMAF v1 and MS-SSIM BD; "
             "J_real = (J full + J low) / 2. Worst-1%: lowest 1% of P frames. Δflicker / "
             "Δ|key pulse|: ours minus anchor, PSNR-Y dB; first key (report only) from the "
             "main pass, periodic keys (guarded) from a 2 s key-period pass on one clip per "
             "class at the two lowest rungs. A point that breaks the buffer (peak > 1.5x) or "
             "overshoots its target by more than 10% + buffer/window (+30% at 150 frames, "
             "30 fps) is a rate-control failure (rate-invalid column); against external "
             "anchors it stays on the curve as a real rate-quality point, ours-vs-ours rows "
             "drop the rung from both arms; clips without a reliable BD are listed under the weaknesses. "
             "Buffer: overrun frames on the strict 1 s leaky bucket (drain = target) and points "
             f"whose bucket peaks above {jreal.NONCOMPLIANT_FULLNESS:g}x the buffer "
             "(non-compliant), ours / anchor; BD is computed without regard to them, and a "
             "weakness against a non-compliant anchor is marked. Definitions: "
             "docs/PROCEDURES.md.", "",
             "Binaries: " + "; ".join(f"{r['encoder']} {r.get('binary_commit') or 'commit unrecorded'}"
                                      f" ({r['date']})" for r in records) + ".", "",
             *([f"Stale rows (stored under the old J_real metrics; re-run them): "
                f"{', '.join(stale)}.", ""] if stale else []),
             "| Codec | Preset | Anchor | J_real | " + " | ".join(h for _, h, _ in BOARD_COLUMNS)
             + " | worse clips (J full) | guards | rate-invalid points (ours / anchor) "
             "| buffer overrun frames (ours / anchor) | non-compliant points (ours / anchor) |",
             "|---|---|---|---|" + "---|" * len(BOARD_COLUMNS) + "---|---|---|---|---|"]
    for record in records:
        dropped = record.get("unreachable", {})
        for name, cmp in record["anchors"].items():
            s, v = cmp["summary"], cmp["verdict"]
            cells = [f"{pct(s[t]['balanced'], u)} [{pct(s[t]['median'], u)}]"
                     for t, _, u in BOARD_COLUMNS]
            anchor = name if cmp["bar"] else f"{name} (context)"
            lines.append(f"| {record['codec']} | {record['preset']} | {anchor} | "
                         f"{pct(v['quality'])} | " + " | ".join(cells)
                         + f" | {s['quality']['worse']}/{s['quality']['scored']} | "
                         f"{len(v['guards'])} | {len(dropped.get(record['encoder'], []))} / "
                         f"{len(dropped.get(name, []))} | {buffer_cells(s)} |")
    lines += ["", "### Per class (bar anchor): J full / J low / worst-1% Y / Δflicker dB", ""]
    classes = sorted({k for r in records for c in r["anchors"].values()
                      for k in c["summary"]["quality"]["classes"]})
    lines += ["| Codec | Preset | " + " | ".join(classes) + " |",
              "|---|---|" + "---|" * len(classes)]
    for record in records:
        s = next(c for c in record["anchors"].values() if c["bar"])["summary"]
        cell = lambda k: " / ".join(pct(s[t]["classes"].get(k), u) for t, u in (
            ("quality", "%"), ("quality_low", "%"), ("bd_worst_psnr_y", "%"),
            ("d_flicker_psnr_y", "dB")))
        lines.append(f"| {record['codec']} | {record['preset']} | "
                     + " | ".join(cell(k) for k in classes) + " |")
    lines += ["", "### Biggest weaknesses (bar anchor; class, term, value)", ""]
    for record in records:
        cmp = next(c for c in record["anchors"].values() if c["bar"])
        worst = "; ".join(f"{k} {t} {pct(v, 'dB' if t.startswith('d_') else '%')}"
                          + (f" ({note})" if note else "")
                          for k, t, v, note in cmp["weaknesses"]) or "none"
        missing = sorted({m for t in ("quality", "quality_low") for m in cmp["summary"][t]["missing"]})
        refused = "; ".join(f"{name} refused {', '.join(points)}"
                            for name, points in record.get("refused", {}).items())
        lines.append(f"- {record['codec']} {record['preset']}: {worst}"
                     + (f" (no J for: {', '.join(missing)})" if missing else "")
                     + (f" ENCODER REFUSED: {refused}" if refused else ""))
    name = f"realboard-{clipset}-" if clipset else "realboard-"
    page = paths.data_root / "reports" / f"{name}{now:%Y-%m-%d-%H%M}.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("\n".join(lines) + "\n")
    return page


# --- the buffer-compliance check (encode only) -----------------------------------------

#: An encode-only point (no gate, no metrics) costs about this fraction of a
#: scored real-use point.
ENCODE_ONLY_FRACTION = 0.3


def buffer_replay(bench: Bench, point: Point) -> dict:
    """Overrun frames and peak fullness of one CBR point on the strict leaky bucket.

    Uses the scored real-use point when the board already measured it;
    otherwise encodes only (no gate, no metrics), caches the replay by the
    point's unscored identity and deletes the bitstream.
    """
    if bench.cached(point):
        result = json.loads(bench.result_path(point).read_text())
        return {k: result[k] for k in ("kbps", "overruns", "max_fullness")}
    identity = bench.identity(point.family, point.clip, point.label, scored=False)
    cache = bench.point_dir(point.family, point.clip) / f"buffer-{point.label}-{identity}.json"

    def replay() -> dict:
        with sibling_temp(cache, point.family.codec.bitstream_suffix) as stream, \
                sibling_temp(cache, ".rate.csv") as rate_csv:
            # Our encoder's log says which input frames were coded (holds).
            rate_log = rate_csv if point.family.own else None
            bench.encode(point, stream, rate_log)
            coded = coded_frames(rate_log) if rate_log else None
            frames = with_held_frames(probe_frames(stream, bench.paths), coded)
            sizes = [f.bytes * 8 for f in frames]
        bucket = replay_buffer(sizes, point.kbps * 1000.0, point.clip.fps, BUFFER_MS)
        return {"kbps": bench.rate_kbps_from_sizes(point, sizes), "overruns": bucket.overruns,
                "max_fullness": bucket.max_fullness}
    return get_or_create(cache, read_json, replay, json.dumps)


def compliance_check(args: argparse.Namespace) -> dict:
    """Buffer compliance of our encoders at the lowest ladder rungs, every VAL clip.

    The question it answers fast: are the remaining overruns the rate floor
    (an encoder that cannot get down to the target) or controller bursts?
    """
    paths = load_paths()
    runner = Runner(paths, EVALUATION_GATE, args.workers, args.over_budget)
    clips = natural_val(paths)
    bench = runner.bench(args.frames)
    label = "@" + args.label if args.label else ""
    families = [Family(f"nanocodec-{c}{label}", Codec(c), Path(args.binary).resolve(),
                       param_args(args.param), own=True, cqp=False)
                for c in args.codecs.split(",")]
    points = [Point(f, clip, kbps=t) for f in families for clip in clips
              for t in clip_targets(clip, LADDER_KBPS)[:args.rungs]]
    cost = sum(point_cpu_seconds(p.clip.width, p.clip.height, bench.frames)
               * REAL_COST_FACTOR * ENCODE_ONLY_FRACTION for p in points if not bench.cached(p))
    check_budget(f"compliance check ({len(points)} encodes)", cost, "fewer --codecs or --rungs",
                 args.over_budget)
    with ThreadPoolExecutor(runner.workers) as pool:
        results = list(pool.map(lambda p: buffer_replay(bench, p), points))
    report: dict = {}
    for point, result in zip(points, results):
        row = report.setdefault(point.family.name, {"overrun_frames": 0, "noncompliant": []})
        row["overrun_frames"] += result["overruns"]
        if jreal.noncompliant(result):
            row["noncompliant"].append(f"{point.clip.name}@{point.kbps}:"
                                       f"{result['max_fullness']:.1f}x")
    for name, row in report.items():
        print(f"{name:32s} overrun frames {row['overrun_frames']:5d}  non-compliant "
              f"{len(row['noncompliant'])}: {' '.join(row['noncompliant'])}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python3 -m nceval.realuse",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("board", help="the J_real scoreboard (VAL natural, 150 frames)")
    b.add_argument("--binary", default=None, help="our nanocodec (default: find_tool)")
    b.add_argument("--rows", help="comma-separated row encoders, e.g. nanocodec-h264")
    b.add_argument("--phase", choices=("all", "anchors"), default="all",
                   help="anchors: only measure (and cache) the rows' anchor curves")
    b.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    b.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    b.add_argument("--over-budget", action="store_true")
    b.add_argument("--presets-dir", default=str(PRESETS_DIR),
                   help="preset files for the rows (a frozen build: its own commit's presets/)")
    b.add_argument("--part", help="I/N: only measure (cache) part I of N class-balanced clip "
                                  "parts of each row; a run without --part then writes the rows")
    b.add_argument("--binary-commit", help="the commit --binary was built from (shown on the page)")
    b.add_argument("--clipset", help="a clip set by name (adv12: the held-out pristine "
                                     "robustness set) or JSON path, instead of VAL natural")
    r = sub.add_parser("render", help="rewrite the page from the stored rows")
    r.add_argument("--clipset", help="the clip set's board (e.g. adv12) instead of VAL's")
    c = sub.add_parser("compliance", help="encode-only buffer check at the lowest rungs")
    c.add_argument("--binary", default=None)
    c.add_argument("--codecs", default="h264,hevc,av1")
    c.add_argument("--param", action="append", default=[], help="name=value for nanocodec")
    c.add_argument("--label", help="names this parameter set in the cache and output")
    c.add_argument("--rungs", type=int, default=2, help="lowest ladder rungs to check")
    c.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    c.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    c.add_argument("--over-budget", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "render":
        paths = load_paths()
        print(render_board(paths, clipset_path(args.clipset, paths).stem if args.clipset
                           else None))
        return
    if args.binary is None:
        args.binary = str(find_tool("nanocodec", load_paths()))
    try:
        board(args) if args.command == "board" else compliance_check(args)
    except OverBudget as refusal:
        print(refusal, file=sys.stderr)
        raise


if __name__ == "__main__":
    main(sys.argv[1:])
