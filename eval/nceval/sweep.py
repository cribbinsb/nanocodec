"""Fixed-QP rate-distortion sweeps (eval/README.md "Methodology").

For each (encoder, clip) the sweep picks quantisers that land near a set of
target bitrates, rather than using one fixed QP list for every encoder:
quantiser scales differ between codecs, and the curves must overlap in
quality for BD-rate to mean anything.

Steps per (encoder, clip):
  1. Estimate bitrate on a coarse quantiser grid: encode a short excerpt per
     grid point, calibrated by one full-length encode (_estimate_rates).
  2. Fit the target bitrates to what this encoder can reach on this clip
     (fit_targets): nominal targets come from the 720p30 range scaled by
     pixel rate; they are capped at 50% of the source bitrate (test rule) and
     moved up when even the coarsest quantiser cannot get down to them (hard
     clips, short clips dominated by the intra frame).
  3. Interpolate log(bitrate) against quantiser and pick the integer
     quantiser nearest each target.
  4. For each picked quantiser: metrics, then store. Our encoder's points
     are scored from its reconstruction while it encodes and, by default,
     decoded by nothing (nceval.gatepolicy: conformance is checked when a
     change lands); anchors are scored from the first standard decoder. An
     explicit gate decodes first, as the policy says.

Curves already measured with the current fingerprint (nceval.fingerprint)
are skipped, so a sweep can be re-run at any time and only does what changed.
Bitstreams are cached under a fingerprint-named directory and kept, so
per-frame data can be recomputed without re-encoding.

Encodes run in parallel across clips; this is quality measurement, not speed
measurement, which is done separately on a pinned core (nceval.speed).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .cachefile import entry_lock, entry_lock_if_free, publish, sibling_temp, write_atomic
from .clips import Clip
from .codecs import Codec
from .conformance import DECODERS, GateResult, gate_stream
from .diskguard import DiskBudgetExceeded, check_disk
from .config import Paths
from .encoders.base import CodingStructure, EncodeJob, EncoderAdapter
from .fingerprint import curve_fingerprint, encoder_fingerprint
from .framestats import coded_at_times, probe_frames, with_held_frames
from .framepipe import FrameHashSink
from .gatepolicy import (EVALUATION_GATE, GateLevel, GatePolicy, PointCheck, combination_key,
                         gate_arguments, point_check, uniform_check)
from .metrics import (MetricSession, StreamSpec, V1_PROFILE, default_xpsnr, gpu_slot_need,
                      profile_metrics, structure_for_profile)
from .point import encode_streamed, run_point
from .results import EncodeRecord, ResultsDb
from .slots import cpu_slots, gpu_slots, low_priority
from .budget import MAX_MINUTES, check as check_budget, estimate_minutes, point_cpu_seconds
from .yuv import FRAME_HASH, YuvFormat

# Target bitrates at the primary operating point (720p30), log-spaced.
REFERENCE_PIXELS_PER_SECOND = 1280 * 720 * 30
REFERENCE_TARGETS_KBPS = (150, 210, 300, 420, 590, 800)
# Bitrate needed grows sub-linearly with pixel rate; 0.75 is a common rule of
# thumb for natural video and only has to put targets in the right region.
PIXEL_RATE_EXPONENT = 0.75
COARSE_GRID_STEP = 6
# Wide quantizer ranges (AV1 qindex 40-255) are probed at about this many
# intervals instead of every COARSE_GRID_STEP: qindex steps are ~4x finer than
# QP steps, and log-rate is smooth enough to interpolate across the gap.
COARSE_GRID_INTERVALS = 8
MIN_TARGETS_PER_CURVE = 4  # BD-rate needs at least 4 points
# A curve must span at least this rate ratio to be worth measuring.
MIN_RATE_SPAN = 2.0
# Keep the lowest target this far above the encoder's floor, so the nearest
# integer quantiser is not always the maximum one.
FLOOR_MARGIN = 1.1
ENCODE_TIMEOUT_SECONDS = 3600
# Length of the excerpt used to probe bitrate against quantiser.
PROBE_SECONDS = 2.0
IVF_FILE_HEADER_BYTES = 32
IVF_FRAME_HEADER_BYTES = 12
# Structure tag of a one-key-frame-per-clip ("long GOP") result set, and the
# intra period its fingerprint template uses (any value longer than a clip).
SINGLE_KEYFRAME_TAG = "1key"
SINGLE_KEYFRAME_TEMPLATE_FRAMES = 1 << 20


@dataclass(frozen=True)
class SweepConfig:
    """One result set: the coding structure shared by every encoder and clip.

    The intra period is given in seconds, because clips have different frame
    rates (24/25/30); each clip gets the nearest whole number of frames.
    `single_keyframe` instead codes one key frame for the whole clip (intra
    period = clip length), so errors accumulating over long P-frame chains
    show up; intra_period_seconds is then ignored.
    """
    intra_period_seconds: float
    temporal_layers: int = 1
    single_keyframe: bool = False
    max_frames: int | None = None  # None: whole clip
    workers: int = 8
    # Re-measure curves that previously failed conformance. Off by default (an
    # unchanged encoder fails the same way), but needed after an environment
    # failure such as a full disk truncating a decode.
    retry_failed: bool = False
    measurement_profile: str = V1_PROFILE
    # Scored points per curve; fewer than the nominal targets keeps the same
    # rate span with wider spacing (a cheaper sanity run). BD-rate needs 4.
    points_per_curve: int = len(REFERENCE_TARGETS_KBPS)
    # XPSNR on/off; None takes the profile's default (metrics.default_xpsnr).
    include_xpsnr: bool | None = None
    # Decoders on the measured points (nceval.gatepolicy; default none).
    gate: GatePolicy = EVALUATION_GATE
    # Run even if estimated over the wall-time limit (nceval.budget; logged).
    over_budget: bool = False

    @property
    def xpsnr(self) -> bool:
        return (default_xpsnr(self.measurement_profile) if self.include_xpsnr is None
                else self.include_xpsnr)

    @property
    def structure_tag(self) -> str:
        period = SINGLE_KEYFRAME_TAG if self.single_keyframe else f"{self.intra_period_seconds:g}s"
        return structure_for_profile(f"ip{period}-tl{self.temporal_layers}",
                                     self.measurement_profile)

    def template_structure(self) -> CodingStructure:
        """The structure of a 30 fps clip, used where no clip is at hand (fingerprints)."""
        frames = (SINGLE_KEYFRAME_TEMPLATE_FRAMES if self.single_keyframe
                  else round(self.intra_period_seconds * 30))
        return CodingStructure(intra_period_frames=frames, temporal_layers=self.temporal_layers)

    def structure_for(self, clip: Clip) -> CodingStructure:
        # Round up to whole temporal GOPs (2^(layers-1) frames), so every intra
        # period holds complete hierarchical-P groups (kvazaar requires it).
        span = 2 ** (self.temporal_layers - 1)
        frames = (clip.frames if self.single_keyframe
                  else max(1, round(self.intra_period_seconds * clip.fps)))
        return CodingStructure(intra_period_frames=math.ceil(frames / span) * span,
                               temporal_layers=self.temporal_layers)


def nominal_targets(clip: Clip) -> list[float]:
    """Target bitrates for a clip, scaled from the 720p30 range by pixel rate."""
    scale = (clip.width * clip.height * clip.fps / REFERENCE_PIXELS_PER_SECOND) ** PIXEL_RATE_EXPONENT
    return [t * scale for t in REFERENCE_TARGETS_KBPS]


# A sweep split for the wall-time limit fills each part to at most this
# fraction of it: the estimate moves with contention while a part runs.
BUDGET_PART_FILL = 0.8
# Top target at 90% of the source cap: leaves room for small parameter-induced
# rate changes and keeps the nearest QP from landing just over the cap.
CAP_HEADROOM = 0.9


def fit_targets(nominal: list[float], floor_kbps: float, cap_kbps: float) -> list[float]:
    """Fit nominal targets into [encoder floor, 50%-of-source cap].

    Keeps the nominal number of points and their span ratio where possible.
    Returns [] if less than MIN_RATE_SPAN of usable range remains.
    """
    span = nominal[-1] / nominal[0]
    lo = max(nominal[0], floor_kbps * FLOOR_MARGIN)
    hi = min(max(nominal[-1], lo * span), cap_kbps)
    if hi / lo < MIN_RATE_SPAN:
        return []
    return [float(t) for t in np.geomspace(lo, hi, len(nominal))]


class Sweep:
    def __init__(self, paths: Paths, db: ResultsDb, harness_commit: str):
        self.paths = paths
        self.db = db
        self.harness_commit = harness_commit
        self.platform = platform.machine()

    def run(self, encoders: list[EncoderAdapter], clips: list[Clip],
            cfg: SweepConfig) -> SweepSummary:
        jobs = [(enc, clip) for enc in encoders for clip in clips
                if enc.supports_temporal_layers(cfg.temporal_layers)]
        check_disk(self.paths.work)  # fail fast, before any work starts
        summary = SweepSummary()
        self._run_jobs(jobs, cfg, summary)
        return summary

    def _run_jobs(self, jobs: list, cfg: SweepConfig, summary: "SweepSummary") -> None:
        """Run jobs that fit the budget; split the rest, re-pricing each part as it starts."""
        parts = self._budget_parts(jobs, cfg)
        if len(parts) > 1:
            for index, part in enumerate(parts, 1):
                print(f"sweep part {index}/{len(parts)} ({len(part)} curves)", flush=True)
                self._run_jobs(part, cfg, summary)
            return
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            for outcome in pool.map(lambda job: self._run_curve_safely(job[0], job[1], cfg), jobs):
                summary.count(outcome)

    def _curve_cpu(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig) -> float:
        """CPU-seconds to (re)measure one curve: rate probes, calibration, points; 0 if current."""
        frames = min(clip.frames, cfg.max_frames or clip.frames)
        if self.db.curve_is_current(encoder.name, cfg.structure_tag, clip.name, frames,
                                    self.platform, self.fingerprint(encoder, cfg, clip),
                                    failures_count_as_current=not cfg.retry_failed):
            return 0.0
        probe_frames = min(frames, round(PROBE_SECONDS * clip.fps))
        probes = len(_coarse_grid(encoder.quantizer_range))
        return (probes * point_cpu_seconds(clip.width, clip.height, probe_frames)
                + (1 + cfg.points_per_curve) * point_cpu_seconds(clip.width, clip.height, frames))

    def _budget_parts(self, jobs: list, cfg: SweepConfig) -> list[list]:
        """The jobs as one priced part, or split into balanced parts to price later.

        A sweep over the limit runs as the fewest class-balanced parts that
        each fill <= BUDGET_PART_FILL of it, one after another, each re-priced
        when it starts (and split again if slots dropped), instead of being
        refused; only a single curve that cannot fit is refused.
        """
        cuts = ("fewer clips or encoders, --max-frames 150, or nceval.screen / "
                "nceval remeasure instead of a sweep")
        costs = {id(job): self._curve_cpu(job[0], job[1], cfg) for job in jobs}
        total = sum(costs.values())
        minutes = estimate_minutes(total)
        if len(jobs) <= 1 or minutes <= MAX_MINUTES:
            check_budget(f"sweep of {len(jobs)} curves", total, cuts, cfg.over_budget)
            return [jobs]
        count = min(len(jobs), math.ceil(minutes / (MAX_MINUTES * BUDGET_PART_FILL)))
        ordered = sorted(jobs, key=lambda j: (j[1].content_class, j[1].name, j[0].name))
        return [ordered[i::count] for i in range(count) if ordered[i::count]]

    def _run_curve_safely(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig) -> str:
        """One broken curve (e.g. an encoder crash) must not stop the whole sweep."""
        try:
            return self._run_curve(encoder, clip, cfg)
        except DiskBudgetExceeded:
            raise  # never carry on filling the disk
        except Exception as err:  # noqa: BLE001 - reported, counted, and the sweep continues
            print(f"ERROR {encoder.name}/{cfg.structure_tag}/{clip.name}: {err}", flush=True)
            return "error"

    def fingerprint(self, encoder: EncoderAdapter, cfg: SweepConfig, clip: Clip) -> str:
        """Identity of one curve: encoder configuration + measurement chain + clip."""
        # The template structure uses a 30 fps clip; the per-clip frame count
        # follows from the (fingerprinted) code, not from the template.
        encoder_fp = encoder_fingerprint(encoder, cfg.template_structure(), self.paths,
                                         cfg.measurement_profile)
        sampling = sampling_identity(encoder.quantizer_range, cfg.points_per_curve,
                                     cfg.xpsnr)
        if sampling:
            encoder_fp = hashlib.sha256(f"{encoder_fp}\n{sampling}".encode()).hexdigest()[:16]
        return curve_fingerprint(encoder_fp, clip)

    def _run_curve(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig) -> str:
        frames = min(clip.frames, cfg.max_frames or clip.frames)
        fingerprint = self.fingerprint(encoder, cfg, clip)
        if self.db.curve_is_current(encoder.name, cfg.structure_tag, clip.name, frames,
                                    self.platform, fingerprint,
                                    failures_count_as_current=not cfg.retry_failed):
            return "current"
        work = curve_work(self.paths, encoder, cfg, clip, fingerprint)
        # Two agents sweeping one curve (a shared anchor) would clear each
        # other's rows and share its stream files: the second waits, then
        # finds the curve current.
        with entry_lock(work):
            if self.db.curve_is_current(encoder.name, cfg.structure_tag, clip.name, frames,
                                        self.platform, fingerprint,
                                        failures_count_as_current=not cfg.retry_failed):
                return "current"
            return self._measure_curve(encoder, clip, cfg, frames, fingerprint, work)

    def _measure_curve(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig,
                       frames: int, fingerprint: str, work: Path) -> str:
        check_disk(self.paths.work)
        _remove_other_versions(work)

        rates = self._estimate_rates(encoder, clip, cfg, frames, work)
        # Headroom below the cap: the top target exactly at the cap left the
        # nearest QP just over it on many clips (3 scored points, no BD).
        targets = fit_targets(nominal_targets(clip), min(rates.values()),
                              clip.max_test_kbps * CAP_HEADROOM)
        if not targets:
            print(f"skip {encoder.name}/{clip.name}: lowest reachable rate "
                  f"{min(rates.values()):.0f} kbps leaves no usable range under the "
                  f"50% cap ({clip.max_test_kbps:.0f} kbps)")
            return "skipped"
        if len(targets) > cfg.points_per_curve:
            targets = [float(t) for t in np.geomspace(targets[0], targets[-1],
                                                      cfg.points_per_curve)]
        quantizers = pick_quantizers(rates, targets, encoder.quantizer_range)
        if len(quantizers) < MIN_TARGETS_PER_CURVE:
            print(f"warning {encoder.name}/{clip.name}: only {len(quantizers)} distinct "
                  f"quantisers for {len(targets)} targets")
        self.db.clear_curve(encoder.name, cfg.structure_tag, clip.name, frames, self.platform,
                            fingerprint)
        try:
            combination = combination_key(fingerprint, {}, clip.name)
            passed = all([self._measure_point(
                encoder, clip, cfg, q, frames, work, fingerprint,
                point_check(cfg.gate, combination, index, len(quantizers)))
                for index, q in enumerate(quantizers)])
        finally:
            # Older harness versions left YUV temporaries here; points now stream.
            for temporary in [*work.rglob("*.recon.yuv"), *work.glob("reference_*f.yuv")]:
                temporary.unlink(missing_ok=True)
        print(f"done {encoder.name}/{cfg.structure_tag}/{clip.name}: {len(quantizers)} points"
              + ("" if passed else " (CONFORMANCE FAILURES)"), flush=True)
        return "measured" if passed else "failed"

    def _estimate_rates(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig,
                        frames: int, work: Path) -> dict[int, float]:
        """Bitrate per coarse-grid quantiser for the full clip, cheaply.

        The grid is encoded on a short excerpt (the first PROBE_SECONDS), then
        calibrated with ONE full-length encode near the middle target: the
        excerpt over-weights the opening intra frame, so its rates are shifted
        (in the log domain) to match the full encode. That full encode is
        usually one of the final points anyway, so a curve costs about 7-8
        full-length encodes instead of ~15.
        """
        grid = _coarse_grid(encoder.quantizer_range)
        probe_frames = min(frames, max(1, round(PROBE_SECONDS * clip.fps)))
        # Full-length probes are final encodes already: share their cache path.
        # Short excerpts stay separate from the calibrated/full-length points.
        probe_work = work if probe_frames == frames else work / f"probe-{probe_frames}f"
        probe = {q: self._rate_probe(encoder, clip, cfg, q, probe_frames,
                                     probe_work) for q in grid}
        if probe_frames == frames:
            return probe
        nominal = nominal_targets(clip)
        middle = math.sqrt(nominal[0] * nominal[-1])
        rough = fit_targets(nominal, min(probe.values()), clip.max_test_kbps * CAP_HEADROOM)
        if rough:
            middle = math.sqrt(rough[0] * rough[-1])
        q_mid = pick_quantizers(probe, [middle], encoder.quantizer_range)[0]
        full_mid = self._rate_probe(encoder, clip, cfg, q_mid, frames, work)
        probe_mid = math.exp(np.interp(q_mid, grid, [math.log(probe[q]) for q in grid]))
        scale = full_mid / probe_mid
        return {q: r * scale for q, r in probe.items()}

    def _rate_probe(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig,
                    quantizer: int, frames: int, work: Path) -> float:
        """Rate calibration streams, checked per the gate policy's uniform check."""
        bitstream, kbps = self._encode(encoder, clip, cfg, quantizer, frames, work)
        gate = self._gate(encoder, bitstream, YuvFormat(clip.width, clip.height), frames,
                          uniform_check(cfg.gate))
        if not gate.passed:
            raise RuntimeError(f"rate probe {encoder.name}/{clip.name}/q{quantizer}: "
                               f"conformance failed: {gate.reason}")
        return kbps

    def _encode(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig, quantizer: int,
                frames: int, work: Path) -> tuple[Path, float]:
        """Encode one point (cached on disk); return (bitstream, kbps).

        The encoder's reconstruction is streamed into per-frame hashes, kept
        beside the bitstream (`_recon_hashes_path`) for the gate; no YUV is kept.
        """
        work.mkdir(parents=True, exist_ok=True)
        bitstream = _stream_path(encoder, work, quantizer, frames)
        recon_missing = encoder.verify_recon() and not _recon_hashes_path(bitstream).is_file()
        if not bitstream.is_file() or recon_missing:
            # Encode to a private temporary name and rename on success, so an
            # interrupted run never leaves a truncated stream in the cache.
            with sibling_temp(bitstream, encoder.codec.bitstream_suffix) as partial:
                job = EncodeJob(clip=clip, quantizer=quantizer,
                                structure=cfg.structure_for(clip), frames=frames,
                                bitstream=partial)
                encoder.check_job(job)
                label = f"{encoder.name}/{clip.name}/q{quantizer}"
                if encoder.verify_recon():
                    recon = FrameHashSink()
                    with cpu_slots(1):
                        encode_streamed(lambda path: encoder.command(
                            replace(job, recon=Path(path)), self.paths),
                            YuvFormat(clip.width, clip.height), [recon])
                    write_atomic(_recon_hashes_path(bitstream), json.dumps(recon.digests))
                else:
                    with cpu_slots(1):
                        _run(encoder.command(job, self.paths), label)
                publish(partial, bitstream)
        kbps = payload_bytes(bitstream, encoder.codec, frames) * 8 / 1000 / (frames / clip.fps)
        return bitstream, kbps

    def _gate(self, encoder: EncoderAdapter, bitstream: Path, fmt: YuvFormat, frames: int,
              check: PointCheck, **options) -> GateResult:
        """The gate; an anchor (no decoder of ours) checked at all gets both
        standard decoders."""
        own = encoder.own_decoder(self.paths)
        if own is None and check is not PointCheck.NONE:
            check = PointCheck.STANDARD
        return gate_stream(bitstream, encoder.codec, fmt, frames, self.paths,
                           expected_hashes=_recon_hashes(encoder, bitstream),
                           **gate_arguments(check, own), **options)

    def _measure_point(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig,
                       quantizer: int, frames: int, work: Path, fingerprint: str,
                       check: PointCheck = PointCheck.STANDARD) -> bool:
        if encoder.verify_recon() and check is PointCheck.NONE:
            return self._score_reconstruction(encoder, clip, cfg, quantizer, frames, work,
                                              fingerprint)
        bitstream, kbps = self._encode(encoder, clip, cfg, quantizer, frames, work)
        try:
            return self._check_and_measure(encoder, clip, cfg, quantizer, frames, work,
                                           fingerprint, bitstream, kbps, check)
        finally:
            # Only the bitstream stays on disk; the reconstruction hashes are
            # needed by the gate only, so a re-measure re-encodes.
            _recon_hashes_path(bitstream).unlink(missing_ok=True)

    def _score_reconstruction(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig,
                              quantizer: int, frames: int, work: Path, fingerprint: str) -> bool:
        """An undecoded point of our encoder: encode it (again, if a rate probe
        cached the stream), scoring the reconstruction as it encodes; the
        times of its coded frames place the held ones."""
        fmt = YuvFormat(clip.width, clip.height)
        spec = StreamSpec(clip.metric_reference, fmt, frames, clip.fps, self.paths)
        components = profile_metrics(spec, cfg.measurement_profile, include_xpsnr=cfg.xpsnr)
        bitstream = _stream_path(encoder, work, quantizer, frames)
        work.mkdir(parents=True, exist_ok=True)
        with sibling_temp(bitstream, encoder.codec.bitstream_suffix) as partial, \
                sibling_temp(bitstream, ".times.txt") as coded_times:
            job = EncodeJob(clip=clip, quantizer=quantizer, structure=cfg.structure_for(clip),
                            frames=frames, bitstream=partial, coded_times=coded_times)
            encoder.check_job(job)
            point = run_point(lambda recon: encoder.command(replace(job, recon=Path(recon)),
                                                            self.paths),
                              partial, encoder.codec, spec, self.paths,
                              components, standard=False)
            coded = coded_at_times(coded_times, clip.fps, frames)
            publish(partial, bitstream)
        kbps = payload_bytes(bitstream, encoder.codec, frames) * 8 / 1000 / (frames / clip.fps)
        record = EncodeRecord(
            encoder=encoder.name, structure=cfg.structure_tag, clip=clip.name,
            quantizer=quantizer, frames=frames,
            bytes=payload_bytes(bitstream, encoder.codec, frames), kbps=kbps,
            passed=True, reason="", platform=self.platform, fingerprint=fingerprint,
            harness_commit=self.harness_commit, gate_level=GateLevel.UNCHECKED.value)
        if not clip.allows_kbps(kbps):
            self.db.store(record, {})
            print(f"exclude {encoder.name}/{clip.name}/q{quantizer}: "
                  f"{kbps:.0f} kbps exceeds source cap {clip.max_test_kbps:.0f}")
            return True
        frame_info = with_held_frames(probe_frames(bitstream, self.paths, picture_types=True), coded)
        if len(frame_info) != frames:
            raise RuntimeError(f"ffprobe saw {len(frame_info)} frames in {bitstream}, "
                               f"expected {frames}")
        self.db.store(record, point.measurement.summary, frame_info, point.measurement.per_frame)
        return True

    def _check_and_measure(self, encoder: EncoderAdapter, clip: Clip, cfg: SweepConfig,
                           quantizer: int, frames: int, work: Path, fingerprint: str,
                           bitstream: Path, kbps: float,
                           check: PointCheck = PointCheck.STANDARD) -> bool:
        fmt = YuvFormat(clip.width, clip.height)
        record = EncodeRecord(
            encoder=encoder.name, structure=cfg.structure_tag, clip=clip.name,
            quantizer=quantizer, frames=frames,
            bytes=payload_bytes(bitstream, encoder.codec, frames), kbps=kbps,
            passed=False, reason="", platform=self.platform,
            fingerprint=fingerprint, harness_commit=self.harness_commit)
        if not clip.allows_kbps(kbps):
            # An estimated target can overshoot. Verify the stream, but do not
            # score or report quality outside the required operating domain.
            gate = self._gate(encoder, bitstream, fmt, frames, check)
            record = replace(record, passed=gate.passed, reason=gate.reason,
                             gate_level=gate.level)
            self.db.store(record, {})
            if not gate.passed:
                print(f"CONFORMANCE FAIL {encoder.name}/{clip.name}/q{quantizer}: {gate.reason}")
                return False
            print(f"exclude {encoder.name}/{clip.name}/q{quantizer}: "
                  f"{kbps:.0f} kbps exceeds source cap {clip.max_test_kbps:.0f}")
            return True
        # Metrics stream from the first standard decoder while the gate runs;
        # a gate that checks proves that decode equals every other decode (and
        # the reconstruction), and a failing gate discards them.
        spec = StreamSpec(clip.metric_reference, fmt, frames, clip.fps, self.paths)
        components = profile_metrics(spec, cfg.measurement_profile,
                                     include_xpsnr=cfg.xpsnr)
        # The decoders + the metrics. Slots first: an idle metric pipeline
        # waiting for slots would hold memory. GPU slots before CPU slots.
        decoders = 1 if check is PointCheck.NONE else len(DECODERS[encoder.codec]) + 1
        slots = decoders + sum(c.cpu_threads for c in components)
        with gpu_slots(gpu_slot_need(components)), cpu_slots(slots), \
                MetricSession(spec, components) as session:
            gate = self._gate(encoder, bitstream, fmt, frames, check,
                              extra_sinks=session.sinks, hold_slots=False)
            record = replace(record, passed=gate.passed, reason=gate.reason,
                             gate_level=gate.level)
            if not gate.passed:
                print(f"CONFORMANCE FAIL {encoder.name}/{clip.name}/q{quantizer}: {gate.reason}")
                self.db.store(record, {})
                return False
            measured = session.result(frames)
        frame_info = with_held_frames(probe_frames(bitstream, self.paths, picture_types=True), gate.coded)
        if len(frame_info) != frames:
            raise RuntimeError(f"ffprobe saw {len(frame_info)} frames in {bitstream}, "
                               f"expected {frames}")
        self.db.store(record, measured.summary, frame_info, measured.per_frame)
        return True


def curve_work(paths: Paths, encoder: EncoderAdapter, cfg: SweepConfig, clip: Clip,
               fingerprint: str) -> Path:
    """One curve version's stream cache; `entry_lock(curve_work(...))` guards it."""
    return paths.work / encoder.name / cfg.structure_tag / clip.name / fingerprint


def _remove_other_versions(work: Path) -> None:
    """Keep one fingerprint directory per curve: older versions are stale.

    A version another run is measuring right now (its curve lock is held) is
    left alone; it goes on a later sweep.
    """
    if work.parent.is_dir():
        for sibling in work.parent.iterdir():
            if sibling != work and sibling.is_dir():
                with entry_lock_if_free(sibling) as free:
                    if free:
                        shutil.rmtree(sibling, ignore_errors=True)


def _stream_path(encoder: EncoderAdapter, work: Path, quantizer: int, frames: int) -> Path:
    # The frame count is part of the name: a shorter encode (a rate probe,
    # an earlier --max-frames run) must never be mistaken for this one.
    return work / f"q{quantizer:03d}-{frames}f{encoder.codec.bitstream_suffix}"


def _recon_hashes_path(bitstream: Path) -> Path:
    """Where the per-frame hashes of an encoder's reconstruction are kept
    (named by yuv.FRAME_HASH: files of another hash are never compared)."""
    return bitstream.with_suffix(f".recon-{FRAME_HASH}.json")


def _recon_hashes(encoder: EncoderAdapter, bitstream: Path) -> list[str] | None:
    """The reconstruction hashes the gate must match (None for anchors)."""
    if not encoder.verify_recon():
        return None
    return json.loads(_recon_hashes_path(bitstream).read_text())


class SweepSummary:
    """Counts of curve outcomes, for the caller's progress report."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def count(self, outcome: str) -> None:
        self.counts[outcome] = self.counts.get(outcome, 0) + 1

    def __str__(self) -> str:
        return ", ".join(f"{n} {k}" for k, n in sorted(self.counts.items())) or "nothing to do"


def payload_bytes(bitstream: Path, codec: Codec, frames: int) -> int:
    """Coded bytes without container overhead (`frames`: the input frames,
    kept for callers; the IVF framing is counted from the file).

    AV1 streams are IVF-wrapped (32-byte file header + 12 bytes per frame);
    that framing is not part of the codec bitstream and would otherwise count
    against AV1 (about 3 kbps at 30 fps). Annex-B start codes are kept: they
    are part of the H.264/HEVC byte stream format.
    """
    size = bitstream.stat().st_size
    if codec in (Codec.AV1, Codec.VP9, Codec.AV2):
        size -= IVF_FILE_HEADER_BYTES + IVF_FRAME_HEADER_BYTES * ivf_frame_count(bitstream)
    return size


def ivf_frame_count(bitstream: Path) -> int:
    """Frames in an IVF file, counted from their headers: a stream of ours
    has fewer than its input when frames were held or dropped (nothing is
    written for them), so the input frame count would overstate the framing."""
    count, offset, size = 0, IVF_FILE_HEADER_BYTES, bitstream.stat().st_size
    with bitstream.open("rb") as source:
        while offset + IVF_FRAME_HEADER_BYTES <= size:
            source.seek(offset)
            offset += IVF_FRAME_HEADER_BYTES + int.from_bytes(source.read(4), "little")
            count += 1
    return count


def pick_quantizers(rates: dict[int, float], targets: list[float],
                    quantizer_range: tuple[int, int]) -> list[int]:
    """Integer quantisers whose interpolated bitrate is nearest each target."""
    qs = np.array(sorted(rates))
    log_r = np.log(np.array([rates[q] for q in qs]))
    lo, hi = quantizer_range
    candidates = np.arange(lo, hi + 1)
    # Bitrate falls with quantiser; interpolate log-rate on the coarse grid.
    predicted = np.interp(candidates, qs, log_r)
    chosen = {int(candidates[np.argmin(np.abs(predicted - math.log(t)))]) for t in targets}
    return sorted(chosen)


def sampling_identity(quantizer_range: tuple[int, int], points_per_curve: int,
                      include_xpsnr: bool = True) -> str:
    """How a curve is sampled and scored, if not the original way ("" otherwise).

    Part of the curve fingerprint, so curves sampled differently (a coarser
    probe grid, fewer points, no XPSNR) are never mixed with older cached
    ones, while curves sampled as before keep their fingerprints.
    """
    parts = [] if include_xpsnr else ["no-xpsnr"]
    parts.append(f"cap-headroom={CAP_HEADROOM}")
    step = _coarse_step(quantizer_range)
    if step != COARSE_GRID_STEP:
        parts.append(f"coarse-grid-step={step}")
    if points_per_curve != len(REFERENCE_TARGETS_KBPS):
        parts.append(f"points={points_per_curve}")
    return ",".join(parts)


def _coarse_step(quantizer_range: tuple[int, int]) -> int:
    lo, hi = quantizer_range
    return max(COARSE_GRID_STEP, math.ceil((hi - lo) / COARSE_GRID_INTERVALS))


def _coarse_grid(quantizer_range: tuple[int, int]) -> list[int]:
    lo, hi = quantizer_range
    step = _coarse_step(quantizer_range)
    grid = list(range(lo, hi + 1, step))
    if grid[-1] != hi:
        grid.append(hi)
    return grid


def _run(cmd: list[str], label: str) -> None:
    proc = subprocess.run(low_priority(cmd), stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=ENCODE_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or proc.stdout).strip().splitlines()[-3:])
        raise RuntimeError(f"encode {label} failed ({proc.returncode}): {tail}")
