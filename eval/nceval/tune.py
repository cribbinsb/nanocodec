"""Parameter tuning: the CLI, shared helpers, and the legacy fixed-QP / TRAIN-proxy search.

The current protocol (owner, 2026-11-11) is nceval.tune_pool, reached through
this module's `search`, `front`, `finalists`, `write-presets` and `agree`
subcommands: J_real (PSNR-YUV, VMAF v1, MS-SSIM every frame) vs x86
callgrind CPU, both relative to the preset tuned, on broad pool clips at
one rung each, Jetson-timed and L3-checked finalists. Read its docstring.

Everything below describes the LEGACY search (`search-legacy`,
`front-legacy`, `write-presets-legacy`, `stats`), kept only to read and
resume archived studies; its fixed TRAIN proxy is what the project overfit
to, so its results are not adoption evidence. The helpers this module also
provides (clip references, streamed points, quantizer calibration,
callgrind instruction cost, study storage) serve nceval.screen's cqp mode,
the scoreboard and nceval.tune_pool.

Legacy description:

Multi-objective tuning of the encoder's heuristic parameters.

The encoder's thresholds and multipliers interact (tuning one at a time
misses the joint optimum), and different speed/quality trade-offs want
different value sets. This module searches a chosen subset of the
registered parameters (`nanocodec params`) with Optuna's multi-objective
TPE sampler and reports the Pareto front of three objectives, all relative
to the encoder's *default* parameters:

  1. BD-rate on PSNR-YUV (6:1:1), from our own reconstruction;
  2. BD-rate on VMAF v1.0.16_3d0h (native geometry, 8-bit, every frame);
  3. instructions per P frame — a deterministic cost (callgrind), not CPU
     time, which on this shared machine varied ×0.68–×1.15 for identical
     settings. --cost-gop 300 instead optimizes amortized GOP300 instructions,
     pricing keyframe work; P/key/GOP costs are still recorded.

A trial is *feasible* only if no content class regresses by more than
`MAX_CLASS_REGRESSION` percent on either metric (Optuna constraint), so a
gain on static scenes cannot buy a loss on low light. With --noise-guard,
all TRAIN noise clips are also constrained to lose no clean-reference quality
at their best measured point or the baseline-covered fixed report bitrates.
Noise clips never enter BD-rate objectives. --noise-report measures the same
noise deltas and strict no-loss status but does not constrain them; these
exploratory Pareto candidates must not be presented as strict guard passes.
--metric-profile psnr-ms-ssim is a separate screening-only objective: native
8-bit raw [0,1] MS-SSIM replaces VMAF, which is not computed or inferred.
--keep-streams preserves gated compressed streams and hashes for later V1
shortlist scoring; raw reconstructions are removed. Completed screening
curves checkpoint before instruction cost, so a cost failure loses no curves.
--search-space FILE selects bounded numeric or categorical domains without
changing registered encoder limits; domain contents are part of study identity.
--extend-noise-range appends gated coarser noise QPs when needed to reach the
scaled200 kbps report point, stopping at the codec maximum rather than
extrapolating. The original grid remains intact; this protocol flag is recorded.
--frames N explores at a cheaper fidelity (e.g. 60 frames on a 6-clip
--train-clips subset); promoted front candidates are re-measured at full
fidelity (nceval.screen proxy-6 / TRAIN) before selection. --class-guard sets
the per-class allowance (a speed point may accept more than the default 3%).
--seed-trials enqueues known-good settings once per study, and --timeout stops
launching trials so each run stays inside the 30-minute budget (the study
resumes on the next run).

Held-out discipline (Mark, 2026-11-06), strictly:

  - The search objective uses TRAIN clips of `clips/main.json` only: a
    class-balanced proxy of `PROXY_CLIPS` (3 per class, distinct sources,
    sizes every codec accepts), `PROXY_FRAMES` frames each at the 10 s
    intra period so P frames dominate, 4 quantisers.
  - Presets are chosen from the Pareto front, then measured on the VAL
    clips through the full harness (`nceval sweep --split val` with
    `nanocodec-<codec>@candidates/<name>`; PSNR-YUV, VMAF v1, XPSNR, every
    stream gated by two standard decoders). That is the honest number,
    reported beside the train number; a wide train/val gap means overfitting.
  - The TEST split is sealed: never used in tuning or selection (the
    coordinator runs it at milestones only). `load_clips` refuses it.
  - VAL is slightly contaminated: the daily quick suite contains
    `bodycam_airport` (a VAL clip) and hand-tuning has looked at it. Treat
    VAL as optimistic; TEST stays pristine for that reason.

Mechanics: one coordinating process owns each study, with up to 8 trial workers
(project rule). SQLite uses WAL/60 s contention timeout and serialized trial
acquisition; measurements remain concurrent. Points are streamed
(nceval.point): reconstructions are scored and hashed while encoding and never
written; studies live in Optuna SQLite files and resume.

Usage (from eval/):

    python3 -m nceval.tune search --codec h264 --binary BIN --params a,b,c --trials 100
    python3 -m nceval.tune front --study NAME
    python3 -m nceval.tune write-presets --codec h264 --study NAME --trials 12:fast,37:better
    python3 -m nceval.tune cost --codec hevc --binary BIN --preset-files F1,F2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import threading
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import geometric_mean

import optuna

from . import buildinfo
from .bdrate import bd_rate
from .cachefile import write_atomic
from .clips import MAX_FRACTION_OF_SOURCE_BITRATE
from .codecs import Codec
from .config import find_tool, load_paths
from .diskguard import check_disk
from .fingerprint import clip_fingerprint, measurement_fingerprint
from .framestats import probe_frames
from .gatepolicy import (EVALUATION_GATE, GatePolicy, combination_key, file_sha256,
                         gate_arguments, point_check, uniform_check)
from . import jreal, ladder, surrogate, tune_realuse
from . import metrics as metric_api
from .metrics import StreamSpec
from .point import PointResult, run_point
from .rate_domain import bounded_quantizers, MIN_POINTS
from .report import NOISE_REPORT_KBPS_720P30, _interpolate_log_rate
from .stages import TRAIN_PROXY
from .sweep import REFERENCE_TARGETS_KBPS, nominal_targets, payload_bytes
from .yuv import YuvFormat

REPO_ROOT = Path(__file__).resolve().parents[2]
TUNING_DIR = load_paths().data_root / "tuning"

# Class-balanced TRAIN proxy (nceval.stages).
PROXY_CLIPS = TRAIN_PROXY
# 150 frames at a 10 s intra period: one key frame, 149 P frames.
PROXY_FRAMES = 150
INTRA_PERIOD_SECONDS = 10
VMAF_SUBSAMPLE = 1
PRIMARY_VMAF_METRIC = "vmaf_v1"
V1_METRIC_PROFILE = "v1"
SCREEN_METRIC_PROFILE = "psnr-ms-ssim"
# Legacy fixed-QP instruction probes, retained for historical scripts only.
# Search quality and cost QPs are calibrated per clip inside its source cap.
QUANTIZERS = {Codec.H264: (24, 28, 32, 36), Codec.HEVC: (26, 31, 36, 41), Codec.VVC: (26, 31, 36, 41),
              Codec.AV1: (100, 140, 180, 220), Codec.VP9: (100, 140, 180, 220),
              Codec.AV2: (100, 140, 180, 220)}
SUFFIX = {codec: codec.bitstream_suffix for codec in Codec}
# Deterministic cost: instructions per P frame on these clips, from two
# callgrind runs each (1 frame, and 1 + COST_P_FRAMES frames). A fixed
# camera (static_cam22) joined crowd and bodycam on 2026-11-09
# (coordinator): fixed cameras are a core edge use case, and levers that save
# CPU only on static scenes got no credit from the two moving clips.
COST_CLIPS = ("crowd_mot17_09", "bodycam_bc1", "static_cam22")
COST_P_FRAMES = 12
MAX_CLASS_REGRESSION = 3.0  # percent, per class, per metric
# Every third trial is the surrogate's (held-out rank correlation 0.80-0.96
# on studies of 47-100 trials; docs/handoff/search.md); TPE keeps the rest.
SURROGATE_EVERY = 3
MAX_WORKERS = 8


@dataclass(frozen=True)
class ClipRef:
    name: str
    yuv: Path
    width: int
    height: int
    fps: float
    frames: int
    content_class: str
    source_kbps: float
    split: str = "train"
    clean_reference: Path | None = None

    @property
    def max_test_kbps(self) -> float:
        return self.source_kbps * MAX_FRACTION_OF_SOURCE_BITRATE

    @property
    def fmt(self) -> YuvFormat:
        return YuvFormat(self.width, self.height)


def load_clips(split: str) -> list[ClipRef]:
    """Clips of one split of main.json. Refuses the sealed TEST split."""
    if split == "test":
        raise SystemExit("the TEST split is sealed: never tune or select on it")
    return [c for c in clipset_refs("main") if c.split == split]


def clipset_refs(name: str) -> list[ClipRef]:
    """Every clip of clip set `name` ($NC_DATA_ROOT/clips/<name>.json),
    TEST included: callers choose the splits they may use."""
    data_root = load_paths().data_root
    raw = json.loads((data_root / "clips" / f"{name}.json").read_text())
    return [ClipRef(c["name"], data_root / c["yuv"], c["width"], c["height"], float(c["fps"]),
                    c["frames"], c["content_class"], c["source_kbps"], c["split"],
                    data_root / c["clean_reference"] if c.get("clean_reference") else None)
            for c in raw["clips"]]


def proxy_clips(include_noise: bool = False) -> list[ClipRef]:
    by_name = {c.name: c for c in load_clips("train")}
    natural = [by_name[n] for n in PROXY_CLIPS]
    noise = [c for c in by_name.values() if c.content_class == "synthetic_noise"]
    if include_noise and (not noise or any(c.clean_reference is None for c in noise)):
        raise ValueError("every TRAIN noise clip needs a clean reference")
    return natural + noise if include_noise else natural


def encode_command(binary: Path, codec: Codec, clip: ClipRef, quantizer: int, frames: int,
                   overrides: dict[str, float], stream: Path,
                   recon: Path | None) -> list[str]:
    cmd = [str(binary), "encode", "--codec", codec.value, "--input", str(clip.yuv),
           "--width", str(clip.width), "--height", str(clip.height), "--fps", f"{clip.fps:g}",
           "--qp", str(quantizer), "--frames", str(frames),
           "--intra-period", str(round(clip.fps * INTRA_PERIOD_SECONDS)),
           "--output", str(stream)]
    if recon:
        cmd += ["--display-recon", str(recon)]
    for name, value in sorted(overrides.items()):
        cmd += ["--param", f"{name}={value!r}"]
    return cmd


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(cmd[:4])}... failed: {proc.stderr.strip()[-400:]}")


def trimmed_reference(clip: ClipRef, frames: int) -> Path:
    """First `frames` frames of a clip, cached under the tuning directory."""
    path = TUNING_DIR / "refs" / f"{clip.name}_{frames}f_{clip_fingerprint(clip)}.yuv"
    expected = frames * clip.fmt.frame_size
    try:
        if path.is_file() and path.stat().st_size == expected:
            return path
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Concurrent codecs/trials share this cache. Publish only after close;
        # each writer owns a distinct same-filesystem temporary file.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as dst:
            temporary = Path(dst.name)
            with clip.yuv.open("rb") as src:
                written = dst.write(src.read(expected))
            if written != expected:
                raise ValueError(f"{clip.name}: reference shorter than {frames} complete frames")
        temporary.replace(path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def study_vmaf_metric(study) -> str:
    metric = study.user_attrs.get("vmaf_metric", "vmaf_neg")
    if metric not in ("vmaf_v1", "vmaf_neg"):
        raise ValueError(f"unknown study VMAF metric {metric!r}")
    return metric


def vmaf_label(metric: str) -> str:
    return ("VMAF v1.0.16_3d0h (native geometry, internal)" if metric == "vmaf_v1"
            else "VMAF v0.6.1neg (legacy)")


@dataclass
class Curve:
    kbps: list[float]
    psnr: list[float]
    vmaf_neg: list[float] = field(default_factory=list)  # Legacy only; never relabelled.
    quantizers: list[int] = field(default_factory=list)
    vmaf_v1: list[float] = field(default_factory=list)
    ms_ssim: list[float] = field(default_factory=list)
    gate_levels: list[str] = field(default_factory=list)  # nceval.gatepolicy.GateLevel
    # SHA-256 of each scored point's stream (nceval.ladder identity dedup).
    stream_sha256: list[str] = field(default_factory=list)


def quality_metric(metric_profile: str) -> str:
    if metric_profile == V1_METRIC_PROFILE:
        return PRIMARY_VMAF_METRIC
    if metric_profile == SCREEN_METRIC_PROFILE:
        return "ms_ssim"
    raise ValueError(f"unknown tuning metric profile {metric_profile!r}")


def selected_train_clips(names: str | None, include_noise: bool) -> list[ClipRef]:
    if not names:
        return proxy_clips(include_noise)
    available = {c.name: c for c in load_clips("train")}
    requested = names.split(",")
    if len(set(requested)) != len(requested) or any(n not in available for n in requested):
        raise ValueError("TRAIN subset must name distinct existing TRAIN clips")
    clips = [available[n] for n in requested]
    if {c.content_class for c in clips if not c.clean_reference} != {
            "bodycam", "crowd", "lowlight", "pan", "static"}:
        raise ValueError("TRAIN subset must represent all five natural content classes")
    if not set(COST_CLIPS) <= set(requested):
        raise ValueError("TRAIN subset must include every instruction-cost clip")
    if include_noise:
        # Keep all noise families, avoiding static-only screening blind spots.
        clips += [c for c in available.values()
                  if c.clean_reference and c.name not in requested]
    return clips


def curve_record(curve: Curve, metric_profile: str = V1_METRIC_PROFILE) -> dict:
    metric = quality_metric(metric_profile)
    if len(getattr(curve, metric)) != len(curve.kbps):
        raise ValueError(f"new study curve requires explicit {metric} scores; legacy or other metric values cannot be reused")
    record = {"kbps": curve.kbps, "psnr": curve.psnr, "quantizers": curve.quantizers,
              metric: getattr(curve, metric)}
    if curve.gate_levels:
        record["gate_levels"] = curve.gate_levels
    return record


def encode_point(binary: Path, codec: Codec, clip: ClipRef, quantizer: int, frames: int,
                 overrides: dict[str, float], stream: Path, metrics=None,
                 own_decoder: Path | None = None, standard: bool = True) -> PointResult:
    """Encode one point, streaming its reconstruction into `metrics(spec)`, and gate it.

    The reconstruction never touches disk: metrics are scored against the
    clip's clean reference while it encodes, and the gate compares its
    per-frame hashes with both standard decoders (and `own_decoder`).
    `standard` False leaves the check to `own_decoder`, or with none to
    nobody (nceval.gatepolicy.gate_arguments).
    """
    metric_clip = replace(clip, yuv=clip.clean_reference) if clip.clean_reference else clip
    spec = StreamSpec(metric_clip.yuv, clip.fmt, frames, clip.fps, load_paths())
    result = run_point(lambda recon: encode_command(binary, codec, clip, quantizer, frames,
                                                    overrides, stream, recon),
                       stream, codec, spec, load_paths(),
                       metrics(spec) if metrics else None, own_decoder, standard=standard)
    if not result.gate.passed:
        raise RuntimeError(f"{clip.name}: conformance failed: {result.gate.reason}")
    return result


def point_metrics(metric_profile: str):
    """The metric components a tuning point is scored with, per profile."""
    if metric_profile == SCREEN_METRIC_PROFILE:
        return lambda spec: [metric_api.PsnrMetric(spec), *metric_api.vmaf_v1_and_ms_ssim(
            spec, vmaf_v1=False, ms_ssim=True)]
    quality_metric(metric_profile)
    return lambda spec: [metric_api.PsnrMetric(spec), *metric_api.vmaf_v1_and_ms_ssim(
        spec, vmaf_subsample=VMAF_SUBSAMPLE)]


def quantizer_bounds(codec: Codec) -> tuple[int, int]:
    if codec == Codec.VVC:
        return (10, 63)
    if codec in (Codec.VP9, Codec.AV2):
        return (1, 255)  # qindex scale, as the nanocodec adapters declare
    return (40, 255) if codec == Codec.AV1 else (10, 51)


def calibrate_quantizers(binary: Path, codec: Codec, clip: ClipRef, frames: int,
                         overrides: dict[str, float], work: Path,
                         gate: GatePolicy = EVALUATION_GATE) -> list[int]:
    """Find this clip's usable rate range from rate probes.

    Probes are checked per `gate.uniform_check`: not decoded (off), by our
    own decoder (this binary) against the reconstruction (sample), or by
    every decoder (full).
    """
    rates = {}
    def rate_at(q):
        if q in rates:
            return rates[q]
        check_disk(work)
        stream = work / f"probe_{clip.name}_{q}{SUFFIX[codec]}"
        try:
            encode_point(binary, codec, clip, q, frames, overrides, stream,
                         **gate_arguments(uniform_check(gate), binary))
            rates[q] = payload_bytes(stream, codec, frames) * 8 / 1000 / (frames / clip.fps)
            return rates[q]
        finally:
            stream.unlink(missing_ok=True)
    targets = nominal_targets(clip)
    chosen = bounded_quantizers(rate_at, quantizer_bounds(codec),
                                [targets[i] for i in (0, 2, 3, 5)], clip.max_test_kbps)
    print(f"calibrated {codec.value}/{clip.name}: cap {clip.max_test_kbps:.0f} kbps, "
          f"QPs {chosen}", flush=True)
    return chosen


def noise_report_floor(clip: ClipRef) -> float:
    return min(NOISE_REPORT_KBPS_720P30) * nominal_targets(clip)[0] / REFERENCE_TARGETS_KBPS[0]


def noise_range_status(curve: Curve, clip: ClipRef, codec: Codec) -> str:
    """Distinguish a sampled-range gap from an observed maximum-QP floor."""
    if not curve.kbps:
        return "unmeasured"
    target = noise_report_floor(clip)
    if min(curve.kbps) <= target <= max(curve.kbps):
        return "covered"
    if min(curve.kbps) > target:
        return ("max_qp_floor" if quantizer_bounds(codec)[1] in curve.quantizers
                else "insufficient_coarse_samples")
    return "insufficient_fine_samples"


def measure_curve(binary: Path, codec: Codec, clip: ClipRef, frames: int,
                  overrides: dict[str, float], work: Path,
                  quantizers: list[int] | None = None,
                  extend_noise_range: bool = False,
                  metric_profile: str = V1_METRIC_PROFILE,
                  keep_streams: bool = False,
                  fixed_grid: bool = False,
                  own_decoder: Path | None = None,
                  gate: GatePolicy = EVALUATION_GATE) -> Curve:
    """Score legal points; optionally extend noise coverage without extrapolation.

    Keep every original QP and append only coarser points, so existing point
    positions/cost probes are unchanged. At most four fixed-step extensions
    reach the codec maximum; overshooting rates still receive no quality score.

    fixed_grid measures only the supplied QPs, including single-point jobs:
    no calibration, minimum-size expansion or replacement of over-cap points.
    own_decoder adds an independent native decode/reconstruction gate before
    scoring, in addition to the existing two-standard-decoder gate.
    keep_streams also preserves over-cap packets and their gate/rate manifests
    under excluded/, with no quality scores; legal points stay under retained/.
    It returns the legal subset (which may be too small for BD-rate), and fails
    if none are legal. Noise-range extension is incompatible with a fixed grid.
    gate selects which decoders check each point (nceval.gatepolicy; our own
    decoder is `own_decoder`, else this binary). Retained streams
    (keep_streams) are certificates: always full.
    """
    if fixed_grid and (not quantizers or extend_noise_range):
        raise ValueError("fixed_grid requires explicit nonempty quantizers and no noise extension")
    check_disk(work)
    metric = quality_metric(metric_profile)
    metric_clip = replace(clip, yuv=clip.clean_reference) if clip.clean_reference else clip
    retained_identity = ({"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                          "clip_fingerprint": clip_fingerprint(clip),
                          "reference_fingerprint": clip_fingerprint(metric_clip)}
                         if keep_streams else {})
    own_decoder_sha = (hashlib.sha256(own_decoder.read_bytes()).hexdigest()
                       if keep_streams and own_decoder is not None else None)
    def retain_point(stream: Path, recon_sha256: str, point_level: str, q: int, kbps: float,
                     fields: dict[str, object], directory: str = "retained") -> None:
        retained = work / directory
        retained.mkdir(exist_ok=True)
        kept = retained / stream.name
        stream.replace(kept)
        manifest = {
            **retained_identity,
            "own_decoder_sha256": own_decoder_sha,
            "own_decoder_conformance": own_decoder is not None,
            "overrides": overrides, "clip": clip.name,
            "frames": frames, "width": clip.width, "height": clip.height,
            "fps": clip.fps, "intra_period": round(clip.fps * INTRA_PERIOD_SECONDS),
            "quantizer": q, "kbps": kbps, "source_cap_kbps": clip.max_test_kbps,
            "metric_profile": metric_profile,
            "conformance": "two decoders and encoder reconstruction passed",
            "gate_level": point_level,
            "stream_sha256": hashlib.sha256(kept.read_bytes()).hexdigest(),
            "recon_sha256": recon_sha256,
            **fields,
        }
        write_atomic(kept.with_suffix(kept.suffix + ".json"),
                     json.dumps(manifest, indent=2) + "\n")

    if keep_streams:
        gate = GatePolicy.FULL
    own_decoder = own_decoder or binary
    combination = combination_key(file_sha256(binary), overrides, clip.name)
    curve = Curve(kbps=[], psnr=[], vmaf_v1=[])
    todo = list(quantizers if quantizers is not None else
                calibrate_quantizers(binary, codec, clip, frames, overrides, work, gate))
    planned = len(todo)
    seen = set()
    max_q = quantizer_bounds(codec)[1]
    target_floor = noise_report_floor(clip) if extend_noise_range and clip.clean_reference else None
    extension_step = max(1, (max_q - max(todo, default=max_q) + 3) // 4)

    def needs_extension():
        return (target_floor is not None and bool(curve.kbps)
                and min(curve.kbps) > target_floor and max(seen) < max_q)

    while todo or (not fixed_grid and (len(curve.kbps) < MIN_POINTS or needs_extension())):
        if not todo:
            if len(curve.kbps) >= MIN_POINTS and needs_extension():
                todo.append(min(max_q, max(seen) + extension_step))
            else:
                if not curve.quantizers:
                    break
                missing = [q for q in range(min(curve.quantizers), max_q + 1) if q not in seen]
                if not missing:
                    break
                todo.append(missing[0])
        q = todo.pop(0)
        if q in seen:
            continue
        seen.add(q)
        stream = work / f"{clip.name}_q{q}{SUFFIX[codec]}"
        check = point_check(gate, combination, len(seen) - 1, max(planned, len(seen)))
        try:
            point = encode_point(binary, codec, clip, q, frames, overrides, stream,
                                 point_metrics(metric_profile),
                                 **gate_arguments(check, own_decoder))
            kbps = payload_bytes(stream, codec, frames) * 8 / 1000 / (frames / clip.fps)
            if kbps > clip.max_test_kbps:
                if keep_streams:
                    # Preserve the rate and gates without scoring an illegal
                    # point or forcing a later diagnostic to encode it again.
                    retain_point(stream, point.recon_sha256, point.gate.level, q, kbps,
                                 {"excluded": True, "reason": "source_bitrate_ceiling"},
                                 "excluded")
                if not fixed_grid and q < max_q:
                    todo.insert(0, q + 1)
                continue
            curve.kbps.append(kbps)
            curve.psnr.append(point.measurement.summary["psnr_yuv"])
            score = point.measurement.summary[metric]
            getattr(curve, metric).append(float(score))
            curve.quantizers.append(q)
            curve.gate_levels.append(point.gate.level)
            curve.stream_sha256.append(hashlib.sha256(stream.read_bytes()).hexdigest())
            if keep_streams:
                retain_point(stream, point.recon_sha256, point.gate.level, q, kbps,
                             {"psnr_yuv": curve.psnr[-1], metric: float(score)})
        finally:
            stream.unlink(missing_ok=True)
    if fixed_grid and not curve.kbps:
        raise ValueError(f"{clip.name}: no requested points below source ceiling")
    if not fixed_grid and len(curve.kbps) < MIN_POINTS:
        raise ValueError(f"{clip.name}: fewer than four scored points below source ceiling")
    order = sorted(range(len(curve.quantizers)), key=curve.quantizers.__getitem__)
    for name in ("quantizers", "kbps", "psnr", metric, "gate_levels", "stream_sha256"):
        values = getattr(curve, name)
        setattr(curve, name, [values[i] for i in order])
    return curve


@dataclass(frozen=True)
class FixedGridJob:
    """One clip's explicit frame prefix and immutable quantizer grid."""
    clip: ClipRef
    frames: int
    quantizers: tuple[int, ...]


def measure_fixed_grid(binary: Path, codec: Codec, jobs: list[FixedGridJob],
                       overrides: dict[str, float], work: Path, *, workers: int = 1,
                       metric_profile: str = V1_METRIC_PROFILE,
                       keep_streams: bool = True,
                       own_decoder: Path | None = None) -> dict[str, Curve]:
    """Measure fixed candidates concurrently, with isolated per-clip directories.

    No calibration, cost probes, or implicit QP extension. Each completed clip
    checkpoints curve.json beside its retained streams/provenance. The caller
    owns resume selection and must use an identity-scoped work directory;
    existing files are not assumed to be current. On
    failure stop submitting clips, cancel pending work, wait for active clips
    to finish their cleanup, and propagate the original exception. Running
    subprocesses are not killed. Metric RAM leasing remains globally bounded.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("fixed-grid workers must be an integer in 1..8")
    names = [job.clip.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("fixed-grid clip names must be unique")
    for job in jobs:
        if (not job.clip.name or job.clip.name in (".", "..")
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
                       for c in job.clip.name)):
            raise ValueError("fixed-grid clip names must be safe file names")
        if (isinstance(job.frames, bool) or not isinstance(job.frames, int)
                or not 1 <= job.frames <= job.clip.frames):
            raise ValueError(f"{job.clip.name}: invalid fixed-grid frame count")
        if (not job.quantizers or len(set(job.quantizers)) != len(job.quantizers)
                or any(isinstance(q, bool) or not isinstance(q, int)
                       or not 0 <= q <= quantizer_bounds(codec)[1] for q in job.quantizers)):
            raise ValueError(f"{job.clip.name}: invalid fixed-grid quantizers")
    quality_metric(metric_profile)
    overrides = dict(overrides)
    identity = evaluation_identity(binary, metric_profile)
    own_decoder_sha = (hashlib.sha256(own_decoder.read_bytes()).hexdigest()
                       if own_decoder is not None else None)
    identities = []
    for job in jobs:
        reference = (replace(job.clip, yuv=job.clip.clean_reference)
                     if job.clip.clean_reference else job.clip)
        identities.append({"evaluation": identity, "codec": codec.value,
                           "overrides": overrides,
                           "clip": clip_fingerprint(job.clip),
                           "reference": clip_fingerprint(reference),
                           "frames": job.frames, "quantizers": list(job.quantizers),
                           "own_decoder_sha256": own_decoder_sha,
                           "keep_streams": keep_streams})
    work.mkdir(parents=True, exist_ok=True)

    def measure(index: int) -> Curve:
        job = jobs[index]
        # Index-based paths cannot escape work even for externally supplied names.
        directory = work / f"clip-{index:04d}"
        directory.mkdir(exist_ok=True)
        curve = measure_curve(binary, codec, job.clip, job.frames, dict(overrides),
                              directory, list(job.quantizers), fixed_grid=True,
                              metric_profile=metric_profile, keep_streams=keep_streams,
                              own_decoder=own_decoder)
        checkpoint = directory / "curve.json"
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as output:
            temporary = Path(output.name)
            try:
                json.dump({"identity": identities[index],
                           "clip": job.clip.name, "frames": job.frames,
                           "requested_quantizers": list(job.quantizers),
                           "metric_profile": metric_profile,
                           "curve": curve_record(curve, metric_profile)}, output, indent=2)
                output.write("\n")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
        return curve

    results = {}
    iterator = iter(range(len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        try:
            for index in iterator:
                pending[pool.submit(measure, index)] = index
                if len(pending) == workers:
                    break
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Check the entire completed batch before scheduling more work.
                completed = [(pending[future], future.result()) for future in done]
                for future in done:
                    del pending[future]
                for index, curve in completed:
                    results[jobs[index].clip.name] = curve
                for _ in completed:
                    index = next(iterator, None)
                    if index is None:
                        break
                    pending[pool.submit(measure, index)] = index
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return {name: results[name] for name in names}


#: Callgrind writes one profile after each call of this symbol: one per coded frame.
FRAME_SYMBOL = "*Encoder*encode_frame"


def instructions(binary: Path, codec: Codec, clip: ClipRef, frames: int,
                 overrides: dict[str, float], work: Path, quantizer: int | None = None,
                 enforce_rate_cap: bool = False, per_frame: bool = False) -> int | list[int]:
    """Instructions executed encoding `frames` frames (callgrind; deterministic).

    `per_frame` returns each coded frame's count from the same single run
    (`--dump-after` FRAME_SYMBOL); the first includes the encoder's start-up.
    """
    out = work / f"cg.{clip.name}.{frames}"
    stream = work / f"cost_{clip.name}{SUFFIX[codec]}"
    q = quantizer if quantizer is not None else QUANTIZERS[codec][2]
    certificate = None
    try:
        if enforce_rate_cap:
            encode_point(binary, codec, clip, q, frames, overrides, stream, standard=False)
            # The key frame is an amortized cost component, not an RD point:
            # over a short probe it would dominate the rate (1 key + 6 P frames
            # exceeded bodycam_bc1's cap at the curve's own QP), so the cap
            # applies to the P frames' rate, as it does on a 10 s GOP curve.
            sizes = [f.bytes for f in probe_frames(stream, load_paths())]
            rate = sum(sizes[1:]) * 8 / 1000 / ((frames - 1) / clip.fps) if frames > 1 else 0.0
            if rate > clip.max_test_kbps:
                raise ValueError(f"{clip.name}: cost probe P frames exceed the source cap")
            certificate = hashlib.sha256(stream.read_bytes()).hexdigest()
        dump = [f"--dump-after={FRAME_SYMBOL}"] if per_frame else []
        run(["valgrind", "--tool=callgrind", f"--callgrind-out-file={out}", *dump,
             *encode_command(binary, codec, clip, quantizer if quantizer is not None else QUANTIZERS[codec][2], frames, overrides,
                             stream, None)])
        if certificate is not None and hashlib.sha256(stream.read_bytes()).hexdigest() != certificate:
            raise ValueError("profiled stream differs from the rate-checked probe's")
        if per_frame:
            return [_callgrind_total(out.with_name(f"{out.name}.{i}")) for i in range(1, frames + 1)]
        return _callgrind_total(out)
    finally:
        for f in [out, *out.parent.glob(f"{out.name}.*")]:
            f.unlink(missing_ok=True)
        stream.unlink(missing_ok=True)


def _callgrind_total(path: Path) -> int:
    for line in path.read_text().splitlines():
        if line.startswith(("summary:", "totals:")):
            return int(line.split()[1])
    raise RuntimeError(f"no instruction total in {path}")


@dataclass
class InstructionCost:
    """Millions of instructions per P/key frame: the geometric mean over COST_CLIPS.

    A geometric mean, so a cost ratio between two settings is the geometric
    mean of the per-clip ratios: every clip weighs the same in percent terms
    (an arithmetic mean let the costliest clip dominate; static P frames cost
    about half of crowd ones).
    """
    p: float
    key: float

    def per_frame(self, intra_period: int) -> float:
        """Amortise the measured key frame over a GOP of `intra_period`."""
        return (self.key + (intra_period - 1) * self.p) / intra_period


def instruction_cost(binary: Path, codec: Codec, overrides: dict[str, float],
                     work: Path, quantizers: dict[str, int] | None = None,
                     p_frames: int | None = None) -> InstructionCost:
    """Separate I/P costs from one gated callgrind run per cost clip (per-frame profiles)."""
    strict_cost = p_frames is not None
    p_frames = COST_P_FRAMES if p_frames is None else p_frames
    if p_frames < 1:
        raise ValueError("cost P-frame count must be positive")
    by_name = {c.name: c for c in load_clips("train")}
    per_frame = []
    keys = []
    for name in COST_CLIPS:
        clip = by_name[name]
        q = quantizers[name] if quantizers else None
        options = {"enforce_rate_cap": True} if strict_cost else {}
        # One run, one profile per frame: the key frame (with start-up, as a
        # one-frame run counts it) and the P frames after it.
        counts = instructions(binary, codec, clip, 1 + p_frames, overrides, work, q,
                              per_frame=True, **options)
        per_frame.append(sum(counts[1:]) / p_frames / 1e6)
        keys.append(counts[0] / 1e6)
    return InstructionCost(geometric_mean(per_frame), geometric_mean(keys))


def p_frame_instructions(binary: Path, codec: Codec, overrides: dict[str, float],
                         work: Path) -> float:
    """Millions of instructions per P frame (compatibility for cost scripts)."""
    return instruction_cost(binary, codec, overrides, work).p


@dataclass
class Evaluation:
    curves: dict[str, Curve]
    p_instructions: float
    key_instructions: float = 0.0


def evaluation_identity(binary: Path, metric_profile: str) -> dict:
    """Compute once per evaluation, shared by its per-curve checkpoints."""
    return {"binary": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "measurement": measurement_fingerprint(),
            "protocol": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metric_implementation": hashlib.sha256(Path(metric_api.__file__).read_bytes()).hexdigest(),
            "metric_profile": metric_profile,
            "secondary_representation": ("raw[0,1]; native yuv420p8; every frame"
                                         if metric_profile == SCREEN_METRIC_PROFILE else "VMAF v1"),
            "ms_ssim_binary": (hashlib.sha256(find_tool("vmaf", load_paths()).read_bytes()).hexdigest()
                               if metric_profile == SCREEN_METRIC_PROFILE else None)}


def checkpointed_curve(binary: Path, codec: Codec, clip: ClipRef,
                       overrides: dict[str, float], work: Path,
                       quantizers: list[int] | None = None, *,
                       extend_noise_range: bool = False,
                       metric_profile: str = V1_METRIC_PROFILE,
                       keep_streams: bool = False,
                       identity: dict | None = None,
                       gate: GatePolicy = EVALUATION_GATE,
                       frames: int = PROXY_FRAMES,
                       fixed_grid: bool = False) -> Curve:
    """Existing curve measurement with an atomic, independently reusable checkpoint.

    An anchor adapter may call this helper in parallel for distinct clips.
    """
    work.mkdir(parents=True, exist_ok=True)
    checkpoints = work / "curve-checkpoints"
    checkpoints.mkdir(exist_ok=True)
    checkpoint = checkpoints / f"{clip.name}.json"
    reference = replace(clip, yuv=clip.clean_reference) if clip.clean_reference else clip
    key = {"evaluation": identity or evaluation_identity(binary, metric_profile),
           "clip": clip_fingerprint(clip), "reference": clip_fingerprint(reference),
           "overrides": overrides, "frames": frames, "quantizers": quantizers,
           "metric_profile": metric_profile, "keep_streams": keep_streams,
           "extend_noise_range": extend_noise_range,
           **({"fixed_grid": True} if fixed_grid else {})}
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text())
        if cached["identity"] != key:
            raise ValueError("curve checkpoint identity changed")
        curve = Curve(**cached["curve"])
        curve_record(curve, metric_profile)
        return curve
    curve = measure_curve(binary, codec, clip, frames, overrides, work, quantizers,
                          extend_noise_range=extend_noise_range,
                          metric_profile=metric_profile, keep_streams=keep_streams, gate=gate,
                          fixed_grid=fixed_grid)
    write_atomic(checkpoint, json.dumps({"identity": key,
        "curve": curve_record(curve, metric_profile)}, indent=2) + "\n")
    return curve


def evaluate(binary: Path, codec: Codec, overrides: dict[str, float],
             work: Path, quantizers: dict[str, list[int]] | None = None,
             clips: list[ClipRef] | None = None,
             extend_noise_range: bool = False,
             metric_profile: str = V1_METRIC_PROFILE,
             keep_streams: bool = False,
             cost_p_frames: int | None = None,
             gate: GatePolicy = EVALUATION_GATE,
             frames: int = PROXY_FRAMES,
             fixed_grid: bool = False,
             cost: InstructionCost | None = None,
             clip_workers: int = 1,
             checkpoint: bool = False,
             cost_quantizers: dict[str, int] | None = None) -> Evaluation:
    """Curves on `clips` (`frames` each: lower is a cheaper, coarser fidelity) and the cost.

    `fixed_grid` measures exactly the given quantizers (a ladder rung's
    subset); a known `cost` (an earlier rung measured it: it does not depend
    on the curves' frames) is not measured again. `clip_workers` measures
    clips concurrently and `checkpoint` keeps each finished curve, so a
    study's anchor (one evaluation, otherwise serial) fits a 30-minute slice
    and resumes where a slice stopped. `cost_quantizers` sets the cost probe's
    QP per cost clip (default: each curve's third QP, the anchor grid's on a
    full curve; a ladder rung's 3-QP subset would put a coarser QP there).
    """
    work.mkdir(parents=True, exist_ok=True)
    try:
        identity = (evaluation_identity(binary, metric_profile)
                    if checkpoint or metric_profile != V1_METRIC_PROFILE or keep_streams else None)

        def curve_of(c: ClipRef) -> Curve:
            qs = quantizers.get(c.name) if quantizers else None
            if identity is None:
                return measure_curve(binary, codec, c, frames, overrides, work,
                                     qs, extend_noise_range=extend_noise_range, gate=gate,
                                     fixed_grid=fixed_grid)
            return checkpointed_curve(binary, codec, c, overrides, work, qs,
                extend_noise_range=extend_noise_range, metric_profile=metric_profile,
                keep_streams=keep_streams, identity=identity, gate=gate, frames=frames,
                fixed_grid=fixed_grid)
        selected = clips if clips is not None else proxy_clips()
        with ThreadPoolExecutor(max(1, min(clip_workers, MAX_WORKERS))) as pool:
            curves = dict(zip((c.name for c in selected), pool.map(curve_of, selected)))
        if cost is None:
            cost_qs = cost_quantizers or {name: curves[name].quantizers[2] for name in COST_CLIPS}
            cost = (instruction_cost(binary, codec, overrides, work, cost_qs)
                    if cost_p_frames is None else
                    instruction_cost(binary, codec, overrides, work, cost_qs,
                                     p_frames=cost_p_frames))
        return Evaluation(curves, cost.p, cost.key)
    finally:
        for f in work.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
        if not any(work.iterdir()):
            work.rmdir()


def rung_quantizers(anchor: dict[str, Curve], rung: "ladder.Rung") -> dict[str, list[int]]:
    """Each clip's anchor quantizers at the rung's indices (all four on the last rung)."""
    return {name: [c.quantizers[i] for i in rung.qp_indices] for name, c in anchor.items()}


def bd_by_clip(test: dict[str, Curve], anchor: dict[str, Curve],
               metric: str, min_points: int = MIN_POINTS) -> dict[str, float]:
    out = {}
    for name, a in anchor.items():
        t = test[name]
        r = bd_rate(a.kbps, getattr(a, metric), t.kbps, getattr(t, metric), min_points)
        if r.bd_rate_percent is None or not r.reliable:
            # Silently excluding this clip would also weaken its class guard
            # and could promote a trial with an unmeasured regression.
            raise ValueError(f"{name}: cannot compare {metric}: insufficient quality overlap")
        out[name] = r.bd_rate_percent
    return out


def noise_deltas(test: dict[str, Curve], anchor: dict[str, Curve],
                 clips: list[ClipRef], metric: str) -> dict[str, dict[str, float]]:
    """Clean-reference quality deltas, never BD-rate on nonmonotonic noise.

    Require candidate coverage wherever the baseline covers a fixed rate.
    Best quality is the best measured point, as in the official report.
    """
    out = {}
    for clip in clips:
        if clip.content_class != "synthetic_noise":
            continue
        a, t = anchor[clip.name], test[clip.name]
        ap = sorted(zip(a.kbps, getattr(a, metric)))
        tp = sorted(zip(t.kbps, getattr(t, metric)))
        if not ap or not tp:
            raise ValueError(f"{clip.name}: missing noise curve")
        deltas = {"best": max(q for _, q in tp) - max(q for _, q in ap)}
        scale = nominal_targets(clip)[0] / REFERENCE_TARGETS_KBPS[0]
        for nominal in NOISE_REPORT_KBPS_720P30:
            aq = _interpolate_log_rate(ap, nominal * scale)
            if aq is None:
                continue
            tq = _interpolate_log_rate(tp, nominal * scale)
            if tq is None:
                raise ValueError(f"{clip.name}: noise curve does not cover {nominal:g} kbps")
            deltas[f"at_{nominal:g}_kbps"] = tq - aq
        out[clip.name] = deltas
    return out


def class_means(per_clip: dict[str, float]) -> dict[str, float]:
    classes = {c.name: c.content_class for c in proxy_clips()}
    grouped: dict[str, list[float]] = {}
    for name, value in per_clip.items():
        if value == value:
            grouped.setdefault(classes[name], []).append(value)
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def mean(values) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def cpu_objective_label(gop_period: int) -> str:
    return f"GOP{gop_period}-instr" if gop_period else "P-instr"


def cpu_objective_cost(cost: InstructionCost, gop_period: int) -> float:
    """Zero retains the historical P objective; a positive period prices keys."""
    if gop_period < 0:
        raise ValueError("cost GOP period must be zero (P) or a positive integer")
    return cost.per_frame(gop_period) if gop_period else cost.p


# ------------------------------------------------------------------ search


#: Prefix of a parameter's non-base frame value (nc-encoder `NONBASE_PREFIX`).
NONBASE_PREFIX = "nonbase."


def specs_of(binary: Path, codec: Codec | None = None) -> dict[str, tuple[float, float, float]]:
    """name -> (default, min, max), from `nanocodec params`.

    With `codec`, defaults are what that codec encodes with (its backend's
    DEFAULT_PARAMS over the engine defaults), so the enqueued default trial
    and seed fill-ins really are the default encoder.
    """
    command = [str(binary), "params"] + (["--codec", codec.value] if codec else [])
    out = subprocess.run(command, capture_output=True, text=True, check=True)
    specs = {}
    for line in out.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, default, rng = line.split("  ")[:3]
        lo, hi = rng.strip("[]").split(", ")
        specs[name] = (float(default), float(lo), float(hi))
    # Non-base frame values (`nonbase.<name>`, nc-encoder `frame_role.rs`)
    # share their parameter's default and range; the encoder rejects the
    # ones that are not re-read per frame.
    specs.update({NONBASE_PREFIX + name: spec for name, spec in list(specs.items())})
    return specs


def storage_of(study: str) -> str:
    return f"sqlite:///{TUNING_DIR / (study + '.sqlite')}"


#: Running trials beat every HEARTBEAT_SECONDS; one silent for STALE_SECONDS
#: (its slice was stopped by --timeout's process limit) is failed and
#: re-enqueued with the same parameters, up to TRIAL_RETRIES times, so the
#: next slice finishes it (real-use points are cached by parameters).
HEARTBEAT_SECONDS = 60
STALE_SECONDS = 300
TRIAL_RETRIES = 2


def study_storage(study: str) -> optuna.storages.RDBStorage:
    """WAL lets reports read during trial writes; wait for brief writer contention."""
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(TUNING_DIR / (study + ".sqlite"), timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    return optuna.storages.RDBStorage(
        storage_of(study), engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=HEARTBEAT_SECONDS, grace_period=STALE_SECONDS,
        heartbeat_stale_trial_callback=optuna.storages.RetryFailedTrialCallback(
            max_retry=TRIAL_RETRIES))


def serialize_trial_acquisition(study: optuna.Study) -> optuna.Study:
    """Acquire queued/new trials atomically within this process's worker pool.

    SQLite ignores SELECT FOR UPDATE. Concurrent queued-trial readers can
    therefore all claim the same WAITING seed, even with WAL and a timeout.
    Lock the complete ask operation, including trial creation, not evaluation.
    Each trial still runs concurrently after it has its own ID and workdir.
    """
    lock = threading.RLock()
    original_ask = study.ask

    def ask(*args, **kwargs):
        with lock:
            return original_ask(*args, **kwargs)

    study.ask = ask
    return study


def constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
    """≤ 0 when feasible: class allowances and optional clean-reference noise guards."""
    return trial.user_attrs.get("constraint", [0.0, 0.0])


def search_space_of(names: list[str], specs: dict[str, tuple[float, float, float]],
                    path: str | None = None) -> dict[str, dict]:
    """Validated bounded float or numeric categorical domains, in study identity.

    An explicit file must cover exactly the selected names so a misspelling or
    missing bound cannot silently fall back to a broad registered range.
    """
    raw = json.loads(Path(path).read_text()) if path else {
        n: {"low": specs[n][1], "high": specs[n][2]} for n in names}
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise ValueError("search space must cover exactly the selected parameters")
    domains = {}
    for name in names:
        domain = raw[name]
        if not isinstance(domain, dict):
            raise ValueError(f"{name}: search domain must be an object")
        _, registered_low, registered_high = specs[name]

        def valid(value):
            return (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and registered_low <= value <= registered_high)

        if "choices" in domain:
            choices = domain["choices"]
            if (set(domain) != {"choices"} or not isinstance(choices, list) or not choices
                    or any(not valid(v) for v in choices) or len(set(choices)) != len(choices)):
                raise ValueError(f"{name}: choices must be distinct finite registered values")
            domains[name] = {"choices": choices}
        else:
            low, high, log = domain.get("low"), domain.get("high"), domain.get("log", False)
            if (set(domain) - {"low", "high", "log"} or not valid(low) or not valid(high)
                    or low > high or not isinstance(log, bool) or (log and low <= 0)):
                raise ValueError(f"{name}: invalid bounded float domain")
            domains[name] = {"low": low, "high": high, "log": log}
    return domains


def suggest_parameters(trial: optuna.Trial, domains: dict[str, dict]) -> dict[str, float]:
    return {name: (trial.suggest_categorical(name, domain["choices"])
                   if "choices" in domain else trial.suggest_float(name, **domain))
            for name, domain in domains.items()}


def domain_contains(domain: dict, value: float) -> bool:
    return value in domain["choices"] if "choices" in domain else domain["low"] <= value <= domain["high"]


def seed_trials(path: str | None, names: list[str], specs: dict[str, tuple[float, float, float]],
                domains: dict[str, dict]) -> list[dict[str, float]]:
    """Known-good settings to enqueue first (JSON list of {name: value}).

    Unlisted parameters take their defaults, so a seed found with a smaller
    parameter set still names a complete point of this space.
    """
    if not path:
        return []
    seeds = json.loads(Path(path).read_text())
    out = []
    for seed in seeds:
        unknown = set(seed) - set(names)
        if unknown:
            raise SystemExit(f"seed names parameters outside --params: {sorted(unknown)}")
        point = {n: float(seed.get(n, specs[n][0])) for n in names}
        if not all(domain_contains(domains[n], v) for n, v in point.items()):
            raise SystemExit(f"seed outside the search space: {seed}")
        out.append(point)
    return out


WARM_START_ATTR = "warm_start"
CQP_OBJECTIVE = "cqp"
REALUSE_OBJECTIVE = "realuse"


def distribution_of(domain: dict) -> optuna.distributions.BaseDistribution:
    if "choices" in domain:
        return optuna.distributions.CategoricalDistribution(domain["choices"])
    return optuna.distributions.FloatDistribution(domain["low"], domain["high"], log=domain["log"])


def warm_start(study: optuna.Study, sources: list[str], names: list[str],
               specs: dict[str, tuple[float, float, float]], domains: dict[str, dict]) -> int:
    """Copy earlier studies' finished trials into `study` as prior knowledge.

    TPE and the surrogate learn from them as from any finished trial, but
    they were measured with another build or protocol, so they are marked
    (`WARM_START_ATTR`) and never enter a front or a preset. Parameters the
    source did not search take their defaults; trials outside this space
    are skipped.
    """
    added = 0
    distributions = {n: distribution_of(domains[n]) for n in names}
    for source in sources:
        old = optuna.load_study(study_name=source, storage=study_storage(source))
        for t in surrogate.training_trials(old.trials):
            if t.user_attrs.get(WARM_START_ATTR) or len(t.values) != 3 or set(t.params) - set(names):
                continue
            params = {n: t.params.get(n, specs[n][0]) for n in names}
            if not all(domain_contains(domains[n], v) for n, v in params.items()):
                continue
            study.add_trial(optuna.trial.create_trial(
                params=params, distributions=distributions, values=t.values,
                user_attrs={WARM_START_ATTR: f"{source}#{t.number}",
                            "constraint": t.user_attrs.get("constraint", [0.0, 0.0])}))
            added += 1
    return added


def search(args: argparse.Namespace) -> None:
    codec = Codec(args.codec)
    binary = Path(args.binary).resolve()
    specs = specs_of(binary, codec)
    names = args.params.split(",")
    for n in names:
        if n not in specs:
            raise SystemExit(f"unknown parameter {n}")
    try:
        domains = search_space_of(names, specs, getattr(args, "search_space", None))
    except (ValueError, OSError) as error:
        raise SystemExit(f"invalid search space: {error}") from error
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    study_name = args.study or f"{codec.value}-{int(time.time())}"

    cost_gop = getattr(args, "cost_gop", 0)
    if cost_gop < 0:
        raise SystemExit("--cost-gop must be zero (P) or a positive integer")
    strict_noise_guard = getattr(args, "noise_guard", False)
    include_noise = strict_noise_guard or getattr(args, "noise_report", False)
    extend_noise_range = getattr(args, "extend_noise_range", False)
    metric_profile = getattr(args, "metric_profile", V1_METRIC_PROFILE)
    secondary_metric = quality_metric(metric_profile)
    keep_streams = getattr(args, "keep_streams", False)
    subset = getattr(args, "train_clips", None)
    clips = (selected_train_clips(subset, include_noise) if subset
             else proxy_clips(True) if include_noise else proxy_clips())
    cost_p_frames = getattr(args, "cost_p_frames", COST_P_FRAMES)
    if cost_p_frames < 1:
        raise SystemExit("--cost-p-frames must be positive")
    grid_path = getattr(args, "quantizer_grid", None)
    grid = json.loads(Path(grid_path).read_text()) if grid_path else None
    if grid is not None:
        low, high = quantizer_bounds(codec)
        if not isinstance(grid, dict) or any(c.name not in grid or not isinstance(grid[c.name], list)
                or len(grid[c.name]) < MIN_POINTS or len(set(grid[c.name])) != len(grid[c.name])
                or any(not isinstance(q, int) or isinstance(q, bool) or not low <= q <= high for q in grid[c.name])
                for c in clips):
            raise SystemExit("quantizer grid must supply at least four distinct legal integer QPs for every selected clip")
        grid = {c.name: grid[c.name] for c in clips}
    evaluation_options = ({"metric_profile": metric_profile, "keep_streams": keep_streams,
                           "gate": GatePolicy(getattr(args, "gate", EVALUATION_GATE.value))}
                          if metric_profile != V1_METRIC_PROFILE or keep_streams else {})
    if cost_p_frames != COST_P_FRAMES:
        evaluation_options["cost_p_frames"] = cost_p_frames
    # Multi-fidelity: a shorter proxy explores cheaply; promoted candidates
    # are re-measured at PROXY_FRAMES (or by nceval.screen) before selection.
    objective_kind = getattr(args, "objective", CQP_OBJECTIVE)
    realuse_study = objective_kind == REALUSE_OBJECTIVE
    frames = getattr(args, "frames", None) or (tune_realuse.DEFAULT_FRAMES if realuse_study
                                               else PROXY_FRAMES)
    if frames != PROXY_FRAMES and not realuse_study:
        evaluation_options["frames"] = frames
    class_guard = getattr(args, "class_guard", MAX_CLASS_REGRESSION)
    real_targets = (tuple(int(i) for i in args.real_targets.split(","))
                    if getattr(args, "real_targets", None) else tune_realuse.DEFAULT_TARGETS)
    flicker_objective = getattr(args, "flicker_objective", False)
    flicker_anchor = getattr(args, "flicker_anchor", None)
    if realuse_study and (include_noise or metric_profile != V1_METRIC_PROFILE):
        raise SystemExit("--objective realuse takes no noise options or --metric-profile")
    if (flicker_objective or flicker_anchor) and not realuse_study:
        raise SystemExit("--flicker-objective and --flicker-anchor need --objective realuse")
    real_fidelities = [tune_realuse.Fidelity(frames, real_targets)]
    real_rungs = None
    if realuse_study and getattr(args, "ladder", None):
        try:
            real_rungs = ladder.parse_ladder(args.ladder, frames, tune_realuse.REAL_RUNG_SIGMAS,
                                             tune_realuse.TARGET_SUBSETS, real_targets)
        except ValueError as error:
            raise SystemExit(f"invalid --ladder: {error}") from error
        real_fidelities = [tune_realuse.Fidelity(r.frames, r.qp_indices) for r in real_rungs]
    rungs = None
    if getattr(args, "ladder", None) and not realuse_study:
        if include_noise:
            raise SystemExit("--ladder screens natural clips only: drop the noise options")
        try:
            rungs = ladder.parse_ladder(args.ladder, frames, ladder.RUNG_SIGMAS)
        except ValueError as error:
            raise SystemExit(f"invalid --ladder: {error}") from error
    anchor_file = TUNING_DIR / f"{study_name}.anchor.json"
    # A path is not a build identity: rebuilding it must never reuse an old
    # anchor or mix old trials with new ones. Include the measurement protocol
    # and input identities too, so a resumed study remains comparable.
    identity = {
        "binary": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "codec": codec.value,
        "params": names,
        "measurement": measurement_fingerprint(),
        "vmaf_metric": PRIMARY_VMAF_METRIC if metric_profile == V1_METRIC_PROFILE else None,
        "metric_profile": metric_profile,
        "secondary_metric": secondary_metric,
        "ms_ssim_binary_sha256": (hashlib.sha256(find_tool("vmaf", load_paths()).read_bytes()).hexdigest()
                                  if metric_profile == SCREEN_METRIC_PROFILE else None),
        "secondary_representation": ("raw[0,1]; native geometry yuv420p8; every frame"
                                     if metric_profile == SCREEN_METRIC_PROFILE else "pinned VMAF v1"),
        "cost_p_frames": cost_p_frames,
        "quantizer_grid": grid,
        "keep_streams": keep_streams,
        "vmaf_protocol": metric_api.vmaf_v1_protocol(load_paths(), VMAF_SUBSAMPLE),
        "protocol": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rate_domain": hashlib.sha256(Path(__file__).with_name("rate_domain.py").read_bytes()).hexdigest(),
        "rate_targets": hashlib.sha256(Path(__file__).with_name("sweep.py").read_bytes()).hexdigest(),
        "clips": {c.name: clip_fingerprint(c) for c in clips},
        "noise_guard": strict_noise_guard,
        "include_noise": include_noise,
        "search_space": domains,
        "cost_gop_period": cost_gop,
        "extend_noise_range": extend_noise_range,
        # Only non-default fidelity/guard settings enter the identity, so
        # studies started before these options existed still resume.
        **({"frames": frames} if frames != PROXY_FRAMES else {}),
        **({"class_guard": class_guard} if class_guard != MAX_CLASS_REGRESSION else {}),
        **({"ladder": [[r.label, r.sigma, r.sigma_cpu] for r in rungs]} if rungs else {}),
        **({"objective": objective_kind, "real_targets": real_targets,
            "flicker_objective": flicker_objective, "flicker_anchor": flicker_anchor,
            "real_ladder": ([[r.label, r.sigma, r.sigma_cpu] for r in real_rungs]
                            if real_rungs else None),
            "real_protocol": hashlib.sha256(
                Path(__file__).with_name("jreal.py").read_bytes()).hexdigest()}
           if realuse_study else {}),
        "noise_report": hashlib.sha256(Path(__file__).with_name("report.py").read_bytes()).hexdigest(),
        "clean_references": {c.name: clip_fingerprint(replace(c, yuv=c.clean_reference))
                             for c in clips if c.clean_reference},
    }
    if anchor_file.is_file():
        data = json.loads(anchor_file.read_text())
        if data.get("identity") != identity:
            raise SystemExit("study anchor does not match this binary, protocol or inputs; "
                             "use a new --study name")
    elif objective_kind == REALUSE_OBJECTIVE:
        # The real-use anchor is the default's CBR curves (cached per point by
        # nceval.cbr); only its CPU is needed here, at the fixed cost QPs.
        work = TUNING_DIR / "work" / f"{study_name}-anchor-cost"
        work.mkdir(parents=True, exist_ok=True)
        cost_options = {"p_frames": cost_p_frames} if cost_p_frames != COST_P_FRAMES else {}
        try:
            base_cost = instruction_cost(binary, codec, {}, work, **cost_options)
        finally:
            for f in work.iterdir():
                f.unlink(missing_ok=True)
            work.rmdir()
        data = {"curves": {}, "p_instructions": base_cost.p, "key_instructions": base_cost.key,
                "binary": str(binary), "params": names, "identity": identity}
        anchor_file.write_text(json.dumps(data, indent=1))
    else:
        base = evaluate(binary, codec, {}, TUNING_DIR / "work" / f"{study_name}-anchor",
                        clips=clips, quantizers=grid, extend_noise_range=extend_noise_range,
                        clip_workers=args.workers, checkpoint=True, **evaluation_options)
        data = {"curves": {k: curve_record(v, metric_profile) for k, v in base.curves.items()},
                "p_instructions": base.p_instructions, "binary": str(binary), "params": names,
                "key_instructions": base.key_instructions,
                "identity": identity}
        anchor_file.write_text(json.dumps(data, indent=1))
    anchor = {k: Curve(**v) for k, v in data["curves"].items()}
    natural_anchor = {c.name: anchor[c.name] for c in clips
                      if c.content_class != "synthetic_noise"} if anchor else {}
    rung_options = {k: v for k, v in evaluation_options.items() if k != "frames"}
    rung_anchors = {}
    for rung in (rungs or ())[:-1]:
        stored = data.setdefault("rungs", {})
        if rung.label not in stored:
            base = evaluate(binary, codec, {}, TUNING_DIR / "work" / f"{study_name}-anchor-{rung.label}",
                            rung_quantizers(anchor, rung), clips=clips, frames=rung.frames,
                            fixed_grid=True, cost=InstructionCost(data["p_instructions"],
                                                                  data["key_instructions"]),
                            clip_workers=args.workers, checkpoint=True, **rung_options)
            stored[rung.label] = {k: curve_record(v, metric_profile) for k, v in base.curves.items()}
            anchor_file.write_text(json.dumps(data, indent=1))
        rung_anchors[rung.label] = {k: Curve(**v) for k, v in stored[rung.label].items()}
    # Do not launch a study whose own baseline cannot produce valid BD curves.
    for metric in ("psnr", secondary_metric) if objective_kind == CQP_OBJECTIVE else ():
        bd_by_clip(natural_anchor, natural_anchor, metric)
        noise_deltas(anchor, anchor, clips, metric)
    real = (tune_realuse.RealUseObjective(binary, codec, [c.name for c in clips],
                                          real_fidelities, args.workers, class_guard,
                                          flicker_objective, flicker_anchor)
            if realuse_study else None)
    base_instructions = data["p_instructions"]
    base_cost = InstructionCost(base_instructions, data["key_instructions"])
    base_objective_cost = cpu_objective_cost(base_cost, cost_gop)
    print(f"defaults: {base_instructions:.1f} M instructions per P frame; "
          f"CPU objective {cpu_objective_label(cost_gop)} {base_objective_cost:.1f} M/frame", flush=True)

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=args.seed,
                                         constraints_func=constraints)
    surrogate_every = getattr(args, "surrogate_every", 0)
    if surrogate_every:
        sampler = surrogate.SurrogateSampler(sampler, surrogate_every, args.seed)
    study = serialize_trial_acquisition(optuna.create_study(
        study_name=study_name, storage=study_storage(study_name), load_if_exists=True,
        directions=["minimize", "minimize", "minimize"], sampler=sampler))
    study.set_user_attr("binary_build", getattr(args, "build_info", None))
    study.set_user_attr("rate_domain", "source-capped-v1")
    study.set_user_attr("metric_profile", metric_profile)
    study.set_user_attr("gate_policy", getattr(args, "gate", EVALUATION_GATE.value))
    study.set_user_attr("secondary_metric", secondary_metric)
    study.set_user_attr("vmaf_metric", PRIMARY_VMAF_METRIC if metric_profile == V1_METRIC_PROFILE else None)
    study.set_user_attr("vmaf_protocol", identity["vmaf_protocol"])
    study.set_user_attr("noise_guard", strict_noise_guard)
    study.set_user_attr("noise_report", include_noise)
    study.set_user_attr("search_space", domains)
    study.set_user_attr("cost_gop_period", cost_gop)
    study.set_user_attr("extend_noise_range", extend_noise_range)
    study.set_user_attr("frames", frames)
    study.set_user_attr("objective", objective_kind)
    study.set_user_attr("flicker_objective", flicker_objective)
    study.set_user_attr("class_guard", class_guard)
    if (metric_profile == V1_METRIC_PROFILE and len(study.trials) == 0
            and all(domain_contains(domains[n], specs[n][0]) for n in names)):
        study.enqueue_trial({n: specs[n][0] for n in names}, user_attrs={"default": True})
    if not study.user_attrs.get("seeded"):
        for seed in seed_trials(getattr(args, "seed_trials", None), names, specs, domains):
            study.enqueue_trial(seed, user_attrs={"seed": True})
        study.set_user_attr("seeded", True)

    def rung_objectives(curves: dict[str, Curve], rung_anchor: dict[str, Curve],
                        cost: InstructionCost, min_points: int
                        ) -> tuple[tuple[float, float, float], list[float]]:
        bd_p = bd_by_clip(curves, rung_anchor, "psnr", min_points)
        bd_v = bd_by_clip(curves, rung_anchor, secondary_metric, min_points)
        guards = [max(class_means(bd_p).values()) - class_guard,
                  max(class_means(bd_v).values()) - class_guard]
        return ((mean(bd_p.values()), mean(bd_v.values()),
                 cpu_objective_cost(cost, cost_gop) / base_objective_cost), guards)

    def climb(trial: optuna.Trial, overrides: dict[str, float]):
        """Measure `trial` up the ladder (nceval.ladder).

        Returns the last rung's Evaluation, or the objective values to
        record when the trial stops early (pruned, or a duplicate).
        """
        cost = None
        reached = {}
        for rung in rungs:
            final = rung is rungs[-1]
            result = evaluate(binary, codec, overrides,
                              TUNING_DIR / "work" / f"{study_name}-{trial.number}-{rung.label}",
                              rung_quantizers(anchor, rung), clips=clips, frames=rung.frames,
                              fixed_grid=not final, cost=cost,
                              cost_quantizers={n: anchor[n].quantizers[2] for n in COST_CLIPS},
                              **rung_options)
            cost = InstructionCost(result.p_instructions, result.key_instructions)
            if rung is rungs[0]:
                twin = duplicate_of(trial, result.curves)
                if twin is not None:
                    return twin
            if final:
                return result
            try:
                values, guards = rung_objectives(result.curves, rung_anchors[rung.label], cost,
                                                 len(rung.qp_indices))
            except ValueError:
                # A rung curve lost a point to the source cap (rates differ at
                # fewer frames) or its overlap: no cheap verdict, climb on.
                continue
            reached[rung.label] = list(values)
            trial.set_user_attr(ladder.RUNG_VALUES_ATTR, reached)
            finished = [t.user_attrs[ladder.RUNG_VALUES_ATTR][rung.label]
                        for t in trial.study.get_trials(deepcopy=False,
                                                        states=(optuna.trial.TrialState.COMPLETE,))
                        if not t.user_attrs.get(ladder.PRUNED_ATTR)
                        and rung.label in t.user_attrs.get(ladder.RUNG_VALUES_ATTR, {})
                        and max(constraints(t)) <= 0]
            if (ladder.clearly_infeasible(guards, rung)
                    or ladder.clearly_dominated(values, finished, rung)):
                trial.set_user_attr(ladder.PRUNED_ATTR, rung.label)
                trial.set_user_attr("constraint", guards)
                return values
        raise AssertionError("a ladder ends with its final rung")

    def duplicate_of(trial: optuna.Trial, curves: dict[str, Curve]):
        """An earlier trial's values when this trial's first-rung streams equal its own."""
        if any(len(c.stream_sha256) != len(c.quantizers) for c in curves.values()):
            return None
        key = ladder.stream_key(identity["binary"], {
            name: list(zip(c.quantizers, c.stream_sha256)) for name, c in curves.items()})
        trial.set_user_attr(ladder.RUNG_KEY_ATTR, key)
        for other in trial.study.get_trials(deepcopy=False,
                                            states=(optuna.trial.TrialState.COMPLETE,)):
            if other.user_attrs.get(ladder.RUNG_KEY_ATTR) == key:
                for name, value in other.user_attrs.items():
                    if name not in ("default", "seed", ladder.RUNG_KEY_ATTR):
                        trial.set_user_attr(name, value)
                trial.set_user_attr(ladder.DUPLICATE_ATTR, other.number)
                return tuple(other.values)
        return None

    def real_use_trial(trial: optuna.Trial, overrides: dict[str, float]
                       ) -> tuple[float, float, float]:
        """J_real terms (CBR ladder) and the CPU ratio, up the real-use rungs; guards as constraints."""
        work = TUNING_DIR / "work" / f"{study_name}-{trial.number}-cost"
        work.mkdir(parents=True, exist_ok=True)
        try:
            options = {"p_frames": cost_p_frames} if cost_p_frames != COST_P_FRAMES else {}
            # The fixed cost QPs (QUANTIZERS), as for the anchor: a CPU probe,
            # not an RD point.
            cost = instruction_cost(binary, codec, overrides, work, **options)
        finally:
            for f in work.iterdir():
                f.unlink(missing_ok=True)
            work.rmdir()
        ratio = cpu_objective_cost(cost, cost_gop) / base_objective_cost
        trial.set_user_attr("instruction_cost", {"p_million": cost.p, "key_million": cost.key,
                                                 "objective_ratio": ratio})
        reached = {}
        for rung, fidelity in zip(real_rungs or (None,), real_fidelities):
            result = real.evaluate(overrides, fidelity)
            values = result.objectives(ratio, flicker_objective)
            if rung is None or rung is real_rungs[-1]:
                break
            reached[rung.label] = list(values)
            trial.set_user_attr(ladder.RUNG_VALUES_ATTR, reached)
            finished = [t.user_attrs[ladder.RUNG_VALUES_ATTR][rung.label]
                        for t in trial.study.get_trials(deepcopy=False,
                                                        states=(optuna.trial.TrialState.COMPLETE,))
                        if not t.user_attrs.get(ladder.PRUNED_ATTR)
                        and rung.label in t.user_attrs.get(ladder.RUNG_VALUES_ATTR, {})
                        and max(constraints(t)) <= 0]
            if (ladder.clearly_infeasible(result.constraints, rung)
                    or ladder.clearly_dominated(values, finished, rung)):
                trial.set_user_attr(ladder.PRUNED_ATTR, rung.label)
                trial.set_user_attr("constraint", result.constraints)
                return values
        verdict = jreal.objective(result.summary, (ratio - 1) * 100)
        trial.set_user_attr("real_use", {
            "j_real": verdict["j_real"], "guards": verdict["guards"],
            "class_quality": result.summary["quality"]["classes"],
            "class_quality_low": result.summary["quality_low"]["classes"],
            "class_flicker": result.summary["d_flicker_psnr_y"]["classes"],
            **{term: result.summary[term]["balanced"] for term, _ in tune_realuse.GUARDS},
            **({f"flicker_vs_{flicker_anchor}": result.summary[f"flicker_vs_{flicker_anchor}"]
                ["balanced"]} if flicker_anchor else {}),
            "compliance": result.summary["compliance"], "refused": result.refused})
        trial.set_user_attr("constraint", result.constraints)
        return values

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        overrides = suggest_parameters(trial, domains)
        if surrogate_every and trial.number in sampler.proposed:
            trial.set_user_attr("surrogate", True)
        if real is not None:
            return real_use_trial(trial, overrides)
        if (metric_profile == SCREEN_METRIC_PROFILE
                and all(overrides[n] == specs[n][0] for n in names)):
            result = Evaluation(anchor, base_cost.p, base_cost.key)
            trial.set_user_attr("anchor_reused", str(anchor_file))
        elif rungs:
            outcome = climb(trial, overrides)
            if isinstance(outcome, tuple):
                return outcome
            result = outcome
        else:
            result = evaluate(binary, codec, overrides,
                              TUNING_DIR / "work" / f"{study_name}-{trial.number}",
                              {name: c.quantizers for name, c in anchor.items()}, clips=clips,
                              extend_noise_range=extend_noise_range, **evaluation_options)
        trial.set_user_attr("curves", {name: curve_record(c, metric_profile) for name, c in result.curves.items()})
        if include_noise:
            trial.set_user_attr("noise_range_status", {
                c.name: noise_range_status(result.curves[c.name], c, codec)
                for c in clips if c.clean_reference})
        trial.set_user_attr("cost_quantizers", {
            name: result.curves[name].quantizers[2] for name in COST_CLIPS})
        bd_p = bd_by_clip(result.curves, natural_anchor, "psnr")
        bd_v = bd_by_clip(result.curves, natural_anchor, secondary_metric)
        class_p, class_v = class_means(bd_p), class_means(bd_v)
        trial.set_user_attr("class_bd_psnr", class_p)
        trial.set_user_attr(f"class_bd_{secondary_metric}", class_v)
        cost = InstructionCost(result.p_instructions, result.key_instructions)
        trial.set_user_attr("instruction_cost", {
            "p_million": cost.p, "key_million": cost.key,
            "ip60_ratio": cost.per_frame(60) / base_cost.per_frame(60),
            "ip300_ratio": cost.per_frame(300) / base_cost.per_frame(300),
            "ip60_million": cost.per_frame(60),
            "ip300_million": cost.per_frame(300),
            "objective_gop_period": cost_gop,
            "objective_ratio": cpu_objective_cost(cost, cost_gop) / base_objective_cost,
        })
        guards = [max(class_p.values()) - class_guard,
                  max(class_v.values()) - class_guard]
        if include_noise:
            noise_constraints = []
            for metric in ("psnr", secondary_metric):
                deltas = noise_deltas(result.curves, anchor, clips, metric)
                trial.set_user_attr(f"noise_{metric}_deltas", deltas)
                noise_constraints.append(-min(v for rows in deltas.values() for v in rows.values()))
            trial.set_user_attr("noise_no_loss_constraint", noise_constraints)
            trial.set_user_attr("noise_no_loss_passed", bool(max(noise_constraints) <= 0))
            if strict_noise_guard:
                guards.extend(noise_constraints)
        trial.set_user_attr("constraint", guards)
        return mean(bd_p.values()), mean(bd_v.values()), cpu_objective_cost(cost, cost_gop) / base_objective_cost

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if getattr(args, "warm_start", None) and not study.user_attrs.get("warm_started"):
        added = warm_start(study, args.warm_start.split(","), names, specs, domains)
        study.set_user_attr("warm_started", args.warm_start)
        print(f"warm start: {added} finished trials from {args.warm_start}", flush=True)
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE
                    and not t.user_attrs.get(WARM_START_ATTR) for t in study.trials)
    # A failing trial (e.g. an encoder crash at an extreme setting) is
    # recorded as failed; the search carries on.
    # --timeout keeps one run inside the 30-minute budget; the study resumes.
    timeout = getattr(args, "timeout", None)
    study.optimize(objective, n_trials=max(0, args.trials - completed),
                   timeout=timeout * 60 if timeout else None,
                   n_jobs=min(args.workers, MAX_WORKERS), catch=(RuntimeError, ValueError))
    print_front(study)


def feasible_front(trials: list[optuna.trial.FrozenTrial]) -> list[optuna.trial.FrozenTrial]:
    """Nondominated completed trials after applying the class guards.

    Trials a ladder stopped early (measured only at a cheap rung) and
    duplicates of an earlier trial are left out (nceval.ladder).

    All three study objectives are minimised. Filtering an unconstrained
    front afterwards can hide useful trials behind infeasible dominators.
    """
    feasible = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
                and t.values is not None and max(constraints(t)) <= 0
                and not t.user_attrs.get(ladder.PRUNED_ATTR)
                and ladder.DUPLICATE_ATTR not in t.user_attrs
                and not t.user_attrs.get(WARM_START_ATTR)]
    return [t for t in feasible if not any(
        all(a <= b for a, b in zip(other.values, t.values))
        and any(a < b for a, b in zip(other.values, t.values))
        for other in feasible)]


def print_front(study: optuna.Study, details: bool = False) -> None:
    if reason := study.user_attrs.get("retired_reason"):
        print(f"study {study.study_name} is retired: {reason}")
        return
    done = [t for t in study.trials if t.values is not None]
    feasible = feasible_front(study.trials)
    noise_note = ("; clean-noise no-loss guards" if study.user_attrs.get("noise_guard") else
                  "; noise measured/reported, not constrained" if study.user_attrs.get("noise_report") else "")
    print(f"study {study.study_name}: {len(done)} trials; feasible Pareto front "
          f"(train proxy, BD vs default parameters; worst class within "
          f"+{study.user_attrs.get('class_guard', MAX_CLASS_REGRESSION)}% on both metrics{noise_note}):")
    cost_gop = study.user_attrs.get("cost_gop_period", 0)
    metric = ("ms_ssim" if study.user_attrs.get("metric_profile") == SCREEN_METRIC_PROFILE
              else study_vmaf_metric(study))
    label = "MS-SSIM (screen only)" if metric == "ms_ssim" else vmaf_label(metric)
    first = "PSNR"
    if study.user_attrs.get("flicker_objective"):
        first, label = "J_real Q", "dFlicker"
    elif study.user_attrs.get("objective") == REALUSE_OBJECTIVE:
        first, label = "J_real Q_full", "Q_low"
    for t in sorted(feasible, key=lambda t: t.values[2]):
        params = ", ".join(f"{k}={v:.4g}" for k, v in sorted(t.params.items()))
        ip300 = t.user_attrs.get("instruction_cost", {}).get("ip300_ratio")
        whole_gop = f"  GOP300 x{ip300:.3f}" if ip300 is not None and cost_gop != 300 else ""
        second_unit = " dB" if study.user_attrs.get("flicker_objective") else "%"
        print(f"  #{t.number:3d}  {first} {t.values[0]:+6.2f}%  {label} {t.values[1]:+6.2f}{second_unit}  "
              f"{cpu_objective_label(cost_gop)} x{t.values[2]:.3f}{whole_gop}  {params}")
        if "noise_no_loss_passed" in t.user_attrs:
            status = "PASS" if t.user_attrs["noise_no_loss_passed"] else "FAIL"
            minima = t.user_attrs["noise_no_loss_constraint"]
            print(f"       strict noise no-loss {status}; worst PSNR {-minima[0]:+.6f} dB, "
                  f"{label} {-minima[1]:+.6f} (report-only candidates are not guard-passed presets)")
        if details:
            for key in ("instruction_cost", "real_use", "noise_range_status", "class_bd_psnr", f"class_bd_{metric}", "noise_psnr_deltas", f"noise_{metric}_deltas"):
                if key in t.user_attrs:
                    print(f"       {key}: {json.dumps(t.user_attrs[key], sort_keys=True)}")


def search_stats(trials: list[optuna.trial.FrozenTrial]) -> dict:
    """Throughput and the share of trials the ladder, dedup and surrogate handled."""
    done = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
            and not t.user_attrs.get(WARM_START_ATTR)]
    if not done:
        return {"trials": 0}
    start = min(t.datetime_start for t in done)
    end = max(t.datetime_complete for t in done)
    hours = max((end - start).total_seconds() / 3600, 1e-9)
    front = {t.number for t in feasible_front(trials)}
    proposed = [t for t in done if t.user_attrs.get("surrogate")]
    sampled = [t for t in done if not t.user_attrs.get("surrogate")
               and not t.user_attrs.get("default") and not t.user_attrs.get("seed")]
    measured = [t for t in done if ladder.DUPLICATE_ATTR not in t.user_attrs]
    return {
        "trials": len(done), "hours": hours, "trials_per_hour": len(done) / hours,
        "pruned": sum(bool(t.user_attrs.get(ladder.PRUNED_ATTR)) for t in measured),
        "duplicates": len(done) - len(measured),
        "front": len(front),
        "surrogate_trials": len(proposed),
        "surrogate_on_front": sum(t.number in front for t in proposed),
        "sampler_trials": len(sampled),
        "sampler_on_front": sum(t.number in front for t in sampled),
        "rung_sigma": rung_sigmas(measured),
    }


def rung_sigmas(trials: list[optuna.trial.FrozenTrial]) -> dict[str, list[float]]:
    """Each rung's pairwise error (PSNR, VMAF BD %) against the final values.

    Measured on the study's own trials that climbed past the rung, so a
    running study checks the ladder's RUNG_SIGMAS on its own content.
    """
    from .fidelity import pair_sigma
    out = {}
    labels = {label for t in trials for label in t.user_attrs.get(ladder.RUNG_VALUES_ATTR, {})}
    for label in sorted(labels):
        climbed = [t for t in trials if label in t.user_attrs.get(ladder.RUNG_VALUES_ATTR, {})
                   and not t.user_attrs.get(ladder.PRUNED_ATTR)]
        if len(climbed) >= 3:
            out[label] = [pair_sigma([t.user_attrs[ladder.RUNG_VALUES_ATTR][label][k] for t in climbed],
                                     [t.values[k] for t in climbed]) for k in (0, 1)]
    return out


def stats(args: argparse.Namespace) -> None:
    study = optuna.load_study(study_name=args.study, storage=study_storage(args.study))
    print(json.dumps(search_stats(study.trials), indent=1, default=str))


def front(args: argparse.Namespace) -> None:
    print_front(optuna.load_study(study_name=args.study, storage=study_storage(args.study)),
                details=getattr(args, "details", False))


def write_presets(args: argparse.Namespace) -> None:
    study = optuna.load_study(study_name=args.study, storage=study_storage(args.study))
    if reason := study.user_attrs.get("retired_reason"):
        raise SystemExit(f"cannot publish presets from retired study: {reason}")
    if study.user_attrs.get("metric_profile") == SCREEN_METRIC_PROFILE:
        raise SystemExit("screen-only study: rescore a frozen shortlist with VMAF v1 before exporting presets")
    metric = study_vmaf_metric(study)
    objective_labels = (("J_real Q (CB BD)", "dFlicker vs default (dB)")
                        if study.user_attrs.get("flicker_objective") else
                        ("J_real Q_full (CB BD)", "J_real Q_low (CB BD)")
                        if study.user_attrs.get("objective") == REALUSE_OBJECTIVE
                        else ("PSNR-YUV BD", vmaf_label(metric) + " BD"))
    for spec in args.trials.split(","):
        number, _, name = spec.partition(":")
        trial = study.trials[int(number)]
        # Candidates stay under presets/candidates/ until validated on VAL.
        path = REPO_ROOT / "presets" / "candidates" / f"{args.codec}-{name or 'trial' + number}.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {args.codec} candidate from tuning study {args.study}, trial {number}.",
                 f"# Train proxy vs defaults: " + ", ".join(
                     f"{label} {value:+.2f}{unit}" for label, value, unit in
                     zip((*objective_labels,
                          cpu_objective_label(study.user_attrs.get("cost_gop_period", 0)) + " x"), trial.values,
                         ("%", "%", ""))) + ".",
                 "# Validate on VAL (nceval sweep --split val) before adopting."]
        for key in ("constraint", "instruction_cost", "real_use", "noise_range_status", "class_bd_psnr", f"class_bd_{metric}", "noise_psnr_deltas", f"noise_{metric}_deltas"):
            if key in trial.user_attrs:
                lines.append(f"# {key}: {json.dumps(trial.user_attrs[key], sort_keys=True)}")
        if "noise_no_loss_passed" in trial.user_attrs:
            lines.append(f"# Strict clean-noise no-loss passed: {trial.user_attrs['noise_no_loss_passed']}; "
                         f"noise constrained in study: {bool(study.user_attrs.get('noise_guard'))}.")
        lines += [f"{k} = {v!r}" for k, v in sorted(trial.params.items())]
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path}")


def read_preset(path: Path) -> dict[str, float]:
    overrides = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            k, v = line.split("=")
            overrides[k.strip()] = float(v)
    return overrides


def cost(args: argparse.Namespace) -> None:
    """Deterministic P-frame instruction cost of preset files vs the defaults."""
    codec = Codec(args.codec)
    binary = Path(args.binary).resolve()
    presets = {"default": {}} | {Path(p).stem: read_preset(Path(p))
                                 for p in args.preset_files.split(",") if p}

    def one(item):
        name, overrides = item
        work = TUNING_DIR / "work" / f"cost-{name}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            clips = {c.name: c for c in load_clips("train")}
            qs = {n: calibrate_quantizers(binary, codec, clips[n], PROXY_FRAMES,
                                          overrides, work)[2] for n in COST_CLIPS}
            return name, instruction_cost(binary, codec, overrides, work, qs).p
        finally:
            for f in work.iterdir():
                f.unlink(missing_ok=True)
            work.rmdir()

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        results = dict(pool.map(one, presets.items()))
    base = results["default"]
    for name, value in results.items():
        print(f"  {name}: {value:.1f} M instructions/P frame (x{value / base:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nceval.tune", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    from . import tune_pool  # the current protocol builds on this module's helpers
    tune_pool.add_commands(sub)
    s = sub.add_parser("search-legacy", help="archived fixed-QP / TRAIN-proxy studies only")
    s.add_argument("--codec", required=True, choices=[c.value for c in Codec])
    s.add_argument("--binary", required=True)
    s.add_argument("--params", required=True, help="comma-separated parameter names")
    s.add_argument("--trials", type=int, default=100)
    s.add_argument("--workers", type=int, default=MAX_WORKERS)
    s.add_argument("--study")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--cost-gop", type=int, default=0,
                   help="CPU objective: 0 keeps historical P instructions; positive period includes amortized key cost")
    s.add_argument("--noise-guard", action="store_true",
                   help="include all TRAIN noise clips with clean-reference no-loss guards")
    s.add_argument("--extend-noise-range", action="store_true",
                   help="append gated coarser noise points toward scaled200kbps, stopping at codec max QP")
    s.add_argument("--noise-report", action="store_true",
                   help="measure all TRAIN clean-reference noise deltas without constraining them")
    s.add_argument("--metric-profile", choices=[V1_METRIC_PROFILE, SCREEN_METRIC_PROFILE],
                   default=V1_METRIC_PROFILE, help="V1 by default; PSNR/MS-SSIM is a distinct screening-only profile")
    s.add_argument("--train-clips", help="comma-separated natural TRAIN subset covering all five classes and both cost clips; noise report adds all TRAIN noise families")
    s.add_argument("--quantizer-grid", help="JSON {clip:[QP,...]} frozen legal baseline grids; no recalibration")
    s.add_argument("--cost-p-frames", type=int, default=COST_P_FRAMES,
                   help="P frames in the separate I/P instruction probe;48 means49 total frames")
    s.add_argument("--gate", choices=[p.value for p in GatePolicy], default=EVALUATION_GATE.value,
                   help="decoders on the measured points (nceval.gatepolicy): off (default; conformance is checked when a change lands), sample or full")
    s.add_argument("--keep-streams", action="store_true", help="retain gated compressed streams and hash manifests for shortlist rescoring, never raw reconstructions")
    s.add_argument("--search-space", help="JSON domains for exactly --params: {low,high,log?} or {choices:[...]}")
    s.add_argument("--frames", type=int,
                   help="frames per clip (default 150; 30 with --objective realuse): fewer is a "
                   "cheaper exploration fidelity (re-measure promoted trials)")
    s.add_argument("--class-guard", type=float, default=MAX_CLASS_REGRESSION,
                   help="largest allowed per-class BD regression (percent, either metric)")
    s.add_argument("--seed-trials", help="JSON list of {param: value} points enqueued in a new study")
    s.add_argument("--timeout", type=float, help="stop starting trials after this many minutes (resume later)")
    s.add_argument("--objective", choices=[CQP_OBJECTIVE, REALUSE_OBJECTIVE], default=CQP_OBJECTIVE,
                   help="cqp: fixed-QP BD proxy; realuse: CBR ladder judged by J_real terms, "
                   "guards as constraints (nceval.tune_realuse)")
    s.add_argument("--real-targets", help="realuse: ladder targets to code (indices into "
                   "realuse.LADDER_KBPS; default 0,3: lowest and highest)")
    s.add_argument("--flicker-objective", action="store_true",
                   help="realuse: objectives J_real quality, class-balanced dFlicker vs the default, CPU")
    s.add_argument("--flicker-anchor", help="realuse: also require flicker no worse than this "
                   "encoder's own rate control at matched rate (e.g. svtav1-m12, x265-superfast)")
    s.add_argument("--surrogate-every", type=int, default=SURROGATE_EVERY,
                   help="every Nth trial (after 20 finished) is the surrogate model's proposal "
                   "(nceval.surrogate); 0 = TPE only")
    s.add_argument("--warm-start", help="comma-separated earlier studies whose finished trials seed the sampler")
    s.add_argument("--ladder", help="multi-fidelity rungs FRAMESxQPS,... ending at --frames x4, e.g. 30x3,60x4 "
                   "(nceval.ladder: stop clearly dominated trials early, reuse identical ones)")
    s.set_defaults(func=search)
    f = sub.add_parser("front-legacy")
    f.add_argument("--study", required=True)
    f.add_argument("--details", action="store_true", help="show full class and clean-reference noise deltas")
    f.set_defaults(func=front)
    st = sub.add_parser("stats", help="throughput, ladder stops, duplicates, surrogate hits")
    st.add_argument("--study", required=True)
    st.set_defaults(func=stats)
    w = sub.add_parser("write-presets-legacy")
    w.add_argument("--codec", required=True)
    w.add_argument("--study", required=True)
    w.add_argument("--trials", required=True, help="trial[:name],... e.g. 12:fast,37:better")
    w.set_defaults(func=write_presets)
    c = sub.add_parser("cost")
    c.add_argument("--codec", required=True, choices=[c.value for c in Codec])
    c.add_argument("--binary", required=True)
    c.add_argument("--preset-files", required=True)
    c.set_defaults(func=cost)
    for command in (s, c):
        command.add_argument("--expect-commit",
                             help="fail at once unless --binary was built from this commit "
                                  f"(also ${buildinfo.EXPECT_ENV})")
    args = parser.parse_args()
    if getattr(args, "workers", 1) > MAX_WORKERS:
        raise SystemExit(f"at most {MAX_WORKERS} workers (project rule)")
    if getattr(args, "binary", None) and hasattr(args, "expect_commit"):
        # Legacy commands; nceval.tune_pool's take a commit and check its build
        # (nceval.levels.build).
        args.build_info = buildinfo.check(Path(args.binary), "binary", args.expect_commit)
    args.func(args)


if __name__ == "__main__":
    main()
