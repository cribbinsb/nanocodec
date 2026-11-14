"""Quality metrics for a verified decode (eval/README.md "Methodology").

Primary:   psnr_yuv (6:1:1), vmaf_v1 (v1.0.16_3d0h at native geometry)
Secondary: psnr_y; opt-in xpsnr_y/xpsnr_yuv (ffmpeg) and ms_ssim (float_ms_ssim)

Measurement profiles (each has its own structure namespace and fingerprint,
so their results are never mixed):

  vmaf-v1          default. libvmaf scores the reconstruction against the
                   clip at their own 8-bit geometry: no upscale, no ffmpeg.
                   An INTERNAL relative metric: the 3d0h model is calibrated
                   for a 1080p display, so these values are not comparable
                   with published (Netflix/AOM CTC) 1080p-protocol VMAF.
  vmaf-v1-display  opt-in, for external comparison: Netflix's documented
                   protocol, bicubic upscale to 1080p 10-bit (ffmpeg) with
                   the encode-side CAMBI hints. Same values as before
                   METRICS_REVISION 5 (it keeps revision 4).
  legacy           the old v0.6.1/v0.6.1neg calls, never relabelled.

XPSNR (ffmpeg's xpsnr filter) is opt-in for vmaf-v1 runs (shortlists);
display and legacy runs keep it on by default, as before.

PSNR is computed from exact per-plane squared-error sums, so its definition
is under our control: per-frame PSNR per plane (MSE 0 is capped at
PSNR_CAP_DB), averaged over frames (JVET convention), then combined 6:1:1.
The sums come from the streamed frames (`PsnrMetric`) or, on our own
encoder's ungated points, from the encoder itself (`EncoderPsnrMetric`,
nceval.point), which then sends its reconstruction straight to VMAF: Python
never touches the pixels. Both give identical numbers.
XPSNR comes from ffmpeg's xpsnr filter. Native VMAF v1 and MS-SSIM come from
VMAFx on CUDA, one run per point on a shared scoring server, by default, or (NC_METRICS_BACKEND=
libvmaf-cpu) from libvmaf and nceval.msssim (libvmaf's float_ms_ssim,
recomputed faster); display and legacy VMAF always from libvmaf.

Every metric is a streaming component (`MetricSession`): the distorted
frames arrive one at a time over a pipe (an encoder's reconstruction or a
decoder's output) and are fanned out to PSNR and MS-SSIM in Python and to
ffmpeg/libvmaf processes through pipes. The reference is read straight from
the clip, only its first `frames` frames, so no trimmed copy, decoded YUV or
1080p 10-bit VMAF display copy is ever written (those cost ~2-3 GB of disk
traffic per 720p point). The file-based functions below stream their files through the
same components, so both routes give identical numbers.

Besides the sequence values, per-frame PSNR and VMAF are returned so the
results database can explain *where* an encoder loses (eval/README.md
"Methodology: Diagnostics").
"""

from __future__ import annotations

import contextlib
import csv
import functools
import json
import hashlib
import math
import os
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .clipframes import ClipFrames
from . import gpuserve
from .config import Paths, find_tool
from .framepipe import (PSNR_CAP_DB, FrameSink, ProcessFailed, ProcessSet, ProcessSink, PsnrSink,
                        psnr_from_mse, pump_frames)
from .msssim import MsSsimSink
from .slots import gpu_slots
from .yuv import YuvFormat, frame_count

# Bump when a metric's definition or invocation changes: results measured
# with an older revision become stale (see nceval.fingerprint).
METRICS_REVISION = 5  # VMAF v1 at native geometry; XPSNR opt-in.
DISPLAY_METRICS_REVISION = 4  # 1080p display upscale, explicit MS-SSIM secondary.
LEGACY_METRICS_REVISION = 3
V1_PROFILE = "vmaf-v1"
DISPLAY_PROFILE = "vmaf-v1-display"
LEGACY_PROFILE = "legacy"
PROFILES = (V1_PROFILE, DISPLAY_PROFILE, LEGACY_PROFILE)
# Structure-tag suffix per profile; the display profile keeps the namespace
# its results were stored under before native scoring existed.
_STRUCTURE_SUFFIX = {V1_PROFILE: "-vmaf-v1n", DISPLAY_PROFILE: "-vmaf-v1", LEGACY_PROFILE: ""}
VMAF_V1_MODEL = Path("share/vmaf/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json")
VMAF_V1_PREPROCESS_REVISION = 1

# Which program scores native VMAF v1 and MS-SSIM (NC_METRICS_BACKEND).
# Default: VMAFx on CUDA (owner decision, 2026-11-11). VMAFx (github.com/VMAFx/vmafx,
# a libvmaf fork; EUPL-1.2, so it is only ever run as an external program
# built outside this repository, never copied in; build recipe in
# docs/handoff/eval-throughput.md "GPU VMAF") has CUDA versions of the
# model's four features and of float_ms_ssim. On 24 encoder reconstructions
# (H.264/HEVC/AV1/VP9, six 720p clips, two rates; 2026-11-11) every
# per-frame VMAF v1 value was identical to libvmaf v3.2.1 at its printed 6
# decimals and MS-SSIM within 2.2e-6 of it. A 720p 150-frame point costs
# ~2.2 CPU-s for VMAF v1 + MS-SSIM (every frame) in one pass there,
# against ~4 CPU-s (libvmaf VMAF v1) + ~7 CPU-s (nceval.msssim, every
# frame) on the CPU.
# `libvmaf-cpu` (libvmaf + nceval.msssim) stays selectable, e.g. where there
# is no GPU. There is no silent fallback: a missing tool or GPU is an error.
# The backend is deliberately not part of any fingerprint or cache key: the
# stored values agree to their stored precision (REAL_METRICS_REVISION).
METRICS_BACKEND_ENV = "NC_METRICS_BACKEND"
LIBVMAF_CPU_BACKEND = "libvmaf-cpu"
VMAFX_CUDA_BACKEND = "vmafx-cuda"
METRICS_BACKENDS = (LIBVMAF_CPU_BACKEND, VMAFX_CUDA_BACKEND)
DEFAULT_METRICS_BACKEND = VMAFX_CUDA_BACKEND
#: The VMAFx `vmaf` tool; default: our patched CUDA build under $NC_DATA_ROOT/ext,
#: VMAFx 1068c699 plus ext/vmafx-patches 0001-0010 (frame-0 fix, MS-SSIM sums
#: on the GPU, scoring servers with a bounded pool, direct reads, wider
#: ADM/CAMBI launches, server inputs relayed with stall timeouts, one-way
#: drains by server generation;
#: docs/handoff/vmafx-gpu.md). A frozen build directory: a new build goes
#: into a new one and this default moves.
VMAFX_ENV = "NC_VMAFX"
VMAFX_DEFAULT = Path("ext/vmafx-serve5/build-cuda/tools/vmaf")
#: "0" scores each point in its own VMAFx process instead of the server pool.
VMAFX_SERVER_ENV = "NC_VMAFX_SERVER"
_NO_GPU_HINT = (f"see docs/handoff/eval-throughput.md \"GPU VMAF\", or set "
                f"{METRICS_BACKEND_ENV}={LIBVMAF_CPU_BACKEND} to score on the CPU")


def metrics_backend() -> str:
    backend = os.environ.get(METRICS_BACKEND_ENV, DEFAULT_METRICS_BACKEND)
    if backend not in METRICS_BACKENDS:
        raise ValueError(f"{METRICS_BACKEND_ENV}={backend!r}: expected one of {METRICS_BACKENDS}")
    return backend


class MetricFailed(RuntimeError):
    """A metric could not score a point: its process failed, its GPU had no
    memory, or its server refused the run. The point has no measurement and
    the run that wanted it fails; this is never an encoder's refusal (which
    nceval.realuse records as a missing point), so it is not a ProcessFailed."""


def vmafx_command(paths: Paths) -> list[str]:
    """How to start one VMAFx scoring run, once it has passed
    `_vmafx_self_check` in this process: the tool, by default as a client of
    a server of its pool (nceval.gpuserve)."""
    tool = Path(os.environ.get(VMAFX_ENV, paths.data_root / VMAFX_DEFAULT))
    if not (tool.is_file() and os.access(tool, os.X_OK)):
        raise FileNotFoundError(f"VMAFx tool not found at {tool} (set {VMAFX_ENV}); {_NO_GPU_HINT}")
    command = [str(tool)]
    if os.environ.get(VMAFX_SERVER_ENV, "1") != "0":
        if VMAFX_ENV not in os.environ:
            _retire_other_pools(tool)
        # Any server of the pool will do: points spread over them at random.
        index = random.randrange(gpuserve.pool_size())
        command += ["--via-server", gpuserve.server_name(tool, index)]
    _vmafx_self_check(tuple(command), paths.anchors / VMAF_V1_MODEL)
    return command


@functools.lru_cache(maxsize=None)
def _retire_other_pools(tool: Path) -> None:
    """Once per process: the default build drains the servers of older builds,
    and fails loudly (MetricFailed) if a newer build's pool is live."""
    try:
        gpuserve.drain_older_builds(tool)
    except gpuserve.OlderThanLivePool as error:
        raise MetricFailed(str(error)) from error


# Self-check input: big enough for every VMAF v1 feature's pyramid.
_SELF_CHECK_FORMAT = YuvFormat(320, 240)
_SELF_CHECK_FRAMES = 2
_SELF_CHECK_SECONDS = 60


@functools.lru_cache(maxsize=None)
def _vmafx_self_check(command: tuple[str, ...], model: Path) -> None:
    """Score two tiny frames the way a point is scored, before any point runs.

    A missing GPU, driver or CUDA extractor then fails once, with one clear
    message, instead of point by point (or, worse, as CPU-speed scoring).
    """
    fmt = _SELF_CHECK_FORMAT
    pattern = bytes(range(256)) * (fmt.frame_size * _SELF_CHECK_FRAMES // 256 + 1)
    with tempfile.TemporaryDirectory(prefix="nc-vmafx-check-") as scratch:
        reference = Path(scratch) / "reference.yuv"
        reference.write_bytes(pattern[:fmt.frame_size * _SELF_CHECK_FRAMES])
        report = Path(scratch) / "report.json"
        cmd = [*command, "-r", str(reference), "-d", "/dev/stdin", "-w", str(fmt.width),
               "-h", str(fmt.height), "-p", "420", "-b", "8",
               "--model", f"path={model}:name=vmaf_v1", "--feature", "float_ms_ssim",
               "--backend", "cuda", "--precision", "max", "--json", "-o", str(report), "-q"]
        try:
            run = subprocess.run(cmd, input=pattern[1:fmt.frame_size * _SELF_CHECK_FRAMES + 1],
                                 capture_output=True, timeout=_SELF_CHECK_SECONDS)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"VMAFx self-check could not run {command[0]}: {error}; "
                               f"{_NO_GPU_HINT}") from error
        if run.returncode != 0 or not report.is_file():
            if run.returncode == gpuserve.STATUS_RETIRED:
                raise MetricFailed(f"VMAFx server of {command[0]} is retired: restart on the "
                                   f"current nceval")
            tail = run.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["no output"]
            raise RuntimeError(f"VMAFx self-check failed (exit {run.returncode}: {tail[0]}); "
                               f"{_NO_GPU_HINT}")
        check_vmafx_report(json.loads(report.read_text()), _SELF_CHECK_FRAMES, ms_ssim=True)


def check_vmafx_report(report: dict, frames: int, ms_ssim: bool) -> None:
    """Every extractor ran on CUDA (and MS-SSIM's did); `frames` rows of finite scores."""
    extractors = {entry["extractor"]: entry["backend"]
                  for entry in report.get("feature_backends", [])}
    if (not extractors or any(backend != "cuda" for backend in extractors.values())
            or (ms_ssim and VMAFX_MS_SSIM_EXTRACTOR not in extractors)):
        raise RuntimeError(f"VMAFx did not score on CUDA (extractors: {extractors}); "
                           f"{_NO_GPU_HINT}")
    rows = report.get("frames", [])
    if len(rows) != frames:
        raise RuntimeError(f"VMAFx scored {len(rows)} frames, expected {frames}")
    for row in rows:
        if not all(math.isfinite(float(value)) for value in row["metrics"].values()):
            raise RuntimeError(f"VMAFx frame {row['frameNum']}: non-finite score {row['metrics']}")


def structure_for_profile(tag: str, profile: str) -> str:
    if profile not in PROFILES:
        raise ValueError(f"unknown measurement profile {profile!r}")
    return _structure_base(tag) + _STRUCTURE_SUFFIX[profile]


def _structure_base(tag: str) -> str:
    for suffix in sorted(filter(None, _STRUCTURE_SUFFIX.values()), key=len, reverse=True):
        if tag.endswith(suffix):
            return tag.removesuffix(suffix)
    return tag


def structure_for_metric(tag: str, metric: str) -> str:
    """The namespace holding `metric`: legacy VMAF lives in the legacy one;
    vmaf_v1 in the display namespace if `tag` names it, else the native one."""
    if metric in ("vmaf", "vmaf_neg"):
        return structure_for_profile(tag, LEGACY_PROFILE)
    if metric == "vmaf_v1":
        display = tag == structure_for_profile(tag, DISPLAY_PROFILE)
        return structure_for_profile(tag, DISPLAY_PROFILE if display else V1_PROFILE)
    return tag


def vmaf_v1_protocol(paths: Paths, subsample: int = 1, display: bool = False) -> dict:
    """The identity of a VMAF v1 measurement, for fingerprints."""
    model = paths.anchors / VMAF_V1_MODEL
    if not display:
        return {"metric": "vmaf_v1", "model": "v1.0.16_3d0h",
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "geometry": "native encoded width/height", "preprocessing": "none",
                "pixel_format": "yuv420p", "bitdepth": 8,
                "cambi": "encoded width/height/8-bit", "subsample": subsample}
    ffmpeg = find_tool("ffmpeg", paths).resolve()
    version = subprocess.run([str(ffmpeg), "-version"], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, check=True).stdout.strip()
    return {"preprocessing_ffmpeg": {
                "binary_sha256": hashlib.sha256(ffmpeg.read_bytes()).hexdigest(),
                "version_build_libraries": version}, "metric": "vmaf_v1", "model": "v1.0.16_3d0h",
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "preprocessing_revision": VMAF_V1_PREPROCESS_REVISION,
            "display": "aspect-preserving1080-height-landscape/1080-width-portrait;nearest-even420",
            "scale": "ffmpeg-bicubic", "pixel_format": "yuv420p10le", "bitdepth": 10,
            "cambi": "original encoded width/height/8-bit", "subsample": subsample}


def vmaf_v1_geometry(fmt: YuvFormat) -> tuple[int, int]:
    even = lambda numerator, denominator: max(2, 2 * ((numerator + denominator) // (2 * denominator)))
    return ((even(fmt.width * 1080, fmt.height), 1080) if fmt.width >= fmt.height else
            (1080, even(fmt.height * 1080, fmt.width)))

VMAF_THREADS = 2  # sweeps run ~14 points in parallel
# Slot weight of one native libvmaf process (VMAF v1, legacy VMAF).
# Paced by the encoder's stream it keeps about one core busy, not
# VMAF_THREADS: on a streamed 720p 150-frame real-use point (2026-11-11,
# quiet machine) VMAF v1 used 3.1 CPU-s and libvmaf's MS-SSIM (every 4th
# frame; now nceval.msssim) 4.7 CPU-s over the 4.9 s encode, beside the
# encoder (1-2 CPU-s) and the Python PSNR/hash reader (0.5 CPU-s; none
# since ungated points score PSNR in the encoder, nceval.point).
NATIVE_VMAF_SLOT_WEIGHT = 1
# A VMAFx run keeps no core busy (a client process that waits, and a thread
# of a shared scoring server: driver calls and input reads); its GPU work is
# bounded by nceval.slots.gpu_slots.
GPU_METRIC_SLOT_WEIGHT = 0
# VMAFx's CUDA MS-SSIM extractor: its presence proves MS-SSIM ran on the GPU.
VMAFX_MS_SSIM_EXTRACTOR = "float_ms_ssim_cuda"
# Display protocol: two single-threaded upscalers plus a 2-thread libvmaf,
# paced by the stream: measured ~2 busy cores per streamed 720p point.
VMAF_V1_DISPLAY_SLOT_WEIGHT = 2
METRIC_TIMEOUT_SECONDS = 1800

_XPSNR_LINE = re.compile(r"XPSNR\s+y:\s*([\d.inf]+)\s+u:\s*([\d.inf]+)\s+v:\s*([\d.inf]+)")


@dataclass
class Measurement:
    summary: dict[str, float]
    per_frame: list[dict[str, float]] = field(default_factory=list)


def weighted_yuv(y: float, u: float, v: float) -> float:
    """The JVET 6:1:1 luma/chroma weighting."""
    return (6.0 * y + u + v) / 8.0


@dataclass(frozen=True)
class StreamSpec:
    """What every metric of one point needs: the reference clip prefix."""
    #: The clip itself (only its first `frames` frames are read), or its
    #: measured frames served by nceval.clipframes (pool clips, rotated starts),
    #: or any other frame server with a `reader()` (nceval.svc_chart's downscale).
    reference: Path | ClipFrames
    fmt: YuvFormat
    frames: int
    fps: float
    paths: Paths | None  # None only for PSNR, which runs no tool

    def reference_path(self) -> Path:
        """A path one metric reads the reference from (a fresh one per call)."""
        if isinstance(self.reference, Path):
            return self.reference
        return self.reference.reader()


class MetricComponent(Protocol):
    """One metric: sinks for the distorted frames, then its Measurement.

    `cpu_threads` is its CPU-slot weight (nceval.slots) while streaming;
    `uses_gpu` says whether it runs a GPU process (nceval.slots.gpu_slots).
    """

    cpu_threads: int
    uses_gpu: bool

    def start(self, processes: ProcessSet, scratch: Path) -> list[FrameSink]: ...

    def measurement(self, processes: ProcessSet) -> Measurement: ...


class PsnrMetric:
    """PSNR per plane per frame, computed in Python from the streamed frames."""

    cpu_threads = 0  # runs on the stream's reader thread
    uses_gpu = False

    def __init__(self, spec: StreamSpec):
        self._spec = spec
        self._sink: PsnrSink | None = None

    def start(self, processes: ProcessSet, scratch: Path) -> list[FrameSink]:
        self._sink = PsnrSink(self._spec.reference_path(), self._spec.fmt)
        return [self._sink]

    def measurement(self, processes: ProcessSet) -> Measurement:
        return psnr_measurement(self._sink.per_plane_psnr())


class EncoderPsnrMetric(PsnrMetric):
    """PSNR from the per-frame, per-plane squared errors our encoder logs
    against the reference while it encodes (`--sse-log`, nc-cli sse_log.rs).

    Same definition as `PsnrMetric` (an exact integer sum per plane, divided
    by the plane size), but no frame passes through Python.
    """

    def __init__(self, spec: StreamSpec, log: Path):
        super().__init__(spec)
        self.log = log

    def start(self, processes: ProcessSet, scratch: Path) -> list[FrameSink]:
        return []

    def encoder_arguments(self) -> list[str]:
        return ["--sse-log", str(self.log), "--sse-reference", str(self._spec.reference_path())]

    def measurement(self, processes: ProcessSet) -> Measurement:
        fmt = self._spec.fmt
        sizes = (fmt.luma_size, fmt.chroma_size, fmt.chroma_size)
        with self.log.open(newline="") as log:
            mse = [tuple(int(row[f"sse_{plane}"]) / size for plane, size in zip("yuv", sizes))
                   for row in csv.DictReader(log)]
        if len(mse) != self._spec.frames:
            raise RuntimeError(f"--sse-log has {len(mse)} frames, expected {self._spec.frames}")
        return psnr_measurement(psnr_from_mse(mse))


def psnr_measurement(per_plane) -> Measurement:
    """Sequence and per-frame PSNR from (frames, 3) per-plane PSNR."""
    y, u, v = per_plane.mean(axis=0)
    summary = {"psnr_y": float(y), "psnr_u": float(u), "psnr_v": float(v),
               "psnr_yuv": weighted_yuv(y, u, v)}
    per_frame = [{"psnr_y": float(r[0]), "psnr_u": float(r[1]), "psnr_v": float(r[2])}
                 for r in per_plane]
    return Measurement(summary, per_frame)


class XpsnrMetric:
    """ffmpeg's xpsnr filter; the distorted frames arrive on its stdin."""

    cpu_threads = 1
    uses_gpu = False

    def __init__(self, spec: StreamSpec):
        self._spec = spec
        self._process: subprocess.Popen | None = None

    def start(self, processes: ProcessSet, scratch: Path) -> list[FrameSink]:
        spec = self._spec
        raw = ["-f", "rawvideo", "-s", f"{spec.fmt.width}x{spec.fmt.height}",
               "-pix_fmt", "yuv420p", "-r", f"{spec.fps:g}"]
        # xpsnr treats its FIRST input as the original: the visual activity
        # weights come from it (vf_xpsnr.c, do_xpsnr). trim keeps exactly the
        # encoded prefix of the clip, so no trimmed copy is needed.
        graph = f"[0:v]trim=end_frame={spec.frames}[ref];[ref][1:v]xpsnr"
        cmd = [str(find_tool("ffmpeg", spec.paths)), "-hide_banner", "-nostdin",
               *raw, "-i", str(spec.reference_path()), *raw, "-i", "pipe:0",
               "-lavfi", graph, "-f", "null", "-"]
        self._process = processes.spawn(cmd, "xpsnr", stdin=subprocess.PIPE)
        return [ProcessSink(self._process)]

    def measurement(self, processes: ProcessSet) -> Measurement:
        match = _XPSNR_LINE.search(processes.stderr_text(self._process))
        if not match:
            raise RuntimeError("could not parse ffmpeg xpsnr output")
        y, u, v = (min(PSNR_CAP_DB, float(x)) for x in match.groups())
        return Measurement({"xpsnr_y": y, "xpsnr_yuv": weighted_yuv(y, u, v)})


class VmafV1DisplayMetric:
    """VMAF v1.0.16_3d0h on the 10-bit 1080p display upscale (opt-in profile).

    Two ffmpeg upscalers (reference prefix from the clip; distorted frames
    from our stdin) feed libvmaf through anonymous pipes named /dev/fd/N.
    The upscale command is exactly `vmaf_v1_display_copy`'s.
    """

    cpu_threads = VMAF_V1_DISPLAY_SLOT_WEIGHT
    uses_gpu = False

    def __init__(self, spec: StreamSpec, subsample: int = 1):
        if subsample < 1:
            raise ValueError("VMAF subsample must be positive")
        model = spec.paths.anchors / VMAF_V1_MODEL
        if not model.is_file():
            raise FileNotFoundError(f"pinned VMAF v1 model missing: {model}")
        self._spec, self._subsample = spec, subsample
        self._json: Path | None = None

    def start(self, processes: ProcessSet, scratch: Path) -> list[FrameSink]:
        spec = self._spec
        self._json = scratch / "vmaf_v1.json"
        ref_read, ref_write = os.pipe()
        dist_read, dist_write = os.pipe()
        try:
            processes.spawn(_display_upscale_command(str(spec.reference_path()), spec.fmt, spec.paths,
                                                     "pipe:1", frames=spec.frames),
                            "vmaf_v1 reference upscale", stdout=ref_write)
            upscaler = processes.spawn(_display_upscale_command("pipe:0", spec.fmt, spec.paths,
                                                                "pipe:1"),
                                       "vmaf_v1 distorted upscale", stdin=subprocess.PIPE,
                                       stdout=dist_write)
            processes.spawn(_vmaf_v1_command(f"/dev/fd/{ref_read}", f"/dev/fd/{dist_read}",
                                             spec.fmt, spec.paths, self._subsample, self._json),
                            "vmaf_v1", pass_fds=(ref_read, dist_read))
        finally:
            for fd in (ref_read, ref_write, dist_read, dist_write):
                os.close(fd)  # the children hold their own copies
        return [ProcessSink(upscaler)]

    def measurement(self, processes: ProcessSet) -> Measurement:
        return _parse_vmaf_v1(self._json)


class NativeVmafMetric:
    """libvmaf at the clip's own 8-bit geometry; distorted frames on stdin.

    The reference is the clip itself, limited by --frame_cnt to the encoded
    prefix. `arguments` select models/features; `parse` reads the JSON.
    """

    cpu_threads = NATIVE_VMAF_SLOT_WEIGHT
    uses_gpu = False

    def __init__(self, spec: StreamSpec, label: str, arguments: list[str], parse,
                 command: list[str] | None = None):
        """`command`: how to run the scoring program; None is the pinned libvmaf."""
        self._spec, self._label, self._arguments, self._parse = spec, label, arguments, parse
        self._command = command
        self._json: Path | None = None

    def start(self, processes: ProcessSet, scratch: Path) -> list[FrameSink]:
        spec = self._spec
        self._json = scratch / f"{self._label}.json"
        command = self._command or [str(find_tool("vmaf", spec.paths))]
        cmd = [*command, "-r", str(spec.reference_path()), "-d", "/dev/stdin",
               "-w", str(spec.fmt.width), "-h", str(spec.fmt.height), "-p", "420", "-b", "8",
               "--frame_cnt", str(spec.frames), *self._arguments,
               "--json", "-o", str(self._json), "-q"]
        return [ProcessSink(processes.spawn(cmd, self._label, stdin=subprocess.PIPE))]

    def measurement(self, processes: ProcessSet) -> Measurement:
        return self._parse(json.loads(self._json.read_text()))


def _vmaf_v1_model_argument(spec: StreamSpec) -> str:
    """The pinned model with CAMBI's native encode-side options, stated explicitly."""
    model = spec.paths.anchors / VMAF_V1_MODEL
    if not model.is_file():
        raise FileNotFoundError(f"pinned VMAF v1 model missing: {model}")
    return (f"path={model}:name=vmaf_v1:cambi.enc_width={spec.fmt.width}"
            f":cambi.enc_height={spec.fmt.height}:cambi.enc_bitdepth=8")


class VmafV1Metric(NativeVmafMetric):
    """VMAF v1.0.16_3d0h at the clip's own 8-bit geometry, by libvmaf on the CPU.

    libvmaf reads the reference prefix from the clip and the distorted frames
    from its stdin; nothing is rescaled.
    """

    def __init__(self, spec: StreamSpec, subsample: int = 1):
        if subsample < 1:
            raise ValueError("VMAF subsample must be positive")
        self._subsample = subsample
        super().__init__(spec, "vmaf_v1", ["--model", _vmaf_v1_model_argument(spec),
                                           "--subsample", str(subsample),
                                           "--threads", str(VMAF_THREADS)],
                         _parse_vmaf_v1_report)


class VmafxMetric(NativeVmafMetric):
    """VMAF v1 and/or MS-SSIM in ONE VMAFx CUDA run.

    One run per point, never one per metric, on a shared scoring server
    (`vmafx_command`): a CUDA context per run cost ~1 s of start-up and
    teardown, more than a 150-frame point's scoring. Scores are printed at
    full precision (--precision max) and VMAF v1 is rounded here to
    libvmaf's printed 6 decimals, so stored VMAF v1 values equal the CPU
    backend's (24/24 parity points). PSNR is ours (exact, our own MSE-0
    cap). Every feature must report a CUDA extractor: a feature that
    silently fell back to the CPU would cost seconds per point.
    """

    cpu_threads = GPU_METRIC_SLOT_WEIGHT
    uses_gpu = True

    def __init__(self, spec: StreamSpec, vmaf_v1: bool = True, ms_ssim: bool = False,
                 vmaf_subsample: int = 1):
        if not (vmaf_v1 or ms_ssim):
            raise ValueError("VMAFx pass with no metric")
        if vmaf_subsample < 1:
            raise ValueError("VMAF subsample must be positive")
        if ms_ssim and vmaf_subsample != 1:
            raise ValueError("MS-SSIM is scored on every frame: no VMAF subsample beside it")
        self._vmaf_v1, self._ms_ssim, self._subsample = vmaf_v1, ms_ssim, vmaf_subsample
        arguments = (["--model", _vmaf_v1_model_argument(spec),
                      "--subsample", str(vmaf_subsample)] if vmaf_v1 else ["--no_prediction"])
        if ms_ssim:
            arguments += ["--feature", "float_ms_ssim"]
        super().__init__(spec, "vmafx", [*arguments, "--backend", "cuda", "--precision", "max"],
                         self._parse_report, vmafx_command(spec.paths))

    def _parse_report(self, report: dict) -> Measurement:
        check_vmafx_report(report, -(-self._spec.frames // self._subsample), self._ms_ssim)
        frames = sorted(report["frames"], key=lambda f: f["frameNum"])
        summary: dict[str, float] = {}
        rows: list[dict[str, float]] = [{} for _ in frames]
        if self._vmaf_v1:
            summary["vmaf_v1"] = _libvmaf_printed(report["pooled_metrics"]["vmaf_v1"]["mean"])
            for row, frame in zip(rows, frames):
                row["vmaf_v1"] = _libvmaf_printed(frame["metrics"]["vmaf_v1"])
        if self._ms_ssim:
            scores = [float(frame["metrics"]["float_ms_ssim"]) for frame in frames]
            summary["ms_ssim"] = sum(scores) / len(scores)
            for row, score in zip(rows, scores):
                row["ms_ssim"] = score
        return Measurement(summary, rows)


def _libvmaf_printed(score: float) -> float:
    """A score as libvmaf's default %.6f JSON prints it."""
    return float(f"{float(score):.6f}")


def vmaf_v1_and_ms_ssim(spec: StreamSpec, vmaf_v1: bool = True, ms_ssim: bool = False,
                        vmaf_subsample: int = 1) -> list[MetricComponent]:
    """The components scoring native VMAF v1 and/or MS-SSIM on `metrics_backend`."""
    if metrics_backend() == VMAFX_CUDA_BACKEND:
        return [VmafxMetric(spec, vmaf_v1, ms_ssim, vmaf_subsample)]
    return ([VmafV1Metric(spec, vmaf_subsample)] if vmaf_v1 else []) + \
        ([MsSsimMetric(spec)] if ms_ssim else [])


def legacy_vmaf_metric(spec: StreamSpec) -> NativeVmafMetric:
    """VMAF v0.6.1 and v0.6.1neg (legacy profile; values are never relabelled)."""
    models = spec.paths.anchors / "share" / "vmaf" / "model"
    arguments = ["--model", f"path={models / 'vmaf_v0.6.1.json'}:name=vmaf",
                 "--model", f"path={models / 'vmaf_v0.6.1neg.json'}:name=vmaf_neg",
                 "--threads", str(VMAF_THREADS)]

    def parse(report: dict) -> Measurement:
        pooled = report["pooled_metrics"]
        frames = sorted(report["frames"], key=lambda f: f["frameNum"])
        return Measurement({"vmaf": pooled["vmaf"]["mean"], "vmaf_neg": pooled["vmaf_neg"]["mean"]},
                           [{"vmaf": f["metrics"]["vmaf"], "vmaf_neg": f["metrics"]["vmaf_neg"]}
                            for f in frames])
    return NativeVmafMetric(spec, "vmaf_legacy", arguments, parse)


class MsSsimMetric:
    """Float MS-SSIM (raw [0, 1] scores) of every frame, on the CPU.

    Computed in-process (nceval.msssim: libvmaf's float_ms_ssim to within
    1e-6, at about a quarter of its CPU) on a worker thread beside the
    stream; the GPU backend scores it in `VmafxMetric` instead.
    """

    cpu_threads = 1  # the scoring thread
    uses_gpu = False

    def __init__(self, spec: StreamSpec):
        self._spec = spec
        self._sink: MsSsimSink | None = None

    def start(self, processes: ProcessSet, scratch: Path) -> list[FrameSink]:
        self._sink = MsSsimSink(self._spec.reference_path(), self._spec.fmt)
        return [self._sink]

    def measurement(self, processes: ProcessSet) -> Measurement:
        scores = self._sink.scores
        return Measurement({"ms_ssim": sum(scores) / len(scores)},
                           [{"ms_ssim": score} for score in scores])


# The real-use metric set (nceval.realuse, J_real in docs/PROCEDURES.md):
# PSNR, VMAF v1 and MS-SSIM, each on every frame. MS-SSIM was scored on
# every 4th frame until 2026-11-11 (libvmaf's MS-SSIM cost ~5x a VMAF v1
# pass); that biased J_real against anything that shifts the QP cadence
# (docs/handoff/realuse.md "Measurement trap"). On the GPU every frame is
# cheap, so the owner moved it to every frame (2026-11-11).
#: Part of every real-use point's cache key (nceval.cbr). Not bumped for the
#: earlier scorer moves (dropping VMAF v0.6.1 NEG, libvmaf -> nceval.msssim,
#: -> VMAFx CUDA): stored PSNR and VMAF v1 were unchanged at their stored
#: precision (VmafxMetric rounds VMAF v1 to libvmaf's 6 decimals; 24/24
#: parity reconstructions identical) and MS-SSIM moved by < 2.2e-6.
REAL_METRICS_REVISION = 1
#: The MS-SSIM protocol of a real-use result, stored IN the result (not its
#: key) so that cached PSNR/VMAF v1 entries stay valid: a cached result with
#: another value is re-measured when it is next used (nceval.cbr), never in
#: bulk (owner decision, 2026-11-11). 1 (absent): every 4th frame; 2: every frame.
REAL_MS_SSIM_REVISION = 2


def real_metrics(spec: StreamSpec) -> list[MetricComponent]:
    """PSNR (per plane), VMAF v1 and MS-SSIM, each on every frame."""
    return [PsnrMetric(spec), *vmaf_v1_and_ms_ssim(spec, ms_ssim=True)]


def default_xpsnr(profile: str) -> bool:
    """XPSNR is opt-in for native VMAF v1 runs and on by default otherwise."""
    return profile != V1_PROFILE


def profile_metrics(spec: StreamSpec, profile: str = V1_PROFILE,
                    include_ms_ssim: bool = False,
                    include_xpsnr: bool | None = None) -> list[MetricComponent]:
    """The metric set of a sweep point (see the module docstring)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown measurement profile {profile!r}")
    components: list[MetricComponent] = [PsnrMetric(spec)]
    if default_xpsnr(profile) if include_xpsnr is None else include_xpsnr:
        components.append(XpsnrMetric(spec))
    if profile == V1_PROFILE:
        return components + vmaf_v1_and_ms_ssim(spec, ms_ssim=include_ms_ssim)
    components.append({DISPLAY_PROFILE: VmafV1DisplayMetric,
                       LEGACY_PROFILE: legacy_vmaf_metric}[profile](spec))
    if include_ms_ssim:
        components += vmaf_v1_and_ms_ssim(spec, vmaf_v1=False, ms_ssim=True)
    return components


def gpu_slot_need(components: list[MetricComponent]) -> int:
    """GPU slots (nceval.slots.gpu_slots) a point scored by `components` holds:
    one per point, however many GPU processes it runs."""
    return int(any(component.uses_gpu for component in components))


#: How long a metric process that broke its input pipe may take to exit.
BROKEN_PIPE_WAIT_SECONDS = 30


def _is_broken_pipe(error: BaseException) -> bool:
    """Whether `error` is a frame producer's failure to write into a closed
    pipe (Python's BrokenPipeError, or a process that died of it)."""
    if isinstance(error, BrokenPipeError):
        return True
    text = str(error)
    return "Broken pipe" in text or "os error 32" in text or "failed (-13)" in text


class MetricSession:
    """Scores one stream of distorted frames with a set of metric components.

        with MetricSession(spec, components) as session:
            frames = pump_frames(source, spec.fmt, [*gate_sinks, *session.sinks])
            measurement = session.result(frames)

    Leaving the block kills any helper process still running (e.g. after a
    failed gate), so an abandoned measurement never lingers.
    """

    def __init__(self, spec: StreamSpec, components: list[MetricComponent]):
        self._spec, self._components = spec, components
        self._processes = ProcessSet()
        self._scratch: tempfile.TemporaryDirectory | None = None
        self.sinks: list[FrameSink] = []

    def __enter__(self) -> "MetricSession":
        self._scratch = tempfile.TemporaryDirectory(prefix="nc-metrics-")
        try:
            for component in self._components:
                self.sinks += component.start(self._processes, Path(self._scratch.name))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._processes.kill_all()
        for sink in self.sinks:
            if isinstance(sink, (PsnrSink, MsSsimSink)):
                # If the stream was cut short: close the reference file and
                # stop MS-SSIM's thread (its error, if any, was already raised).
                with contextlib.suppress(Exception):
                    sink.finish()
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    @property
    def cpu_threads(self) -> int:
        return sum(component.cpu_threads for component in self._components)

    @property
    def gpu_slots(self) -> int:
        return gpu_slot_need(self._components)

    def failure_behind(self, error: BaseException) -> MetricFailed | None:
        """The metric failure that caused `error` (the producer of the frames
        failing on a broken pipe into a metric process), or None when the
        producer failed on its own."""
        if not _is_broken_pipe(error):
            return None
        try:
            self._processes.wait_all(BROKEN_PIPE_WAIT_SECONDS)
        except ProcessFailed as failure:
            return MetricFailed(f"metric failed: {failure}")
        return MetricFailed(f"a metric process stopped reading its frames: {error}")

    def result(self, frames_streamed: int) -> Measurement:
        """Wait for the helpers and combine every component's Measurement."""
        if frames_streamed != self._spec.frames:
            raise ValueError(f"streamed {frames_streamed} frames, expected {self._spec.frames}")
        try:
            self._processes.wait_all(METRIC_TIMEOUT_SECONDS)
        except ProcessFailed as failure:
            raise MetricFailed(f"metric failed: {failure}") from failure
        measurement = combine([c.measurement(self._processes) for c in self._components])
        # Point sets start with PSNR, which has a row for every frame.
        whole = bool(self._components) and isinstance(self._components[0], PsnrMetric)
        check_measurement(measurement, self._spec.frames if whole else None)
        return measurement


#: Scores that must always be finite (XPSNR may legitimately be inf).
FINITE_METRICS = ("psnr_y", "psnr_u", "psnr_v", "psnr_yuv", "vmaf_v1", "ms_ssim")


def check_measurement(measurement: Measurement, frames: int | None) -> None:
    """A point's scores: `frames` per-frame rows (None: unchecked), every
    FINITE_METRICS value finite. A NaN or a short row list never reaches a
    result file."""
    if frames is not None and len(measurement.per_frame) != frames:
        raise RuntimeError(f"{len(measurement.per_frame)} per-frame metric rows, expected {frames}")
    for where, values in [("summary", measurement.summary),
                          *((f"frame {i}", row) for i, row in enumerate(measurement.per_frame))]:
        bad = {key: values[key] for key in FINITE_METRICS
               if key in values and not math.isfinite(values[key])}
        if bad:
            raise RuntimeError(f"non-finite metric scores in {where}: {bad}")


def combine(parts: list[Measurement]) -> Measurement:
    """Merge summaries and per-frame rows; the first part sets the row count.

    (PSNR comes first in every multi-metric set and has a row per frame; a
    lone subsampled VMAF keeps its subsampled rows.)
    """
    summary: dict[str, float] = {}
    per_frame: list[dict[str, float]] = [{} for _ in parts[0].per_frame]
    for part in parts:
        summary.update(part.summary)
        if part.per_frame and len(part.per_frame) != len(per_frame):
            raise RuntimeError("per-frame metric counts disagree")
        for row, values in zip(per_frame, part.per_frame):
            row.update(values)
    return Measurement(summary, per_frame)


def score_files(reference: Path, distorted: Path, fmt: YuvFormat, fps: float, paths: Paths,
                components) -> Measurement:
    """Stream a distorted YUV file through `components(spec)` (equal lengths)."""
    frames = frame_count(distorted, fmt)
    if frames == 0 or frame_count(reference, fmt) != frames:
        raise ValueError(f"frame counts differ or are empty: "
                         f"{frame_count(reference, fmt)} vs {frames}")
    spec = StreamSpec(reference, fmt, frames, fps, paths)
    parts = components(spec)
    with gpu_slots(gpu_slot_need(parts)), MetricSession(spec, parts) as session, \
            distorted.open("rb") as source:
        return session.result(pump_frames(source, fmt, session.sinks))


def psnr(reference: Path, distorted: Path, fmt: YuvFormat) -> Measurement:
    return score_files(reference, distorted, fmt, 0.0, None, lambda spec: [PsnrMetric(spec)])


def xpsnr(reference: Path, distorted: Path, fmt: YuvFormat, fps: float,
          paths: Paths) -> Measurement:
    return score_files(reference, distorted, fmt, fps, paths, lambda spec: [XpsnrMetric(spec)])


def vmaf(reference: Path, distorted: Path, fmt: YuvFormat, paths: Paths) -> Measurement:
    return score_files(reference, distorted, fmt, 0.0, paths,
                       lambda spec: [legacy_vmaf_metric(spec)])


def vmaf_v1(reference: Path, distorted: Path, fmt: YuvFormat, paths: Paths,
            subsample: int = 1, display: bool = False) -> Measurement:
    """The pinned v1.0.16_3d0h model at native geometry, or with `display` on
    Netflix's 1080p 10-bit bicubic display protocol (models_v1.md)."""
    if display:
        return score_files(reference, distorted, fmt, 0.0, paths,
                           lambda spec: [VmafV1DisplayMetric(spec, subsample)])
    return score_files(reference, distorted, fmt, 0.0, paths,
                       lambda spec: vmaf_v1_and_ms_ssim(spec, vmaf_subsample=subsample))


def ms_ssim(reference: Path, distorted: Path, fmt: YuvFormat, paths: Paths) -> Measurement:
    return score_files(reference, distorted, fmt, 0.0, paths,
                       lambda spec: vmaf_v1_and_ms_ssim(spec, vmaf_v1=False, ms_ssim=True))


def all_metrics(reference: Path, distorted: Path, fmt: YuvFormat, fps: float,
                paths: Paths, profile: str = V1_PROFILE,
                include_ms_ssim: bool = False, include_xpsnr: bool | None = None) -> Measurement:
    return score_files(reference, distorted, fmt, fps, paths,
                       lambda spec: profile_metrics(spec, profile, include_ms_ssim,
                                                    include_xpsnr))


def _display_upscale_command(source: str, fmt: YuvFormat, paths: Paths, dest: str,
                             frames: int | None = None) -> list[str]:
    """ffmpeg bicubic upscale to the v1 display geometry, 10-bit (fixed protocol)."""
    width, height = vmaf_v1_geometry(fmt)
    limit = ["-frames:v", str(frames)] if frames is not None else []
    return [str(find_tool("ffmpeg", paths)), "-hide_banner", "-nostdin", "-v", "error",
            "-threads", "1", "-filter_threads", "1", "-f", "rawvideo",
            "-pixel_format", "yuv420p", "-video_size", f"{fmt.width}x{fmt.height}",
            "-i", source, *limit,
            "-vf", f"scale={width}:{height}:flags=bicubic,format=yuv420p10le",
            "-pix_fmt", "yuv420p10le", "-threads", "1", "-f", "rawvideo", "-y", dest]


def _vmaf_v1_command(reference_display: str, distorted_display: str, fmt: YuvFormat,
                     paths: Paths, subsample: int, out: Path) -> list[str]:
    model = paths.anchors / VMAF_V1_MODEL
    width, height = vmaf_v1_geometry(fmt)
    return [str(find_tool("vmaf", paths)), "-r", reference_display, "-d", distorted_display,
            "-w", str(width), "-h", str(height), "-p", "420", "-b", "10", "--model",
            f"path={model}:name=vmaf_v1:cambi.enc_width={fmt.width}:cambi.enc_height={fmt.height}:cambi.enc_bitdepth=8",
            "--threads", str(VMAF_THREADS), "--subsample", str(subsample),
            "--json", "-o", str(out), "-q"]


def _parse_vmaf_v1(report_path: Path) -> Measurement:
    return _parse_vmaf_v1_report(json.loads(report_path.read_text()))


def _parse_vmaf_v1_report(data: dict) -> Measurement:
    frames = sorted(data["frames"], key=lambda f: f["frameNum"])
    return Measurement({"vmaf_v1": float(data["pooled_metrics"]["vmaf_v1"]["mean"])},
                       [{"vmaf_v1": float(f["metrics"]["vmaf_v1"])} for f in frames])


def vmaf_v1_display_copy(source: Path, fmt: YuvFormat, dest: Path, paths: Paths) -> None:
    """The v1 display-geometry 10-bit copy of `source`, as a file.

    Only for callers that still keep display files (nceval.cbr); the point
    pipeline pipes the same command instead (`VmafV1DisplayMetric`).
    """
    subprocess.run(_display_upscale_command(str(source), fmt, paths, str(dest)),
                   stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)


def vmaf_v1_score(reference_display: Path, distorted_display: Path, fmt: YuvFormat,
                  paths: Paths, scratch: Path, subsample: int = 1) -> Measurement:
    """Score two `vmaf_v1_display_copy` outputs with the pinned v1 model."""
    out = scratch / f"{distorted_display.stem}.vmaf.json"
    subprocess.run(_vmaf_v1_command(str(reference_display), str(distorted_display), fmt, paths,
                                    subsample, out),
                   stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
    try:
        return _parse_vmaf_v1(out)
    finally:
        out.unlink()
