"""CBR rate-control benchmark: whole-sequence quality at a target bitrate.

A rate controller is scored as a rate-quality curve: each clip is coded at
several target bitrates (all at most half the source bitrate), and the
*achieved* bitrate is plotted against sequence PSNR-YUV and VMAF v1. Curves
are compared by BD-rate against

  - the same encoder at fixed QP (the cost of rate control itself; QPs are
    chosen to span the CBR curve's achieved rates), and
  - the reference encoders' own CBR modes (SVT-AV1 M12 --rtc for AV1,
    openh264 realtime bitrate mode for H.264).

Each CBR point also reports its rate error, strict-CBR buffer overruns,
worst-1% frame quality and frame-to-frame quality flicker (cbr_stats.py).

Every point is streamed (nceval.point): metrics are computed from the
encoder's reconstruction (or, for anchors, the first standard decoder); an
explicit --gate also hash-checks decoders against it (nceval.gatepolicy).
No video is written; each bitstream is deleted once its packet sizes are
read, and only per-point JSON results are kept.

usage (from eval/):
  python3 -m nceval.cbr run --binary BIN [--codecs h264,hevc,av1] [--frames 150]
      [--clips a,b] [--intra-period 300] [--workers 4] [--param name=value ...]
      [--label NAME [--baseline-binary BASE]] [--anchors | --no-anchors]
      [--clipset FILE] [--extra-anchors x264-veryfast,x265-superfast]

--extra-anchors adds x264 veryfast (H.264) and x265 superfast (HEVC) in their
own real-time ABR+VBV mode with their default AQ (contributed by the
adversarial review). SVT's CBR is also compared with our HEVC.

With --baseline-binary, --binary is a candidate: it is coded in CBR only and
judged against the baseline build's (cached) CBR and fixed-QP curves.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import json
import math
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from .bdrate import bd_rate
from .buildinfo import info_of
from .cachefile import entry_lock, get_or_create, read_json, sibling_temp, write_atomic
from .cbr_stats import flicker, replay_buffer, weighted_yuv_per_frame, worst_quality
from .clipframes import ClipFrames
from .clips import Clip, load_clipset
from .codecs import Codec
from .config import Paths, find_tool, load_paths
from .conformance import gate_stream
from . import svcbase
from .gatepolicy import (EVALUATION_GATE, GatePolicy, PointCheck, anchor_check, combination_key,
                         gate_arguments, level_satisfies, point_check)
from .diskguard import check_disk
from .encoders.base import fps_fraction
from .fingerprint import measurement_fingerprint
from .framepipe import ProcessFailed
from .framestats import coded_at_times, probe_frames, with_held_frames
from .metrics import (REAL_METRICS_REVISION, REAL_MS_SSIM_REVISION, MetricSession, PsnrMetric, StreamSpec,
                      gpu_slot_need, real_metrics, vmaf_v1_and_ms_ssim)
from .point import coded_format, run_point
from .slots import cpu_slots, gpu_slots, nvenc_slots
from .stages import STAGES, decide
from .sweep import PIXEL_RATE_EXPONENT, REFERENCE_PIXELS_PER_SECOND, payload_bytes
from .yuv import YuvFormat

#: Target bitrates for a 1280x720 30 fps clip; other clips scale by pixel rate.
REFERENCE_TARGETS_KBPS = (200, 350, 600, 1000, 1500)
#: Receiver buffer for every encoder, and the strict-CBR replay.
BUFFER_MS = 1000
#: SVT's starting and steady-state buffer levels (its RTC CBR examples use 60%).
SVT_BUFFER_LEVEL_MS = 600
DEFAULT_FRAMES = 150
DEFAULT_INTRA_PERIOD = 300  # 10 s at 30 fps: matches the BD-rate sweeps
DEFAULT_CLIPS = ("crowd_mot17_09", "pan_jaad_0117", "static_meva_hospital",
                 "lowlight_city_night", "handheld_mot17_11")
#: Coarse fixed-QP grids used to price the rate of each quantizer.
COARSE_GRIDS = {Codec.H264: range(18, 52, 3), Codec.HEVC: range(18, 52, 3),
                Codec.VVC: range(18, 64, 3),
                Codec.AV1: range(30, 256, 15), Codec.VP9: range(30, 256, 15),
                Codec.AV2: range(30, 256, 15)}
SVT_QP_GRID = range(20, 64, 3)
#: Extra fixed-QP points past each end of the CBR rate span, so the fixed-QP
#: curve covers the whole CBR curve for BD-rate.
CQP_SPAN_MARGIN = 1.25
CQP_POINTS = 4
#: A CBR point more than this far above its target did not reach it (the
#: encoder ran out of quantizer range): it is reported as unreachable and
#: left out of that family's BD curves, so controller cost is not confused
#: with an encoder's rate floor.
UNREACHABLE_RATE_ERROR = 0.10
#: Anchor programs (Family.kind); "" is our own encoder.
ANCHOR_KINDS = ("svt", "openh264", "x264", "x265", "vpx", "aom", "nvenc")
#: Each anchor program's preset when a family names none: the bars and context
#: anchors the board was seeded with. A family at its kind's default preset
#: keeps the cache identity it had before presets were configurable.
DEFAULT_ANCHOR_PRESETS = {"svt": "12", "x264": "veryfast", "x265": "superfast"}
#: Anchor family names that carry a preset (`<prefix><preset>`, e.g.
#: x264-medium, svtav1-m8), by kind.
#: libvpx / libaom presets are their realtime --cpu-used speeds (libvpx-vp9-cpu7).
ANCHOR_PRESET_PREFIXES = {"svt": "svtav1-m", "x264": "x264-", "x265": "x265-",
                          "vpx": "libvpx-vp9-cpu", "aom": "libaom-av1-cpu"}
ANCHOR_TOOLS = {"svt": "SvtAv1EncApp", "x264": "x264", "x265": "x265", "vpx": "vpxenc",
                "aom": "aomenc"}
ANCHOR_CODECS = {"svt": Codec.AV1, "x264": Codec.H264, "x265": Codec.HEVC, "vpx": Codec.VP9,
                 "aom": Codec.AV1}
#: NVIDIA NVENC through ffmpeg: fixed-function GPU hardware, a chart reference on the
#: quality axis only (its speed is not CPU speed). Family -> (codec, ffmpeg encoder,
#: preset, tune): the default preset with low-latency tuning, as streaming
#: services use it, and the slowest preset with high-quality tuning, NVENC's best
#: forward-only quality. They run in the ffmpeg the harness decodes with
#: (find_tool "ffmpeg"), whose libavcodec the encoder version is.
NVENC_FAMILIES = {
    "nvenc-h264": (Codec.H264, "h264_nvenc", "p4", "ll"),
    "nvenc-hevc": (Codec.HEVC, "hevc_nvenc", "p4", "ll"),
    "nvenc-av1": (Codec.AV1, "av1_nvenc", "p4", "ll"),
    "nvenc-h264-p7hq": (Codec.H264, "h264_nvenc", "p7", "hq"),
    "nvenc-hevc-p7hq": (Codec.HEVC, "hevc_nvenc", "p7", "hq"),
    "nvenc-av1-p7hq": (Codec.AV1, "av1_nvenc", "p7", "hq"),
}
#: The profile each NVENC codec is asked for (av1_nvenc has only main): H.264 High,
#: like x264 (h264_nvenc defaults to Main, without 8x8 transforms).
NVENC_PROFILES = {Codec.H264: ("-profile:v", "high"), Codec.HEVC: ("-profile:v", "main"),
                  Codec.AV1: ()}
#: ffmpeg muxer writing each codec's bitstream_suffix format.
NVENC_MUXERS = {Codec.H264: "h264", Codec.HEVC: "hevc", Codec.AV1: "ivf"}
WORK_SUBDIR = "ratecontrol/cbr"
METRICS = ("psnr_yuv", "vmaf_v1", "worst_psnr_yuv", "worst_vmaf_v1")
#: The metrics a staged verdict needs to agree on (nceval.screen.decide).
STAGE_METRICS = ("psnr_yuv", "vmaf_v1")
#: BD-rate points per curve: 4 normally; a staged proxy-4 stage uses 3.
MIN_BD_POINTS = 4


@dataclass(frozen=True)
class Family:
    """One encoder configuration that can run both CBR and fixed QP."""
    name: str
    codec: Codec
    tool: Path
    extra: tuple[str, ...] = ()  # e.g. --param pairs for nanocodec
    own: bool = False  # our encoder: reconstruction is also gated
    #: A candidate measured in CBR only, judged against the baseline family
    #: of the same codec (whose fixed-QP curve is shared).
    variant: bool = False
    #: Measure this family's fixed-QP curve (anchors: only on request).
    cqp: bool = True
    #: Anchors: which program (ANCHOR_KINDS) and which of our codecs it is a
    #: bar for. Empty for our own encoder.
    kind: str = ""
    compares: tuple[Codec, ...] = ()
    #: Anchors: the program's preset ("" = DEFAULT_ANCHOR_PRESETS[kind]).
    preset: str = ""

    @property
    def hardware(self) -> bool:
        """A GPU encoder (NVENC): it holds an NVENC slot while it encodes."""
        return self.kind == "nvenc"

    @property
    def anchor_preset(self) -> str:
        return self.preset or DEFAULT_ANCHOR_PRESETS[self.kind]

    def command(self, clip: Clip, frames: int, intra: int, out: Path, recon: Path | None,
                *, kbps: int | None = None, qp: int | None = None,
                layers: int = 1) -> list[str]:
        num, den = fps_fraction(clip.fps)
        common = {"w": str(clip.width), "h": str(clip.height), "n": str(frames)}
        if self.own:
            mode = ["--bitrate", str(kbps), "--buffer-ms", str(BUFFER_MS)] if kbps else \
                ["--qp", str(qp)]
            cmd = [str(self.tool), "encode", "--codec", self.codec.value, "--input",
                   str(clip.yuv), "--width", common["w"], "--height", common["h"],
                   "--fps", f"{num}/{den}", "--frames", common["n"], "--intra-period",
                   str(intra), "--temporal-layers", str(layers), "--output", str(out), *mode,
                   *self.extra]
            return cmd + (["--display-recon", str(recon)] if recon else [])
        if self.kind in ("x264", "x265"):
            return self._x26x_command(clip, frames, intra, out, kbps, qp, layers)
        if self.kind in ("vpx", "aom"):
            return self._libvpx_aom_command(clip, frames, intra, out, kbps, qp, layers)
        if self.kind == "nvenc":
            return self._nvenc_command(clip, frames, intra, out, kbps, layers)
        if self.kind == "svt":
            # Fixed QP mirrors the svtav1-m12 sweep adapter (AQ off); CBR keeps
            # SVT's own RTC CBR defaults (AQ included): that is the bar.
            mode = ["--rc", "2", "--tbr", str(kbps), "--buf-sz", str(BUFFER_MS),
                    "--buf-initial-sz", str(SVT_BUFFER_LEVEL_MS),
                    "--buf-optimal-sz", str(SVT_BUFFER_LEVEL_MS)] if kbps else \
                ["--rc", "0", "--aq-mode", "0", "--qp", str(qp)]
            return [str(self.tool), "--rtc", "1", "--preset", self.anchor_preset,
                    "--pred-struct", "1",
                    "--hierarchical-levels", str(layers - 1), *mode, "--keyint", str(intra),
                    "--scd", "0",
                    "--lp", "1", "--fps-num", str(num), "--fps-denom", str(den),
                    "-w", common["w"], "-h", common["h"], "-n", common["n"],
                    "-i", str(clip.yuv), "-b", str(out)]
        # openh264-baseline settings (encoders/openh264.py), frame skip off so
        # every frame is coded and scored.
        mode = ["--bitrate", str(kbps)] if kbps else ["--qp", str(qp)]
        return [str(self.tool), "--input", str(clip.yuv), "--output", str(out),
                "--width", common["w"], "--height", common["h"], "--fps", f"{clip.fps:g}",
                "--frames", common["n"], "--usage", "camera-realtime", "--complexity", "medium",
                "--cabac", "0", *mode, "--intra-period", str(intra), "--temporal-layers", str(layers),
                "--adaptive-quant", "0", "--scene-change", "0", "--background-detection", "1",
                "--denoise", "0", "--ltr", "0", "--frame-skip", "0", "--threads", "1"]

    def _x26x_command(self, clip: Clip, frames: int, intra: int, out: Path,
                      kbps: int | None, qp: int | None, layers: int) -> list[str]:
        """x264 / x265 at the family's preset, zero latency, flat P.

        CBR is their ABR with a VBV of the shared buffer (their usual
        real-time CBR), AQ at its default; fixed QP mirrors the sweep
        adapters (encoders/anchors.py).
        """
        if layers != 1:
            raise ValueError(f"{self.name}: only flat P is configured")
        num, den = fps_fraction(clip.fps)
        mode = (["--bitrate", str(kbps), "--vbv-maxrate", str(kbps),
                 "--vbv-bufsize", str(round(kbps * BUFFER_MS / 1000))] if kbps
                else ["--qp", str(qp)])
        common = ["--tune", "zerolatency", "--bframes", "0", "--rc-lookahead", "0",
                  "--no-scenecut", "--keyint", str(intra), "--min-keyint", str(intra),
                  *mode, "--input-res", f"{clip.width}x{clip.height}",
                  "--fps", f"{num}/{den}", "--frames", str(frames)]
        if self.kind == "x264":
            return [str(self.tool), "--preset", self.anchor_preset, *common, "--threads", "1",
                    "--output", str(out), str(clip.yuv)]
        return [str(self.tool), "--preset", self.anchor_preset, *common, "--frame-threads", "1",
                "--pools", "none", "--no-wpp", "--input-depth", "8",
                "--input", str(clip.yuv), "--output", str(out)]

    def _libvpx_aom_command(self, clip: Clip, frames: int, intra: int, out: Path,
                            kbps: int | None, qp: int | None, layers: int) -> list[str]:
        """libvpx VP9 / libaom AV1 in their realtime mode (as WebRTC uses them).

        One pass, no lag, one thread, `--cpu-used` = the family's preset. CBR
        uses the shared 1 s buffer with SVT's 60% start/optimal levels; fixed
        QP pins the quantizer (`--end-usage=q`, min = max = cq level).
        """
        if layers != 1:
            raise ValueError(f"{self.name}: only flat P is configured")
        num, den = fps_fraction(clip.fps)
        mode = (["--end-usage=cbr", f"--target-bitrate={kbps}", f"--buf-sz={BUFFER_MS}",
                 f"--buf-initial-sz={SVT_BUFFER_LEVEL_MS}",
                 f"--buf-optimal-sz={SVT_BUFFER_LEVEL_MS}"] if kbps
                else ["--end-usage=q", f"--cq-level={qp}", f"--min-q={qp}", f"--max-q={qp}",
                      "--disable-warning-prompt"])  # min = max q is a warning, not an error
        codec = ["--codec=vp9", "--rt"] if self.kind == "vpx" else ["--rt"]
        return [str(self.tool), *codec, "--passes=1", "--lag-in-frames=0",
                "--error-resilient=0", "--threads=1", f"--cpu-used={self.preset}", *mode,
                f"--kf-min-dist={intra}", f"--kf-max-dist={intra}", "--i420",
                f"--width={clip.width}", f"--height={clip.height}", f"--fps={num}/{den}",
                f"--limit={frames}", "--ivf", "-o", str(out), str(clip.yuv)]

    def _nvenc_command(self, clip: Clip, frames: int, intra: int, out: Path,
                       kbps: int | None, layers: int) -> list[str]:
        """ffmpeg's NVENC encoder (`extra`: encoder, preset, tune) in low-delay CBR.

        Like the x264 anchor: CBR with a peak of the target and the shared
        1 s buffer, no B-frames, no lookahead, no scene-cut keys, the fixed key
        interval, frames output as they are coded; NVENC's own defaults
        otherwise (adaptive quantisation off). CBR only: no fixed-QP curve.
        """
        if layers != 1 or not kbps:
            raise ValueError(f"{self.name}: only flat-P CBR is configured")
        num, den = fps_fraction(clip.fps)
        rate = f"{kbps}k"
        return [str(self.tool), "-hide_banner", "-nostdin", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{clip.width}x{clip.height}",
                "-framerate", f"{num}/{den}", "-i", str(clip.yuv), "-frames:v", str(frames),
                *self.extra, *NVENC_PROFILES[self.codec],
                "-rc", "cbr", "-b:v", rate, "-maxrate", rate,
                "-bufsize", f"{round(kbps * BUFFER_MS / 1000)}k",
                "-bf", "0", "-rc-lookahead", "0", "-no-scenecut", "1", "-g", str(intra),
                "-zerolatency", "1", "-delay", "0",
                "-f", NVENC_MUXERS[self.codec], "-y", str(out)]

    @property
    def quantizer_grid(self) -> range:
        return COARSE_GRIDS[self.codec] if self.own else (
            SVT_QP_GRID if self.kind == "svt" else COARSE_GRIDS[Codec.H264])


@dataclass(frozen=True)
class Point:
    family: Family
    clip: Clip
    kbps: int | None = None  # CBR target
    qp: int | None = None  # fixed quantizer
    #: The decoders that check this point (nceval.gatepolicy).
    check: PointCheck = PointCheck.STANDARD

    @property
    def label(self) -> str:
        return f"cbr{self.kbps}" if self.kbps else f"qp{self.qp}"


@dataclass
class Bench:
    paths: Paths
    frames: int
    intra: int
    work: Path
    #: Temporal layers (1 flat; 2/3 dyadic hierarchical P; SVT --hierarchical-levels n-1).
    layers: int = 1
    tool_hashes: dict[Path, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)  # shared caches
    #: The metric protocol's fingerprint (nceval.fingerprint), part of every
    #: scored point's identity.
    measurement: str = field(default_factory=measurement_fingerprint)
    #: "classic" (PSNR + VMAF v1) or "real" (nceval.realuse: + MS-SSIM and
    #: the per-frame series J_real's temporal terms need).
    metric_set: str = "classic"
    #: Reuse a stored real-use result whatever its MS-SSIM protocol (the
    #: published chart: families a refresh does not touch keep their stored
    #: numbers; every-4th-frame vs every-frame MS-SSIM barely moves a
    #: sequence mean, coordinator 2026-11-11). Points not stored are always
    #: measured on the current protocol.
    stored_ms_ssim: bool = False
    #: Streams encoded elsewhere (tools/dev/aws_heldout_chart.py: a rented machine
    #: encodes, only bitstreams come back): every point's encode is taken from
    #: `prebuilt/<prebuilt_key>` instead of run, and a point without one fails.
    #: Scoring is the anchors' path for every family: metrics from the first
    #: standard decoder, and our decodes must equal the encoder's displayed
    #: pictures (their per-frame digests came with the stream).
    prebuilt: Path | None = None

    def tool_hash(self, family: Family) -> str:
        with self.lock:
            if family.tool not in self.tool_hashes:
                self.tool_hashes[family.tool] = hashlib.sha256(
                    family.tool.read_bytes()).hexdigest()
            return self.tool_hashes[family.tool]

    def identity(self, family: Family, clip: Clip, label: str, scored: bool = True) -> str:
        """Cache key: binary content, settings and input; never file paths.

        Scored points also key on the metric protocol (nceval's measurement
        fingerprint), so results from a changed VMAF protocol are never
        compared with older ones; rate probes carry no metrics.
        """
        parts = [self.tool_hash(family), family.extra, clip.name, clip.yuv.stat().st_size,
                 self.frames, self.intra, label, BUFFER_MS, SVT_BUFFER_LEVEL_MS]
        if clip.start_frame:  # clips measured from frame 0 keep their identities
            parts.append(f"start={clip.start_frame}")
        if self.layers != 1:  # flat points keep their existing identities
            parts.append(f"layers={self.layers}")
        if family.preset and family.preset != DEFAULT_ANCHOR_PRESETS.get(family.kind):
            parts.append(f"preset={family.preset}")  # default presets keep their identity
        if self.spatial(family):  # their results carry the base layer too
            parts.append("svc-base")
        if scored:
            parts.append(self.measurement)
            if self.metric_set != "classic":  # classic points keep their identities
                parts.append(f"metrics={self.metric_set}-{REAL_METRICS_REVISION}")
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]

    def prebuilt_key(self, point: Point) -> str:
        """A point's files under `prebuilt` (and the remote job's id): the encode's
        own identity (no metric protocol), so they serve any scoring of it."""
        identity = self.identity(point.family, point.clip, point.label, scored=False)
        return f"{point.clip.name}/{point.family.name}/{point.label}-{identity}"

    def point_dir(self, family: Family, clip: Clip) -> Path:
        directory = self.work / clip.name / family.name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def frames_of(self, clip: Clip) -> ClipFrames:
        """The point's input frames (nceval.clipframes: one decode, any readers)."""
        return ClipFrames(clip, self.frames, self.work / "frames")

    @staticmethod
    def spatial(family: Family) -> bool:
        """Our two-layer settings (spatial SVC): their points also score the base
        layer (nceval.svcbase)."""
        return family.own and svcbase.is_spatial(family.extra)

    def command(self, point: Point, out: Path, recon: str | None,
                rate_log: Path | None = None, coded_times: Path | None = None,
                source: ClipFrames | None = None, layer_md5: Path | None = None) -> list[str]:
        """The encode command; `source` serves its input (else the clip's own file)."""
        clip = replace(point.clip, yuv=source.reader()) if source else point.clip
        cmd = point.family.command(clip, self.frames, self.intra, out, recon,
                                   kbps=point.kbps, qp=point.qp, layers=self.layers)
        return (cmd + (["--rate-log", str(rate_log)] if rate_log else [])
                + (["--timestamps-out", str(coded_times)] if coded_times else [])
                + (["--layer-md5", str(layer_md5)] if layer_md5 else []))

    def encode(self, point: Point, out: Path, rate_log: Path | None = None,
               source: ClipFrames | None = None) -> None:
        """Encode without a reconstruction (rate probes and anchors).

        `source`: the point's frames when the caller also scores them."""
        with contextlib.ExitStack() as stack:
            if source is None:
                source = stack.enter_context(self.frames_of(point.clip))
            cmd = self.command(point, out, None, rate_log, source=source)
            # the NVENC slot first, like GPU slots (nceval.slots)
            with nvenc_slots() if point.family.hardware else contextlib.nullcontext(), \
                    cpu_slots(1):
                proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                                      text=True)
        if proc.returncode != 0:
            tail = " | ".join((proc.stderr or proc.stdout).strip().splitlines()[-3:])
            raise RuntimeError(f"{point.family.name} {point.clip.name} {point.label}: {tail}")

    def rate_kbps(self, point: Point, bitstream: Path) -> float:
        bits = payload_bytes(bitstream, point.family.codec, self.frames) * 8
        return bits * point.clip.fps / self.frames / 1000.0

    def price(self, family: Family, clip: Clip, qp: int) -> float:
        """Achieved rate of one fixed-QP encode, without gating or scoring."""
        point = Point(family, clip, qp=qp)
        cache = self.point_dir(family, clip) / f"rate-{point.label}-" \
            f"{self.identity(family, clip, point.label, scored=False)}.json"

        def encode() -> float:
            with sibling_temp(cache, family.codec.bitstream_suffix) as out:
                self.encode(point, out)
                return self.rate_kbps(point, out)
        return get_or_create(cache, lambda path: read_json(path, "kbps"), encode,
                             lambda kbps: json.dumps({"kbps": kbps}))

    def result_path(self, point: Point) -> Path:
        return self.point_dir(point.family, point.clip) / \
            f"{point.label}-{self.identity(point.family, point.clip, point.label)}.json"

    def cached(self, point: Point) -> bool:
        """Whether `point` is already measured and reusable (for pricing a run)."""
        return self._reusable(point, self.result_path(point)) is not None

    def components(self, spec: StreamSpec) -> list:
        return real_metrics(spec) if self.metric_set == "real" else metric_components(spec)

    def measure(self, point: Point) -> dict:
        """Encode, gate, score and summarize one point (cached by identity).

        Streamed (nceval.point): our encoder's reconstruction is scored while
        it encodes and checked by the decoders `point.check` asks for;
        anchors, which output no reconstruction, are scored from the first
        standard decoder (the gate, when it runs). Only the bitstream is
        written, to a private temporary name, and it is deleted once its
        packet sizes are read. Concurrent requests for one point (two
        agents' screens sharing an anchor) encode it once (nceval.cachefile).
        """
        result_path = self.result_path(point)
        cached = self._reusable(point, result_path)
        if cached is not None:
            return cached
        with entry_lock(result_path):
            cached = self._reusable(point, result_path)
            if cached is not None:
                return cached
            result = self._measure_uncached(point, result_path)
            write_atomic(result_path, json.dumps(result, indent=1))
            return result

    def _reusable(self, point: Point, result_path: Path) -> dict | None:
        """The cached result, unless missing, this run needs a stronger gate,
        or (real-use set) its MS-SSIM is of an older protocol: such a point
        is re-measured lazily, when used (metrics.REAL_MS_SSIM_REVISION).
        The reconstruction is not kept, so MS-SSIM cannot be re-scored
        alone; the deterministic re-encode reproduces PSNR and VMAF v1."""
        cached = read_json(result_path)
        if cached is None or not level_satisfies(cached.get("gate_level", ""), point.check):
            return None
        if self.metric_set == "real" and not self.stored_ms_ssim and \
                cached.get("ms_ssim_revision", 1) != REAL_MS_SSIM_REVISION:
            return None
        return cached

    def _measure_uncached(self, point: Point, result_path: Path) -> dict:
        family, clip = point.family, point.clip
        check_disk(self.paths.work)
        fmt = YuvFormat(clip.width, clip.height)
        if clip.clean_reference and not clip.direct:
            raise ValueError(f"{clip.name}: a clean reference needs a raw clip from frame 0")
        # Our encoder says which input frames it coded (holds, drops) and, in
        # CBR, logs the key loan still outstanding at the end.
        spatial = self.spatial(family)
        if spatial and self.prebuilt is not None:
            raise ValueError(f"{family.name}: prebuilt streams carry no layer log")
        with self.frames_of(clip) as source, \
                sibling_temp(result_path, family.codec.bitstream_suffix) as stream, \
                sibling_temp(result_path, ".rate.csv") as rate_csv, \
                sibling_temp(result_path, ".times.txt") as coded_times, \
                sibling_temp(result_path, ".layers.csv") as layer_md5:
            spec = StreamSpec(clip.clean_reference or source, fmt, self.frames, clip.fps,
                              self.paths)
            rate_log = rate_csv if family.own and point.kbps else None
            if self.prebuilt is not None:
                gate, measured = self._score_prebuilt(point, stream, rate_log, coded_times,
                                                      spec)
            elif family.own:
                outcome = run_point(lambda recon: self.command(
                                        point, stream, recon, rate_log, coded_times, source,
                                        layer_md5 if spatial else None),
                                    stream,
                                    family.codec, spec, self.paths, self.components(spec),
                                    rate_log=rate_log, **gate_arguments(point.check, family.tool))
                gate, measured = outcome.gate, outcome.measurement
            else:
                self.encode(point, stream, source=source)
                components = self.components(spec)
                # GPU slots before the gate's CPU slots (nceval.slots).
                with gpu_slots(gpu_slot_need(components)), \
                        MetricSession(spec, components) as session:
                    gate = gate_stream(stream, family.codec, fmt, self.frames, self.paths,
                                       extra_sinks=session.sinks,
                                       extra_threads=session.cpu_threads,
                                       standard=point.check is PointCheck.STANDARD)
                    measured = session.result(self.frames) if gate.passed else None
            if not gate.passed or measured is None:
                raise RuntimeError(f"conformance failed: {family.name} {clip.name} "
                                   f"{point.label}: {gate.reason}")
            coded = coded_at_times(coded_times, clip.fps, self.frames) if family.own else None
            frames = with_held_frames(probe_frames(stream, self.paths), coded)
            sizes = [f.bytes * 8 for f in frames]
            loan_bits = final_key_loan_bits(rate_log) if rate_log else None
            base = self._base_layer(point, stream, layer_md5, coded, source) \
                if spatial else None
        result = summarize(point, self.rate_kbps_from_sizes(point, sizes), sizes,
                           measured.summary["psnr_yuv"], measured.per_frame,
                           measured.summary["vmaf_v1"],
                           [f["vmaf_v1"] for f in measured.per_frame])
        if loan_bits is not None:
            add_loan_adjusted_rate(result, loan_bits, point.clip.fps)
        keys = [f.frame_type == "I" for f in frames]
        if self.metric_set == "real":
            add_real_series(result, measured, keys)
        if base is not None:
            result["base"] = self._base_result(point, *base, keys)
        result["gate_level"] = gate.level
        if family.own:  # which build this point measured (nceval.buildinfo)
            result["binary_commit"] = info_of(family.tool)["commit"]
        return result

    def _base_layer(self, point: Point, stream: Path, layer_md5: Path,
                    coded: list[bool] | None, source: ClipFrames) -> tuple[list[int], object]:
        """A two-layer point's base layer (operating point 1): its per-input-frame
        bits and its scores at the base size (nceval.svcbase)."""
        coded = coded or [True] * self.frames
        bits = svcbase.per_input(svcbase.stream_layer_bits(stream, point.family.codec)[1],
                                 coded)
        measured = svcbase.own_base_layer(stream, point.family.codec, point.family.tool,
                                          layer_md5, point.clip, self.frames, coded,
                                          self.work / "frames", self.paths, source)
        return bits, measured

    def _base_result(self, point: Point, bits: list[int], measured, keys: list[bool]) -> dict:
        """The base layer's result, in the point's own layout, at its own rate against
        its share of the target (svcbase.BASE_RATE_SHARE): what a base-only
        receiver gets."""
        base_point = replace(point, kbps=svcbase.split_rate(point.kbps)[0]) if point.kbps \
            else point
        result = summarize(base_point, sum(bits) * point.clip.fps / self.frames / 1000.0, bits,
                           measured.summary["psnr_yuv"], measured.per_frame,
                           measured.summary["vmaf_v1"],
                           [f["vmaf_v1"] for f in measured.per_frame])
        if self.metric_set == "real":
            add_real_series(result, measured, keys)
        return result

    def _score_prebuilt(self, point: Point, stream: Path, rate_log: Path | None,
                        coded_times: Path, spec: StreamSpec):
        """Gate and score a stream encoded elsewhere (see `prebuilt`); its rate log
        and coded-frame times are copied to where the local encode puts them."""
        family = point.family
        base = self.prebuilt / self.prebuilt_key(point)
        failed = Path(f"{base}.failed")
        if not Path(f"{base}.done").exists() and failed.exists():
            reason = " | ".join(json.loads(failed.read_text())["stderr"][-2:])
            error = ProcessFailed if family.own else RuntimeError  # ours: refused (Runner)
            raise error(f"{family.name} {point.clip.name} {point.label} (prebuilt): {reason}")
        if not Path(f"{base}.done").exists():
            raise RuntimeError(f"no prebuilt stream for {family.name} {point.clip.name} "
                               f"{point.label}: {base}")
        shutil.copyfile(f"{base}{family.codec.bitstream_suffix}", stream)
        expected = None
        if family.own:
            shutil.copyfile(f"{base}.times.txt", coded_times)
            if rate_log:
                shutil.copyfile(f"{base}.rate.csv", rate_log)
            expected = Path(f"{base}.recon.txt").read_text().split()
        components = self.components(spec)
        # GPU slots before the gate's CPU slots (nceval.slots).
        with gpu_slots(gpu_slot_need(components)), MetricSession(spec, components) as session:
            # frames coded below the stream size (adaptive resolution) are compared
            # as the player shows them, as in nceval.point
            gate = gate_stream(stream, family.codec, spec.fmt, self.frames, self.paths,
                               expected_hashes=expected, extra_sinks=session.sinks,
                               extra_threads=session.cpu_threads,
                               coded_sizes=coded_format(rate_log, spec.fmt) if family.own
                               else None,
                               **gate_arguments(point.check, family.tool if family.own
                                                else None))
            return gate, session.result(self.frames) if gate.passed else None

    def rate_kbps_from_sizes(self, point: Point, sizes: list[int]) -> float:
        if len(sizes) != self.frames:
            raise RuntimeError(f"{point.family.name} {point.clip.name}: ffprobe saw "
                               f"{len(sizes)} packets, expected {self.frames}")
        return sum(sizes) * point.clip.fps / self.frames / 1000.0


def final_key_loan_bits(rate_log: Path) -> int:
    """The key loan still outstanding after the last frame (nanocodec --rate-log)."""
    with rate_log.open(newline="") as log:
        rows = list(csv.DictReader(log))
    if not rows:
        raise RuntimeError(f"{rate_log}: empty rate log")
    return int(rows[-1]["key_loan_bits"])


def add_loan_adjusted_rate(result: dict, loan_bits: int, fps: float) -> None:
    """Rate error with the outstanding key loan excluded.

    By design our controller lends a key frame's excess bits and repays them
    over `rate.key_repay_seconds`; a run shorter than that ends with part of
    the loan unpaid, which reads as overshoot in the raw rate error. Same
    sign convention as `rate_error`; anything left over is a real overshoot.
    """
    frames = len(result["frame_bits"])
    repaid_kbps = (sum(result["frame_bits"]) - loan_bits) * fps / frames / 1000.0
    result.update(key_loan_bits=loan_bits,
                  loan_adjusted_rate_error=repaid_kbps / result["target_kbps"] - 1.0)


def reachable(point: dict) -> bool:
    return overshoot(point) <= UNREACHABLE_RATE_ERROR


def overshoot(point: dict) -> float:
    """How far a CBR point went over its target (0 when under).

    The target bitrate is a ceiling, not a rate to hit (Mark, 2026-11-08):
    the encoder may underspend when bits stop buying quality, so only
    overshoot is an error. Our points exclude the by-design key loan.
    """
    error = point.get("loan_adjusted_rate_error", point.get("rate_error", 0.0))
    return max(0.0, error)


def metric_components(spec: StreamSpec) -> list:
    """Per-frame PSNR and every-frame VMAF v1 (worst-1% needs every frame)."""
    return [PsnrMetric(spec), *vmaf_v1_and_ms_ssim(spec)]


def add_real_series(result: dict, measured, keys: list[bool]) -> None:
    """The real-use metrics (nceval.realuse): MS-SSIM, per-frame PSNR-Y, keys
    (per-frame VMAF v1 is in every result: summarize)."""
    if len(keys) != len(measured.per_frame):
        raise RuntimeError(f"{len(keys)} frame types for {len(measured.per_frame)} frames")
    result.update(ms_ssim=measured.summary["ms_ssim"], ms_ssim_revision=REAL_MS_SSIM_REVISION,
                  frame_psnr_y=[f["psnr_y"] for f in measured.per_frame], frame_key=keys)


def summarize(point: Point, kbps: float, sizes: list[int], psnr_yuv: float,
              psnr_frames: list[dict], vmaf_mean: float, vmaf_frames: list[float]) -> dict:
    per_yuv = weighted_yuv_per_frame([f["psnr_y"] for f in psnr_frames],
                                     [f["psnr_u"] for f in psnr_frames],
                                     [f["psnr_v"] for f in psnr_frames])
    result = {"family": point.family.name, "clip": point.clip.name, "label": point.label,
              "kbps": kbps, "psnr_yuv": psnr_yuv, "vmaf_v1": vmaf_mean,
              "worst_psnr_yuv": worst_quality(per_yuv), "worst_vmaf_v1": worst_quality(vmaf_frames),
              "flicker_psnr_y": flicker([f["psnr_y"] for f in psnr_frames]),
              "flicker_vmaf_v1": flicker(vmaf_frames), "frame_bits": sizes,
              "frame_psnr_yuv": per_yuv, "frame_vmaf_v1": vmaf_frames}
    if point.kbps:
        replay = replay_buffer(sizes, point.kbps * 1000.0, point.clip.fps, BUFFER_MS)
        result.update(target_kbps=point.kbps, rate_error=kbps / point.kbps - 1.0,
                      overruns=replay.overruns, max_fullness=replay.max_fullness,
                      max_window_ratio=replay.max_window_ratio)
    return result


def clip_targets(clip: Clip, reference: tuple[int, ...] = REFERENCE_TARGETS_KBPS,
                 indices: tuple[int, ...] | None = None) -> list[int]:
    """`reference` targets (720p30 kbps) scaled to the clip's pixel rate, within its cap.

    `indices` keeps a staged subset of the clip's own legal targets (e.g. the
    outer three of four); a clip with fewer legal targets keeps them all.
    """
    scale = (clip.width * clip.height * clip.fps / REFERENCE_PIXELS_PER_SECOND) \
        ** PIXEL_RATE_EXPONENT
    legal = [round(t * scale) for t in reference if clip.allows_kbps(t * scale)]
    if indices is None or len(legal) < len(reference):
        return legal
    return [legal[i] for i in indices]


def cqp_quantizers(rates: dict[int, float], cbr_kbps: list[float]) -> list[int]:
    """Fixed QPs whose rates span the achieved CBR rates, evenly in log rate.

    Four points are the BD-rate minimum; the ends sit CQP_SPAN_MARGIN
    beyond the CBR curve so the overlap covers all of it.
    """
    low, high = min(cbr_kbps) / CQP_SPAN_MARGIN, max(cbr_kbps) * CQP_SPAN_MARGIN
    goals = [low * (high / low) ** (i / (CQP_POINTS - 1)) for i in range(CQP_POINTS)]
    chosen = {min(rates, key=lambda q: abs(math.log(rates[q] / g))) for g in goals}
    return sorted(chosen)


def families(args: argparse.Namespace, paths: Paths) -> list[Family]:
    binary = Path(args.binary)
    extra = tuple(x for p in args.param for x in ("--param", p))
    extra += ("--preset", args.preset) if args.preset else ()
    suffix = f"@{args.label}" if args.label else ""
    chosen = []
    variant = args.baseline_binary is not None
    for codec in (Codec(c) for c in args.codecs.split(",")):
        if variant:
            base_extra = tuple(x for p in args.baseline_param for x in ("--param", p))
            chosen.append(Family(f"nanocodec-{codec.value}", codec, Path(args.baseline_binary),
                                 base_extra, own=True, cqp=args.cqp))
        chosen.append(Family(f"nanocodec-{codec.value}{suffix}", codec, binary, extra, own=True,
                             variant=variant, cqp=args.cqp))
    if args.anchors:
        ours = {Codec(c) for c in args.codecs.split(",")}
        chosen += [f for f in anchor_families(paths, Path(args.openh264_harness),
                                              args.anchor_cqp, args.extra_anchors.split(","))
                   if ours & set(f.compares)]
    return chosen


def anchor_families(paths: Paths, openh264_harness: Path, cqp: bool = False,
                    extra: list[str] = ()) -> list[Family]:
    """The anchors' own real-time rate control: SVT M12 --rtc CBR (the AV1 and
    HEVC bar), openh264 RC (the H.264 bar) and, in `extra`, preset anchors by
    name (preset_anchor; the board's context anchors are x264-veryfast and
    x265-superfast)."""
    families = [
        Family("svtav1-m12", Codec.AV1, find_tool("SvtAv1EncApp", paths), cqp=cqp, kind="svt",
               compares=(Codec.AV1, Codec.HEVC)),
        Family("openh264-baseline", Codec.H264, openh264_harness, cqp=cqp, kind="openh264",
               compares=(Codec.H264,)),
    ]
    return families + [preset_anchor(name, paths, cqp) for name in extra if name]


def preset_anchor(name: str, paths: Paths, cqp: bool = False) -> Family:
    """An x264 / x265 / SVT-AV1 family by name (`x264-medium`, `x265-ultrafast`,
    `svtav1-m8`): the program's real-time CBR (or fixed QP) at that preset; or an
    NVENC family (NVENC_FAMILIES, CBR only)."""
    if name in NVENC_FAMILIES:
        return nvenc_family(name, paths)
    for kind, prefix in ANCHOR_PRESET_PREFIXES.items():
        if name.startswith(prefix) and len(name) > len(prefix):
            codec = ANCHOR_CODECS[kind]
            compares = (Codec.AV1, Codec.HEVC) if kind == "svt" else (codec,)
            return Family(name, codec, find_tool(ANCHOR_TOOLS[kind], paths), cqp=cqp, kind=kind,
                          compares=compares, preset=name.removeprefix(prefix))
    raise ValueError(f"unknown anchor family {name!r}: expected "
                     f"{', '.join(p + '<preset>' for p in ANCHOR_PRESET_PREFIXES.values())}")


def nvenc_family(name: str, paths: Paths) -> Family:
    """An NVENC_FAMILIES family. Its `extra` (the encoder, preset and tune
    arguments) keys its cache identity: the three codecs share one ffmpeg."""
    codec, encoder, preset, tune = NVENC_FAMILIES[name]
    return Family(name, codec, find_tool("ffmpeg", paths), extra=("-c:v", encoder, "-preset", preset,
                                                    "-tune", tune),
                  cqp=False, kind="nvenc", compares=(codec,))


def curve_points(bench: Bench, family: Family, clip: Clip, policy: GatePolicy, *,
                 targets: list[int] = (), quantizers: list[int] = ()) -> list[Point]:
    """One curve's points, each with the check the gate policy gives it."""
    count = len(targets) + len(quantizers)
    combination = combination_key(bench.tool_hash(family), {"args": " ".join(family.extra)},
                                  clip.name)

    def check(index: int) -> PointCheck:
        return (point_check(policy, combination, index, count) if family.own
                else anchor_check(policy))
    return ([Point(family, clip, kbps=t, check=check(i)) for i, t in enumerate(targets)]
            + [Point(family, clip, qp=q, check=check(len(targets) + i))
               for i, q in enumerate(quantizers)])


def measure_all(bench: Bench, fams: list[Family], clips: list[Clip], workers: int,
                policy: GatePolicy, targets: tuple[int, ...],
                indices: tuple[int, ...] | None = None) -> tuple[dict, dict]:
    """All CBR points, then fixed-QP points chosen to span each CBR curve."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cbr_jobs = {(f.name, c.name, p.kbps): pool.submit(bench.measure, p)
                    for f in fams for c in clips
                    for p in curve_points(bench, f, c, policy, targets=clip_targets(c, targets, indices))}
        price_jobs = {(f.name, c.name, q): pool.submit(bench.price, f, c, q)
                      for f in fams if f.cqp and not f.variant for c in clips
                      for q in f.quantizer_grid}
        cbr = {k: j.result() for k, j in cbr_jobs.items()}
        rates = {k: j.result() for k, j in price_jobs.items()}
        cqp_jobs = {}
        for f in (f for f in fams if f.cqp and not f.variant):
            for c in clips:
                grid = {q: rates[(f.name, c.name, q)] for q in f.quantizer_grid}
                achieved = [cbr[(f.name, c.name, t)]["kbps"]
                            for t in clip_targets(c, targets, indices)]
                for p in curve_points(bench, f, c, policy,
                                      quantizers=cqp_quantizers(grid, achieved)):
                    cqp_jobs[(f.name, c.name, p.qp)] = pool.submit(bench.measure, p)
        cqp = {k: j.result() for k, j in cqp_jobs.items()}
    return cbr, cqp


def run(args: argparse.Namespace) -> dict:
    paths = load_paths()
    layers = f"-L{args.temporal_layers}" if args.temporal_layers != 1 else ""
    work = paths.work / WORK_SUBDIR / f"f{args.frames}-ip{args.intra_period}{layers}"
    bench = Bench(paths, args.frames, args.intra_period, work, args.temporal_layers)
    clipset = load_clipset(Path(args.clipset) if args.clipset else paths.clips / "main.json",
                           paths.data_root)
    by_name = {c.name: c for c in clipset.clips}
    fams = families(args, paths)
    targets = tuple(int(t) for t in args.targets.split(","))
    if args.staged:
        return run_staged(args, bench, fams, by_name, targets)
    clips = select_clips(by_name, args.clips.split(","))
    report = measure_and_report(args, bench, fams, clips, targets)
    print_report(report)
    return report


def select_clips(by_name: dict[str, Clip], names: list[str] | tuple[str, ...]) -> list[Clip]:
    clips = [by_name[n] for n in names]
    for clip in clips:
        if clip.split == "test":
            raise ValueError(f"{clip.name} is a sealed TEST clip")
    return clips


def measure_and_report(args: argparse.Namespace, bench: Bench, fams: list[Family],
                       clips: list[Clip], targets: tuple[int, ...], suffix: str = "",
                       indices: tuple[int, ...] | None = None) -> dict:
    cbr, cqp = measure_all(bench, fams, clips, args.workers, GatePolicy(args.gate), targets,
                           indices)
    count = len(indices) if indices is not None else len(targets)
    report = build_report(fams, clips, cbr, cqp, min_points=min(count, MIN_BD_POINTS))
    out = bench.work / f"report{'-' + args.label if args.label else ''}{suffix}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"report: {out}")
    return report


def run_staged(args: argparse.Namespace, bench: Bench, fams: list[Family],
               by_name: dict[str, Clip], targets: tuple[int, ...]) -> dict:
    """A candidate screen in nceval.screen's TRAIN stages (proxy-4 on the outer
    three targets, then proxy-6 on all four), stopping once every candidate
    comparison against base CBR is clearly better or any is clearly worse.
    The stage sigmas are nceval.screen's fixed-QP ones (provisional for CBR).
    """
    # screen -> tune -> tune_realuse -> realuse -> cbr: imported here, not at
    # module level, so that chain never meets a half-initialised cbr.
    if not any(f.variant for f in fams):
        raise ValueError("--staged needs a candidate (--baseline-binary and --label)")
    report: dict = {}
    for stage in STAGES:
        if stage.clips is None:
            break  # the full TRAIN proxy is run explicitly with --clips
        clips = select_clips(by_name, stage.clips)
        report = measure_and_report(args, bench, fams, clips, targets, f"-{stage.name}",
                                    stage.qp_indices)
        verdicts = {}
        for c in report["comparisons"]:
            if c["anchor"].endswith(" cbr") and "@" in c["test"]:
                verdicts[c["test"]] = decide({m: c["mean"][m] for m in STAGE_METRICS},
                                             stage.sigma)
        report["stage"] = {"name": stage.name, "sigma": stage.sigma, "verdicts": verdicts}
        print_report(report)
        print(f"== stage {stage.name} (sigma {stage.sigma:.1f}%): {verdicts}", flush=True)
        if all(v == "better" for v in verdicts.values()) or "worse" in verdicts.values():
            break
    return report


def curve(points: list[dict]) -> tuple[list[float], dict[str, list[float]]]:
    points = sorted(points, key=lambda p: p["kbps"])
    return [p["kbps"] for p in points], {m: [p[m] for p in points] for m in METRICS}


def bd(anchor: list[dict], test: list[dict],
       min_points: int = MIN_BD_POINTS) -> dict[str, float | None]:
    a_rate, a_q = curve(anchor)
    t_rate, t_q = curve(test)
    result = {}
    for metric in METRICS:
        try:
            r = bd_rate(a_rate, a_q[metric], t_rate, t_q[metric], min_points=min_points)
            result[metric] = r.bd_rate_percent if r.reliable else None
        except ValueError:
            result[metric] = None
    return result


def build_report(fams: list[Family], clips: list[Clip], cbr: dict, cqp: dict,
                 min_points: int = MIN_BD_POINTS) -> dict:
    """BD-rate comparisons per clip plus CBR behaviour means per family."""
    def points(table: dict, family: str, clip: str) -> list[dict]:
        return [v for (f, c, _), v in table.items()
                if f == family and c == clip and reachable(v)]

    comparisons = []
    for codec in {f.codec for f in fams}:
        own = [f for f in fams if f.codec is codec and f.own and not f.variant]
        variants = [f for f in fams if f.codec is codec and f.variant]
        anchors = [f for f in fams if not f.own and codec in f.compares]
        pairs = [(f"{f.name} cqp", f.name, cqp, f.name, cbr) for f in own + anchors if f.cqp]
        pairs += [(f"{a.name} cbr", a.name, cbr, f.name, cbr) for f in own for a in anchors]
        for v in variants:
            pairs += [(f"{b.name} {mode}", b.name, table, v.name, cbr)
                      for b in own for mode, table in (("cbr", cbr), ("cqp", cqp))]
            pairs += [(f"{a.name} cbr", a.name, cbr, v.name, cbr) for a in anchors]
        for anchor_label, anchor_family, anchor_table, test_family, test_table in pairs:
            per_clip = {c.name: bd(points(anchor_table, anchor_family, c.name),
                                   points(test_table, test_family, c.name), min_points)
                        for c in clips}
            comparisons.append({"test": f"{test_family} cbr", "anchor": anchor_label,
                                "per_clip": per_clip, "mean": mean_bd(per_clip)})
    behaviour = {}
    for f in fams:
        rows = [v for (name, _, _), v in cbr.items() if name == f.name and reachable(v)]
        unreachable = sorted(f"{c} {t}" for (name, c, t), v in cbr.items()
                             if name == f.name and not reachable(v))
        behaviour[f.name] = {
            "unreachable": unreachable,
            "mean_overshoot": mean([overshoot(r) for r in rows]),
            "max_overshoot": max((overshoot(r) for r in rows), default=None),
            "mean_rate_error": mean([r["rate_error"] for r in rows]),
            "overruns": sum(r["overruns"] for r in rows),
            "max_fullness": max((r["max_fullness"] for r in rows), default=None),
            "max_window_ratio": max((r["max_window_ratio"] for r in rows), default=None),
            "flicker_psnr_y": mean([r["flicker_psnr_y"] for r in rows]),
            "flicker_vmaf_v1": mean([r["flicker_vmaf_v1"] for r in rows]),
            "cqp_flicker_psnr_y": mean([v["flicker_psnr_y"] for (n, _, _), v in cqp.items()
                                        if n == f.name]),
        }
    points_table = {f"{f} {c} {t}": {k: v for k, v in r.items() if not k.startswith("frame_")}
                    for table in (cbr, cqp) for (f, c, t), r in table.items()}
    return {"comparisons": comparisons, "behaviour": behaviour, "points": points_table}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def mean_bd(per_clip: dict) -> dict[str, float | None]:
    return {m: mean([r[m] for r in per_clip.values() if r[m] is not None]) for m in METRICS}


def print_report(report: dict) -> None:
    def fmt(v: float | None) -> str:
        return f"{v:+7.2f}%" if v is not None else "    n/a "
    print(f"{'test':28} {'anchor':28} " + " ".join(f"{m:>15}" for m in METRICS))
    for c in report["comparisons"]:
        print(f"{c['test']:28} {c['anchor']:28} " +
              " ".join(f"{fmt(c['mean'][m]):>15}" for m in METRICS))
        for clip, row in c["per_clip"].items():
            print(f"   {clip:54} " + " ".join(f"{fmt(row[m]):>15}" for m in METRICS))
    print(f"\n{'family':28} {'over':>7} {'maxover':>7} {'rate':>7} {'overrun':>7} {'maxfull':>7} "
          f"{'win/tgt':>7} {'flkPSNR':>7} {'flkVMAF':>7} {'cqpFlk':>7}")
    for name, b in report["behaviour"].items():
        def num(key: str, spec: str) -> str:
            return format(b[key], spec) if b[key] is not None else "    n/a"
        print(f"{name:28} {num('mean_overshoot', '7.3f')} {num('max_overshoot', '7.3f')} "
              f"{num('mean_rate_error', '+7.3f')} "
              f"{b['overruns']:7d} {num('max_fullness', '7.2f')} "
              f"{num('max_window_ratio', '7.2f')} {num('flicker_psnr_y', '7.3f')} "
              f"{num('flicker_vmaf_v1', '7.3f')} {num('cqp_flicker_psnr_y', '7.3f')}")
        if b["unreachable"]:
            print(f"   unreachable (>{UNREACHABLE_RATE_ERROR:.0%} over target): "
                  + ", ".join(b["unreachable"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nceval.cbr", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--binary", required=True, help="our nanocodec binary")
    parser.add_argument("--codecs", default="h264,hevc,av1")
    parser.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--targets", default=",".join(map(str, REFERENCE_TARGETS_KBPS)),
                        help="720p30 reference targets in kbps, scaled per clip")
    parser.add_argument("--intra-period", type=int, default=DEFAULT_INTRA_PERIOD)
    parser.add_argument("--temporal-layers", type=int, choices=(1, 2, 3), default=1,
                        help="dyadic hierarchical-P layers (SVT: --hierarchical-levels n-1)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--param", action="append", default=[], help="name=value for nanocodec")
    parser.add_argument("--preset", help="nanocodec --preset NAME")
    parser.add_argument("--label", help="suffix naming this nanocodec variant")
    parser.add_argument("--baseline-param", action="append", default=[],
                        help="name=value for the baseline build (candidate mode)")
    parser.add_argument("--staged", action="store_true",
                        help="candidate screen in nceval.screen's TRAIN stages (proxy-4, proxy-6)")
    parser.add_argument("--baseline-binary",
                        help="judge --binary (CBR only) against this build's cached curves")
    parser.add_argument("--anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-anchors", default="",
                        help="comma-separated: x264-veryfast, x265-superfast")
    parser.add_argument("--clipset", help="clip set JSON (default: the main set)")
    parser.add_argument("--gate", choices=[g.value for g in GatePolicy],
                        default=EVALUATION_GATE.value,
                        help="decoders on the measured points (nceval.gatepolicy; default off: "
                             "conformance is checked when a change lands)")
    parser.add_argument("--cqp", action=argparse.BooleanOptionalAction, default=True,
                        help="measure our encoder's fixed-QP curves (controller cost)")
    parser.add_argument("--anchor-cqp", action=argparse.BooleanOptionalAction, default=False,
                        help="also measure the anchors' fixed-QP curves (diagnostic)")
    parser.add_argument("--openh264-harness", default=str(
        Path.home() / "nanocodec-data/work/ratecontrol/bin/openh264-harness"),
        help="openh264-harness build with --bitrate support")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be 1..8")
    if args.baseline_binary and not args.label:
        parser.error("--baseline-binary needs --label for the candidate")
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
