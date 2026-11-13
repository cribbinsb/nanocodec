"""One measured point of our encoder, streamed end to end: encode, gate, score.

Ungated (the default, nceval.gatepolicy: conformance is checked when a
change lands), Python never touches a pixel:

    encoder --sse-log ──────────────► per-frame squared errors ─► PSNR
    encoder --display-recon /dev/fd/N ──► VMAF (VMAFx stdin)

With a gate, or a metric that needs the frames in Python (or a binary
older than `--sse-log`), the reconstruction is read and fanned out here:

    encoder --display-recon /dev/fd/N ──► per-frame hash + SHA-256 + metric sinks
    standard decoder 1 ─────────► per-frame hash ┐
    standard decoder 2 ─────────► per-frame hash ├─ all equal the recon hashes?
    our decoder (optional) ─────► per-frame hash ┘

Metrics are computed from the encoder's reconstruction while it encodes,
so no decode is ever needed for metrics, and no video is written: the only
file a point leaves is its bitstream. With a gate, the decoders' hashes
prove the reconstruction equals every decode, and a point whose gate fails
returns no measurement.

Used by nceval.tune, nceval.screen and tools/dev/fastbd.py; nceval.sweep
measures anchor encoders (which have no reconstruction output) by attaching
the same metric sinks to the first standard decoder instead.
"""

from __future__ import annotations

import csv
import functools
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .codecs import Codec
from .config import Paths
from .conformance import GateResult, coded_format_from_layer_md5, gate_stream
from .gatepolicy import GateLevel
from .framestats import UNCODED_FRAME_TYPES
from .framepipe import (FrameSink, FrameHashSink, PipeOutput, ProcessFailed, ProcessSet,
                        ProcessSink, Sha256Sink)
from .metrics import (EncoderPsnrMetric, Measurement, MetricComponent, MetricSession,
                      PsnrMetric, StreamSpec, gpu_slot_need)
from .slots import cpu_slots, gpu_slots
from .display import switched_format
from .yuv import SwitchedFormat, YuvFormat

ENCODE_TIMEOUT_SECONDS = 3600

# Builds the encoder command line given the path its reconstruction goes to.
EncodeCommand = Callable[[str], list[str]]


@dataclass
class PointResult:
    gate: GateResult
    recon_sha256: str | None  # None when ungated: nothing read the reconstruction
    measurement: Measurement | None  # None when not requested or the gate failed


def encode_streamed(command: EncodeCommand, fmt: YuvFormat, sinks: list[FrameSink],
                    env: dict[str, str] | None = None) -> int:
    """Run the encoder with its reconstruction piped into `sinks`; return its frames.

    The caller holds the CPU slots (nceval.slots) for the encoder and sinks.
    """
    output = PipeOutput(fmt, sinks)
    with ProcessSet() as processes:
        try:
            process = processes.spawn(command(output.path), "encode",
                                      pass_fds=output.pass_fds, log_stdout=True, env=env)
        except BaseException:
            output.close_unstarted()
            raise
        output.start()
        try:
            process.wait(ENCODE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise ProcessFailed("encode timed out") from None
        finally:
            if process.poll() is None:
                process.kill()
            output.join(ENCODE_TIMEOUT_SECONDS)
        if process.returncode:
            raise ProcessFailed(f"encode failed ({process.returncode}): "
                                f"{processes.stderr_text(process)[-400:]}")
    return output.frames


def run_point(command: EncodeCommand, bitstream: Path, codec: Codec, spec: StreamSpec,
              paths: Paths, metrics: list[MetricComponent] | None = None,
              own_decoder: Path | None = None,
              env: dict[str, str] | None = None, standard: bool = True,
              serial_gate: bool = False, rate_log: Path | None = None,
              layer_md5: Path | None = None) -> PointResult:
    """Encode (scoring the reconstruction with `metrics`), then gate the stream.

    `spec.reference` is the metric reference clip, read only for its first
    `spec.frames` frames. `command` must write `bitstream`; `env` is the
    encoder's environment (default: inherited). `standard` False leaves the
    check to `own_decoder`, or with none to nobody (nceval.gatepolicy
    gate_arguments).
    `rate_log` is the encoder's `--rate-log`, if it writes one: its coded
    sizes tell the gate which frames were coded below the stream size
    (adaptive resolution). The reconstruction, and so every metric, is
    always at the stream size, as a player displays it (nceval.display).
    """
    gated = standard or own_decoder is not None
    if not gated:
        measurement = (score_ungated(command, spec, metrics, env) if metrics
                       else encode_unscored(command, env))
        return PointResult(GateResult(True, level=GateLevel.UNCHECKED.value), None, measurement)
    recon_hashes, recon_sha = FrameHashSink(), Sha256Sink()
    measurement = None
    if metrics:
        # Slots first: an idle metric pipeline waiting for slots would hold
        # memory. GPU slots before CPU slots (nceval.slots).
        with gpu_slots(gpu_slot_need(metrics)), \
                cpu_slots(1 + sum(m.cpu_threads for m in metrics)), \
                MetricSession(spec, metrics) as session:
            try:
                frames = encode_streamed(command, spec.fmt,
                                         [recon_hashes, recon_sha, *session.sinks], env=env)
            except (ProcessFailed, BrokenPipeError) as error:
                raise_metric_failure(session, error)
                raise
            measurement = session.result(frames)
    else:
        with cpu_slots(1):
            frames = encode_streamed(command, spec.fmt, [recon_hashes, recon_sha], env=env)
    if frames != spec.frames:
        raise ProcessFailed(f"encoder reconstructed {frames} frames, expected {spec.frames}")
    coded = coded_format(rate_log, spec.fmt) or coded_format_from_layer_md5(layer_md5, spec.fmt)
    gate = gate_stream(bitstream, codec, spec.fmt, spec.frames, paths,
                       expected_hashes=recon_hashes.digests, own_decoder=own_decoder,
                       standard=standard, serial=serial_gate,
                       coded_sizes=coded)
    return PointResult(gate, recon_sha.hexdigest(), measurement if gate.passed else None)


def score_ungated(command: EncodeCommand, spec: StreamSpec, metrics: list[MetricComponent],
                  env: dict[str, str] | None) -> Measurement:
    """Score an ungated point, reading its reconstruction in Python only if
    a metric needs the frames there.

    The encoder logs PSNR's squared errors itself (`--sse-log`; not in
    builds before it, which then fall back to `PsnrMetric`), and when what
    is left takes the frames through one process's stdin (VMAFx), the
    encoder writes its reconstruction straight into it.
    """
    with gpu_slots(gpu_slot_need(metrics)), \
            cpu_slots(1 + sum(m.cpu_threads for m in metrics)), \
            tempfile.TemporaryDirectory(prefix="nc-point-") as scratch:
        extra: list[str] = []
        if writes_sse_log(command(os.devnull)[0]):
            psnr = EncoderPsnrMetric(spec, Path(scratch) / "sse.csv")
            metrics = [psnr if type(m) is PsnrMetric else m for m in metrics]
            extra = psnr.encoder_arguments()

        def encode(recon: str) -> list[str]:
            return command(recon) + extra

        with MetricSession(spec, metrics) as session:
            sinks = session.sinks
            try:
                if len(sinks) > 1 or (sinks and not isinstance(sinks[0], ProcessSink)):
                    return session.result(encode_streamed(encode, spec.fmt, sinks, env=env))
                if not sinks:
                    run_encoder(encode(os.devnull), (), env)
                else:
                    fd = sinks[0].fileno()
                    run_encoder(encode(f"/dev/fd/{fd}"), (fd,), env, on_start=sinks[0].finish)
            except (ProcessFailed, BrokenPipeError) as error:
                raise_metric_failure(session, error)
                raise
            return session.result(spec.frames)


def raise_metric_failure(session: MetricSession, error: BaseException) -> None:
    """When the encoder (or the frame fan-out) failed because a metric died
    under it, raise that as MetricFailed: an encoder failure here is otherwise
    taken for the encoder refusing the point (nceval.realuse), which would
    turn a failed metric into a silently missing point."""
    failure = session.failure_behind(error)
    if failure is not None:
        raise failure from error


def encode_unscored(command: EncodeCommand, env: dict[str, str] | None) -> None:
    """An ungated point with no metrics: the reconstruction goes nowhere."""
    with cpu_slots(1):
        run_encoder(command(os.devnull), (), env)


def run_encoder(cmd: list[str], pass_fds: tuple[int, ...], env: dict[str, str] | None,
                on_start: Callable[[], None] = lambda: None) -> None:
    """Run the encoder to completion. `on_start` runs once it is spawned:
    dropping our copy of the pipe it writes, its reader sees EOF when it exits."""
    with ProcessSet() as processes:
        try:
            process = processes.spawn(cmd, "encode", pass_fds=pass_fds, log_stdout=True,
                                      env=env)
        finally:
            on_start()
        try:
            process.wait(ENCODE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise ProcessFailed("encode timed out") from None
        if process.returncode:
            raise ProcessFailed(f"encode failed ({process.returncode}): "
                                f"{processes.stderr_text(process)[-400:]}")


@functools.lru_cache(maxsize=None)
def _usage_mentions_sse_log(binary: str, size: int, mtime_ns: int) -> bool:
    usage = subprocess.run([binary], stdin=subprocess.DEVNULL, capture_output=True, text=True)
    return "--sse-log" in usage.stdout + usage.stderr


def writes_sse_log(binary: str) -> bool:
    """Whether our encoder `binary` has `--sse-log` (older builds silently
    ignore unknown options, so ask its usage text)."""
    stat = Path(binary).stat()
    return _usage_mentions_sse_log(binary, stat.st_size, stat.st_mtime_ns)


def coded_format(rate_log: Path | None, fmt: YuvFormat) -> SwitchedFormat | None:
    """The coded sizes of a stream that codes frames smaller than `fmt`,
    from its `--rate-log` (`width`, `height` columns of the frames in the
    stream: dropped and held frames leave nothing); `None` when every frame
    has the stream size or there is no log."""
    if rate_log is None or not rate_log.exists():
        return None
    with rate_log.open(newline="") as log:
        rows = [row for row in csv.DictReader(log) if row["type"] not in UNCODED_FRAME_TYPES]
    if not rows or "width" not in rows[0]:
        return None
    sizes = [(int(row["width"]), int(row["height"])) for row in rows]
    if all(size == (fmt.width, fmt.height) for size in sizes):
        return None
    switched = switched_format(sizes)
    return switched if isinstance(switched, SwitchedFormat) else SwitchedFormat(switched, ())
