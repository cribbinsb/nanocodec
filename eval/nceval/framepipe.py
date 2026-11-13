"""Frame streams: raw I420 video moved through pipes, never through files.

A measured point used to write ~1.6-2.9 GB of temporary YUV per 720p
150-frame point (recon, two decodes, a trimmed reference, two 1080p 10-bit
VMAF display copies), which saturated the shared NVMe and left decoders and
metrics in iowait (docs/handoff/eval-throughput.md). Everything here instead
moves frames through pipes and FIFOs one frame at a time:

  - `FrameSink`s consume frames: per-frame and SHA-256 hashers, the PSNR accumulator,
    and `ProcessSink`, which feeds another program's stdin (an upscaler or
    the xpsnr filter).
  - `pump_frames` reads whole frames from a pipe and hands each to every sink.
    A failing sink is detached and its error re-raised at the end, but the
    pipe is always drained, so the producer can never block forever.
  - `PipeOutput` gives a tool that insists on an output *path* (encoders'
    --recon, JM, HM) an anonymous pipe named /dev/fd/N instead.

Memory per stream is one frame plus OS pipe buffers.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, Protocol

import cv2
import numpy as np

from .slots import low_priority
from .yuv import YuvFormat, frame_digest, frame_format_at

PSNR_CAP_DB = 100.0
_PEAK_SQUARED = 255.0 ** 2


class FrameSink(Protocol):
    """Something that consumes a stream of frames, one at a time."""

    def consume(self, frame: bytes) -> None: ...

    def finish(self) -> None: ...


class FrameHashSink:
    """One digest per frame (the conformance comparison, as yuv.frame_hashes)."""

    def __init__(self) -> None:
        self.digests: list[str] = []

    def consume(self, frame: bytes) -> None:
        self.digests.append(frame_digest(frame))

    def finish(self) -> None:
        pass


class Sha256Sink:
    """SHA-256 of the whole concatenated stream (equals sha256 of the file)."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def consume(self, frame: bytes) -> None:
        self._hash.update(frame)

    def finish(self) -> None:
        pass

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


class ReferenceFrames:
    """Sequential reader of the first frames of a raw clip, without copying it."""

    def __init__(self, path: Path, fmt: YuvFormat):
        self._file = path.open("rb")
        self._fmt = fmt

    def next(self) -> bytes:
        frame = self._file.read(self._fmt.frame_size)
        if len(frame) != self._fmt.frame_size:
            raise ValueError("reference has fewer frames than the distorted stream")
        return frame

    def close(self) -> None:
        self._file.close()


class PsnrSink:
    """Per-frame, per-plane MSE against the reference, accumulated while streaming.

    Each plane's sum of squared differences is an exact integer: |a - b| in
    uint8 (OpenCV's SIMD absdiff), squared in uint16 (<= 255^2 fits), summed
    in uint64; then divided by the plane size. About half the CPU of an
    int16 difference and int64 dot product, with identical results.
    """

    def __init__(self, reference: Path, fmt: YuvFormat):
        self._reference = ReferenceFrames(reference, fmt)
        self._planes = ((0, fmt.luma_size), (fmt.luma_size, fmt.luma_size + fmt.chroma_size),
                        (fmt.luma_size + fmt.chroma_size, fmt.frame_size))
        self.mse: list[tuple[float, float, float]] = []

    def consume(self, frame: bytes) -> None:
        difference = cv2.absdiff(np.frombuffer(self._reference.next(), dtype=np.uint8),
                                 np.frombuffer(frame, dtype=np.uint8)).ravel()
        squares = difference.astype(np.uint16)
        squares *= squares
        self.mse.append(tuple(float(squares[lo:hi].sum(dtype=np.uint64)) / (hi - lo)
                              for lo, hi in self._planes))

    def finish(self) -> None:
        self._reference.close()

    def per_plane_psnr(self) -> np.ndarray:
        """(frames, 3) PSNR in dB, MSE 0 capped at PSNR_CAP_DB."""
        return psnr_from_mse(self.mse)


def psnr_from_mse(mse: list[tuple[float, float, float]]) -> np.ndarray:
    """(frames, 3) PSNR in dB of per-frame, per-plane MSE; MSE 0 is capped at PSNR_CAP_DB."""
    mse = np.array(mse, dtype=np.float64).reshape(-1, 3)
    with np.errstate(divide="ignore"):
        return np.minimum(PSNR_CAP_DB, 10.0 * np.log10(_PEAK_SQUARED / mse))


class ProcessSink:
    """Writes every frame to a running process's stdin; closing it signals EOF."""

    def __init__(self, process: subprocess.Popen):
        if process.stdin is None:
            raise ValueError("ProcessSink needs a process started with stdin=PIPE")
        self._stdin = process.stdin

    def consume(self, frame: bytes) -> None:
        self._stdin.write(frame)

    def fileno(self) -> int:
        """The process's stdin, for a producer that writes to it directly
        (nceval.point); `finish` then just drops our copy."""
        return self._stdin.fileno()

    def finish(self) -> None:
        self._stdin.close()


class FrameStreamError(RuntimeError):
    """The stream was not a whole number of frames, or a sink failed."""


def shown_frames(decoded: list[str], shown: list[str]) -> list[bool] | None:
    """Which of the pictures shown at each input frame (`shown`: frame hashes of our
    encoder's --display-recon) were decoded frames; None if `decoded` (hashes
    of a decoder's frames) is not `shown` with held frames left out.

    A frame our encoder held (or dropped) is not coded, and the decoder
    shows the previous picture again: `shown` repeats it there. Each decoded
    frame is matched to the next shown picture; a shown picture it does
    not match must repeat the one before. Where a coded frame happens to
    reconstruct exactly as the previous one, which of the equal pictures was
    coded is ambiguous, but the pictures shown are the same either way.
    """
    coded, at = [], 0
    for digest in decoded:
        while at < len(shown) and shown[at] != digest:
            if not coded or shown[at] != shown[at - 1]:
                return None
            coded.append(False)
            at += 1
        if at == len(shown):
            return None
        coded.append(True)
        at += 1
    rest = shown[at:]
    # Trailing held frames repeat the last decoded picture.
    if rest and (not coded or any(digest != shown[at - 1] for digest in rest)):
        return None
    return coded + [False] * len(rest)


class HeldFrameSink:
    """Feeds `sinks` the picture shown at every input frame, from a decoder's
    frames: each frame our encoder held (`shown` repeats the previous
    picture's hash there, see `shown_frames`) is filled in as a repeat of the
    last decoded frame, so metrics see one picture per input frame."""

    def __init__(self, sinks: list[FrameSink], shown: list[str]):
        self._sinks, self._shown = sinks, shown
        self._at = 0
        self._last: bytes | None = None

    def consume(self, frame: bytes) -> None:
        digest = frame_digest(frame)
        while self._at < len(self._shown) and self._shown[self._at] != digest:
            self._repeat()
        if self._at == len(self._shown):
            raise FrameStreamError("decoded frames are not the encoder's shown pictures")
        self._forward(frame)

    def finish(self) -> None:
        while self._at < len(self._shown):
            self._repeat()
        for sink in self._sinks:
            sink.finish()

    def _repeat(self) -> None:
        held = self._last is not None and self._shown[self._at] == self._shown[self._at - 1]
        if not held:
            raise FrameStreamError("decoded frames are not the encoder's shown pictures")
        self._forward(self._last)

    def _forward(self, frame: bytes) -> None:
        for sink in self._sinks:
            sink.consume(frame)
        self._last = frame
        self._at += 1


def pump_frames(source: BinaryIO, fmt: YuvFormat, sinks: list[FrameSink]) -> int:
    """Hand every whole frame of `source` to each sink; return the frame count.

    A sink that raises is detached (its error is re-raised after the stream
    ends) so the remaining sinks and, above all, the producer keep going: a
    producer blocked on a full pipe would otherwise hang the point.
    """
    live = list(sinks)
    errors: list[BaseException] = []
    frames = 0
    while True:
        # A switched stream (SwitchedFormat) changes frame size mid-stream.
        size = frame_format_at(fmt, frames).frame_size
        buffer = bytearray(size)
        filled = _read_full(source, memoryview(buffer))
        if filled == 0:
            break
        if filled != size:
            errors.append(FrameStreamError(
                f"stream ends with a partial frame ({filled} of {size} bytes)"))
            break
        frame = bytes(buffer)
        frames += 1
        for sink in list(live):
            try:
                sink.consume(frame)
            except BaseException as err:  # detach; keep draining for the producer
                errors.append(err)
                live.remove(sink)
                _finish_quietly(sink)
    for sink in live:
        try:
            sink.finish()
        except BaseException as err:
            errors.append(err)
    if errors:
        raise errors[0]
    return frames


def _read_full(source: BinaryIO, view: memoryview) -> int:
    filled = 0
    while filled < len(view):
        count = source.readinto(view[filled:])
        if not count:
            break
        filled += count
    return filled


def _finish_quietly(sink: FrameSink) -> None:
    try:
        sink.finish()
    except BaseException:  # the first error is already recorded
        pass


class PipeOutput:
    """An anonymous pipe a child process writes frames to, named by a path.

    Tools that insist on an output *path* (an encoder's --recon, JM, HM,
    aomdec) are given `path` = /dev/fd/N; on Linux opening it reopens the
    inherited pipe, so nothing touches the filesystem. Usage:

        output = PipeOutput(fmt, sinks)
        process = subprocess.Popen([..., output.path], pass_fds=output.pass_fds)
        output.start()          # closes our write end, starts the reader
        process.wait(); output.join()

    Because the child inherits the write end, EOF arrives when the child
    exits, even if it never opened the path (no FIFO open() deadlocks).
    """

    def __init__(self, fmt: YuvFormat, sinks: list[FrameSink]):
        self._read_fd, self._write_fd = os.pipe()
        self._fmt, self._sinks = fmt, sinks
        self.frames = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._read, name="frame-pipe", daemon=True)

    @property
    def path(self) -> str:
        return f"/dev/fd/{self._write_fd}"

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self._write_fd,)

    def start(self) -> "PipeOutput":
        """Call once the writer process exists: drop our write end, start reading."""
        os.close(self._write_fd)
        self._thread.start()
        return self

    def close_unstarted(self) -> None:
        """Release both ends when the writer could not be started."""
        os.close(self._write_fd)
        os.close(self._read_fd)

    def _read(self) -> None:
        try:
            with open(self._read_fd, "rb", buffering=0) as source:
                self.frames = pump_frames(source, self._fmt, self._sinks)
        except BaseException as err:
            self._error = err

    def join(self, timeout: float | None = None) -> None:
        """Wait for the reader (after the writer exited); re-raise its error."""
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("frame pipe reader did not finish")
        if self._error is not None:
            raise self._error


class ProcessFailed(RuntimeError):
    """A helper process of a frame stream exited with an error."""


class ProcessSet:
    """The helper processes of one streamed measurement, killed together on exit.

    stderr goes to an anonymous temporary file (a few KB of text), never to a
    pipe nobody drains: a full stderr pipe would stall the process and, with
    it, the whole frame stream.
    """

    def __init__(self) -> None:
        self._processes: list[tuple[subprocess.Popen, BinaryIO, str]] = []

    def __enter__(self) -> "ProcessSet":
        return self

    def __exit__(self, *exc) -> None:
        self.kill_all()

    def spawn(self, cmd: list[str], label: str, *, stdin=subprocess.DEVNULL,
              stdout=subprocess.DEVNULL, pass_fds: tuple[int, ...] = (),
              log_stdout: bool = False, cwd: Path | None = None,
              env: dict[str, str] | None = None) -> subprocess.Popen:
        """Start `cmd`; `log_stdout` sends its stdout to the stderr log too."""
        stderr = tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(low_priority(cmd), stdin=stdin, stdout=stderr if log_stdout else stdout,
                                       stderr=stderr, pass_fds=pass_fds, cwd=cwd, env=env)
        except BaseException:
            stderr.close()
            raise
        self._processes.append((process, stderr, label))
        return process

    def wait_all(self, timeout: float) -> None:
        """Wait for every process; raise ProcessFailed naming the first failure."""
        for process, _, label in self._processes:
            try:
                process.wait(timeout)
            except subprocess.TimeoutExpired:
                raise ProcessFailed(f"{label} timed out") from None
        for process, stderr, label in self._processes:
            if process.returncode != 0:
                raise ProcessFailed(f"{label} failed ({process.returncode}): "
                                    f"{self.stderr_text(process)[-400:]}")

    def stderr_text(self, process: subprocess.Popen) -> str:
        for candidate, stderr, _ in self._processes:
            if candidate is process:
                stderr.seek(0)
                return stderr.read().decode(errors="replace")
        raise KeyError("not a process of this set")

    def kill_all(self) -> None:
        for process, stderr, _ in self._processes:
            if process.poll() is None:
                process.kill()
                process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
            stderr.close()
        self._processes.clear()


class FileSink:
    """Writes frames to an open file (only for callers that still need a YUV file)."""

    def __init__(self, output: BinaryIO):
        self._output = output

    def consume(self, frame: bytes) -> None:
        self._output.write(frame)

    def finish(self) -> None:
        self._output.flush()
