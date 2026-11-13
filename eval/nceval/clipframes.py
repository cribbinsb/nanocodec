"""A clip's measured frames, served to every reader of a point through its own FIFO.

A point reads its clip several times at once: the encoder's input, and the
metric references (PSNR, VMAF v1 and MS-SSIM each read the reference prefix;
nceval.metrics). A raw clip measured from frame 0 is read straight from its
file, as before. Anything else -- a pool clip stored as a high-bitrate encode
(nceval.pool), or a raw clip measured from a rotated start frame -- is served
through FIFOs, one per `reader()` call, so no YUV file is ever written:

  - encoded clips: one ffmpeg decode, from the nearest key frame before the
    start (input seeking; the pool stores 1 s key intervals), fanned out to
    every reader (`SharedDecode`);
  - raw clips from a rotated start: a copy of the file's bytes from the
    start frame's offset, per reader (no decode to share).

The shared decode keeps memory bounded: it holds at most WINDOW_FRAMES
decoded frames, the span between the slowest and the fastest reader, and
stops decoding (backpressure) while that span is full. A decode per reader
cost ~3 decodes per point (17 ffmpeg processes among the top CPU users of a
loaded 8-worker screen, 2026-11-12); an earlier fan-out that held the whole
150-frame prefix (~210 MB at 720p) per point took an 8-worker screen past
its 8 GB memory cap (2026-11-11). Measured on 12 pool points (4 clips incl.
B-grade downscaled, rotated starts; loaded machine, 2026-11-12): our
encoder's points went from 3 decodes to 1, anchors' from 3 to 2 (a 720p
150-frame decode is ~2.4 CPU-s there); every per-frame metric unchanged;
mean 14.6 -> 13.3 CPU-s per point.

Readers are not all alike, and the fan-out handles each case without
holding more frames:

  - readers open their FIFOs at different times (a metric starts before the
    encoder): a reader joins a decode while its first frame is still held;
    one made after that starts the next decode, shared by later readers
    (anchors encode first, then score: two decodes, not three);
  - a path built but never opened (a command line built only to inspect
    it) must not stall the others: when a reader that has opened its FIFO
    is starved by a full window, readers at the window's start that never
    opened are dropped from the decode;
  - a reader that stops reading while others wait for frames (a deadlock
    no bound can rule out, e.g. a reader that needs more than the window
    ahead) is dropped after STALL_SECONDS.

A dropped reader that reads later continues from its own decode, skipping
the frames it already had: decoding is deterministic, so every reader gets
the same bytes either way.
"""

from __future__ import annotations

import collections
import errno
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .clips import Clip
from .config import find_tool
from .slots import low_priority

#: Back-off while a FIFO's reader has not opened it yet.
OPEN_POLL_SECONDS = 0.01
#: How long close() waits for a writer thread (a reader that died unblocks it).
JOIN_SECONDS = 5.0
#: Input-seek this far before the start frame's timestamp (in frames), so
#: rounding never drops the start frame; ffmpeg's accurate seek discards
#: every decoded frame before the seek point.
SEEK_BACKOFF_FRAMES = 0.5
#: Decoded frames a shared decode holds at most (~11 MB at 720p, ~25 MB at
#: 1080p, whatever the reader count). A point's readers move in step with
#: the encoder (each metric reads reference frame n once the encoder has
#: reconstructed frame n), a frame or two apart, so this never limits a
#: healthy point.
WINDOW_FRAMES = 8
#: A reader that holds a full window while another reader starves, and
#: reads nothing for this long, is dropped to its own decode. Only a
#: deadlock or a reader stalled for seconds gets here; dropping one costs
#: one decode, never a wrong frame.
STALL_SECONDS = 2.0
#: How often a decode blocked on a full window re-checks for a stall.
STALL_POLL_SECONDS = 0.05


class ClipFrames:
    """Frames [clip.start_frame, clip.start_frame + frames) of `clip`.

    Use as a context manager; `reader()` returns a path that yields exactly
    those frames once (the clip's own file when it can be read directly).
    """

    def __init__(self, clip: Clip, frames: int, scratch: Path):
        self.clip, self.frames = clip, frames
        self.frame_size = clip.width * clip.height * 3 // 2
        self._scratch = scratch
        self._directory: Path | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []
        self._processes: list[subprocess.Popen] = []
        self._errors: list[BaseException] = []
        self._decodes: list[SharedDecode] = []

    def __enter__(self) -> ClipFrames:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # A consumer that failed (an encoder refusing the clip) breaks the
        # pipe its frames were served on: its own error is the one to report,
        # not the broken pipe that follows from it.
        self.close(raise_errors=exc_type is None)

    @property
    def decodes_started(self) -> int:
        """ffmpeg decodes run so far (shared and per-reader)."""
        with self._lock:
            return len(self._processes)

    def reader(self) -> Path:
        """A path that yields the frames once (a fresh FIFO per call)."""
        if self.clip.direct:
            return self.clip.yuv
        with self._lock:
            if self._directory is None:
                self._scratch.mkdir(parents=True, exist_ok=True)
                self._directory = Path(tempfile.mkdtemp(prefix="nc-frames-", dir=self._scratch))
            fifo = self._directory / f"reader{len(self._threads)}.yuv"
            os.mkfifo(fifo)
            membership = self._join_decode() if self.clip.encoded else None
            thread = threading.Thread(target=self._serve, args=(fifo, membership), daemon=True)
            self._threads.append(thread)
        thread.start()
        return fifo

    def close(self, raise_errors: bool = True) -> None:
        self._closed.set()
        with self._lock:
            processes = list(self._processes)
            decodes = list(self._decodes)
        for decode in decodes:
            decode.stop()
        for process in processes:
            if process.poll() is None:
                process.kill()
        for thread in self._threads:
            thread.join(JOIN_SECONDS)
        for process in processes:
            process.wait()
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
        if self._errors and raise_errors:
            raise RuntimeError(f"{self.clip.name}: serving its frames failed") from self._errors[0]

    def _join_decode(self) -> tuple[SharedDecode, int]:
        """Join the current shared decode, or start the next one (lock held)."""
        if self._decodes:
            membership = self._decodes[-1].join()
            if membership is not None:
                return self._decodes[-1], membership
        decode = SharedDecode(self, self._closed)
        self._decodes.append(decode)
        return decode, decode.join()

    def _serve(self, fifo: Path, membership: tuple[SharedDecode, int] | None) -> None:
        descriptor = self._open_for_writing(fifo)
        if descriptor is None:
            if membership is not None:
                membership[0].leave(membership[1])
            return  # never opened (e.g. a command line built only to inspect it)
        os.set_blocking(descriptor, True)
        try:
            if membership is None:
                self._copy_into(descriptor)
            else:
                self._fan_out_into(descriptor, *membership)
        except BrokenPipeError:
            pass  # the reader stopped early; it reports its own error
        except BaseException as error:  # reported by close(); the reader sees EOF
            if not self._closed.is_set():
                self._errors.append(error)
        finally:
            os.close(descriptor)

    def _open_for_writing(self, fifo: Path) -> int | None:
        """Wait (without blocking close) until the reader opens the FIFO."""
        while not self._closed.is_set():
            try:
                return os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as error:
                if error.errno != errno.ENXIO:
                    raise
            time.sleep(OPEN_POLL_SECONDS)
        return None

    def _copy_into(self, descriptor: int) -> None:
        with self.clip.yuv.open("rb") as source:
            source.seek(self.clip.start_frame * self.frame_size)
            for index in range(self.frames):
                frame = source.read(self.frame_size)
                if len(frame) != self.frame_size:
                    raise RuntimeError(f"{self.clip.name}: frame {self.clip.start_frame + index} "
                                       "is past the end of the clip")
                _write_all(descriptor, frame)

    def _fan_out_into(self, descriptor: int, decode: SharedDecode, member: int) -> None:
        """Copy the shared decode's frames; if dropped from it, finish from our own."""
        try:
            decode.opened(member)
            while (frame := decode.next(member)) is not None:
                _write_all(descriptor, frame)
        except Dropped as dropped:
            self._decode_into(descriptor, skip=dropped.position)
        finally:
            decode.leave(member)

    def start_decode(self, stdout) -> subprocess.Popen | None:
        """Start an ffmpeg decode of the frames (None once closed)."""
        with self._lock:
            if self._closed.is_set():
                return None
            process = subprocess.Popen(low_priority(decode_command(self.clip, self.frames)),
                                       stdin=subprocess.DEVNULL, stdout=stdout,
                                       stderr=subprocess.PIPE)
            self._processes.append(process)
            return process

    def check_decode(self, process: subprocess.Popen) -> None:
        """Wait for a decode that was read to its end; raise if it failed."""
        stderr = process.stderr.read()
        process.stderr.close()
        if process.wait() != 0 and not self._closed.is_set():
            raise RuntimeError(f"ffmpeg failed decoding {self.clip.yuv} ({process.returncode}): "
                               f"{stderr.decode(errors='replace').strip()[-300:]}")

    def report(self, error: BaseException) -> None:
        """Record an error for close() to raise."""
        if not self._closed.is_set():
            self._errors.append(error)

    def _decode_into(self, descriptor: int, skip: int = 0) -> None:
        """A decode of our own, its first `skip` frames discarded."""
        if not skip:
            process = self.start_decode(descriptor)
            if process is not None:
                self.check_decode(process)
            return
        process = self.start_decode(subprocess.PIPE)
        if process is None:
            return
        try:
            for index in range(self.frames):
                frame = _read_frame(process.stdout, self.frame_size)
                if frame is None:
                    break
                if index >= skip:
                    _write_all(descriptor, frame)
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdout.close()
        self.check_decode(process)


class Dropped(Exception):
    """A reader was dropped from a shared decode after `position` frames."""

    def __init__(self, position: int):
        super().__init__(position)
        self.position = position


class SharedDecode:
    """One ffmpeg decode of a clip's frames, fanned out to its members.

    Frames [base, produced) are held; `base` is the slowest member's
    position, and the decode waits while WINDOW_FRAMES are held. A member
    can join only while frame 0 is still held. Started when its first
    member opens its FIFO; stopped once every member has left.
    """

    def __init__(self, frames: ClipFrames, closed: threading.Event):
        self._source, self._closed = frames, closed
        self._condition = threading.Condition()
        self._held: collections.deque[bytes] = collections.deque()
        self._base = self._produced = 0
        self._ended = self._started = self._stopped = False
        #: Member -> its next frame; dropped and departed members are removed.
        self._positions: dict[int, int] = {}
        #: Dropped member -> the frames it had read.
        self._dropped: dict[int, int] = {}
        self._opened: set[int] = set()
        self._members = 0
        self._progress = time.monotonic()

    def join(self) -> int | None:
        """A new member at frame 0, or None if frame 0 is gone."""
        with self._condition:
            if self._base or self._stopped:
                return None
            member = self._members
            self._members += 1
            self._positions[member] = 0
            return member

    def opened(self, member: int) -> None:
        """`member` opened its FIFO: start decoding if nobody had."""
        with self._condition:
            self._opened.add(member)
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._decode, name="clip-decode", daemon=True).start()

    def next(self, member: int) -> bytes | None:
        """`member`'s next frame (None at the end); raises Dropped."""
        with self._condition:
            while True:
                if member in self._dropped:
                    raise Dropped(self._dropped[member])
                position = self._positions[member]
                if position < self._produced:
                    self._positions[member] = position + 1
                    frame = self._held[position - self._base]
                    self._release()
                    return frame
                if self._ended or self._closed.is_set():
                    return None
                self._condition.wait()

    def leave(self, member: int) -> None:
        """`member` is done (or never opened): it holds no frames any more."""
        with self._condition:
            self._positions.pop(member, None)
            self._release()
            if not self._positions:
                self._stopped = True
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def _release(self) -> None:
        """Forget frames every member has read (condition held)."""
        base = min(self._positions.values(), default=self._produced)
        if base > self._base:
            for _ in range(base - self._base):
                self._held.popleft()
            self._base = base
            self._progress = time.monotonic()
            self._condition.notify_all()

    def _decode(self) -> None:
        process = self._source.start_decode(subprocess.PIPE)
        try:
            if process is not None:
                self._pump(process)
        except BaseException as error:
            self._source.report(error)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
            with self._condition:
                self._ended = True
                self._condition.notify_all()

    def _pump(self, process: subprocess.Popen) -> None:
        """Read the decode's frames into the window until done or stopped."""
        wanted = self._source.frames
        while self._produced < wanted and self._wait_for_room():
            frame = _read_frame(process.stdout, self._source.frame_size)
            if frame is None:
                break
            with self._condition:
                self._held.append(frame)
                self._produced += 1
                self._release()
                self._condition.notify_all()
        if self._produced == wanted:
            process.stdout.close()
            self._source.check_decode(process)
        elif not (self._stopped or self._closed.is_set()):
            self._source.check_decode(process)
            raise RuntimeError(f"ffmpeg decoded {self._produced} of {wanted} frames "
                               f"of {self._source.clip.yuv}")

    def _wait_for_room(self) -> bool:
        """Wait until a frame may be added; False once stopped."""
        with self._condition:
            while len(self._held) >= WINDOW_FRAMES and not self._stopped:
                self._drop_stalled()
                if len(self._held) >= WINDOW_FRAMES:
                    self._condition.wait(STALL_POLL_SECONDS)
            return not (self._stopped or self._closed.is_set())

    def _drop_stalled(self) -> None:
        """Drop the members at the window's start that keep a starved member
        waiting: any that never opened their FIFO at once, the others after
        STALL_SECONDS without progress (condition held)."""
        if not any(self._positions.get(m) == self._produced for m in self._opened):
            return
        stalled = time.monotonic() - self._progress > STALL_SECONDS
        for member, position in list(self._positions.items()):
            if position == self._base and (stalled or member not in self._opened):
                self._dropped[member] = position
                del self._positions[member]
        self._release()
        self._condition.notify_all()


def _read_frame(stream, size: int) -> bytes | None:
    """One whole frame from `stream`; None at its end (or a partial frame)."""
    frame = stream.read(size)
    return frame if len(frame) == size else None


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view):]


def decode_command(clip: Clip, frames: int) -> list[str]:
    """ffmpeg decoding `frames` frames of an encoded clip from its start frame."""
    seek = []
    if clip.start_frame:
        seek = ["-ss", f"{(clip.start_frame - SEEK_BACKOFF_FRAMES) / clip.fps:.6f}"]
    return [str(find_tool("ffmpeg")), "-nostdin", "-v", "error", "-threads", "1", *seek,
            "-i", str(clip.yuv), "-frames:v", str(frames), "-f", "rawvideo",
            "-pix_fmt", "yuv420p", "pipe:1"]
