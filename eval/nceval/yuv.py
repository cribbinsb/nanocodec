"""Raw 8-bit 4:2:0 (I420) YUV file helpers.

All decoded output in the harness is normalised to this one format, so
comparisons and metrics never depend on a decoder's output conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xxhash

#: The per-frame digest every conformance comparison uses (decodes against
#: each other and against the encoder's reconstruction, the verified-stream
#: cache, sweep recon files). Frames are compared, not authenticated, so a
#: non-cryptographic 128-bit hash is enough: xxh3-128 hashes a 720p frame in
#: ~0.09 ms where MD5 took ~1.6 ms, and a point hashes 2-4 streams of 150
#: frames (2026-11-11). Part of every stored digest's identity.
FRAME_HASH = "xxh3_128"


def frame_digest(frame: bytes | memoryview | np.ndarray) -> str:
    """The FRAME_HASH hex digest of one raw frame."""
    return xxhash.xxh3_128_hexdigest(frame)


@dataclass(frozen=True)
class YuvFormat:
    width: int
    height: int

    @property
    def luma_size(self) -> int:
        return self.width * self.height

    @property
    def chroma_size(self) -> int:
        return (self.width // 2) * (self.height // 2)

    @property
    def frame_size(self) -> int:
        return self.luma_size + 2 * self.chroma_size



@dataclass(frozen=True)
class SwitchedFormat:
    """Frames whose size changes mid-stream (an encoder resolution switch):
    `first` until the first switch, then each `(frame index, format)` from
    that frame on. Its `width` and `height` are the first format's."""
    first: YuvFormat
    switches: tuple[tuple[int, YuvFormat], ...]

    @property
    def width(self) -> int:
        return self.first.width

    @property
    def height(self) -> int:
        return self.first.height

    def at(self, index: int) -> YuvFormat:
        """The format of frame `index`."""
        fmt = self.first
        for start, switched in self.switches:
            if index >= start:
                fmt = switched
        return fmt


def frame_format_at(fmt: YuvFormat | SwitchedFormat, index: int) -> YuvFormat:
    """The format of frame `index` of a fixed or switched stream."""
    return fmt.at(index) if isinstance(fmt, SwitchedFormat) else fmt

def frame_count(path: Path, fmt: YuvFormat) -> int:
    size = path.stat().st_size
    if size % fmt.frame_size:
        raise ValueError(f"{path}: size {size} is not a whole number of "
                         f"{fmt.width}x{fmt.height} I420 frames")
    return size // fmt.frame_size


def open_frames(path: Path, fmt: YuvFormat) -> np.ndarray:
    """Memory-map a YUV file as an array of shape (frames, frame_size)."""
    n = frame_count(path, fmt)
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(n, fmt.frame_size))


def split_planes(frame: np.ndarray, fmt: YuvFormat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_end = fmt.luma_size
    u_end = y_end + fmt.chroma_size
    cw, ch = fmt.width // 2, fmt.height // 2
    return (frame[:y_end].reshape(fmt.height, fmt.width),
            frame[y_end:u_end].reshape(ch, cw),
            frame[u_end:].reshape(ch, cw))


def frame_hashes(path: Path, fmt: YuvFormat) -> list[str]:
    """One digest per frame: lets a mismatch report *which* frame differs."""
    frames = open_frames(path, fmt)
    return [frame_digest(frames[i].tobytes()) for i in range(len(frames))]


def first_mismatch(a: list[str], b: list[str]) -> int | None:
    """Index of the first differing frame, len() if only lengths differ, else None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None
