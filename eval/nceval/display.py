"""The player model: decoded frames coded smaller than the stream, scaled to it.

An adaptive-resolution stream (nanocodec `resolution.adaptive`, AV1) codes
some frames at half size per dimension and signals the stream size as every
frame's render size. Decoders output each frame at its coded size; a player
scales it to the render size on the GPU, sampling bilinearly with pixel
centres aligned. Metrics must see what the viewer sees, so every frame is
scored at the stream size after exactly that scaling.

`scale_frame` is the bit-exact twin of `nc_dsp::display` (the Rust model the
encoder's `--recon` output uses): integer positions in 1/65536 sample, 8-bit
weights, a horizontal pass kept unrounded, one rounding at the end, each
4:2:0 plane on its own grid. The conformance gate then compares the
encoder's (scaled) reconstruction with every decoder's output scaled here,
and the decoders' coded-size outputs with each other
(`conformance.gate_stream`, `coded=`). `tests/test_display.py` pins both
implementations to one fixture.
"""

from __future__ import annotations

import numpy as np

from .yuv import SwitchedFormat, YuvFormat, frame_format_at, split_planes

_POSITION_BITS = 16
_WEIGHT_BITS = 8
_WEIGHT_ONE = 1 << _WEIGHT_BITS


def _taps(size_in: int, size_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First and second input index and the second's weight per output sample."""
    out = np.arange(size_out, dtype=np.int64)
    centre = ((2 * out + 1) * size_in) << _POSITION_BITS
    position = np.maximum(centre // (2 * size_out) - (1 << (_POSITION_BITS - 1)), 0)
    first = np.minimum(position >> _POSITION_BITS, size_in - 1)
    weight = (position >> (_POSITION_BITS - _WEIGHT_BITS)) & (_WEIGHT_ONE - 1)
    return first, np.minimum(first + 1, size_in - 1), weight


def scale_plane(plane: np.ndarray, height: int, width: int) -> np.ndarray:
    """`plane` (uint8, rows x columns) scaled to `height` x `width`."""
    if plane.shape == (height, width):
        return plane
    x0, x1, wx = _taps(plane.shape[1], width)
    y0, y1, wy = _taps(plane.shape[0], height)
    p = plane.astype(np.int64)
    rows = p[:, x0] * (_WEIGHT_ONE - wx) + p[:, x1] * wx
    total = (rows[y0] * (_WEIGHT_ONE - wy)[:, None] + rows[y1] * wy[:, None]
             + (1 << (2 * _WEIGHT_BITS - 1)))
    return (total >> (2 * _WEIGHT_BITS)).astype(np.uint8)


def scale_frame(frame: bytes, coded: YuvFormat, shown: YuvFormat) -> bytes:
    """One I420 frame of the `coded` size as displayed at the `shown` size."""
    if coded == shown:
        return frame
    planes = split_planes(np.frombuffer(frame, dtype=np.uint8), coded)
    sizes = [(shown.height, shown.width)] + [(shown.height // 2, shown.width // 2)] * 2
    return b"".join(scale_plane(p, h, w).tobytes() for p, (h, w) in zip(planes, sizes))


class DisplaySink:
    """Scales each frame of a switched stream to the stream size, then hands
    it to `sinks` (a FrameSink; frames arrive in order)."""

    def __init__(self, coded: SwitchedFormat | YuvFormat, shown: YuvFormat, sinks: list) -> None:
        self._coded, self._shown, self._sinks = coded, shown, sinks
        self._index = 0

    def consume(self, frame: bytes) -> None:
        scaled = scale_frame(frame, frame_format_at(self._coded, self._index), self._shown)
        self._index += 1
        for sink in self._sinks:
            sink.consume(scaled)

    def finish(self) -> None:
        for sink in self._sinks:
            sink.finish()


def switched_format(sizes: list[tuple[int, int]]) -> SwitchedFormat | YuvFormat:
    """The frame format of a stream whose frame i is coded at `sizes[i]`."""
    first = YuvFormat(*sizes[0])
    switches, current = [], first
    for index, size in enumerate(sizes):
        fmt = YuvFormat(*size)
        if fmt != current:
            switches.append((index, fmt))
            current = fmt
    return SwitchedFormat(first, tuple(switches)) if switches else first
