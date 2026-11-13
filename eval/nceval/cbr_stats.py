"""Per-stream statistics for constant-bitrate (CBR) rate-control runs.

Pure functions over per-frame packet sizes and per-frame quality, shared by
the CBR benchmark (cbr.py). They are encoder-independent: anchors and our own
encoder are judged by the same replay of their packets, never by an encoder's
self-reported buffer state.

Buffer model: a constant-rate channel drains the encoder-side buffer at the
target rate; each coded frame adds its packet. The buffer holds `buffer_ms` of
target-rate data and starts empty. A frame that leaves more than that queued
is an overrun: a receiver with that buffer would have stalled. This is the
strict CBR envelope (drain = target, not a higher peak rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Worst-frame quality is the 1st percentile of per-frame quality.
WORST_PERCENTILE = 1.0
MS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class BufferReplay:
    overruns: int  # frames after which the buffer held more than its capacity
    max_fullness: float  # peak occupancy as a fraction of capacity
    max_window_ratio: float  # worst one-second bitrate / target


def replay_buffer(frame_bits: Sequence[int], target_bps: float, fps: float,
                  buffer_ms: float) -> BufferReplay:
    """Leaky-bucket replay of the packets against the strict CBR channel."""
    capacity = target_bps * buffer_ms / MS_PER_SECOND
    drain = target_bps / fps
    occupancy, peak, overruns = 0.0, 0.0, 0
    for bits in frame_bits:
        occupancy += bits
        peak = max(peak, occupancy)
        if occupancy > capacity:
            overruns += 1
        occupancy = max(0.0, occupancy - drain)
    window = max(1, round(fps))
    sums = np.convolve(np.asarray(frame_bits, dtype=float), np.ones(window), mode="valid")
    worst_window = float(sums.max()) if len(sums) else float(sum(frame_bits))
    return BufferReplay(overruns=overruns, max_fullness=peak / capacity,
                        max_window_ratio=worst_window / target_bps)


def worst_quality(per_frame: Sequence[float]) -> float:
    """1st-percentile frame quality: what the worst moments look like."""
    return float(np.percentile(np.asarray(per_frame, dtype=float), WORST_PERCENTILE))


def flicker(per_frame: Sequence[float]) -> float:
    """Mean absolute frame-to-frame quality change (temporal quality pumping)."""
    values = np.asarray(per_frame, dtype=float)
    return float(np.abs(np.diff(values)).mean()) if len(values) > 1 else 0.0


def weighted_yuv_per_frame(y: Sequence[float], u: Sequence[float],
                           v: Sequence[float]) -> list[float]:
    """Per-frame JVET 6:1:1 PSNR-YUV (the sequence value averages planes first)."""
    return [(6.0 * a + b + c) / 8.0 for a, b, c in zip(y, u, v)]
