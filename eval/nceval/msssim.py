"""Float MS-SSIM equal to libvmaf's `float_ms_ssim`, at about a quarter of its CPU.

libvmaf's MS-SSIM (its iqa port: Wang's 5-scale MS-SSIM on the luma plane,
11-tap Gaussian window with sigma 1.5, 9/7 biorthogonal low-pass before each
2x decimation, symmetric borders) is scalar C: on a 720p frame it costs
~125-190 ms, which made it the most expensive part of a real-use point (4.7
of 13-15 CPU-s, scoring every 4th frame; 2026-11-11). This is the same
computation on OpenCV's SIMD filters, in row strips so that a strip's dozen
intermediate planes stay in cache (whole-frame planes spill to memory and
run 4x slower on the loaded machine): ~35 ms per 720p frame.

It reproduces libvmaf's definition step by step (libvmaf/src/feature/
ms_ssim.c, iqa/ssim_tools.c, iqa/convolve.c, iqa/decimate.c, v3.2.1):

  - each scale's SSIM uses only the positions where the 11x11 window lies
    inside the image ("valid" convolution), with the separable taps;
  - decimation applies the 2-D 9x9 table at every even position with
    symmetric (edge-repeating) borders, giving ceil(w/2) x ceil(h/2);
  - variances are clamped at 0, contrast uses sqrt(var_x * var_y), and a
    negative covariance on a flat window counts as 0 (libvmaf's fix for
    identical flat regions);
  - the score is the product over scales of mean(c)^beta * mean(s)^gamma,
    times mean(l)^alpha at the coarsest scale, each mean taken by magnitude
    (`_power`).

Only the floating-point summation order differs (float32 SIMD instead of
double scalar sums): per-frame scores agree with libvmaf within 1e-6
(tests/test_msssim.py checks it against the pinned libvmaf build), below
the 6 decimals libvmaf prints, so cached libvmaf scores stay comparable.

The filter tables are libvmaf's (BSD+Patent, Copyright 2016-2020 Netflix,
Inc.; THIRD_PARTY.md).
"""

from __future__ import annotations

import math
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from .yuv import YuvFormat

# The point pipeline accounts one CPU slot per metric (nceval.slots):
# OpenCV must not fan out into its own thread pool.
cv2.setNumThreads(1)

#: 11-tap Gaussian window (sigma 1.5), libvmaf's g_gaussian_window_h/_v.
GAUSSIAN_TAPS = np.array([0.001028, 0.007599, 0.036001, 0.109361, 0.213006, 0.266012,
                          0.213006, 0.109361, 0.036001, 0.007599, 0.001028], np.float32)
WINDOW = len(GAUSSIAN_TAPS)
#: The 9x9 9/7 biorthogonal low-pass applied before each decimation, libvmaf's
#: g_lpf: its decimation uses this 2-D table, not the separable taps (which
#: differ from it in the sixth decimal).
LOW_PASS_2D = np.array([
    [0.000714, -0.000450, -0.002090, 0.007132, 0.016114, 0.007132, -0.002090, -0.000450, 0.000714],
    [-0.000450, 0.000283, 0.001316, -0.004490, -0.010146, -0.004490, 0.001316, 0.000283, -0.000450],
    [-0.002090, 0.001316, 0.006115, -0.020867, -0.047149, -0.020867, 0.006115, 0.001316, -0.002090],
    [0.007132, -0.004490, -0.020867, 0.071207, 0.160885, 0.071207, -0.020867, -0.004490, 0.007132],
    [0.016114, -0.010146, -0.047149, 0.160885, 0.363505, 0.160885, -0.047149, -0.010146, 0.016114],
    [0.007132, -0.004490, -0.020867, 0.071207, 0.160885, 0.071207, -0.020867, -0.004490, 0.007132],
    [-0.002090, 0.001316, 0.006115, -0.020867, -0.047149, -0.020867, 0.006115, 0.001316, -0.002090],
    [-0.000450, 0.000283, 0.001316, -0.004490, -0.010146, -0.004490, 0.001316, 0.000283, -0.000450],
    [0.000714, -0.000450, -0.002090, 0.007132, 0.016114, 0.007132, -0.002090, -0.000450, 0.000714],
], np.float32)
#: Per-scale exponents (Wang et al. 2003): luminance only at the coarsest scale.
ALPHAS = (0.0, 0.0, 0.0, 0.0, 0.1333)
BETAS = GAMMAS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
SCALES = len(BETAS)
DYNAMIC_RANGE = 255.0
C1 = (0.01 * DYNAMIC_RANGE) ** 2
C2 = (0.03 * DYNAMIC_RANGE) ** 2
C3 = C2 / 2.0
#: Output rows per strip: ~12 float planes of (STRIP_ROWS + 10) x width fit
#: in L2/L3 at 720p; 32-64 measured fastest (8: 1.6x, whole frame: 4x slower).
STRIP_ROWS = 32
#: libvmaf's border mode for decimation (x < 0 reads -1-x): OpenCV's REFLECT.
SYMMETRIC_BORDER = cv2.BORDER_REFLECT


def ms_ssim(reference: np.ndarray, distorted: np.ndarray) -> float:
    """MS-SSIM of two 8-bit luma planes (h, w), as libvmaf's float_ms_ssim."""
    if reference.shape != distorted.shape:
        raise ValueError(f"plane shapes differ: {reference.shape} vs {distorted.shape}")
    height, width = reference.shape
    if min(height, width) >> (SCALES - 1) < WINDOW:
        raise ValueError(f"{width}x{height} is too small for {SCALES}-scale MS-SSIM")
    x, y = reference.astype(np.float32), distorted.astype(np.float32)
    score = 1.0
    for scale in range(SCALES):
        luminance, contrast, structure = _ssim_terms(x, y)
        score *= (_power(luminance, ALPHAS[scale]) * _power(contrast, BETAS[scale])
                  * _power(structure, GAMMAS[scale]))
        if scale + 1 < SCALES:
            x, y = _decimate(x), _decimate(y)
    return score


def _power(mean: float, exponent: float) -> float:
    """|mean| ** exponent, as VMAFx's float_ms_ssim and its CUDA extractor (the
    default metrics backend) take each scale's term.

    A structure mean can be negative: at a coarse scale of a starved stream the
    decoded detail anti-correlates with the source (an x265 ultrafast 35 kbps
    frame: -0.12 at scale 2). Upstream libvmaf's Wang path then multiplies in
    pow(negative, fractional) = NaN, and math.pow raised; libvmaf's own
    reduction (the Rouse path) and VMAFx take the magnitude, so both backends
    agree and the frame keeps a finite score.
    """
    return math.pow(abs(mean), exponent)


def _decimate(plane: np.ndarray) -> np.ndarray:
    return cv2.filter2D(plane, -1, LOW_PASS_2D, borderType=SYMMETRIC_BORDER)[::2, ::2]


def _window_mean(strip: np.ndarray) -> np.ndarray:
    """The Gaussian window's mean at every position where it fits in `strip`."""
    edge = WINDOW // 2
    filtered = cv2.sepFilter2D(strip, -1, GAUSSIAN_TAPS, GAUSSIAN_TAPS,
                               borderType=SYMMETRIC_BORDER)
    return filtered[edge:-edge, edge:-edge]


def _ssim_terms(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Mean luminance, contrast and structure terms over the valid positions."""
    rows = x.shape[0] - WINDOW + 1
    sums = np.zeros(3)
    for top in range(0, rows, STRIP_ROWS):
        end = min(rows, top + STRIP_ROWS) + WINDOW - 1
        sums += _strip_sums(x[top:end], y[top:end])
    return tuple(sums / (rows * (x.shape[1] - WINDOW + 1)))


def _strip_sums(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu_x, mu_y = _window_mean(x), _window_mean(y)
    var_x = np.maximum(_window_mean(x * x) - mu_x * mu_x, 0.0)
    var_y = np.maximum(_window_mean(y * y) - mu_y * mu_y, 0.0)
    covariance = _window_mean(x * y) - mu_x * mu_y
    sigma_xy = np.sqrt(var_x * var_y)
    covariance[(covariance < 0) & (sigma_xy <= 0)] = 0.0
    luminance = (2.0 * mu_x * mu_y + C1) / (mu_x * mu_x + mu_y * mu_y + C1)
    contrast = (2.0 * sigma_xy + C2) / (var_x + var_y + C2)
    structure = (covariance + C3) / (sigma_xy + C3)
    return np.array([term.sum(dtype=np.float64) for term in (luminance, contrast, structure)])


class MsSsimSink:
    """Scores every `subsample`-th streamed frame's luma on a worker thread.

    A FrameSink (nceval.framepipe): the stream's reader only hands frames
    over, so scoring runs beside it like a metric process would. The
    reference luma is read from the clip at the same frame index.
    """

    #: Frames queued for the worker before the stream waits (memory bound).
    QUEUE_FRAMES = 2

    def __init__(self, reference: Path, fmt: YuvFormat, subsample: int = 1):
        if subsample < 1:
            raise ValueError("MS-SSIM subsample must be positive")
        self._reference = reference.open("rb")
        self._fmt, self._subsample = fmt, subsample
        self._index = 0
        self.scores: list[float] = []
        self._error: BaseException | None = None
        self._queue: queue.Queue = queue.Queue(self.QUEUE_FRAMES)
        self._finished = False
        self._worker = threading.Thread(target=self._score, name="ms-ssim", daemon=True)
        self._worker.start()

    def consume(self, frame: bytes) -> None:
        # The reference is read in order, never sought: an encoded clip's frames
        # arrive through a FIFO (nceval.clipframes), which cannot seek.
        reference = self._reference.read(self._fmt.frame_size)
        if len(reference) != self._fmt.frame_size:
            raise ValueError("reference has fewer frames than the distorted stream")
        if self._index % self._subsample == 0:
            self._queue.put((reference[:self._fmt.luma_size], frame))
        self._index += 1

    def finish(self) -> None:
        """End of stream: wait for the worker; re-raise its error. Idempotent."""
        if not self._finished:
            self._finished = True
            self._queue.put(None)
            self._worker.join()
            self._reference.close()
        if self._error is not None:
            raise self._error

    def _score(self) -> None:
        shape = (self._fmt.height, self._fmt.width)
        while (item := self._queue.get()) is not None:
            if self._error is not None:
                continue  # keep draining so the stream never blocks
            reference, frame = item
            try:
                self.scores.append(ms_ssim(
                    np.frombuffer(reference, np.uint8).reshape(shape),
                    np.frombuffer(frame, np.uint8, self._fmt.luma_size).reshape(shape)))
            except BaseException as error:
                self._error = error
