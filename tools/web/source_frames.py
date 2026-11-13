"""The demo's live source frames, offline: a clip's decoded YUV planes placed in
a view exactly as nc-web `SourceFrame` places them (crates/nc-web/src/source.rs).

The page codes a video's own decoded planes (web/capture-worker.js): at the
file's size they pass through unchanged; at another size each plane is scaled
with the player model's bilinear filter (nc-dsp `display`, whose bit-exact
Python twin is eval/nceval/display.py) into the largest centred rectangle of
the source's aspect ratio, with even size and offsets, and the rest is black
(Y 16 in limited range, 0 in full; Cb, Cr 128). make-reference-streams.py codes
the same pixels, and web/tests/capture-identity.py checks the browser against
this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval.display import scale_plane  # noqa: E402

LIMITED_BLACK, NEUTRAL_CHROMA = 16, 128


def placement(source: tuple[int, int], view: tuple[int, int]) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the source in the view: nc-web `Placement::of`."""
    (sw, sh), (vw, vh) = source, view

    def even(numerator: int, denominator: int) -> int:
        return 2 * ((numerator + denominator) // (2 * denominator))

    if sw * vh >= sh * vw:
        width, height = vw, min(even(sh * vw, sw), vh)
    else:
        width, height = min(even(sw * vh, sh), vw), vh
    width, height = max(width, 1), max(height, 1)
    return (vw - width) // 4 * 2, (vh - height) // 4 * 2, width, height


def view_frame(planes: list[np.ndarray], view: tuple[int, int], full_range: bool = False) -> bytes:
    """One frame's Y, Cb, Cr planes (uint8 arrays) as packed I420 of the view size."""
    vw, vh = view
    source = (planes[0].shape[1], planes[0].shape[0])
    x, y, w, h = placement(source, view)
    black = 0 if full_range else LIMITED_BLACK
    out = []
    for index, plane in enumerate(planes):
        if index == 0:
            canvas = np.full((vh, vw), black, np.uint8)
            rect = (x, y, w, h)
        else:
            canvas = np.full(((vh + 1) // 2, (vw + 1) // 2), NEUTRAL_CHROMA, np.uint8)
            rect = (x // 2, y // 2, (w + 1) // 2, (h + 1) // 2)
        rx, ry, rw, rh = rect
        canvas[ry:ry + rh, rx:rx + rw] = scale_plane(plane, rh, rw)
        out.append(canvas.tobytes())
    return b"".join(out)


def split_i420(frame: bytes, width: int, height: int) -> list[np.ndarray]:
    """Packed I420 bytes as Y, Cb, Cr arrays."""
    cw, ch = (width + 1) // 2, (height + 1) // 2
    data = np.frombuffer(frame, np.uint8)
    y = data[:width * height].reshape(height, width)
    cb = data[width * height:width * height + cw * ch].reshape(ch, cw)
    cr = data[width * height + cw * ch:width * height + 2 * cw * ch].reshape(ch, cw)
    return [y, cb, cr]
