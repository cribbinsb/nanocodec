"""Hindsight inheritance maps of key frames: the key-frame study's allocation oracle.

    maps = key_maps(read_analysis_log(path), strengths=(1, 2, 3))

A key's cells are not worth the same: the frames after it copy some (a still
background) for seconds and overwrite others (a moving person) within a few
frames. MB-tree (x264) and TPL (libaom, SVT-AV1) measure that with a
lookahead and give each block a quantizer offset from its propagated reuse;
this is the same measurement with full hindsight, from a first coding of the
stream (`nanocodec encode --analysis-log`, `key_oracle.rs`), so it bounds
what any causal signal could reach.

Propagation, per key period, from its last frame back to the key: every 8x8
cell of frame t sends `(intra + received) * max(0, 1 - inter / intra)` to
the cells of frame t - 1 its motion vector points at, by area. `intra` is the
Gaussian SAD estimate from the cell's sum of squared deviations (as
`layer_propagation.rs`), `inter` the 8x8 motion search's SAD. The key cell's
offset is `-strength * log2((intra + received) / intra)` QP units (finer
where later frames reuse it): MB-tree's closed form, whose strength
5 * (1 - qcompress) is 2 at x264's default. Rate control budgets the
offsets like any block AQ, so the key keeps its bits and they move between
its cells.

The motion field is against the previous coded frame (flat single-layer
streams); a held frame repeats the last analysis and is skipped. Where the
coded size changes (`resolution.rs`), nothing propagates across the change.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Bytes of one cell record: u32 sum of squared deviations, u32 SAD, i16 x, i16 y.
CELL_DTYPE = np.dtype([("ssd", "<u4"), ("sad", "<u4"), ("mvx", "<i2"), ("mvy", "<i2")])
HEADER = struct.Struct("<IHHB")
CELL = 8
#: SAD of a Gaussian residual of variance v over 64 samples: 64 * sqrt(2 v / pi).
GAUSS_SAD = 64.0 * math.sqrt(2.0 / math.pi)


@dataclass
class FrameRecord:
    index: int
    cols: int
    rows: int
    motion: bool
    cells: np.ndarray  # CELL_DTYPE, rows x cols


def read_analysis_log(path: Path) -> list[FrameRecord]:
    """Every record of an `--analysis-log` file, held frames' repeats dropped."""
    data = Path(path).read_bytes()
    out: list[FrameRecord] = []
    pos = 0
    while pos < len(data):
        index, cols, rows, motion = HEADER.unpack_from(data, pos)
        pos += HEADER.size
        count = cols * rows
        cells = np.frombuffer(data, CELL_DTYPE, count, pos).reshape(rows, cols)
        pos += count * CELL_DTYPE.itemsize
        if out and out[-1].cells.shape == cells.shape and np.array_equal(out[-1].cells, cells):
            continue  # a held frame: the analyzer's last frame again
        out.append(FrameRecord(index, cols, rows, bool(motion), cells))
    return out


def intra_cost(cells: np.ndarray) -> np.ndarray:
    return np.maximum(GAUSS_SAD * np.sqrt(cells["ssd"].astype(np.float64) / 64.0), 1.0)


def send_back(record: FrameRecord, received: np.ndarray) -> np.ndarray:
    """What frame `record` (with its own `received`) sends to the frame before it."""
    intra = intra_cost(record.cells)
    inter = record.cells["sad"].astype(np.float64)
    amount = (intra + received) * np.clip(1.0 - inter / intra, 0.0, 1.0)
    rows, cols = amount.shape
    out = np.zeros_like(amount)
    ys, xs = np.mgrid[0:rows, 0:cols]
    # the reference block's top-left corner, in cells (positive vector: to the right / below)
    fx = xs + record.cells["mvx"] / CELL
    fy = ys + record.cells["mvy"] / CELL
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ax, ay = fx - x0, fy - y0
    for dx, dy, w in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)),
                      (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        tx, ty = x0 + dx, y0 + dy
        inside = (tx >= 0) & (tx < cols) & (ty >= 0) & (ty < rows) & (w > 0)
        np.add.at(out, (ty[inside], tx[inside]), (amount * w)[inside])
    return out


def key_reuse(records: list[FrameRecord], horizon: int | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per key (input frame): ((intra + received) / intra, intra) of each cell, over
    the frames of its period (at most `horizon` of them)."""
    keys = [i for i, r in enumerate(records) if not r.motion]
    out = {}
    for k, start in enumerate(keys):
        end = keys[k + 1] if k + 1 < len(keys) else len(records)
        if horizon:
            end = min(end, start + 1 + horizon)
        received = np.zeros(records[end - 1].cells.shape)
        for t in range(end - 1, start, -1):
            if records[t].cells.shape != records[t - 1].cells.shape:
                received = np.zeros(records[t - 1].cells.shape)
                continue
            received = send_back(records[t], received)
        intra = intra_cost(records[start].cells)
        out[records[start].index] = ((intra + received) / intra, intra)
    return out


def offsets(reuse: np.ndarray, intra: np.ndarray, strength: float, centred: bool) -> np.ndarray:
    """MB-tree's offset (QP units, finer where reused). `centred`: less its mean
    weighted by the cells' intra cost (where the key's bits are), so the map moves
    bits between cells without asking rate control to undo a uniform shift."""
    offset = -strength * np.log2(reuse)
    if centred:
        offset -= np.average(offset, weights=intra)
    return offset


def map_text(maps: dict[int, np.ndarray]) -> str:
    """A `--key-offsets` file of `maps` (by input frame)."""
    lines = []
    for frame, grid in sorted(maps.items()):
        lines.append(f"frame {frame}")
        lines.extend(" ".join(f"{v:.3f}" for v in row) for row in grid)
    return "\n".join(lines) + "\n"


def key_maps(records: list[FrameRecord], strengths, horizon: int | None = None) -> dict:
    """{label: --key-offsets text} and the reuse summary of each key. A strength
    label is a number (MB-tree's offsets as they are) or `c` and a number (centred)."""
    reuse = key_reuse(records, horizon)
    texts = {}
    for label in strengths:
        label = str(label)
        centred = label.startswith("c")
        strength = float(label.removeprefix("c"))
        texts[label] = map_text({f: offsets(r, intra, strength, centred)
                                 for f, (r, intra) in reuse.items()})
    summary = {f: {"log2_reuse_mean": float(np.log2(r).mean()),
                   "log2_reuse_sd": float(np.log2(r).std())} for f, (r, _) in reuse.items()}
    return {"texts": texts, "summary": summary}
