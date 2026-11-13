"""H.264 / HEVC coding-tool activity of a stream, decoder-side.

H.264 (ours and openh264 alike):
  - macroblock types from libavcodec's `-debug mb_type` map (skip, intra,
    inter partition) per frame;
  - motion vectors from libavcodec's exported side data (PyAV,
    `flags2=+export_mvs`), quarter-pel, for the sub-pel phase histogram;
  - deblocking enable per slice from the slice headers.
HEVC: libavcodec exports neither block types nor vectors, so only the slice
headers are read (SAO luma/chroma and deblocking enable shares); block-level
HEVC activity needs a decoder-side parser we do not have yet.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from ..codecs import Codec
from ..config import Paths, find_tool
from .activity import Activity, mean_activity, subpel_phase_shares
from .headers import HeaderTrace, trace_headers

QUARTER_PEL = 4
# libavcodec's mb_type debug letters (libavcodec/mpegutils.c).
SKIP_LETTERS = {"S", "d", "g"}
INTRA_LETTERS = {"i", "I", "A", "P"}
PARTITION_NAMES = {" ": "16x16", "-": "16x8", "|": "8x16", "+": "8x8"}
MB_CELL = 3  # characters per macroblock in the map
_NEW_FRAME = re.compile(r"New frame, type: (\S)")
_MAP_ROW = re.compile(r"\]\s+(\d+) (.*)$")
H264_I_SLICES = {2, 7}  # slice_type I (and "all slices I")


def avc_activity(stream: Path, codec: Codec, paths: Paths) -> dict[str, Activity]:
    trace = trace_headers(stream, paths)
    if codec is Codec.HEVC:
        return hevc_activity(trace)
    vectors = motion_vector_phases(stream)
    # ffmpeg also decodes (and prints) frames while probing the input: the
    # real decode is the last len(vectors) maps.
    maps = mb_type_maps(stream, paths)[-len(vectors):] if vectors else []
    frames = []
    for index, cells in enumerate(maps):
        phases = vectors[index] if index < len(vectors) else {}
        header = trace.frames[index] if index < len(trace.frames) else {}
        frames.append(h264_frame_activity(cells, phases, header))
    return mean_activity(frames)


def h264_frame_activity(cells: list[str], phases: dict[str, float],
                        header: dict[str, int]) -> Activity:
    kinds = Counter(c[0] for c in cells)
    total = max(1, len(cells))
    partitions = Counter(PARTITION_NAMES.get(c[1], "other") for c in cells
                         if c[0] not in INTRA_LETTERS and c[0] not in SKIP_LETTERS)
    intra = sum(kinds[k] for k in INTRA_LETTERS)
    frame_type = "I" if header.get("slice_type") in H264_I_SLICES or intra == total else "P"
    return Activity(
        frames=1, frame_type=frame_type,
        skip_share=sum(kinds[k] for k in SKIP_LETTERS) / total,
        intra_share=intra / total,
        histograms={"block_sizes": {k: v / total for k, v in partitions.items()},
                    "subpel_phases": phases},
        filters={"deblocking_enabled_share":
                 0.0 if header.get("disable_deblocking_filter_idc", 0) == 1 else 1.0})


def hevc_activity(trace: HeaderTrace) -> dict[str, Activity]:
    pps_off = trace.params.get("pps_deblocking_filter_disabled_flag", 0)
    frames = []
    for header in trace.frames:
        deblock_off = header.get("slice_deblocking_filter_disabled_flag", pps_off)
        frames.append(Activity(
            frames=1, frame_type="I" if header.get("slice_type") == 2 else "P",
            skip_share=float("nan"), intra_share=float("nan"),
            filters={"sao_luma_share": float(header.get("slice_sao_luma_flag", 0)),
                     "sao_chroma_share": float(header.get("slice_sao_chroma_flag", 0)),
                     "deblocking_enabled_share": 0.0 if deblock_off else 1.0}))
    return mean_activity(frames)


def parse_mb_type_log(lines: list[str]) -> list[list[str]]:
    """Per frame, the 3-character cell of every macroblock in raster order."""
    frames: list[list[str]] = []
    for line in lines:
        if _NEW_FRAME.search(line):
            frames.append([])
            continue
        row = _MAP_ROW.search(line)
        if not row or not frames:
            continue
        body = row.group(2)
        if not body.strip() or body.replace(" ", "").isdigit():
            continue  # the column-number header line
        frames[-1].extend(body[i:i + MB_CELL].ljust(MB_CELL)
                          for i in range(0, len(body.rstrip()), MB_CELL))
    return frames


def mb_type_maps(stream: Path, paths: Paths) -> list[list[str]]:
    cmd = [str(find_tool("ffmpeg", paths)), "-nostdin", "-threads", "1", "-debug", "mb_type",
           "-i", str(stream), "-f", "null", "-"]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          check=True)
    return parse_mb_type_log(proc.stderr.splitlines())


def motion_vector_phases(stream: Path) -> list[dict[str, float]]:
    """Per decoded frame, the area-weighted sub-pel phase shares of its vectors."""
    import av  # optional dependency (PyAV); only H.264 activity needs it

    out = []
    with av.open(str(stream)) as container:
        video = container.streams.video[0]
        video.codec_context.options = {"flags2": "+export_mvs"}
        video.thread_type = "NONE"
        for frame in container.decode(video):
            side = frame.side_data.get("MOTION_VECTORS")
            if side is None:
                out.append({})
                continue
            mvs = side.to_ndarray()
            if mvs.size == 0:
                out.append({})
                continue
            scale = np.maximum(mvs["motion_scale"], 1)
            rows = mvs["motion_y"] * QUARTER_PEL // scale
            cols = mvs["motion_x"] * QUARTER_PEL // scale
            weights = mvs["w"].astype(np.int64) * mvs["h"]
            out.append(_weighted_phases(rows, cols, weights))
    return out


def _weighted_phases(rows: np.ndarray, cols: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Sub-pel phase shares weighted by partition area (16x16 counts 4x an 8x8)."""
    unit = max(1, int(weights.min()))
    repeats = np.maximum(1, weights // unit)
    return subpel_phase_shares(np.repeat(rows, repeats), np.repeat(cols, repeats), QUARTER_PEL)
