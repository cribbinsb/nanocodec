"""Coding-tool usage of an H.264 or HEVC stream, from our decoders' analysis.

`nanocodec stat` decodes the stream with analysis on and writes, per
picture, tallies of codec-neutral block records (crates/nc-common
`stream_stats`): statistic -> value -> [count, weight], where the weight is
the luma area unless the key says otherwise. This module turns those into a
key-frame / P-frame summary: block, prediction and transform sizes and
shapes, intra mode classes, motion vector precision and sizes, residual
coverage, QP behaviour, SAO and bit split. `streamstats.py --codec
hevc|h264` drives it; docs/analysis/soc-encoder-tools.md is an example.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from pathlib import Path

# Value names of the record enums (nc_common::stream_stats).
PRED = ("intra", "inter", "skip", "merge")
SHAPE = ("whole", "halfH", "halfV", "quarter", "qTop", "qBot", "qLeft", "qRight",
         "sixteenth", "raw")
INTRA_CLASS = ("planar", "dc", "angular", "other")
PRECISION = ("int", "half", "quarter", "eighth")
SAO_TYPE = ("off", "band", "edge")
SAO_SIGNAL = ("mergeL", "mergeU", "explicit")
# Size codes are w * SIZE_CODE_SCALE + h; intra mode by size is
# size_code * INTRA_MODE_SCALE + mode.
SIZE_CODE_SCALE = 1000
INTRA_MODE_SCALE = 100
BITS_PER_BYTE = 8
PERCENT = 100
# A frame-QP step counts towards a reversal when larger than this.
QP_REVERSAL_STEP = 0.5


# The analysis build of the CLI (`--features analysis`, docs/PROCEDURES.md
# "Inspect a stream"): normal builds have no `stat` command.
ANALYSIS_BIN = Path(os.environ.get("NC_ANALYSIS_BIN",
                                   Path.home() / "nanocodec-data" / "targets" / "analysis" / "release"))


def run_stat(stream: Path, codec: str, out: Path) -> list[dict]:
    """Decode `stream` with `nanocodec stat` and return its per-frame tallies."""
    binary = ANALYSIS_BIN / "nanocodec"
    if not binary.is_file():
        raise FileNotFoundError(f"{binary}: build the CLI with --features analysis (docs/PROCEDURES.md)")
    subprocess.run([str(binary), "stat", "--codec", codec, "--input", str(stream),
                    "--json", str(out)], check=True)
    return json.loads(out.read_text())["frames"]


def _merge(frames: list[dict]) -> dict[str, dict[int, list[int]]]:
    out: dict[str, dict[int, list[int]]] = {}
    for frame in frames:
        for key, values in frame.items():
            target = out.setdefault(key, {})
            for value, (count, weight) in values.items():
                tally = target.setdefault(int(value), [0, 0])
                tally[0] += count
                tally[1] += weight
    return out


def size_name(code: int) -> str:
    w, h = divmod(code, SIZE_CODE_SCALE)
    return f"{w}" if w == h else f"{w}x{h}"


def _share(d: dict, key: str, label=None, by_count: bool = False) -> dict:
    values = d.get(key, {})
    index = 0 if by_count else 1
    total = sum(t[index] for t in values.values()) or 1
    return {(label(v) if label else v): round(PERCENT * t[index] / total, 1)
            for v, t in sorted(values.items())}


def _total(d: dict, key: str, value: int | None = None) -> int:
    return sum(t[1] for v, t in d.get(key, {}).items() if value is None or v == value)


def summarise(frames: list[dict]) -> dict:
    """Shares (% of luma area unless noted) over `frames`."""
    d = _merge(frames)
    area = _total(d, "cb_size") or 1
    bits = _total(d, "slice_bytes") * BITS_PER_BYTE or 1
    intra_by_size: dict[str, int] = {}
    modes_by_size: dict[str, set[int]] = {}
    for value, (_, weight) in d.get("intra_mode_by_size", {}).items():
        size, mode = divmod(value, INTRA_MODE_SCALE)
        intra_by_size[size_name(size)] = intra_by_size.get(size_name(size), 0) + weight
        modes_by_size.setdefault(size_name(size), set()).add(mode)
    intra_area = sum(intra_by_size.values()) or 1
    coded = d.get("cb_coded", {})
    coded_area = sum(t[1] for t in coded.values()) or 1
    deltas = d.get("cb_qp_delta", {})
    return {
        "cb_size": _share(d, "cb_size", size_name),
        **{f"cb_size_{p}": _share(d, f"cb_size_{p}", size_name) for p in ("intra", "inter", "skip")},
        "cb_pred": _share(d, "cb_pred", lambda v: PRED[v]),
        "cb_shape_intra": _share(d, "cb_shape_intra", lambda v: SHAPE[v]),
        "cb_shape_inter": _share(d, "cb_shape_inter", lambda v: SHAPE[v]),
        "pb_pred": _share(d, "pb_pred", lambda v: PRED[v]),
        "pb_size_inter": _share(d, "pb_size", size_name),
        "merge_idx": _share(d, "merge_idx"),
        "ref_idx": _share(d, "ref_idx"),
        "mv_precision": _share(d, "mv_precision", lambda v: PRECISION[v]),
        "mv_zero": _share(d, "mv_zero").get(1, 0),
        "mv_size_bucket": _share(d, "mv_size"),
        "mvd_size_bucket": _share(d, "mvd_size"),
        "intra_class": _share(d, "intra_class", lambda v: INTRA_CLASS[v]),
        "intra_pb_size": {k: round(PERCENT * w / intra_area, 1) for k, w in sorted(intra_by_size.items())},
        "intra_modes_by_pb_size": {k: len(v) for k, v in sorted(modes_by_size.items())},
        "intra_chroma_mode": _share(d, "intra_chroma_mode"),
        "tb_size_intra": _share(d, "tb_size_intra", size_name),
        "tb_size_inter": _share(d, "tb_size_inter", size_name),
        "tb_depth_intra": _share(d, "tb_depth_intra"),
        "tb_depth_inter": _share(d, "tb_depth_inter"),
        "tb_coded": {p: _share(d, f"tb_coded_{p}").get(1, 0) for p in ("y", "cb", "cr")},
        "luma_coded_area": round(PERCENT * _total(d, "tb_coded_y", 1) / area, 1),
        "cb_any_residual": round(PERCENT * sum(t[1] for v, t in coded.items() if v) / coded_area, 1),
        "transform_skip": _share(d, "tb_transform_skip").get(1, 0),
        "nonzero_per_1000_px": round(1000 * _total(d, "tb_nonzero") / area, 1),
        "cb_qp": _share(d, "cb_qp"),
        "qp_delta": {"coded": sum(t[0] for t in deltas.values()),
                     "nonzero_per_frame": round(sum(t[0] for v, t in deltas.items() if v) / len(frames), 1),
                     "range": [min(deltas), max(deltas)] if deltas else None},
        "sao_luma": _share(d, "filter_luma", lambda v: SAO_TYPE[v], by_count=True),
        "sao_chroma": _share(d, "filter_chroma", lambda v: SAO_TYPE[v], by_count=True),
        "sao_signal": _share(d, "filter_signal", lambda v: SAO_SIGNAL[v], by_count=True),
        "bits_share": {"residual": round(PERCENT * _total(d, "tb_bits") / bits, 1),
                       "mvd": round(PERCENT * _total(d, "mvd_bits") / bits, 1),
                       "sao": round(PERCENT * _total(d, "filter_bits") / bits, 1),
                       **{f"in_{PRED[v]}_blocks": round(PERCENT * t[1] / bits, 1)
                          for v, t in sorted(d.get("cb_bits", {}).items())}},
    }


def _frame_qp(frame: dict) -> float:
    """Area-mean QP of a frame's coding blocks."""
    qps = frame["cb_qp"]
    return sum(int(v) * w for v, (_, w) in qps.items()) / sum(w for _, w in qps.values())


def rate_behaviour(frames: list[dict]) -> dict:
    """Key-frame cost and how frame size and QP move over the P frames."""
    qps = [_frame_qp(f) for f in frames]
    sizes = [_total(_merge([f]), "slice_bytes") for f in frames]
    p_sizes, p_qps = sizes[1:], qps[1:]
    steps = [b - a for a, b in zip(p_qps, p_qps[1:])]
    reversals = sum(1 for a, b in zip(steps, steps[1:])
                    if a * b < 0 and abs(a) > QP_REVERSAL_STEP and abs(b) > QP_REVERSAL_STEP)
    return {"key_bytes": sizes[0], "key_qp": round(qps[0], 1),
            "p_bytes_mean": round(statistics.mean(p_sizes)),
            "p_bytes_cv": round(PERCENT * statistics.pstdev(p_sizes) / statistics.mean(p_sizes), 1),
            "p_qp_mean": round(statistics.mean(p_qps), 1), "p_qp_sd": round(statistics.pstdev(p_qps), 2),
            "p_qp_mean_abs_step": round(statistics.mean(abs(s) for s in steps), 2),
            "p_qp_reversals": reversals}


def header_facts(frames: list[dict]) -> dict:
    """Parameter-set and slice-header values seen anywhere in the stream."""
    keys = sorted({k for f in frames for k in f
                   if k.startswith(("sps_", "pps_", "slice_")) and k not in ("slice_bytes", "slice_qp")})
    return {k: sorted({int(v) for f in frames for v in f.get(k, {})}) for k in keys}


def analyse(frames: list[dict]) -> dict:
    """Header facts, rate behaviour and key / P summaries of a stream."""
    is_key = [2 in {int(v) for v in f.get("slice_type", {})} for f in frames]
    keys = [f for f, k in zip(frames, is_key) if k]
    inter = [f for f, k in zip(frames, is_key) if not k]
    return {"header": header_facts(frames), "rate": rate_behaviour(frames),
            "key": summarise(keys) if keys else None, "P": summarise(inter) if inter else None}
