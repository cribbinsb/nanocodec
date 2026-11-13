"""Standard stats map of a coded stream (eval/README.md "Methodology: Diagnostics").

Extracted decoder-side from the bitstream, so it works the same for any
encoder, ours or an anchor. One codec-neutral schema per frame:

    {
      "index", "type" ("I"/"P"), "bytes", "q" (codec-native base quantiser),
      "bits":  {category: bits}       # headers, partition, intra_mode, ref,
                                      # inter_mode, mv, skip, tx_type,
                                      # coeff, cdef, other
      "area":  {"by_size": {"64x64": share, ...},
                "by_mode": {"intra", "nearest", "near", "new", "global",
                            "other_inter": share},
                "skip": share},       # share of picture area, 0..1
      "tx_area": {"32x32": share, ...},   # coded (non-skip) area only
      "mv": {"new_blocks", "new_area", "mean_abs_pel", "subpel_share"},
      "delta_q": {"min", "max"},      # per-block quantiser actually used
      ["psnr_y", "psnr_u", "psnr_v"]  # when a source is given
    }

Coefficient bits cannot be split by plane: the decoder's accounting labels
symbols by syntax element, not plane.

AV1 is read with libaom's inspection decoder (`aom-inspect`, built by
`tools/anchors/build_anchors.sh aominspect`). H.264 and HEVC can be added later
behind the same schema.

    python3 -m nceval.streamstats STREAM.ivf [--source SRC.yuv --width W
        --height H] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import find_tool, load_paths

# Inspection output flags: block size, tx size, mode, skip, MVs, delta-q,
# accounting. Everything else stays off to keep the JSON small.
INSPECT_FLAGS = ("-bs", "-ts", "-m", "-s", "-mv", "-dq", "-a")
MI_SIZE = 4  # AV1 mode-info unit in luma samples
EIGHTH_PEL = 8

# libaom accounting symbol names -> codec-neutral bit categories.
AV1_SYMBOL_CATEGORY = {
    "read_partition": "partition",
    "read_skip_txfm": "skip",
    "read_cdef": "cdef",
    "read_intra_mode": "intra_mode",
    "read_intra_mode_uv": "intra_mode",
    "cfl:signs": "intra_mode",
    "cfl:alpha_u": "intra_mode",
    "cfl:alpha_v": "intra_mode",
    "read_coeffs_txb": "coeff",
    "read_coeffs_reverse_2d": "coeff",
    "read_coeffs_reverse": "coeff",
    "read_golomb": "coeff",
    "av1_read_tx_type": "tx_type",
    "read_is_inter_block": "ref",
    "read_block_reference_mode": "ref",
    "read_ref_frames": "ref",
    "read_inter_mode": "inter_mode",
    "read_drl_idx": "inter_mode",
    "read_mv": "mv",
    "read_mv_component": "mv",
}

AV1_MODE_CLASS = {
    "NEARESTMV": "nearest", "NEARMV": "near", "NEWMV": "new", "GLOBALMV": "global",
}


# ----------------------------------------------------------------------------
# Container and decoder plumbing


def ivf_frame_sizes(path: Path) -> list[int]:
    data = path.read_bytes()
    header_len = struct.unpack_from("<H", data, 6)[0]
    sizes, pos = [], header_len
    while pos + 12 <= len(data):
        size = struct.unpack_from("<I", data, pos)[0]
        sizes.append(size)
        pos += 12 + size
    return sizes


def inspect_frames(stream: Path, flags: tuple[str, ...] = INSPECT_FLAGS) -> Iterator[dict]:
    """Yield libaom's per-frame inspection objects one at a time.

    The tool prints one JSON array; at 720p each frame is several MB, so the
    array is split on its top-level object boundaries instead of being parsed
    whole.
    """
    tool = find_tool("aom-inspect", load_paths())
    with subprocess.Popen([str(tool), str(stream), *flags],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True) as proc:
        assert proc.stdout is not None
        lines: list[str] = []
        for line in proc.stdout:
            if line.startswith("{"):
                lines = [line]
            elif lines and line.rstrip() in ("}", "},"):
                lines.append("}")
                yield json.loads("".join(lines))
                lines = []
            elif lines:
                lines.append(line)
    if proc.returncode not in (0, None):
        raise RuntimeError(f"aom-inspect failed on {stream} ({proc.returncode})")


# ----------------------------------------------------------------------------
# Per-frame summary


def summarise_av1_frame(frame: dict, frame_bytes: int) -> dict:
    sizes = np.array(frame["blockSize"])
    total_mi = sizes.size
    size_names = _invert(frame["blockSizeMap"])
    mode_names = _invert(frame["modeMap"])
    tx_names = _invert(frame["transformSizeMap"])

    by_size = _area_shares(sizes, size_names, lambda n: n.removeprefix("BLOCK_").lower())
    skipped = np.array(frame["skip"]) == frame["skipMap"]["SKIP"]
    skip = float(skipped.mean())
    # Transform sizes of coded (non-skip) blocks only, as shares of the whole
    # picture, so they sum to 1 - skip.
    coded_tx = np.array(frame["transformSize"])[~skipped]
    tx_area = {k: v * coded_tx.size / total_mi for k, v in
               _area_shares(coded_tx, tx_names, lambda n: n.removeprefix("TX_").lower()).items()
               } if coded_tx.size else {}

    modes = np.array(frame["mode"])
    by_mode: dict[str, float] = defaultdict(float)
    for value, count in zip(*np.unique(modes, return_counts=True)):
        name = mode_names[int(value)]
        cls = AV1_MODE_CLASS.get(name, "other_inter" if name.endswith("MV") else "intra")
        by_mode[cls] += count / total_mi

    mv = _mv_stats(frame, modes, sizes, size_names, mode_names)
    bits = _bit_categories(frame, frame_bytes)
    delta_q = np.array(frame["delta_q"])
    return {
        "type": "I" if frame["frameType"] == 0 else "P",
        "bytes": frame_bytes,
        "q": frame["baseQIndex"],
        "bits": bits,
        "area": {"by_size": by_size, "by_mode": dict(by_mode), "skip": skip},
        "tx_area": tx_area,
        "mv": mv,
        "delta_q": {"min": int(delta_q.min()), "max": int(delta_q.max())},
    }


def _invert(mapping: dict) -> dict[int, str]:
    return {v: k for k, v in mapping.items()}


def _area_shares(grid: np.ndarray, names: dict[int, str], label) -> dict[str, float]:
    values, counts = np.unique(grid, return_counts=True)
    return {label(names[int(v)]): c / grid.size for v, c in zip(values, counts)}


def _block_mi_area(name: str) -> int:
    w, h = name.removeprefix("BLOCK_").split("X")
    return (int(w) // MI_SIZE) * (int(h) // MI_SIZE)


def _mv_stats(frame: dict, modes: np.ndarray, sizes: np.ndarray,
              size_names: dict[int, str], mode_names: dict[int, str]) -> dict:
    new_code = frame["modeMap"].get("NEWMV")
    mvs = np.array(frame["motionVectors"])[..., :2]  # (row, col) of ref 0, 1/8 pel
    inter = np.isin(modes, [frame["modeMap"][m] for m in AV1_MODE_CLASS
                            if m in frame["modeMap"]])
    new = modes == new_code if new_code is not None else np.zeros_like(inter)
    mi_area = np.vectorize(lambda s: _block_mi_area(size_names[int(s)]))(sizes)
    stats = {
        "new_area": float(new.mean()),
        # Each block covers mi_area units, so summing 1/mi_area counts blocks.
        "new_blocks": float((1.0 / mi_area[new]).sum()),
        "mean_abs_pel": 0.0,
        "subpel_share": 0.0,
    }
    if inter.any():
        sel = mvs[inter]
        stats["mean_abs_pel"] = float(np.abs(sel).mean() / EIGHTH_PEL)
        stats["subpel_share"] = float(np.mean(np.any(sel % EIGHTH_PEL != 0, axis=-1)))
    return stats


def _bit_categories(frame: dict, frame_bytes: int) -> dict[str, float]:
    names = frame["symbolsMap"]
    bits: dict[str, float] = defaultdict(float)
    for entry in frame["symbols"]:
        if len(entry) == 3:  # [symbol id, bits in 1/8 bit, samples]
            category = AV1_SYMBOL_CATEGORY.get(names[entry[0]], "other")
            bits[category] += entry[1] / 8.0
        # Two-element entries mark a new (x, y) context, not a symbol.
    # This residual includes headers AND symbol-accounting/coder differences.
    # It is not a direct header-size measurement; use trace_headers for that.
    bits["headers"] = max(0.0, frame_bytes * 8 - sum(bits.values()))
    return dict(bits)


# ----------------------------------------------------------------------------
# Quality, aggregation and CLI


def per_frame_psnr(stream: Path, source: Path, width: int, height: int,
                   frames: int) -> list[tuple[float, float, float]]:
    """Decode with ffmpeg/dav1d and return per-frame (Y, U, V) PSNR, film
    grain not applied (as the gate scores it, nceval.conformance)."""
    paths = load_paths()
    with tempfile.TemporaryDirectory(dir=paths.work) as tmp:
        decoded = Path(tmp) / "decoded.yuv"
        subprocess.run([str(find_tool("ffmpeg", paths)), "-v", "error", "-nostdin",
                        "-export_side_data", "film_grain", "-c:v", "libdav1d", "-i", str(stream), "-f", "rawvideo",
                        "-pix_fmt", "yuv420p", "-y", str(decoded)], check=True)
        luma, chroma = width * height, (width // 2) * (height // 2)
        frame_len = luma + 2 * chroma
        ref = np.memmap(source, dtype=np.uint8, mode="r")
        dec = np.memmap(decoded, dtype=np.uint8, mode="r")
        out = []
        for i in range(frames):
            a = ref[i * frame_len:(i + 1) * frame_len].astype(np.int32)
            b = dec[i * frame_len:(i + 1) * frame_len].astype(np.int32)
            planes = [(0, luma), (luma, luma + chroma), (luma + chroma, frame_len)]
            out.append(tuple(_psnr(a[s:e], b[s:e]) for s, e in planes))
        return out


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse == 0 else min(100.0, 10 * np.log10(255.0 ** 2 / mse))


def analyse(stream: Path, source: Path | None = None, width: int = 0,
            height: int = 0) -> list[dict]:
    sizes = ivf_frame_sizes(stream)
    frames = [dict(index=i, **summarise_av1_frame(f, sizes[i]))
              for i, f in enumerate(inspect_frames(stream))]
    if source is not None:
        for f, (y, u, v) in zip(frames, per_frame_psnr(stream, source, width, height,
                                                       len(frames))):
            f.update(psnr_y=y, psnr_u=u, psnr_v=v)
    return frames


def summarise_clip(frames: list[dict], kind: str = "P") -> dict:
    """Mean over frames of one type; bits are per frame, shares are averaged."""
    sel = [f for f in frames if f["type"] == kind]
    if not sel:
        return {}

    def mean_of(get) -> dict[str, float]:
        acc: dict[str, float] = defaultdict(float)
        for f in sel:
            for k, v in get(f).items():
                acc[k] += v / len(sel)
        return dict(sorted(acc.items()))

    summary = {
        "frames": len(sel),
        "bytes": sum(f["bytes"] for f in sel) / len(sel),
        "bits": mean_of(lambda f: f["bits"]),
        "area_by_size": mean_of(lambda f: f["area"]["by_size"]),
        "area_by_mode": mean_of(lambda f: f["area"]["by_mode"]),
        "skip": sum(f["area"]["skip"] for f in sel) / len(sel),
        "tx_area": mean_of(lambda f: f["tx_area"]),
        "mv": mean_of(lambda f: f["mv"]),
        "q": sum(f["q"] for f in sel) / len(sel),
    }
    if "psnr_y" in sel[0]:
        summary["psnr"] = {p: sum(f[f"psnr_{p}"] for f in sel) / len(sel) for p in "yuv"}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stream", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write per-frame stats here")
    args = parser.parse_args(argv)
    frames = analyse(args.stream, args.source, args.width, args.height)
    if args.json:
        args.json.write_text(json.dumps(frames))
    print(json.dumps({"I": summarise_clip(frames, "I"), "P": summarise_clip(frames, "P")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
