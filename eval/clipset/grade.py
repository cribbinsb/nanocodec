#!/usr/bin/env python3
"""Grade every clip's source quality for the screen pool, and draw contact sheets.

    python3 clipset/grade.py main broad_independent_v1 layergate_fit layergate_fit2 long pool

Our encoder sees clean camera frames on the device; a clip cut from a badly
compressed source mostly tests transcoding of its artefacts (owner,
2026-11-11). So each clip's SOURCE is measured at its own size, on a few
frames spread over the clip:

  source_bpp_native  bits per pixel per frame of the source (inventory
                     record; a clip with none is its own source)
  source_bpp         the same bits over the clip's stored pixel rate
  blockiness         mean luma step across 8-pixel block edges over the
                     mean step elsewhere (1.0: no block structure; DCT
                     codecs at low rates show their grid, ~1.3 and up)
  noise_sigma        Immerkaer's fast noise estimate (luma levels)
  source_resolution  the source's WxH; downscale = source / clip height
  dataset            where the source comes from (draws spread datasets)

and graded:

  A clean     bpp >= A_MIN_BPP and blockiness <= A_MAX_BLOCKINESS:
              usable at native size;
  B mediocre  bpp >= C_MAX_BPP and blockiness <= C_MIN_BLOCKINESS:
              usable only downscaled 2-3x (nceval.pool), which removes
              most of its artefacts;
  C poor      anything else: never used.

Lossless (pristine) sources are A. The fields are written into each
<set>.json (nothing else changes; nceval.clips reads them and nceval.pool
applies the grade). One PNG crop per clip the pool accepts goes to
~/nanocodec-data/work/pool-sheets/<class>/, with one contact sheet per class
(sheet-<class>.png), outside the repo, for eyeballing the pool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DATA_ROOT = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
DEFAULT_INVENTORY = DEFAULT_DATA_ROOT / "inventory" / "inventory.json"
SHEETS = DEFAULT_DATA_ROOT / "work" / "pool-sheets"
SAMPLE_FRAMES = 4
BLOCK = 8
A_MIN_BPP = 0.15
A_MAX_BLOCKINESS = 1.15
C_MAX_BPP = 0.03  # below this the source is poor at any size
C_MIN_BLOCKINESS = 1.5
MIN_POOR_DOWNSCALE = 2.0  # as nceval.pool
PRISTINE_MARKERS = ("/clips/pristine/", "/clips/xiph/", "/clips/svt/")
CROP = (384, 216)
SHEET_COLUMNS = 8
PARALLEL = 6
DIGITS = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()
    inventory = {r["path"]: r for r in json.loads(args.inventory.read_text())["clips"]}
    crops: dict[str, list[Path]] = {}
    for name in args.sets:
        path = args.data_root / "clips" / f"{name}.json"
        clipset = json.loads(path.read_text())
        entries = [e for e in clipset["clips"] if not e.get("clean_reference")]
        with ThreadPoolExecutor(PARALLEL) as pool:
            grades = list(pool.map(lambda e: grade(e, inventory, args.data_root), entries))
        for entry, fields in zip(entries, grades):
            entry.update(fields)
            if accepted(entry):
                crop = crop_png(entry, args.data_root)
                if crop:
                    crops.setdefault(entry["content_class"], []).append(crop)
        path.write_text(json.dumps(clipset, indent=1) + "\n")
        counts = {g: sum(e.get("grade") == g for e in entries) for g in "ABC"}
        print(f"{path}: {counts}", flush=True)
    for cls, paths in crops.items():
        sheet(cls, paths)
    return 0


def grade(entry: dict, inventory: dict, data_root: Path) -> dict:
    stored = Path(entry["yuv"]) if Path(entry["yuv"]).is_absolute() else data_root / entry["yuv"]
    source = inventory.get(entry.get("source"))
    pristine = any(m in str(entry.get("source", "")) for m in PRISTINE_MARKERS) and \
        entry.get("form") != "camera"
    pixels = entry["width"] * entry["height"]
    if source and source.get("source_kbps") and source.get("fps"):
        bits = source["source_kbps"] * 1000
        native = bits / (source["width"] * source["height"] * source["fps"])
        effective = bits / (pixels * min(float(entry["fps"]), source["fps"]))
        width, height = source["width"], source["height"]
        frames = sample_frames(Path(entry["source"]), source["width"], source["height"],
                               float(source.get("duration_s") or 0) or None,
                               start=float(entry.get("start", 0.0)))
    else:  # its own source: a pristine excerpt or a camera-like encode of one
        native = effective = entry["source_kbps"] * 1000 / (pixels * float(entry["fps"]))
        width, height = entry["width"], entry["height"]
        frames = sample_stored(stored, entry)
    block = float(np.mean([blockiness(f) for f in frames])) if frames else None
    noise = float(np.mean([noise_sigma(f) for f in frames])) if frames else None
    if pristine:
        letter = "A"
    elif block is None:
        letter = "B" if native >= C_MAX_BPP else "C"
    elif native >= A_MIN_BPP and block <= A_MAX_BLOCKINESS:
        letter = "A"
    elif native >= C_MAX_BPP and block <= C_MIN_BLOCKINESS:
        letter = "B"
    else:
        letter = "C"
    rounded = lambda v: None if v is None else round(v, DIGITS)  # noqa: E731
    return {"source_bpp_native": rounded(native), "source_bpp": rounded(effective),
            "blockiness": rounded(block), "noise_sigma": rounded(noise),
            "source_resolution": f"{width}x{height}",
            "downscale": round(min(width, height) / min(entry["width"], entry["height"]), 3),
            "grade": letter, "dataset": dataset(entry)}


def accepted(entry: dict) -> bool:
    """The grade rule as nceval.pool applies it (A anywhere, B downscaled, never C)."""
    return entry["grade"] == "A" or (entry["grade"] == "B" and
                                     entry["downscale"] >= MIN_POOR_DOWNSCALE)


def dataset(entry: dict) -> str:
    source = str(entry.get("source") or entry.get("origin") or "")
    for marker in ("/tracking_original/", "/downloaded_datasets/other/", "/clips/"):
        if marker in source:
            return source.split(marker, 1)[1].split("/", 1)[0].lower()
    if source.startswith("/mldata/video/"):
        rest = source.removeprefix("/mldata/video/")
        return "video/" + rest.split("/", 1)[0] if "/" in rest else "video"
    return "other"


def sample_frames(path: Path, width: int, height: int, duration: float | None,
                  start: float) -> list[np.ndarray]:
    """SAMPLE_FRAMES luma frames of the source around the clip's cut."""
    span = min(duration - start, 20.0) if duration else 5.0
    frames = []
    for index in range(SAMPLE_FRAMES):
        t = start + span * (index + 0.5) / SAMPLE_FRAMES
        frame = decode_luma(["-ss", f"{t:.3f}", "-i", str(path)], width, height)
        if frame is not None:
            frames.append(frame)
    return frames


def sample_stored(path: Path, entry: dict) -> list[np.ndarray]:
    width, height, count = entry["width"], entry["height"], int(entry["frames"])
    frames = []
    for index in range(SAMPLE_FRAMES):
        number = count * (2 * index + 1) // (2 * SAMPLE_FRAMES)
        if path.suffix == ".yuv":
            with path.open("rb") as f:
                f.seek(number * width * height * 3 // 2)
                data = f.read(width * height)
            if len(data) == width * height:
                frames.append(np.frombuffer(data, np.uint8).reshape(height, width))
        else:
            frame = decode_luma(["-ss", f"{number / float(entry['fps']):.3f}", "-i", str(path)],
                                width, height)
            if frame is not None:
                frames.append(frame)
    return frames


def decode_luma(inputs: list[str], width: int, height: int) -> np.ndarray | None:
    """One frame's luma at the source's coded size (no scaling: block edges stay on grid)."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-noautorotate", *inputs,
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
                         capture_output=True).stdout
    if len(out) != width * height:
        return None
    return np.frombuffer(out, np.uint8).reshape(height, width)


def blockiness(luma: np.ndarray) -> float:
    """Mean |step| across 8-pixel block edges over the mean |step| elsewhere."""
    y = luma.astype(np.int16)
    ratios = []
    for steps in (np.abs(np.diff(y, axis=1)), np.abs(np.diff(y, axis=0)).T):
        edge = np.zeros(steps.shape[1], bool)
        edge[BLOCK - 1::BLOCK] = True
        inner = steps[:, ~edge].mean()
        ratios.append(steps[:, edge].mean() / inner if inner > 0 else 1.0)
    return float(np.mean(ratios))


def noise_sigma(luma: np.ndarray) -> float:
    """Immerkaer (1996): sigma from the mean |Laplacian difference| response."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = np.abs(cv2.filter2D(luma.astype(np.float32), -1, kernel))[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) * response.mean() / 6)


def crop_png(entry: dict, data_root: Path) -> Path | None:
    """A centre crop of the stored clip's middle frame, at stored size."""
    stored = Path(entry["yuv"]) if Path(entry["yuv"]).is_absolute() else data_root / entry["yuv"]
    w, h = entry["width"], entry["height"]
    middle = int(entry["frames"]) // 2
    if stored.suffix == ".yuv":
        cmd = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-i", str(stored),
               "-vf", f"select=eq(n\\,{middle})"]
    else:
        cmd = ["-ss", f"{middle / float(entry['fps']):.3f}", "-i", str(stored)]
    cw, ch = min(CROP[0], w), min(CROP[1], h)
    out = SHEETS / entry["content_class"] / f"{entry['name']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    label = f"{entry['name']} {entry['grade']} {entry['source_bpp_native']:.2f}bpp".replace(
        ":", " ")
    vf = (f"crop={cw}:{ch},pad={CROP[0]}:{CROP[1]}:0:0,"
          f"drawtext=text='{label}':fontcolor=yellow:fontsize=12:x=3:y=3")
    if "-vf" in cmd:
        cmd[cmd.index("-vf") + 1] += "," + vf
    else:
        cmd += ["-vf", vf]
    done = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *cmd, "-frames:v", "1",
                           str(out)], capture_output=True)
    return out if done.returncode == 0 and out.exists() else None


def sheet(cls: str, crops: list[Path]) -> None:
    tiles = [cv2.imread(str(p)) for p in sorted(crops)]
    tiles = [t for t in tiles if t is not None]
    rows = []
    for start in range(0, len(tiles), SHEET_COLUMNS):
        row = tiles[start:start + SHEET_COLUMNS]
        row += [np.zeros_like(tiles[0])] * (SHEET_COLUMNS - len(row))
        rows.append(np.hstack(row))
    if rows:
        cv2.imwrite(str(SHEETS / f"sheet-{cls}.png"), np.vstack(rows))
        print(f"sheet {cls}: {len(tiles)} clips -> {SHEETS / f'sheet-{cls}.png'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
