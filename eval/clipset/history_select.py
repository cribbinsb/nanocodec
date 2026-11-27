#!/usr/bin/env python3
"""BP-3's history subset (docs/benchmark-procedure.md section 14): 10 clips from each of ours
(heldout_2026-11-23b), VC-1K and AOM-CTC, by greedy maximum coverage over fixed strata.

    python3 eval/clipset/history_select.py features   # -> HISTORY_FEATURES (decodes clips)
    python3 eval/clipset/history_select.py select     # -> HISTORY_SET (prints the coverage)

`features` measures, on the raw benchmark clips (`<name>.yuv.zst`), every FEATURE_STEP-th
frame's luma: SI and TI (ITU-T P.910: the maximum over the sampled frames of the standard
deviation of the Sobel magnitude, and of the difference from the previous frame), mean luma,
and the grade's noise estimate (`grade.noise_sigma`). Only VC-1K and AOM-CTC are measured:
heldout_2026-11-23b is sealed (release-measure only), so its content is not analysed; its noise
comes from the grade already recorded in its clip set, and its SI, TI and mean luma are
unknown (no cell).

`select` applies section 14's rule, written here before any history measurement:
  - strata: content class (VC-1K: its motion/lighting/environment stratum), resolution band, frame-rate band, SI and TI tertiles (over the
    measured clips), noise flag (noise_sigma in the top quarter of the three sets), low-light
    flag (mean luma under LOW_LIGHT_LUMA), source dataset. A clip covers one cell per stratum
    whose value is known;
  - the rate-behaviour stratum (the spread of per-clip BD-J across the encoders in the 0.6.0
    chart data) is NOT applied: the published 0.6.0 data holds no per-clip BD-J for these
    sets (`web/results/data/chart.json` is per family), so it would need a new computation
    over the sealed set's points. Stated in HISTORY_SET;
  - each pick takes, over the datasets still short of PER_DATASET, the clip covering the most
    still-uncovered cells, ties broken by SHA-256 of SEED + name;
  - at most MAX_PER_CLASS clips of one class and MAX_PER_SOURCE of one source within a dataset;
  - the coverage table compares each cell's share in the subset with the full sets; a cell
    with a full-set share of COVERAGE_SHARE or more left empty refuses the subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval" / "clipset"))

from grade import noise_sigma  # noqa: E402

DATA = Path.home() / "nanocodec-data"
CLIPSET = REPO / "eval" / "clipset"
HISTORY_FEATURES = CLIPSET / "history_features.json"
HISTORY_SET = CLIPSET / "history-set.json"
#: dataset -> (manifest, raw clip directory, measured here)
DATASETS = {
    "ours": (CLIPSET / "heldout_2026-11-23b.manifest.json", None, False),
    "vc1k": (CLIPSET / "vc1k_2026-11-18.manifest.json", DATA / "clips/bench_raw/vc1k", True),
    "aomctc": (CLIPSET / "aomctc_bp3.manifest.json", DATA / "clips/aomctc_bp3", True),
}
#: The grade's noise estimate, recorded in our set's clip-set file.
OURS_GRADES = DATA / "clips" / "heldout_2026-11-23b.json"
FEATURE_STEP = 5
SEED = "history-2026-11-23"
PER_DATASET = 10
MAX_PER_CLASS = 3
MAX_PER_SOURCE = 2
COVERAGE_SHARE = 0.10
LOW_LIGHT_LUMA = 60
NOISY_QUANTILE = 0.75
TERTILES = (1 / 3, 2 / 3)
#: Resolution bands by the picture's shorter side (lines); the first band whose floor it
#: reaches.
RESOLUTION_BANDS = ((900, "1080p"), (630, "720p"), (450, "540p"), (315, "360p"),
                    (0, "270p/240p"))


def fps_value(fps: str | float) -> float:
    if "/" in str(fps):
        num, den = map(int, str(fps).split("/"))
        return num / den
    return float(fps)


def fps_band(fps: float) -> str:
    if fps <= 15.5:
        return "<=15"
    if 23.9 <= fps <= 25.1:
        return "24-25"
    if 29.9 <= fps <= 30.1:
        return "29.97-30"
    if fps >= 49.9:
        return ">=50"
    return "other"


def resolution_band(width: int, height: int) -> str:
    short = min(width, height)
    return next(name for floor, name in RESOLUTION_BANDS if short >= floor)


def luma_frames(path: Path, width: int, height: int, frames: int):
    """(index, luma) of every frame of a raw I420 clip, streamed."""
    size = width * height * 3 // 2
    with subprocess.Popen(["zstd", "-dcq", str(path)], stdout=subprocess.PIPE) as raw:
        for index in range(frames):
            frame = raw.stdout.read(size)
            if len(frame) != size:
                raise SystemExit(f"refused: {path} holds fewer than {frames} frames")
            yield index, np.frombuffer(frame, np.uint8, width * height).reshape(height, width)


def measure(job: tuple[str, Path, int, int, int]) -> tuple[str, dict]:
    name, path, width, height, frames = job
    si, ti, luma, noise, previous = [], [], [], [], None
    for index, y in luma_frames(path, width, height, frames):
        if index % FEATURE_STEP == 0 and previous is not None:
            f = y.astype(np.float32)
            gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
            si.append(float(np.sqrt(gx * gx + gy * gy)[1:-1, 1:-1].std()))
            ti.append(float((f - previous).std()))
            luma.append(float(f.mean()))
            noise.append(noise_sigma(y))
        previous = y.astype(np.float32)
    return name, {"si": round(max(si), 2), "ti": round(max(ti), 2),
                  "mean_luma": round(float(np.mean(luma)), 2),
                  "noise_sigma": round(float(np.mean(noise)), 4)}


def features(jobs: int) -> None:
    work = []
    for dataset, (manifest, raw, measured) in DATASETS.items():
        if not measured:
            continue
        for c in json.loads(manifest.read_text())["clips"]:
            work.append((c["name"], raw / f"{c['name']}.yuv.zst", c["width"], c["height"],
                         c["frames"]))
    with ProcessPoolExecutor(jobs) as pool:
        out = dict(pool.map(measure, work))
    HISTORY_FEATURES.write_text(json.dumps(
        {"_comment": f"eval/clipset/history_select.py features: every {FEATURE_STEP}th frame's "
                     "luma (with the frame before it for TI); VC-1K and AOM-CTC only.",
         "clips": dict(sorted(out.items()))}, indent=1) + "\n")
    print(f"features of {len(out)} clips -> {HISTORY_FEATURES}")


def tertile(value: float | None, cuts: list[float]) -> str | None:
    if value is None:
        return None
    return ("low", "mid", "high")[sum(value > c for c in cuts)]


def load_clips() -> list[dict]:
    measured = json.loads(HISTORY_FEATURES.read_text())["clips"]
    grades = {c["name"]: c for c in json.loads(OURS_GRADES.read_text())["clips"]}
    clips = []
    for dataset, (manifest, _, _) in DATASETS.items():
        for c in json.loads(manifest.read_text())["clips"]:
            feats = measured.get(c["name"], {})
            source = (c.get("family") if dataset != "aomctc" else c["licence"])
            noise = feats.get("noise_sigma", grades.get(c["name"], {}).get("noise_sigma"))
            # VC-1K's clips are all class "videocall"; its own strata are its classes.
            cls = c["stratum"] if dataset == "vc1k" else c["content_class"]
            clips.append({"name": c["name"], "dataset": dataset, "class": cls,
                          "source": source, "width": c["width"], "height": c["height"],
                          "fps": fps_value(c["fps"]), "si": feats.get("si"),
                          "ti": feats.get("ti"), "mean_luma": feats.get("mean_luma"),
                          "noise_sigma": noise})
    return clips


def strata(clips: list[dict]) -> None:
    """Each clip's cell per stratum (None: unknown, no cell)."""
    def cuts(key: str) -> list[float]:
        values = [c[key] for c in clips if c[key] is not None]
        return [float(np.quantile(values, q)) for q in TERTILES]

    si_cuts, ti_cuts = cuts("si"), cuts("ti")
    noisy = float(np.quantile([c["noise_sigma"] for c in clips if c["noise_sigma"] is not None],
                              NOISY_QUANTILE))
    for c in clips:
        c["cells"] = {
            "class": f"{c['dataset']}:{c['class']}",
            "resolution": resolution_band(c["width"], c["height"]),
            "fps": fps_band(c["fps"]),
            "si": tertile(c["si"], si_cuts),
            "ti": tertile(c["ti"], ti_cuts),
            "noisy": None if c["noise_sigma"] is None else c["noise_sigma"] >= noisy,
            "low_light": None if c["mean_luma"] is None else c["mean_luma"] < LOW_LIGHT_LUMA,
            "dataset": c["dataset"],
        }


def cells(clip: dict) -> set[tuple[str, object]]:
    return {(k, v) for k, v in clip["cells"].items() if v is not None}


def tie_key(name: str) -> str:
    return hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()


def greedy(clips: list[dict]) -> list[dict]:
    picked: list[dict] = []
    covered: set = set()
    while len(picked) < PER_DATASET * len(DATASETS):
        def allowed(c: dict) -> bool:
            mine = [p for p in picked if p["dataset"] == c["dataset"]]
            return (c not in picked and len(mine) < PER_DATASET
                    and sum(p["class"] == c["class"] for p in mine) < MAX_PER_CLASS
                    and sum(p["source"] == c["source"] for p in mine) < MAX_PER_SOURCE)

        candidates = [c for c in clips if allowed(c)]
        if not candidates:
            raise SystemExit("refused: the class and source caps leave no candidate")
        best = min(candidates, key=lambda c: (-len(cells(c) - covered), tie_key(c["name"])))
        picked.append(best)
        covered |= cells(best)
    return picked


def coverage(clips: list[dict], picked: list[dict]) -> tuple[list[dict], list[str]]:
    full = Counter(cell for c in clips for cell in cells(c))
    subset = Counter(cell for c in picked for cell in cells(c))
    rows, empty = [], []
    for (stratum, value), count in sorted(full.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        known_full = sum(1 for c in clips if c["cells"][stratum] is not None)
        known_sub = sum(1 for c in picked if c["cells"][stratum] is not None)
        share = count / known_full
        rows.append({"stratum": stratum, "cell": str(value), "full": count,
                     "full_share": round(share, 3), "subset": subset[(stratum, value)],
                     "subset_share": round(subset[(stratum, value)] / known_sub, 3)})
        if share >= COVERAGE_SHARE and not subset[(stratum, value)]:
            empty.append(f"{stratum}={value}")
    return rows, empty


def select() -> None:
    clips = load_clips()
    strata(clips)
    picked = greedy(clips)
    rows, empty = coverage(clips, picked)
    print(f"{'stratum':10} {'cell':22} {'full':>5} {'share':>6} {'subset':>6} {'share':>6}")
    for r in rows:
        print(f"{r['stratum']:10} {r['cell']:22} {r['full']:5} {r['full_share']:6.3f} "
              f"{r['subset']:6} {r['subset_share']:6.3f}")
    if empty:
        raise SystemExit(f"refused: strata with a full-set share >= {COVERAGE_SHARE} left "
                         f"empty: {empty}")
    HISTORY_SET.write_text(json.dumps({
        "_comment": ("BP-3 history subset (docs/benchmark-procedure.md section 14), drawn once "
                     "by eval/clipset/history_select.py select. Fixed: never redrawn."),
        "seed": SEED,
        "rule": {"per_dataset": PER_DATASET, "max_per_class": MAX_PER_CLASS,
                 "max_per_source": MAX_PER_SOURCE, "coverage_share": COVERAGE_SHARE,
                 "feature_step": FEATURE_STEP, "low_light_luma": LOW_LIGHT_LUMA,
                 "noisy_quantile": NOISY_QUANTILE},
        "not_applied": {
            "rate_behaviour": "no per-clip BD-J in the published 0.6.0 chart data",
            "ours_si_ti_luma": "heldout_2026-11-23b is sealed: its content is not analysed"},
        "clips": [{"name": c["name"], "dataset": c["dataset"], "class": c["class"],
                   "cells": {k: v for k, v in c["cells"].items() if v is not None}}
                  for c in picked],
        "coverage": rows}, indent=1) + "\n")
    print(f"{len(picked)} clips -> {HISTORY_SET}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("features", "select"))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    features(a.jobs) if a.command == "features" else select()
    return 0


if __name__ == "__main__":
    sys.exit(main())
