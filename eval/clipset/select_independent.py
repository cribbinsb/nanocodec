#!/usr/bin/env python3
"""Select never-used sources for the independent test (a build_pool.py selection).

    python3 clipset/select_independent.py --out clipset/independent.json [--count 120]
    python3 clipset/build_pool.py clipset/independent.json   # -> clips/independent.json
    python3 clipset/grade.py independent                     # grades A/B/C; nceval.pool rules

The held-out chart check (tools/dev/aws_heldout_chart.py `select`) draws from
the screening pool, which adoption screens and tuning studies have used since
2026-11-11. This draws only sources that NO clip set has ever used: every
inventory source (eval/inventory, /mldata) whose family key (nceval.pool.
family_key) and path appear in no clip set under the data root's clips/ and in
no selection file under eval/ (TRAIN, VAL, TEST, adv12, the chart's clips, the
pool, the layer-gate fits, long, quick, smoke, ...).

Rules, all fixed here so the draw is reproducible and nobody picks clips:

  - a camera recording: no derived composite (video/long_composite), no
    edited or re-uploaded content (inventory tags), no hard cut in the
    inventory's sample, a known camera class (auto_class);
  - at least START_S + the excerpt's length, >= MIN_SOURCE_FPS;
  - a source the pool's grading could accept (nceval.pool.BPP_FLOOR): stored
    at native size (<= 720 lines) when its own bits per pixel reach the floor,
    else downscaled 2x when that lifts it over the floor (grade B's rule);
    grade.py then grades the stored clips and the pool's rule decides;
  - one source per family, round robin over the classes and, within a class,
    over its datasets (largest first), each in a fixed SHA-256 order of the
    path seeded with SEED.

The excerpt is long enough for MEASURED_FRAMES frames at the stored rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval.pool import BPP_FLOOR, MIN_POOR_DOWNSCALE, family_key  # noqa: E402

DATA_ROOT = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
EVAL = Path(__file__).resolve().parents[1]
SEED = "independent-2026-11-13"
#: Inventory camera classes -> clip content classes (nceval.pool.STRATA).
CLASSES = {"static": "static", "crowd-busy": "crowd", "handheld-bodycam": "bodycam",
           "pan": "pan", "low-light": "lowlight", "drone-aerial": "drone"}
EXCLUDED_GROUPS = ("video/long_composite",)
EXCLUDED_TAGS = ("derived", "edited", "web-upload")
START_S = 2.0
MEASURED_FRAMES = 150
#: Frames past MEASURED_FRAMES the excerpt keeps (a clean cut at the end).
SPARE_FRAMES = 15
MIN_SOURCE_FPS = 10.0
MAX_FPS = 30.0  # build_pool.MAX_FPS: faster sources are halved
MAX_HEIGHT = 720
#: The poorest source the pool's grading accepts at any size (grade.C_MAX_BPP).
C_MAX_BPP = 0.03
_MLDATA = re.compile(r"/mldata/[^\"']+")


def used_sources(own_name: str) -> tuple[set[str], set[str]]:
    """Every source path and family key any clip set or selection file names
    (but this selection's own, `own_name`, so a rerun reproduces it)."""
    paths: set[str] = set()
    files = [*(DATA_ROOT / "clips").glob("*.json"), *EVAL.rglob("*.json")]
    for path in (f for f in files if f.stem != own_name):
        text = path.read_text(errors="replace")
        paths |= set(_MLDATA.findall(text))
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        for clip in data.get("clips", []) if isinstance(data, dict) else []:
            if isinstance(clip, dict):
                for key in ("family", "source", "name"):
                    if clip.get(key):
                        paths.add(str(clip[key]))
    return paths, {family_key(p) for p in paths}


def stored_height(record: dict) -> int | None:
    """The stored clip's height the pool's grading could accept, or None."""
    fps = record["fps"]
    bpp = record["source_kbps"] * 1000 / (record["width"] * record["height"] * fps)
    native = min(record["height"], MAX_HEIGHT)
    native_bpp = bpp * (record["height"] / native) ** 2
    if record["height"] <= MAX_HEIGHT and native_bpp >= BPP_FLOOR:
        return native
    half = min(record["height"] / MIN_POOR_DOWNSCALE, MAX_HEIGHT)
    if bpp >= C_MAX_BPP and bpp * MIN_POOR_DOWNSCALE ** 2 >= BPP_FLOOR:
        return int(half) - int(half) % 2
    return None


def excerpt_seconds(fps: float) -> float:
    stored_fps = fps / 2 if fps > MAX_FPS else fps
    return math.ceil((MEASURED_FRAMES + SPARE_FRAMES) / stored_fps * 10) / 10


def candidate(record: dict, used_paths: set[str], used_families: set[str]) -> dict | None:
    if record.get("error") or record["path"] in used_paths or \
            family_key(record["path"]) in used_families:
        return None
    stats = record.get("stats") or {}
    if record.get("source_group", "").startswith(EXCLUDED_GROUPS) or \
            set(record.get("tags", [])) & set(EXCLUDED_TAGS) or \
            stats.get("scene_cuts", 1) != 0 or record.get("auto_class") not in CLASSES:
        return None
    fps = record.get("fps") or 0.0
    if fps < MIN_SOURCE_FPS or not record.get("source_kbps"):
        return None
    seconds = excerpt_seconds(fps)
    height = stored_height(record)
    if height is None or (record.get("duration_s") or 0) < START_S + seconds:
        return None
    dataset = record.get("source_group", "").split("/")[-1] or "other"
    name = re.sub(r"[^a-z0-9]+", "_", f"ind_{dataset}_{Path(record['path']).stem}".lower())
    entry = {"name": name.strip("_"), "source": record["path"], "start": START_S,
             "seconds": seconds, "content_class": CLASSES[record["auto_class"]]}
    if height != min(record["height"], MAX_HEIGHT):
        entry["height"] = height
    return {"entry": entry, "dataset": dataset, "family": family_key(record["path"])}


def spread(candidates: list[dict], count: int) -> list[dict]:
    """One per family, round robin over classes, then datasets (largest first)."""
    rank = lambda c: hashlib.sha256(f"{SEED}/{c['entry']['source']}".encode()).hexdigest()  # noqa
    queues: dict[str, dict[str, list[dict]]] = {}
    for c in sorted(candidates, key=rank):
        queues.setdefault(c["entry"]["content_class"], {}).setdefault(c["dataset"], []).append(c)
    turn = {cls: 0 for cls in queues}
    chosen, families = [], set()
    while len(chosen) < count and any(any(q.values()) for q in queues.values()):
        for cls in sorted(queues):
            datasets = sorted((d for d, q in queues[cls].items() if q),
                              key=lambda d: (-len(queues[cls][d]), d))
            while datasets:
                dataset = datasets[turn[cls] % len(datasets)]
                turn[cls] += 1
                queue = queues[cls][dataset]
                while queue and queue[0]["family"] in families:
                    queue.pop(0)
                if queue:
                    pick = queue.pop(0)
                    chosen.append(pick)
                    families.add(pick["family"])
                    break
                datasets = [d for d in datasets if queues[cls][d]]
            if len(chosen) == count:
                break
    return sorted(chosen, key=lambda c: c["entry"]["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--inventory", type=Path,
                        default=DATA_ROOT / "inventory" / "inventory.json")
    args = parser.parse_args()
    used_paths, used_families = used_sources(args.out.stem)
    records = json.loads(args.inventory.read_text())["clips"]
    candidates = [c for r in records if (c := candidate(r, used_paths, used_families))]
    chosen = spread(candidates, args.count)
    by_class: dict[str, int] = {}
    for c in chosen:
        by_class[c["entry"]["content_class"]] = by_class.get(c["entry"]["content_class"], 0) + 1
    name = args.out.stem
    args.out.write_text(json.dumps({
        "name": name,
        "_comment": f"Never-used sources for the independent test: written by "
                    f"eval/clipset/select_independent.py (seed {SEED}), "
                    f"{len(chosen)} of {len(candidates)} candidates.",
        "clips": [c["entry"] for c in chosen]}, indent=1) + "\n")
    print(f"{len(chosen)} of {len(candidates)} candidates ({len(records)} inventory sources, "
          f"{len(used_families)} used family keys), "
          f"{len({c['dataset'] for c in chosen})} datasets, by class {by_class}; wrote {args.out}")


if __name__ == "__main__":
    main()
