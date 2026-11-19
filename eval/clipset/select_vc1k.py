#!/usr/bin/env python3
"""Select VC-1K phone video-call clips, stratified (build_pool.py selections): the
held-out measurement set, and a disjoint development set for the screening pool.

    python3 clipset/select_vc1k.py --out clipset/vc1k.json [--count 100]
    python3 clipset/build_pool.py clipset/vc1k.json \\
        --inventory ~/nanocodec-data/inventory/vc1k.json   # -> clips/vc1k_2026-11-18.json
    python3 clipset/select_vc1k.py --dev --out clipset/vc1k_dev.json \\
        --exclude clipset/vc1k_dev_excluded.json
    python3 clipset/build_pool.py clipset/vc1k_dev.json \\
        --inventory ~/nanocodec-data/inventory/vc1k_dev.json   # -> clips/vc1k_dev.json
    python3 clipset/grade.py vc1k_dev
    # then add every non-A clip to vc1k_dev_excluded.json and repeat until all are A

VC-1K (github.com/facebookresearch/VC-1K, CC BY-NC 4.0) is ~15 s phone
recordings in the phones' own H.264, 720p (mostly portrait 720x1280), 24-30
fps. Its file names carry the collection's labels:
`<device>_<environment>_<motion>motion_<lighting>light_<W>x<H>_<fps>_<id>`.
Only the 930 even-frame-rate clips (the dataset's all_even_fps.txt) are used;
video_with_jitter_fps is not.

Licence: non-commercial, and most clips show participants' faces. The results
site publishes numbers and clip names only, never frames, crops or streams of
these clips, and credits the dataset (nceval.publicnames).

Rules, all fixed here so the draw is reproducible and nobody picks clips:

  - strata: motion (slow, medium, fast) x lighting (dark, low, medium, good)
    x environment (indoor, outdoor), 24 in all;
  - each stratum filled evenly: a stratum smaller than the even share gives
    all its clips, the rest share what is left (water filling); the
    remainder after the even share goes to strata in a fixed SHA-256 order;
  - within a stratum, device (Android, iOS) x orientation (portrait,
    landscape) groups are taken in turn, in a fixed order, each group in a
    fixed SHA-256 order of the file name seeded with SEED;
  - names in --exclude (a JSON {name: reason}) are skipped: clips the build
    dropped or cut below MEASURED_FRAMES (a scene cut), each with its reason.

The selection also writes the Fixed QP sample (FIXED_QP_SAMPLE clips, one per
motion x lighting cell, the environment alternating; the first pick of the
stratum) as `fixed_qp_sample`: build_pool.py carries it into the clip set, and
`--qp-clipset` writes it as a clip set of its own for `nceval.fixedqp gap
--clipset`.

`--dev` (owner, 2026-11-18: our development clips had no phone video-call
content) draws DEV_COUNT clips by the same rules with its own seed from the
clips the measurement selection (MEASUREMENT_SELECTION, its exclusions too)
does not hold, so the measurement set stays held out; nceval.pool also seals
it by family. Each dev clip runs from START_S to the end of the recording
(room for the pool's start-frame rotation), and every DEV_VAL_EVERY-th clip
in stratum order is VAL, the rest TRAIN (main's TRAIN:VAL is about 2:1). A
dev clip whose source grade the pool would not use at its native size
(grade.py B or C) goes into the exclusions with its grade and is replaced by
its stratum's next clip, until all are grade A (coordinator, 2026-11-18: no
downscaled forms; a stratum with no A clips left gives its share away). The
dev set is internal only: nceval.site.DEV_ONLY_FILES keeps it off the public
mirror, and no site view reads it.

Each clip is cut from START_S, long enough for MEASURED_FRAMES plus spare
frames, at its native displayed size (rotation applied), and stored like the
held-out sets (build_pool.py camera form). Its source rate is the original
file's video bit rate (the inventory stub this writes), so the
<= 50 %-of-source rule caps the ladder on each clip's own phone encode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA_ROOT = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
SOURCE_DIR = Path("/mldata/downloaded_datasets/other/vc1k")
EVEN_FPS_LIST = "all_even_fps.txt"
NAME = "vc1k_2026-11-18"
SEED = "vc1k-2026-11-18"
DEFAULT_COUNT = 100
DEV_NAME = "vc1k_dev"
DEV_SEED = "vc1k-dev-2026-11-18"
DEV_COUNT = 30
DEV_VAL_EVERY = 3
MEASUREMENT_SELECTION = Path(__file__).resolve().parent / "vc1k.json"
#: The site's dataset name and this set's one content class (nceval.clips.CONTENT_CLASSES):
#: the strata below, not the pool's camera classes, describe the content.
DATASET = "vc1k"
CONTENT_CLASS = "videocall"
START_S = 2.0
MEASURED_FRAMES = 150
SPARE_FRAMES = 15  # as select_independent.py: a clean cut past the measured frames
FIXED_QP_SAMPLE = 12
PROBE_WORKERS = 8
MOTIONS = ("slow", "medium", "fast")
LIGHTS = ("dark", "low", "medium", "good")
ENVIRONMENTS = ("indoor", "outdoor")
_NAME = re.compile(r"^(android|ios)_(indoor|outdoor)_(slow|medium|fast)motion_"
                   r"(dark|low|medium|good)light_(\d+)x(\d+)_(\d+)_(\d+)\.\w+$", re.IGNORECASE)


def labels(file: str) -> dict:
    match = _NAME.match(file)
    if not match:
        raise ValueError(f"{file}: not a VC-1K clip name")
    device, environment, motion, light, width, height, _fps, _id = match.groups()
    return {"device": device.lower(), "environment": environment, "motion": motion,
            "lighting": light,
            "orientation": "portrait" if int(height) > int(width) else "landscape"}


def stratum(entry: dict) -> str:
    return f"{entry['motion']}/{entry['lighting']}/{entry['environment']}"


def hashed(text: str, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def quotas(sizes: dict[str, int], count: int, seed: str = SEED) -> dict[str, int]:
    """Water filling: the largest even share every stratum can give (a small stratum
    gives all it has), the remainder one each to the strata that still have clips,
    in a fixed hash order."""
    share = 0
    while sum(min(n, share + 1) for n in sizes.values()) <= count:
        share += 1
        if share > max(sizes.values()):
            break
    quota = {s: min(n, share) for s, n in sizes.items()}
    spare = sorted((s for s, n in sizes.items() if n > share), key=lambda s: hashed(s, seed))
    for s in spare[:count - sum(quota.values())]:
        quota[s] += 1
    return quota


def interleaved(entries: list[dict], seed: str = SEED) -> list[dict]:
    """A stratum's clips, device x orientation groups in turn (in a fixed hash order, so
    neither the common portrait Android clips nor the rare ones always come first)."""
    groups: dict[tuple, list[dict]] = {}
    for e in sorted(entries, key=lambda e: hashed(e["file"], seed)):
        groups.setdefault((e["device"], e["orientation"]), []).append(e)
    queues = sorted(groups.values(), key=lambda g: hashed("/".join(g[0][k] for k in (
        "device", "orientation")), seed))
    out = []
    while any(queues):
        for queue in queues:
            if queue:
                out.append(queue.pop(0))
    return out


def probe(file: str) -> dict:
    """The displayed size (rotation applied), frame rate and video bit rate."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,bit_rate:stream_side_data=rotation",
                          "-of", "json", str(SOURCE_DIR / file)],
                         capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    rotation = next((int(d["rotation"]) for d in stream.get("side_data_list", [])
                     if "rotation" in d), 0)
    width, height = stream["width"], stream["height"]
    if rotation % 180:
        width, height = height, width
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    return {"width": width, "height": height, "fps": num / den,
            "source_kbps": round(int(stream["bit_rate"]) / 1000, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the build_pool.py selection")
    parser.add_argument("--count", type=int, help=f"default {DEFAULT_COUNT}, --dev {DEV_COUNT}")
    parser.add_argument("--dev", action="store_true",
                        help="the development set for the pool, disjoint from the measurement set")
    parser.add_argument("--exclude", type=Path, help="JSON {clip name: reason} to skip")
    parser.add_argument("--inventory", type=Path,
                        help="the source-rate stub build_pool.py --inventory reads "
                             "(default inventory/vc1k.json, --dev inventory/vc1k_dev.json)")
    parser.add_argument("--qp-clipset", type=Path,
                        help="also write the Fixed QP sample of this built clip set "
                             f"(clips/{NAME}.json) as its own clip set here")
    args = parser.parse_args()
    if args.qp_clipset:
        return write_qp_clipset(args.qp_clipset)
    name, seed = (DEV_NAME, DEV_SEED) if args.dev else (NAME, SEED)
    count = args.count or (DEV_COUNT if args.dev else DEFAULT_COUNT)
    inventory = args.inventory or DATA_ROOT / "inventory" / f"{'vc1k_dev' if args.dev else 'vc1k'}.json"
    excluded = json.loads(args.exclude.read_text()) if args.exclude else {}
    held_out = measurement_names() if args.dev else set()
    files = [f.strip() for f in (SOURCE_DIR / EVEN_FPS_LIST).read_text().splitlines() if f.strip()]
    entries = [{"file": f, "name": clip_name(f), **labels(f)} for f in files]
    candidates = [e for e in entries if e["name"] not in excluded and e["name"] not in held_out]
    by_stratum: dict[str, list[dict]] = {}
    for e in candidates:
        by_stratum.setdefault(stratum(e), []).append(e)
    quota = quotas({s: len(v) for s, v in by_stratum.items()}, count, seed)
    chosen = {s: interleaved(v, seed)[:quota[s]] for s, v in sorted(by_stratum.items())}
    picked = [e for s in sorted(chosen) for e in chosen[s]]
    with ThreadPoolExecutor(PROBE_WORKERS) as pool:
        probes = dict(zip((e["file"] for e in picked), pool.map(probe, (e["file"] for e in picked))))
    clips = []
    for i, e in enumerate(picked):
        p = probes[e["file"]]
        length = ({"split": "val" if i % DEV_VAL_EVERY == DEV_VAL_EVERY - 1 else "train"}
                  if args.dev else
                  {"seconds": round((MEASURED_FRAMES + SPARE_FRAMES) / p["fps"], 3)})
        clips.append({"name": e["name"], "source": str(SOURCE_DIR / e["file"]), "start": START_S,
                      **length, "content_class": CONTENT_CLASS, "height": p["height"],
                      "dataset": DATASET, "stratum": stratum(e),
                      **{k: e[k] for k in ("device", "orientation", "environment", "motion",
                                           "lighting")}})
    strata = {s: {"available": len(by_stratum[s]), "chosen": len(chosen[s])}
              for s in sorted(by_stratum)}
    keys = ("device", "orientation", "environment", "motion", "lighting", *(
        ("split",) if args.dev else ()))
    selection = {
        "name": name,
        "_comment": "VC-1K (Meta, CC BY-NC 4.0, github.com/facebookresearch/VC-1K): numbers "
                    "only, never frames or streams; eval/clipset/select_vc1k.py" + (
                        "; development set, internal only (never published)" if args.dev else ""),
        "seed": seed, "count": len(clips), "candidates": len(candidates),
        "excluded": excluded, "strata": strata,
        "mix": {key: count_by(clips, key) for key in keys},
        **({"held_out": "eval/clipset/vc1k.json"} if args.dev else
           {"fixed_qp_sample": fixed_qp_sample(chosen)}),
        "clips": clips}
    args.out.write_text(json.dumps(selection, indent=1) + "\n")
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps({"clips": [
        {"path": str(SOURCE_DIR / f), **probes[f]} for f in sorted(probes)]}, indent=1) + "\n")
    print(f"wrote {args.out}: {len(clips)} clips from {len(candidates)} candidates in "
          f"{len(strata)} strata; {inventory}")
    for s, row in strata.items():
        print(f"  {s:22s} {row['chosen']:3d} of {row['available']}")
    return 0


def measurement_names() -> set[str]:
    """The measurement selection's clips and the ones its build dropped: never dev clips."""
    selection = json.loads(MEASUREMENT_SELECTION.read_text())
    return {c["name"] for c in selection["clips"]} | set(selection["excluded"])


def clip_name(file: str) -> str:
    return "vc1k_" + Path(file).stem.lower()


def count_by(clips: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in clips:
        counts[c[key]] = counts.get(c[key], 0) + 1
    return dict(sorted(counts.items()))


def fixed_qp_sample(chosen: dict[str, list[dict]]) -> list[str]:
    """One clip per motion x lighting cell, the environment alternating from cell to
    cell (the other one where a cell's stratum has no clip): the first pick of it."""
    sample = []
    for i, (motion, light) in enumerate((m, li) for m in MOTIONS for li in LIGHTS):
        order = ENVIRONMENTS if i % 2 == 0 else ENVIRONMENTS[::-1]
        for environment in order:
            picks = chosen.get(f"{motion}/{light}/{environment}")
            if picks:
                sample.append(picks[0]["name"])
                break
    return sample[:FIXED_QP_SAMPLE]


def write_qp_clipset(out: Path) -> int:
    built = json.loads((DATA_ROOT / "clips" / f"{NAME}.json").read_text())
    names = set(built["fixed_qp_sample"])
    clips = [c for c in built["clips"] if c["name"] in names]
    out.write_text(json.dumps({"name": out.stem, "_comment": built["_comment"],
                               "sample_of": NAME, "clips": clips}, indent=1) + "\n")
    print(f"wrote {out}: {len(clips)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
