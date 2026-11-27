#!/usr/bin/env python3
"""Select the sealed held-out chart set from source collections nothing has used.

    python3 clipset/select_heldout.py used      # -> clipset/heldout_2026-11-23.used.json
    python3 clipset/select_heldout.py select    # -> clipset/heldout_2026-11-23.json and
        # <data root>/inventory/heldout_2026-11-23.sources.json
    python3 clipset/build_pool.py clipset/heldout_2026-11-23.json \
        --inventory $NC_DATA_ROOT/inventory/heldout_2026-11-23.sources.json
    python3 clipset/grade.py heldout_2026-11-23 --no-sheets \
        --inventory $NC_DATA_ROOT/inventory/heldout_2026-11-23.sources.json
    python3 clipset/select_heldout.py finalize  # -> the kept clips, clipset/*.outcome.json

Why (docs/analysis/results-audit-2026-11-23.md finding 8): `independent` was
fresh per file only; 113 of its 150 clips came from the dataset directories the
development clips are cut from. Here the unit of "unused" is the source
COLLECTION (a dataset or capture set), not the file.

`used` lists every /mldata source directory that any commit of the repository
(all refs, every diff), any clip set under the data root's clips/, the
learned-search datasets (nn-data/) or the tuning records (tuning/) has ever
named, and counts every mention of each candidate collection's tokens there.
`select` refuses to run unless every candidate collection has zero mentions.

Candidates (COLLECTIONS) are fixed here, each with the content class its kind of
camera gives (the classes of nceval.pool.STRATA). Rules, all fixed so the draw is
reproducible and nobody picks clips:

  - at least MIN_SOURCE_FPS, long enough for MEASURED_FRAMES (+ spare) frames
    after START_MIN_S; a start time drawn from the seed within the source;
  - a source the pool's grading could accept (select_independent.stored_height:
    native size when its bits per pixel reach nceval.pool.BPP_FLOOR, else 2x
    down); grade.py grades the stored clips afterwards and the C grades leave;
  - stored heights rotate over HEIGHTS within each class (never upscaled), so
    every class is seen at several sizes;
  - one source per family (a collection's capture session or participant),
    PER_CLASS + RESERVE per class, round robin over the class's collections, each
    in a fixed SHA-256 order of the path seeded with SEED; `finalize` keeps the
    first PER_CLASS per class, in draw order, that the build and the grade accept.

The set is sealed (nceval.sealing): only release and publish measurement reads it.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clipset.grade import accepted  # noqa: E402
from clipset.select_independent import MAX_FPS, SPARE_FRAMES, stored_height  # noqa: E402
from nceval.pool import BPP_FLOOR  # noqa: E402

DATA_ROOT = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
EVAL = Path(__file__).resolve().parents[1]
REPO = EVAL.parent
NAME = "heldout_2026-11-23"
SEED = NAME
PREFIX = "ho_"
MEASURED_FRAMES = 150
PER_CLASS = 15
#: Drawn past PER_CLASS per class, in draw order, to replace the clips the build drops
#: (a scene cut too early, a short excerpt) and the grade refuses (`finalize`).
RESERVE = 10
HEIGHTS = (720, 540, 360)
START_MIN_S = 2.0
NAME_HASH_DIGITS = 6
MIN_SOURCE_FPS = 10.0
#: Frame rates ffprobe reports for variable-rate captures above any real camera's.
MAX_PLAUSIBLE_FPS = 120.0
_MLDATA = re.compile(r"/mldata/[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+ -]+){0,3}")
#: The data root's directories whose JSON/text files name the sources measured or
#: trained on (clip sets, learned-search data, tuning records).
DATA_DIRS = ("clips", "nn-data", "tuning")
TEXT_SUFFIXES = {".json", ".txt", ".md", ".csv", ".toml", ".jsonl"}


@dataclass(frozen=True)
class Collection:
    """One source collection: its files, its camera kind's class and its families."""
    key: str
    description: str
    globs: tuple[str, ...]
    content_class: str
    #: What one capture session (family) is: "file", "dir" (the file's directory)
    #: or "set" (the file name's first `_` field: a Caltech drive).
    family_rule: str
    #: Lower-case strings whose mention anywhere would mean the collection was used.
    tokens: tuple[str, ...]
    #: A regex the file name must match (None: every file of the globs).
    name_pattern: str | None = None
    #: Variable-rate captures whose nominal rate is not the real one (VCD2 webcams
    #: deliver 15, 20 or 30 fps under a nominal 30): the rate is counted instead.
    counted_rate: bool = False


OTHER = "/mldata/downloaded_datasets/other"
COLLECTIONS = (
    Collection("vcd2", "Microsoft VCD2 camera-native talking heads (FFV1 lossless webcams)",
               (f"{OTHER}/vcd2/*.mkv", f"{OTHER}/vcd2_more/*.mkv"), "videocall", "file",
               ("vcd2", "vcdpublic"), counted_rate=True),
    Collection("epic_kitchens", "EPIC-KITCHENS-100 head-mounted GoPro (slice)",
               (f"{OTHER}/epic_kitchens_100_slice/P*/*.MP4",), "bodycam", "dir",
               ("epic_kitchens", "epic-kitchens")),
    Collection("ugc_vlog", "YouTube UGC dataset, Vlog category (handheld)",
               (f"{OTHER}/youtube_ugc/original_videos_h264/Vlog_*.mp4",), "bodycam", "file",
               ("youtube_ugc", "ugc_dataset")),
    Collection("caltech", "Caltech Pedestrian vehicle-mounted camera",
               ("/mldata/tracking/caltech_pedestrian/video/*.mp4",), "pan", "set",
               ("caltech",), r"set\d\d_V\d{3}\.mp4"),
    Collection("ucf_crimes", "UCF-Crime surveillance cameras",
               (f"{OTHER}/UCF_Crimes/UCF_Crimes/Videos/*/*.mp4",), "static", "file",
               ("ucf_crime", "ucf-crime")),
    Collection("wm_city", "Wikimedia Commons city traffic", (f"{OTHER}/wikimedia_commons_videos/"
               "city_traffic/*",), "static", "file", ("wikimedia_commons_videos",)),
    Collection("wm_festivals", "Wikimedia Commons festivals", (f"{OTHER}/wikimedia_commons_videos/"
               "festivals/*",), "crowd", "file", ("wikimedia_commons_videos",)),
    Collection("wm_music", "Wikimedia Commons music and dance", (f"{OTHER}/"
               "wikimedia_commons_videos/music_dance/*",), "crowd", "file",
               ("wikimedia_commons_videos",)),
    Collection("wm_drone", "Wikimedia Commons aerial drone", (f"{OTHER}/wikimedia_commons_videos/"
               "aerial_drone/*",), "drone", "file", ("wikimedia_commons_videos",)),
    Collection("wm_night", "Wikimedia Commons night and low light", (f"{OTHER}/"
               "wikimedia_commons_videos/night_lowlight/*",), "lowlight", "file",
               ("wikimedia_commons_videos",)),
    Collection("wm_texture", "Wikimedia Commons water, weather, landscape",
               tuple(f"{OTHER}/wikimedia_commons_videos/{c}/*"
                     for c in ("water", "weather", "landscape_mountain")), "texture", "file",
               ("wikimedia_commons_videos",)),
    Collection("stock", "Vincent Wolf stock footage (4K)",
               (f"{OTHER}/vincent_wolf_stock/*/*.mp4",), "texture", "dir", ("vincent_wolf",)),
)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".ogv", ".mov", ".avi"}


def collection_files(c: Collection) -> list[str]:
    paths = sorted(Path(p) for pattern in c.globs for p in glob.glob(pattern))
    return [str(p) for p in paths if p.suffix.lower() in VIDEO_SUFFIXES and
            (c.name_pattern is None or re.fullmatch(c.name_pattern, p.name))]


def family(c: Collection, path: str) -> str:
    """The capture session a source file belongs to (Collection.family_rule)."""
    if c.family_rule == "set":
        return str(Path(path).parent / Path(path).name.split("_")[0])
    return str(Path(path).parent) if c.family_rule == "dir" else path


# ---------------------------------------------------------------- used ---------

def source_dir(path: str) -> str:
    """The collection directory a /mldata path belongs to (dataset level)."""
    parts = path.rstrip("/").split("/")[1:]
    depth = 4 if parts[1:2] == ["downloaded_datasets"] else 3
    return "/" + "/".join(parts[:depth])


#: This selection's own files: they name the candidates by construction.
OWN_FILES = ("eval/clipset/select_heldout.py", f"eval/clipset/{NAME}*",
             "docs/analysis/heldout-2026-11-23.md")


def history_text() -> str:
    excludes = [f":(exclude,glob){f}" for f in OWN_FILES]
    return subprocess.run(["git", "-C", str(REPO), "log", "--all", "-p", "--no-color", "--",
                           ".", *excludes],
                          capture_output=True, check=True).stdout.decode(errors="replace")


def data_texts() -> list[tuple[str, str]]:
    texts = []
    for name in DATA_DIRS:
        for path in sorted((DATA_ROOT / name).rglob("*")):
            if path.suffix in TEXT_SUFFIXES and path.is_file() and \
                    not path.name.startswith(NAME):
                texts.append((str(path), path.read_text(errors="replace")))
    return texts


def token_pattern(token: str) -> re.Pattern:
    """A whole-word match: "vcd2" must not count inside "vvcd2" (a profile file name)."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])")


def used(out: Path) -> dict:
    """Every source directory named in the repository's history and the data root's
    records, and every mention of a candidate collection."""
    sources = {"git history (all refs)": history_text(), **dict(data_texts())}
    dirs: dict[str, int] = {}
    mentions: dict[str, dict[str, int]] = {c.key: {} for c in COLLECTIONS}
    for where, text in sources.items():
        for path in _MLDATA.findall(text):
            if len(path.split("/")) >= 4:
                dirs[source_dir(path)] = dirs.get(source_dir(path), 0) + 1
        low = text.lower()
        for c in COLLECTIONS:
            hits = sum(len(token_pattern(t).findall(low)) for t in c.tokens) + \
                sum(low.count(g.split("*")[0].lower()) for g in c.globs)
            if hits:
                label = where if where.startswith("git") else "data root"
                mentions[c.key][label] = mentions[c.key].get(label, 0) + hits
    report = {"name": NAME,
              "_comment": "Written by eval/clipset/select_heldout.py used: the /mldata source "
                          "directories (dataset level) named anywhere in the repository's history "
                          "or the data root's clip sets, nn-data and tuning records, with their "
                          "mention counts; and each candidate collection's mentions (all zero for "
                          "a fresh collection).",
              "scanned": {"git_history_refs": "--all", "data_root_dirs": list(DATA_DIRS),
                          "data_root_files": len(sources) - 1},
              "used_source_dirs": dict(sorted(dirs.items())),
              "candidate_mentions": mentions}
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report


# --------------------------------------------------------------- select --------

def probe(path: str, counted_rate: bool) -> dict | None:
    """The source's size, rate, length and bitrate (ffprobe). With `counted_rate`, the
    frame rate is its frames over its duration, to whole frames per second."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          *(["-count_packets"] if counted_rate else []), "-show_entries",
                          "stream=codec_name,width,height,avg_frame_rate,bit_rate,"
                          "nb_read_packets:format=duration,bit_rate", "-of", "json", path],
                         capture_output=True, text=True)
    try:
        data = json.loads(out.stdout)
        stream, fmt = data["streams"][0], data["format"]
        duration = float(fmt.get("duration") or 0)
        num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
        fps = round(int(stream["nb_read_packets"]) / duration) if counted_rate else \
            num / den if den else 0.0
        kbps = float(stream.get("bit_rate") or fmt.get("bit_rate") or 0) / 1000
        return {"path": path, "codec": stream["codec_name"], "width": int(stream["width"]),
                "height": int(stream["height"]), "fps": round(fps, 3),
                "duration_s": duration, "source_kbps": round(kbps, 1),
                **({"counted_rate": True} if counted_rate else {})}
    except (KeyError, IndexError, ValueError, ZeroDivisionError, json.JSONDecodeError):
        return None


def rank(text: str, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}/{text}".encode()).hexdigest()


def excerpt_seconds(fps: float) -> float:
    stored_fps = fps / 2 if fps > MAX_FPS else fps
    return math.ceil((MEASURED_FRAMES + SPARE_FRAMES) / stored_fps * 10) / 10


def candidate(c: Collection, record: dict) -> dict | None:
    fps = record["fps"]
    if not MIN_SOURCE_FPS <= fps <= MAX_PLAUSIBLE_FPS or not record["source_kbps"]:
        return None
    seconds = excerpt_seconds(fps)
    room = record["duration_s"] - seconds - START_MIN_S
    height = stored_height(record)
    if room < 0 or height is None:
        return None
    # a seeded start anywhere the excerpt fits, so long sources are not all cut at 2 s
    start = START_MIN_S + round(int(rank(record["path"])[:8], 16) / 16 ** 8 * room, 1)
    stem = re.sub(r"[^a-z0-9]+", "_", Path(record["path"]).stem.lower()).strip("_")[:40]
    return {"collection": c.key, "family": family(c, record["path"]), "max_height": height,
            "entry": {"name": f"{PREFIX}{c.key}_{stem}", "source": record["path"],
                      "start": start, "seconds": seconds, "content_class": c.content_class,
                      "family": family(c, record["path"]),
                      **({"source_fps": fps} if c.counted_rate else {})}}


def draw(candidates: list[dict], quotas: dict[str, int] | None = None,
         seed: str = SEED) -> list[dict]:
    """PER_CLASS + RESERVE per class (or `quotas` + RESERVE): one per family, round robin
    over the class's collections, heights rotating over HEIGHTS (capped by what the source
    allows)."""
    by_class: dict[str, dict[str, list[dict]]] = {}
    for cand in sorted(candidates, key=lambda x: rank(x["entry"]["source"], seed)):
        by_class.setdefault(cand["entry"]["content_class"], {}) \
            .setdefault(cand["collection"], []).append(cand)
    chosen = []
    for cls in sorted(by_class):
        queues, families, picks, turn = by_class[cls], set(), [], 0
        wanted = (PER_CLASS if quotas is None else quotas.get(cls, 0)) + RESERVE
        while len(picks) < wanted and any(queues.values()):
            keys = sorted(k for k, q in queues.items() if q)
            queue = queues[keys[turn % len(keys)]]
            turn += 1
            while queue and queue[0]["family"] in families:
                queue.pop(0)
            if queue:
                pick = queue.pop(0)
                families.add(pick["family"])
                target = HEIGHTS[len(picks) % len(HEIGHTS)]
                height = min(target, pick["max_height"])
                pick["entry"]["height"] = height - height % 2
                pick["entry"]["draw"] = len(picks)
                picks.append(pick)
        chosen += picks
    return sorted(unique_names(chosen), key=lambda x: x["entry"]["name"])


def unique_names(chosen: list[dict]) -> list[dict]:
    """Names cut from long file names can collide (two Wikimedia uploads of one series):
    those take a short hash of their source path."""
    counts: dict[str, int] = {}
    for x in chosen:
        counts[x["entry"]["name"]] = counts.get(x["entry"]["name"], 0) + 1
    for x in chosen:
        if counts[x["entry"]["name"]] > 1:
            x["entry"]["name"] += "_" + rank(x["entry"]["source"])[:NAME_HASH_DIGITS]
    return chosen


def select(used_report: dict, out: Path, sources_out: Path) -> None:
    dirty = {k: v for k, v in used_report["candidate_mentions"].items() if v}
    if dirty:
        raise SystemExit(f"candidate collections already mentioned (not fresh): {dirty}")
    records, candidates = [], []
    for c in COLLECTIONS:
        for path in collection_files(c):
            record = probe(path, c.counted_rate)
            if record:
                records.append(record)
                if (cand := candidate(c, record)):
                    candidates.append(cand)
    chosen = draw(candidates)
    sources_out.write_text(json.dumps({
        "_comment": f"ffprobe records of {NAME}'s candidate sources (select_heldout.py): the "
                    "inventory build_pool.py and grade.py read for their source bitrates.",
        "clips": records}, indent=1) + "\n")
    classes: dict[str, int] = {}
    for x in chosen:
        classes[x["entry"]["content_class"]] = classes.get(x["entry"]["content_class"], 0) + 1
    out.write_text(json.dumps({
        "name": NAME,
        "_comment": f"The sealed held-out chart set (docs/analysis/heldout-2026-11-23.md): written "
                    f"by eval/clipset/select_heldout.py (seed {SEED}), {len(chosen)} of "
                    f"{len(candidates)} candidates from {len(COLLECTIONS)} collections no "
                    "development, tuning, screening or chart set has used.",
        "sealed": "release-only (nceval.sealing)",
        "clips": [x["entry"] for x in chosen]}, indent=1) + "\n")
    print(f"{len(chosen)} of {len(candidates)} candidates ({len(records)} probed), by class "
          f"{dict(sorted(classes.items()))}; wrote {out}")


def finalize(clips_dir: Path, outcome_out: Path) -> None:
    """Keep the first PER_CLASS clips of each class, in draw order, that were built,
    graded usable (grade.accepted, nceval.pool.BPP_FLOOR for a B) and hold
    MEASURED_FRAMES frames; record every drawn clip's outcome."""
    selection = json.loads((EVAL / "clipset" / f"{NAME}.json").read_text())
    set_path = clips_dir / f"{NAME}.json"
    built = json.loads(set_path.read_text())
    by_name = {e["name"]: e for e in built["clips"]}
    kept, outcome = [], {}
    for spec in sorted(selection["clips"], key=lambda e: (e["content_class"], e["draw"])):
        entry = by_name.get(spec["name"])
        if entry is None:
            reason = "dropped by the build (scene cut or short excerpt)"
        elif "grade" not in entry:
            reason = "not graded"
        elif not accepted(entry) or (entry["grade"] == "B" and
                                     (entry.get("source_bpp") or 0.0) < BPP_FLOOR):
            reason = f"source grade {entry['grade']} (refused)"
        elif entry["frames"] < MEASURED_FRAMES:
            reason = f"{entry['frames']} frames"
        elif sum(e["content_class"] == spec["content_class"] for e in kept) >= PER_CLASS:
            reason = "reserve (class full)"
        else:
            reason = "kept"
            kept.append(entry)
        outcome[spec["name"]] = reason
    set_path.write_text(json.dumps({**{k: v for k, v in built.items() if k != "clips"},
                                    "clips": sorted(kept, key=lambda e: e["name"])},
                                   indent=1) + "\n")
    outcome_out.write_text(json.dumps({
        "_comment": f"Every drawn clip of {NAME} and what became of it "
                    "(eval/clipset/select_heldout.py finalize).", "clips": outcome},
        indent=1) + "\n")
    classes: dict[str, int] = {}
    for e in kept:
        classes[e["content_class"]] = classes.get(e["content_class"], 0) + 1
    print(f"kept {len(kept)} clips, by class {dict(sorted(classes.items()))}; wrote {set_path}")


# --------------------------------------------------------------- redraw --------

#: The re-sealed set (docs/results/sealed-log.md entry 1): heldout_2026-11-23 less the clips
#: seen in design (sealed_exclusions.json), each replaced, with pan and crowd topped up.
REDRAW_NAME = "heldout_2026-11-23b"
#: Clips per class in the re-sealed set: PER_CLASS, pan and crowd topped up to TOP_UP
#: (BP-3 section 3: the thin classes, decided 2026-11-23).
TOP_UP = 12
TOP_UP_CLASSES = ("pan", "crowd")


def redraw_quotas(first_outcome: dict, selection: dict, seen: set[str]) -> dict[str, int]:
    """New clips wanted per class: the target less the first set's kept, unseen clips."""
    classes = {e["name"]: e["content_class"] for e in selection["clips"]}
    kept: dict[str, int] = {}
    for name, reason in first_outcome["clips"].items():
        if reason == "kept" and name not in seen:
            kept[classes[name]] = kept.get(classes[name], 0) + 1
    return {cls: max(0, (TOP_UP if cls in TOP_UP_CLASSES else PER_CLASS) - kept.get(cls, 0))
            for cls in sorted(set(classes.values()))}


def redraw(sources: Path, out: Path) -> None:
    """Draw the replacements from the first draw's probed sources, seeded with REDRAW_NAME,
    never reusing a family (capture session) the first draw drew from."""
    clipset = EVAL / "clipset"
    selection = json.loads((clipset / f"{NAME}.json").read_text())
    outcome = json.loads((clipset / f"{NAME}.outcome.json").read_text())
    seen = {c["name"] for c in json.loads((clipset / "sealed_exclusions.json").read_text())["clips"]}
    quotas = redraw_quotas(outcome, selection, seen)
    drawn_families = {e["family"] for e in selection["clips"]}
    by_path = {f: c for c in COLLECTIONS for f in collection_files(c)}
    candidates = []
    for record in json.loads(sources.read_text())["clips"]:
        c = by_path.get(record["path"])
        cand = candidate(c, record) if c else None
        if cand and cand["family"] not in drawn_families:
            cand["entry"]["name"] = cand["entry"]["name"].replace(PREFIX, "hob_", 1)
            candidates.append(cand)
    chosen = draw(candidates, quotas, REDRAW_NAME)
    out.write_text(json.dumps({
        "name": REDRAW_NAME,
        "_comment": f"Replacements and the pan/crowd top-up for {REDRAW_NAME} (select_heldout.py "
                    f"redraw, seed {REDRAW_NAME}): quotas {quotas} plus {RESERVE} reserve per "
                    f"class, from {len(candidates)} candidates of families {NAME} never drew.",
        "sealed": "release-only (nceval.sealing)",
        "quotas": quotas,
        "clips": [x["entry"] for x in chosen]}, indent=1) + "\n")
    print(f"quotas {quotas}; drew {len(chosen)} of {len(candidates)}; wrote {out}")


def finalize_redraw(clips_dir: Path) -> None:
    """The re-sealed set: the first set's kept clips less the seen ones, plus, per class, the
    first `quotas` redrawn clips in draw order that were built, graded usable and hold
    MEASURED_FRAMES frames. Every redrawn clip's outcome is recorded."""
    clipset = EVAL / "clipset"
    seen = {c["name"] for c in json.loads((clipset / "sealed_exclusions.json").read_text())["clips"]}
    first = json.loads((clips_dir / f"{NAME}.json").read_text())
    drawn = json.loads((clipset / f"{REDRAW_NAME}.draw.json").read_text())
    built_path = clips_dir / f"{REDRAW_NAME}.json"
    built = json.loads(built_path.read_text())
    by_name = {e["name"]: e for e in built["clips"]}
    kept = [e for e in first["clips"] if e["name"] not in seen]
    added: dict[str, int] = {}
    outcome = {}
    for spec in sorted(drawn["clips"], key=lambda e: (e["content_class"], e["draw"])):
        entry, cls = by_name.get(spec["name"]), spec["content_class"]
        if entry is None:
            reason = "dropped by the build (scene cut or short excerpt)"
        elif "grade" not in entry:
            reason = "not graded"
        elif not accepted(entry) or (entry["grade"] == "B" and
                                     (entry.get("source_bpp") or 0.0) < BPP_FLOOR):
            reason = f"source grade {entry['grade']} (refused)"
        elif entry["frames"] < MEASURED_FRAMES:
            reason = f"{entry['frames']} frames"
        elif added.get(cls, 0) >= drawn["quotas"].get(cls, 0):
            reason = "reserve (class full)"
        else:
            reason = "kept"
            added[cls] = added.get(cls, 0) + 1
            kept.append(entry)
        outcome[spec["name"]] = reason
    built_path.write_text(json.dumps({**{k: v for k, v in built.items() if k != "clips"},
                                      "name": REDRAW_NAME,
                                      "clips": sorted(kept, key=lambda e: e["name"])},
                                     indent=1) + "\n")
    (clipset / f"{REDRAW_NAME}.outcome.json").write_text(json.dumps({
        "_comment": f"Every redrawn clip of {REDRAW_NAME} and what became of it "
                    "(eval/clipset/select_heldout.py finalize-redraw).",
        "short_of_quota": {c: q - added.get(c, 0) for c, q in drawn["quotas"].items()
                           if added.get(c, 0) < q},
        "clips": outcome}, indent=1) + "\n")
    classes: dict[str, int] = {}
    for e in kept:
        classes[e["content_class"]] = classes.get(e["content_class"], 0) + 1
    print(f"kept {len(kept)} clips ({sum(added.values())} redrawn), by class "
          f"{dict(sorted(classes.items()))}; wrote {built_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("used", "select", "finalize", "redraw", "finalize-redraw"))
    parser.add_argument("--used", type=Path, default=EVAL / "clipset" / f"{NAME}.used.json")
    parser.add_argument("--out", type=Path, default=EVAL / "clipset" / f"{NAME}.json")
    parser.add_argument("--sources", type=Path,
                        default=DATA_ROOT / "inventory" / f"{NAME}.sources.json")
    args = parser.parse_args()
    if args.step == "used":
        report = used(args.used)
        fresh = [k for k, v in report["candidate_mentions"].items() if not v]
        print(f"{len(report['used_source_dirs'])} used source directories; fresh candidate "
              f"collections: {fresh}; mentioned: "
              f"{ {k: v for k, v in report['candidate_mentions'].items() if v} }")
    elif args.step == "finalize-redraw":
        finalize_redraw(DATA_ROOT / "clips")
    elif args.step == "redraw":
        redraw(args.sources, EVAL / "clipset" / f"{REDRAW_NAME}.draw.json")
    elif args.step == "select":
        select(json.loads(args.used.read_text()), args.out, args.sources)
    else:
        finalize(DATA_ROOT / "clips", EVAL / "clipset" / f"{NAME}.outcome.json")


if __name__ == "__main__":
    main()
