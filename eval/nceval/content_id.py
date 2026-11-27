"""Content identity of video sources, for the benchmark procedure's disjointness rule.

docs/benchmark-procedure.md (V3): no dataset clip may share content with the PGO training
clips, the tuning sets or another dataset. File names alone do not say this: the same Xiph
derf shoot is `old_town_cross_420_720p50` in the PGO set and `OldTownCross_1920x1080p50` in
AOM CTC A2. `content_id` normalises a name (case, separators, resolution, frame rate, bit
depth and chroma tokens dropped), then maps it through `ALIASES`, a curated table of the
names one content goes by. `SHOOTS` groups contents filmed together (same camera and grain),
which the rule also keeps apart.

    content_id(name) -> str           canonical content of a source name
    shoot(name) -> str                 the content's shoot (itself when not in SHOOTS)
    overlaps(a, b) -> list[(x, y)]     name pairs of two lists sharing a content or a shoot
    source_overlaps(a, b) -> list      manifest entries of two sets cut from one source file
    collection_overlaps(a, b) -> list  manifest entries of two sets from one source collection

Names cannot show that two private clips with different names come from one source file, so
the rule also compares the manifests' source paths (`source_overlaps`, as
eval/clipset/select_heldout.py does when it excludes used sources), and their collections
(`collection_overlaps`): the PGO training set (eval/clipset/pgo_training_2026-11-23.json, NASA
and NOAA footage) must share no collection with any dataset (BP-3.1 task T5).
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

#: Tokens that describe a format, not the content (lower case, separators removed).
_FORMAT_TOKENS = re.compile(
    r"(\d{3,4}x\d{3,4}(p\d{2,3})?|\d{3,4}p\d{0,3}"
    r"|3840|2160|1920|1080|1280|720|640|480|360|270"
    r"|\d+fps|\d+bit|420|422|444|p\d{2,3}|5994|2997|2398|sdr|hdr|downscaled|ter|v\d)")
#: Names a content goes by, normalised (as `_normalise` returns them) -> canonical id.
ALIASES = {
    "oldtowncross": "old_town_cross",
    "crowdrun": "crowd_run",
    "duckstakeoff": "ducks_take_off",
    "intotree": "in_to_tree",
    "parkjoy": "park_joy",
    "parkrun": "park_run",
    "mobcal": "mobcal",
    "shields": "shields",
    "stockholm": "stockholm",
}
#: Contents filmed in one shoot, which the rule keeps apart as if they were one content:
#: the SVT 2006 HD set (OldTownCross, CrowdRun, DucksTakeOff, InToTree, ParkJoy: same camera,
#: film stock and grain), and SVT's earlier 720p50 "ter" set (ParkRun, Shields, Mobcal,
#: Stockholm). The second grouping is the reviewer's knowledge of the Xiph collection
#: (review 2, N8), adopted as the safe side: AOM CTC A4's Stockholm then collides with the
#: PGO clip 720p50_parkrun_ter.
SHOOTS = {**{cid: "svt_hd_2006" for cid in
             ("old_town_cross", "crowd_run", "ducks_take_off", "in_to_tree", "park_joy")},
          **{cid: "svt_ter_720p50" for cid in ("park_run", "shields", "mobcal", "stockholm")}}


def _normalise(name: str) -> str:
    stem = name.rsplit("/", 1)[-1].lower()
    stem = re.sub(r"\.(y4m|yuv|mp4|mkv|ivf|zst)$", "", stem)
    stem = stem.replace("-", "_")
    parts = [p for p in stem.split("_") if p and not _FORMAT_TOKENS.fullmatch(p)]
    return "".join(parts)


def content_id(name: str) -> str:
    key = _normalise(name)
    return ALIASES.get(key, key)


def shoot(name: str) -> str:
    cid = content_id(name)
    return SHOOTS.get(cid, cid)


def overlaps(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    """Pairs (x from a, y from b) sharing a content or a shoot."""
    index: dict[str, list[str]] = {}
    for y in b:
        index.setdefault(shoot(y), []).append(y)
    return [(x, y) for x in a for y in index.get(shoot(x), [])]


def source_overlaps(a: list[dict], b: list[dict]) -> list[tuple[str, str]]:
    """(name in a, name in b) for manifest entries ({"name", "source"}) cut from one file."""
    index: dict[str, list[str]] = {}
    for entry in b:
        index.setdefault(str(entry["source"]), []).append(entry["name"])
    return [(entry["name"], other) for entry in a for other in index.get(str(entry["source"]), [])]


#: Directories that hold many collections, one per subdirectory (a source's collection is the
#: subdirectory below one of these; elsewhere it is the source's own directory).
COLLECTION_ROOTS = ("/mldata/downloaded_datasets/other", "/mldata/downloaded_datasets",
                    "/mldata/tracking_original", "/mldata/tracking", "/mldata/video")


def collection(source: str) -> str:
    """The collection a source file belongs to: e.g. /mldata/downloaded_datasets/other/nasa_videos."""
    path = PurePosixPath(source)
    for root in COLLECTION_ROOTS:
        if path.is_relative_to(root) and len(path.relative_to(root).parts) > 1:
            return str(PurePosixPath(root) / path.relative_to(root).parts[0])
    return str(path.parent)


def collection_overlaps(a: list[dict], b: list[dict]) -> list[tuple[str, str]]:
    """(name in a, name in b) for manifest entries ({"name", "source"}) of one collection."""
    index: dict[str, list[str]] = {}
    for entry in b:
        index.setdefault(collection(str(entry["source"])), []).append(entry["name"])
    return [(entry["name"], other) for entry in a
            for other in index.get(collection(str(entry["source"])), [])]
