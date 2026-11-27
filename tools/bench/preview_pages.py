"""The results site's clips and SVC pages' data (tools/bench/preview.py writes them beside bp3.json).

  - clips.json: each dataset's clips as the benchmark coded them, from the dataset's committed
    manifest (names and numbers only: no path, no frame, no stream; VC-1K is numbers and names
    only by its licence) with the rungs the shared ladder gives each clip (nceval.rate_ladder,
    the definition the benchmark's planner reads), and, per timing machine's report, which
    families' curves on that clip were malformed (the report's own `missing`), so a reader sees
    why a clip is out of the common set. Nothing is measured here.
  - svc.json: the SVC chart's record (`python3 -m nceval.svc_chart site`, SVC_RECORD) as it is,
    with the timing machines' names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from nceval import rate_ladder  # noqa: E402

#: Each dataset's committed manifest as the benchmark ran it (BP-4: the 600-frame cut where
#: the source allows it).
MANIFESTS = {"ours": "eval/clipset/heldout_2026-11-23b-bp4.manifest.json",
             "vc1k": "eval/clipset/vc1k_2026-11-18-bp4.manifest.json",
             "aomctc": "eval/clipset/aomctc_bp3.manifest.json",
             "fast": "eval/clipset/heldout_2026-11-23b-bp4.manifest.json"}
#: A dataset that is a fixed subset of another's manifest: its clip-set file (`clips`: [{name}]).
SUBSETS = {"fast": "eval/clipset/fast-subset.json"}
#: The record of the dataset clips a tuning set used (procedure section 4, V3).
TUNED_ON = REPO / "eval" / "clipset" / "tuned_on.json"
#: The 600-frame window (procedure D7): coded where the source allows it.
LONG_WINDOW = 600
WINDOW_SEPARATOR = "@"
DROP_SUFFIX = "+drop"
#: The SVC chart's site record (nceval.svc_chart `site`).
SVC_RECORD = REPO / "docs" / "results" / "svc-site.json"


def windows(clip: dict) -> list[int]:
    """The windows a clip was coded in: its own length, plus 150 read from a 600 encode."""
    frames = int(clip["frames"])
    return [150, frames] if frames >= LONG_WINDOW else [frames]


def rungs(clip: dict) -> list[int]:
    """The clip's targets on the shared ladder, within its source-rate cap."""
    targets = [rate_ladder.scaled_kbps(r, clip["width"], clip["height"], clip["fps"])
               for r in rate_ladder.LADDER_720P30_KBPS]
    return [t for t in targets if rate_ladder.allowed(t, float(clip["source_kbps"]))]


def clip_row(clip: dict) -> dict:
    num, den = rate_ladder.frame_rate(clip["fps"])
    return {"name": clip["name"], "class": clip.get("stratum") or clip["content_class"],
            "width": clip["width"], "height": clip["height"], "fps": round(num / den, 3),
            "windows": windows(clip), "source_kbps": round(float(clip["source_kbps"])),
            "rungs": rungs(clip), "status": clip.get("status")}


def lost_by_clip(report: dict, machine: str) -> dict[str, dict[str, list[str]]]:
    """{clip: {window: [family, ...]}}: the families whose curves on that clip window were
    malformed in a report (a +drop row's loss does not take the clip out of the common set)."""
    lost: dict[str, dict[str, list[str]]] = {}
    for family, row in report["series"][machine]["families"].items():
        for key in row["missing"]:
            name, _, window = key.partition(WINDOW_SEPARATOR)
            lost.setdefault(name, {}).setdefault(window or "all", []).append(family)
    return lost


def tuned_on() -> dict[str, str]:
    """{clip: the tuning set that used it} from the in-sample record."""
    record = json.loads(TUNED_ON.read_text())
    return {c["name"]: f"{name} ({entry['used']})"
            for name, entry in record["sets"].items() for c in entry["clips"]}


def manifest_clips(key: str) -> list[dict]:
    """A dataset's manifest clips, restricted to its subset file where it is a subset."""
    clips = json.loads((REPO / MANIFESTS[key]).read_text())["clips"]
    if key not in SUBSETS:
        return clips
    names = {c["name"] for c in json.loads((REPO / SUBSETS[key]).read_text())["clips"]}
    return [c for c in clips if c["name"] in names]


def clips_data(reports: Path, datasets, report_path) -> dict:
    """Each dataset's clips and, per timing machine it is measured on, its report's losses."""
    out = {}
    tuned = tuned_on()
    for dataset in datasets:
        views = {}
        for machine in dataset.machines:
            path = report_path(reports, dataset.key, machine)
            if path.is_file():
                report = json.loads(path.read_text())
                series = report["series"][machine]
                views[machine] = {"view_clips": series["view_clips"],
                                  "common_clips": series["common_clips"],
                                  "lost": lost_by_clip(report, machine)}
        out[dataset.key] = {"name": dataset.name, "about": dataset.about, "doc": dataset.doc,
                            "manifest": SUBSETS.get(dataset.key, MANIFESTS[dataset.key]),
                            "clips": [clip_row(c) | {"tuned_on": tuned.get(c["name"])}
                                      for c in manifest_clips(dataset.key)], "views": views}
    return {"datasets": out, "drop_suffix": DROP_SUFFIX,
            "ladder_kbps": list(rate_ladder.LADDER_720P30_KBPS)}


def svc_data(machines: dict) -> dict | None:
    """The SVC record with the timing machines' labels, or None before it is measured."""
    if not SVC_RECORD.is_file():
        return None
    return {**json.loads(SVC_RECORD.read_text()), "machines": machines,
            "source": str(SVC_RECORD.relative_to(REPO))}
