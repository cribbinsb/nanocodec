"""The history page's data for the local benchmark preview (tools/bench/preview.py).

One frame per day's build, from the BP-4 history report of each timing machine (procedure
section 14; tools/bench/report.py over the history run), `<reports>/history-<machine>.json`.
That one report holds every day's builds as families `nanocodec-<codec>@<preset>~<day>`
(a key every 150 frames) and `nanocodec-<codec>@<preset>~<day>~k600` (one key in 600 frames),
the reference encoders and the zero line, timed on the same machines and at the same half load
as the main pages. The key-every-150 families and the references share the report's common set;
the single-key families are on their own long-source clips (report.py `OWN_CLIP_ROWS`).

    <reports>/history-<machine>.json     the report stage's JSON, one per timing machine
    <days>                               [{day, commit, full, codecs}]: each day's build; a day
                                         it does not list is the report's own `nanocodec`
                                         build (the BP-4 benchmark build)

Every figure is the report's: each family's headline BD-J (its `bd`: the common set, or its own
clips for a single-key family; `clips` is the clips it averages, each clip's windows averaged
once) with its interval and speed, the references' the same way from the same report, and each
day's compliance as the worst over our families.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

DEFAULT_DAYS = Path.home() / "nanocodec-data" / "bench" / "bp4-prep" / "history" / "days-bp4.json"
#: The earlier history page's captions (per day: summary, structural landings, per-codec
#: items). Only the structural landings are reused: the per-codec items quote figures of the
#: withdrawn procedure.
DEFAULT_CAPTIONS = Path.home() / "nanocodec-data" / "work" / "history-anim" / "captions-draft.json"
DATA_NAME = "history.json"
#: The history reports' name in the main reports directory (`history-<machine>.json`).
REPORT_NAME = "history"
OWN_PREFIX = "nanocodec-"
OWN_BINARY = "nanocodec"
#: A history family: codec, preset, day and the key-interval variant (none: a key every 150).
HISTORY_FAMILY = re.compile(r"^nanocodec-(\w+)@(\w+)~(\d{4}-\d{2}-\d{2})(?:~(\w+))?$")
#: The benchmark build's own families (no day): the main run on the history clips.
BENCHMARK_FAMILY = re.compile(r"^nanocodec-\w+@\w+$")
#: The key-interval variants (procedure section 14, long windows), the primary first.
KEYS = {
    "k150": "a key every 150 frames",
    "k600": "one key in 600 frames",
}
PRIMARY_KEY = "k150"
COMMIT_SHOWN = 9
#: Procedure section 14, V17: on the current build, every family's subset-minus-full BD-J
#: offset lies within this many points of the median offset, with the same front membership.
#: Owner decision 2026-11-27: a failure no longer withdraws the graph; it is shown with the
#: failed check beside it (read it for the trend across days, not the level).
TRACKING_LIMIT = 1.0
#: The full set the history subset is checked against: the "ours" report of the same machine.
FULL_REPORT = "ours"
DROP_SUFFIX = "+drop"
BENCHMARK_BUILD = "BP-4 benchmark build"

NOTES = (
    "CBR only, strict buffer (bound 1.0), the first key frame's lookahead off. Builds from "
    "before --strict-buffer (2026-11-22) set the bound with rate.recode_overflow=1.0. The "
    "history run's CBR series starts with the 2026-11-09 build (CBR rate control first landed "
    "on 2026-11-07).",
    "Timed like the main pages: the same bare-metal machines, at half load, with each day's "
    "build compiled for that machine. The machine selector switches between the x86 and Arm "
    "reports.",
    "Every BD-J of the primary series, ours and the references', is over the report's common "
    "set: the history clips on which every family with a key every 150 frames, of every day, "
    "has well-formed curves; each clip counts once, its 150- and 600-frame windows averaged. "
    "The single-key families (one key in 600 frames) are coded only on the long-source clips, "
    "so they are shown on those clips, their own set, and do not define the common set. The "
    "clip count is in each frame's heading.",
    "Key interval: a key every 150 frames is the primary series; one key in 600 frames is the "
    "long-window view of key-frame and start-up behaviour (procedure section 14). The last "
    "build (the BP-4 benchmark build) was coded with one key in 600 frames only, so it is the "
    "last frame of that view alone.",
    "References: the fixed builds of the same report, drawn as the same points on every frame, "
    "with their own front.",
    "Compliance is each frame's worst over our families. AV2 key-frame positions are unchecked.",
)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def point(row: dict) -> dict:
    """A family's common-set BD-J with its interval, and its speed."""
    j = (row.get("bd") or {}).get("j") or {}
    return {"j": j.get("mean"), "low": j.get("low"), "high": j.get("high"),
            "speed": row.get("speed_mpix_per_cpu_s"), "clips": row.get("clips")}


def dig(record: dict | None, *keys):
    for key in keys:
        record = record.get(key) if isinstance(record, dict) else None
    return record


def compliance(families: dict) -> list[dict]:
    """The day's compliance: the worst of each figure over our families, as labelled rows."""
    ours = [row["compliance"] for f, row in families.items()
            if f.startswith(OWN_PREFIX) and row.get("compliance")]

    def worst(*keys):
        values = [v for v in (dig(c, *keys) for c in ours) if isinstance(v, (int, float))]
        return max(values) if values else None

    def lowest(*keys):
        values = [v for v in (dig(c, *keys) for c in ours) if isinstance(v, (int, float))]
        return min(values) if values else None

    rows = [
        ("worst buffer peak at its own rate", worst("own_rate", "worst_peak_buffers"), "{:.3f} buffers"),
        ("worst buffer peak at the target rate", worst("target_rate", "worst_peak_buffers"), "{:.3f} buffers"),
        ("largest share of points over 1.0 buffers (own rate)", worst("own_rate", "share_points_over_1"), "{:.2%}"),
        ("largest share of points over 1.0 buffers (target rate)", worst("target_rate", "share_points_over_1"), "{:.2%}"),
        ("achieved / target rate, lowest family mean", lowest("mean_rate_over_target"), "{:.3f}"),
        ("achieved / target rate, highest family mean", worst("mean_rate_over_target"), "{:.3f}"),
    ]
    return [{"label": label, "value": form.format(value)} for label, value, form in rows
            if value is not None]


def by_day(families: dict) -> dict[tuple[str, str], dict[str, dict]]:
    """Our history families grouped by (key variant, day): {"<codec>@<preset>": row}. The
    benchmark build's own families (`nanocodec-<codec>@<preset>`, from the main run on the
    history clips, primary key interval) are the newest day's primary frame: the benchmark
    build is the newest build the history coded."""
    days: dict[tuple[str, str], dict[str, dict]] = {}
    benchmark: dict[str, dict] = {}
    for family, row in families.items():
        if not row.get("clips"):
            continue
        match = HISTORY_FAMILY.match(family)
        if match:
            codec, preset, day, variant = match.groups()
            days.setdefault((variant or PRIMARY_KEY, day), {})[f"{codec}@{preset}"] = row
        elif BENCHMARK_FAMILY.match(family):
            benchmark[family.removeprefix(OWN_PREFIX)] = row
    if benchmark and days:
        days[(PRIMARY_KEY, max(day for _, day in days))] = benchmark
    return days


def frame(day: str, rows: dict[str, dict], build: dict, captions: dict, finite) -> dict:
    """One day's frame at one key interval."""
    rungs: dict[str, list[str]] = {}
    for key in rows:
        codec, _, preset = key.partition("@")
        rungs.setdefault(codec, []).append(preset)
    return {
        "id": day, "label": build.get("label", day), "tick": day[5:],
        "commit": build["commit"][:COMMIT_SHOWN], "codecs": sorted(rungs),
        "rungs": rungs, "caption": list((captions.get(day) or {}).get("structural") or []),
        "view": finite({"clips": max(r["clips"] for r in rows.values()),
                        "values": {k: point(r) for k, r in rows.items()},
                        "compliance": compliance({OWN_PREFIX + k: r for k, r in rows.items()})}),
    }


def references(families: dict, describe, finite) -> dict:
    """The reference encoders as fixed points, from the same report and the same common set
    (the +drop rows, on their own clips, and the metric-tuned twins are not drawn)."""
    shown = {f: row for f, row in families.items()
             if not f.startswith(OWN_PREFIX) and row.get("clips")
             and not describe(f)["drop"] and not describe(f)["metric_tuned"]}
    return finite({f: point(row) | {k: describe(f)[k] for k in ("label", "owner")}
                   for f, row in shown.items()})


def machine_view(report: dict, machine: str, builds: dict[str, dict], captions: dict,
                 describe, finite, source: Path) -> dict:
    """One machine's history: a frame list per key interval, oldest first, and the references."""
    body = report["series"][machine]
    families = body["families"]
    own = dig(report, "provenance", "encoders", OWN_BINARY) or {}
    benchmark = {"commit": own.get("commit") or "", "label": BENCHMARK_BUILD}
    series = {key: [frame(day, rows, builds.get(day, benchmark), captions, finite)
                    for (variant, day), rows in sorted(by_day(families).items(),
                                                       key=lambda item: item[0][1])
                    if variant == key]
              for key in KEYS}
    return {
        "source": str(source), "procedure": dig(report, "provenance", "procedure", "report"),
        "view_clips": body["view_clips"], "common_clips": body["common_clips"],
        "zero_line": body["zero_line"], "series": series,
        "references": references(families, describe, finite),
    }


def tracking_check(history: dict, full: dict) -> dict:
    """V17 over the families both reports hold for the current build (our benchmark build's
    own families and the references; not the earlier days, not the +drop rows, which are on
    their own clips): each one's subset-minus-full BD-J offset against the median offset, and
    the front membership among them."""
    families, full_families = history["families"], full["families"]
    offsets = {f: row["bd"]["j"]["mean"] - full_families[f]["bd"]["j"]["mean"]
               for f, row in families.items()
               if not HISTORY_FAMILY.match(f) and not f.endswith(DROP_SUFFIX) and row.get("clips")
               and (full_families.get(f) or {}).get("clips")}
    median = statistics.median(offsets.values())
    outside = {f: o for f, o in offsets.items() if abs(o - median) > TRACKING_LIMIT}
    front = sorted(set(history["front"]).intersection(offsets))
    full_front = sorted(set(full["front"]).intersection(offsets))
    return {"families": len(offsets), "median_offset": median, "limit": TRACKING_LIMIT,
            "outside": dict(sorted(outside.items())), "offsets": dict(sorted(offsets.items())),
            "front_same": front == full_front, "passed": not outside and front == full_front,
            "subset_clips": history["common_clips"], "full_clips": full["common_clips"]}


def history_data(reports: Path, days_path: Path, captions_path: Path, machines: list[str],
                 shared: dict, describe, finite, report_path) -> dict:
    """The history page's data from each machine's history report found under `reports`;
    `shared` is the preview's name and the machines of its selector."""
    builds = {d["day"]: d for d in read_json(days_path) or []}
    captions = read_json(captions_path) or {}
    views, missing = {}, []
    for machine in machines:
        path = report_path(reports, REPORT_NAME, machine)
        report = read_json(path)
        if report is None:
            missing.append(str(path))
            continue
        full = read_json(report_path(reports, FULL_REPORT, machine))
        check = full and tracking_check(report["series"][machine], full["series"][machine])
        views[machine] = machine_view(report, machine, builds, captions, describe, finite, path)
        views[machine]["tracking_check"] = check and finite(check)
    return shared | {"keys": KEYS, "primary_key": PRIMARY_KEY, "notes": list(NOTES),
                     "views": views, "missing": missing}
