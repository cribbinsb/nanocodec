#!/usr/bin/env python3
"""The benchmark results pages: the benchmark reports as the results site's data, and a local preview.

    python3 tools/bench/preview.py --out /abs/dir     # every report found under --reports

Reads the report stage's JSON of each dataset on each timing machine, `<dataset>-<machine>.json`
(tools/bench/report.py; docs/benchmark-procedure.md sections 8-9; CBR only since BP-4), and
writes one data file, `results/data/bp3.json`, with both machines' views (the pages show one
machine at a time and switch between them), that the pages in tools/bench/pages/ render with
the results site's own chart, table and style code (web/results/chart.js, common.js,
frontier.js, domain.js, site.css), copied beside them. Nothing is measured, averaged or rounded
here: every figure is a report's, and a report that is not there yet is listed as missing, so a
later run of the same command adds it.

The output is a local, self-contained directory (no clip, frame or stream; numbers only). It
is the results section of the site (web/ with this directory as results/): nothing here
publishes; the owner decides when it is published (2026-11-27: BP-4 published).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import preview_history

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "tools" / "bench" / "pages"
SITE = REPO / "web" / "results"
PROCEDURE_DOC = REPO / "docs" / "benchmark-procedure.md"
PROCEDURE_VERSION = re.compile(r"^# Benchmark procedure \((BP-[\d.]+),")
#: Where the report stage writes this round's reports, one `<dataset>-<machine>.json` each.
DEFAULT_REPORTS = Path.home() / "nanocodec-data" / "bench" / "bp4-prep" / "reports"
#: The results site's own modules and styles the pages use unchanged.
SHARED_ASSETS = ("chart.js", "common.js", "domain.js", "frontier.js", "site.css")
ROOT_ASSETS = ("site-header.css", "icon.svg")
DATA_NAME = "bp3.json"
NOT_RECORDED = "not recorded"

CODEC_ORDER = ("h264", "hevc", "vvc", "av1", "vp9", "av2")
CODEC_NAMES = {"h264": "H.264", "hevc": "HEVC", "vvc": "VVC", "av1": "AV1", "vp9": "VP9",
               "av2": "AV2"}
OWN_PREFIX = "nanocodec-"
TUNE_SEPARATOR = "~"
DROP_SUFFIX = "+drop"
#: Reference families (procedure section 3): name prefix, project, codec, label pattern.
REFERENCES = (
    ("x264-", "x264", "h264", "x264 {}"),
    ("x265-", "x265", "hevc", "x265 {}"),
    ("svtav1-m", "SVT-AV1", "av1", "SVT-AV1 M{}"),
    ("libaom-av1-cpu", "libaom", "av1", "libaom rt {}"),
    ("libvpx-vp9-cpu", "libvpx", "vp9", "libvpx VP9 rt {}"),
    ("kvazaar-", "Kvazaar", "hevc", "Kvazaar {}"),
    ("vvenc-", "VVenC", "vvc", "VVenC low delay {}"),
    ("openh264-", "openh264", "h264", "openh264 {}"),
)
#: The metric-oriented tunes (section 3's PSNR and SSIM twins); openh264's ~cabac and ~aq
#: twins are other configurations, not metric tunes.
METRIC_TUNES = ("psnr", "ssim")
#: The report's metric keys, in the pages' order (J first: the composite).
METRICS = ("j", "psnr_yuv", "vmaf_v1", "ms_ssim")


@dataclass(frozen=True)
class Machine:
    """A timing series of procedure section 2: the pages' selector label, the instance type,
    and how it is timed (section 6, with its measured half-load against alone factor)."""
    label: str
    name: str
    about: str


#: The timing series (procedure sections 2 and 6), x86 first: the pages open on it.
MACHINES = {
    "c8i": Machine("x86 (Intel Granite Rapids)", "c8i.metal-48xl (Intel Granite Rapids)",
                   "AWS c8i.metal-48xl (bare metal; Intel Xeon 6975P-C, Granite Rapids, 96 "
                   "cores), loaded speed at half load: one longest-first queue of the machine's "
                   "timed jobs on every second physical core (against alone, ×1.00-1.07, "
                   "procedure section 6)"),
    "c8g": Machine("Arm (Graviton 4)", "c8g.metal-24xl (Graviton 4)",
                   "AWS c8g.metal-24xl (bare metal; Graviton 4, 96 cores), with the aarch64 "
                   "builds, loaded speed at half load: one longest-first queue of the machine's "
                   "timed jobs on every second physical core (against alone, ×1.01-1.23, "
                   "procedure section 6)"),
}


@dataclass(frozen=True)
class Dataset:
    """A dataset of procedure section 4, as the pages name it."""
    key: str
    name: str
    about: str


DATASETS = (
    Dataset("ours", "\"ours\" held-out set",
            "the project's held-out clips (eval/clipset/heldout_2026-11-23b.manifest.json)"),
    Dataset("vc1k", "VC-1K", "phone video calls from Meta's VC-1K dataset (CC BY-NC 4.0); "
                             "numbers only"),
    Dataset("aomctc", "AOM-CTC", "the AOM common test conditions' sequences "
                                 "(eval/clipset/aomctc_bp3.manifest.json), A2 included"),
)
#: The report's headline set of a family averaged over the common set; a +drop row's headline
#: is its own clips (section 9).
COMMON_HEADLINE = "common set"
OWN_BINARY = "nanocodec"
#: The reference encoders: the name shown and the programs of it a report's provenance names.
REFERENCE_BUILDS = (
    ("x264", ("x264",)),
    ("x265", ("x265",)),
    ("SVT-AV1", ("SvtAv1EncApp",)),
    ("libaom", ("aomenc", "svc_encoder_rtc")),
    ("libvpx", ("vpxenc",)),
    ("Kvazaar", ("kvazaar",)),
    ("VVenC", ("vvencFFapp",)),
    ("openh264", ("openh264-harness",)),
)
COMMIT_SHOWN = 9


def report_path(reports: Path, dataset: str, machine: str) -> Path:
    return reports / f"{dataset}-{machine}.json"


def procedure_version() -> str:
    """The procedure's version, from the document's own title."""
    match = PROCEDURE_VERSION.match(PROCEDURE_DOC.read_text().splitlines()[0])
    if not match:
        raise ValueError(f"{PROCEDURE_DOC}: no version in the title")
    return match.group(1)


def describe(family: str) -> dict:
    """Owner, codec, preset, tune and label of a report family."""
    drop = family.endswith(DROP_SUFFIX)
    base, _, tune = family.removesuffix(DROP_SUFFIX).partition(TUNE_SEPARATOR)
    suffix = (f" {TUNE_SEPARATOR}{tune}" if tune else "") + (f" {DROP_SUFFIX}" if drop else "")
    if base.startswith(OWN_PREFIX):
        codec, _, preset = base.removeprefix(OWN_PREFIX).partition("@")
        return {"owner": "nanocodec", "codec": codec, "preset": preset.capitalize(), "tune": tune,
                "drop": drop, "ours": True, "metric_tuned": False,
                "label": f"nanocodec {CODEC_NAMES[codec]} {preset.capitalize()}{suffix}"}
    for prefix, owner, codec, pattern in REFERENCES:
        if base.startswith(prefix):
            preset = base.removeprefix(prefix)
            return {"owner": owner, "codec": codec, "preset": preset, "tune": tune, "drop": drop,
                    "ours": False, "metric_tuned": tune in METRIC_TUNES,
                    "label": pattern.format(preset) + suffix}
    raise ValueError(f"unknown family {family!r}")


def finite(value):
    """A report value for JSON: NaN and infinities (no clips to average) become null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def family_values(row: dict, dropped: dict) -> dict:
    """One family's figures in a view: its headline BDs (the common set; a +drop row's own
    clips) and the clips they average, its own anchor-valid clip count, speed, compliance."""
    common = row["headline_set"] == COMMON_HEADLINE
    return finite({
        "bd": row["bd"] if common else row["bd_own_set"],
        "clips": row["clips"] if common else row["own_clips"], "own_only": not common,
        "own_clips": row["own_clips"], "lost": len(row["missing"]),
        "speed": row["speed_mpix_per_cpu_s"], "speed_raw": row["speed_raw_mpix_per_cpu_s"],
        "speed_clips": row["speed_clips"],
        "compliance": row["compliance"], "exclusion_check": row["exclusion_check"],
        "envelope_dropped": dropped,
    })


def same_streams(families: dict) -> dict[str, str]:
    """Tune twins this report measured identically, every BD mean on both clip sets and both
    clip counts: {twin: kept}. Section 3: two twins that are the same encoder are shown once.
    The bootstrap intervals are not compared: each family draws its own resamples."""
    def figures(row):
        means = [row[clip_set][m]["mean"] for clip_set in ("bd", "bd_own_set") for m in METRICS]
        return json.dumps([means, row["clips"], row["own_clips"]])
    kept: dict[tuple[str, str], str] = {}
    twins = {}
    for family in sorted(families):
        base = family.removesuffix(DROP_SUFFIX).partition(TUNE_SEPARATOR)[0]
        first = kept.setdefault((base, figures(families[family])), family)
        if first != family:
            twins[family] = first
    return twins


def commit_date(commit: str) -> str | None:
    """A commit's date in UTC as git records it, or None where this checkout lacks it."""
    shown = subprocess.run(
        ["git", "-C", str(REPO), "show", "-s", "--date=format-local:%Y-%m-%d %H:%M UTC",
         "--format=%cd", commit], capture_output=True, text=True, env={"TZ": "UTC"}, check=False)
    return shown.stdout.strip() or None


def build_identity(encoders: dict, used: set[str]) -> dict:
    """The builds a report's points were coded with, as its provenance checked them by SHA-256
    against the build records (V25): ours (commit, its date, version, how it was built) and
    each reference encoder the runs used (`used`: their manifests' programs; pinned ref and
    commit, LTO, PGO). A binary the provenance does not name is None: the pages print it as
    not recorded."""
    ours = encoders.get(OWN_BINARY)
    own = ours and {
        "commit": ours["commit"][:COMMIT_SHOWN], "date": commit_date(ours["commit"]),
        "version": ours["version"].removesuffix(ours["commit"]).strip(),
        "build": ours["build"], "sha256": ours["sha256"]}
    references = []
    for name, programs in REFERENCE_BUILDS:
        if not used.intersection(programs):
            continue
        tool = next((encoders[p] for p in programs if p in encoders), None)
        references.append({"name": name, "programs": list(programs)} | (
            {"version": tool["ref"], "commit": tool["commit"][:COMMIT_SHOWN],
             "lto": tool["lto"], "pgo": tool["pgo"]} if tool else {"version": None}))
    stores = sorted({e["store"] for e in encoders.values() if e.get("store")})
    return {"nanocodec": own or None, "references": references,
            "record": ", ".join(stores) or NOT_RECORDED}


def recorded(manifests: list[dict], key: str) -> list[dict]:
    """The entries of a manifest list (patches, exceptions) over a report's runs, each once."""
    entries = [e for m in manifests for e in m.get(key) or []]
    return [e for i, e in enumerate(entries) if e not in entries[:i]]


def view(dataset: Dataset, machine: str, path: Path) -> dict:
    """One report (a dataset timed on one machine) as a page view."""
    report = json.loads(path.read_text())
    body = report["series"][machine]
    families = body["families"]
    twins = same_streams(families)
    provenance = report.get("provenance", {})
    checked = provenance.get("encoders", {})
    manifests = report["manifests"]
    return {
        "id": f"{dataset.key}-{machine}", "dataset": dataset.key, "machine": machine,
        "procedure": (provenance.get("procedure") or {}).get("report") or NOT_RECORDED,
        "view_clips": body["view_clips"], "common_clips": body["common_clips"],
        "zero_line": body["zero_line"], "front": body["front"],
        "timing_series": report["timing_series"] or machine,
        "quality_series": report["quality_series"] or machine,
        "phase_name": manifests[0]["phase_name"], "phase": manifests[0]["phase"],
        "procedure_commits": sorted({m["procedure_commit"] for m in manifests}),
        "patches": recorded(manifests, "patches"),
        "exceptions": recorded(manifests, "exceptions"),
        "binaries": manifests[0]["binaries"] | {
            name: encoder["sha256"] for name, encoder in checked.items()
            if name in manifests[0]["binaries"] and encoder.get("sha256")},
        "build": build_identity(checked, set(manifests[0]["binaries"])),
        "envelope": report["envelope"], "speed_provisional": report["speed_provisional"],
        "refused_windows": report["refused_boxes"],
        "same_streams": twins, "source": str(path),
        "values": {f: family_values(row, report["envelope_dropped"].get(f, {}))
                   for f, row in families.items() if f not in twins},
    }


def machines() -> dict:
    """The timing machines as the pages show them, in the selector's order."""
    return {key: {"label": m.label, "name": m.name, "about": m.about} for key, m in MACHINES.items()}


def site_data(reports: Path) -> dict:
    """Every report found under `reports`, on every machine, and the ones not there yet."""
    views, missing = [], []
    for machine in MACHINES:
        for dataset in DATASETS:
            path = report_path(reports, dataset.key, machine)
            if path.is_file():
                views.append(view(dataset, machine, path))
            else:
                missing.append({"id": f"{dataset.key}-{machine}", "dataset": dataset.key,
                                "machine": machine, "path": str(path)})
    families = sorted({f for v in views for f in v["values"]})
    return {
        "current_procedure": procedure_version(),
        "datasets": {d.key: {"name": d.name, "about": d.about} for d in DATASETS},
        "machines": machines(), "metrics": list(METRICS),
        "codecs": CODEC_NAMES, "codec_order": list(CODEC_ORDER),
        "points": {f: describe(f) for f in families},
        "views": views, "missing": missing,
    }


def write_site(data: dict, out: Path, history: dict | None = None) -> Path:
    """The preview directory: the pages, the shared site assets and the data."""
    results = out / "results"
    (results / "data").mkdir(parents=True, exist_ok=True)
    for page in sorted(PAGES.iterdir()):
        shutil.copyfile(page, results / page.name)
    for name in SHARED_ASSETS:
        shutil.copyfile(SITE / name, results / name)
    for name in ROOT_ASSETS:
        shutil.copyfile(SITE.parent / name, out / name)
    (results / "data" / DATA_NAME).write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    if history is not None:
        (results / "data" / preview_history.DATA_NAME).write_text(
            json.dumps(history, indent=1, allow_nan=False) + "\n")
    return results / "index.html"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="the preview directory (absolute)")
    ap.add_argument("--reports", type=Path, default=DEFAULT_REPORTS,
                    help="where the <dataset>-<machine>.json reports are")
    ap.add_argument("--days", type=Path, default=preview_history.DEFAULT_DAYS,
                    help="the history days' builds (days.json: day, commit, codecs)")
    ap.add_argument("--captions", type=Path, default=preview_history.DEFAULT_CAPTIONS,
                    help="the earlier history page's captions (their structural landings)")
    a = ap.parse_args(argv)
    if not a.out.is_absolute():
        ap.error("--out takes an absolute path")
    data = site_data(a.reports)
    shared = {"current_procedure": data["current_procedure"],
              "machines": machines(), "codecs": CODEC_NAMES}
    history = preview_history.history_data(a.reports, a.days, a.captions, list(MACHINES), shared,
                                           describe, finite, report_path)
    index = write_site(data, a.out, history)
    for machine, h in history["views"].items():
        print(f"history-{machine}: " + ", ".join(
            f"{key} {len(frames)} frames" for key, frames in h["series"].items())
              + f", common set {h['common_clips']} of {h['view_clips']} ({h['source']})")
    for path in history["missing"]:
        print(f"history: no report found ({path})")
    for v in data["views"]:
        print(f"{v['id']}: {len(v['values'])} families, common set {v['common_clips']} of "
              f"{v['view_clips']} clips ({v['source']})")
        own = v["build"]["nanocodec"]
        print(f"  nanocodec build: {own['commit'] + ', ' + str(own['date']) if own else 'NOT RECORDED'}")
    for m in data["missing"]:
        print(f"{m['id']}: no report found ({m['path']})")
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
