"""Backfill the lever ledger (nceval.levers) from every result measured before it.

    python3 -m nceval.levers backfill [--doc-rows FILE] [--into LEDGER]

Sources, each turned into rows by the same converters the live writers use:

  - result JSONs under the data root's work/, tuning/, results/ and agent
    worktrees: nceval.screen (real levels, real tracks, the fixed-QP proxy and
    its older format), nceval.levels final-check, nceval.fixedqp ab and cost,
    and the hand-rolled callgrind cost files (`costs` with `p_million` and
    inline params, measured against their parameter-free label);
  - tune studies on the current protocol (tuning/*.sqlite, `protocol`
    pool-jreal-*): every complete trial and every L3-checked finalist. The
    older three-objective studies (fixed-QP BD of PSNR and VMAF NEG) are not
    J_real and are left out;
  - sensitivity sweeps (tuning/sens-*/results.json), with each parameter's
    preset value from the sweep's own table as the base value;
  - rows extracted from the analysis and handoff docs (--doc-rows, e.g.
    eval/results/sources/doc-rows-2026-11-15.jsonl: one reading of 33 docs' tables;
    kind doc, level doc:<as stated>).

Point caches (realuse/, lens/, screen-cache/, ...) hold single encodes, not
comparisons, and are skipped. Appending is deduplicated by row id, so a
rerun adds only what is new.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import levers
from .config import load_paths

#: Directories holding single points or build trees, never comparisons.
SKIP_DIRS = {"target", ".git", "node_modules", "crates", "lens", "cache", "points",
             "screen-cache", "realuse", "fuzz", "pkg", "decode-work", "heldout-chart",
             "svc-chart", "ratecontrol", "cloud-results", "standards-eval", "steady-state",
             "persistence", "curves", "curve-checkpoints", "retained", "bench", "web"}
#: Result files start with one of these keys (a cheap filter before json.load).
MARKERS = ('"mode"', '"candidates"', '"costs"', '"old_commit"', '"anchor_binary"',
           '"status"')
MAX_BYTES = 50_000_000
CODECS = ("h264", "hevc", "av1", "vp9", "av2", "vvc")
#: Labels of the reference arm in hand-rolled cost files.
COST_ANCHORS = ("anchor", "accepted", "baseline", "base", "default", "preset", "exact")


def file_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")


def result_files(roots: list[Path]):
    for root in roots:
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS and not d.startswith("f1")]
            for name in files:
                path = Path(directory) / name
                if not name.endswith(".json"):
                    continue
                try:
                    size = path.stat().st_size
                    if not 50 < size < MAX_BYTES:
                        continue
                    with path.open() as source:
                        head = source.read(600)
                    if not any(marker in head for marker in MARKERS):
                        continue
                    data = json.loads(path.read_text())
                except (OSError, ValueError, UnicodeDecodeError):
                    continue
                if isinstance(data, dict):
                    yield path, data


def classify(data: dict) -> str | None:
    if "old_commit" in data and "codecs" in data:
        return "final-check"
    if "candidates" in data and "rotation" in data and "codec" in data:
        return "fixedqp-ab"
    if "costs" in data:
        return "cost"
    if "anchor_binary" in data and ("stages" in data or "levels" in data) and "codec" in data:
        return "screen"
    return None


def codec_of(params: dict, name: str) -> str | None:
    prefixes = {k.split(".")[0] for k in params}
    found = [c for c in CODECS if c in prefixes]
    if found:
        return found[0]
    return next((c for c in CODECS if re.search(rf"(^|[^a-z]){c}([^a-z0-9]|$)", name)), None)


def handrolled_cost_rows(data: dict, path: Path, date: str) -> list[dict]:
    """`costs: {label: {p_million|p_instructions, params}}` against the parameter-free arm."""
    costs = {label: c for label, c in data["costs"].items() if isinstance(c, dict)}
    value = lambda c: c.get("p_million") or c.get("p_minstr") or c.get("p_instructions")
    anchors = [label for label, c in costs.items() if not c.get("params") and value(c)
               and any(word in label for word in COST_ANCHORS)]
    if not anchors:
        return []
    base = value(costs[anchors[0]])
    rows: list[dict] = []
    for label, c in costs.items():
        params = levers.pairs_to_dict(c.get("params") or {})
        codec = data.get("codec") or codec_of(params, path.name)
        if not params or not value(c) or codec is None:
            continue
        rows += levers.cost_rows(codec, data.get("preset") or levers.MEDIUM,
                                 {label: (params, 100 * (value(c) / base - 1))},
                                 levers.CALLGRIND_P, data.get("source_commit"), str(path),
                                 "backfill", date)
    return rows


def json_rows(roots: list[Path]) -> list[dict]:
    files = list(result_files(roots))
    # fixedqp cost files name their candidates by label: the ab runs of the same
    # directory carry each label's pairs.
    labels: dict[Path, dict[str, list[str]]] = {}
    for path, data in files:
        if classify(data) == "fixedqp-ab":
            for candidate in data.get("candidates") or []:
                labels.setdefault(path.parent, {})[candidate["label"]] = candidate["param"]
    rows: list[dict] = []
    for path, data in files:
        kind, date, source = classify(data), file_date(path), str(path)
        try:
            if kind == "screen":
                rows += levers.screen_rows(data, source, "backfill", date)
            elif kind == "final-check":
                rows += levers.final_check_rows(data, source, "backfill", date)
            elif kind == "fixedqp-ab":
                rows += levers.fixedqp_ab_rows(data, source, "backfill", date)
            elif kind == "cost" and all("dcpu_pct" in c for c in data["costs"].values()):
                known = {**labels.get(path.parent.parent, {}), **labels.get(path.parent, {})}
                rows += levers.fixedqp_cost_rows(data, known, source, "backfill", date)
            elif kind == "cost":
                rows += handrolled_cost_rows(data, path, date)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            print(f"skipped {path}: {type(error).__name__}: {error}", flush=True)
    return rows


def study_attrs(connection: sqlite3.Connection) -> dict:
    return {k: json.loads(v) for k, v in
            connection.execute("select key, value_json from study_user_attributes")}


def study_rows(path: Path) -> tuple[list[dict], dict]:
    """Rows of one current-protocol tune study, and its attrs."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        attrs = study_attrs(connection)
        if not str(attrs.get("protocol", "")).startswith("pool-jreal"):
            return [], attrs
        trials: dict[int, dict] = {}
        for trial_id, number in connection.execute(
                "select trial_id, number from trials where state = 'COMPLETE'"):
            params = dict(connection.execute(
                "select param_name, param_value from trial_params where trial_id = ?",
                (trial_id,)))
            values = dict(connection.execute(
                "select objective, value from trial_values where trial_id = ?", (trial_id,)))
            user = {k: json.loads(v) for k, v in connection.execute(
                "select key, value_json from trial_user_attributes where trial_id = ?",
                (trial_id,))}
            trials[number] = {"params": params, "values": values, "attrs": user}
    finally:
        connection.close()
    if 0 not in trials:
        return [], attrs
    start = trials[0]["params"]
    date = file_date(path)
    rows = []
    for number, trial in trials.items():
        j = trial["values"].get(0)
        cpu = trial["attrs"].get("cpu", trial["values"].get(1))
        if number == 0 or j is None:
            continue
        row = levers.trial_row(attrs, number, trial["params"], start, j, cpu, trial["attrs"],
                               str(path), "backfill", date)
        if row["overrides"]:
            rows.append(row)
    rows += levers.finalist_rows(attrs, attrs.get("finalists") or {},
                                 {n: t["params"] for n, t in trials.items()}, start, str(path),
                                 "backfill", date)
    return rows, attrs


def sweep_presets(table: Path) -> dict[str, float]:
    """parameter -> preset value from a sweep's sensitivity.md table."""
    values = {}
    if table.is_file():
        for line in table.read_text().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3 and "." in cells[0]:
                try:
                    values[cells[0]] = float(cells[1])
                except ValueError:
                    continue
    return values


SWEEP_NAME = re.compile(r"sens-(?P<codec>[a-z0-9]+)-(?P<preset>.+)-(?P<key>\d{4}-\d{2}-\d{2})$")


def sweep_rows(tuning: Path, commits: dict[tuple, str]) -> list[dict]:
    rows = []
    for directory in sorted(tuning.glob("sens-*")):
        match = SWEEP_NAME.match(directory.name)
        store = directory / "results.json"
        if not match or not store.is_file():
            continue
        codec, preset, key = match["codec"], match["preset"], match["key"]
        rows += levers.sensitivity_rows(
            codec, preset, json.loads(store.read_text()),
            sweep_presets(directory / "sensitivity.md"), str(store), "backfill",
            file_date(store), commits.get((codec, preset, key)), key)
    return rows


def registered_names(binary: Path | None) -> set[str]:
    """Every parameter name the build registers (any codec), `nonbase.` forms included."""
    names: set[str] = set()
    for codec in CODECS:
        names |= set(levers.codec_defaults(binary, codec) or {})
    return names | {"nonbase." + n for n in names}


def doc_date(source: str) -> str | None:
    """The date of the doc's last commit (rows extracted without a stated date)."""
    path = source.split(":")[0]
    try:
        out = subprocess.run(["git", "-C", str(levers.REPO_ROOT), "log", "-1", "--format=%cs",
                              "--", path], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip() or None


def doc_rows(path: Path, binary: Path | None) -> list[dict]:
    """Rows extracted from the docs' tables (kind doc). An override whose name the
    build does not register (the extraction inferred it from the row's wording) is
    never stored as an override: the row keeps it in its change text only."""
    known = registered_names(binary)
    dates: dict[str, str | None] = {}
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("codec") not in CODECS:
            continue
        try:
            overrides = {k: float(v) for k, v in (raw.get("overrides") or {}).items()}
        except (TypeError, ValueError, AttributeError):
            overrides = {}
        change = raw.get("change")
        if overrides and (not known or set(overrides) - known):
            change = f"{change} (doc; unregistered: " + " ".join(
                f"{k}={v:g}" for k, v in overrides.items()) + ")"
            overrides = {}
        level = str(raw.get("level") or "?")
        source = raw.get("source") or str(path)
        if not raw.get("date"):
            dates.setdefault(source.split(":")[0], doc_date(source))
        fields = {f: raw.get(f) for f in levers.FIELDS if f in raw and f != "id"}
        fields.update(overrides=overrides, change=change, kind="doc", source=source,
                      level=level if level in ("L1", "L2", "L3") else f"doc:{level}",
                      base=raw.get("base") or levers.MEDIUM, base_params=None,
                      date=raw.get("date") or dates.get(source.split(":")[0]))
        rows.append(levers.make_row(**fields))
    return rows


def newest_build(paths) -> Path | None:
    builds = sorted((paths.data_root / "builds").glob("*/nanocodec"),
                    key=lambda p: p.stat().st_mtime)
    return builds[-1] if builds else None


def run(into: Path, doc_file: str | None) -> None:
    paths = load_paths()
    tuning = paths.data_root / "tuning"
    roots = [paths.work, tuning, paths.data_root / "results", paths.data_root / "worktrees"]
    rows = json_rows([r for r in roots if r.is_dir()])
    print(f"result files: {len(rows)} rows", flush=True)
    commits: dict[tuple, str] = {}
    study_count = 0
    for study in sorted(tuning.glob("*.sqlite")):
        try:
            found, attrs = study_rows(study)
        except sqlite3.DatabaseError as error:
            print(f"skipped {study}: {error}", flush=True)
            continue
        if found:
            study_count += 1
            rows += found
        if attrs.get("codec") and attrs.get("commit"):
            commits[(attrs["codec"], attrs.get("preset"), attrs.get("rotation"))] = attrs["commit"]
    print(f"tune studies: {study_count} studies, {len(rows)} rows so far", flush=True)
    rows += sweep_rows(tuning, commits)
    if doc_file:
        rows += doc_rows(Path(doc_file), newest_build(paths))
    added = levers.append(rows, into)
    print(f"backfill: {len(rows)} rows found, {added} new in {into}", flush=True)
