"""The lever ledger: every measured change, its J gain and CPU cost, in one place.

One row per measured change (a parameter set or a code change against a base),
so a preset at any CPU charge k can be derived by replaying the data instead
of re-running it (owner, 2026-11-15):

    python3 -m nceval.levers replay --codec hevc --k 0.03 [--base medium]
    python3 -m nceval.levers table  --codec hevc [--base medium]
    python3 -m nceval.levers coverage
    python3 -m nceval.levers sync          # spool rows into the repo ledger
    python3 -m nceval.levers backfill      # nceval.levers_backfill: old results

Storage. The ledger is `eval/results/levers.jsonl` in the repo (append-only,
merge=union). Measurements run in many worktrees, most never landed, so the
tools append to the machine-wide spool `$NC_DATA_ROOT/results/levers.jsonl`
(NC_LEVERS_SPOOL overrides; tests point it at a temporary file); `sync`
copies its new rows into the repo ledger. Readers take the union.

Writers: nceval.screen (every mode), nceval.fixedqp ab/cost, the tune
studies (each trial, sensitivity point and finalist; nceval.tune_pool),
nceval.levels final-check and the standards lens (`nceval.ctc compare
--ledger BASE`) append when they finish (`record_*`). CPU from a timing
session (tools/dev/aws_timing.py's summary.json) goes in as cost rows with
`levers cost`; the read side joins it to the same change's quality rows.

Row schema (absent values are null; percentages are candidate vs base):

    id          sha256[:16] of the dedup key: codec, base, base_params, rate,
                overrides, change, level, draw, commit
    codec       h264 | hevc | av1 | vp9 | av2 | vvc
    base        the preset measured against (medium, fast, ...; "custom" when
                the anchor's settings match no preset)
    base_params anchor --param pairs beyond the base preset {name: value}
    base_values the anchor's value of every overridden parameter (staleness)
    rate        cbr | cqp | lens (the AOM CTC LD lens of nceval.ctc: 12 public
                sequences, six quantizers)
    overrides   the candidate's change {name: value}; {} for a code change
    change      text label ("code a1b2c3d->e4f5a6b", a sweep label, ...)
    level       L1 | L2 | L3 | final-check | tune-f1 | tune-f2 | tune-ladder |
                sensitivity | fixedqp | lens | cost | screen:<stage> | cqp:<stage> |
                doc:<as stated>
    draw        rotation key or clip set; n clips; frames per clip
    j lo hi     J_real quality delta (percent, CPU term NOT included) and its
                interval (90 %, nceval.interval). Lens rows: the change in the
                lens's J against x264 medium (candidate minus base, BD points)
    weighted    lens rows: the change in the AOM Weighted BD-rate against x264
                medium (candidate minus base, BD points)
    psnr vmaf msssim  per-metric BD-rate (percent): PSNR-YUV, VMAF v1, MS-SSIM
    guards      guard messages, [] when none tripped, null if not measured
    x86_pct x86_how jetson_pct jetson_how   the CPU change and how it was measured
    k verdict   the CPU charge the run decided at and its decision
    commit date source kind   binary commit, YYYY-MM-DD, origin file, and
                live | backfill | doc

CPU is joined on read: a row without CPU takes it from a cost row of the same
codec, base and overrides (`resolved`), with x86_how naming that row.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_paths
from .encoders.nanocodec import preset_values

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "eval" / "results" / "levers.jsonl"
SPOOL_ENV = "NC_LEVERS_SPOOL"
FIELDS = ("id", "codec", "base", "base_params", "base_values", "rate", "overrides", "change",
          "level", "draw", "n", "frames", "j", "lo", "hi", "psnr", "vmaf", "msssim", "weighted",
          "guards",
          "x86_pct", "x86_how", "jetson_pct", "jetson_how", "k", "verdict", "commit", "date",
          "source", "kind")
KEY_FIELDS = ("codec", "base", "base_params", "rate", "overrides", "change", "level", "draw",
              "commit")
MEDIUM = "medium"
#: Percent values are kept to 1e-4 (a thousandth of the smallest decision margin).
MEASURED = ("j", "lo", "hi", "psnr", "vmaf", "msssim", "weighted", "x86_pct", "jetson_pct")
MEASURE_DECIMALS = 4
LEVEL_NAMES = {"quick": "L1", "adopt": "L2", "final": "L3"}
#: Evidence order when one change has several rows: the first level present wins.
EVIDENCE = ("L3", "final-check", "L2", "tune-ladder", "tune-f2", "L1", "sensitivity",
            "tune-f1", "fixedqp", "lens")
#: Jetson CPU counts twice the x86 change (owner: Jetson primary, ~2:1; tune_pool).
JETSON_WEIGHT = 2.0
#: Replay decision, as nceval.interval decides L3: adopt below -ADOPT_MIN_GAIN_PCT
#: with the upper bound at most ADOPT_UPPER_PCT; reject when the bound is above 0.
ADOPT_MIN_GAIN_PCT = 0.05
ADOPT_UPPER_PCT = 0.3
#: A CPU change smaller than this (percent) counts as free: its break-even k is noise.
CPU_NOISE_PCT = 0.05
#: Values closer than this are the same setting (float round trips in JSON/TOML).
VALUE_TOLERANCE = 1e-9
CALLGRIND_P = "x86 callgrind instructions per P frame (tune.instruction_cost)"
CALLGRIND_TUNE = "x86 callgrind instructions, key + P frames amortised (tune_pool.cpu_cost)"


def spool_path() -> Path:
    override = os.environ.get(SPOOL_ENV)
    return Path(override) if override else load_paths().data_root / "results" / "levers.jsonl"


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def row_id(row: dict) -> str:
    key = json.dumps([row.get(f) for f in KEY_FIELDS], sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def make_row(**fields) -> dict:
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown ledger fields: {sorted(unknown)}")
    row = {f: fields.get(f) for f in FIELDS}
    row["base"] = preset_stem(row["base"], row["codec"] or "")
    row["base_params"] = dict(sorted((row["base_params"] or {}).items()))
    row["overrides"] = dict(sorted((row["overrides"] or {}).items()))
    row["date"] = row["date"] or today()
    row["id"] = row_id(row)
    return row


def compact(row: dict) -> dict:
    """The stored form: null fields left out, measured values to MEASURE_DECIMALS."""
    out = {}
    for name, value in row.items():
        if value is None:
            continue
        out[name] = round(value, MEASURE_DECIMALS) if name in MEASURED and \
            isinstance(value, float) else value
    return out


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            stored = json.loads(line)
            rows.append({f: stored.get(f) for f in FIELDS})
    return rows


def append(rows: list[dict], path: Path | None = None) -> int:
    """Append rows whose id the file lacks (under a lock); returns how many were new."""
    path = path or spool_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as out:
        fcntl.flock(out, fcntl.LOCK_EX)
        out.seek(0)
        seen = {json.loads(line)["id"] for line in out if line.strip()}
        new = []
        for row in rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                new.append(row)
        out.seek(0, os.SEEK_END)
        for row in new:
            out.write(json.dumps(compact(row), separators=(",", ":")) + "\n")
    return len(new)


def record(rows: list[dict], what: str) -> None:
    """Live append from a measuring tool: never fails the run, but says so loudly."""
    try:
        added = append(rows)
        print(f"levers: {added} new row(s) of {what} in {spool_path()}", flush=True)
    except (OSError, ValueError) as error:
        print(f"\n!!! levers: could not record {what}: {error}\n", file=sys.stderr, flush=True)


def load() -> list[dict]:
    """The union of the repo ledger and the spool, deduplicated by id."""
    rows: dict[str, dict] = {}
    for path in (LEDGER, spool_path()):
        for row in read_rows(path):
            rows.setdefault(row["id"], row)
    return list(rows.values())


# --- turning tool results into rows -------------------------------------------------------

def pairs_to_dict(pairs) -> dict[str, float]:
    """--param pairs (["a=1", ...]) or a {name: value} mapping, as {name: float}."""
    if isinstance(pairs, dict):
        return {k: float(v) for k, v in pairs.items()}
    out = {}
    for pair in pairs or []:
        name, _, value = str(pair).partition("=")
        out[name] = float(value)
    return out


def diff(test: dict[str, float], anchor: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in test.items() if k not in anchor or not same(anchor[k], v)}


def same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= VALUE_TOLERANCE * max(1, abs(a))


_DEFAULTS: dict[tuple[str, str], dict[str, float] | None] = {}


def codec_defaults(binary: Path | str | None, codec: str) -> dict[str, float] | None:
    """`nanocodec params --codec` defaults of a binary (None if it is gone)."""
    if not binary or not Path(binary).is_file():
        return None
    key = (str(Path(binary).resolve()), codec)
    if key not in _DEFAULTS:
        try:
            out = subprocess.run([str(binary), "params", "--codec", codec], capture_output=True,
                                 text=True, check=True, timeout=60).stdout
            _DEFAULTS[key] = {line.split("  ")[0]: float(line.split("  ")[1])
                              for line in out.splitlines()
                              if line.strip() and not line.startswith("#")}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            _DEFAULTS[key] = None
    return _DEFAULTS[key]


def presets_of(binary: Path | str | None) -> Path:
    directory = Path(binary).parent / "presets" if binary else None
    return directory if directory and directory.is_dir() else REPO_ROOT / "crates/nc-encoder/presets"


def name_base(codec: str, anchor: dict[str, float], presets: Path) -> tuple[str, dict]:
    """(preset stem, the anchor's params beyond it): the shipped preset the anchor's
    --param set contains exactly (medium: none), else ("custom", all of them)."""
    if not anchor:
        return MEDIUM, {}
    from .encoders.nanocodec import preset_values, shipped_presets
    try:
        stems = [s for s in shipped_presets(codec, presets) if s != MEDIUM]
        for stem in stems:
            values = preset_values(presets, codec, stem)
            if values and all(k in anchor and same(anchor[k], v) for k, v in values.items()):
                return stem, diff(anchor, values)
    except (OSError, ValueError, KeyError):
        pass
    return "custom", anchor


def base_values(binary, codec: str, anchor: dict[str, float],
                overrides: dict[str, float]) -> dict[str, float] | None:
    """The anchor's value of each overridden parameter (its params, else the binary's
    defaults); None if a value is unknown."""
    if not overrides:
        return None
    defaults = codec_defaults(binary, codec) or {}
    values = {}
    for name in overrides:
        base_name = name.removeprefix("nonbase.")
        value = anchor.get(name, defaults.get(base_name))
        if value is None:
            return None
        values[name] = value
    return values


def binary_commit(binary, recorded: dict | None = None) -> str | None:
    if recorded and recorded.get("commit"):
        return recorded["commit"]
    if binary and Path(binary).is_file():
        from . import buildinfo
        try:
            return buildinfo.info_of(Path(binary)).get("commit")
        except (OSError, subprocess.SubprocessError):
            return None
    return None


def side_change(binary: str, anchor_binary: str, commit: str | None,
                anchor_commit: str | None) -> str | None:
    if Path(binary).resolve() == Path(anchor_binary).resolve() or binary == anchor_binary:
        return None
    if commit and anchor_commit and commit != anchor_commit:
        return f"code {anchor_commit[:9]}->{commit[:9]}"
    return f"code {Path(anchor_binary).name}->{Path(binary).name}"


def balanced(summary: dict, term: str) -> float | None:
    value = summary.get(term)
    return value.get("balanced") if isinstance(value, dict) else None


def metrics_of(summary: dict) -> dict:
    """Per-metric BD-rates of a real-mode stage summary (nceval.jreal.summarise)."""
    return {"psnr": balanced(summary, "bd_psnr_yuv"), "vmaf": balanced(summary, "bd_vmaf_v1"),
            "msssim": balanced(summary, "bd_ms_ssim")}


def screen_rows(result: dict, source: str, kind: str = "live", date: str | None = None) -> list[dict]:
    """Rows of an nceval.screen result (real: levels or tracks; cqp proxy)."""
    if result.get("status") not in (None, "done") or "codec" not in result:
        return []
    codec, binary = result["codec"], result.get("binary")
    anchor_binary = result.get("anchor_binary") or binary
    builds = result.get("build_info") or {}
    commit = binary_commit(binary, builds.get("candidate"))
    anchor_commit = binary_commit(anchor_binary, builds.get("anchor")) if anchor_binary != binary \
        else commit
    test, anchor = pairs_to_dict(result.get("params")), pairs_to_dict(result.get("anchor_params"))
    overrides = diff(test, anchor)
    base, extra = name_base(codec, anchor, presets_of(anchor_binary))
    common = dict(codec=codec, base=base, base_params=extra, overrides=overrides,
                  base_values=base_values(anchor_binary, codec, anchor, overrides),
                  change=side_change(binary, anchor_binary, commit, anchor_commit),
                  commit=commit, date=date, source=source, kind=kind)
    if "levels" in result:
        return levels_rows(result, common)
    stages = result.get("stages") or []
    # Screens before the `mode` field were all the fixed-QP proxy (no J_real verdict).
    proxy = result.get("mode") == "cqp-proxy" or (
        result.get("mode") is None and not any("verdict" in s for s in stages))
    rate = "cqp" if proxy else "cbr"
    return [stage_row(stage, rate, result.get("clips"), common)
            for stage in last_stages(stages)]


def last_stages(stages: list[dict]) -> list[dict]:
    """The deciding (last) stage of each track: earlier proxy stages are nested in it."""
    by_track: dict[str, dict] = {}
    for stage in stages:
        if "stopped_early" not in stage:
            track = "val" if stage["stage"].startswith("val") else "main"
            by_track[stage.get("track", track)] = stage
    return list(by_track.values())


def stage_row(stage: dict, rate: str, clips, common: dict) -> dict:
    summary, verdict = stage.get("summary") or {}, stage.get("verdict") or {}
    if rate == "cqp":
        metric = lambda m: (summary.get(m) or {}).get("mean")
        j = None
        metrics = {"psnr": metric("psnr"), "vmaf": metric("vmaf_v1"), "msssim": None}
    else:
        j = verdict.get("quality")
        metrics = metrics_of(summary)
    n = next((v.get("clips", v.get("scored")) for v in summary.values()
              if isinstance(v, dict) and ("clips" in v or "scored" in v)), None)
    dcpu = verdict.get("dcpu_pct")
    return make_row(**common, rate=rate, level=f"{'cqp' if rate == 'cqp' else 'screen'}:"
                    f"{stage['stage']}", draw=clips, n=n or len(stage.get("per_clip") or {}),
                    frames=stage.get("frames"), j=j, **metrics, guards=verdict.get("guards"),
                    x86_pct=dcpu, x86_how="--dcpu given to the screen" if dcpu is not None
                    else None, k=verdict.get("k_cpu"), verdict=stage.get("decision"))


def levels_rows(result: dict, common: dict) -> list[dict]:
    """One row per pool level (L1/L2/L3): disjoint clips, each its own evidence."""
    draw = (result.get("draw") or {}).get("rotation")
    stages = {s["stage"]: s for s in result.get("stages") or []}
    rows = []
    for level, stats in (result.get("levels") or {}).items():
        stage = stages.get(level, {})
        verdict = stage.get("verdict") or {}
        dcpu, k = verdict.get("dcpu_pct"), verdict.get("k_cpu")
        shift = (k or 0.0) * (dcpu or 0.0)  # the interval includes k * dCPU; J here does not
        unshift = lambda v: None if v is None else v - shift
        rows.append(make_row(
            **common, rate="cbr", level=LEVEL_NAMES.get(level, level), draw=draw,
            n=stats.get("n"), frames=stage.get("frames") or (result.get("draw") or {}).get("frames"),
            j=unshift(stats.get("mean")), lo=unshift(stats.get("lo")), hi=unshift(stats.get("hi")),
            **metrics_of(stage.get("summary") or {}), guards=stats.get("guards"),
            x86_pct=dcpu, x86_how="--dcpu given to the screen" if dcpu is not None else None,
            k=k, verdict=stats.get("decision")))
    return rows


def final_check_rows(report: dict, source: str, kind: str = "live",
                     date: str | None = None) -> list[dict]:
    """Two builds (or a preset change) on L3, per codec: a code/preset change row.

    A preset-only comparison (one commit, a different presets/ beside each
    binary) is told apart by the candidate's preset values: the ones that
    differ from the anchor's are the row's overrides, and a hash of all of
    them goes into the change label (so a removed setting also changes the
    id). Without them every such check on one commit had the same id and
    all but the first were dropped as duplicates (2026-11-16)."""
    rows = []
    change = f"build {str(report.get('old_commit') or report.get('old'))[:9]}->" \
             f"{str(report.get('new_commit') or report.get('new'))[:9]}"
    for codec, result in (report.get("codecs") or {}).items():
        overrides, tag = preset_change(report, codec)
        common = dict(codec=codec, base=report.get("preset") or MEDIUM, overrides=overrides,
                      change=change + tag, commit=report.get("new_commit") or None, date=date,
                      source=source, kind=kind)
        for row in levels_rows(result, common):
            row["level"] = "final-check"
            row["id"] = row_id(row)
            rows.append(row)
    return rows


def preset_change(report: dict, codec: str) -> tuple[dict[str, float], str]:
    """The candidate preset's differing values and a change-label suffix naming
    its preset files, for a final-check whose arms' presets/ differ; ({}, "")
    otherwise. A presets/ dir that no longer exists is named by its path."""
    presets, stem = report.get("presets") or {}, report.get("preset")
    if not stem or not presets.get("new") or presets.get("new") == presets.get("old"):
        return {}, ""

    def values(directory: str | None) -> dict[str, float] | None:
        try:
            return preset_values(Path(directory), codec, stem) if directory else None
        except OSError:
            return None
    old, new = values(presets.get("old")), values(presets.get("new"))
    if old is None or new is None:
        return {}, f" presets {presets['new']}"
    if old == new:
        return {}, ""
    digest = hashlib.sha256(json.dumps(sorted(new.items())).encode()).hexdigest()[:8]
    return diff(new, old), f" presets {digest}"


def fixedqp_ab_rows(result: dict, source: str, kind: str = "live",
                    date: str | None = None) -> list[dict]:
    codec = result["codec"]
    anchor = pairs_to_dict(result.get("anchor_param"))
    base = result.get("preset") or MEDIUM
    rows = []
    for candidate in result.get("candidates") or []:
        if candidate.get("refused") or "summary" not in candidate:
            continue
        summary = candidate["summary"]
        overrides = diff(pairs_to_dict(candidate.get("param")), anchor)
        rows.append(make_row(
            codec=codec, base=base, base_params=anchor, rate="cqp", overrides=overrides,
            change=candidate.get("label"), level="fixedqp", draw=result.get("rotation"),
            n=summary.get("n"), frames=result.get("frames"), j=summary.get("j"),
            lo=summary.get("lo"), hi=summary.get("hi"), psnr=summary.get("psnr_yuv"),
            vmaf=summary.get("vmaf_v1"), msssim=summary.get("ms_ssim"),
            commit=result.get("binary_commit"), date=date, source=source, kind=kind,
            verdict=f"level {result.get('level')}"))
    return rows


def cost_rows(codec: str, base: str, costs: dict[str, tuple[dict, float]], how: str,
              commit: str | None, source: str, kind: str = "live",
              date: str | None = None, rate: str = "cqp") -> list[dict]:
    """CPU-only rows: {label: (overrides, dCPU percent)} measured against `base`."""
    return [make_row(codec=codec, base=base, rate=rate, overrides=overrides, change=label,
                     level="cost", x86_pct=dcpu, x86_how=how, commit=commit, date=date,
                     source=source, kind=kind)
            for label, (overrides, dcpu) in costs.items() if overrides or label != "anchor"]


def fixedqp_cost_rows(result: dict, labels: dict[str, list[str]], source: str,
                      kind: str = "live", date: str | None = None,
                      commit: str | None = None) -> list[dict]:
    """nceval.fixedqp cost: {label: {p_minstr, dcpu_pct}}; `labels` maps label -> pairs."""
    costs = {label: (pairs_to_dict(labels.get(label, [])), row["dcpu_pct"])
             for label, row in result["costs"].items()
             if label != "anchor" and label in labels}
    return cost_rows(result["codec"], result.get("preset") or MEDIUM, costs, CALLGRIND_P,
                     commit, source, kind, date)


def lens_row(codec: str, base: str, base_params: dict[str, float], overrides: dict[str, float],
             change: str, condition: str, n: int, weighted: float | None, j: float | None,
             commit: str | None, source: str, kind: str = "live",
             date: str | None = None) -> dict:
    """One standards-lens comparison: `weighted` and `j` are candidate minus base, in BD
    points against x264 medium (nceval.ctc compare)."""
    return make_row(codec=codec, base=base, base_params=base_params, rate="lens",
                    overrides=diff(overrides, base_params), change=change, level="lens",
                    draw=condition, n=n, weighted=weighted, j=j, commit=commit, date=date,
                    source=source, kind=kind)


def timing_cost_rows(summary: dict, codec: str, base: str, base_point: str,
                     points: dict[str, dict[str, float]], how: str, commit: str | None,
                     source: str, rate: str, date: str | None = None) -> list[dict]:
    """Cost rows from a timing session's summary ({point: {repeat-N: {fps}}}, as
    tools/dev/aws_timing.py writes it): each point's time against `base_point`'s, in
    percent, with the mean fps over the repeats. `points` maps point -> its overrides."""
    def fps(point: str) -> float:
        repeats = [r["fps"] for name, r in summary[point].items()
                   if name.startswith("repeat") and r.get("fps")]
        if not repeats:
            raise SystemExit(f"{source}: no timed repeat of {point}")
        return sum(repeats) / len(repeats)
    base_fps = fps(base_point)
    costs = {point: (overrides, (base_fps / fps(point) - 1) * 100)
             for point, overrides in points.items()}
    return cost_rows(codec, base, costs, how, commit, source, date=date, rate=rate)


def preset_stem(name: str | None, codec: str) -> str:
    """A study's preset name (default, av1-fast, h264-fastest, ...) as a ladder stem."""
    if not name or name in ("default", MEDIUM, "Default"):
        return MEDIUM
    stem = name.removeprefix(f"{codec}-")
    return {"fastest": "veryfast", "svc": "l2-medium", "svc-fast": "l2-fast"}.get(stem, stem)


def trial_row(attrs: dict, number: int, params: dict, start: dict, j: float, cpu: float | None,
              trial_attrs: dict, source: str, kind: str = "live",
              date: str | None = None) -> dict:
    """One tune study trial (nceval.tune_pool): a Pareto-search point vs the preset."""
    codec = attrs["codec"]
    overrides = diff({**attrs.get("preset_values", {}), **params},
                     {**attrs.get("preset_values", {}), **start})
    fidelity = trial_attrs.get("fidelity")
    stage = trial_attrs.get("stage_1") or {}
    stopped = fidelity == 1
    return make_row(
        codec=codec, base=preset_stem(attrs.get("preset"), codec), rate="cbr",
        overrides=overrides, base_values={n: start.get(n, attrs.get("preset_values", {}).get(n))
                                          for n in overrides} or None,
        change=f"{Path(source).stem}#{number}", level=f"tune-{'f' + str(fidelity) if isinstance(fidelity, int) else fidelity}",
        draw=attrs.get("rotation"), n=stage.get("n") if stopped else len(attrs.get("clips") or []),
        frames=attrs.get("frames"), j=j, lo=stage.get("lo") if stopped else None,
        hi=stage.get("hi") if stopped else None,
        x86_pct=None if cpu is None else 100 * (cpu - 1), x86_how=CALLGRIND_TUNE, k=attrs.get("k"),
        commit=attrs.get("commit"), date=date, source=source, kind=kind)


def finalist_rows(attrs: dict, finalists: dict, trials: dict[int, dict], start: dict,
                  source: str, kind: str = "live", date: str | None = None) -> list[dict]:
    """A study's L3-checked finalists: L3 J with x86 and Jetson CPU."""
    rows = []
    for number, result in finalists.items():
        level = result.get("l3")
        params = trials.get(int(number))
        if not level or params is None:
            continue
        row = trial_row(attrs, int(number), params, start, None, None, {}, source, kind, date)
        k = attrs.get("k") or 0.0
        shift = k * (result.get("dcpu") or 0.0)
        unshift = lambda v: None if v is None else v - shift
        row.update(level="L3", n=level.get("n"), j=unshift(level.get("mean")),
                   lo=unshift(level.get("lo")), hi=unshift(level.get("hi")),
                   guards=level.get("guards"), x86_pct=result.get("dcpu_x86"),
                   jetson_pct=result.get("dcpu_jetson"),
                   jetson_how="Jetson cycles (tune_pool.jetson_ratios)"
                   if result.get("dcpu_jetson") is not None else None,
                   verdict=level.get("decision"))
        row["id"] = row_id(row)
        rows.append(row)
    return rows


def sensitivity_rows(codec: str, preset: str, results: dict, preset_of: dict[str, float],
                     source: str, kind: str = "live", date: str | None = None,
                     commit: str | None = None, draw: str | None = None) -> list[dict]:
    """One-at-a-time sweep points (tune_pool.sensitivity): the cleanest gradients."""
    rows = []
    for key, r in results.items():
        if key.startswith("_") or r.get("j") is None:
            continue
        name = r["name"]
        rows.append(make_row(
            codec=codec, base=preset_stem(preset, codec), rate="cbr",
            overrides={name: r["value"]},
            base_values={name: preset_of[name]} if name in preset_of else None,
            level="sensitivity", draw=draw, n=r.get("n"), j=r["j"], lo=r.get("lo"),
            hi=r.get("hi"), x86_pct=100 * (r["cpu"] - 1) if r.get("cpu") else None,
            x86_how=CALLGRIND_TUNE, commit=commit, date=date, source=source, kind=kind))
    return rows


# --- reading: CPU join, staleness, replay -------------------------------------------------

def change_key(row: dict) -> tuple:
    return (row["codec"], row["base"], json.dumps(row["base_params"], sort_keys=True),
            json.dumps(row["overrides"], sort_keys=True),
            row["change"] if not row["overrides"] else None)


def combined_cpu(row: dict) -> float | None:
    x86, jet = row.get("x86_pct"), row.get("jetson_pct")
    if jet is None:
        return x86
    if x86 is None:
        return jet
    return (JETSON_WEIGHT * jet + x86) / (JETSON_WEIGHT + 1)


def resolved(rows: list[dict]) -> list[dict]:
    """J rows with CPU joined from another row of the same change: a cost-only row or
    any measured CPU of the same overrides on the same base (nearest date). The CPU of
    a parameter set does not depend on the rate mode or clips that judged its J."""
    costs: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("x86_pct") is not None and row["overrides"] and "joined" not in str(
                row.get("x86_how")):
            costs.setdefault(change_key(row), []).append(row)
    out = []
    for row in rows:
        if row["j"] is not None and combined_cpu(row) is None and change_key(row) in costs:
            match = min(costs[change_key(row)], key=lambda c: abs(
                datetime.fromisoformat(c["date"]) - datetime.fromisoformat(row["date"])))
            row = {**row, "x86_pct": match["x86_pct"],
                   "x86_how": f"joined: {match['x86_how']} ({match['source']})"}
        out.append(row)
    return out


def current_values(codec: str, base: str, binary: Path | None) -> dict[str, float] | None:
    """The current base's value of every parameter: the binary's defaults + the preset."""
    from .encoders.nanocodec import preset_values
    defaults = codec_defaults(binary, codec) if binary else None
    if defaults is None:
        return None
    try:
        preset = preset_values(REPO_ROOT / "crates/nc-encoder/presets", codec, base)
    except (OSError, ValueError, KeyError):
        return None
    return {**defaults, **preset}


def base_status(row: dict, current: dict[str, float] | None) -> str:
    """current | in-base (already adopted) | stale (the base moved) | unknown."""
    if not row["overrides"]:
        return "code" if row.get("change", "") and str(row["change"]).startswith(("code", "build")) \
            else "unknown"
    if current is None:
        return "unknown"
    if all(same(current.get(n.removeprefix("nonbase.")), v) for n, v in row["overrides"].items()):
        return "in-base"
    if row["base_params"] and any(not same(current.get(n), v) for n, v in row["base_params"].items()):
        return "stale"
    known = row.get("base_values")
    if not known:
        return "unknown"
    return "current" if all(same(current.get(n.removeprefix("nonbase.")), v)
                            for n, v in known.items()) else "stale"


def charged(row: dict, k: float) -> tuple[float | None, float | None, float | None]:
    """(J, lo, hi) + k * dCPU; None where unmeasured."""
    cpu = combined_cpu(row) or 0.0
    add = lambda v: None if v is None else v + k * cpu
    return add(row["j"]), add(row["lo"]), add(row["hi"])


def decide(row: dict, k: float) -> str:
    j, lo, hi = charged(row, k)
    if j is None:
        return "no-j"
    if combined_cpu(row) is None and k:
        return "no-cpu"
    if lo is None or hi is None:
        return "lead" if j < -ADOPT_MIN_GAIN_PCT else "trail" if j > ADOPT_MIN_GAIN_PCT else "flat"
    if j < -ADOPT_MIN_GAIN_PCT and hi <= ADOPT_UPPER_PCT:
        return "better"
    return "worse" if lo > 0 else "ambiguous"


def strongest(rows: list[dict]) -> dict:
    rank = lambda r: next((i for i, level in enumerate(EVIDENCE) if r["level"] == level),
                          len(EVIDENCE))
    return min(rows, key=lambda r: (rank(r), -(r.get("n") or 0), r["date"]))


def quality(row: dict) -> float | None:
    """The row's quality change: J, or for a lens row stated in Weighted only, Weighted."""
    return row["j"] if row["j"] is not None or row["rate"] != "lens" else row.get("weighted")


def label(row: dict) -> str:
    if row["overrides"]:
        return " ".join(f"{k}={v:g}" for k, v in row["overrides"].items())
    return str(row["change"])


def selected(args: argparse.Namespace) -> tuple[list[dict], dict | None]:
    rows = [r for r in resolved(load()) if r["codec"] == args.codec and quality(r) is not None
            and (args.base == "all" or r["base"] == args.base)
            and (args.rate == "all" or r["rate"] == args.rate)]
    binary = current_binary(args)
    current = current_values(args.codec, args.base, binary) if args.base != "all" else None
    if current is None and args.base != "all":
        print(f"(no current defaults: base status unknown; pass --binary)", file=sys.stderr)
    return rows, current


def current_binary(args: argparse.Namespace) -> Path | None:
    if args.binary and args.binary != "none":
        if Path(args.binary).is_file():
            return Path(args.binary)
        from . import levels
        return levels.build(args.binary, load_paths(), "levers")[0]
    return None


def replay(args: argparse.Namespace) -> None:
    rows, current = selected(args)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(change_key(row), []).append(row)
    listing = []
    for group in groups.values():
        best = strongest(group)
        status = base_status(best, current)
        if status == "stale" and not args.stale:
            continue
        j, lo, hi = charged(best, args.k)
        listing.append((j, decide(best, args.k), status, best, len(group)))
    listing.sort(key=lambda item: item[0])
    print(f"{args.codec} base {args.base}, k {args.k:g}: {len(listing)} changes "
          f"(J + k*dCPU, percent; stale rows {'shown' if args.stale else 'hidden'})")
    print(f"{'verdict':9s} {'J+kC':>7s} {'J':>7s} {'dCPU':>7s} {'level':12s} {'n':>3s} "
          f"{'base':8s} {'rows':>4s}  change")
    for j, verdict, status, row, count in listing:
        if args.only and verdict not in args.only.split(","):
            continue
        cpu = combined_cpu(row)
        fmt = lambda v: "    n/a" if v is None else f"{v:+7.2f}"
        print(f"{verdict:9s} {fmt(j)} {fmt(row['j'])} {fmt(cpu)} {row['level'][:12]:12s} "
              f"{row.get('n') or 0:3d} {status:8s} {count:4d}  {label(row)[:110]}")


def table(args: argparse.Namespace) -> None:
    """Each change's gradient: ΔJ, ΔCPU and its break-even k (-ΔJ/ΔCPU)."""
    rows, current = selected(args)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if combined_cpu(row) is not None:
            groups.setdefault(change_key(row), []).append(row)
    lines = []
    for group in groups.values():
        best = strongest(group)
        j, cpu = quality(best), combined_cpu(best)
        if abs(cpu) < CPU_NOISE_PCT:
            quadrant, breakeven = ("free gain" if j < 0 else "free loss"), None
        elif (j < 0) == (cpu < 0):
            quadrant, breakeven = ("win-win" if j < 0 else "lose-lose"), None
        else:
            quadrant, breakeven = ("quality arm" if j < 0 else "speed arm"), -j / cpu
        lines.append((quadrant, breakeven, best, base_status(best, current), j))
    order = {"win-win": 0, "free gain": 1, "quality arm": 2, "speed arm": 3, "free loss": 4,
             "lose-lose": 5}
    # Quality arms pay off while k < break-even (largest first); speed arms while k >
    # break-even (smallest first).
    lines.sort(key=lambda x: (order[x[0]], -x[1] if x[0] == "quality arm"
                              else x[1] if x[0] == "speed arm" else x[4]))
    print(f"{args.codec} base {args.base}: {len(lines)} changes with J and CPU. Quality arms "
          f"pay while k < k*, speed arms while k > k*.")
    lens = args.rate == "lens"
    if lens:
        print("lens rows: J and Weighted are BD points against x264 medium, candidate minus "
              "base (k* from J, or Weighted where J is not stated); W/C is Weighted points "
              "per 1 % of time.")
    print(f"{'quadrant':11s} {'k*':>7s} {'J':>7s} {'dCPU':>7s} "
          + (f"{'W':>7s} {'W/C':>7s} " if lens else "")
          + f"{'level':12s} {'n':>3s} {'base':8s}  change")
    for quadrant, breakeven, row, status, _ in lines:
        fmt = lambda v: "    n/a" if v is None else f"{v:+7.2f}"
        cpu = combined_cpu(row)
        weighted = row.get("weighted")
        per_cpu = None if weighted is None or abs(cpu) < CPU_NOISE_PCT else weighted / cpu
        print(f"{quadrant:11s} {fmt(breakeven)} {fmt(row['j'])} {fmt(cpu)} "
              + (f"{fmt(weighted)} {fmt(per_cpu)} " if lens else "")
              + f"{row['level'][:12]:12s} {row.get('n') or 0:3d} {status:8s}  {label(row)[:100]}")


def coverage(_: argparse.Namespace) -> None:
    rows = resolved(load())
    print(f"{len(rows)} rows ({LEDGER} + {spool_path()})")
    codecs = sorted({r["codec"] for r in rows})
    print(f"{'codec':6s} {'rows':>6s} {'J':>6s} {'CPU':>6s} {'J+CPU':>6s} {'no J':>6s} "
          f"{'no CPU':>6s} {'doc':>5s} {'changes':>8s}")
    for codec in codecs:
        mine = [r for r in rows if r["codec"] == codec]
        has_j = [r for r in mine if r["j"] is not None]
        has_cpu = [r for r in mine if combined_cpu(r) is not None]
        both = [r for r in has_j if combined_cpu(r) is not None]
        print(f"{codec:6s} {len(mine):6d} {len(has_j):6d} {len(has_cpu):6d} {len(both):6d} "
              f"{len(mine) - len(has_j):6d} {len(mine) - len(has_cpu):6d} "
              f"{sum(r['kind'] == 'doc' for r in mine):5d} "
              f"{len({change_key(r) for r in mine}):8d}")
    levels: dict[str, int] = {}
    for row in rows:
        levels[row["level"].split(":")[0]] = levels.get(row["level"].split(":")[0], 0) + 1
    print("levels: " + ", ".join(f"{k} {v}" for k, v in sorted(levels.items(), key=lambda x: -x[1])))


def param_file(path: str) -> dict[str, float]:
    """k=v lines (# comments and blanks skipped; the last value of a name wins)."""
    lines = [line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()]
    return pairs_to_dict([line.replace(" ", "") for line in lines if "=" in line])


def record_timing(args: argparse.Namespace) -> None:
    points = {}
    for spec in args.point:
        name, _, path = spec.partition("=")
        points[name] = param_file(path)
    record(timing_cost_rows(json.loads(Path(args.summary).read_text()), args.codec, args.base,
                            args.base_point, points, args.how, args.commit, args.summary,
                            args.rate), "timing costs")


def sync(_: argparse.Namespace) -> None:
    added = append(read_rows(spool_path()), LEDGER)
    print(f"{added} new row(s) from {spool_path()} into {LEDGER}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python3 -m nceval.levers",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "table"):
        p = sub.add_parser(name)
        p.add_argument("--codec", required=True)
        p.add_argument("--base", default=MEDIUM, help="preset stem the rows were measured "
                       "against (default medium, i.e. Default); 'all' for every base")
        p.add_argument("--rate", default="cbr", choices=("cbr", "cqp", "lens", "all"))
        p.add_argument("--binary", default="HEAD",
                       help="current build (a path, or a commit built once into "
                       "builds/; default HEAD) whose defaults and presets decide which rows' "
                       "bases are stale; 'none' skips the check")
    replay_parser = sub.choices["replay"]
    replay_parser.add_argument("--k", type=float, required=True, help="CPU charge per 1%% CPU")
    replay_parser.add_argument("--stale", action="store_true", help="also list stale rows")
    replay_parser.add_argument("--only", help="verdicts to list, e.g. better,lead")
    sub.add_parser("coverage")
    sub.add_parser("sync")
    cost = sub.add_parser("cost", help="cost rows from a timing session's summary.json "
                          "(tools/dev/aws_timing.py)")
    cost.add_argument("--summary", required=True)
    cost.add_argument("--codec", required=True)
    cost.add_argument("--base", required=True, help="preset stem of the base point")
    cost.add_argument("--base-point", required=True, help="the summary's name of the base")
    cost.add_argument("--point", action="append", required=True,
                      help="POINT=PARAMFILE: a timed point and its overrides (k=v lines)")
    cost.add_argument("--how", required=True, help="what was timed, e.g. 'c8a.large chart "
                      "path (aws_timing), mean fps of 2 repeats'")
    cost.add_argument("--rate", default="lens", choices=("cbr", "cqp", "lens"),
                      help="the rows' rate mode (CPU joins across modes on read)")
    cost.add_argument("--commit", help="the timed build's commit")
    backfill = sub.add_parser("backfill", help="nceval.levers_backfill: every old result")
    backfill.add_argument("--doc-rows", help="JSONL of rows extracted from docs (kind doc)")
    backfill.add_argument("--into", default=str(LEDGER), help="ledger to append to")
    backfill.add_argument("--docs-only", action="store_true",
                          help="append only --doc-rows (no rescan of the result files)")
    args = parser.parse_args(argv)
    if args.command == "backfill":
        from . import levers_backfill
        levers_backfill.run(Path(args.into), args.doc_rows, args.docs_only)
        return
    {"replay": replay, "table": table, "coverage": coverage, "sync": sync,
     "cost": record_timing}[args.command](args)


if __name__ == "__main__":
    main()
