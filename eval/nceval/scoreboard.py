"""The scoreboard: where each codec and preset stands against its bar, on one page.

    python3 -m nceval scoreboard [--status docs/STATUS.md] [--over-budget]

Rows (CLAUDE.md, docs/ARCHITECTURE.md §1, PROCEDURES adoption bars): H.264 vs
openh264-baseline; HEVC and AV1, Default and Fast, vs SVT-AV1 M12 (rtc); VP9
vs our own HEVC and AV2 vs our own AV1 (Default rows; their bars are BD ~ 0
and BD <= 0 against those rows). For each row:

  - BD-rate on PSNR-YUV and native VMAF v1, per content class and overall, on
    VAL-150 (the natural VAL clips, first 150 frames, 10 s intra period),
    against the anchors' native curves (no decoding: nceval.gatepolicy);
  - host instructions per P frame and per key frame (callgrind, the
    geometric mean over tune.COST_CLIPS), the deterministic CPU measure;
  - the latest Jetson native-time ratio vs the anchor, read from
    `JETSON_FILE`, which the timing agents append to (one JSON object per
    line: {"date", "codec", "preset", "anchor", "ratio", "binary_sha256"?});
  - the worst clip (highest mean of the two BD-rates).

Context comparisons (Mark, 2026-11-08; GPL tools run as external binaries
only): every row is also compared with x264 ultrafast/veryfast and x265
ultrafast/superfast (single-threaded, zerolatency, no B-frames, fixed QP;
encoders/anchors.py), and every anchor's callgrind instructions per frame on
the same cost clips are listed, so CPU can be read against the bars.
openh264/SVT stay the primary bars.

Everything is cached: the curves are ordinary fingerprinted sweep results
(`nceval sweep --max-frames 150`). A new binary changes every row's
fingerprint, so each row first runs a canonical probe: 30-frame encodes of
one static and one motion clip at two QPs with that row's preset (seconds).
If those bitstreams are byte-identical to the ones behind the row's stored
curves, the curves are re-keyed to the new fingerprint instead of encoded
again (a change that touches neither clip at either QP can slip through; the
probe was chosen to exercise static skip and motion paths). Otherwise the
row is re-encoded on its previous QP grid (nceval.remeasure: no rate
probes). Instruction counts are cached by (binary hash, params, codec).
Anchors and our encoders are measured as separate budgeted phases. The page is
written to $NC_DATA_ROOT/reports/scoreboard-<date>.md, and the section between
STATUS_BEGIN and STATUS_END in the status file is replaced with its tables.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import report, tune
from .cachefile import write_atomic
from .clips import ClipSet, load_clipset
from .codecs import Codec
from .config import Paths, find_tool
from .encoders import resolve_encoder
from .encoders.base import EncodeJob
from .gatepolicy import EVALUATION_GATE, file_sha256
from .results import ResultsDb
from .rows import ROWS, Row
from .slots import cpu_slots, low_priority
from .remeasure import Remeasure, Source, stored_points
from .sweep import Sweep, SweepConfig

VAL_FRAMES = 150
CONTEXT_ANCHORS = ("x264-ultrafast", "x264-veryfast", "x265-ultrafast", "x265-superfast")
CANONICAL_CLIPS = ("static_cam29", "pan_jaad_0101")  # one static, one motion (VAL)
CANONICAL_FRAMES = 30
CANONICAL_QPS = {Codec.H264: (30, 42), Codec.HEVC: (30, 42), Codec.VVC: (30, 42), Codec.AV1: (100, 200),
                 Codec.VP9: (100, 200), Codec.AV2: (100, 200)}
METRICS = (("psnr_yuv", "PSNR"), ("vmaf_v1", "VMAF"))
STATUS_BEGIN = "<!-- scoreboard:begin -->"
STATUS_END = "<!-- scoreboard:end -->"


def jetson_file(paths: Paths) -> Path:
    return paths.data_root / "results" / "jetson-timing.jsonl"


def preset_key(codec: str, preset: str) -> str:
    """'hevc-fast', 'Fast', 'fast' -> 'fast'; the timing agents name presets freely."""
    preset = preset.lower()
    return preset.removeprefix(f"{codec}-") if preset != codec else "default"


def anchor_key(anchor: str) -> str:
    """'svtav1-m12-rtc' -> 'svtav1-m12': the timing log names anchors freely."""
    return anchor.removesuffix("-rtc")


def latest_jetson(paths: Paths) -> dict[tuple[str, str, str], dict]:
    """Per (codec, preset, anchor): the latest batch of per-clip Jetson ratios, summarised.

    The log is append-only; the latest batch is every record sharing the last
    record's binary hash (or date, if it has none). Its ratio is the geometric
    mean over that batch's clips.
    """
    records: dict[tuple[str, str, str], list[dict]] = {}
    try:
        lines = jetson_file(paths).read_text().splitlines()
    except FileNotFoundError:
        return {}
    for line in lines:
        try:
            record = json.loads(line)
            key = (record["codec"], preset_key(record["codec"], record["preset"]),
                   anchor_key(record.get("anchor", "")))
            float(record["ratio"])
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        records.setdefault(key, []).append(record)
    latest = {}
    for key, entries in records.items():
        last = entries[-1]
        batch_id = lambda r: r.get("binary_sha256") or r.get("date")
        batch = [r for r in entries if batch_id(r) == batch_id(last)]
        ratios = [float(r["ratio"]) for r in batch]
        latest[key] = {"ratio": math.exp(sum(math.log(x) for x in ratios) / len(ratios)),
                       "clips": len(batch), "date": last.get("date", "?"),
                       "anchor": last.get("anchor", "?")}
    return latest


@dataclass(frozen=True)
class Cost:
    instructions: tune.InstructionCost
    current: bool  # False: the latest count from an older binary (--no-measure)


def _cached_cost(directory: Path, codec: str, overrides: list) -> tune.InstructionCost | None:
    """The newest cached count for this codec and preset, whatever the binary."""
    best = None
    for path in directory.glob("*.json") if directory.is_dir() else ():
        record = json.loads(path.read_text())
        _, params, record_codec, *_ = json.loads(record["identity"])
        if record_codec == codec and params == json.loads(json.dumps(overrides)):
            if best is None or path.stat().st_mtime > best[0]:
                best = (path.stat().st_mtime, record)
    return tune.InstructionCost(best[1]["p"], best[1]["key"]) if best else None


def instruction_cost(row: Row, paths: Paths, measure_new: bool = True) -> Cost | None:
    """Callgrind P/key instructions, cached by (binary hash, params, codec).

    Without `measure_new` nothing is run: an uncached binary shows the newest
    count of an older binary, marked as such, or nothing.
    """
    adapter = resolve_encoder(row.encoder)
    binary = find_tool("nanocodec", paths)
    overrides = sorted(dict(getattr(adapter, "overrides", {})).items())
    identity = json.dumps([file_sha256(binary), overrides, row.codec.value,
                           tune.COST_CLIPS, tune.COST_P_FRAMES])
    directory = paths.data_root / "screen-cache" / "instructions"
    path = directory / f"{hashlib.sha256(identity.encode()).hexdigest()[:24]}.json"
    if path.is_file():
        record = json.loads(path.read_text())
        return Cost(tune.InstructionCost(record["p"], record["key"]), True)
    if not measure_new:
        older = _cached_cost(directory, row.codec.value, overrides)
        return Cost(older, False) if older else None
    overrides = dict(overrides)
    paths.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nc-scoreboard-", dir=paths.work) as work:
        cost = tune.instruction_cost(binary, row.codec, overrides, Path(work))
    write_atomic(path, json.dumps({"identity": identity, "p": cost.p, "key": cost.key}))
    return Cost(cost, True)


def canonical_signature(row: Row, clips: ClipSet, cfg: SweepConfig, paths: Paths) -> str:
    """SHA-256 over the row's canonical probe bitstreams (seconds of encoding)."""
    encoder = resolve_encoder(row.encoder)
    by_name = {c.name: c for c in clips.clips}
    digest = hashlib.sha256()
    # The preset is part of the identity: a preset change must never be
    # re-keyed even if it happens not to alter these particular probes.
    digest.update(json.dumps(sorted(getattr(encoder, "overrides", {}).items())).encode())
    paths.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nc-canonical-", dir=paths.work) as work:
        for name in CANONICAL_CLIPS:
            clip = by_name[name]
            for q in CANONICAL_QPS[row.codec]:
                stream = Path(work) / f"{name}-{q}{row.codec.bitstream_suffix}"
                job = EncodeJob(clip=clip, quantizer=q, structure=cfg.structure_for(clip),
                                frames=CANONICAL_FRAMES, bitstream=stream)
                with cpu_slots(1):
                    subprocess.run(low_priority(encoder.command(job, paths)),
                                   stdin=subprocess.DEVNULL, capture_output=True, check=True)
                digest.update(f"{name}:{q}:".encode() + stream.read_bytes())
    return digest.hexdigest()


def row_state_path(encoder: str, paths: Paths) -> Path:
    return paths.data_root / "screen-cache" / "scoreboard-rows" / f"{encoder}.json"


def reuse_identical(row: Row, signature: str, clips: ClipSet, sweep: Sweep, cfg: SweepConfig,
                    paths: Paths) -> int:
    """Re-key the row's stored curves when the canonical probe is unchanged."""
    try:
        state = json.loads(row_state_path(row.encoder, paths).read_text())
    except (FileNotFoundError, ValueError):
        return 0
    if state.get("signature") != signature:
        return 0
    encoder, relabelled = resolve_encoder(row.encoder), 0
    for clip in clips.clips:
        old = state["fingerprints"].get(clip.name)
        new = sweep.fingerprint(encoder, cfg, clip)
        frames = min(clip.frames, VAL_FRAMES)
        if old and old != new and sweep.db.curve_is_current(
                encoder.name, cfg.structure_tag, clip.name, frames, sweep.platform, old):
            sweep.db.relabel_curve(encoder.name, cfg.structure_tag, clip.name, frames,
                                   sweep.platform, old, new)
            relabelled += 1
    return relabelled


def save_row_state(row: Row, signature: str, clips: ClipSet, sweep: Sweep, cfg: SweepConfig,
                   paths: Paths) -> None:
    encoder = resolve_encoder(row.encoder)
    path = row_state_path(row.encoder, paths)
    write_atomic(path, json.dumps({"signature": signature, "fingerprints": {
        c.name: sweep.fingerprint(encoder, cfg, c) for c in clips.clips}}))


def val_150(paths: Paths) -> ClipSet:
    clipset = load_clipset(paths.clips / "main.json", paths.data_root)
    return ClipSet(clipset.name, tuple(c for c in clipset.clips if c.split == "val"
                                       and c.content_class not in report.SEPARATE_CLASSES))


def measure(paths: Paths, db: ResultsDb, clips: ClipSet, commit: str, over_budget: bool,
            rows: tuple[Row, ...] = ROWS) -> None:
    """Bring every row's and anchor's VAL-150 curves up to date.

    Anchors: a cached sweep (encoded once per anchor build). Our encoders: a
    curve measured before (an older binary) is re-encoded on its stored QP
    grid; only a curve never measured is swept with rate probes.
    """
    cfg = SweepConfig(intra_period_seconds=10, max_frames=VAL_FRAMES, gate=EVALUATION_GATE,
                      over_budget=over_budget)
    sweep = Sweep(paths, db, commit)
    # External anchors are swept here; our own encoders used as anchors (VP9
    # vs HEVC, AV2 vs AV1) are rows themselves and measured below.
    ours = {r.encoder for r in ROWS}
    external = [r.anchor for r in ROWS if r.anchor not in ours] + list(CONTEXT_ANCHORS)
    for anchor in dict.fromkeys(external):
        sweep.run([resolve_encoder(anchor)], list(clips.clips), cfg)
    source = [Source(paths.results_db, cfg.structure_tag)]
    for row in rows:  # one budgeted phase per row
        encoder = resolve_encoder(row.encoder)
        signature = canonical_signature(row, clips, cfg, paths)
        reused = reuse_identical(row, signature, clips, sweep, cfg, paths)
        if reused:
            print(f"{row.encoder}: canonical probe unchanged; {reused} curves re-keyed, "
                  "not encoded", flush=True)
        stale, new = [], []
        for clip in clips.clips:
            if db.curve_is_current(encoder.name, cfg.structure_tag, clip.name,
                                   min(clip.frames, VAL_FRAMES), sweep.platform,
                                   sweep.fingerprint(encoder, cfg, clip)):
                continue
            # A stored VAL-150 curve of an older build is re-encoded on its
            # grid; matching the length in the query (not on the first point
            # found) keeps full-length curves of the same clip from sending
            # it back to a rate-probe sweep.
            stored = stored_points(source, encoder.name, clip.name, min(clip.frames, VAL_FRAMES))
            (stale if stored else new).append(clip)
        if stale:
            runner = Remeasure(sweep, cfg, source)
            for line in runner.run([(encoder, c) for c in stale], over_budget):
                print(line, flush=True)
        if new:
            sweep.run([encoder], new, cfg)
        save_row_state(row, signature, clips, sweep, cfg, paths)


def curve_fingerprints(paths: Paths, db: ResultsDb, commit: str):
    """`fingerprint_of(encoder, clip)`: the configuration each scoreboard curve was measured with.

    A row's own curves: the fingerprints its state file recorded after the
    last measurement; anchors: their current sweep fingerprint. Other agents'
    sweeps of the same encoder name live beside them in the shared database
    and are never read by the scoreboard.
    """
    cfg = SweepConfig(intra_period_seconds=10, max_frames=VAL_FRAMES, gate=EVALUATION_GATE)
    sweep = Sweep(paths, db, commit)
    states, encoders = {}, {}

    def fingerprint_of(encoder: str, clip) -> str | None:
        if encoder not in states:
            try:
                states[encoder] = json.loads(
                    row_state_path(encoder, paths).read_text())["fingerprints"]
            except (FileNotFoundError, ValueError, KeyError):
                states[encoder] = {}
        if clip.name in states[encoder]:
            return states[encoder][clip.name]
        if encoder not in encoders:
            encoders[encoder] = resolve_encoder(encoder)
        return sweep.fingerprint(encoders[encoder], cfg, clip)
    return fingerprint_of


def bd_tables(db: ResultsDb, clips: ClipSet, fingerprint_of=None) -> dict:
    structure = SweepConfig(intra_period_seconds=10).structure_tag
    tables = {}
    for row in ROWS:
        entry = {"metrics": {}, "clips": {}}
        for metric, label in METRICS:
            rows = report.compute(db, clips, row.anchor, metric, structure, platform.machine(),
                                  [row.encoder], max_frames=VAL_FRAMES,
                                  fingerprint_of=fingerprint_of)
            summary = report.summarise(rows).get(row.encoder)
            entry["metrics"][label] = summary
            for clip_row in rows:
                if clip_row.result and clip_row.result.reliable:
                    entry["clips"].setdefault(clip_row.clip, {})[label] = \
                        clip_row.result.bd_rate_percent
        tables[row] = entry
    return tables


def context_tables(db: ResultsDb, clips: ClipSet, fingerprint_of=None) -> dict:
    """{row: {context anchor: {"PSNR": overall, "VMAF": overall}}} (None if unusable)."""
    structure = SweepConfig(intra_period_seconds=10).structure_tag
    tables = {}
    for row in ROWS:
        tables[row] = {}
        for anchor in CONTEXT_ANCHORS:
            cell = {}
            for metric, label in METRICS:
                rows = report.compute(db, clips, anchor, metric, structure, platform.machine(),
                                      [row.encoder], max_frames=VAL_FRAMES,
                                      fingerprint_of=fingerprint_of)
                summary = report.summarise(rows).get(row.encoder)
                cell[label] = summary.overall if summary else None
            tables[row][anchor] = cell
    return tables


def anchor_instruction_cost(name: str, paths: Paths,
                            measure_new: bool = True) -> tune.InstructionCost | None:
    """Callgrind P/key instructions of an external anchor on the cost clips, cached.

    Same method as our rows (tune.instruction_cost): one key frame, then
    COST_P_FRAMES more, at the middle of the anchor's quantizer range.
    """
    adapter = resolve_encoder(name)
    sample = adapter.command(EncodeJob(clip=_cost_clips(paths)[0], quantizer=0,
                                       structure=SweepConfig(10).template_structure(),
                                       frames=1, bitstream=Path("/x")), paths)
    identity = json.dumps([name, file_sha256(Path(sample[0])), tune.COST_CLIPS,
                           tune.COST_P_FRAMES])
    path = (paths.data_root / "screen-cache" / "anchor-instructions" /
            f"{hashlib.sha256(identity.encode()).hexdigest()[:24]}.json")
    if path.is_file():
        record = json.loads(path.read_text())
        return tune.InstructionCost(record["p"], record["key"])
    if not measure_new:
        return None
    lo, hi = adapter.quantizer_range
    per_frame, keys = [], []
    paths.work.mkdir(parents=True, exist_ok=True)
    for clip in _cost_clips(paths):
        totals = []
        for frames in (1, 1 + tune.COST_P_FRAMES):
            with tempfile.TemporaryDirectory(prefix="nc-anchor-cg-", dir=paths.work) as work:
                out = Path(work) / "callgrind.out"
                job = EncodeJob(clip=clip, quantizer=(lo + hi) // 2,
                                structure=SweepConfig(10).structure_for(clip), frames=frames,
                                bitstream=Path(work) / f"s{adapter.codec.bitstream_suffix}")
                with cpu_slots(1):
                    subprocess.run(low_priority(["valgrind", "--tool=callgrind",
                                                 f"--callgrind-out-file={out}",
                                                 *adapter.command(job, paths)]),
                                   stdin=subprocess.DEVNULL, capture_output=True, check=True)
                totals.append(next(int(line.split()[1]) for line in out.read_text().splitlines()
                                   if line.startswith(("summary:", "totals:"))))
        keys.append(totals[0] / 1e6)
        per_frame.append((totals[1] - totals[0]) / tune.COST_P_FRAMES / 1e6)
    cost = tune.InstructionCost(sum(per_frame) / len(per_frame), sum(keys) / len(keys))
    write_atomic(path, json.dumps({"identity": identity, "p": cost.p, "key": cost.key}))
    return cost


def _cost_clips(paths: Paths) -> list:
    clipset = load_clipset(paths.clips / "main.json", paths.data_root)
    by_name = {c.name: c for c in clipset.clips}
    return [by_name[name] for name in tune.COST_CLIPS]


def render_context(context: dict, anchor_costs: dict, jetson: dict | None = None) -> str:
    pct = lambda v: "n/a" if v is None else f"{v:+.1f}%"
    jetson = jetson or {}

    def timing(row: Row, anchor: str) -> str:
        record = jetson.get((row.codec.value, row.preset.lower(), anchor))
        return f"; Jetson {record['ratio']:.2f}x" if record else ""
    lines = ["", "### Context: vs x264 / x265 (PSNR / VMAF BD, VAL-150, and our/their Jetson "
             "CPU time where measured; not the bars)", "",
             "| Codec | Preset | " + " | ".join(CONTEXT_ANCHORS) + " |",
             "|" + "---|" * (2 + len(CONTEXT_ANCHORS))]
    for row, cells in context.items():
        lines.append(f"| {row.codec.value} | {row.preset} | " + " | ".join(
            f"{pct(cells[a]['PSNR'])} / {pct(cells[a]['VMAF'])}{timing(row, a)}"
            for a in CONTEXT_ANCHORS) + " |")
    lines += ["", "| Anchor | P / key instr. (M) |", "|---|---|"]
    lines += [f"| {name} | {cost.p:.1f} / {cost.key:.1f} |" for name, cost in anchor_costs.items()]
    return "\n".join(lines)


def worst_clip(entry: dict) -> str:
    scored = {clip: sum(v.values()) / len(v) for clip, v in entry["clips"].items()
              if len(v) == len(METRICS)}
    if not scored:
        return "n/a"
    clip = max(scored, key=scored.get)
    return f"{clip} {scored[clip]:+.1f}%"


def render(tables: dict, costs: dict, jetson: dict, classes: list[str], when: str,
           pending: dict[str, str] | None = None) -> str:
    pct = lambda v: "n/a" if v is None else f"{v:+.1f}%"
    lines = [f"## Scoreboard ({when}; VAL-150, native VMAF v1)", "",
             "| Codec | Preset | Anchor | PSNR | VMAF | " + " | ".join(classes) +
             " | P / key instr. (M) | Jetson ratio | Worst clip |",
             "|" + "---|" * (8 + len(classes))]
    for row, entry in tables.items():
        if pending and row.encoder in pending:
            lines.append(f"| {row.codec.value} | {row.preset} | {row.anchor} | pending: "
                         f"{pending[row.encoder]} |" + " |" * (5 + len(classes)))
            continue
        psnr, vmaf = entry["metrics"]["PSNR"], entry["metrics"]["VMAF"]
        per_class = [f"{pct(psnr.by_class.get(c) if psnr else None)} / "
                     f"{pct(vmaf.by_class.get(c) if vmaf else None)}" for c in classes]
        cost = costs.get(row)
        record = jetson.get((row.codec.value, row.preset.lower(), anchor_key(row.anchor)))
        lines.append(
            f"| {row.codec.value} | {row.preset} | {row.anchor} | "
            f"{pct(psnr.overall if psnr else None)} | {pct(vmaf.overall if vmaf else None)} | "
            + " | ".join(per_class) + " | "
            + (f"{cost.instructions.p:.1f} / {cost.instructions.key:.1f}"
               + ("" if cost.current else " (older binary)") if cost else "n/a") + " | "
            + (f"{record['ratio']:.2f}x ({record['clips']} clips, {record['date']})"
               if record else "n/a")
            + f" | {worst_clip(entry)} |")
    lines += ["", "BD-rate is vs the anchor (negative = fewer bits); per-class cells are "
              "PSNR / VMAF. Instructions: callgrind, host, per P and per key frame, geometric mean over "
              f"{', '.join(tune.COST_CLIPS)}. Jetson: geometric mean of the latest batch of per-clip native-time ratios in the timing log."]
    return "\n".join(lines)


def rewrite_status(status: Path, section: str) -> None:
    """Replace the scoreboard section of the status file (appended if absent)."""
    text = status.read_text() if status.is_file() else ""
    block = f"{STATUS_BEGIN}\n{section}\n{STATUS_END}"
    if STATUS_BEGIN in text and STATUS_END in text:
        head, rest = text.split(STATUS_BEGIN, 1)
        text = head + block + rest.split(STATUS_END, 1)[1]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    write_atomic(status, text)


def run(paths: Paths, db: ResultsDb, commit: str, status: Path | None,
        over_budget: bool = False, only: list[str] | None = None, measure_rows: bool = True,
        pending: dict[str, str] | None = None) -> Path:
    """Measure (all rows, or only these encoder names) and write the page.

    `measure_rows` False writes the page from what is stored (no encodes, no
    callgrind); `pending` rows (encoder -> note) show the note instead of numbers.
    """
    clips = val_150(paths)
    unknown = (set(only or ()) | set(pending or ())) - {r.encoder for r in ROWS}
    if unknown:
        raise SystemExit(f"unknown scoreboard rows: {', '.join(sorted(unknown))}")
    active = tuple(r for r in ROWS if (not only or r.encoder in only)
                   and r.encoder not in (pending or {}))
    if measure_rows:
        measure(paths, db, clips, commit, over_budget, active)
    fingerprint_of = curve_fingerprints(paths, db, commit)
    tables = bd_tables(db, clips, fingerprint_of)
    costs = {row: instruction_cost(row, paths, measure_rows) for row in ROWS
             if row.encoder not in (pending or {})}
    classes = sorted({c.content_class for c in clips.clips})
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    jetson = latest_jetson(paths)
    page = render(tables, costs, jetson, classes, when, pending)
    anchors = list(dict.fromkeys([r.anchor for r in ROWS if r.anchor not in
                                  {x.encoder for x in ROWS}] + list(CONTEXT_ANCHORS)))
    anchor_costs = {name: cost for name in anchors
                    if (cost := anchor_instruction_cost(name, paths, measure_rows))}
    page += "\n" + render_context(context_tables(db, clips, fingerprint_of), anchor_costs,
                                  jetson)
    out = paths.data_root / "reports" / f"scoreboard-{when.replace(' ', '-').replace(':', '')}.md"
    write_atomic(out, page + "\n")
    if status is not None:
        rewrite_status(status, page)
    print(page)
    return out
