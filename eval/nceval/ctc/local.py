"""Run the standards lens on this machine for one encoder variant, and compare with stored curves.

    python3 -m nceval.ctc local --source-dir DIR --run-dir DIR --encoder nanocodec-av1
        [--label NAME] [--param k=v ...] [--conditions aom-ld] [--sequences A,B]
        [--kinds qp,cbr] [--x264 qp|cbr|psnr-qp|psnr-cbr] [--workers 4]
        [--references DIR] [--skip ID,...]
    python3 -m nceval.ctc compare --results DIR[,DIR...] [--points CSV[,CSV...]]
        [--anchor x264-medium] [--paired-with ENCODER] [--stored-tests ENC,...] [--json OUT]
        [--ledger BASE]

`local` is the lens's iteration loop for encoder work: our encoders are fast,
and the anchors' curves are already measured (the results page's
points.csv), so only the variant under test is coded. It stages a run
directory exactly as `stage` does (the same jobs, command lines and worker)
with three differences:

- our streams are not decoded: evaluation scores the encoder's shown
  pictures, and conformance is the landing check (docs/PROCEDURES.md §2);
- `--param` pairs are appended to our command, and the variant is recorded
  under `<encoder>+<label>`, so variants and the stored Default sit side by
  side in one comparison;
- x264 can also run in the chart's modes (`--x264`): its own CBR (ABR with a
  VBV of the shared 1 s buffer, `nceval.cbr`) at the rates its fixed-QP
  points reached, and `--tune psnr`. That is how the gap between this lens
  and our chart is decomposed (docs/analysis/standards-lens-improvement-2026-11-14.md).

Any other anchor of the condition (`--encoder x265-slow`) runs its staged
fixed-QP jobs unchanged, so a curve the stored results lack can be added
locally (HM and VVenC from `--references`, the directory `stage` takes).
`--skip` leaves out named jobs: a re-run that adds XPSNR to a stored curve
codes only the points that curve has (HM's WestWindEasy QP 22 never finished
within a launch's limit).

Sources: raw I420 files named `<sequence>.yuv` in `--source-dir` with at
least the condition's frames (`worker.py prepare`, or any download of the
URLs in `sequences.py` cut to length).

`compare` prints, per test curve (encoder and kind), the mean BD-rate over
sequences against the anchor, for the AOM Weighted figure, our J and its
three terms; with `--paired-with`, also against that curve directly (a
variant against its baseline: the same sequences and points).

`--ledger BASE` (an encoder curve, e.g. `nanocodec-av1@faster` or a local
variant) also records each of our other test curves in the lever ledger
(`nceval.levers`, level `lens`): its Weighted and J against x264 medium minus
BASE's, with the overrides taken from the variant records `local` leaves
beside the results (`variants.jsonl`: encoder, parameters, build commit).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean

from ..config import load_paths
from ..cbr import BUFFER_MS
from . import plan as planner
from .conditions import CONDITIONS
from .report import (BD_WEIGHTS, J_METRICS, REPORTED_METRICS, Curves, Point, bd, bd_j,
                     load_points)
from ..metrics import vmafx_tool

DEFAULT_POINTS = Path(__file__).parents[3] / "docs/results/standards-ctc-2026-11-14/points.csv"
WORKER = Path(__file__).with_name("worker.py")
VMAF_V1_MODEL = Path("share/vmaf/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json")
#: x264's chart modes: (CBR, metric tune). `qp` is the lens's own setting.
X264_MODES = {"qp": (False, None), "cbr": (True, None), "psnr-qp": (False, "psnr"),
              "psnr-cbr": (True, "psnr")}
X264 = "x264-medium"
MS_PER_SECOND = 1000
#: The figures `compare` prints, in order.
FIGURES = ("Weighted", "J", *J_METRICS, *REPORTED_METRICS)
#: Written by `local` beside the result records (not *.json: load_points reads those).
VARIANTS_FILE = "variants.jsonl"


def variant_name(encoder: str, label: str) -> str:
    return f"{encoder}+{label}" if label else encoder


def renamed(job: dict, encoder: str) -> dict:
    """`job` recorded under another encoder name (its id and bitstream follow)."""
    condition, sequence, _, point = job["id"].split("/")
    key = f"{condition}/{sequence}/{encoder}/{point}"
    bitstream = str(planner.STREAMS / f"{key}{Path(job['bitstream']).suffix}")
    encode = [bitstream if a == job["bitstream"] else a for a in job["encode"]]
    return {**job, "id": key, "encoder": encoder, "bitstream": bitstream, "encode": encode,
            **({"rate_of": f"{condition}/{sequence}/{encoder}/q{job['point']}"}
               if job["kind"] == "cbr" else {})}


def own_variant(job: dict, params: list[str], label: str) -> dict:
    """Our job with `params` added, not decoded, under the variant's name."""
    encode = list(job["encode"])
    for pair in params:
        encode += ["--param", pair]
    return renamed({**job, "encode": encode, "decoders": []}, variant_name(job["encoder"], label))


def with_metrics(jobs: list[dict], metrics: str) -> list[dict]:
    return [{**job, "metrics": metrics} for job in jobs]


def x264_variant(job: dict, mode: str) -> list[dict]:
    """The x264 job in one of the chart's modes: [fixed QP] or [fixed QP, CBR twin]
    (the CBR twin's target is the rate the fixed-QP point reached, as for ours)."""
    cbr, tune = X264_MODES[mode]
    encode = list(job["encode"])
    if tune:
        at = encode.index("--tune") + 1
        encode[at] = f"{encode[at]},{tune}"
    name = variant_name(job["encoder"], "" if mode == "qp" else f"{tune or 'default'}")
    fixed = renamed({**job, "encode": encode}, name)
    if not cbr:
        return [fixed]
    at = encode.index("--qp")
    # The VBV holds BUFFER_MS of the target: with the shared 1 s buffer its size in
    # kbit is the target in kbps, so the worker's one placeholder fills both.
    assert BUFFER_MS == MS_PER_SECOND
    rate = ["--bitrate", "{KBPS}", "--vbv-maxrate", "{KBPS}", "--vbv-bufsize", "{KBPS}"]
    twin = renamed({**job, "kind": "cbr", "id": job["id"].replace("/q", "/cbr"),
                    "encode": [*encode[:at], *rate, *encode[at + 2:]]}, name)
    return [fixed, twin]


def local_jobs(args: argparse.Namespace, source_dir: Path) -> list[dict]:
    """The jobs of one local run; an encoder, kind or sequence that expands to none is an error
    (a misspelt rung or anchor once ran as an empty, "finished" run)."""
    jobs = _local_jobs(args, source_dir)
    if not jobs:
        raise SystemExit(f"{args.encoder} (kinds {args.kinds}, conditions {args.conditions}, "
                         f"sequences {args.sequences or 'all'}): no jobs; the lens's encoders "
                         "are nceval.ctc.conditions.ALL_ENCODERS")
    return jobs


def _local_jobs(args: argparse.Namespace, source_dir: Path) -> list[dict]:
    jobs = []
    for name in args.conditions.split(","):
        condition = CONDITIONS[name]
        sequences = args.sequences.split(",") if args.sequences else None
        condition = replace(condition, encoders=tuple(e for e in condition.encoders
                                                      if e.name == args.encoder),
                            sequences=tuple(s for s in condition.sequences
                                            if sequences is None or s.name in sequences))
        jobs += planner.jobs_for(condition, load_paths(), source_dir)
    kinds = set(args.kinds.split(","))
    if args.encoder == X264:
        return [v for j in jobs if j["kind"] == "qp" for v in x264_variant(j, args.x264)]
    if not args.encoder.startswith("nanocodec-"):  # another anchor: its staged jobs as they are
        if args.param or args.label:
            raise SystemExit(f"{args.encoder}: --param and --label are for our encoders")
        return [j for j in jobs if j["kind"] == "qp"]
    return [own_variant(j, args.param, args.label) for j in jobs if j["kind"] in kinds]


def tool_paths(references: Path | None = None) -> dict[str, Path]:
    """The anchors' tools and ours, plus the reference software's bin/ and cfg/ files
    where `references` is given (the layout `stage --references` takes)."""
    paths = load_paths()
    tools = {p.name: p for p in (paths.anchors / "bin").iterdir()}
    tools["nanocodec"] = paths.nanocodec_bin / "nanocodec"
    if references is not None:
        tools.update({p.name: p for d in ("bin", "cfg") for p in (references / d).iterdir()})
    return tools


def xpsnr_tool() -> Path | None:
    """The VMAFx tool that scores XPSNR, or None (said loudly) where this machine has
    none, e.g. the build box: Weighted and J do not read XPSNR, so the lens still runs."""
    try:
        return vmafx_tool(load_paths())
    except FileNotFoundError as missing:
        print(f"lens: XPSNR not scored ({missing})", file=sys.stderr, flush=True)
        return None


def local(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    jobs = with_metrics(local_jobs(args, Path(args.source_dir).resolve()), args.metrics)
    skipped = set(args.skip.split(",")) if args.skip else set()
    unknown = skipped - {job["id"] for job in jobs}
    if unknown:
        raise SystemExit(f"--skip: no such job: {', '.join(sorted(unknown))}")
    jobs = [job for job in jobs if job["id"] not in skipped]
    tools = tool_paths(Path(args.references) if args.references else None)
    if not run_dir.exists():
        planner.stage(run_dir, jobs, tools, [WORKER, load_paths().anchors / VMAF_V1_MODEL],
                      xpsnr_tool())
    print(f"{len(jobs)} jobs in {run_dir}", flush=True)
    subprocess.run([sys.executable, "-I", WORKER.name, "run", "--workers", str(args.workers)],
                   cwd=run_dir, check=True)
    if args.encoder.startswith("nanocodec-"):
        write_variant(run_dir / "out" / "results", args, tools["nanocodec"])


def write_variant(results: Path, args: argparse.Namespace, binary: Path) -> None:
    """The variant's name, base encoder, parameters and build, for `compare --ledger`."""
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                             check=True).stdout.split()
    record = {"variant": variant_name(args.encoder, args.label), "encoder": args.encoder,
              "params": args.param, "commit": version[-1] if version else None}
    results.mkdir(parents=True, exist_ok=True)
    with open(results / VARIANTS_FILE, "a") as out:
        out.write(json.dumps(record) + "\n")


def read_variants(results: list[Path]) -> dict[str, dict]:
    """{variant name: record} from every results directory's variants.jsonl."""
    variants = {}
    for directory in results:
        path = directory / VARIANTS_FILE
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    variants[record["variant"]] = record
    return variants


def ledger_rows(curves: Curves, tests: list[tuple[str, str, str]], anchor: tuple[str, str],
                base: tuple[str, str], variants: dict[str, dict], source: str) -> list[dict]:
    """Lever-ledger rows of our test curves against `base` (nceval.levers.lens_row)."""
    from .. import levers
    rows = []
    for condition, encoder, kind in tests:
        if (encoder, kind) == base or not encoder.startswith("nanocodec-") \
                or encoder not in variants:
            continue
        record, base_record = variants[encoder], variants.get(base[0], {})
        codec, _, preset = record["encoder"].removeprefix("nanocodec-").partition("@")
        base_codec, _, base_preset = base_record.get(
            "encoder", base[0]).removeprefix("nanocodec-").partition("@")
        if base_codec != codec or (base_preset or levers.MEDIUM) != (preset or levers.MEDIUM):
            print(f"ledger: {encoder} is not a variant of {base[0]}'s preset; not recorded",
                  file=sys.stderr)
            continue
        test_means, n, _ = mean_figures(curves, condition, anchor, (encoder, kind))
        base_means, _, _ = mean_figures(curves, condition, anchor, base)
        delta = lambda f: None if test_means[f] is None or base_means[f] is None \
            else round(test_means[f] - base_means[f], 2)
        rows.append(levers.lens_row(
            codec, preset or levers.MEDIUM, levers.pairs_to_dict(base_record.get("params")),
            levers.pairs_to_dict(record["params"]), encoder.split("+", 1)[-1], condition, n,
            delta("Weighted"), delta("J"), record.get("commit"), source))
    return rows


def stored_points(path: Path) -> list[Point]:
    """Points from a results page's points.csv."""
    quality_columns = None
    points = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            quality_columns = quality_columns or [c for c in row if c not in
                                                  ("condition", "sequence", "encoder", "kind",
                                                   "q", "kbps", "user_s")]
            points.append(Point(row["condition"], row["sequence"], row["encoder"], row["kind"],
                                int(row["q"]), float(row["kbps"]), float(row["user_s"]),
                                {c: float(row[c]) for c in quality_columns if row[c] != ""}))
    return points


def figures(anchor: list[Point], test: list[Point]) -> dict[str, float | None]:
    """One sequence's BD-rates for FIGURES (None: no reliable overlap)."""
    planes = [bd(anchor, test, p) for p in BD_WEIGHTS]
    values = {"Weighted": sum(w * r.value for w, r in zip(BD_WEIGHTS.values(), planes))
              if all(r.value is not None and r.reliable for r in planes) else None}
    j = bd_j(anchor, test)
    values["J"] = j.value if j.reliable else None
    for metric in J_METRICS:
        r = bd(anchor, test, metric)
        values[metric] = r.value if r.reliable else None
    for metric in REPORTED_METRICS:  # None where either curve was not scored for it
        r = bd(anchor, test, metric) if all(metric in p.quality for p in anchor + test) else None
        values[metric] = r.value if r is not None and r.reliable else None
    return values


def mean_figures(curves: Curves, condition: str, anchor: tuple[str, str],
                 test: tuple[str, str]) -> tuple[dict[str, float | None], int, float]:
    """Mean over sequences of each figure, the sequence count and the user-time ratio."""
    per_sequence, times = [], []
    for sequence in CONDITIONS[condition].sequences:
        a = curves.get(condition, sequence.name, *anchor)
        t = curves.get(condition, sequence.name, *test)
        if a and t:
            per_sequence.append(figures(a, t))
            times.append(sum(p.user_s for p in t) / max(sum(p.user_s for p in a), 1e-9))
    means = {f: (mean(vs) if (vs := [s[f] for s in per_sequence if s[f] is not None]) else None)
             for f in FIGURES}
    return means, len(per_sequence), mean(times) if times else float("nan")


def reported_counts(curves: Curves, condition: str, anchor: tuple[str, str],
                    test: tuple[str, str]) -> dict[str, int]:
    """Per reported metric, the sequences its mean is over (`<metric> n`): the ones
    where it can be computed, as the CBR charts' BD-XPSNR (chart.reported_pair)."""
    counts = dict.fromkeys(REPORTED_METRICS, 0)
    for sequence in CONDITIONS[condition].sequences:
        a = curves.get(condition, sequence.name, *anchor)
        t = curves.get(condition, sequence.name, *test)
        if a and t:
            values = figures(a, t)
            for metric in REPORTED_METRICS:
                counts[metric] += values[metric] is not None
    return {f"{metric} n": n for metric, n in counts.items()}


def encode_fps(curves: Curves, condition: str, test: tuple[str, str]) -> float | None:
    """Frames per CPU second over every point of the curve (user time where it was coded)."""
    frames = CONDITIONS[condition].frames
    points = [p for s in CONDITIONS[condition].sequences
              for p in curves.get(condition, s.name, *test)]
    seconds = sum(p.user_s for p in points)
    return round(frames * len(points) / seconds, 3) if seconds > 0 else None


def curve_key(spec: str, kind: str) -> tuple[str, str]:
    """`ENCODER` or `ENCODER:KIND` as a curve's (encoder, kind), `kind` by default."""
    encoder, _, named = spec.partition(":")
    return encoder, named or kind


def compare(args: argparse.Namespace) -> None:
    measured = load_points([Path(d) for d in args.results.split(",")])
    tests = sorted({(p.condition, p.encoder, p.kind) for p in measured})
    # A curve measured here replaces the stored one of the same name (a local
    # re-run of the Default on a newer build is the variants' baseline).
    replaced = {(p.condition, p.sequence, p.encoder, p.kind) for p in measured}
    stored = [p for path in args.points.split(",") for p in stored_points(Path(path))]
    curves = Curves(measured + [p for p in stored if (p.condition, p.sequence, p.encoder, p.kind)
                                not in replaced])
    if args.stored_tests:  # stored curves reported beside the measured ones
        conditions = sorted({t[0] for t in tests})
        tests += [(c, *curve_key(spec, "qp")) for c in conditions
                  for spec in args.stored_tests.split(",")]
    header = " | ".join(FIGURES)
    rows = []
    known = {(c, e, k) for c, _, e, k in curves.groups}
    absent = [f"{e} {k}" for c, e, k in tests if (c, e, k) not in known]
    if absent:
        raise SystemExit(f"no points for: {', '.join(absent)}")
    for condition, encoder, kind in tests:
        bars = [curve_key(args.anchor, "qp")]
        if args.paired_with:
            bars.append(curve_key(args.paired_with, kind))
        for bar in bars:
            means, n, time = mean_figures(curves, condition, bar, (encoder, kind))
            cells = " | ".join("—" if v is None else f"{v:+.1f}" for v in means.values())
            print(f"{condition} | {encoder} {kind} vs {bar[0]} {bar[1]} | {cells} | n={n} "
                  f"| time {time:.2f}x   ({header})")
            rows.append({"condition": condition, "test": encoder, "kind": kind,
                         "anchor": bar[0], "anchor_kind": bar[1], "n": n,
                         "fps": encode_fps(curves, condition, (encoder, kind)),
                         **{f: None if v is None else round(v, 2) for f, v in means.items()},
                         **reported_counts(curves, condition, bar, (encoder, kind))})
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
    if args.ledger:
        from .. import levers
        results = [Path(d) for d in args.results.split(",")]
        levers.record(ledger_rows(curves, tests, curve_key(args.anchor, "qp"),
                                  curve_key(args.ledger, "qp"), read_variants(results),
                                  args.results), "lens comparisons")


def main_local(sub) -> None:
    p = sub.add_parser("local", help="code one variant on this machine")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--param", action="append", default=[], help="k=v, repeatable")
    p.add_argument("--conditions", default="aom-ld")
    p.add_argument("--sequences")
    p.add_argument("--kinds", default="qp", help="our curves: qp, cbr or qp,cbr")
    p.add_argument("--x264", default="qp", choices=sorted(X264_MODES))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--metrics", default="lean", choices=("lean", "ctc"),
                   help="lean: only what Weighted and J read (worker.py vmaf_process)")
    p.add_argument("--references", help="bin/ and cfg/ of the reference software (HM, VVenC)")
    p.add_argument("--skip", help="job ids (condition/sequence/encoder/qN) not to code")
    p.set_defaults(handler=local)
    c = sub.add_parser("compare", help="BD-rates of local results against stored curves")
    c.add_argument("--results", required=True)
    c.add_argument("--points", default=str(DEFAULT_POINTS))
    c.add_argument("--anchor", default=X264, help="ENCODER or ENCODER:KIND (default kind qp)")
    c.add_argument("--paired-with", help="ENCODER or ENCODER:KIND")
    c.add_argument("--stored-tests", help="ENCODER[:KIND],...: stored curves to report too")
    c.add_argument("--json", help="also write the rows here (one per test and bar)")
    c.add_argument("--ledger", help="ENCODER[:KIND]: record our variants against this base "
                   "curve in the lever ledger (nceval.levers, level lens)")
    c.set_defaults(handler=compare)

