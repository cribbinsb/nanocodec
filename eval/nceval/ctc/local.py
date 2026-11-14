"""Run the standards lens on this machine for one encoder variant, and compare with stored curves.

    python3 -m nceval.ctc local --source-dir DIR --run-dir DIR --encoder nanocodec-av1
        [--label NAME] [--param k=v ...] [--conditions aom-ld] [--sequences A,B]
        [--kinds qp,cbr] [--x264 qp|cbr|psnr-qp|psnr-cbr] [--workers 4]
    python3 -m nceval.ctc compare --results DIR[,DIR...] [--points CSV[,CSV...]]
        [--anchor x264-medium] [--paired-with ENCODER]

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

Sources: raw I420 files named `<sequence>.yuv` in `--source-dir` with at
least the condition's frames (`worker.py prepare`, or any download of the
URLs in `sequences.py` cut to length).

`compare` prints, per test curve (encoder and kind), the mean BD-rate over
sequences against the anchor, for the AOM Weighted figure, our J and its
three terms; with `--paired-with`, also against that curve directly (a
variant against its baseline: the same sequences and points).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean

from ..config import load_paths
from ..cbr import BUFFER_MS
from . import plan as planner
from .conditions import CONDITIONS
from .report import BD_WEIGHTS, J_METRICS, Curves, Point, bd, bd_j, load_points

DEFAULT_POINTS = Path(__file__).parents[3] / "docs/results/standards-ctc-2026-11-14/points.csv"
WORKER = Path(__file__).with_name("worker.py")
VMAF_V1_MODEL = Path("share/vmaf/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json")
#: x264's chart modes: (CBR, metric tune). `qp` is the lens's own setting.
X264_MODES = {"qp": (False, None), "cbr": (True, None), "psnr-qp": (False, "psnr"),
              "psnr-cbr": (True, "psnr")}
X264 = "x264-medium"
MS_PER_SECOND = 1000
#: The figures `compare` prints, in order.
FIGURES = ("Weighted", "J", *J_METRICS)


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
    if not args.encoder.startswith("nanocodec-"):
        raise SystemExit(f"{args.encoder}: only our encoders and {X264} run locally")
    return [own_variant(j, args.param, args.label) for j in jobs if j["kind"] in kinds]


def tool_paths() -> dict[str, Path]:
    paths = load_paths()
    tools = {p.name: p for p in (paths.anchors / "bin").iterdir()}
    tools["nanocodec"] = paths.nanocodec_bin / "nanocodec"
    return tools


def local(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    jobs = with_metrics(local_jobs(args, Path(args.source_dir).resolve()), args.metrics)
    if not run_dir.exists():
        planner.stage(run_dir, jobs, tool_paths(), [WORKER, load_paths().anchors / VMAF_V1_MODEL])
    print(f"{len(jobs)} jobs in {run_dir}", flush=True)
    subprocess.run([sys.executable, "-I", WORKER.name, "run", "--workers", str(args.workers)],
                   cwd=run_dir, check=True)


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
    header = " | ".join(FIGURES)
    for condition, encoder, kind in tests:
        bars = [curve_key(args.anchor, "qp")]
        if args.paired_with:
            bars.append(curve_key(args.paired_with, kind))
        for bar in bars:
            means, n, time = mean_figures(curves, condition, bar, (encoder, kind))
            cells = " | ".join("—" if v is None else f"{v:+.1f}" for v in means.values())
            print(f"{condition} | {encoder} {kind} vs {bar[0]} {bar[1]} | {cells} | n={n} "
                  f"| time {time:.2f}x   ({header})")


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
    p.set_defaults(handler=local)
    c = sub.add_parser("compare", help="BD-rates of local results against stored curves")
    c.add_argument("--results", required=True)
    c.add_argument("--points", default=str(DEFAULT_POINTS))
    c.add_argument("--anchor", default=X264, help="ENCODER or ENCODER:KIND (default kind qp)")
    c.add_argument("--paired-with", help="ENCODER or ENCODER:KIND")
    c.set_defaults(handler=compare)

