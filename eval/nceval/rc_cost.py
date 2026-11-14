"""Two reporting lines beside the chart: fixed QP against fixed QP, and each encoder's own rate-control cost.

    python3 -m nceval.rc_cost --binary BIN [--encoders nanocodec-av1,nanocodec-hevc] --out CSV

The chart (nceval.chart) compares every encoder in its own CBR with x264 medium
in x264's own CBR. That mixes coding efficiency with rate control, and x264's
zero-latency CBR (1 s VBV, no lookahead) is expensive for x264 itself
(docs/analysis/standards-lens-improvement-2026-11-14.md). On the chart's clips
and ladder (natural VAL, 150 frames, one key, LADDER_KBPS at 720p30, no
low-rung extension) this measures, per encoder, both its CBR curve and a
fixed-QP curve spanning the same rates (`cbr.measure_all`), for ours and for
the x264 medium anchor at its default tuning and at `--tune psnr`, and writes
one CSV row per pairing:

- `cbr_vs_cbr`: the chart's pairing (without the low-rung rule);
- `qp_vs_qp`: fixed QP against the anchor at fixed QP (coding efficiency);
- `rc_cost`: an encoder's CBR against its own fixed QP (what its rate control
  costs it; the anchor column then names the encoder itself).

BD-J is the project's J (nceval.jreal: the mean of the full and lower-half
BD-rates over PSNR-YUV, VMAF v1 and MS-SSIM), class-balanced over clips with a
reliable BD; the three terms are the full-range BD-rates. Points are cached
(nceval.cbr.Bench), so a re-run after the chart measures nothing new.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from . import buildinfo, jreal
from .cbr import Family, measure_all, preset_anchor
from .config import load_paths
from .gatepolicy import EVALUATION_GATE
from .realuse import DEFAULT_FRAMES, LADDER_KBPS, Runner, natural_val, own_family

ANCHOR = "x264-medium"
TUNED_ANCHOR = "x264-medium~psnr"
WORKERS = 4
COLUMNS = ("line", "test", "test_mode", "anchor", "anchor_mode", "bd_j", "bd_psnr_yuv",
           "bd_vmaf_v1", "bd_ms_ssim", "clips", "frames", "ladder_kbps", "binary_commit")


def clip_terms(anchor: list[dict], test: list[dict]) -> dict[str, float | None]:
    """One clip's full and lower-half BD-rates per J metric, and their means."""
    row: dict[str, float | None] = {}
    for metric in jreal.HEADLINE_METRICS:
        row[metric] = jreal.curve_bd(anchor, test, metric)
        row[f"low_{metric}"] = jreal.curve_bd(anchor, test, metric, True)
    for name, prefix in (("full", ""), ("low", "low_")):
        values = [row[f"{prefix}{m}"] for m in jreal.HEADLINE_METRICS]
        row[name] = None if None in values else sum(values) / len(values)
    return row


def pairing(curves: dict, clips, anchor: tuple[str, str], test: tuple[str, str]) -> dict:
    """Class-balanced BD-J and terms of `test` against `anchor` ((family, mode) each)."""
    per_clip = {c.name: {"class": c.content_class,
                         **clip_terms(curves[(*anchor, c.name)], curves[(*test, c.name)])}
                for c in clips}
    full, low = (jreal.aggregate(per_clip, t)["balanced"] for t in ("full", "low"))
    return {"bd_j": None if full is None or low is None else (full + low) / 2,
            **{f"bd_{m}": jreal.aggregate(per_clip, m)["balanced"]
               for m in jreal.HEADLINE_METRICS},
            "clips": jreal.aggregate(per_clip, "full")["scored"]}


def curves_by_mode(cbr: dict, cqp: dict) -> dict:
    """{(family, "cbr"|"qp", clip): points sorted by rate}."""
    curves: dict = {}
    for mode, results in (("cbr", cbr), ("qp", cqp)):
        for (family, clip, _), point in results.items():
            curves.setdefault((family, mode, clip), []).append(point)
    return {key: sorted(points, key=lambda p: p["kbps"]) for key, points in curves.items()}


def pairings(ours: list[str]) -> list[tuple[str, tuple[str, str], tuple[str, str]]]:
    lines = []
    for anchor in (ANCHOR, TUNED_ANCHOR):
        for test in ours:
            lines += [("cbr_vs_cbr", (anchor, "cbr"), (test, "cbr")),
                      ("qp_vs_qp", (anchor, "qp"), (test, "qp"))]
    return lines + [("rc_cost", (f, "qp"), (f, "cbr")) for f in (*ours, ANCHOR, TUNED_ANCHOR)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", required=True)
    parser.add_argument("--encoders", default="nanocodec-av1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    paths, binary = load_paths(), Path(args.binary)
    ours = args.encoders.split(",")
    families: list[Family] = [preset_anchor(ANCHOR, paths, cqp=True),
                              preset_anchor(TUNED_ANCHOR, paths, cqp=True)]
    families += [replace(own_family(name, binary), cqp=True) for name in ours]
    clips = natural_val(paths)
    runner = Runner(paths, EVALUATION_GATE, WORKERS, over_budget=False)
    cbr, cqp = measure_all(runner.bench(DEFAULT_FRAMES), families, clips, WORKERS,
                           EVALUATION_GATE, LADDER_KBPS)
    curves = curves_by_mode(cbr, cqp)
    commit = buildinfo.info_of(binary).get("commit") or ""
    rows = []
    for line, anchor, test in pairings(ours):
        result = pairing(curves, clips, anchor, test)
        rows.append({"line": line, "test": test[0], "test_mode": test[1], "anchor": anchor[0],
                     "anchor_mode": anchor[1],
                     **{k: (round(v, 1) if isinstance(v, float) else v)
                        for k, v in result.items()},
                     "frames": DEFAULT_FRAMES, "ladder_kbps": "/".join(map(str, LADDER_KBPS)),
                     "binary_commit": commit})
        print(f"{line:10s} {test[0]:28s} {test[1]:3s} vs {anchor[0]:18s} {anchor[1]:3s} "
              f"J {rows[-1]['bd_j']}  clips {rows[-1]['clips']}", flush=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
