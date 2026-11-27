#!/usr/bin/env python3
"""The README's chart: BD-J against encode speed, from one benchmark report, in the style of
the earlier README charts (nceval.chart.svg_chart).

    PYTHONPATH=eval python3 tools/bench/readme_chart.py --reports /abs/reports/dir \\
        --out docs/images/bd-vs-speed-bp4.svg

Reads the report stage's JSON (tools/bench/report.py, docs/benchmark-procedure.md sections
8-9), `<dataset>-<machine>.json` of one timing machine (Arm, Graviton 4, by default: one panel reads
at README width, two did not), and draws each family's BD-J on the report's headline set (the
common set) against its loaded speed (megapixels per CPU-second). The +drop rows (frames
dropped, their own clip sets) and the metric-tuned variants are left out, as on the results
site's default view. Nothing is measured or averaged here: every figure is the report's own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nceval.chart import ChartPoint, RateCounts, SpeedPlatform, pareto, svg_chart  # noqa: E402
from preview import describe  # noqa: E402  (the results site's family labels)

#: The timing machines as the reports name them: the chart title's machine and its axis.
MACHINES = {
    "c8i": SpeedPlatform("", "", "encode speed, megapixels per CPU-second, half load (log scale; faster →)",
                         "AWS c8i.metal-48xl (Intel Granite Rapids), half load"),
    "c8g": SpeedPlatform("", "", "encode speed, megapixels per CPU-second, half load (log scale; faster →)",
                         "AWS c8g.metal-24xl (Graviton 4), half load"),
}
HEADING = "Bitrate vs encode speed"
ZERO_LINE = "x264 medium (tune psnr)"
NOTE = "Same clips for every point (the common set). Speed: timed with half the machine's cores busy."


def points(report: Path, machine: str) -> list[ChartPoint]:
    """The report's charted families on its headline set, as chart points (speed in the
    chart's fps slot: the axis is megapixels per CPU-second)."""
    series = json.loads(report.read_text())["series"][machine]
    out = []
    for family, row in series["families"].items():
        about = describe(family)
        j = (row.get("bd") or {}).get("j") or {}
        if about["drop"] or about["tune"] or j.get("mean") is None \
                or not row.get("speed_mpix_per_cpu_s"):
            continue
        label = about["label"].removeprefix("nanocodec ") if about["ours"] else about["label"]
        out.append(ChartPoint(
            family=family, label=label, ours=about["ours"], bd_j=j["mean"],
            bd_psnr=((row["bd"].get("psnr_yuv") or {}).get("mean")),
            bd_vmaf=((row["bd"].get("vmaf_v1") or {}).get("mean")), iqr=None,
            scored=row.get("clips", 0), clips=row.get("clips", 0), missing_classes=(),
            rate=RateCounts(), refused=0, fps=row["speed_mpix_per_cpu_s"]))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, required=True,
                        help="directory with <dataset>-<machine>.json report files")
    parser.add_argument("--dataset", default="ours", help="report dataset (default: ours)")
    parser.add_argument("--machine", default="c8g", choices=MACHINES)
    parser.add_argument("--out", type=Path, required=True, help="SVG file to write")
    parser.add_argument("--subtitle", default="held-out camera clips, common set; CBR, "
                        "benchmark procedure BP-4")
    args = parser.parse_args(argv)
    chart = points(args.reports / f"{args.dataset}-{args.machine}.json", args.machine)
    args.out.write_text(svg_chart(chart, pareto(chart), args.subtitle, MACHINES[args.machine],
                                  heading=HEADING, note=NOTE, anchor_label=ZERO_LINE))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
