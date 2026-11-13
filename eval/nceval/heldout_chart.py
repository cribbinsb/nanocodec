"""The held-out check's chart: BD-J on clips no decision has used, against x86 speed.

The overfitting check (docs/results/heldout-chart-*.md, tools/dev/aws_heldout_chart.py)
re-measures every chart family on held-out clips. This draws its BD-J vs x264 medium
on the main chart's x86 speed axis, in the main chart's style (nceval.chart.svg_chart),
so the two read side by side: the held-out BD-J from the check's CSV, the speed from
the main chart's x86 CSV (the same builds and streams' timing).

    python3 -m nceval.heldout_chart --heldout ../docs/results/heldout-chart-2026-11-12.csv \\
        --x86 ../docs/results/bd-vs-speed-x86.csv --out ../docs/results/heldout-chart-2026-11-12.svg
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .chart import X86, ChartPoint, pareto, svg_chart

HEADING = "Held-out clips: bitrate vs single-threaded encode speed"
NOTE = ("BD-J on 71 held-out clips (no tuning, screen or chart decision used them); speed as "
        "in the main x86 chart. Hollow: flagged.")


def heldout_points(heldout_csv: Path, x86_csv: Path) -> list[ChartPoint]:
    """Every family of the held-out check with its held-out BD-J and its chart x86 fps."""
    speed = {r["family"]: r for r in csv.DictReader(x86_csv.open())}
    points = []
    for row in csv.DictReader(heldout_csv.open()):
        timed = speed.get(row["family"], {})
        fps = float(timed["fps_x86"]) if timed.get("fps_x86") else None
        points.append(ChartPoint(
            family=row["family"], label=row["label"], ours=row["ours"] == "1",
            bd_j=float(row["bd_j_heldout_pct"]), bd_psnr=float(row["bd_psnr_heldout_pct"]),
            bd_vmaf=float(row["bd_vmaf_heldout_pct"]), iqr=None,
            scored=int(row["clips_scored"]), clips=int(row["clips"]), missing_classes=(),
            rate_invalid=0, refused=0, fps=fps,
            speed=float(timed["speed_vs_anchor"]) if timed.get("speed_vs_anchor") else None))
    return points


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python3 -m nceval.heldout_chart",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--heldout", type=Path, required=True, help="the check's CSV")
    parser.add_argument("--x86", type=Path, required=True, help="the main chart's x86 CSV")
    parser.add_argument("--out", type=Path, required=True, help="the SVG to write")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--stamp", default="", help="the small bottom-right line")
    parser.add_argument("--note", default=NOTE, help="the footnote under the chart")
    parser.add_argument("--heading", default=HEADING)
    args = parser.parse_args(argv)
    # families without an x86 time (not on the main chart) have no place on the speed axis
    points = [p for p in heldout_points(args.heldout, args.x86) if p.fps is not None]
    args.out.write_text(svg_chart(points, pareto(points), args.subtitle, X86, args.stamp,
                                  heading=args.heading, note=args.note))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
