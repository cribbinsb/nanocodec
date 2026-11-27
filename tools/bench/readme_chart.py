#!/usr/bin/env python3
"""The README's chart: BD-J against encode speed, from one dataset's benchmark reports.

    python3 tools/bench/readme_chart.py --reports /abs/reports/dir --out docs/images/bd-vs-speed.svg

Reads the report stage's JSON (tools/bench/report.py, docs/benchmark-procedure.md sections
8-9), `<dataset>-<machine>.json` for each timing machine, and draws one panel per machine:
each family's BD-J on the report's headline set (the common set) against its loaded speed
(megapixels per CPU-second), our presets joined per codec, the references in grey. The +drop
rows (frames dropped, their own clip sets) and the metric-tuned variants (`~psnr`, `~ssim`,
`~cabac`) are left out, as on the results site's default view. Nothing is measured or
averaged here: every figure is the report's own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

#: The timing machines, in panel order, as the reports name them and as the chart names them.
MACHINES = {"c8i": "x86: AWS c8i.metal-48xl (Intel Granite Rapids)",
            "c8g": "Arm: AWS c8g.metal-24xl (Graviton 4)"}
#: Our codecs, in legend order, and their colours (categorical, fixed order).
CODECS = {"vvc": ("VVC", "#6a4ce2"), "av2": ("AV2", "#ec4647"), "hevc": ("HEVC", "#257de0"),
          "av1": ("AV1", "#f47230"), "vp9": ("VP9", "#008a17"), "h264": ("H.264", "#f2a600")}
OURS_PREFIX = "nanocodec-"
REFERENCE_COLOUR = "#8a8a8a"
TEXT = "#191919"
TEXT_SECONDARY = "#666666"
GRID = "#e6e6e6"
#: References named on the chart, with each label's offset in points (the rest are unlabelled
#: grey points): one setting of each project at least, placed clear of the other marks.
LABELLED = {"x264-medium": ("x264 medium", (6, 2)), "x264-ultrafast": ("x264 ultrafast", (-8, -10)),
            "x265-medium": ("x265 medium", (-6, -12)), "svtav1-m12": ("SVT-AV1 M12", (8, -4)),
            "libaom-av1-cpu10": ("libaom rt 10", (-40, -13)), "libvpx-vp9-cpu9": ("libvpx rt 9", (6, 2)),
            "kvazaar-veryfast": ("Kvazaar veryfast", (-6, 7)), "openh264-baseline": ("openh264", (6, 2))}
#: Figure width in inches at 100 dpi: 1000 px, the width GitHub shows a README image at.
WIDTH_IN, HEIGHT_IN = 10.0, 4.6


def families(report: Path, machine: str) -> dict[str, dict]:
    """The report's charted families on its headline set: {family: {"j", "speed"}}."""
    series = json.loads(report.read_text())["series"][machine]
    return {name: {"j": row["bd"]["j"]["mean"], "speed": row["speed_mpix_per_cpu_s"]}
            for name, row in series["families"].items()
            if row.get("clips") and row.get("speed_mpix_per_cpu_s")
            and "+drop" not in name and "~" not in name}


def panel(ax, rows: dict[str, dict], title: str) -> None:
    for codec, (label, colour) in CODECS.items():
        points = sorted((r["speed"], r["j"]) for f, r in rows.items()
                        if f.startswith(f"{OURS_PREFIX}{codec}@"))
        xs, ys = zip(*points)
        ax.plot(xs, ys, "-o", color=colour, lw=2, ms=5, label=f"nanocodec {label}", zorder=3)
    refs = {f: r for f, r in rows.items() if not f.startswith(OURS_PREFIX)}
    ax.scatter([r["speed"] for r in refs.values()], [r["j"] for r in refs.values()],
               s=28, facecolors="white", edgecolors=REFERENCE_COLOUR, lw=1.5, zorder=2,
               label="reference encoders")
    for family, (name, offset) in LABELLED.items():
        if family in refs:
            ax.annotate(name, (refs[family]["speed"], refs[family]["j"]), xytext=offset,
                        textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY,
                        ha="right" if offset[0] < 0 else "left")
    ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 400])
    ax.set_xticklabels(["10", "20", "50", "100", "200", "400"])
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel("encode speed, Mpx per CPU-second (log; right is faster)", fontsize=8.5,
                  color=TEXT_SECONDARY)
    ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, required=True,
                        help="directory with <dataset>-<machine>.json report files")
    parser.add_argument("--dataset", default="ours", help="report dataset (default: ours)")
    parser.add_argument("--out", type=Path, required=True, help="SVG file to write")
    args = parser.parse_args()
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "svg.hashsalt": "nanocodec"})  # stable ids: diffs show real changes
    fig, axes = plt.subplots(1, len(MACHINES), figsize=(WIDTH_IN, HEIGHT_IN), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (machine, title) in zip(axes, MACHINES.items()):
        panel(ax, families(args.reports / f"{args.dataset}-{machine}.json", machine), title)
    axes[0].set_ylabel("BD-J vs x264 medium --tune psnr\n(lower is fewer bits)", fontsize=8.5,
                       color=TEXT_SECONDARY)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.out, metadata={"Date": None})


if __name__ == "__main__":
    main()
