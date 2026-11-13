"""Write the sanity report: dated markdown for people, JSON for tools.

Flags come first, most severe first, each with its evidence and suspected
cause; the per-clip tables (BD-rate vs reference, tool activity, decay)
follow so a flag can be checked without rerunning anything.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .checks import SanityReport
from .flags import Severity

PREAMBLE = ("Every flag below is a suspected bug to root-cause, not a result to disclose "
            "(docs/PROCEDURES.md, 'Sanity report'). Outliers in particular need a cause, "
            "not a footnote.")
# Decay-table rows: trends steeper than this (dB/frame), flagged or not.
DECAY_TABLE_SLOPE = -0.01
TOOL_COLUMNS = ("skip", "intra", "cdef/sao", "lf/deblock", "subpel int", "top block", "top tx")


def write_report(report: SanityReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    markdown = out_dir / f"sanity-{stamp}.md"
    data = out_dir / f"sanity-{stamp}.json"
    markdown.write_text("\n".join(render_markdown(report)) + "\n")
    data.write_text(json.dumps({"created": report.created, "identity": report.identity,
                                "flags": [f.to_json() for f in report.flags],
                                "tables": report.tables}, indent=1, default=str) + "\n")
    return markdown, data


def render_markdown(report: SanityReport) -> list[str]:
    lines = [f"# Sanity report {report.created}", "", PREAMBLE, "",
             f"nanocodec `{report.identity.get('nanocodec')}` "
             f"(sha256 {str(report.identity.get('nanocodec_sha256'))[:12]}); "
             f"results `{report.identity.get('results_db')}`.", ""]
    lines += _summary(report)
    lines += ["## Flags", ""]
    for severity in sorted(Severity, reverse=True):
        chosen = [f for f in report.flags if f.severity == severity]
        if not chosen:
            continue
        lines += [f"### {severity.name} ({len(chosen)})", ""]
        for flag in chosen:
            lines.append(f"- **[{flag.check}] {flag.subject}**: {flag.finding}")
            if flag.suspected_cause:
                lines.append(f"  - suspected cause: {flag.suspected_cause}")
            if flag.evidence:
                lines.append(f"  - evidence: `{json.dumps(flag.evidence, default=str)[:600]}`")
        lines.append("")
    for name, tables in report.tables.items():
        lines += _set_tables(name, tables)
    return lines


def _summary(report: SanityReport) -> list[str]:
    counts: dict[tuple[str, str], int] = {}
    for flag in report.flags:
        key = (flag.check, flag.severity.name)
        counts[key] = counts.get(key, 0) + 1
    checks = sorted({c for c, _ in counts})
    lines = ["| check | " + " | ".join(s.name for s in sorted(Severity, reverse=True)) + " |",
             "|---|" + "---:|" * len(Severity)]
    for check in checks:
        cells = [str(counts.get((check, s.name), 0)) for s in sorted(Severity, reverse=True)]
        lines.append(f"| {check} | " + " | ".join(cells) + " |")
    return lines + [""]


def _set_tables(name: str, tables: dict) -> list[str]:
    lines = [f"## Set {name}", ""]
    for pair, clips in tables.get("bd_rate", {}).items():
        lines += [f"**BD-rate {pair}** (positive = we need more bits)", "",
                  "| clip | " + " | ".join(next(iter(clips.values()), {}).keys()) + " |",
                  "|---|" + "---:|" * len(next(iter(clips.values()), {}))]
        for clip, values in clips.items():
            cells = [f"{v:+.1f}%" if v is not None else "n/a" for v in values.values()]
            lines.append(f"| {clip} | " + " | ".join(cells) + " |")
        lines.append("")
    declining = {k: v for k, v in tables.get("decay", {}).items()
                 if v and v["slope"] < DECAY_TABLE_SLOPE}
    if declining:
        lines += [f"**Per-frame Y-PSNR trend over P frames** (slope below "
                  f"{DECAY_TABLE_SLOPE} dB/frame; reference at the nearest rate)", "",
                  "| point | P frames | slope | reference | first/last tenth dB | "
                  "reference first/last |", "|---|---:|---:|---:|---|---|"]
        for key, v in sorted(declining.items(), key=lambda kv: kv[1]["slope"]):
            ref = v["reference_slope"]
            lines.append(f"| {key} | {v['p_frames']} | {v['slope']:+.4f} | "
                         f"{'' if ref is None else f'{ref:+.4f}'} | "
                         f"{v['first_tenth_db']} / {v['last_tenth_db']} | "
                         f"{v['reference_first_last_db'] or ''} |")
        lines.append("")
    if tables.get("tools"):
        lines += ["**Tool activity, P frames** (ours / reference at the nearest rate)", "",
                  "| stream | " + " | ".join(TOOL_COLUMNS) + " |",
                  "|---|" + "---|" * len(TOOL_COLUMNS)]
        for key, row in tables["tools"].items():
            ours = row["ours"].get("P", {})
            theirs = row["theirs"].get("P", {})
            cells = [_pair(_tool_value(ours, c), _tool_value(theirs, c)) for c in TOOL_COLUMNS]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _tool_value(activity: dict, column: str) -> str:
    if not activity:
        return ""
    filters, hist = activity.get("filters", {}), activity.get("histograms", {})
    if column == "skip":
        return _share(activity.get("skip_share"))
    if column == "intra":
        return _share(activity.get("intra_share"))
    if column == "cdef/sao":
        return _share(filters.get("cdef_active_share", filters.get("sao_luma_share")))
    if column == "lf/deblock":
        value = filters.get("loop_filter_level_y")
        return f"{value:.1f}" if value is not None else _share(
            filters.get("deblocking_enabled_share"))
    if column == "subpel int":
        return _share(hist.get("subpel_phases", {}).get("integer"))
    key = "block_sizes" if column == "top block" else "tx_sizes"
    top = max(hist.get(key, {}).items(), key=lambda kv: kv[1], default=None)
    return f"{top[0]} {top[1]:.0%}" if top else ""


def _share(value) -> str:
    return "" if value is None or value != value else f"{value:.0%}"


def _pair(ours: str, theirs: str) -> str:
    return f"{ours} / {theirs}" if theirs else ours
