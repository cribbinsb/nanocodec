"""Markdown tables for the results page, from `report.report`'s summaries.

Signs as in the CTCs: a negative BD-rate means fewer bits than the anchor at
equal quality. Each cell is the mean over sequences with a reliable BD-rate;
`n` is how many. EncT is the geometric mean of per-sequence encode user-time
ratios (test / anchor).
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median

from .conditions import CONDITIONS, JVET_CTC_QPS, Condition
from .report import enc_time_ratio
from .sequences import Tier

OVERALL = "overall"
AOM_COLUMNS = ("Weighted", "Y", "U", "V", "APSNR-YUV", "MS-SSIM", "VMAF", "J")
JVET_COLUMNS = ("Y-fa@ctc", "U-fa@ctc", "V-fa@ctc", "Y-fa", "J")
CLASS_COLUMNS = ("Y", "U", "V", "Weighted", "J")


def _cell(summary: dict, key: tuple) -> str:
    value = summary.get(key)
    return "—" if value is None else f"{value[0]:+.1f}"


def _count(summary: dict, key: tuple) -> str:
    value = summary.get(key)
    return "0" if value is None else str(value[3])


def _time(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3g}x" if value >= 0.01 else f"1/{1 / value:,.0f}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _header(columns: list[str]) -> str:
    return _row(columns) + "\n" + _row(["---"] * len(columns))


def _sequences_with(curves, condition: Condition, encoder: str) -> list[str]:
    return [s.name for s in condition.sequences if curves.get(condition.name, s.name, encoder)]


def overall_table(data: dict, condition: Condition, anchor: str, columns: tuple[str, ...],
                  kinds: tuple[str, ...] = ("qp",)) -> str:
    """One row per test encoder: overall BD-rates vs `anchor`, n, EncT."""
    summary, curves = data["all"], data["curves"]
    lines = [_header(["Encoder", *columns, "n", "EncT", f"{columns[0]}, free tier only"])]
    for encoder in condition.encoders:
        if encoder.name == anchor:
            continue
        for kind in kinds:
            if kind == "cbr" and not (encoder.own and condition.cbr_extra):
                continue
            key = (condition.name, anchor, encoder.name, kind)
            if (*key, columns[0], OVERALL) not in summary:
                continue
            names = _sequences_with(curves, condition, encoder.name)
            enc_t = enc_time_ratio(curves, condition, anchor, encoder.name, names) \
                if kind == "qp" else None
            label = f"{encoder.name}" + (" (CBR extra)" if kind == "cbr" else "")
            lines.append(_row([label, *(_cell(summary, (*key, c, OVERALL)) for c in columns),
                               _count(summary, (*key, columns[0], OVERALL)), _time(enc_t),
                               _cell(data["free"], (*key, columns[0], OVERALL))]))
    return "\n".join(lines)


def class_table(data: dict, condition: Condition, anchor: str, test: str) -> str:
    """The CTC's per-class layout for one test encoder vs one anchor."""
    summary = data["all"]
    classes = sorted({s.sequence_class for s in condition.sequences})
    lines = [_header(["Class", *CLASS_COLUMNS, "n"])]
    for cls in [*classes, OVERALL]:
        key = (condition.name, anchor, test, "qp")
        lines.append(_row([cls, *(_cell(summary, (*key, c, cls)) for c in CLASS_COLUMNS),
                           _count(summary, (*key, "Y", cls))]))
    return "\n".join(lines)


def rate_table(data: dict, condition: Condition, encoders: tuple[str, ...]) -> str:
    """Median achieved kbps per class and quantiser point (what the CTC points give)."""
    rates: dict[tuple, list[float]] = defaultdict(list)
    classes = {s.name: s.sequence_class for s in condition.sequences}
    for p in data["points"]:
        if p.condition == condition.name and p.kind == "qp" and p.encoder in encoders:
            rates[(p.encoder, classes[p.sequence], p.q)].append(p.kbps)
    lines = [_header(["Encoder", "Class", "Point", "Median kbps", "Range kbps", "n"])]
    for (encoder, cls, q), values in sorted(rates.items()):
        lines.append(_row([encoder, cls, str(q), f"{median(values):,.0f}",
                           f"{min(values):,.0f}-{max(values):,.0f}", str(len(values))]))
    return "\n".join(lines)


def tier_note(data: dict) -> str:
    tiers = defaultdict(set)
    for r in data["rows"]:
        tiers[Tier(r["tier"])].add(r["sequence"])
    return ", ".join(f"{t.value}: {len(s)} sequences" for t, s in sorted(tiers.items()))


def page_tables(data: dict) -> str:
    """Every table the results page shows, as one markdown document."""
    aom, jvet = CONDITIONS["aom-ld"], CONDITIONS["jvet-e"]
    parts = [f"Licence tiers in these tables: {tier_note(data)}."]
    for anchor in aom.anchors:
        parts += [f"### AOM LD: overall BD-rate vs {anchor}",
                  overall_table(data, aom, anchor, AOM_COLUMNS, ("qp", "cbr"))]
    for test in (e.name for e in aom.encoders if e.own and "@" not in e.name):
        parts += [f"### AOM LD per class: {test} vs hm-ldp", class_table(data, aom, "hm-ldp", test)]
    parts += [f"### JVET LD-P class E: BD-rate vs hm-ldp (CTC = QP {JVET_CTC_QPS[0]}-"
              f"{JVET_CTC_QPS[-1]}; Y-fa = all six points)",
              overall_table(data, jvet, "hm-ldp", JVET_COLUMNS)]
    parts += ["### Rates the quantiser points give (AOM LD)",
              rate_table(data, aom, ("hm-ldp", "nanocodec-av1", "nanocodec-h264")),
              "### Rates the quantiser points give (JVET class E)",
              rate_table(data, jvet, ("hm-ldp", "nanocodec-hevc"))]
    return "\n\n".join(parts) + "\n"
