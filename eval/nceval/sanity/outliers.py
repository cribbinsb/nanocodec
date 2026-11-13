"""Check 2: per-clip outliers against the reference encoders.

A mean BD-rate hides single clips that are far worse (bodycam_airport at
+115% vs SVT sat in reports for weeks as a "disclosed outlier"). Every clip
more than the limit worse than its reference encoder, on either metric, is a
bug to root-cause, not a number to disclose.
"""

from __future__ import annotations

from ..bdrate import bd_rate
from .curves import CurveStore
from .flags import Flag, Severity, severity_by_excess

# Percentage points beyond the limit that make an outlier HIGH / CRITICAL.
HIGH_EXCESS_PERCENT = 10.0
CRITICAL_EXCESS_PERCENT = 35.0
MIN_BD_POINTS = 4


def clip_bd(store: CurveStore, test: str, reference: str, clip: str,
            metric: str) -> tuple[float | None, str]:
    """(BD-rate %, note) of test vs reference on one clip; None with a reason if unusable."""
    t_kbps, t_q = store.curve(test, clip).series(metric)
    r_kbps, r_q = store.curve(reference, clip).series(metric)
    if len(t_kbps) < MIN_BD_POINTS or len(r_kbps) < MIN_BD_POINTS:
        return None, f"too few scored points ({len(t_kbps)} test, {len(r_kbps)} reference)"
    try:
        result = bd_rate(r_kbps, r_q, t_kbps, t_q)
    except ValueError as err:
        return None, str(err)
    if result.bd_rate_percent is None:
        return None, "curves do not overlap in quality"
    note = "" if result.reliable else f"low overlap ({result.overlap_fraction:.2f})"
    return result.bd_rate_percent, note


def check_outliers(store: CurveStore, clips: list[str], references: dict[str, str],
                   metrics: tuple[str, ...], limit: float, encoders: tuple[str, ...],
                   label: str) -> tuple[list[Flag], dict]:
    """Flags plus the full per-clip BD table (for the report)."""
    flags, table = [], {}
    for test, reference in references.items():
        if test not in encoders or reference not in encoders:
            continue
        for clip in clips:
            if not (store.curve(test, clip).points and store.curve(reference, clip).points):
                continue  # a missing curve is a harness flag, not an outlier
            values, notes = {}, {}
            for metric in metrics:
                values[metric], notes[metric] = clip_bd(store, test, reference, clip, metric)
            table.setdefault(f"{test} vs {reference}", {})[clip] = values
            bad = {m: v for m, v in values.items() if v is not None and v > limit}
            missing = {m: notes[m] for m, v in values.items() if v is None}
            subject = f"{label}/{clip}: {test} vs {reference}"
            if bad:
                excess = max(bad.values()) - limit
                flags.append(Flag(
                    check="outliers",
                    severity=severity_by_excess(excess, HIGH_EXCESS_PERCENT,
                                                CRITICAL_EXCESS_PERCENT),
                    subject=subject,
                    finding=(f"BD-rate {_fmt(values)} vs {reference} exceeds +{limit:g}%: "
                             f"needs root-causing, not disclosure"),
                    evidence={"bd_rate_percent": values, "notes": notes, "limit": limit},
                    suspected_cause=outlier_cause(bad, values)))
            elif missing:
                flags.append(Flag(
                    check="outliers", severity=Severity.MEDIUM, subject=subject,
                    finding="BD-rate could not be computed, so this clip is unchecked",
                    evidence={"bd_rate_percent": values, "notes": notes},
                    suspected_cause="missing or degenerate curve (see harness flags)"))
    return flags, table


def outlier_cause(bad: dict[str, float], values: dict[str, float | None]) -> str:
    if len(bad) == len([v for v in values.values() if v is not None]):
        return ("loss on every metric: a tool off or broken on this content "
                "(check tool activity and per-frame decay)")
    metric = next(iter(bad))
    return (f"loss on {metric} only: allocation between planes / perceptual tuning "
            f"diverges from the reference on this content")


def _fmt(values: dict[str, float | None]) -> str:
    return ", ".join(f"{m} {v:+.1f}%" if v is not None else f"{m} n/a"
                     for m, v in values.items())
