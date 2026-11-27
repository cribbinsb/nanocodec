"""What every CBR chart and CBR number says about the 1 s buffer: one source for the results
site's buffer page (web/results/buffer.html, nceval.site.buffer_data), the CBR chart
captions, the method page and the chart pages
(nceval.chart render).

The figures come from the buffer-use report (nceval.buffer_use, the newest REPORT_GLOB in
docs/results): how full each encoder's buffer gets on the published charts, measured at
the target and at each stream's own rate. The position (our re-code ceiling, the strict
mode) and the quality it costs are stated here with their sources.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

HEADING = "Rate control and the buffer"
#: The site's page on it (web/results/), and its public address.
PAGE = "buffer.html"
URL = f"https://cribbinsb.github.io/nanocodec/results/{PAGE}"

RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results"
REPORT_GLOB = "cbr-buffer-use-*.json"
#: The chart view whose figures the one-paragraph statement quotes (the largest set).
STATEMENT_VIEW = "heldout-psnr"
#: References whose buffer setting is not a bound (no buffer option): quoted apart.
UNBOUNDED_REFERENCES = ("openh264-baseline",)
OURS_PREFIX = "nanocodec-"

#: Our rate control's buffer settings, before and after (docs/handoff/rate-control.md,
#: "Buffer as a hard bound (0.6.1)"): the re-code ceiling `rate.recode_overflow` (a frame
#: is re-coded when it would leave more than this many buffers queued), the key cap
#: `rate.key_cap_buffer`, and the strict mode's flag.
POSITION = {
    "release_before": "0.6.0", "ceiling_before": 1.4,
    "release": "0.6.1", "ceiling": 1.15, "key_cap": 0.8,
    "strict_ceiling": 1.0, "strict_flag": "--strict-buffer", "strict_c_field": "strict_buffer",
    "parameter": "rate.recode_overflow", "key_parameter": "rate.key_cap_buffer",
}

#: What each encoder is asked for in the CBR charts, in words, with the chart family whose
#: harness command line (cbr.Family.command, shown from chart.json) the page prints.
REQUESTS = (
    ("nanocodec", "nanocodec-av1", "its own CBR with a 1 s buffer."),
    ("x264, x265", "x264-medium~psnr", "ABR held by a hard VBV of one second (maximum rate = "
                                       "the target), zero latency, no B-frames, no lookahead."),
    ("SVT-AV1", "svtav1-m12", "real-time CBR with a 1 s buffer."),
    ("libaom, libvpx", "libaom-av1-cpu8", "real-time CBR with a 1 s buffer."),
    ("openh264", "openh264-baseline", "its bitrate mode, frame skipping off; it has no buffer "
                                      "setting."),
)

#: The quality the 0.6.1 settings cost against 0.6.0's, and each arm's buffer peaks
#: (nceval.rc_oracle on the medium presets; docs/handoff/rate-control.md, "Default 1.15
#: buffers, strict 1.0 as an option").
COST_GLOB = "cbr-buffer-cost-*.json"


@functools.cache
def report() -> dict:
    """The newest buffer-use report (nceval.buffer_use)."""
    return json.loads(sorted(RESULTS.glob(REPORT_GLOB))[-1].read_text())


def report_path() -> str:
    return f"docs/results/{sorted(RESULTS.glob(REPORT_GLOB))[-1].name}"


@functools.cache
def cost() -> dict:
    """The newest cost record (COST_GLOB)."""
    return json.loads(sorted(RESULTS.glob(COST_GLOB))[-1].read_text())


def cost_path() -> str:
    return f"docs/results/{sorted(RESULTS.glob(COST_GLOB))[-1].name}"


def bd_range(values: list[float]) -> str:
    low, high = min(values), max(values)
    return f"{low:+.2f} %" if low == high else f"{low:+.2f} to {high:+.2f} %"


def cost_sentence() -> str:
    """What the 0.6.1 default and strict mode cost against 0.6.0's buffer, per window."""
    record = cost()
    windows = sorted({r["frames"] for r in record["rows"]})
    per = lambda key: " and ".join(  # noqa: E731
        f"{bd_range([r[key] for r in record['rows'] if r['frames'] == w])} over {w} frames"
        for w in windows)
    return (f"Against 0.6.0's buffer, on the {record['preset']} presets, the "
            f"{POSITION['ceiling']:g}-buffer default costs {per('j_ceiling')} in BD-J (bits at "
            f"equal quality), and strict mode {per('j_strict')} ({record['runs'][0].lower()}{record['runs'][1:]}).")


def view(view_id: str) -> dict:
    return next(v for v in report()["views"] if v["view"] == view_id)


def percent_range(values: list[float]) -> str:
    low, high = round(min(values)), round(max(values))
    return f"{low} %" if low == high else f"{low}-{high} %"


def over_one(view_id: str, which: str) -> str:
    """The range of the share of points over one buffer at their own rate: ours, the bounded
    references or the unbounded ones."""
    rows = view(view_id)["rows"]
    pick = {"ours": lambda f: f.startswith(OURS_PREFIX),
            "bounded": lambda f: not f.startswith(OURS_PREFIX) and f not in UNBOUNDED_REFERENCES,
            "unbounded": lambda f: f in UNBOUNDED_REFERENCES}[which]
    return percent_range([r["own_over"]["1"] for f, r in rows.items() if pick(f)])


def caption() -> str:
    """One line beside a CBR chart or number: the measured buffer use it rests on."""
    held = view(STATEMENT_VIEW)
    return (f"At each stream's own rate, {over_one(STATEMENT_VIEW, 'ours')} of nanocodec's points in "
            f"these CBR charts exceed one buffer, against {over_one(STATEMENT_VIEW, 'bounded')} of "
            f"the buffer-bounded references' ({held['clips']} held-out clips): see {HEADING}.")


@functools.cache
def statement() -> str:
    """The one-paragraph statement beside every CBR chart."""
    held = view(STATEMENT_VIEW)
    p = POSITION
    text = (
        "Every encoder in the CBR charts is asked for a 1 s buffer at the target rate; openh264 "
        "has no buffer setting. Measured at each stream's own achieved rate (the rate it is "
        f"plotted at), on the {held['clips']} held-out clips of the {p['release_before']} charts "
        f"nanocodec exceeds one buffer on {over_one(STATEMENT_VIEW, 'ours')} of points, the other "
        f"references on {over_one(STATEMENT_VIEW, 'bounded')} and openh264 on "
        f"{over_one(STATEMENT_VIEW, 'unbounded')}. From {p['release']} nanocodec re-codes any "
        f"frame that would leave more than {p['ceiling']:g} buffers queued, and "
        f"{p['strict_flag']} holds it to exactly one buffer. The CBR charts will be re-measured "
        f"on {p['release']}; the Fixed QP charts have no rate control and are not affected.")
    return f"{text} {cost_sentence()}"


#: The facts table's columns: (buffer_use FACTS name, heading, unit).
FACT_COLUMNS = (("own_over_1", "over 1 buffer at own rate", "%"),
                ("own_over_1.15", "over 1.15 buffers at own rate", "%"),
                ("target_over_1", "over 1 buffer at the target", "%"),
                ("own_p95", "95th-percentile peak at own rate", " buffers"),
                ("key_buffers_median", "first key frame, median", " buffers"),
                ("key_buffers_p90", "first key frame, 90th percentile", " buffers"),
                ("window_ratio_median", "worst 1 s window / target, median", "×"))


def fact_cell(value: list[float], unit: str) -> str:
    digits = 0 if unit == "%" else 2
    return f"{value[0]:.{digits}f}{unit} ({value[1]:.{digits}f}–{value[2]:.{digits}f})"


def facts_table(view_id: str = STATEMENT_VIEW) -> str:
    """Every encoder's measured buffer facts on one chart view, each with its 95 % clip
    bootstrap interval (nceval.buffer_use `intervals`; results audit 2026-11-23,
    finding 11): the same figures for ours and every reference. Empty for a report made
    before the intervals were recorded."""
    rows = view(view_id)["rows"]
    if not all("intervals" in r for r in rows.values()):
        return ""
    lines = ["| Encoder | points | " + " | ".join(h for _, h, _ in FACT_COLUMNS) + " |",
             "|---|---:|" + "---:|" * len(FACT_COLUMNS)]
    for name, r in rows.items():
        cells = " | ".join(fact_cell(r["intervals"][key], unit) for key, _, unit in FACT_COLUMNS)
        lines.append(f"| {name} | {r['points']} | {cells} |")
    return "\n".join(lines) + "\n"
