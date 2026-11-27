"""The CPU charge k of a preset change: the slope of its codec's speed/quality front.

A preset change is accepted if and only if it improves the Pareto front (owner,
2026-11-22). The acceptance screens (nceval.quick, nceval.train, nceval.screen)
judge J_real = quality + k * dCPU; a fixed k (0.1 at medium, 0.6 on the fast rungs)
accepted CPU bought more dearly than the front between the rungs buys it (the
2026-11-22 history frame: front J gap +1.0, speed 0.93x). So k at rung R is the
front's own exchange rate around R, as `tune front` judges a between-rungs study
(nceval.tune_pool.chord_of): moving along the front scores 0, below it negative.

The slopes come from one front of every shipped rung's BD-J against x264 medium and its
speed, on one basis, through each codec's own front (`codec_fronts`): since BP-3.10 a
benchmark report (tools/bench/report.py report.json: the "ours" CBR run's c8i series,
`generate --report`), before that a history frame (tools/dev/history_chart.py) or a rendered
chart (c8a fps). For rung R:

  * between two front points: the chord from the nearest slower to the nearest
    faster front point (R itself excluded);
  * at an end: the front's end chord on that side (R to its neighbour when R is on
    the front, else the two front points nearest R).

The chart's J is BD-rate against x264, a screen's J is BD-rate against the rung
itself; a 1 % saving at the rung is (1 + J_R / 100) chart points, so
k = -dJ_chart / ((1 + J_R / 100) * 100 * dln CPU), CPU = 1 / speed (fps, or a report's
megapixels per CPU-second: only speed ratios enter k).

The table is checked in (`TABLE`) and regenerated with each history frame
(`python3 -m nceval.front_k generate`, also run by history_chart.py report) or
preset change; every screen prints its source and date (`describe`).

One measurement per decision (owner, 2026-11-23): the screens judge CBR quality, so
a slope is only taken from a front whose fps time CBR encodes, the chart's own quality
encodes (tools/dev/history_chart.py records the timing plan's rate mode on each frame,
`TIMING_MODE_KEY`). Frames timed at fixed QP (every frame before the CBR re-timing),
or whose fps are chained by paired timing to such a frame (`CHAINED_KEY`), are refused
when generating; a table that does not say it came from CBR timing is refused when
read, so no screen charges a k from another timing mode (an explicit --k still works,
loudly).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .encoders.nanocodec import MEDIUM

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "eval" / "configs" / "front_k.json"
HISTORY = REPO / "web" / "results" / "data" / "history.json"
#: The k each rung used before front slopes (checkrun.PRESET_K then; reported only).
FIXED_K_BEFORE = {MEDIUM: 0.1, "slow": 0.1, "slower": 0.0, "veryslow": 0.0}
FIXED_K_SPEED_RUNGS = 0.6
#: A history frame's timing rate mode (history_chart.py new_day) and the frame its fps
#: are chained to by paired timing; the table's record of the mode it came from.
TIMING_MODE_KEY = "timing_rate_mode"
CHAINED_KEY = "fps_chained_from"
#: The screens' rate mode: the only timing a slope may come from.
SCREEN_RATE_MODE = "cbr"


class NoSlope(SystemExit):
    """The table has no slope for that rung: regenerate it, or give --k."""


def chart_key(codec: str, rung: str) -> str:
    """History's preset key: `hevc` for medium, `hevc@fast` otherwise."""
    return codec if rung == MEDIUM else f"{codec}@{rung}"


def rung_of(key: str) -> tuple[str, str]:
    codec, _, rung = key.partition("@")
    return codec, rung or MEDIUM


def chord_k(slow: dict, fast: dict, j_rung: float) -> float:
    """The chord's slope in screen units (J % at the rung per 1 % CPU); `slow` and
    `fast` are {"j", "fps"}."""
    dj = slow["j"] - fast["j"]
    dln_cpu = math.log(fast["fps"] / slow["fps"])
    return -dj / ((1 + j_rung / 100) * 100 * dln_cpu)


def bracket(rung: str, points: dict[str, dict], front: list[str]) -> tuple[str, str]:
    """The two front points whose chord prices `rung` (slower first; module doc)."""
    fps = points[rung]["fps"]
    others = sorted((p for p in front if p != rung), key=lambda p: points[p]["fps"])
    slower = [p for p in others if points[p]["fps"] < fps]
    faster = [p for p in others if points[p]["fps"] > fps]
    on_front = rung in front
    if slower and faster:
        return slower[-1], faster[0]
    if faster:
        return (rung, faster[0]) if on_front else (faster[0], faster[1])
    if slower:
        return (slower[-1], rung) if on_front else (slower[-2], slower[-1])
    raise NoSlope(f"{rung}: its codec's front has no other point")


def one_sided(rung: str, points: dict[str, dict], front: list[str]) -> dict[str, float | None]:
    """The chords from a front rung to its slower and faster neighbours (reported only)."""
    if rung not in front:
        return {"toward_slower": None, "toward_faster": None}
    ordered = sorted(front, key=lambda p: points[p]["fps"])
    i, j = ordered.index(rung), points[rung]["j"]
    return {"toward_slower": chord_k(points[ordered[i - 1]], points[rung], j) if i else None,
            "toward_faster": chord_k(points[rung], points[ordered[i + 1]], j)
            if i + 1 < len(ordered) else None}


def slopes(day: dict) -> dict[str, dict[str, dict]]:
    """{codec: {rung: entry}} for every rung of a history day with J and fps."""
    points = {k: {"j": v["j"], "fps": v["fps_x86"]} for k, v in day["values"].items()
              if v.get("j") is not None and v.get("fps_x86")}
    table: dict[str, dict[str, dict]] = {}
    for key in sorted(points):
        codec, rung = rung_of(key)
        front = [p for p in day["codec_fronts"].get(codec, []) if p in points]
        slow, fast = bracket(key, points, front)
        table.setdefault(codec, {})[rung] = {
            "k": round(chord_k(points[slow], points[fast], points[key]["j"]), 4),
            "chord": [slow, fast], "on_front": key in front, "j": points[key]["j"],
            "fps": points[key]["fps"],
            **{side: None if v is None else round(v, 4)
               for side, v in one_sided(key, points, front).items()}}
    return table


def generate(history: Path, day: str | None = None) -> dict:
    record = json.loads(history.read_text())
    days = record["days"] if day is None else [d for d in record["days"] if d["date"] == day]
    if not days:
        raise SystemExit(f"{history}: no day {day}")
    try:
        source = str(history.resolve().relative_to(REPO))
    except ValueError:
        source = str(history)
    return generate_from(days[-1], source, record["days"])


#: Our families in a chart CSV (nceval.chart render): `nanocodec-<codec>[@rung]`.
OWN_PREFIX = "nanocodec-"


def chart_day(csv_path: Path, timing_path: Path, date: str, commit: str) -> dict:
    """A history-day-shaped record of a rendered chart (its x86 CSV: BD-J vs x264 medium
    and c8a fps of every shipped rung), whose speed and quality are the same encodes:
    `timing_path` (the CSV's x86 timing record) must time the chart's own CBR encodes."""
    import csv

    timing = json.loads(timing_path.read_text())
    rows = [r for r in csv.DictReader(csv_path.open()) if r["family"].startswith(OWN_PREFIX)
            and r["bd_j_pct"] and r["fps_x86"]]
    values = {r["family"].removeprefix(OWN_PREFIX): {"j": float(r["bd_j_pct"]),
                                                    "fps_x86": float(r["fps_x86"])}
              for r in rows}
    return {"date": date, "commit": commit, "values": values,
            "codec_fronts": codec_fronts(values),
            TIMING_MODE_KEY: timing.get("rate_mode")}


#: A benchmark report's suffixes of rows that are not a shipped rung: twins and +drop rows.
REPORT_EXTRA_ROWS = ("~", "+")


def report_day(report_path: Path, series: str, date: str, commit: str) -> dict:
    """A history-day-shaped record of a BP-3 benchmark report (tools/bench/report.py): each of
    our rungs' BD-J against x264 medium on the series' common clip set and its speed
    (megapixels per CPU-second, normalised by the box's timing references), from the same
    encodes. Refused unless the report is CBR (the screens' rate mode) with checked speed."""
    report = json.loads(report_path.read_text())
    if report.get("speed_provisional"):
        raise NoSlope(f"{report_path}: speed is provisional (V6 failures): no slope from it")
    if series not in report["series"]:
        raise NoSlope(f"{report_path}: no series {series!r} (has {sorted(report['series'])})")
    rows = {name.removeprefix(OWN_PREFIX): row
            for name, row in report["series"][series]["families"].items()
            if name.startswith(OWN_PREFIX) and not any(m in name for m in REPORT_EXTRA_ROWS)
            and row.get("bd") and row.get("speed_mpix_per_cpu_s")}
    if not rows:
        raise NoSlope(f"{report_path}: series {series} has none of our rungs with BD and speed")
    cbr = all(row.get("compliance") is not None for row in rows.values())
    values = {}
    for name, row in rows.items():
        codec, _, rung = name.partition("@")
        values[chart_key(codec, rung or MEDIUM)] = {"j": round(row["bd"]["j"]["mean"], 2),
                                                    "fps_x86": row["speed_mpix_per_cpu_s"]}
    return {"date": date, "commit": commit, "values": values, "codec_fronts": codec_fronts(values),
            TIMING_MODE_KEY: SCREEN_RATE_MODE if cbr else "qp"}


def codec_fronts(values: dict[str, dict]) -> dict[str, list[str]]:
    """Each codec's Pareto front over its rungs: no other rung both faster and lower J."""
    fronts: dict[str, list[str]] = {}
    for key, v in values.items():
        codec, _ = rung_of(key)
        dominated = any(o is not v and rung_of(k)[0] == codec and o["fps_x86"] >= v["fps_x86"]
                        and o["j"] <= v["j"] and (o["fps_x86"], o["j"]) != (v["fps_x86"], v["j"])
                        for k, o in values.items())
        if not dominated:
            fronts.setdefault(codec, []).append(key)
    return fronts


def timing_problem(day: dict, days: list[dict]) -> str | None:
    """Why `day`'s fps are not CBR timing (None when they are): its own mode, and that of
    every frame its paired fps are chained to."""
    by_date = {d["date"]: d for d in days}
    seen = day
    while True:
        mode = seen.get(TIMING_MODE_KEY)
        if mode != SCREEN_RATE_MODE:
            how = "no timing rate mode recorded (timed at fixed QP before 2026-11-23)" \
                if mode is None else f"timed in {mode}"
            return (f"frame {day['date']}" + (f" (fps chained to {seen['date']})"
                                              if seen is not day else "") + f": {how}")
        chained = seen.get(CHAINED_KEY)
        if chained is None:
            return None
        if chained not in by_date:
            return f"frame {day['date']}: fps chained to {chained}, which is not in the history"
        seen = by_date[chained]


def generate_from(chosen: dict, source: str, days: list[dict] | None = None) -> dict:
    """The table of one history day (`source` names where it was read); refused unless
    its fps time CBR encodes (timing_problem)."""
    problem = timing_problem(chosen, days or [chosen])
    if problem:
        raise NoSlope(f"{source}: {problem}; the screens judge {SCREEN_RATE_MODE} quality, "
                      f"so their CPU weight may not come from this front")
    return {"source": source, "date": chosen["date"], "commit": chosen["commit"],
            "basis": "BD-J vs x264 medium (28 held-out clips) and c8a fps of every shipped "
                     "rung, both on the chart's CBR encodes; k = -dJ / ((1 + J_rung/100) * "
                     "100 * dln CPU) on the codec's front (nceval.front_k)",
            TIMING_MODE_KEY: SCREEN_RATE_MODE, "rungs": slopes(chosen)}


def shown(path: Path) -> str:
    """`path` as the repository names it (else as given)."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def load(path: Path | None = None) -> dict:
    """The table (default TABLE), refused unless it came from CBR timing (the screens'
    rate mode)."""
    path = path or TABLE
    if not path.is_file():
        raise NoSlope(f"no front-slope table at {path}: python3 -m nceval.front_k generate")
    table = json.loads(path.read_text())
    if table.get(TIMING_MODE_KEY) != SCREEN_RATE_MODE:
        raise NoSlope(f"{path}: slopes of {table.get('date')} from "
                      f"{table.get(TIMING_MODE_KEY) or 'fixed-QP'} timing, while the screens "
                      f"judge {SCREEN_RATE_MODE} quality (owner 2026-11-23: never combine "
                      f"different measurements); regenerate it from a CBR-timed history frame "
                      f"(python3 -m nceval.front_k generate) or give --k")
    return table


def describe(table: dict) -> str:
    return (f"front slopes of {table['date']} (build {table['commit']}, {table['source']}, "
            f"{shown(TABLE)})")


def rung_k(codec: str, rung: str, table: dict | None = None) -> tuple[float, str]:
    """(k, its source) for a change at `codec`'s `rung`, from the checked-in table."""
    table = table or load()
    entry = table["rungs"].get(codec, {}).get(rung)
    if entry is None:
        raise NoSlope(f"{codec}@{rung}: not in {shown(TABLE)} ({describe(table)}); "
                      f"regenerate it (python3 -m nceval.front_k generate) or give --k")
    slow, fast = entry["chord"]
    return entry["k"], f"front slope {slow} -> {fast}, {describe(table)}"


def resolve(codec: str, rung: str | None, explicit: float | None) -> tuple[float, str]:
    """The k a screen charges, and its source: an explicit --k (said loudly, as it is
    not the front's), else the rung's front slope."""
    if explicit is not None:
        front = ""
        if rung is not None:
            try:
                front = f"; the front slope at {codec}@{rung} is {rung_k(codec, rung)[0]:.4g}"
            except NoSlope:
                front = f"; {codec}@{rung} has no front slope"
        print(f"!!! --k {explicit:g} overrides the front slope{front}: this verdict does not "
              f"say whether the change improves the front", file=sys.stderr, flush=True)
        return explicit, f"--k {explicit:g} (explicit, not the front slope)"
    if rung is None:
        raise NoSlope(f"no rung to price CPU at: give --preset RUNG (its front slope) or --k")
    return rung_k(codec, rung)


def fixed_k_before(rung: str) -> float:
    return FIXED_K_BEFORE.get(rung, FIXED_K_SPEED_RUNGS)


def show(table: dict) -> str:
    lines = [describe(table),
             f"{'codec':6} {'rung':9} {'J':>7} {'fps':>6} {'front k':>8} {'fixed k':>8} "
             f"{'->slower':>8} {'->faster':>8}  chord"]
    fmt = lambda v: f"{v:8.4f}" if v is not None else f"{'-':>8}"
    for codec, rungs in table["rungs"].items():
        for rung, e in sorted(rungs.items(), key=lambda r: r[1]["fps"]):
            lines.append(f"{codec:6} {rung:9} {e['j']:+7.2f} {e['fps']:6.1f} {e['k']:8.4f} "
                         f"{fixed_k_before(rung):8.2f} {fmt(e['toward_slower'])} "
                         f"{fmt(e['toward_faster'])}  {' -> '.join(e['chord'])}"
                         + ("" if e["on_front"] else "  (off the front)"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python3 -m nceval.front_k", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help=f"write {TABLE.relative_to(REPO)} from a history frame "
                       "or a rendered chart")
    g.add_argument("--history", type=Path, default=HISTORY)
    g.add_argument("--day", help="the frame's date (default: the last)")
    g.add_argument("--report", type=Path, help="a BP-3 benchmark report.json (the 'ours' CBR "
                   "run; with --series, --date and --commit): the BP-3.10 source")
    g.add_argument("--series", default="c8i", help="the report's machine series (default c8i)")
    g.add_argument("--chart", type=Path, help="a rendered chart's x86 CSV instead of a history "
                   "frame (with --timing, --date and --commit)")
    g.add_argument("--timing", type=Path, help="the chart's x86 timing record (CBR)")
    g.add_argument("--date", help="the chart's measurement date")
    g.add_argument("--commit", help="the build its rungs were measured on")
    g.add_argument("--out", type=Path, default=TABLE)
    sub.add_parser("show", help="print the table against the fixed k used before")
    args = p.parse_args(argv)
    if args.command == "generate":
        if args.report:
            if not (args.date and args.commit):
                raise SystemExit("--report needs --date and --commit")
            table = generate_from(report_day(args.report, args.series, args.date, args.commit),
                                  shown(args.report))
            table["basis"] = (f"BP-3.10 benchmark report, series {args.series}: BD-J vs x264 "
                              "medium on the series' common clip set and speed (Mpx per "
                              "CPU-s, reference-normalised; in the fps fields) of every shipped "
                              "rung, from the same CBR encodes; k = -dJ / ((1 + J_rung/100) * "
                              "100 * dln CPU) on the codec's front (nceval.front_k)")
        elif args.chart:
            if not (args.timing and args.date and args.commit):
                raise SystemExit("--chart needs --timing, --date and --commit")
            table = generate_from(chart_day(args.chart, args.timing, args.date, args.commit),
                                  shown(args.chart))
            table["basis"] = ("BD-J vs x264 medium on the chart's clips and c8a fps of every "
                              "shipped rung, timed on every one of the chart's own CBR quality "
                              "encodes; k = -dJ / ((1 + J_rung/100) * 100 * dln CPU) on the "
                              "codec's front (nceval.front_k)")
        else:
            table = generate(args.history, args.day)
        args.out.write_text(json.dumps(table, indent=1) + "\n")
        print(f"wrote {args.out}")
    else:
        table = load()
    print(show(table))


if __name__ == "__main__":
    main()
