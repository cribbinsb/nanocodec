#!/usr/bin/env python3
"""The independent test's tables: BD per metric with intervals, and the rate-control audit.

    python3 tools/dev/independent_report.py --clipset FILE --streams DIR --build DIR \\
        --references x264-faster,svtav1-m9,svtav1-m11 --extras 30 --out DIR
    python3 tools/dev/independent_report.py --chart --out DIR      # the chart's own clips

Reads the scored points of a run of tools/dev/aws_heldout_chart.py (the same
options select the same points; nothing is encoded or scored: a point not
in the cache is counted as missing) and writes CSV tables and their markdown
into OUT (eval/nceval/independent.py defines every measure):

With `--local` (the open subset, docs/results/reproducibility.md) the points are
encoded and scored here instead (`--measure` scores what is missing first).

  bd.csv / bd.md          every family vs x264 medium: PSNR-Y, PSNR-YUV, VMAF v1,
                          MS-SSIM and BD-J with 95% bootstrap intervals, on all
                          clips and on each clip group (CLIP_GROUPS), per class
                          (BD-J), beside the published chart's BD-J
  chart_method.csv        BD-J by nceval.chart's own method (chained clips) per group
  rate.csv, frames.csv, tails.csv, strict.csv, variants.csv, ladder.csv, decode.csv
                          the audit (docs/results/rate-control-audit-*.md)

`--chart` reads the published chart's points instead (its clips, its builds per
family as docs/results/bd-vs-speed-builds.json records them) for the audit's
tables that need no extras.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))
import aws_heldout_chart as heldout  # noqa: E402
from nceval import independent as ind  # noqa: E402
from nceval.cbr import Family  # noqa: E402
from nceval.chart import (ANCHOR, OURS, REFERENCES, chart_clips, chart_curves,  # noqa: E402
                          chart_families, low_ladder, low_rung_clips, quality_points)
from nceval.config import load_paths  # noqa: E402
from nceval.gatepolicy import EVALUATION_GATE  # noqa: E402
from nceval.realuse import DEFAULT_FRAMES, LADDER_KBPS, Runner  # noqa: E402
from nceval.cbr import clip_targets  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PUBLISHED_CSV = REPO / "docs/results/bd-vs-speed.csv"
#: Clip groups by name prefix: the never-used sources (eval/clipset/
#: select_independent.py) and the screening pool's held-out clips (yesterday's).
CLIP_GROUPS = {"fresh": "ind_", "pool": "pool_"}
LABELS = dict(OURS + REFERENCES)


def label(name: str) -> str:
    return LABELS.get(name, name)


class Points:
    """Every cached point of the run, as independent.terms, by (family, clip), each
    tagged with its rung set ("ladder", "low", "high")."""

    def __init__(self, runner: Runner, families: list[Family], clips: list, extra_clips: list,
                 variants: list[Family]):
        self.curves: dict[tuple[str, str], list[dict]] = {}
        self.missing: Counter = Counter()
        self.gate_levels: Counter = Counter()
        ladders = {"ladder": lambda c: clip_targets(c, LADDER_KBPS), "low": low_ladder,
                   "high": heldout.high_ladder}
        for rung, ladder in ladders.items():
            self.add(runner.points(families, clips, DEFAULT_FRAMES, None, ladder=ladder), rung)
            if extra_clips:
                self.add(runner.points(variants, extra_clips, DEFAULT_FRAMES, None,
                                       ladder=ladder), rung)

    def add(self, jobs: list, rung: str) -> None:
        for bench, point in jobs:
            if not bench.cached(point):
                self.missing[(point.family.name, rung)] += 1
                continue
            result = bench.measure(point)
            self.gate_levels[(point.family.name, result.get("gate_level", ""))] += 1
            terms = ind.terms(result)
            terms["rung"] = rung
            self.curves.setdefault((point.family.name, point.clip.name), []).append(terms)

    def select(self, rungs: dict[str, set[str]]) -> dict:
        """Curves with, per clip, only the points of the rung sets `rungs[clip]`."""
        return {(f, c): [p for p in pts if p["rung"] in rungs.get(c, ())]
                for (f, c), pts in self.curves.items()}


def main_rungs(points: Points, families: list[Family], clips: list) -> dict[str, set[str]]:
    """The chart's rule: the ladder, and the low rungs on the clips where some family
    has no direct overlap with the anchor on the ladder (chart.low_rung_clips)."""
    ladder = points.select({c.name: {"ladder"} for c in clips})
    # a family with no ladder points at all (not run) does not decide the low rungs
    present = [f for f in families if any(ladder.get((f.name, c.name)) for c in clips)]
    for family in present:
        for clip in clips:
            ladder.setdefault((family.name, clip.name), [])
    low = {c.name for c in low_rung_clips(ladder, present, clips)}
    return {c.name: {"ladder", "low"} if c.name in low else {"ladder"} for c in clips}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def num(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def groups(clips: list) -> dict[str, list[str]]:
    out = {"all": [c.name for c in clips]}
    for name, prefix in CLIP_GROUPS.items():
        chosen = [c.name for c in clips if c.name.startswith(prefix)]
        if chosen:
            out[name] = chosen
    return out


# --- the BD tables ------------------------------------------------------------------------

def bd_tables(points: Points, families: list[Family], clips: list, out: Path) -> None:
    classes = {c.name: c.content_class for c in clips}
    curves = points.select(main_rungs(points, families, clips))
    published = {r["family"]: r for r in csv.DictReader(PUBLISHED_CSV.open())}
    rows, md_rows, class_rows = [], [], []
    for family in families[1:]:
        for group, names in groups(clips).items():
            per_clip = ind.family_bd(curves, ANCHOR, family.name, names)
            summary = ind.summarise(per_clip, classes)
            row = {"family": family.name, "label": label(family.name), "ours": int(family.own),
                   "group": group, "clips": len(names),
                   "bd_j_chart_pct": published.get(family.name, {}).get("bd_j_pct", "")}
            for metric, est in summary.items():
                row.update({f"{metric}_pct": num(est.value), f"{metric}_lo": num(est.low),
                            f"{metric}_hi": num(est.high), f"{metric}_clips": est.clips})
            rows.append(row)
            if group == "all":
                md_rows.append({"Family": label(family.name),
                                **{name: summary[m].text() for m, name in
                                   ((ind.J, "BD-J"), ("psnr_y", "PSNR-Y"),
                                    ("psnr_yuv", "PSNR-YUV"), ("vmaf_v1", "VMAF v1"),
                                    ("ms_ssim", "MS-SSIM"))},
                                "clips (J)": f"{summary[ind.J].clips}/{len(names)}",
                                "chart BD-J": row["bd_j_chart_pct"]})
                by_class: dict[str, list[str]] = {}
                for name in names:
                    by_class.setdefault(classes[name], []).append(name)
                class_row = {"Family": label(family.name)}
                for cls, members in sorted(by_class.items()):
                    values = [per_clip[m][ind.J] for m in members if per_clip[m][ind.J] is not None]
                    class_row[f"{cls} ({len(members)})"] = \
                        f"{statistics.fmean(values):+.1f}" if values else "n/a"
                class_rows.append(class_row)
    md_rows.sort(key=lambda r: float(r["BD-J"].split()[0]) if r["BD-J"] != "n/a" else 1e9)
    write_csv(out / "bd.csv", rows)
    (out / "bd.md").write_text(
        md_table(md_rows, list(md_rows[0])) + "\n" + md_table(class_rows, list(class_rows[0]))
        + "\n" + group_table(rows))


#: Other anchors every family is also compared with, when the run has them: x264
#: medium tuned for PSNR / SSIM (psy optimisations off) instead of its defaults.
ALT_ANCHORS = ("x264-medium~psnr", "x264-medium~ssim")


def alt_anchor_table(points: Points, families: list[Family], clips: list, out: Path) -> None:
    """Every family vs each ALT_ANCHORS family, per metric with intervals."""
    classes = {c.name: c.content_class for c in clips}
    curves = points.select(main_rungs(points, families, clips))
    names = [c.name for c in clips]
    rows, md_rows = [], []
    for alt in [a for a in ALT_ANCHORS if any(f.name == a for f in families)]:
        for family in families:
            if family.name == alt:
                continue
            summary = ind.summarise(ind.family_bd(curves, alt, family.name, names), classes)
            row = {"anchor": alt, "family": family.name}
            for metric, est in summary.items():
                row.update({f"{metric}_pct": num(est.value), f"{metric}_lo": num(est.low),
                            f"{metric}_hi": num(est.high), f"{metric}_clips": est.clips})
            rows.append(row)
            md_rows.append({"vs": alt, "Family": label(family.name),
                            **{n: summary[m].text() for m, n in
                               ((ind.J, "BD-J"), ("psnr_y", "PSNR-Y"), ("psnr_yuv", "PSNR-YUV"),
                                ("vmaf_v1", "VMAF v1"), ("ms_ssim", "MS-SSIM"))}})
    if rows:
        write_csv(out / "alt_anchor.csv", rows)
        (out / "alt_anchor.md").write_text(md_table(md_rows, list(md_rows[0])))


def group_table(rows: list[dict]) -> str:
    """BD-J per clip group, side by side."""
    by_family: dict[str, dict] = {}
    for row in rows:
        entry = by_family.setdefault(row["family"], {"Family": row["label"],
                                                     "chart": row["bd_j_chart_pct"]})
        entry[row["group"]] = (f"{row['j_pct']} [{row['j_lo']}, {row['j_hi']}]"
                               if row["j_pct"] else "n/a")
    entries = sorted(by_family.values(), key=lambda e: float(e["all"].split()[0])
                     if e.get("all", "n/a") != "n/a" else 1e9)
    columns = ["Family", *[g for g in ("all", "fresh", "pool") if g in entries[0]], "chart"]
    return md_table(entries, columns)


def chart_method(runner: Runner, families: list[Family], clips: list, out: Path) -> None:
    """BD-J by the chart's own code (nceval.chart.quality_points) on each clip group."""
    rows = []
    for group, names in groups(clips).items():
        chosen = [c for c in clips if c.name in names]
        quality = quality_points(runner, families, chosen, DEFAULT_FRAMES)
        for p in quality.points:
            rows.append({"family": p.family, "group": group, "bd_j_pct": num(p.bd_j),
                         "bd_psnr_yuv_pct": num(p.bd_psnr), "bd_vmaf_pct": num(p.bd_vmaf),
                         "clips_scored": p.scored, "clips": p.clips,
                         "chained": len(p.chained), "rate_invalid_points": p.rate_invalid,
                         "excluded_clips": len(quality.excluded)})
    write_csv(out / "chart_method.csv", rows)


# --- the audit ----------------------------------------------------------------------------

def cbr_points(points: Points, family: str, rungs: set[str]) -> list[dict]:
    return [p for (f, _), pts in points.curves.items() if f == family for p in pts
            if p["rung"] in rungs]


def audit_tables(points: Points, families: list[Family], clips: list, out: Path) -> None:
    rate_rows, frame_rows, tail_rows = [], [], []
    for family in families:
        ladder = cbr_points(points, family.name, {"ladder"})
        if not ladder:
            continue
        r = ind.rate_audit(ladder)
        rate_rows.append({
            "family": family.name, "points": r["points"],
            "ratio_mean": num(r["ratio"]["mean"], 3), "ratio_median": num(r["ratio"]["median"], 3),
            "ratio_p10": num(r["ratio"]["p10"], 3), "ratio_p90": num(r["ratio"]["p90"], 3),
            "ratio_max": num(r["ratio"]["max"], 3),
            "loan_ratio_median": num(r["loan_ratio"]["median"], 3),
            "over_10pct": r["over_10pct"], "under_10pct": r["under_10pct"],
            "fullness_median": num(r["fullness"]["median"]),
            "fullness_p90": num(r["fullness"]["p90"]), "fullness_max": num(r["fullness"]["max"]),
            "points_over_buffer": r["over_buffer"], "points_over_1_5": r["over_1_5"],
            "violation_s_mean": num(r["violation_seconds"]["mean"]),
            "violation_s_max": num(r["violation_seconds"]["max"]),
            "first_frame_buffers_median": num(r["first_frame_buffers"]["median"]),
            "first_frame_buffers_max": num(r["first_frame_buffers"]["max"]),
            "rate_valid": r["rate_valid"], "valid_by_allowance": r["relies_on_allowance"],
            "strict_valid": r["strict_valid"]})
        f = ind.frames_audit(ladder)
        frame_rows.append({"family": family.name, **{k: v for k, v in f.items() if k != "keys"},
                           "keys_per_stream_max": f["keys"]["max"]})
        t = ind.tails_audit(ladder)
        tail_rows.append({"family": family.name,
                          **{f"{k}_median": num(v["median"]) for k, v in t.items()},
                          **{f"{k}_p10": num(v["p10"]) for k, v in t.items()}})
    write_csv(out / "rate.csv", rate_rows)
    write_csv(out / "frames.csv", frame_rows)
    write_csv(out / "tails.csv", tail_rows)
    matched_tail_table(points, families, clips, out)
    strict_table(points, families, clips, out)
    decode_table(points, out)


def matched_tail_table(points: Points, families: list[Family], clips: list, out: Path) -> None:
    """Tails at matched achieved rate vs x264 medium, class-balanced with intervals."""
    classes = {c.name: c.content_class for c in clips}
    curves = points.select(main_rungs(points, families, clips))
    rows = []
    for family in families[1:]:
        per_clip = {c.name: ind.matched_tails(curves.get((ANCHOR, c.name), []),
                                              curves.get((family.name, c.name), []))
                    for c in clips if curves.get((family.name, c.name))}
        row = {"family": family.name}
        for term in ("bd_worst_psnr_y", "bd_worst_vmaf_v1", "d_first_pulse_psnr_y",
                     "d_psnr_y_stdev"):
            est = ind.bootstrap({c: v[term] for c, v in per_clip.items()
                                 if v[term] is not None}, classes)
            row.update({term: num(est.value), f"{term}_lo": num(est.low),
                        f"{term}_hi": num(est.high), f"{term}_clips": est.clips})
        rows.append(row)
    write_csv(out / "tails_matched.csv", rows)


def strict_table(points: Points, families: list[Family], clips: list, out: Path) -> None:
    """BD-J vs x264 medium on all points and on only the points that meet the strict
    buffer rule (ind.strict_ok, for every encoder alike), same clips."""
    classes = {c.name: c.content_class for c in clips}
    curves = points.select(main_rungs(points, families, clips))
    names = [c.name for c in clips]
    rows = []
    for family in families[1:]:
        every_rows = ind.family_bd(curves, ANCHOR, family.name, names)
        strict_rows = ind.family_bd(curves, ANCHOR, family.name, names, ind.strict_ok)
        both = [n for n in names if strict_rows[n][ind.J] is not None
                and every_rows[n][ind.J] is not None]
        every = ind.summarise(every_rows, classes)[ind.J]
        same = ind.summarise({n: every_rows[n] for n in both}, classes)[ind.J]
        strict = ind.summarise({n: strict_rows[n] for n in both}, classes)[ind.J]
        rows.append({"family": family.name, "bd_j_all": num(every.value),
                     "bd_j_all_clips": every.clips, "bd_j_all_same_clips": num(same.value),
                     "bd_j_strict": num(strict.value), "bd_j_strict_lo": num(strict.low),
                     "bd_j_strict_hi": num(strict.high), "same_clips": len(both),
                     "strict_points": sum(ind.strict_ok(p) for n in names
                                          for p in curves.get((family.name, n), [])),
                     "points": sum(len(curves.get((family.name, n), [])) for n in names)})
    write_csv(out / "strict.csv", rows)


def decode_table(points: Points, out: Path) -> None:
    write_csv(out / "decode.csv", [{"family": f, "gate_level": level, "points": n}
                                   for (f, level), n in sorted(points.gate_levels.items())])


def variant_tables(points: Points, families: list[Family], variants: list[Family],
                   extra_clips: list, out: Path) -> None:
    """On the extras' clips: each variant vs x264 medium and vs its base family; every
    family's BD-J by rung range."""
    classes = {c.name: c.content_class for c in extra_clips}
    names = [c.name for c in extra_clips]
    rungs = main_rungs(points, families, extra_clips)
    curves = points.select(rungs)
    rows = []
    for variant in variants:
        base = variant.name.split("+")[0].split("~")[0]
        vs_anchor = ind.summarise(ind.family_bd(curves, ANCHOR, variant.name, names), classes)
        vs_base = ind.summarise(ind.family_bd(curves, base, variant.name, names), classes)
        base_anchor = ind.summarise(ind.family_bd(curves, ANCHOR, base, names), classes)
        rows.append({"variant": variant.name, "base": base,
                     "base_vs_anchor_j": base_anchor[ind.J].text(),
                     "variant_vs_anchor_j": vs_anchor[ind.J].text(),
                     "variant_vs_base_j": vs_base[ind.J].text(),
                     "variant_vs_base_psnr_yuv": vs_base["psnr_yuv"].text(),
                     "variant_vs_base_vmaf": vs_base["vmaf_v1"].text(),
                     "variant_vs_base_ms_ssim": vs_base["ms_ssim"].text()})
    write_csv(out / "variants.csv", rows)
    by_range = {"chart (ladder + low where needed)": rungs,
                "ladder only": {n: {"ladder"} for n in names},
                "ladder + high": {n: {"ladder", "high"} for n in names}}
    upper = points.select({n: {"ladder", "high"} for n in names})
    ladder_rows = []
    for family in families[1:]:
        row = {"family": family.name}
        for title, chosen in by_range.items():
            est = ind.summarise(ind.family_bd(points.select(chosen), ANCHOR, family.name,
                                              names), classes)[ind.J]
            row[title], row[f"{title} clips"] = est.text(), est.clips
        # the upper range: each clip's top two ladder rungs and the high rungs
        per_clip = {}
        for name in names:
            floor = upper_floor(upper.get((ANCHOR, name), []))
            per_clip.update(ind.family_bd(upper, ANCHOR, family.name, [name],
                                          lambda p, floor=floor: p["target_kbps"] >= floor))
        est = ind.summarise(per_clip, classes)[ind.J]
        row["upper (top 2 ladder + high)"], row["upper clips"] = est.text(), est.clips
        ladder_rows.append(row)
    write_csv(out / "ladder.csv", ladder_rows)


def upper_floor(anchor_points: list[dict]) -> float:
    """The second-highest ladder target of a clip (the upper range's lowest rung)."""
    targets = sorted({p["target_kbps"] for p in anchor_points if p["rung"] == "ladder"})
    return targets[-2] if len(targets) >= 2 else 0.0


# --- the run's points ---------------------------------------------------------------------

def chart_setup(workers: int, fallback: Path | None) -> tuple[Runner, list[Family], list]:
    """The published chart's families on their recorded builds, and its clips; a
    family whose recorded build is gone from the builds directory runs on
    `fallback` (--build), and is reported."""
    record = json.loads(heldout.BUILDS_FILE.read_text())
    root = heldout.BUILDS_ROOT
    base = root / record["base"]
    dirs = {}
    for family, build in record["families"].items():
        directory = root / build
        if not (directory / heldout.SCORING_BINARY).is_file():
            if fallback is None:
                raise SystemExit(f"{family}: build {build} is gone; pass --build")
            print(f"{family}: build {build} is gone, using {fallback.name}", flush=True)
            directory = fallback
        dirs[family] = directory
    binaries = {f: d / heldout.SCORING_BINARY for f, d in dirs.items()}
    presets = {f: d / "presets" for f, d in dirs.items()}
    families = chart_families(base / heldout.SCORING_BINARY, base / "presets", None,
                              binaries, presets)
    return Runner(load_paths(), EVALUATION_GATE, workers, over_budget=True), families, \
        chart_clips(None)


def chart_extras(build: Path, references: list[str], chart: list[Family]) -> list[Family]:
    """On the chart's clips: our Default of each codec on `build` (`<family>=<build>`)
    and its strict variant (aws_heldout_chart.EXTRA_VARIANTS), and `references`."""
    from dataclasses import replace
    from nceval.cbr import preset_anchor
    from nceval.realuse import own_family
    families = []
    for name in heldout.OWN_DEFAULTS:
        base = own_family(name, build / heldout.SCORING_BINARY, build / "presets")
        base = replace(base, name=f"{name}={build.name}")
        strict = heldout.EXTRA_VARIANTS["strict"]
        families += [base, replace(base, name=f"{base.name}+strict",
                                   extra=base.extra + tuple(x for p in strict
                                                            for x in ("--param", p)))]
    by_name = {f.name: f for f in chart}
    for name in references:  # a profile of a chart family (openh264's is no preset anchor)
        base, _, profile = name.partition("~")
        families.append(replace(by_name[base], name=name, profile=profile)
                        if profile and base in by_name else preset_anchor(name, load_paths()))
    return families


def measure_chart_extras(runner: Runner, families: list[Family], extras: list[Family],
                         clips: list) -> None:
    """Score the extras' ladder and, on the chart's low-rung clips, their low rungs."""
    runner.measure("chart extras", extras, clips, DEFAULT_FRAMES, budgeted=False)
    probe = Points(runner, families, clips, [], [])
    rungs = main_rungs(probe, families, clips)
    low = [c for c in clips if "low" in rungs[c.name]]
    runner.measure("chart extras low rungs", extras, low, DEFAULT_FRAMES, budgeted=False,
                   ladder=low_ladder)


#: The decode check's sample: per codec, our Default on every chart clip at these
#: real-use ladder indices.
DECODE_RUNGS = (0, 2)
DECODE_COMPARED = ("kbps", "psnr_yuv", "vmaf_v1", "ms_ssim", "frame_bits")


def decode_check(args: argparse.Namespace) -> None:
    """Chart points re-encoded in a scratch work root with the full gate (both
    standard decoders and ours must equal the encoder's reconstruction frame by
    frame), their scores compared with the chart's stored, reconstruction-scored
    results."""
    import os
    from nceval.gatepolicy import GatePolicy
    build = Path(args.build).resolve() if args.build else None
    stored_runner, families, clips = chart_setup(args.workers, build)
    ours = [f for f in families if f.name in heldout.OWN_DEFAULTS]
    stored = {(p.family.name, p.clip.name, p.kbps): b.measure(p)
              for b, p in stored_runner.points(ours, clips, DEFAULT_FRAMES, DECODE_RUNGS)
              if b.cached(p)}
    os.environ["NC_WORK"] = str(args.out / "decode-work")
    runner = Runner(load_paths(), GatePolicy.FULL, args.workers, over_budget=True)
    jobs = [(b, p) for b, p in runner.points(ours, clips, DEFAULT_FRAMES, DECODE_RUNGS)
            if (p.family.name, p.clip.name, p.kbps) in stored]
    results = runner.run_jobs(jobs)
    rows = []
    for (_, point), result in zip(jobs, results):
        old = stored[(point.family.name, point.clip.name, point.kbps)]
        row = {"family": point.family.name, "clip": point.clip.name, "kbps": point.kbps,
               "gate_level": result["gate_level"] if result else "FAILED"}
        for key in DECODE_COMPARED:
            row[f"{key}_equal"] = int(bool(result) and result.get(key) == old.get(key))
        row["ms_ssim_diff"] = (abs(result["ms_ssim"] - old["ms_ssim"])
                               if result and "ms_ssim" in old else "")
        rows.append(row)
    write_csv(args.out / "decode_check.csv", rows)
    print(f"{len(rows)} points; all equal: "
          + ", ".join(f"{k} {sum(r[f'{k}_equal'] for r in rows)}" for k in DECODE_COMPARED)
          + f"; gate levels {Counter(r['gate_level'] for r in rows)}")


#: The metrics cross-check's sample size and seed (cloud CPU libvmaf vs local VMAFx).
GPU_CHECK_POINTS = 200
GPU_CHECK_SEED = "independent-gpu-check"
GPU_COMPARED = ("psnr_yuv", "vmaf_v1", "ms_ssim")


def gpu_check(args: argparse.Namespace) -> None:
    """GPU_CHECK_POINTS of the run's cloud-scored points (a fixed hashed draw over all
    ladder points) scored again here from the same streams with the default (VMAFx
    GPU) metrics in a scratch work root; the differences per metric."""
    import hashlib
    import os
    run = heldout.setup(args, args.workers)
    jobs = [(b, p) for b, p in run.runner.points(run.families, run.clips, DEFAULT_FRAMES, None)
            if b.cached(p)]
    rank = lambda job: hashlib.sha256(  # noqa: E731
        f"{GPU_CHECK_SEED}/{job[1].family.name}/{job[1].clip.name}/{job[1].kbps}".encode()
    ).hexdigest()
    sample = sorted(jobs, key=rank)[:GPU_CHECK_POINTS]
    cloud = [b.measure(p) for b, p in sample]
    os.environ["NC_WORK"] = str(args.out / "gpu-work")
    os.environ.pop("NC_METRICS_BACKEND", None)
    local_args = argparse.Namespace(**{**vars(args), "workers": args.workers})
    local = heldout.setup(local_args, args.workers)
    by_key = {(p.family.name, p.clip.name, p.kbps): (b, p) for b, p in
              local.runner.points(local.families, local.clips, DEFAULT_FRAMES, None)}
    local_jobs = [by_key[(p.family.name, p.clip.name, p.kbps)] for _, p in sample]
    results = local.runner.run_jobs(local_jobs)
    rows = []
    for (_, point), old, new in zip(sample, cloud, results):
        row = {"family": point.family.name, "clip": point.clip.name, "kbps": point.kbps}
        for key in GPU_COMPARED:
            row[f"{key}_cloud"] = old[key]
            row[f"{key}_gpu"] = new[key] if new else ""
            row[f"{key}_diff"] = abs(new[key] - old[key]) if new else ""
        row["frame_bits_equal"] = int(bool(new) and new["frame_bits"] == old["frame_bits"])
        rows.append(row)
    write_csv(args.out / "gpu_check.csv", rows)
    for key in GPU_COMPARED:
        diffs = [r[f"{key}_diff"] for r in rows if r[f"{key}_diff"] != ""]
        print(f"{key}: {len(diffs)} points, max |diff| {max(diffs):.3g}, "
              f"mean {statistics.fmean(diffs):.3g}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chart", action="store_true", help="the published chart's points")
    parser.add_argument("--clipset")
    parser.add_argument("--streams", default=str(heldout.DEFAULT_STREAMS))
    parser.add_argument("--build")
    parser.add_argument("--references")
    parser.add_argument("--families")
    parser.add_argument("--clips", type=int)
    parser.add_argument("--extras", type=int, default=0)
    parser.add_argument("--session")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", default="bd,chart,audit,variants")
    parser.add_argument("--chart-extras", action="store_true",
                        help="with --chart: our Defaults on --build, strict, --references")
    parser.add_argument("--measure", action="store_true",
                        help="with --chart-extras or --local: score the points first")
    parser.add_argument("--local", action="store_true",
                        help="encode and score here (no --streams; e.g. the open subset)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tables = set(args.tables.split(","))
    if "decode-check" in tables:
        decode_check(args)
        return
    if "gpu-check" in tables:
        gpu_check(args)
        return
    if args.chart:
        build = Path(args.build).resolve() if args.build else None
        runner, families, clips = chart_setup(args.workers, build)
        extra_clips, variants = [], []
        if args.chart_extras:
            extras = chart_extras(build, args.references.split(",") if args.references
                                  else [], families)
            if args.measure:
                measure_chart_extras(runner, families, extras, clips)
            families = families + extras
            variants = [f for f in extras if "+" in f.name or "~" in f.name]
            extra_clips = clips
    else:
        run = heldout.setup(args, args.workers)
        runner, families, clips = run.runner, run.families, run.clips
        extra_clips, variants = run.extra_clips, run.variants
        if args.local:  # encoded and scored here (the open subset), not from --streams
            runner = Runner(load_paths(), EVALUATION_GATE, args.workers, over_budget=True)
            if args.measure:
                chart_curves(runner, families, clips, DEFAULT_FRAMES, "local run")
    points = Points(runner, families, clips, [] if args.chart else extra_clips, variants)
    print(f"{sum(len(v) for v in points.curves.values())} points; missing: "
          f"{dict(points.missing) or 'none'}", flush=True)
    if "bd" in tables:
        bd_tables(points, families, clips, args.out)
        alt_anchor_table(points, families, clips, args.out)
    if "chart" in tables and not args.chart:
        chart_method(runner, families, clips, args.out)
    if "audit" in tables:
        audit_tables(points, list(dict.fromkeys(families + variants)), clips, args.out)
    if "variants" in tables and variants:
        main_families = [f for f in families if f not in variants]
        variant_tables(points, main_families, variants, extra_clips, args.out)
    print(f"wrote {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    main(sys.argv[1:])
