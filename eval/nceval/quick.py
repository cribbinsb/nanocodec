"""Stage 1 of the three-stage checks: the quick screen of one idea (KILL or CONTINUE).

    python3 -m nceval.quick --codec av1 --preset fast --param a.b=1 [--param ...] \\
        [--dcpu X86_PCT] [--dcpu-jetson PCT] [--binary BIN] [--json OUT]

Anchor: the preset as it is on the head build (tools/dev/binary_for.sh HEAD,
or --binary); candidate: the preset plus the --param values. Two measurements
run side by side, each on its own share of the workers:

  L1     the pool's quick level on the day's draw (nceval.levels, 24 clips,
         150 frames, single-point): the broad ranker.
  probe  the final check's protocol in miniature: PROBE_FRAMES frames from
         frame 0, a key every PROBE_KEY_SECONDS, PROBE_RUNGS of the CBR
         ladder, on PROBE_CLIPS clips with that many frames, one per
         PROBE_CLASSES class first (the classes whose long windows killed
         four L1/L2 passers in the 600-frame final check on 2026-11-21).
         The clips come from the day's L2 families (L1's when a class has
         none there), so the probe is disjoint from L1 and from L3, which
         the bundle's final check measures. Each candidate point is priced
         on the anchor's points at the same rungs (jreal.point_quality); the
         periodic-key pulse is the 10 s keys' (jreal.compare_clip).

Verdict (it ranks and kills; it never adopts): KILL when L1 is confidently
worse (nceval.interval's L1 stop), the probe's interval is above
+STOP_LOWER_PCT, its class-balanced pulse is over the final check's pulse
guard, one clip pulses over PROBE_PULSE_WORST_DB, or one probe class loses
more than PROBE_CLASS_KILL_PCT. Otherwise CONTINUE: queue the candidate with
`python3 -m nceval.train add ... --quick-json OUT`.

Points stream through the shared engine (nceval.realuse.Runner: machine-wide
slots, the point cache, no decoded files). The anchor's points are shared by
every idea screened on the same head and day, so a screen usually measures
only its 24 + 12 candidate points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import checkrun, cloud, interval, jreal, levels, levers, pool
from .cbr import Family
from .clips import Clip
from .config import load_paths
from .gatepolicy import EVALUATION_GATE, GatePolicy
from .realuse import DEFAULT_FRAMES, DEFAULT_WORKERS, Runner

PROBE_FRAMES = levels.LONG_FRAMES
PROBE_KEY_SECONDS = levels.LONG_KEY_SECONDS
#: The two lowest ladder rungs: the low-rate region we target, where a periodic key's
#: budget is tightest (the same rungs as the periodic-key passes, realuse.PERIODIC_RUNGS).
PROBE_RUNGS = (0, 1)
PROBE_CLIPS = 6
PROBE_CLASSES = ("static", "lowlight", "bodycam", "drone", "videocall")
#: Loose kill thresholds (stage 1 ranks; the bundle's L3 and final check decide):
#: twice the final check's worst-clip pulse guard and class guard.
PROBE_PULSE_WORST_DB = 2 * jreal.PULSE_WORST_CLIP_DB
PROBE_CLASS_KILL_PCT = 2 * jreal.CLASS_GUARD_PCT
KILL, CONTINUE = "KILL", "CONTINUE"
#: The stage's wall-time limit (minutes): its target is 5, so this is a stall.
DEADLINE_MINUTES = 20.0


def probe_clips(draw: pool.Draw, frames: int = PROBE_FRAMES) -> list[Clip]:
    """The probe's clips for the draw: one per PROBE_CLASSES class (L2 families
    first, then L1's), then the rest of L2's long clips by the draw's key."""
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f"probe\0{draw.key}".encode()).digest()[:8], "big"))
    long_of = lambda level: sorted((replace(c, start_frame=0) for c in draw.levels[level]
                                    if c.frames >= frames), key=lambda c: c.name)
    adopt, quick = long_of(pool.ADOPT), long_of(pool.QUICK)
    chosen: list[Clip] = []
    for content_class in PROBE_CLASSES:
        for source in (adopt, quick):
            members = [c for c in source if c.content_class == content_class]
            if members:
                chosen.append(rng.choice(members))
                break
    rest = [c for c in adopt if c not in chosen]
    rng.shuffle(rest)
    return (chosen + rest)[:PROBE_CLIPS]


def probe_rows(runner: Runner, anchor: Family, test: Family, clips: list[Clip]) -> dict:
    """Per clip {"class", "quality" (= "quality_low"), "pulse"} of the probe."""
    curves = runner.measure("probe (long window)", [anchor, test], clips, PROBE_FRAMES,
                            key_seconds=PROBE_KEY_SECONDS, ladder=checkrun.rungs(PROBE_RUNGS))
    refused = {e.split("@")[0] for e in runner.refused.get(test.name, ())}
    rows = {}
    for clip in clips:
        if clip.name in refused:
            continue
        a, t, _, _ = jreal.pair_rungs(curves[(anchor.name, clip.name)],
                                      curves[(test.name, clip.name)])
        qualities = [jreal.point_quality(a, p) for p in t] if len(a) >= 2 else []
        if not qualities or None in qualities:
            print(f"probe: {clip.name}: no paired rate-valid rungs, left out", flush=True)
            continue
        quality = sum(qualities) / len(qualities)
        terms = jreal.compare_clip(a, t, min_points=2)
        rows[clip.name] = {"class": clip.content_class, "quality": quality,
                           "quality_low": quality,
                           # as the final check judges it: only as far as the key got worse
                           "pulse": jreal.key_attributed_pulse(
                               terms["d_periodic_pulse_psnr_y"], terms["d_periodic_key_psnr_y"]),
                           "raw_pulse": terms["d_periodic_pulse_psnr_y"],
                           "key": terms["d_periodic_key_psnr_y"]}
    return rows


def class_means(rows: dict, term: str) -> dict[str, float]:
    classes: dict[str, list[float]] = {}
    for row in rows.values():
        if row.get(term) is not None:
            classes.setdefault(row["class"], []).append(row[term])
    return {name: sum(v) / len(v) for name, v in sorted(classes.items())}


def summarise_rows(rows: dict, dcpu: float | None, k: float) -> dict:
    """Interval with CPU, plus the CPU-free quality and per-class quality."""
    stats = interval.interval(rows, dcpu, k)
    quality = interval.interval(rows, None, k)["mean"]
    return {**stats, "quality": quality, "classes": class_means(rows, "quality")}


def probe_summary(rows: dict, dcpu: float | None, k: float) -> dict:
    pulses = class_means(rows, "pulse")
    values = [r["pulse"] for r in rows.values() if r.get("pulse") is not None]
    return {**summarise_rows(rows, dcpu, k),
            "pulse": sum(pulses.values()) / len(pulses) if pulses else None,
            "pulse_max": max(values) if values else None, "pulse_classes": pulses,
            "clips": sorted(rows)}


def verdict(l1: dict, probe: dict) -> tuple[str, list[str]]:
    reasons = []
    if l1["decision"] == interval.WORSE:
        reasons.append(f"L1 confidently worse: {interval.describe(l1)}")
    if probe["lo"] is not None and probe["lo"] > interval.STOP_LOWER_PCT:
        reasons.append(f"probe confidently worse: {interval.describe(probe)}")
    if probe["pulse"] is not None and probe["pulse"] > jreal.PULSE_GUARD_DB:
        reasons.append(f"probe pulse {probe['pulse']:+.2f} dB > {jreal.PULSE_GUARD_DB:+.2f}")
    if probe["pulse_max"] is not None and probe["pulse_max"] > PROBE_PULSE_WORST_DB:
        reasons.append(f"probe worst-clip pulse {probe['pulse_max']:+.2f} dB > "
                       f"{PROBE_PULSE_WORST_DB:+.2f}")
    for name, value in probe["classes"].items():
        if value > PROBE_CLASS_KILL_PCT:
            reasons.append(f"probe class {name} {value:+.2f}% > {PROBE_CLASS_KILL_PCT:+.2f}%")
    return (KILL if reasons else CONTINUE), reasons


def quick(args: argparse.Namespace) -> dict:
    paths = load_paths()
    binary = Path(args.binary).resolve() if args.binary else checkrun.landed_binary()
    presets = checkrun.presets_beside(binary)
    base = checkrun.preset_params(presets, args.codec, args.preset)
    change = levers.pairs_to_dict(args.param)
    checkrun.check_known(binary, args.codec, change)
    k = args.k if args.k is not None else checkrun.preset_k(args.preset)
    dcpu = checkrun.combined_dcpu(args.dcpu, args.dcpu_jetson)
    anchor, test = checkrun.families(args.codec, binary, base, {**base, **change})
    draw = pool.draw(pool.load_pool(paths, levels.usable), pool.rotation_key(args.rotation),
                     DEFAULT_FRAMES)
    probe = probe_clips(draw)
    half = max(1, args.workers // 2)
    runners = [Runner(paths, GatePolicy(EVALUATION_GATE.value), half, args.over_budget)
               for _ in range(2)]
    l1_clips = draw.levels[pool.QUICK]
    costs = [checkrun.price(runners[0], anchor, test, l1_clips, DEFAULT_FRAMES,
                            test_ladder=checkrun.single_rung(l1_clips)),
             checkrun.price(runners[1], anchor, test, probe, PROBE_FRAMES,
                            checkrun.rungs(PROBE_RUNGS), key_seconds=PROBE_KEY_SECONDS)]
    print(f"quick {args.codec}@{args.preset} {' '.join(checkrun.pairs(change))} (k {k:g}, "
          f"dCPU {dcpu if dcpu is not None else 'n/a'}), draw {draw.key}, build {binary}:\n"
          f"  L1     {len(l1_clips)} clips, {DEFAULT_FRAMES} f, single-point: "
          f"{costs[0][1]} new points\n"
          f"  probe  {len(probe)} clips ({', '.join(f'{c.name}/{c.content_class}' for c in probe)}),"
          f" {PROBE_FRAMES} f, keys every {PROBE_KEY_SECONDS:g} s, rungs {PROBE_RUNGS}: "
          f"{costs[1][1]} new points\n"
          f"  side by side, each on {half} workers: ~{checkrun.minutes(costs[0][0] + costs[1][0]):.1f}"
          f" min wall expected; abort after {DEADLINE_MINUTES:g} min",
          flush=True)
    screen_args = argparse.Namespace(frames=str(DEFAULT_FRAMES), rotation=draw.key,
                                     max_level=1, levels=(pool.QUICK,), dcpu=dcpu, k=k,
                                     all_stages=False, final_estimator=levels.SINGLE,
                                     final_frames=None)
    started = time.monotonic()
    with checkrun.Deadline("nceval.quick", DEADLINE_MINUTES), ThreadPoolExecutor(2) as threads:
        l1_job = threads.submit(levels.run, runners[0], (anchor, test), paths, screen_args)
        probe_job = threads.submit(probe_rows, runners[1], anchor, test, probe)
        l1_result, rows = l1_job.result(), probe_job.result()
    stage = l1_result["stages"][0]
    l1 = {**summarise_rows(stage["per_clip"], dcpu, k),
          "decision": l1_result["levels"][pool.QUICK]["decision"],
          "guards": l1_result["levels"][pool.QUICK]["guards"]}
    probe_stats = probe_summary(rows, dcpu, k)
    decision, reasons = verdict(l1, probe_stats)
    wall = time.monotonic() - started
    report = {"status": "done", "stage": "quick", "verdict": decision, "reasons": reasons,
              "codec": args.codec, "preset": args.preset, "params": change,
              "binary": str(binary), "rotation": draw.key, "k": k, "dcpu": dcpu,
              "dcpu_x86": args.dcpu, "dcpu_jetson": args.dcpu_jetson, "l1": l1,
              "probe": {**probe_stats, "per_clip": rows}, "wall_seconds": round(wall),
              "screen": {"mode": "real", "codec": args.codec, "binary": str(binary),
                         "params": checkrun.pairs({**base, **change}),
                         "anchor_binary": str(binary), "anchor_params": checkrun.pairs(base),
                         "frames": str(DEFAULT_FRAMES), **l1_result}}
    print(format_report(report), flush=True)
    return report


def format_report(report: dict) -> str:
    l1, probe = report["l1"], report["probe"]
    worst = lambda classes: max(classes.items(), key=lambda kv: kv[1], default=("-", 0.0))
    pulse = "n/a" if probe["pulse"] is None else f"{probe['pulse']:+.2f} dB (worst clip " \
        f"{probe['pulse_max']:+.2f})"
    lines = [f"L1    {interval.describe(l1)} -> {l1['decision'].upper()}; worst class "
             f"{worst(l1['classes'])[0]} {worst(l1['classes'])[1]:+.2f}%",
             f"probe {interval.describe(probe)}; pulse {pulse}; worst class "
             f"{worst(probe['classes'])[0]} {worst(probe['classes'])[1]:+.2f}%",
             f"quick -> {report['verdict']} ({report['wall_seconds'] / 60:.1f} min)"
             + (": " + "; ".join(report["reasons"]) if report["reasons"] else "")]
    return "\n".join(lines)


def ledger_rows(report: dict, source: str) -> list[dict]:
    """L1's rows (levers.screen_rows) and one row for the probe."""
    rows = levers.screen_rows(report["screen"], source)
    common = {k: rows[0][k] for k in ("codec", "base", "base_params", "base_values",
                                      "overrides", "change", "commit")} if rows else {
        "codec": report["codec"], "base": report["preset"], "overrides": report["params"]}
    probe = report["probe"]
    shift = report["k"] * (report["dcpu"] or 0.0)  # the row's J and interval exclude CPU
    unshift = lambda value: None if value is None else value - shift
    rows.append(levers.make_row(
        **common, rate="cbr", level="quick-probe", draw=report["rotation"], n=probe["n"],
        frames=PROBE_FRAMES, j=probe["quality"],
        lo=unshift(probe["lo"]), hi=unshift(probe["hi"]),
        guards=report["reasons"], x86_pct=report["dcpu_x86"], jetson_pct=report["dcpu_jetson"],
        x86_how="--dcpu given to nceval.quick" if report["dcpu_x86"] is not None else None,
        jetson_how="--dcpu-jetson given to nceval.quick"
        if report["dcpu_jetson"] is not None else None,
        k=report["k"], verdict=report["verdict"], source=source, kind="live"))
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python3 -m nceval.quick", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codec", required=True)
    p.add_argument("--preset", required=True, help="medium, fast, faster, veryfast, slow, slower, ...")
    p.add_argument("--param", action="append", required=True, help="the change, name=value")
    p.add_argument("--dcpu", type=float, help="x86 CPU change in %% (callgrind)")
    p.add_argument("--dcpu-jetson", type=float, help="Jetson CPU change in %% (cycles)")
    p.add_argument("--k", type=float, help="CPU charge (default: the preset's, "
                   "checkrun.PRESET_K)")
    p.add_argument("--binary", help="build to measure (default: the landed cache's HEAD)")
    p.add_argument("--rotation", help="pool rotation key (default: today, UTC)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--over-budget", action="store_true")
    p.add_argument("--json", help="write the report here (train add --quick-json reads it)")
    p.add_argument("--local", action="store_true",
                   help="measure here (default: a warm rented box, nceval.cloud)")
    args = p.parse_args(argv)
    cloud.check_fresh()
    if not cloud.runs_here(args.local):
        forwarded = list(argv if argv is not None else sys.argv[1:])
        if not args.binary:  # the box has no build cache: name the build it measures
            forwarded += ["--binary", str(checkrun.landed_binary())]
        cloud.offload("nceval.quick", forwarded, args.json, cloud.default_json("quick"))
    try:
        report = quick(args)
    except BaseException as error:
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"status": "error", "reason": f"{type(error).__name__}: {error}"}) + "\n")
        raise
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str) + "\n")
    levers.record(ledger_rows(report, args.json or "nceval.quick"), "nceval.quick")
    sys.exit(0)


if __name__ == "__main__":
    main()
