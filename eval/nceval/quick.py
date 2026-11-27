"""Stage 1 of the three-stage checks: the quick screen of one idea (KILL or CONTINUE).

    python3 -m nceval.quick --codec av1 --preset fast --param a.b=1 [--param ...] \\
        [--base-param a.b=1 ...] [--cpu-plan PLAN | --cpu-summary SUMMARY]
        [--key-shares 150,300] [--binary BIN] [--json OUT]

Anchor: the preset as it is on the head build (tools/dev/binary_for.sh HEAD,
or --binary); candidate: the preset plus the --param values. Two measurements
run side by side, each on its own share of the workers:

  L1     the pool's quick level on the day's draw (nceval.levels, 24 clips,
         150 frames, single-point): the broad ranker.
  probe  the final check's protocol in miniature, at every key share
         (nceval.key_shares: each share's window from frame 0 with a key every
         share period, CPU at that share), PROBE_RUNGS of the CBR
         ladder, on PROBE_CLIPS clips with that many frames, one per
         PROBE_CLASSES class first (the classes whose long windows killed
         four L1/L2 passers in the 600-frame final check on 2026-11-21).
         The clips come from the day's L2 families (L1's when a class has
         none there), so the probe is disjoint from L1 and from L3, which
         the bundle's final check measures. Each candidate point is priced
         on the anchor's points at the same rungs (jreal.point_quality); the
         periodic-key pulse is the share's keys' (jreal.compare_clip).

CPU (nceval.decision_cpu): `--cpu-plan PLAN` writes the declared subset of
this screen's own candidate encodes (L1's and each probe share's: same clips,
start frames, rates, frames and key period) and exits;
`tools/dev/aws_cpu_ab.py --decision-plan PLAN` times them on rented cores;
`--cpu-summary SUMMARY` charges L1 and each probe share the CPU of its own
encodes, on the plan's draw (a summary of other encodes is refused).

Verdict (it ranks and kills; it never adopts): KILL when L1 is confidently
worse (nceval.interval's L1 stop, at L1's own CPU), or at any key share
the probe's interval is above
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

from . import checkrun, cloud, decision_cpu, interval, jreal, key_shares, levels, levers, pool
from .cbr import DEFAULT_INTRA_PERIOD, Bench, Family, Point
from .clips import Clip
from .config import load_paths
from .gatepolicy import EVALUATION_GATE, GatePolicy
from .realuse import DEFAULT_FRAMES, DEFAULT_WORKERS, Runner

#: The two lowest ladder rungs: the low-rate region we target, where a periodic key's
#: budget is tightest (the same rungs as the periodic-key passes, realuse.PERIODIC_RUNGS).
PROBE_RUNGS = (0, 1)
PROBE_LADDER = checkrun.rungs(PROBE_RUNGS)
#: The screen's decision name in its CPU plan and aws_cpu_ab summary.
CPU_DECISION = "quick"
PROBE_CLIPS = 6
PROBE_CLASSES = ("static", "lowlight", "bodycam", "drone", "videocall")
#: Loose kill thresholds (stage 1 ranks; the bundle's L3 and final check decide):
#: twice the final check's worst-clip pulse guard and class guard.
PROBE_PULSE_WORST_DB = 2 * jreal.PULSE_WORST_CLIP_DB
PROBE_CLASS_KILL_PCT = 2 * jreal.CLASS_GUARD_PCT
KILL, CONTINUE = "KILL", "CONTINUE"
#: The stage's wall-time limit (minutes): its target is 5, so this is a stall.
DEADLINE_MINUTES = 20.0


def probe_clips(draw: pool.Draw, frames: int) -> list[Clip]:
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


def probe_rows(runner: Runner, anchor: Family, test: Family, clips: list[Clip],
               share: key_shares.KeyShare) -> dict:
    """Per clip {"class", "quality" (= "quality_low"), "pulse"} of the probe at `share`."""
    curves = runner.measure(f"probe (key share {share.label})", [anchor, test], clips,
                            share.window, key_frames=share.period, ladder=PROBE_LADDER)
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


def mean_of(values: list[float | None]) -> float | None:
    """The equal-weight mean over key shares (None when any is missing)."""
    return None if not values or None in values else sum(values) / len(values)


def worst_of_shares(shares: dict[int, dict]) -> dict:
    """The probe's pulse, worst-clip pulse and per-class quality, each the worst over
    the key shares (what nceval.train ranks and groups on)."""
    worst = lambda values: max((v for v in values if v is not None), default=None)
    classes: dict[str, float] = {}
    for share in shares.values():
        for name, value in share["classes"].items():
            classes[name] = max(value, classes.get(name, value))
    return {"pulse": worst(s["pulse"] for s in shares.values()),
            "pulse_max": worst(s["pulse_max"] for s in shares.values()),
            "classes": dict(sorted(classes.items()))}


def probe_summary(rows: dict, dcpu: float | None, k: float) -> dict:
    pulses = class_means(rows, "pulse")
    values = [r["pulse"] for r in rows.values() if r.get("pulse") is not None]
    return {**summarise_rows(rows, dcpu, k),
            "pulse": sum(pulses.values()) / len(pulses) if pulses else None,
            "pulse_max": max(values) if values else None, "pulse_classes": pulses,
            "clips": sorted(rows)}


def verdict(l1: dict, probes: dict[int, dict]) -> tuple[str, list[str]]:
    """KILL when L1 or the probe at any key share kills (`probe_reasons`)."""
    reasons = []
    if l1["decision"] == interval.WORSE:
        reasons.append(f"L1 confidently worse: {interval.describe(l1)}")
    for period, probe in sorted(probes.items()):
        reasons += [f"1/{period}: {r}" for r in probe_reasons(probe)]
    return (KILL if reasons else CONTINUE), reasons


def probe_reasons(probe: dict) -> list[str]:
    reasons = []
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
    return reasons


def l1_args(draw: pool.Draw, cpu: decision_cpu.Charge | None, k: float) -> argparse.Namespace:
    """L1 as nceval.levels runs it for this screen (and its CPU plan lists it)."""
    return argparse.Namespace(frames=str(DEFAULT_FRAMES), rotation=draw.key, max_level=1,
                              levels=(pool.QUICK,), cpu=cpu, dcpu=None, k=k,
                              all_stages=False, final_estimator=levels.SINGLE,
                              final_frames=None)


def quality_jobs(runner: Runner, test: Family, draw: pool.Draw,
                 shares: list[key_shares.KeyShare]) -> list[tuple[Bench, Point]]:
    """The candidate's encodes of this screen: L1's (levels.quality_jobs) and each probe
    share's (probe_rows' own measurement)."""
    jobs = levels.quality_jobs(runner, test, draw, (pool.QUICK,), l1_args(draw, None, 0.0))
    for share in shares:
        jobs += runner.points([test], probe_clips(draw, share.window), share.window, None,
                              key_frames=share.period, ladder=PROBE_LADDER)
    return jobs


def setting(args: argparse.Namespace) -> tuple[Path, dict, dict, Family, Family]:
    """(binary, base values, change, anchor, candidate) of the idea screened."""
    binary = Path(args.binary).resolve() if args.binary else checkrun.landed_binary()
    presets = checkrun.presets_beside(binary)
    base = {**checkrun.preset_params(presets, args.codec, args.preset),
            **levers.pairs_to_dict(args.base_param)}
    change = levers.pairs_to_dict(args.param)
    checkrun.check_known(binary, args.codec, {**levers.pairs_to_dict(args.base_param), **change})
    anchor, test = checkrun.families(args.codec, binary, base, {**base, **change})
    return binary, base, change, anchor, test


def cpu_plan_entry(args: argparse.Namespace) -> dict:
    """This screen's CPU plan (nceval.decision_cpu) on its draw."""
    paths = load_paths()
    _, _, _, anchor, test = setting(args)
    draw = pool.draw(pool.load_pool(paths, levels.usable), pool.rotation_key(args.rotation),
                     DEFAULT_FRAMES)
    runner = Runner(paths, GatePolicy(EVALUATION_GATE.value), 1, True)
    jobs = quality_jobs(runner, test, draw, key_shares.parse_periods(args.key_shares))
    return decision_cpu.plan_entry(CPU_DECISION, args.codec, args.preset, draw.key, jobs,
                                   anchor, test, decision_cpu.build_of(anchor.tool),
                                   decision_cpu.build_of(test.tool))


def resolve_cpu(args: argparse.Namespace) -> decision_cpu.Charge | None:
    """The verified CPU charge (--cpu-summary, or --cpu-charge on a box), else None;
    --cpu-plan writes the plan and exits. Sets args.rotation to the plan's draw."""
    if args.cpu_summary:
        found = decision_cpu.load_charge(Path(args.cpu_summary), CPU_DECISION)
        rotation = decision_cpu.plan_rotation(Path(args.cpu_summary), CPU_DECISION)
        if args.rotation and args.rotation != rotation:
            raise decision_cpu.CpuMismatch(f"--rotation {args.rotation}: the CPU was measured "
                                           f"on draw {rotation}")
        args.rotation = rotation
    elif args.cpu_charge:
        found = decision_cpu.parse(args.cpu_charge)
    else:
        if args.cpu_plan:
            decision_cpu.write_plan(Path(args.cpu_plan), [cpu_plan_entry(args)])
            sys.exit(0)
        return None
    return decision_cpu.verify(found, cpu_plan_entry(args)["digest"], "nceval.quick")


def l1_kills(l1_result: dict) -> bool:
    return l1_result["levels"][pool.QUICK]["decision"] == interval.WORSE


def staged(run_l1, run_probes: dict) -> tuple[dict, dict]:
    """L1, then each key share's probe (side by side) unless L1 kills: (L1's result,
    {period: probe rows}, empty when skipped)."""
    l1_result = run_l1()
    if l1_kills(l1_result):
        print("probes skipped: L1 kills (staged probes)", flush=True)
        return l1_result, {}
    with ThreadPoolExecutor(len(run_probes) or 1) as threads:
        jobs = {period: threads.submit(run) for period, run in run_probes.items()}
        return l1_result, {period: job.result() for period, job in jobs.items()}


def quick(args: argparse.Namespace) -> dict:
    paths = load_paths()
    binary, base, change, anchor, test = setting(args)
    k, k_source = checkrun.preset_k(args.codec, args.preset, args.k)
    shares = key_shares.parse_periods(args.key_shares)
    cpu = args.cpu
    dcpu = None if cpu is None else cpu.dcpu(DEFAULT_FRAMES, DEFAULT_INTRA_PERIOD)
    dcpu_shares = {s.period: None if cpu is None else cpu.dcpu(s.window, s.period)
                   for s in shares}
    draw = pool.draw(pool.load_pool(paths, levels.usable), pool.rotation_key(args.rotation),
                     DEFAULT_FRAMES)
    probes = {s.period: probe_clips(draw, s.window) for s in shares}
    # L1 then the probes (staged, below): each stage gets every worker
    runners = [Runner(paths, GatePolicy(EVALUATION_GATE.value), args.workers, args.over_budget)
               for _ in range(2)]
    l1_clips = draw.levels[pool.QUICK]
    l1_cost = checkrun.price(runners[0], anchor, test, l1_clips, DEFAULT_FRAMES,
                             test_ladder=checkrun.single_rung(l1_clips))
    probe_costs = {s.period: checkrun.price(runners[1], anchor, test, probes[s.period],
                                            s.window, PROBE_LADDER,
                                            key_frames=s.period) for s in shares}
    dcpu_text = decision_cpu.describe(cpu)
    print(f"quick {args.codec}@{args.preset} {' '.join(checkrun.pairs(change))} (k {k:.4g}, "
          f"dCPU {dcpu_text}), draw {draw.key}, build {binary}:\n"
          f"  k     {k_source}\n"
          f"  L1     {len(l1_clips)} clips, {DEFAULT_FRAMES} f, single-point: "
          f"{l1_cost[1]} new points\n"
          + "".join(
              f"  probe  {s.describe()}, {len(probes[s.period])} clips "
              f"({', '.join(f'{c.name}/{c.content_class}' for c in probes[s.period])}), "
              f"rungs {PROBE_RUNGS}: {probe_costs[s.period][1]} new points\n" for s in shares)
          + f"  L1, then the probes unless L1 kills, on {args.workers} workers: up to ~"
          f"{checkrun.minutes(l1_cost[0] + sum(c[0] for c in probe_costs.values())):.1f}"
          f" min wall expected; abort after {DEADLINE_MINUTES:g} min",
          flush=True)
    screen_args = l1_args(draw, cpu, k)
    started = time.monotonic()
    with checkrun.Deadline("nceval.quick", DEADLINE_MINUTES):
        # Staged probes (owner, 2026-11-25): the probes are ~83 % of a quick's frames, and an
        # L1 KILL is a KILL whatever they show (`verdict`), so they run only when L1 does
        # not kill: the verdict is the same as running them beside L1, and every candidate
        # that passes L1 still gets every key share (the key-share rule for adoption).
        l1_result, share_rows = staged(
            lambda: levels.run(runners[0], (anchor, test), paths, screen_args),
            {s.period: (lambda s=s: probe_rows(runners[1], anchor, test, probes[s.period], s))
             for s in shares})
        l1_killed = l1_kills(l1_result)
    stage = l1_result["stages"][0]
    l1 = {**summarise_rows(stage["per_clip"], dcpu, k),
          "decision": l1_result["levels"][pool.QUICK]["decision"],
          "guards": l1_result["levels"][pool.QUICK]["guards"]}
    probe_shares = {p: {**probe_summary(rows, dcpu_shares[p], k), "dcpu": dcpu_shares[p],
                        "frames": key_shares.KeyShare(p).window, "per_clip": rows}
                    for p, rows in share_rows.items()}
    probe_stats = {**key_shares.combined_stats(probe_shares),
                   "quality": mean_of([s["quality"] for s in probe_shares.values()]),
                   **worst_of_shares(probe_shares), "shares": probe_shares}
    decision, reasons = verdict(l1, probe_shares)
    wall = time.monotonic() - started
    report = {"status": "done", "stage": "quick", "verdict": decision, "reasons": reasons,
              "codec": args.codec, "preset": args.preset, "params": change,
              "binary": str(binary), "rotation": draw.key, "k": k, "k_source": k_source,
              "dcpu": dcpu, "dcpu_shares": dcpu_shares, "cpu": decision_cpu.record(cpu),
              "l1": l1,
              "probe": probe_stats, "probes_skipped": l1_killed, "wall_seconds": round(wall),
              "screen": {"mode": "real", "codec": args.codec, "binary": str(binary),
                         "params": checkrun.pairs({**base, **change}),
                         "anchor_binary": str(binary), "anchor_params": checkrun.pairs(base),
                         "frames": str(DEFAULT_FRAMES), **l1_result}}
    print(format_report(report), flush=True)
    return report


def format_report(report: dict) -> str:
    l1, probe = report["l1"], report["probe"]
    worst = lambda classes: max(classes.items(), key=lambda kv: kv[1], default=("-", 0.0))
    lines = [f"L1    {interval.describe(l1)} -> {l1['decision'].upper()}; worst class "
             f"{worst(l1['classes'])[0]} {worst(l1['classes'])[1]:+.2f}%",
             *(probe_line(period, share) for period, share in sorted(probe["shares"].items())),
             f"probe mean of key shares: {interval.describe(probe)}",
             f"k     {report['k']:.4g}: {report['k_source']}",
             f"quick -> {report['verdict']} ({report['wall_seconds'] / 60:.1f} min)"
             + (": " + "; ".join(report["reasons"]) if report["reasons"] else "")]
    return "\n".join(lines)


def probe_line(period, share: dict) -> str:
    """One key share's probe: J + k dCPU at that share's CPU, pulse, worst class."""
    worst = max(share["classes"].items(), key=lambda kv: kv[1], default=("-", 0.0))
    pulse = "n/a" if share["pulse"] is None else f"{share['pulse']:+.2f} dB (worst clip " \
        f"{share['pulse_max']:+.2f})"
    dcpu = "n/a" if share["dcpu"] is None else f"{share['dcpu']:+.2f}%"
    return (f"probe 1/{period} ({share['frames']} f, dCPU {dcpu}): {interval.describe(share)}; "
            f"pulse {pulse}; worst class {worst[0]} {worst[1]:+.2f}%")


def ledger_rows(report: dict, source: str) -> list[dict]:
    """L1's rows (levers.screen_rows) and one row for the probe."""
    rows = levers.screen_rows(report["screen"], source)
    common = {k: rows[0][k] for k in ("codec", "base", "base_params", "base_values",
                                      "overrides", "change", "commit")} if rows else {
        "codec": report["codec"], "base": report["preset"], "overrides": report["params"]}
    for period, probe in sorted(report["probe"]["shares"].items()):
        rows.append(probe_ledger_row(report, common, period, probe, source))
    return rows


def probe_ledger_row(report: dict, common: dict, period, probe: dict, source: str) -> dict:
    """The ledger row of the probe at one key share (its level names the share)."""
    shift = report["k"] * (probe["dcpu"] or 0.0)  # the row's J and interval exclude CPU
    unshift = lambda value: None if value is None else value - shift
    return levers.make_row(
        **common, rate="cbr", level=f"quick-probe-1/{period}", draw=report["rotation"],
        n=probe["n"], frames=probe["frames"], j=probe["quality"],
        lo=unshift(probe["lo"]), hi=unshift(probe["hi"]),
        guards=report["reasons"], x86_pct=probe["dcpu"],
        x86_how=decision_cpu.LEDGER_HOW if probe["dcpu"] is not None else None,
        k=report["k"], verdict=report["verdict"], source=source, kind="live")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python3 -m nceval.quick", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codec", required=True)
    p.add_argument("--preset", required=True, help="medium, fast, faster, veryfast, slow, slower, ...")
    p.add_argument("--param", action="append", required=True, help="the change, name=value")
    p.add_argument("--base-param", action="append", default=[],
                   help="name=value on both sides (a base not yet a default, e.g. a release "
                   "candidate's rate settings): the change is judged on top of it")
    p.add_argument("--cpu-plan", help="write this screen's CPU plan (its own encodes' "
                   "declared subset, nceval.decision_cpu) here and exit")
    p.add_argument("--cpu-summary", help="tools/dev/aws_cpu_ab.py --decision-plan's summary of "
                   "this screen's plan: L1 and each probe share charged k * dCPU of their own "
                   "encodes (refused if measured on others); the screen runs on its draw")
    p.add_argument("--cpu-charge", help=argparse.SUPPRESS)  # a verified --cpu-summary, boxed
    p.add_argument("--key-shares", help="key periods in frames (default 150,300; "
                   "nceval.key_shares, which must include both)")
    p.add_argument("--k", type=float, help="CPU charge (default: the rung's front slope, "
                   "nceval.front_k; an explicit k is warned about loudly)")
    p.add_argument("--binary", help="build to measure (default: the landed cache's HEAD)")
    p.add_argument("--rotation", help="pool rotation key (default: today, UTC)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--over-budget", action="store_true")
    p.add_argument("--json", help="write the report here (train add --quick-json reads it)")
    p.add_argument("--local", action="store_true",
                   help="measure here (default: a warm rented box, nceval.cloud)")
    args = p.parse_args(argv)
    cloud.check_fresh()
    if not args.binary:  # the plan, the check and the box measure this one build
        args.binary = str(checkrun.landed_binary())
    args.cpu = resolve_cpu(args)
    if args.cpu is not None:
        print(f"dCPU per group: {decision_cpu.describe(args.cpu)}", flush=True)
    if not cloud.runs_here(args.local):
        forwarded = list(argv if argv is not None else sys.argv[1:])
        if "--binary" not in forwarded:
            forwarded += ["--binary", args.binary]
        if args.cpu is not None:
            forwarded = [*decision_cpu.without_local_cpu(forwarded), "--cpu-charge",
                         args.cpu.text(), "--rotation", args.rotation]
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
