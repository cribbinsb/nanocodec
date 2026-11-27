"""The shared A/B screen: candidate vs anchor encoder settings, staged and cached.

Two modes (`--mode`):

  real  (default) real-use rate control: both builds in CBR on a target
        ladder, achieved-rate curves, decided by J_real (nceval.realuse,
        nceval.jreal; docs/PROCEDURES.md "Adoption rule: J_real"). Its
        default `--clips adopt` runs the pool levels on the day's rotating
        draw (nceval.levels: L1 quick, L2 adopt, L3 final), each decided on
        a bootstrap interval of the per-clip J deltas (nceval.interval).
  cqp   the FIXED-QP PROXY described below: fast, but it credits temporal
        QP cadence that real rate control does not, measures only PSNR-YUV
        and VMAF v1, and the full curve rather than the low-rate region.
        Use it to rank many ideas quickly; never as adoption evidence.

The rest of this docstring describes the fixed-QP proxy; the real mode uses
the same stages, sigmas, caching and budget splitting.

Every agent used to keep a private screen.py/tr.py/ab.py with its own point
loop, gates, caching and BD code. This is the one runner:

    python3 -m nceval.screen --codec av1 --binary BIN --param a.b=1 \\
        [--anchor-binary BIN0] [--anchor-param ...] [--clips train|val|a,b,c] \\
        [--no-staged] [--frames 150|full] [--qps 40,65,...] [--gate off|sample|full] [--json OUT]

Staged (the default for `--clips train` and `--clips val`): the BD-rate over
the whole set is estimated hierarchically, stopping as soon as the answer is
clear. TRAIN (screens, sample gate by default):

    stage    clips  frames  QPs per clip   sigma of the estimate of the 15-clip BD
    proxy-4  4      60      3 (outer)      2.4 %
    proxy-6  6      150     4              1.0 %
    train    18     150     4              exact (this is the screen's target; 15 camera
                                           clips + 3 pristine texture clips since 2026-11-09)

A stage decides "better" when every metric's BD + MARGIN_SIGMAS*sigma < 0,
"worse" when any metric's BD - MARGIN_SIGMAS*sigma > 0, and otherwise moves
on. The clip sets and sigmas are learned from history (2026-11-08): 288
(candidate, metric) A/B comparisons from five tune studies (h264, hevc, av1;
PSNR-YUV and VMAF-NEG; 150 frames, 4 QPs on each clip's anchor grid); the
sigma is the leave-one-study-out RMSE of the subset mean against the
15-clip mean (sd of that target across candidates: 7.4 %). Keeping the
outer three of four QPs costs nothing measurable on the 4-clip set (RMSE
1.39 vs 1.35 % in-sample). Stages are nested, and every point is cached, so
moving to the next stage only encodes what is new.

VAL-150 (confirmation, every point enters the verdict): val-proxy-5 (one clip per class, outer
3 QPs, provisional sigma 2.3 %) -> all natural VAL clips (outer 3 QPs, exact).
Full-length VAL sweeps are for milestones only.

Points: each (binary, params, clip, frames, QP, gate) point is encoded once
through the streamed engine (tune.encode_point; no decoder by default,
nceval.gatepolicy; under an explicit sample gate our own decoder on every
point and the standard decoders on the first QP of each curve, every point
under full) and cached in ~/nanocodec-data/screen-cache. An undecoded run
reuses a point measured under any gate.
The QP grid is calibrated once per (anchor, clip, frames) to four legal
points (tune.calibrate_quantizers), unless --qps gives a fixed grid.

Besides BD-rate, each clip reports the paired matched-QP signal: Delta-bits
and Delta-PSNR / Delta-VMAF at the two middle measured QPs, which often
shows a clear win or loss on its own.

Screens decide what to try; real-use screens (nceval.realuse) are the
adoption evidence (docs/PROCEDURES.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from . import baseline_state, buildinfo, checkrun, cloud, decision_cpu, front_k, levers, pool, tune
from .bdrate import bd_rate
from .codecs import Codec
from .encoders.nanocodec import MEDIUM
from .config import load_paths
from .fingerprint import measurement_fingerprint
from .budget import (MAX_MINUTES, OverBudget, check as check_budget, estimate_minutes,
                     point_cpu_seconds)
from .cachefile import identity_cached, read_json
from .gatepolicy import (EVALUATION_GATE, GatePolicy, PointCheck, gate_arguments,
                         uniform_check)
from .stages import (ALL_FOUR, MARGIN_SIGMAS, METRICS, PART_FILL, STAGES, VAL_STAGES, Stage,
                     balanced_parts, cache_root, calibrated_plan, decide, history_path)
from .sweep import payload_bytes


MAX_WORKERS = 8
CALIBRATION_PROBES = 8  # typical gated probe encodes to calibrate one anchor grid


@dataclass(frozen=True)
class Side:
    """One encoder configuration of the A/B."""
    binary: Path
    overrides: dict

    def key(self) -> list:
        return [buildinfo.stream_identity(self.binary), sorted(self.overrides.items())]


def _cache_path(kind: str, codec: Codec, clip: tune.ClipRef, identity: dict) -> Path:
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cache_root() / codec.value / f"{kind}-{clip.name}-{key[:20]}.json"


def _cached(kind: str, codec: Codec, clip: tune.ClipRef, identity: dict, compute):
    """A small JSON result cached by its identity, computed once however many ask."""
    return identity_cached(_cache_path(kind, codec, clip, identity), identity, compute)


def _stored(kind: str, codec: Codec, clip: tune.ClipRef, identity: dict):
    """The cached result of `identity`, or None (never computes)."""
    record = read_json(_cache_path(kind, codec, clip, identity))
    expected = json.loads(json.dumps(identity, sort_keys=True))
    return record["value"] if record is not None and record["identity"] == expected else None


def _clip_identity(clip: tune.ClipRef, frames: int) -> dict:
    reference = replace(clip, yuv=clip.clean_reference) if clip.clean_reference else clip
    return {"clip": tune.clip_fingerprint(clip), "reference": tune.clip_fingerprint(reference),
            "frames": frames, "intra_period_seconds": tune.INTRA_PERIOD_SECONDS}


def _scratch():
    work = load_paths().work
    work.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="nc-screen-", dir=work)


def calibrated_grid(side: Side, codec: Codec, clip: tune.ClipRef, frames: int,
                    gate: GatePolicy) -> list[int]:
    """Four legal QPs for this clip, calibrated once on the anchor and cached."""
    def calibrate():
        with _scratch() as work:
            return sorted(tune.calibrate_quantizers(side.binary, codec, clip, frames,
                                                    side.overrides, Path(work), gate))
    return _cached("grid", codec, clip, {"side": side.key(), "codec": codec.value,
                                         **_clip_identity(clip, frames)}, calibrate)


def cached_point(side: Side, codec: Codec, clip: tune.ClipRef, frames: int, quantizer: int,
                 check: PointCheck) -> dict:
    """One gated, scored point: {"q", "kbps", "psnr", "vmaf_v1", "gate"} or excluded."""
    def measure():
        with _scratch() as work:
            stream = Path(work) / f"q{quantizer}{tune.SUFFIX[codec]}"
            point = tune.encode_point(side.binary, codec, clip, quantizer, frames,
                                      side.overrides, stream,
                                      tune.point_metrics(tune.V1_METRIC_PROFILE),
                                      **gate_arguments(check, side.binary))
            kbps = payload_bytes(stream, codec, frames) * 8 / 1000 / (frames / clip.fps)
        if kbps > clip.max_test_kbps:
            return {"q": quantizer, "kbps": kbps, "excluded": True, "gate": point.gate.level}
        summary = point.measurement.summary
        return {"q": quantizer, "kbps": kbps, "psnr": summary["psnr_yuv"],
                "vmaf_v1": summary["vmaf_v1"], "gate": point.gate.level}
    identity = {"side": side.key(), "codec": codec.value, "q": quantizer,
                "standard_decoders": check is PointCheck.STANDARD,
                "measurement": measurement_fingerprint(), **_clip_identity(clip, frames)}
    if check is PointCheck.NONE:
        # Any gate measured the same numbers: reuse a decoded point first.
        for standard in (True, False):
            stored = _stored("point", codec, clip, {**identity, "standard_decoders": standard})
            if stored is not None:
                return stored
        identity["decoded"] = False  # never taken for a decoded point
    return _cached("point", codec, clip, identity, measure)


def clip_result(anchor: list[dict], test: list[dict]) -> dict:
    """BD-rate per metric (None if unusable) and the matched-QP paired signal."""
    legal = lambda points: [p for p in points if not p.get("excluded")]
    a, t = legal(anchor), legal(test)
    result: dict = {"points": [len(a), len(t)],
                    "gate_levels": sorted({p["gate"] for p in test})}
    for metric in METRICS:
        bd = (bd_rate([p["kbps"] for p in a], [p[metric] for p in a],
                      [p["kbps"] for p in t], [p[metric] for p in t], min_points=3)
              if len(a) >= 3 and len(t) >= 3 else None)
        result[metric] = bd.bd_rate_percent if bd is not None and bd.reliable else None
    common = sorted({p["q"] for p in a} & {p["q"] for p in t})
    middle = common[len(common) // 2 - 1: len(common) // 2 + 1] if len(common) >= 2 else []
    by_q = lambda points: {p["q"]: p for p in points}
    pa, pt = by_q(a), by_q(t)
    result["matched"] = ({"qps": middle,
                          "bits_pct": 100 * sum(pt[q]["kbps"] / pa[q]["kbps"] - 1
                                                for q in middle) / len(middle),
                          **{f"d_{m}": sum(pt[q][m] - pa[q][m] for q in middle) / len(middle)
                             for m in METRICS}} if middle else None)
    return result


def record_history(clip_set: str, codec: Codec, stages: list[dict]) -> None:
    """Append each stage's mean BD of a screen; complete screens calibrate sigmas."""
    entry = {"set": clip_set, "codec": codec.value,
             "stages": {s["stage"]: {m: s["summary"][m]["mean"] for m in METRICS}
                        for s in stages}}
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(entry) + "\n")


def select_clips(spec: str) -> list[tune.ClipRef]:
    if spec == "train":
        return tune.proxy_clips()
    if spec == "val":
        return [c for c in tune.load_clips("val") if not c.clean_reference]
    available = {c.name: c for split in ("train", "val") for c in tune.load_clips(split)}
    names = spec.split(",")
    if any(n not in available for n in names):
        # As in real mode: the natural, non-TEST clips of the extra sets.
        from . import realuse  # realuse builds on this module's stages
        for clipset in realuse.EXTRA_SCREEN_CLIPSETS:
            available.update({c.name: c for c in tune.clipset_refs(clipset)
                              if c.split != "test" and c.clean_reference is None})
    missing = [n for n in names if n not in available]
    if missing:
        raise SystemExit(f"unknown TRAIN/VAL clips: {', '.join(missing)}")
    return [available[n] for n in names]


def parse_params(pairs: list[str]) -> dict[str, float]:
    overrides = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not value:
            raise SystemExit(f"--param needs name=value, got {pair!r}")
        overrides[name] = float(value)
    return overrides


#: Set when the --param / --anchor-param lists already hold the rung's preset values
#: (nceval.train passes them itself; the cloud offload passes them expanded).
RUNG_PARAMS_FLAG = "--rung-params-included"


def rung_params(binary: Path, codec: Codec, rung: str | None) -> list[str]:
    """`rung`'s preset values for `binary`, from the presets of its own build (the
    `presets/` beside a tools/dev/binary_for.sh build, as nceval.quick reads them), as
    name=value pairs ([] for medium, the engine defaults). A binary without them is
    refused: a screen of a rung must encode that rung on both sides, never medium."""
    if not rung or rung == MEDIUM:
        return []
    presets = checkrun.presets_beside(binary.resolve())
    return checkrun.pairs(checkrun.preset_params(presets, codec.value, rung))


def missing_pairs(given: list[str], rung_pairs: list[str]) -> list[str]:
    """The rung's pairs the caller did not set (the caller's values win)."""
    named = {pair.partition("=")[0] for pair in given}
    return [p for p in rung_pairs if p.partition("=")[0] not in named]


def apply_rung(args: argparse.Namespace) -> list[str]:
    """Both sides encode `--preset`'s rung: its preset values go under each side's
    --param / --anchor-param (each side from its own build's presets). Returns the
    arguments to add to the command line for a re-run (the offload)."""
    if args.rung_params_included or not args.preset or args.preset == MEDIUM:
        return []
    codec = Codec(args.codec)
    test = missing_pairs(args.param, rung_params(Path(args.binary), codec, args.preset))
    anchor = missing_pairs(args.anchor_param, rung_params(
        Path(args.anchor_binary or args.binary), codec, args.preset))
    args.param = test + args.param
    args.anchor_param = anchor + args.anchor_param
    args.rung_params_included = True
    print(f"rung {args.preset}: {len(test)} preset parameters added to the candidate, "
          f"{len(anchor)} to the anchor", flush=True)
    return ([x for p in test for x in ("--param", p)]
            + [x for p in anchor for x in ("--anchor-param", p)] + [RUNG_PARAMS_FLAG])


class Screen:
    def __init__(self, codec: Codec, anchor: Side, test: Side, frames: str,
                 fixed_qps: tuple[int, ...] | None, gate: GatePolicy, workers: int):
        self.codec, self.anchor, self.test = codec, anchor, test
        self.frames, self.fixed_qps, self.gate = frames, fixed_qps, gate
        self.workers = min(workers, MAX_WORKERS)

    def frames_of(self, clip: tune.ClipRef, cap: int | None = None) -> int:
        frames = clip.frames if self.frames == "full" else min(clip.frames, int(self.frames))
        return min(frames, cap) if cap else frames

    def grid(self, clip: tune.ClipRef, cap: int | None = None) -> list[int]:
        if self.fixed_qps:
            return sorted(self.fixed_qps)
        return calibrated_grid(self.anchor, self.codec, clip, self.frames_of(clip, cap), self.gate)

    def run(self, clips: list[tune.ClipRef], qp_indices: tuple[int, ...] | None,
            frames_cap: int | None = None) -> dict:
        """Per-clip results for these clips (QPs: all of the grid, or these indices of it)."""
        with ThreadPoolExecutor(self.workers) as pool:
            grids = dict(zip((c.name for c in clips),
                             pool.map(lambda c: self.grid(c, frames_cap), clips)))
            jobs = []
            for clip in clips:
                grid = grids[clip.name]
                chosen = grid if qp_indices is None or len(grid) != 4 else [grid[i] for i in qp_indices]
                for index, q in enumerate(chosen):
                    check = (PointCheck.STANDARD if index == 0 and self.gate is GatePolicy.SAMPLE
                             else uniform_check(self.gate))
                    jobs += [(side, clip, q, check) for side in ("anchor", "test")]
            points = list(pool.map(
                lambda j: cached_point(getattr(self, j[0]), self.codec, j[1],
                                       self.frames_of(j[1], frames_cap), j[2], j[3]), jobs))
        curves: dict[str, dict] = {}
        for (side, clip, _, _), point in zip(jobs, points):
            curves.setdefault(clip.name, {"anchor": [], "test": [], "class": clip.content_class})
            curves[clip.name][side].append(point)
        return {name: {"class": c["class"], **clip_result(c["anchor"], c["test"])}
                for name, c in curves.items()}


def summarise(rows: dict) -> dict:
    summary = {}
    for metric in METRICS:
        values = [r[metric] for r in rows.values() if r[metric] is not None]
        classes: dict[str, list[float]] = {}
        for r in rows.values():
            if r[metric] is not None:
                classes.setdefault(r["class"], []).append(r[metric])
        complete = len(values) == len(rows)
        summary[metric] = {"mean": sum(values) / len(values) if values and complete else None,
                           "classes": {k: sum(v) / len(v) for k, v in sorted(classes.items())},
                           "clips_scored": len(values), "clips": len(rows)}
    return summary


def worst_case_cpu(clips: list[tune.ClipRef], plan, runner: "Screen",
                   fixed: tuple[int, ...] | None) -> float:
    """CPU-seconds if every stage runs and nothing is cached (grids included)."""
    cpu = 0.0
    for stage in plan or (Stage("all", None, ALL_FOUR, 0.0),):
        for clip in clips:
            if stage.clips is not None and clip.name not in stage.clips:
                continue
            qps = len(fixed) if fixed else len(stage.qp_indices)
            point = point_cpu_seconds(clip.width, clip.height,
                                      runner.frames_of(clip, stage.frames))
            cpu += 2 * qps * point + (0 if fixed else CALIBRATION_PROBES * point)
    return cpu  # nested stages share points, so this over-counts slightly


def run_stage(runner: "Screen", stage: Stage, chosen: list[tune.ClipRef], staged: bool,
              fixed: tuple[int, ...] | None, over_budget: bool) -> dict:
    """Price the stage, then run it; a stage over budget runs in parts.

    Each stage is priced before it runs, so a clear early stage never pays for
    the later ones (cached points make the price an upper bound). Any stage
    (proxy or whole set) that does not fit runs as the fewest class-balanced
    parts that each fit, each priced and run on its own. Only a part that is a
    single clip and still too big is refused.
    """
    qps = stage.qp_indices if staged else None
    cuts = "--clips with fewer clips, --frames 60, --qps with fewer QPs, or staged mode"
    cost = worst_case_cpu(chosen, (stage,), runner, fixed)
    minutes = estimate_minutes(cost)
    if len(chosen) > 1 and minutes > MAX_MINUTES:
        count = min(len(chosen), math.ceil(minutes / (MAX_MINUTES * PART_FILL)))
        rows = {}
        for index, part in enumerate(balanced_parts(chosen, count), 1):
            print(f"screen stage {stage.name}: part {index}/{count} ({len(part)} clips)",
                  flush=True)
            # Re-priced when it starts: if free slots dropped meanwhile, the
            # part splits again rather than being refused.
            rows.update(run_stage(runner, stage, part, staged, fixed, over_budget))
        return rows
    check_budget(f"screen stage {stage.name} ({len(chosen)} clips)", cost, cuts, over_budget)
    return runner.run(chosen, qps, stage.frames)


def screen(args: argparse.Namespace) -> dict:
    codec = Codec(args.codec)
    test = Side(Path(args.binary).resolve(), parse_params(args.param))
    anchor = Side(Path(args.anchor_binary).resolve() if args.anchor_binary else test.binary,
                  parse_params(args.anchor_param))
    if (test.binary, test.overrides) == (anchor.binary, anchor.overrides):
        raise SystemExit("candidate and anchor are identical")
    fixed = tuple(int(q) for q in args.qps.split(",")) if args.qps else None
    gate = GatePolicy(args.gate or EVALUATION_GATE.value)
    runner = Screen(codec, anchor, test, args.frames, fixed, gate, args.workers)
    clips = select_clips(args.clips)
    plan = {"train": STAGES, "val": VAL_STAGES}.get(args.clips) if args.staged else None
    if plan:
        plan, note = calibrated_plan(args.clips, plan)
        print(f"stages: {note}", flush=True)
    stages = []
    for stage in plan or (Stage("all", None, ALL_FOUR, 0.0),):
        chosen = [c for c in clips if stage.clips is None or c.name in stage.clips]
        rows = run_stage(runner, stage, chosen, plan is not None, fixed, args.over_budget)
        summary = summarise(rows)
        decision = decide({m: summary[m]["mean"] for m in METRICS}, stage.sigma)
        stages.append({"stage": stage.name, "sigma": stage.sigma, "frames": stage.frames,
                       "decision": decision,
                       "per_clip": rows, "summary": summary})
        print(format_stage(stages[-1]), flush=True)
        if decision in ("better", "worse") and not args.all_stages:
            break
    if plan:
        record_history(args.clips, codec, stages)
    return {"codec": codec.value, "binary": str(test.binary), "params": test.overrides,
            "anchor_binary": str(anchor.binary), "anchor_params": anchor.overrides,
            "clips": args.clips, "frames": args.frames, "gate": gate.value,
            "staged": plan is not None, "decision": stages[-1]["decision"], "stages": stages}


def format_stage(stage: dict) -> str:
    pct = lambda v: "   n/a" if v is None else f"{v:+6.2f}%"
    frames = f", {stage['frames']} frames" if stage.get("frames") else ""
    lines = [f"== stage {stage['stage']} (sigma {stage['sigma']:.1f}%{frames})",
             f"{'clip':26s} {'class':9s} {'PSNR-YUV':>8s} {'VMAFv1n':>8s}  matched-QP: bits  dPSNR  dVMAF"]
    for name, row in sorted(stage["per_clip"].items(), key=lambda kv: (kv[1]["class"], kv[0])):
        m = row["matched"]
        paired = (f"{m['bits_pct']:+6.2f}% {m['d_psnr']:+.3f} {m['d_vmaf_v1']:+.3f}"
                  if m else "n/a")
        lines.append(f"{name:26s} {row['class'][:9]:9s} {pct(row['psnr']):>8s} "
                     f"{pct(row['vmaf_v1']):>8s}  {paired}")
    summary = stage["summary"]
    lines.append("mean: " + ", ".join(f"{m} {pct(summary[m]['mean'])}" for m in METRICS))
    margin = MARGIN_SIGMAS * stage["sigma"]
    lines.append(f"decision: {stage['decision'].upper()}"
                 + (f" (margin {margin:.1f}% = {MARGIN_SIGMAS:g} sigma)" if margin else
                    " (the whole set)"))
    return "\n".join(lines)


def pool_levels(spec: str) -> tuple[str, ...]:
    """`--levels quick,final` as pool levels in the pool's order."""
    wanted = {level.strip() for level in spec.split(",") if level.strip()}
    unknown = wanted - set(pool.LEVELS)
    if unknown or not wanted:
        raise argparse.ArgumentTypeError(f"pool levels are {', '.join(pool.LEVELS)}; "
                                         f"got {spec!r}")
    return tuple(level for level in pool.LEVELS if level in wanted)


def cpu_charge(args: argparse.Namespace) -> tuple[float, str]:
    """The CPU weight k and its source: without a CPU measurement none is charged; else
    --k, or --preset's front slope."""
    if args.cpu_summary is None and args.cpu_charge is None:
        return args.k or 0.0, "no CPU measured: no CPU charge"
    if args.k is None and args.preset is None:
        raise SystemExit("a CPU charge needs --preset RUNG (its front slope prices the CPU) "
                         "or an explicit --k")
    return front_k.resolve(args.codec, args.preset, args.k)


#: The screen's decision name in its CPU plan and aws_cpu_ab summary (--cpu-decision:
#: nceval.train names each arm's screens in one plan).
CPU_DECISION = "screen"


def cpu_plan_entry(args: argparse.Namespace) -> dict:
    """The screen's CPU plan (nceval.decision_cpu): the candidate's encodes at the levels
    it runs on its draw, as levels.run measures them."""
    from . import levels, realuse  # realuse builds on this module's stages
    if args.mode != "real" or (args.clips or realuse.ADOPT) != realuse.ADOPT:
        raise SystemExit("a CPU charge is for the adoption levels (--mode real --clips "
                         "adopt); fixed clip sets rank ideas on quality alone")
    anchor, test = realuse.screen_families(args)
    realuse.set_final_estimator(args)
    paths = load_paths()
    runner = realuse.Runner(paths, EVALUATION_GATE, 1, over_budget=True)
    draw = levels.draw_for(paths, args)
    jobs = levels.quality_jobs(runner, test, draw, levels.chosen_levels(args), args)
    return decision_cpu.plan_entry(args.cpu_decision, args.codec, args.preset, draw.key, jobs,
                                   anchor, test, decision_cpu.build_of(anchor.tool),
                                   decision_cpu.build_of(test.tool))


def resolve_cpu(args: argparse.Namespace) -> list[str]:
    """Write the CPU plan (--cpu-plan), or verify the CPU measurement (--cpu-summary,
    or the compact --cpu-charge an offloaded screen receives) against this screen's own
    encodes and set `args.cpu`; returns the arguments that carry it to a box."""
    args.cpu, args.dcpu = None, None
    if args.cpu_summary:
        found = decision_cpu.load_charge(Path(args.cpu_summary), args.cpu_decision)
        rotation = decision_cpu.plan_rotation(Path(args.cpu_summary), args.cpu_decision)
        if args.rotation and args.rotation != rotation:
            raise decision_cpu.CpuMismatch(f"--rotation {args.rotation}: the CPU was measured "
                                           f"on draw {rotation}")
        args.rotation = rotation
        print(f"CPU measured on draw {rotation}: the screen runs on it", flush=True)
    elif args.cpu_charge:
        found = decision_cpu.parse(args.cpu_charge)
    else:
        if args.cpu_plan:
            decision_cpu.write_plan(Path(args.cpu_plan), [cpu_plan_entry(args)])
            sys.exit(0)
        return []
    args.cpu = decision_cpu.verify(found, cpu_plan_entry(args)["digest"], "nceval.screen")
    print(f"dCPU per group: {decision_cpu.describe(args.cpu)}", flush=True)
    return ["--cpu-charge", args.cpu.text(), "--rotation", args.rotation]


def main() -> None:
    p = argparse.ArgumentParser(prog="python3 -m nceval.screen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codec", required=True, choices=[c.value for c in Codec])
    p.add_argument("--binary", required=True, help="candidate nanocodec binary")
    p.add_argument("--param", action="append", default=[], help="candidate name=value")
    p.add_argument("--anchor-binary", help="anchor binary (default: the candidate binary)")
    p.add_argument("--anchor-param", action="append", default=[], help="anchor name=value")
    p.add_argument("--clips",
                   help="adopt (real mode's default: the pool levels on the day's draw, "
                        "nceval.levels), train (fixed class-balanced proxy; cqp mode's "
                        "default), val, or comma-separated TRAIN/VAL names (ranking only)")
    p.add_argument("--max-level", type=int, choices=(1, 2, 3), default=3,
                   help="--clips adopt: stop after this level (3, the default, is adoption "
                        "evidence; 1-2 rank ideas)")
    p.add_argument("--levels", type=pool_levels,
                   help="--clips adopt: run only these pool levels, comma-separated "
                        f"({','.join(pool.LEVELS)}), in order; overrides --max-level (the "
                        "bundle checks of nceval.train run L3 alone)")
    p.add_argument("--final-estimator", choices=("single", "ladder"),
                   help="--clips adopt: L3 per clip, one rung on the anchor's curve or the "
                        "full ladder (default: single for a parameter-only change, i.e. "
                        "one binary; ladder for code changes and anything reshaping the curve)")
    p.add_argument("--final-frames", type=int, nargs="?", const=600,
                   help="--clips adopt: L3 on the long window at every key share "
                        "(nceval.key_shares: 600 f with keys every 150, 900 f with keys every "
                        "300): for changes to rate, keys, cadence or key-frame quality and a "
                        "preset bundle's final check")
    p.add_argument("--rotation",
                   help="--clips adopt: the pool rotation key (default: today's UTC date); "
                        "only to reproduce a recorded run")
    p.add_argument("--no-staged", dest="staged", action="store_false",
                   help="measure every clip at once instead of the staged TRAIN/VAL screen")
    p.add_argument("--all-stages", action="store_true",
                   help="run every stage even after a clear decision (calibrates the stage sigmas)")
    p.add_argument("--frames", default=str(tune.PROXY_FRAMES), help="frames per clip, or 'full'")
    p.add_argument("--qps", help="fixed quantizer grid (default: 4 QPs calibrated per clip)")
    p.add_argument("--gate", choices=[g.value for g in GatePolicy],
                   help="decoders on the measured points (nceval.gatepolicy): off (default; "
                        "conformance is checked when a change lands), sample or full")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--over-budget", action="store_true",
                   help="run even if estimated over the 30-min limit (logged)")
    p.add_argument("--json", help="also write the result here")
    p.add_argument("--local", action="store_true",
                   help="measure here (default: a warm rented box, nceval.cloud; --mode cqp "
                        "always runs here)")
    p.add_argument("--mode", choices=("real", "cqp"), default="real",
                   help="real: CBR ladder judged by J_real (default); cqp: the fixed-QP proxy")
    p.add_argument("--cpu-plan",
                   help="write this screen's CPU plan here and exit: its own encodes' declared "
                        "subset (nceval.decision_cpu), for tools/dev/aws_cpu_ab.py "
                        "--decision-plan")
    p.add_argument("--cpu-summary",
                   help="aws_cpu_ab.py's summary of this screen's CPU plan: each level is "
                        "charged k * dCPU of its own encodes (refused if measured on others); "
                        "the screen runs on the plan's draw")
    p.add_argument("--cpu-charge", help=argparse.SUPPRESS)  # a verified --cpu-summary, boxed
    p.add_argument("--cpu-decision", default=CPU_DECISION,
                   help="this screen's decision name in a plan or summary of several "
                        f"(default {CPU_DECISION}; nceval.train names each arm's screens)")
    p.add_argument("--key-shares",
                   help="key periods in frames of the long-window L3 (default 150,300: "
                        "nceval.key_shares, which must include both)")
    p.add_argument("--preset",
                   help="the rung screened: both sides encode its preset parameters (each "
                        "from the presets/ beside its binary, under --param / --anchor-param), "
                        "and in real mode its front slope (nceval.front_k) is the CPU weight "
                        "k of J_real")
    p.add_argument(RUNG_PARAMS_FLAG, action="store_true",
                   help="the --param / --anchor-param lists already hold --preset's values "
                        "(nceval.train, the cloud offload): do not add them again")
    p.add_argument("--k", type=float,
                   help="real mode: an explicit CPU weight k of J_real in place of the rung's "
                        "front slope (warned about loudly)")
    p.add_argument("--expect-commit",
                   help="fail at once unless --binary was built from this commit "
                        f"(also ${buildinfo.EXPECT_ENV}; nceval.buildinfo)")
    args = p.parse_args()
    rung_argv = apply_rung(args)
    cpu_argv = resolve_cpu(args)
    args.k, args.k_source = cpu_charge(args)
    print(f"k {args.k:.4g}: {args.k_source}", flush=True)
    if args.mode == "real" and (cold := baseline_state.notice()):
        print(cold, flush=True)
    if args.mode == "real" and not cloud.runs_here(args.local):
        argv = decision_cpu.without_local_cpu(sys.argv[1:]) if cpu_argv else sys.argv[1:]
        cloud.offload("nceval.screen", argv + rung_argv + cpu_argv, args.json,
                      cloud.default_json("screen"))
    builds = {"candidate": buildinfo.check(Path(args.binary), "candidate", args.expect_commit)}
    if args.anchor_binary:
        builds["anchor"] = buildinfo.check(Path(args.anchor_binary), "anchor", measured=False)
    write = lambda result: (Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
                            if args.json else None)
    # The JSON is always written, so a waiter keyed on it never hangs.
    try:
        if args.mode == "real":
            from . import realuse  # realuse builds on this module's stages
            args.clips = args.clips or realuse.ADOPT
            result = realuse.screen(args)
        else:
            print("mode: FIXED-QP PROXY (not adoption evidence; see --mode real)", flush=True)
            args.clips = args.clips or "train"
            result = {"mode": "cqp-proxy", **screen(args)}
    except OverBudget as refusal:
        write({"status": "refused", "reason": str(refusal)})
        raise
    except BaseException as error:
        write({"status": "error", "reason": f"{type(error).__name__}: {error}"})
        raise
    write({"status": "done", **result, "k_source": args.k_source, "build_info": builds,
           "cpu": decision_cpu.record(args.cpu)})
    levers.record(levers.screen_rows({**result, "build_info": builds},
                                     args.json or "nceval.screen"), "nceval.screen")


if __name__ == "__main__":
    main()
