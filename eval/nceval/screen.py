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
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from . import buildinfo, tune
from .bdrate import bd_rate
from .codecs import Codec
from .config import load_paths
from .fingerprint import measurement_fingerprint
from .budget import (MAX_MINUTES, OverBudget, check as check_budget, estimate_minutes,
                     point_cpu_seconds)
from .cachefile import identity_cached, read_json
from .gatepolicy import (EVALUATION_GATE, GatePolicy, PointCheck, file_sha256, gate_arguments,
                         uniform_check)
from .jreal import K_CPU
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
        return [file_sha256(self.binary), sorted(self.overrides.items())]


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
    p.add_argument("--final-estimator", choices=("single", "ladder"),
                   help="--clips adopt: L3 per clip, one rung on the anchor's curve or the "
                        "full ladder (default: single for a parameter-only change, i.e. "
                        "one binary; ladder for code changes and anything reshaping the curve)")
    p.add_argument("--final-frames", type=int, nargs="?", const=600,
                   help="--clips adopt: L3 on this long window (600 when given without a "
                        "value; keys every 10 s, not pooled with L2): for changes to rate, "
                        "keys, cadence or key-frame quality and a preset bundle's final check "
                        "(nceval.levels.LONG_FRAMES)")
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
    p.add_argument("--mode", choices=("real", "cqp"), default="real",
                   help="real: CBR ladder judged by J_real (default); cqp: the fixed-QP proxy")
    p.add_argument("--dcpu", type=float,
                   help="real mode: the candidate's CPU change in %% (callgrind), folded "
                        "into J_real as k * dCPU")
    p.add_argument("--k", type=float, default=K_CPU,
                   help=f"real mode: the CPU weight k of J_real (default {K_CPU}, Fast and "
                        "engine changes; 0.1 for Default quality arms, PROCEDURES "
                        "\"Preset roles\")")
    p.add_argument("--expect-commit",
                   help="fail at once unless --binary was built from this commit "
                        f"(also ${buildinfo.EXPECT_ENV}; nceval.buildinfo)")
    args = p.parse_args()
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
    write({"status": "done", **result, "build_info": builds})


if __name__ == "__main__":
    main()
