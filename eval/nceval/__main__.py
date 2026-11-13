"""Command line for the evaluation harness. Run from the eval/ directory.

The maintained entry point is `reference`, which runs the whole anchor
evaluation from eval/configs/reference.json, idempotently:

    python3 -m nceval reference [--config FILE] [--only sweep|speed|report]
                                [--platform-remote jetson] [--force-speed]

The individual steps are also available for ad-hoc work:

    python3 -m nceval sweep  --clipset FILE --encoders a,b --intra-period-seconds 10
    python3 -m nceval report --clipset FILE --anchor openh264-baseline --metric psnr_yuv
    python3 -m nceval speed  --clipset FILE --clip NAME --encoders a,b
    python3 -m nceval sanity [--only measure|checks]   (docs/PROCEDURES.md)
    python3 -m nceval remeasure --clipset FILE --encoders a,b --source DB:STRUCTURE
    python3 -m nceval scoreboard   (after each milestone landing)
    python3 -m nceval gpu-status   (VMAFx scoring servers: memory, load, idle time)

See eval/README.md for the full workflow.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from . import report, scoreboard, speed
from .clips import ClipSet, load_clipset
from .config import load_paths
from .encoders import ENCODERS, EncoderAdapter, resolve_encoder
from .reference import DEFAULT_CONFIG, STEPS, Reference, Selection
from .sanity.config import DEFAULT_CONFIG as SANITY_CONFIG
from .results import ResultsDb
from .sweep import Sweep, SweepConfig, nominal_targets
from .remeasure import Remeasure, Source
from .gatepolicy import EVALUATION_GATE, GatePolicy
from .metrics import DISPLAY_PROFILE, LEGACY_PROFILE, V1_PROFILE

PRIMARY_METRICS = ("psnr_yuv", "vmaf_v1")  # xpsnr_y is opt-in (metrics.default_xpsnr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


def _cmd_reference(args) -> int:
    paths = load_paths()
    selection = Selection(anchors=_csv(args.anchors), suites=_csv(args.suites),
                          structures=_csv(args.structures))
    runner = Reference(Path(args.config), paths, ResultsDb(paths.results_db), _harness_commit(),
                       selection)
    steps = tuple(args.only.split(",")) if args.only else STEPS
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; steps are {', '.join(STEPS)}")
    runner.run(steps, args.platform_remote, args.force_speed)
    return 0


def _cmd_gpu_status(args) -> int:
    from . import gpuserve
    print(gpuserve.format_status(gpuserve.servers()))
    return 0


def _cmd_sanity(args) -> int:
    from .sanity.runner import run_sanity
    return run_sanity(args, _harness_commit())


def _cmd_sweep(args) -> int:
    paths, db, clipset, cfg = _setup(args)
    sweeper = Sweep(paths, db, _harness_commit())
    summary = sweeper.run(_encoders(args.encoders), list(clipset.clips), cfg)
    print(f"curves: {summary}")
    print(report.format_failures(db, [c.name for c in clipset.clips]))
    return 0 if not db.failures([c.name for c in clipset.clips]) else 1


def _cmd_remeasure(args) -> int:
    paths, db, clipset, cfg = _setup(args)
    sources = [Source(Path(database), structure)
               for database, _, structure in (s.rpartition(":") for s in args.source)]
    runner = Remeasure(Sweep(paths, db, _harness_commit()), cfg, sources)
    jobs = [(e, c) for e in _encoders(args.encoders) for c in clipset.clips]
    for line in runner.run(jobs, args.over_budget, args.workers):
        print(line, flush=True)
    print(report.format_failures(db, [c.name for c in clipset.clips]))
    return 0 if not db.failures([c.name for c in clipset.clips]) else 1


def _cmd_scoreboard(args) -> int:
    paths = load_paths()
    status = None if args.no_status else Path(args.status)
    pending = dict(item.split("=", 1) for item in args.pending)
    out = scoreboard.run(paths, ResultsDb(paths.results_db), _harness_commit(), status,
                         args.over_budget, args.rows.split(",") if args.rows else None,
                         not args.no_measure, pending)
    print(f"written {out}" + ("" if status is None else f"; status section in {status}"))
    return 0


def _cmd_report(args) -> int:
    _, db, clipset, cfg = _setup(args)
    for metric in args.metric.split(","):
        rows = report.compute(db, clipset, args.anchor, metric, cfg.structure_tag,
                              platform.machine(), max_frames=args.max_frames)
        print(report.format_report(rows, args.anchor, metric, args.per_clip))
        print()
    print(report.format_failures(db, [c.name for c in clipset.clips]))
    print(report.format_gate_levels(db, [c.name for c in clipset.clips]))
    return 0


def _cmd_speed(args) -> int:
    paths, db, clipset, cfg = _setup(args)
    matches = [c for c in clipset.clips if c.name == args.clip]
    if not matches:
        print(f"clip {args.clip!r} not in clip set", file=sys.stderr)
        return 2
    clip = matches[0]
    frames = min(clip.frames, cfg.max_frames or clip.frames)
    encoders = _encoders(args.encoders)
    if args.quantizers:
        quantizers = _parse_quantizers(args.quantizers)
    else:
        target = args.target_kbps or sorted(nominal_targets(clip))[2]
        quantizers = speed.choose_quantizers(db, encoders, cfg.structure_tag, clip,
                                             platform.machine(), target)
    results = speed.measure(encoders, clip, cfg.structure_for(clip), quantizers, frames,
                            args.core, args.runs, paths)
    if args.json:
        # Machine-readable output for a remote caller (nceval reference); the
        # caller stores the results in its own database.
        print(json.dumps([r.to_dict() for r in results]))
        return 0
    sweeper = Sweep(paths, db, _harness_commit())
    fingerprints = {e.name: sweeper.fingerprint(e, cfg, clip) for e in encoders}
    speed.store(results, db, cfg.structure_tag, clip, frames, platform.machine(), args.core,
                args.runs, fingerprints, _harness_commit())
    print(speed.format_speed(results, args.anchor, clip, speed.machine_conditions()))
    return 0


def _setup(args):
    paths = load_paths()
    db = ResultsDb(paths.results_db)
    clipset = load_clipset(Path(args.clipset), paths.data_root)
    if args.split:
        splits = set(args.split.split(","))
        clipset = ClipSet(name=clipset.name,
                          clips=tuple(c for c in clipset.clips if c.split in splits))
    cfg = SweepConfig(intra_period_seconds=args.intra_period_seconds,
                      temporal_layers=args.temporal_layers, max_frames=args.max_frames,
                      single_keyframe=args.single_keyframe,
                      workers=getattr(args, "workers", 1),
                      retry_failed=getattr(args, "retry_failed", False),
                      gate=GatePolicy(getattr(args, "gate", EVALUATION_GATE.value)),
                      over_budget=getattr(args, "over_budget", False),
                      measurement_profile=_profile(args))
    return paths, db, clipset, cfg


def _profile(args) -> str:
    """Legacy if asked for (or a legacy metric is reported), the opt-in 1080p
    display protocol with --vmaf-display, else native VMAF v1."""
    if getattr(args, "legacy_metrics", False) or any(
            m in ("vmaf_neg", "vmaf") for m in getattr(args, "metric", "").split(",")):
        return LEGACY_PROFILE
    return DISPLAY_PROFILE if getattr(args, "vmaf_display", False) else V1_PROFILE


def _encoders(names: str) -> list[EncoderAdapter]:
    # `nanocodec-<codec>@<preset>` is our encoder with presets/<preset>.toml.
    resolved = {n: resolve_encoder(n) for n in names.split(",")}
    unknown = [n for n, e in resolved.items() if e is None]
    if unknown:
        raise SystemExit(f"unknown encoder(s) {unknown}; known: {', '.join(ENCODERS)} "
                         f"(and nanocodec-<codec>@<preset>)")
    return list(resolved.values())


def _csv(text: str | None) -> tuple[str, ...]:
    return tuple(t for t in (text or "").split(",") if t)


def _parse_quantizers(text: str) -> dict[str, tuple[int, float]]:
    """'enc=q:kbps,enc2=q:kbps' -> {enc: (q, kbps)}."""
    out = {}
    for item in text.split(","):
        name, value = item.split("=")
        q, kbps = value.split(":")
        out[name] = (int(q), float(kbps))
    return out


def _harness_commit() -> str:
    repo = Path(__file__).resolve().parents[2]

    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                              text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("-dirty" if git("status", "--porcelain") else "")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nceval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)

    ref = sub.add_parser("reference", help="the maintained anchor evaluation (all steps)")
    ref.add_argument("--config", default=str(DEFAULT_CONFIG))
    ref.add_argument("--only", help=f"comma-separated subset of: {', '.join(STEPS)}")
    ref.add_argument("--platform-remote", help="also time on this remote (config 'remotes')")
    ref.add_argument("--force-speed", action="store_true",
                     help="re-time even if speed results are current")
    ref.add_argument("--anchors", help="restrict to these anchors, run in this order")
    ref.add_argument("--suites", help="restrict to these suites (e.g. main,long)")
    ref.add_argument("--structures", help="restrict to these structure tags (e.g. ip10s-tl1)")
    ref.set_defaults(handler=_cmd_reference)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--clipset", required=True, help="clip set JSON (eval/clipset)")
        sp.add_argument("--intra-period-seconds", type=float, default=10.0,
                        help="time between IDRs (converted to frames per clip)")
        sp.add_argument("--single-keyframe", action="store_true",
                        help="one key frame for the whole clip (long GOP); overrides the period")
        sp.add_argument("--temporal-layers", type=int, default=1, choices=(1, 2, 3))
        sp.add_argument("--max-frames", type=int, help="encode at most this many frames")
        sp.add_argument("--split", help="comma-separated splits (train,val,test)")
        sp.add_argument("--legacy-metrics", action="store_true",
                        help="use legacy v0.6.1neg measurement namespace; vmaf_neg reports autodetect it")
        sp.add_argument("--vmaf-display", action="store_true",
                        help="VMAF v1 on Netflix's 1080p display protocol (ffmpeg upscale), for "
                             "external comparison; default is native geometry (internal)")

    sweep = sub.add_parser("sweep", help="fixed-QP RD sweep + metrics")
    common(sweep)
    sweep.add_argument("--encoders", required=True, help="comma-separated anchor names")
    sweep.add_argument("--workers", type=int, default=8)
    sweep.add_argument("--gate", choices=[p.value for p in GatePolicy],
                       default=EVALUATION_GATE.value,
                       help="decoders on the measured points (nceval.gatepolicy; default off: "
                            "conformance is checked when a change lands)")
    sweep.add_argument("--over-budget", action="store_true",
                       help="run even if estimated over the 30-min limit (logged)")
    sweep.add_argument("--retry-failed", action="store_true",
                       help="re-measure curves that failed conformance (after an "
                            "environment failure such as a full disk)")
    sweep.set_defaults(handler=_cmd_sweep)

    rem = sub.add_parser("remeasure", help="re-measure stored curves in the current profile: "
                                           "anchors' kept streams rescored, ours re-encoded "
                                           "on the stored QP grid (nceval.remeasure)")
    common(rem)
    rem.add_argument("--encoders", required=True)
    rem.add_argument("--source", action="append", required=True,
                     help="DB:STRUCTURE holding the stored curves, searched in order")
    rem.add_argument("--workers", type=int, default=2)
    rem.add_argument("--over-budget", action="store_true",
                     help="run even if estimated over the 30-min limit (logged)")
    rem.add_argument("--gate", choices=[p.value for p in GatePolicy],
                     default=EVALUATION_GATE.value,
                     help="decoders on the measured points (nceval.gatepolicy; default off)")
    rem.set_defaults(handler=_cmd_remeasure)

    board = sub.add_parser("scoreboard", help="one page: every codec/preset vs its bar "
                                              "(VAL-150 BD, instructions, Jetson, worst clip)")
    board.add_argument("--status", default=str(Path(__file__).resolve().parents[2] / "docs" / "STATUS.md"),
                       help="status file whose scoreboard section is rewritten")
    board.add_argument("--no-status", action="store_true", help="only write the dated page")
    board.add_argument("--no-measure", action="store_true",
                       help="write the page from stored results only (no encodes, no callgrind)")
    board.add_argument("--pending", action="append", default=[], metavar="ENCODER=NOTE",
                       help="show NOTE instead of numbers for this row (e.g. a codec being rebuilt)")
    board.add_argument("--rows", help="measure only these rows (encoder names, e.g. "
                                      "nanocodec-av1); the page still shows every row")
    board.add_argument("--over-budget", action="store_true",
                       help="run even if estimated over the 30-min limit (logged)")
    board.set_defaults(handler=_cmd_scoreboard)

    rep = sub.add_parser("report", help="BD-rate tables from the results database")
    common(rep)
    rep.add_argument("--anchor", default="openh264-baseline")
    rep.add_argument("--metric", default=",".join(PRIMARY_METRICS),
                     help="comma-separated metric names")
    rep.add_argument("--per-clip", action="store_true")
    rep.set_defaults(handler=_cmd_report)

    spd = sub.add_parser("speed", help="pinned single-core timing")
    common(spd)
    spd.add_argument("--clip", required=True)
    spd.add_argument("--encoders", required=True)
    spd.add_argument("--anchor", default="openh264-baseline")
    spd.add_argument("--core", type=int, default=2, help="CPU core to pin to")
    spd.add_argument("--runs", type=int, default=speed.MIN_TIMED_RUNS)
    spd.add_argument("--target-kbps", type=float,
                     help="time each encoder at its sweep point nearest this rate")
    spd.add_argument("--quantizers", help="explicit 'enc=q:kbps,...' instead of the database")
    spd.add_argument("--json", action="store_true", help="print results as JSON, store nothing")
    spd.set_defaults(handler=_cmd_speed)
    gpu = sub.add_parser("gpu-status", help="VMAFx scoring servers: state, GPU memory, idle time "
                         "(nceval.gpuserve)")
    gpu.set_defaults(handler=_cmd_gpu_status)
    san = sub.add_parser("sanity", help="automatic sanity report (ordering, outliers, tools, "
                                         "decay, harness)")
    san.add_argument("--config", default=str(SANITY_CONFIG))
    san.add_argument("--only", help="comma-separated subset of: measure,checks")
    san.add_argument("--sets", help="restrict to these measured sets (config 'sets')")
    san.add_argument("--clips", help="only these clips (comma-separated) of each set")
    san.add_argument("--clip-range", help="only clips FIRST-LAST (1-based) of each set, to split "
                                          "a set into runs that fit the time budget")
    san.add_argument("--workers", type=int, default=4)
    san.add_argument("--out", help="report directory (default $NC_DATA_ROOT/reports)")
    san.set_defaults(handler=_cmd_sanity)
    return p


if __name__ == "__main__":
    sys.exit(main())
