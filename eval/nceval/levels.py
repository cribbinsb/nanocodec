"""The pool levels: L1 quick, L2 adopt, L3 final, on the day's rotating draw.

`nceval.screen` (real mode, `--clips adopt`, the default) runs the levels of
nceval.pool's draw in order, candidate vs anchor in CBR on the ladder:

    level   clips                                       per clip
    quick   L1: the key's chosen set (pool.L1_SETS),    one rung, single-point
            else 24 drawn families, every stratum
    adopt   48 more, disjoint                           full ladder
    final   every other pool family (one clip each)     single-point (parameter-only
                                                        changes) or full ladder

Single-point (nceval.singlepoint, owner 2026-11-11): one rung per clip,
priced on the anchor's cached full curve; on recorded screens it predicted
held-out clips better than the full ladder on a quarter of the clips. A
single-point level whose per-clip deltas trend with the rung (the change
reshapes the curve) is re-measured on the full ladder. L3's estimator:
`--final-estimator` (screen) / `--estimator` (final-check); by default
single-point when the arms differ only in parameters (one binary, or
commits with identical crates/), the full ladder for code changes; charts
and structural changes keep the full ladder.

Each level is decided on its per-clip J deltas (nceval.interval): L1 and L2
stop only a confidently worse candidate; L3 adopts when the interval
favours it, judged on L2 and L3's clips together.
`--max-level 2` stops after L2 (ranking ideas; not adoption evidence). The
draw (rotation key, clips and start frames) is in the result JSON.

L3 runs with NC_GPU_SLOTS raised to FINAL_GPU_SLOTS unless already set (as
the chart re-scores do: L3 is one long job that keeps the GPU busy).

`python3 -m nceval.levels final-check --old A --new B --codecs h264,av1`
compares two builds (commits or binaries) on the day's L3: after every
preset bundle, and daily.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import filecmp
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import buildinfo, interval, jreal, pool, singlepoint
from .cbr import Family, clip_targets
from .clips import Clip
from .codecs import Codec
from .config import Paths, load_paths
from .encoders.nanocodec import read_preset
from .gatepolicy import EVALUATION_GATE, GatePolicy
from .realuse import (DEFAULT_FRAMES, DEFAULT_WORKERS, LADDER_KBPS, Runner, Track,
                      add_periodic_pulse, compare, format_stage, one_per_class, param_args,
                      run_step)
from .stages import ALL_FOUR, OUTER_THREE, Stage

RUNGS = {pool.QUICK: OUTER_THREE, pool.ADOPT: ALL_FOUR, pool.FINAL: ALL_FOUR}
#: Level estimators: one rung per clip on the anchor's curve (nceval.singlepoint),
#: or the full ladder.
SINGLE, LADDER, AUTO = "single", "ladder", "auto"
FINAL_GPU_SLOTS = 16
GPU_SLOTS_ENV = "NC_GPU_SLOTS"
#: Levels whose clips L3's adoption interval pools (same full ladder).
FINAL_POOLED = (pool.ADOPT, pool.FINAL)
#: The long-window L3 (coordinator, 2026-11-13): 150 frames with one key reward
#: an unrepaid first-key loan (the audit's long clips lose 4-9 BD-J points from
#: 150 to 600 frames), so changes to rate, keys, cadence or key-frame quality,
#: and the final check of a preset bundle, decide on LONG_FRAMES with a key
#: every LONG_KEY_SECONDS (`final_frames`); L1, L2 and study trials stay at 150
#: frames as cheap rankers. Only the L3 clips with LONG_FRAMES frames take part
#: (`long_clips`: about a third of the pool), from frame 0. Such an L3 is not
#: pooled with L2 (a different window).
LONG_FRAMES = 600
LONG_KEY_SECONDS = 10.0


def usable(clip: Clip) -> bool:
    """Enough ladder rungs under the clip's 50% cap for a BD-rate."""
    return len(clip_targets(clip, LADDER_KBPS)) >= jreal.BD_MIN_POINTS


def run(runner: Runner, families: tuple[Family, Family], paths: Paths,
        args: argparse.Namespace, levels: tuple[str, ...] | None = None) -> dict:
    """Run the levels in order, stopping at a confident loss; the result fields.

    L1 is single-point; L3 is single-point when `args.final_estimator` says so
    (small changes), else the full ladder. A single-point level whose deltas
    trend with the rung is re-measured on the full ladder (singlepoint.reshaping).
    `args.final_frames` (LONG_FRAMES) runs L3 on the long window, keys every
    LONG_KEY_SECONDS, unpooled.
    """
    levels = levels or pool.LEVELS[:args.max_level]
    frames = int(args.frames)
    draw = pool.draw(pool.load_pool(paths, usable), pool.rotation_key(args.rotation), frames)
    print(f"pool draw {draw.key}: " + ", ".join(
        f"{level} {len(draw.levels[level])} clips" for level in levels), flush=True)
    stages, per_level = [], {}
    for level in levels:
        if level == pool.FINAL:
            os.environ.setdefault(GPU_SLOTS_ENV, str(FINAL_GPU_SLOTS))
        if not draw.levels[level]:
            print(f"{level}: no pool families left for this level", flush=True)
            continue
        started = time.monotonic()
        long_window = level == pool.FINAL and getattr(args, "final_frames", None)
        level_frames = args.final_frames if long_window else frames
        key_seconds = LONG_KEY_SECONDS if long_window else None
        clips = long_clips(draw.levels[level], level_frames) if long_window \
            else draw.levels[level]
        if long_window:
            print(f"{level}: long window on the {len(clips)} of {len(draw.levels[level])} "
                  f"clips with {level_frames} frames, from frame 0", flush=True)
        result = measure_level(runner, families, clips, level_frames, level,
                               estimator_of(level, args), args, key_seconds)
        result["frames"] = level_frames
        rows = result["per_clip"]
        if level == pool.FINAL and not long_window:
            pooled = {name: row for s in stages if s["stage"] in FINAL_POOLED
                      for name, row in s["per_clip"].items()}
            rows = {**pooled, **rows}
        if level == pool.FINAL:
            pulse = one_per_class(clips)
            add_periodic_pulse(runner, *families, pulse, frames, result["per_clip"], level)
        if level == pool.FINAL or result["estimator"] == SINGLE:
            result["summary"] = jreal.summarise(result["per_clip"])
            result["verdict"] = jreal.objective(result["summary"], args.dcpu, 0.0, args.k)
        stats = interval.interval(rows, args.dcpu, args.k)
        result["interval"] = stats
        result["decision"] = interval.decide(stats, final=level == pool.FINAL)
        wall = time.monotonic() - started
        per_level[level] = {"decision": result["decision"], "estimator": result["estimator"],
                            **stats, "guards": result["verdict"]["guards"],
                            "wall_seconds": round(wall)}
        stages.append(result)
        if result["estimator"] == LADDER:
            print(format_stage(result), flush=True)
        print(f"{level} ({result['estimator']}): {interval.describe(stats)}"
              + (f" ({level_frames} f, keys every {LONG_KEY_SECONDS:g} s)" if long_window
                 else " (L2+L3 pooled)" if level == pool.FINAL else "")
              + f" -> {result['decision'].upper()} ({wall / 60:.1f} min)"
              + (f"; guards (judge): {'; '.join(result['verdict']['guards'])}"
                 if result["verdict"]["guards"] else ""), flush=True)
        if result["decision"] == interval.WORSE:
            break
    last = stages[-1]["decision"]
    decision = last if last in (interval.WORSE, interval.BETTER, interval.AMBIGUOUS) else \
        f"not adopted (stopped after {stages[-1]['stage']})"
    print(f"levels -> {decision.upper()}", flush=True)
    return {"clips": "adopt", "staged": True, "decision": decision, "levels": per_level,
            "draw": draw.record(), "stages": stages}


def long_clips(clips: list[Clip], frames: int) -> list[Clip]:
    """The level's clips long enough for the long window, measured from frame 0.

    A rotated start or a shorter source would end the clip early (the encoder
    then refuses the point, which silently shrank the first long L3s to the
    clips that happened to fit); only about a third of the pool has 600 frames.
    """
    return [dataclasses.replace(c, start_frame=0) for c in clips if c.frames >= frames]


def estimator_of(level: str, args: argparse.Namespace) -> str:
    """L1 single-point; L2 full ladder; L3 as `args.final_estimator` (default ladder)."""
    if level == pool.QUICK:
        return SINGLE
    if level == pool.FINAL:
        return getattr(args, "final_estimator", None) or LADDER
    return LADDER


def measure_level(runner: Runner, families: tuple[Family, Family], clips: list[Clip],
                  frames: int, level: str, estimator: str, args: argparse.Namespace,
                  key_seconds: float | None = None) -> dict:
    """One level's per-clip rows ({"stage", "estimator", "per_clip", ...}); `key_seconds`
    sets a key period (the long-window L3), else one key."""
    if estimator == SINGLE:
        rows = singlepoint.single_points(runner, *families, clips,
                                         singlepoint.assign_rungs(clips), frames,
                                         budgeted=True, key_seconds=key_seconds)
        change, t_value = singlepoint.rung_trend(rows)
        if not singlepoint.reshaping(rows):
            return {"stage": level, "estimator": SINGLE, "per_clip": rows,
                    "rung_trend": [change, t_value]}
        print(f"{level}: single-point deltas trend with the rung ({change:+.2f}% over the "
              f"ladder, t {t_value:+.1f}): the change reshapes the curve, full ladder",
              flush=True)
    if key_seconds:
        curves = runner.measure(f"{level} long window", list(families), clips, frames,
                                key_seconds=key_seconds)
        rows = compare(curves, families[0].name, families[1].name, clips, paired=True)
        summary = jreal.summarise(rows)
        return {"stage": level, "estimator": LADDER, "sigma": 0.0, "frames": frames,
                "per_clip": rows, "summary": summary,
                "verdict": jreal.objective(summary, args.dcpu, 0.0, args.k)}
    track = Track(level, clips, None, [])
    run_step(runner, families, [(track, Stage(level, None, RUNGS[level], 0.0))], frames, True,
             args, early_stop=False, announce=False)
    return {**track.stages[-1], "estimator": LADDER}


# --- final-check: two builds on the day's L3 -------------------------------------------

#: What decides whether two commits' arms can differ: the encoder's sources and presets.
CODE_INPUTS = ("crates", "presets")


def build(spec: str, paths: Paths, role: str) -> tuple[Path, str]:
    """(binary, commit) for a binary path or a commit (built once, cached by commit).

    A commit is built in `role`'s own git worktree and target dir (one per arm,
    so each stays incremental against itself): `git checkout` rewrites only the
    files that changed, with fresh mtimes, so cargo's freshness check is right.
    (Builds from a `git archive` into one shared target dir kept the commit
    time as every file's mtime, and cargo reused the previous commit's
    binary: a VP9 final check came out at exactly +0.00%.) Every binary,
    cached or new, must say it is `commit` (nanocodec --version)."""
    if Path(spec).is_file():
        return Path(spec).resolve(), ""
    repo = Path(__file__).resolve().parents[2]
    commit = git(repo, "rev-parse", f"{spec}^{{commit}}")
    binary = paths.data_root / "builds" / commit[:12] / "nanocodec"
    if binary.is_file():
        problem = buildinfo.stamp_problem(buildinfo.info_of(binary), commit)
        if problem is None:
            return binary, commit
        print(f"cached build {binary} is unusable ({problem}): rebuilding", flush=True)
    tree = paths.data_root / "worktrees" / f"final-check-{role}"
    target = paths.data_root / "targets" / f"final-check-{role}"
    with tree_lock(tree):
        if not (tree / ".git").exists():
            git(repo, "worktree", "add", "--detach", str(tree), commit)
        git(tree, "checkout", "-q", "--detach", commit)
        built = cargo_build(tree, target)
        buildinfo.verify_built(built, commit)
        presets = binary.parent / "presets"
        shutil.rmtree(presets, ignore_errors=True)
        shutil.copytree(tree / "presets", presets)
        partial = binary.with_name(binary.name + ".partial")
        shutil.copy2(built, partial)
        partial.rename(binary)  # present means complete
    buildinfo.verify_built(binary, commit)
    return binary, commit


def cargo_build(tree: Path, target: Path) -> Path:
    """The release CLI of `tree`, built into `target`."""
    subprocess.run(["cargo", "build", "--release", "-j8", "-p", "nc-cli"], cwd=tree,
                   env={**os.environ, "CARGO_TARGET_DIR": str(target),
                        "PATH": f"{Path.home() / '.cargo/bin'}:{os.environ['PATH']}"},
                   check=True)
    return target / "release" / "nanocodec"


@contextmanager
def tree_lock(tree: Path):
    """One final check at a time checks out and builds in a role's tree."""
    tree.parent.mkdir(parents=True, exist_ok=True)
    with tree.with_name(tree.name + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def same_code(old_commit: str, new_commit: str) -> bool:
    """Whether two commits have identical crates/ and presets/ trees."""
    if not (old_commit and new_commit):
        return False
    repo = Path(__file__).resolve().parents[2]
    return all(git(repo, "rev-parse", f"{old_commit}:{part}")
               == git(repo, "rev-parse", f"{new_commit}:{part}") for part in CODE_INPUTS)


def final_estimator(choice: str, old_commit: str, new_commit: str) -> str:
    """The L3 estimator: `choice`, or for "auto" single-point when only presets differ
    (a parameter-only change: the encoder's sources are identical), else the ladder."""
    if choice != AUTO:
        return choice
    if not (old_commit and new_commit):
        return LADDER
    repo = Path(__file__).resolve().parents[2]
    same = git(repo, "rev-parse", f"{old_commit}:crates") == \
        git(repo, "rev-parse", f"{new_commit}:crates")
    return SINGLE if same else LADDER


def no_change_reason(old: tuple[Path, str], new: tuple[Path, str]) -> str | None:
    """Why the two arms cannot differ (None: they can); SystemExit when the arms'
    binaries are byte-identical although their code differs (a stale build)."""
    (old_binary, old_commit), (new_binary, new_commit) = old, new
    if same_code(old_commit, new_commit):
        return f"no code change: {old_commit[:12]} and {new_commit[:12]} have identical " \
               f"{' and '.join(f'{p}/' for p in CODE_INPUTS)}"
    if filecmp.cmp(old_binary, new_binary, shallow=False):
        raise SystemExit(f"old {old_binary} and new {new_binary} are byte-identical, but "
                         f"their code differs (or is unknown): one is a stale build; "
                         f"refusing to compare them")
    return None


def preset_args(binary: Path, preset: str | None) -> list[str]:
    """--param pairs of `preset` from the build's own presets (repo's for a bare binary)."""
    if not preset:
        return []
    directory = binary.parent / "presets"
    if not directory.is_dir():
        directory = Path(__file__).resolve().parents[2] / "presets"
    values = read_preset(directory / f"{preset}.toml")
    return [f"{k}={v:g}" for k, v in sorted(values.items())]


def final_check(args: argparse.Namespace) -> dict:
    paths = load_paths()
    (old, old_commit), (new, new_commit) = arms = (build(args.old, paths, "old"),
                                                   build(args.new, paths, "new"))
    report = {"old": args.old, "old_commit": old_commit, "new": args.new,
              "new_commit": new_commit, "preset": args.preset, "codecs": {},
              "build_info": {"old": buildinfo.check(old, "old", measured=False),
                             "new": buildinfo.check(new, "new", measured=False)}}
    estimator = final_estimator(args.estimator, old_commit, new_commit)
    report["estimator"] = estimator
    print(f"L3 estimator: {estimator}", flush=True)
    unchanged = no_change_reason(*arms)
    if unchanged:
        print(unchanged, flush=True)
        report["no_change"] = unchanged
        return report
    for name in args.codecs.split(","):
        codec = Codec(name)
        anchor = Family(f"nanocodec-{codec.value}", codec, old,
                        param_args(preset_args(old, args.preset)), own=True, cqp=False)
        test = Family(f"nanocodec-{codec.value}@candidate", codec, new,
                      param_args(preset_args(new, args.preset)), own=True, cqp=False)
        runner = Runner(paths, GatePolicy(EVALUATION_GATE.value), args.workers, args.over_budget)
        screen_args = argparse.Namespace(frames=args.frames, rotation=args.rotation,
                                         max_level=len(pool.LEVELS), dcpu=None,
                                         k=jreal.K_CPU, all_stages=False,
                                         final_estimator=estimator,
                                         final_frames=args.final_frames or None)
        print(f"== {codec.value}: {args.old} -> {args.new}", flush=True)
        report["codecs"][codec.value] = run(runner, (anchor, test), paths, screen_args,
                                            (pool.FINAL,))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python3 -m nceval.levels",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("final-check", help="two builds on the day's L3 (bundles, daily)")
    f.add_argument("--old", required=True, help="commit or nanocodec binary (the anchor)")
    f.add_argument("--new", required=True, help="commit or nanocodec binary (the candidate)")
    f.add_argument("--codecs", default="h264,hevc,av1")
    f.add_argument("--preset", help="preset name (each build's own presets/), e.g. h264-fast")
    f.add_argument("--rotation", help="rotation key to reproduce (default: today, UTC)")
    f.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    f.add_argument("--final-frames", type=int, default=LONG_FRAMES,
                   help="the bundle's L3 window, keys every LONG_KEY_SECONDS (0: --frames, "
                        "one key)")
    f.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    f.add_argument("--over-budget", action="store_true")
    f.add_argument("--estimator", choices=(AUTO, SINGLE, LADDER), default=AUTO,
                   help="L3 per clip: single (one rung on the anchor's curve; small changes), "
                        "ladder (full ladder; structural changes, charts) or auto (single "
                        "when only presets differ)")
    f.add_argument("--json", help="also write the report here")
    args = parser.parse_args(argv)
    report = final_check(args)
    print(json.dumps({c: {"decision": r["decision"], **r["levels"][pool.FINAL]}
                      for c, r in report["codecs"].items()}, indent=1, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
