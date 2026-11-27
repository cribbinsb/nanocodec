"""nceval.nndata command line.

  plan      the jobs of a dataset version (one clip per pool family, sealed sets out)
  run       run a job list (on a rented box: nc-remote run ... python3 -m nceval.nndata run)
  validate  validate shards
  release   move a run's kept shards into an immutable version (or --part) with its manifest
  assemble  one version of released parts
  derive    jobs derived from a job list: a repeat (determinism, timing noise) or a lever
            toggled (label stability), on a subset of its clips
  retry     qp-mid jobs for the qp-low jobs a run quarantined for their rate
  compare   two shards of one job: determinism, timing noise and label shift
  quality   a version's determinism, timing noise and label shift against a repeat run
  card      the coverage report and dataset card of a run or a version
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..config import load_paths
from ..pool import load_pool
from dataclasses import replace

from . import coverage, generate, quality, shard, version
from .validate import validate


def one_per_family(clips):
    """One clip per source family: the encoded copy first (small to send), then by name."""
    families = defaultdict(list)
    for clip in clips:
        families[clip.family].append(clip)
    return [sorted(members, key=lambda c: (not c.encoded, c.name))[0]
            for _, members in sorted(families.items())]


def cmd_plan(args) -> int:
    paths = load_paths()
    clips = one_per_family(load_pool(paths))
    clips = clips[::max(1, args.every)]
    if args.limit:
        clips = clips[:args.limit]
    ffmpeg = str(generate.find_ffmpeg())
    rungs = tuple(args.rungs.split(",")) if args.rungs else None
    jobs = generate.plan(clips, args.codecs.split(","), paths.data_root, args.seed,
                         generate.EXCERPTS, ffmpeg, rungs)
    if args.skip_rate_failures:
        jobs = generate.without_rate_failures(jobs, json.loads(Path(args.skip_rate_failures).read_text()))
    generate.write_jobs(jobs, Path(args.out))
    inputs = sorted({str(Path.home() / j.input_file) for j in jobs})
    Path(args.out).with_suffix(".inputs").write_text("\n".join(inputs) + "\n")
    print(f"{len(jobs)} jobs over {len(clips)} clips -> {args.out} (+ .inputs: {len(inputs)} files)")
    return 0


def cmd_run(args) -> int:
    jobs = generate.read_jobs(Path(args.jobs))
    binary = Path(args.binary).resolve()
    tally = generate.run_jobs(jobs, binary, generate.binary_provenance(binary), Path(args.out),
                              str(generate.find_ffmpeg()), args.workers)
    print(json.dumps({k: v for k, v in tally.items() if k != "problems"}))
    return 0


def cmd_validate(args) -> int:
    bad = 0
    for path in args.paths:
        problems = validate(shard.load(Path(path)))
        bad += bool(problems)
        print(f"{path}: {'ok' if not problems else '; '.join(problems)}")
    return 1 if bad else 0


def cmd_release(args) -> int:
    root = Path(args.root) if args.root else version.ROOT
    target = version.release([Path(run) for run in args.run], args.version, root, args.part,
                             args.workers)
    print(f"released {target}")
    return 0


def cmd_derive(args) -> int:
    jobs = generate.read_jobs(Path(args.jobs))
    clips = sorted({j.clip for j in jobs})[::max(1, args.every)]
    presets, rates = args.preset.split(","), args.rate.split(",")
    chosen = [j for j in jobs if j.clip in clips and j.preset in presets and j.rate in rates
              and j.codec in args.codecs.split(",") and not j.params]
    params = tuple(args.param)
    tools = tuple(args.tool)
    unknown = set(tools) - set(generate.TOOL_CATALOGUE)
    if unknown:
        raise SystemExit(f"unknown tools {sorted(unknown)}; known: {sorted(generate.TOOL_CATALOGUE)}")
    derived = [replace(j, params=params, tools=tools) for j in chosen]
    generate.write_jobs(derived, Path(args.out))
    Path(args.out).with_suffix(".inputs").write_text(
        "\n".join(sorted({str(Path.home() / j.input_file) for j in derived})) + "\n")
    print(f"{len(derived)} jobs -> {args.out}")
    return 0


def cmd_retry(args) -> int:
    jobs = generate.read_jobs(Path(args.jobs))
    summaries = [json.loads((Path(run) / "summary.json").read_text()) for run in args.runs]
    retries = generate.rate_retries(jobs, summaries)
    generate.write_jobs(retries, Path(args.out))
    Path(args.out).with_suffix(".inputs").write_text(
        "\n".join(sorted({str(Path.home() / j.input_file) for j in retries})) + "\n")
    print(f"{len(retries)} qp-mid jobs -> {args.out}")
    return 0


def cmd_compare(args) -> int:
    a, b = shard.load(Path(args.a)), shard.load(Path(args.b))
    print(json.dumps({"determinism": quality.determinism(a, b),
                      "timing_noise": quality.timing_noise(a, b),
                      "label_shift": quality.label_shift(a, b)}, indent=1))
    return 0


def cmd_quality(args) -> int:
    """Repeat and lever-toggled shards of `run` against the version's shards of the same
    job (by name; a toggled job's name carries its +params)."""
    released = {str(p.relative_to(p.parents[2])).split("/", 1)[1]: p
                for p in version.shard_dirs(Path(args.version_dir))}
    report = {"repeats": [], "shifts": []}
    for path in version.shard_dirs(Path(args.run) / "shards"):
        name = str(path.relative_to(Path(args.run) / "shards"))
        config, clip = name.split("/", 1)
        base_name = config.split("+", 1)[0] + "/" + clip
        if base_name not in released:
            continue
        a, b = shard.load(released[base_name]), shard.load(path)
        if "+" in config:
            report["shifts"].append({"job": name, **quality.label_shift(a, b)})
        else:
            report["repeats"].append({"job": name, "determinism": quality.determinism(a, b),
                                      "timing_noise": quality.timing_noise(a, b)})
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(quality.summarise(report), indent=1))
    return 0


def cmd_card(args) -> int:
    root = Path(args.path)
    shards = version.shard_dirs(root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    manifest_path = root / version.MANIFEST
    manifest_sha = json.loads(manifest_path.read_text())["sha256"] if manifest_path.exists() else None
    limitations = json.loads(Path(args.limitations).read_text()) if args.limitations else []
    proposals = json.loads(Path(args.proposals).read_text()) if args.proposals else {}
    out = Path(args.out or root)
    out.mkdir(parents=True, exist_ok=True)
    rep = coverage.write(out, args.version, shards, summary, manifest_sha, limitations, proposals)
    print(json.dumps({"shards": rep["shards"], "superblocks": rep["superblocks"], "gaps": rep["gaps"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m nceval.nndata", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("plan")
    p.add_argument("--codecs", default="av1,hevc")
    p.add_argument("--seed", required=True, help="start-frame draw seed (recorded per job)")
    p.add_argument("--every", type=int, default=1, help="every Nth family only")
    p.add_argument("--rungs", help="these rungs (comma-separated) for every codec, not RUNGS")
    p.add_argument("--skip-rate-failures", help="an earlier version's summary.json: drop its qp-low rate failures")
    p.add_argument("--limit", type=int, default=0, help="first N clips only (pipe-clean runs)")
    p.add_argument("--out", required=True)
    p.set_defaults(handler=cmd_plan)
    p = commands.add_parser("run")
    p.add_argument("--jobs", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(handler=cmd_run)
    p = commands.add_parser("validate")
    p.add_argument("paths", nargs="+")
    p.set_defaults(handler=cmd_validate)
    p = commands.add_parser("release")
    p.add_argument("--run", required=True, action="append",
                   help="a run's output directory (shards/ inside); repeatable")
    p.add_argument("--version", required=True, help="vN (with --part: the part's name)")
    p.add_argument("--root", help="where versions live (default ~/nanocodec-data/nn-data)")
    p.add_argument("--part", action="store_true", help="a part of a version (assemble joins them)")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(handler=cmd_release)
    p = commands.add_parser("unstage")
    p.add_argument("staging")
    p.add_argument("run")
    p.set_defaults(handler=lambda a: print(version.unstage(Path(a.staging), Path(a.run))) or 0)
    p = commands.add_parser("assemble")
    p.add_argument("--version", required=True)
    p.add_argument("parts", nargs="+")
    p.set_defaults(handler=lambda a: print(version.assemble([Path(x) for x in a.parts], a.version)) or 0)
    p = commands.add_parser("derive")
    p.add_argument("--jobs", required=True)
    p.add_argument("--codecs", default="av1,hevc")
    p.add_argument("--preset", default="medium")
    p.add_argument("--rate", default="qp-high")
    p.add_argument("--every", type=int, default=8, help="every Nth clip by name")
    p.add_argument("--param", action="append", default=[], help="NAME=VALUE (none: a repeat)")
    p.add_argument("--tool", action="append", default=[], help="a TOOL_CATALOGUE name to label")
    p.add_argument("--out", required=True)
    p.set_defaults(handler=cmd_derive)
    p = commands.add_parser("retry")
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+", help="runs whose qp-low rate quarantines get qp-mid jobs")
    p.set_defaults(handler=cmd_retry)
    p = commands.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(handler=cmd_compare)
    p = commands.add_parser("quality")
    p.add_argument("--version-dir", required=True)
    p.add_argument("--run", required=True, help="a run of repeat and lever-toggled jobs")
    p.add_argument("--out", required=True)
    p.set_defaults(handler=cmd_quality)
    p = commands.add_parser("card")
    p.add_argument("path", help="a run (with summary.json) or a released version")
    p.add_argument("--version", required=True)
    p.add_argument("--limitations", help="JSON list of known limitations")
    p.add_argument("--proposals", help="JSON {gap area: [clip, ...]}")
    p.add_argument("--out", help="directory for CARD.md and coverage.json (default: path)")
    p.set_defaults(handler=cmd_card)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
