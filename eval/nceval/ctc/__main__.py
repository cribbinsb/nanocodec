"""Command line for the standards lens. Run from the eval/ directory.

    python3 -m nceval.ctc stage --run-dir DIR --references DIR [--conditions aom-ld,jvet-e]
                                [--sequences A,B] [--encoders X,Y] [--points N]
                                [--source-dir /mnt/ctc] [--shard I/N]
                                [--skip-done RESULTS[,RESULTS...]]
    python3 -m nceval.ctc report --results DIR[,DIR...] --out DIR   (points.csv, bd.csv, tables.md)
    python3 -m nceval.ctc local | compare ...   (one variant on this machine: ctc/local.py)

`stage` writes a run directory for ctc/worker.py (jobs, sources, tools as
symlinks). `--references` is a directory with bin/{TAppEncoder,EncoderApp,
vvencFFapp} and cfg/ holding the three JVET configuration files (built by the
operator from the pinned tags, see the results page). Every other tool comes
from the anchors prefix and our build (`nceval.config.load_paths`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import load_paths
from . import plan as planner
from .local import main_local
from .report import report
from .tables import page_tables

WORKER = Path(__file__).with_name("worker.py")
DAV2D_CROP = Path(__file__).parents[1] / "dav2d_crop.py"
VMAF_V1_MODEL = Path("share/vmaf/model/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json")
REFERENCE_TOOLS = ("TAppEncoder", "EncoderApp", "vvencFFapp")


def tool_paths(references: Path) -> dict[str, Path]:
    """Every tool and configuration file a staged run may name, by file name."""
    paths = load_paths()
    tools = {p.name: p for p in (paths.anchors / "bin").iterdir()}
    tools["nanocodec"] = paths.nanocodec_bin / "nanocodec"
    tools.update({name: references / "bin" / name for name in REFERENCE_TOOLS})
    tools.update({p.name: p for p in (references / "cfg").iterdir()})
    return tools


def finished_ids(results: list[Path]) -> set[str]:
    """Job ids with a result already home (a rerun stages only the rest)."""
    return {str(p.relative_to(d).with_suffix("")) for d in results for p in d.rglob("*.json")}


def with_twin_rate(job: dict, results: list[Path]) -> dict:
    """A CBR job whose fixed-QP twin is already home carries that twin's rate."""
    if job["kind"] != "cbr":
        return job
    for directory in results:
        twin = directory / f"{job['rate_of']}.json"
        if twin.exists():
            return {**job, "target_kbps": json.loads(twin.read_text())["kbps"]}
    return job


def stage(args: argparse.Namespace) -> None:
    split = lambda value: value.split(",") if value else None  # noqa: E731
    jobs = planner.plan(args.conditions.split(","), load_paths(), Path(args.source_dir),
                        split(args.sequences), split(args.encoders), args.points)
    if args.skip_done:
        results = [Path(d) for d in args.skip_done.split(",")]
        done = finished_ids(results)
        jobs = [with_twin_rate(j, results) for j in jobs if j["id"] not in done]
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        jobs = planner.shard(jobs, index, count)
    run_dir = Path(args.run_dir)
    planner.stage(run_dir, jobs, tool_paths(Path(args.references)),
                  [WORKER, DAV2D_CROP, load_paths().anchors / VMAF_V1_MODEL])
    print(f"staged {len(jobs)} jobs, {len(planner.tools_of(jobs))} tools in {run_dir}")


def report_command(args: argparse.Namespace) -> None:
    summary = report([Path(d) for d in args.results.split(",")], Path(args.out))
    (Path(args.out) / "tables.md").write_text(page_tables(summary))
    print(f"{len(summary['points'])} points, {len(summary['rows'])} BD rows -> {args.out}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="nceval.ctc", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--references", required=True)
    s.add_argument("--conditions", default="aom-ld,jvet-e")
    s.add_argument("--sequences")
    s.add_argument("--encoders")
    s.add_argument("--points", type=int)
    s.add_argument("--shard", help="I/N: the I-th of N cost-balanced parts")
    s.add_argument("--skip-done", help="result directories (comma-separated) whose jobs are done")
    s.add_argument("--source-dir", default=str(planner.SOURCE_DIR))
    s.set_defaults(handler=stage)
    r = sub.add_parser("report")
    r.add_argument("--results", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(handler=report_command)
    main_local(sub)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main(sys.argv[1:])
