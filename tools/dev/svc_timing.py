#!/usr/bin/env python3
"""The SVC chart's encode speed on a rented spot box: x86 (Granite Rapids) or Arm (Graviton 4).

    python3 tools/dev/svc_timing.py --arch amd64 --binary BIN --plan PLAN --out DIR
    python3 tools/dev/svc_timing.py --arch arm64 --source-commit C --plan PLAN --out DIR

PLAN is nceval.svc_chart's timing plan (`svc_chart plan`: each point's command on the
timing clips at the 600 kbps rung, x264 medium among them). Each plan entry is timed alone
on one pinned physical core of a benchmark timing box (docs/aws-cost-guide.md row 1:
c8i.2xlarge, or c8g.2xlarge for Arm; spot, eu-north-1), `--rounds` times in rotated order
(tools/dev/svc_timing_box.py), and the CPU time per frame is reported per point: the median
over rounds per clip, the geometric mean over clips.

Arm: our encoder is built on the box from `--source-commit` (a `git archive` of it; the
plain `release` profile, as the x86 `--binary` should be) and libvpx's
vp9_spatial_svc_encoder from LIBVPX_TAG (no aarch64 build of it is kept); the other
references are the benchmark's Graviton builds (ARM_TOOLS).

Reclaim-safe: every encode's record comes back within PULL_SECONDS (aws_session.pulling)
and a relaunch with the same --out uploads the records so far, so the box codes only the
runs still missing. Every box is terminated on exit (aws_session.Session).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))
from aws_session import Session, SessionSpec, say  # noqa: E402
from aws_timing import clip_archive, runtime_libraries  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ANCHORS = Path.home() / "nanocodec-data" / "anchors"
#: The timing boxes (docs/aws-cost-guide.md, workload 1: benchmark absolute timing).
MACHINES = {"amd64": "c8i.2xlarge", "arm64": "c8g.2xlarge"}
REGION = "eu-north-1"
#: Where each reference program comes from, per architecture: the benchmark's LTO + PGO
#: builds (procedure section 7) where they hold it, else the plain anchor build.
X86_TOOLS = {"x264": ANCHORS / "x86_64-opt-bp4/bin/x264",
             "svc_encoder_rtc": ANCHORS / "x86_64-opt-bp4/bin/svc_encoder_rtc",
             "vp9_spatial_svc_encoder": ANCHORS / "x86_64/bin/vp9_spatial_svc_encoder"}
ARM_TOOLS = {"x264": ANCHORS / "aarch64-graviton-opt-bp4/bin/x264",
             "svc_encoder_rtc": ANCHORS / "aarch64-graviton-opt-bp4/bin/svc_encoder_rtc"}
#: libvpx's release whose example the quality points use (nceval.svc_chart: v1.16.0).
LIBVPX_TAG = "v1.16.0"
#: The core the encodes are pinned to (cores 0-1 are left to the system).
TIMED_CPU = 2
REMOTE = "svctiming"
FRAMES = 150


def stage(stage_dir: Path, plan: dict, binary: Path | None, source_commit: str | None,
          arch: str, done: Path) -> None:
    """The upload: the runner, the plan, the programs, the clips (150 frames, zstd), our
    source for an Arm build, and the records of an earlier attempt."""
    from nceval.config import load_paths
    from nceval.realuse import main_clips

    (stage_dir / "bin").mkdir(parents=True)
    (stage_dir / "clips").mkdir()
    (stage_dir / "out").mkdir()
    tools = X86_TOOLS if arch == "amd64" else ARM_TOOLS
    names = {e["argv"][0].removeprefix("{BIN}/") for p in plan["points"].values()
             for e in p.values()}
    for name in sorted(names):
        if name == "nanocodec" and binary:
            (stage_dir / "bin" / name).symlink_to(binary.resolve())
        elif name in tools:
            (stage_dir / "bin" / name).symlink_to(tools[name])
    clips = main_clips(load_paths())
    for name in sorted({c for p in plan["points"].values() for c in p}):
        (stage_dir / "clips" / f"{name}.yuv.zst").symlink_to(clip_archive(clips[name], FRAMES))
    shutil.copyfile(REPO / "tools/dev/svc_timing_box.py", stage_dir / "svc_timing_box.py")
    (stage_dir / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    if source_commit:
        with open(stage_dir / "source.tar", "wb") as tar:
            subprocess.run(["git", "-C", str(REPO), "archive", source_commit], stdout=tar,
                           check=True)
    if done.exists():
        shutil.copyfile(done, stage_dir / "out" / done.name)


def arm_builds() -> str:
    """Remote shell building our encoder (release profile) and libvpx's SVC example."""
    return (
        "sudo apt-get -qq update && sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y "
        "build-essential git nasm yasm zstd > /dev/null && "
        "curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal > /dev/null && "
        "mkdir -p src && tar -C src -xf source.tar && "
        f"{{ ( git clone -q --depth 1 --branch {LIBVPX_TAG} https://chromium.googlesource.com/webm/libvpx "
        "vpx && cd vpx && ./configure --enable-examples --disable-unit-tests --disable-docs "
        "--disable-vp8 > /dev/null && make -j4 > /dev/null && "
        "cp examples/vp9_spatial_svc_encoder ../bin/ ) & vpx=$!; } && "
        "( cd src && ~/.cargo/bin/cargo build -q --release -p nanocodec-cli -j4 && "
        "cp target/release/nanocodec ../bin/ ) && wait $vpx")


def remote_command(arch: str, rounds: int) -> str:
    cd = f"cd ~/{REMOTE} && "
    build = arm_builds() + " && " if arch == "arm64" else ""
    return (cd + build + "command -v zstd >/dev/null && "
            "for c in clips/*.zst; do zstd -d -q --rm \"$c\"; done && "
            + runtime_libraries() + "! ldd bin/* | grep 'not found' && bin/nanocodec --version && "
            "lscpu > out/lscpu.txt && "
            f"python3 -I svc_timing_box.py --plan plan.json --bin bin --clips clips "
            f"--out out/runs.jsonl --rounds {rounds} --cpu {TIMED_CPU}")


def summarize(runs: Path, plan: dict) -> dict:
    """Per point: CPU ms per frame (median over rounds per clip, geometric mean over clips)
    and frames per CPU-second."""
    by_entry: dict[tuple[str, str], list[float]] = {}
    for line in runs.read_text().splitlines():
        record = json.loads(line)
        by_entry.setdefault((record["point"], record["clip"]), []).append(record["cpu_seconds"])
    points = {}
    for point, clips in plan["points"].items():
        medians = {c: statistics.median(by_entry[(point, c)]) * 1000 / FRAMES
                   for c in clips if (point, c) in by_entry}
        if len(medians) != len(clips):
            continue
        ms = math.exp(sum(math.log(v) for v in medians.values()) / len(medians))
        points[point] = {"ms_per_frame": ms, "fps": 1000 / ms, "clips": medians,
                         "rounds": min(len(by_entry[(point, c)]) for c in clips)}
    return points


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=sorted(MACHINES), required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--binary", type=Path, help="our x86 release build (amd64)")
    ap.add_argument("--source-commit", help="the commit our Arm build is made from (arm64)")
    ap.add_argument("--points", help="comma-separated plan points (default: all)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=2.0, help="session budget, USD")
    ap.add_argument("--lifetime", type=int, default=75, help="hard instance lifetime, min")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.arch == "amd64") != bool(a.binary) or (a.arch == "arm64") != bool(a.source_commit):
        ap.error("amd64 takes --binary, arm64 --source-commit")
    plan = json.loads(a.plan.read_text())
    if a.points:
        plan["points"] = {n: v for n, v in plan["points"].items() if n in a.points.split(",")}
    if a.arch == "arm64":
        plan["points"] = {n: v for n, v in plan["points"].items()
                          if all(e["argv"][0].removeprefix("{BIN}/") in
                                 {*ARM_TOOLS, "nanocodec", "vp9_spatial_svc_encoder"}
                                 for e in v.values())}
    a.out.mkdir(parents=True, exist_ok=True)
    runs = a.out / "runs.jsonl"
    stage_dir = a.out / "stage"
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage(stage_dir, plan, a.binary, a.source_commit, a.arch, runs)
    spec = SessionSpec(region=REGION, instance_type=MACHINES[a.arch], arch=a.arch,
                       purpose="svc-timing", lifetime_minutes=a.lifetime, budget_usd=a.budget,
                       root_gb=30 if a.arch == "arm64" else 12)
    try:
        with Session(spec) as session:
            session.upload(stage_dir, f"~/{REMOTE}")
            with session.pulling(f"~/{REMOTE}/out", a.out / "pulled", "svc-timing"):
                session.run_detached(remote_command(a.arch, a.rounds),
                                     f"~/{REMOTE}/out/run.log", a.lifetime - 5,
                                     stall_minutes=20)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        pulled = a.out / "pulled" / runs.name
        if pulled.exists():
            shutil.copyfile(pulled, runs)
    summary = {"machine": MACHINES[a.arch], "region": REGION, "rounds": a.rounds,
               "date": datetime.date.today().isoformat(), "timed_cpu": TIMED_CPU,
               "points": summarize(runs, plan)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for name, p in sorted(summary["points"].items(), key=lambda kv: -kv[1]["fps"]):
        say(f"{name}: {p['fps']:.1f} fps ({p['rounds']} rounds)")


if __name__ == "__main__":
    main()
