"""The partition transplant's identity and work, per rung and clip (runs on the build box).

    python3 -I eval/distill/replay_check.py --binary target/gate/nanocodec --arms ARMS.json \\
        --clips-dir ~/nanocodec-data/clips/quick --out OUT.json [--frames 150] [--bitrate 600]

ARMS.json (written on the PC by `python3 -m distill.arms_json`): a list of
{"codec", "rung", "arm", "params": [k=v ...]}, the params on top of `--preset rung`.

Per (arm, clip), three encodes of the clip's first frames in CBR:
  fast   the rung alone;
  search the rung with the arm's settings (the slowest rung's partition search), recording
         its partitions (`--partition-record`);
  replay the rung coding those partitions (`--partition-replay`), no search.
It reports whether replay's stream equals search's (then replay is exactly the search's
coding at the replay's cost: a perfect partition guide), each encode's user-space
instructions (`perf stat`, else none) and CPU seconds, and the stream sizes. Nothing but
the logs and streams is written, beside OUT (deleted at the end; `--keep-logs` keeps the
logs, for tools/dev/aws_cpu_ab.py to replay them on the CPU targets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLIPS = {  # the four quick clips of the CPU A/B tool (tools/dev/aws_cpu_ab.py)
    "crowd_mot17_09": (1280, 720, "30"),
    "bodycam_airport": (1280, 720, "29.97"),
    "pan_jaad_0117": (1280, 720, "30"),
    "lowlight_pp22_004": (1280, 720, "24"),
}


def perf_available() -> bool:
    try:
        done = subprocess.run(["perf", "stat", "-x,", "-e", "instructions:u", "true"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and "instructions" in done.stderr and "<not" not in done.stderr


def run(argv: list[str], perf: bool) -> dict:
    """Run one encode; its instructions (perf) and the children's CPU seconds."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    command = (["perf", "stat", "-x,", "-e", "instructions:u", "--"] if perf else []) + argv
    done = subprocess.run(command, capture_output=True, text=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if done.returncode != 0:
        raise SystemExit(f"{' '.join(argv)}: {done.stderr[-2000:]}")
    instructions = None
    if perf:
        for line in done.stderr.splitlines():
            fields = line.split(",")
            if len(fields) > 2 and fields[2].startswith("instructions"):
                instructions = int(fields[0])
    # Threads share RUSAGE_CHILDREN; the caller runs encodes in threads, so this is only
    # indicative. Instructions are the measure.
    return {"instructions": instructions, "cpu_s": after.ru_utime - before.ru_utime}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(job: dict, args, work: Path, perf: bool) -> dict:
    name = job["clip"]
    width, height, fps = CLIPS[name]
    tag = f"{job['codec']}-{job['rung']}-{job['arm']}-{name}"
    common = [str(args.binary), "encode", "--codec", job["codec"], "--preset", job["rung"],
              "--input", str(args.clips_dir / f"{name}.yuv"), "--width", str(width),
              "--height", str(height), "--fps", fps, "--frames", str(args.frames),
              "--bitrate", str(args.bitrate)]
    arm = [a for p in job["params"] for a in ("--param", p)]
    out = {v: work / f"{tag}-{v}.bin" for v in ("fast", "search", "replay")}
    log = work / f"{tag}.plog"
    result = {"codec": job["codec"], "rung": job["rung"], "arm": job["arm"], "clip": name}
    result["fast"] = run(common + ["--output", str(out["fast"])], perf)
    result["search"] = run(common + arm + ["--output", str(out["search"]),
                                           "--partition-record", str(log)], perf)
    result["replay"] = run(common + ["--output", str(out["replay"]),
                                     "--partition-replay", str(log)], perf)
    for variant, path in out.items():
        result[variant]["bytes"] = path.stat().st_size
    result["identical"] = digest(out["search"]) == digest(out["replay"])
    result["log_bytes"] = log.stat().st_size
    for path in out.values():
        path.unlink()
    if args.keep_logs:
        kept = args.keep_logs / f"{job['codec']}-{job['rung']}" / f"{name}.plog"
        kept.parent.mkdir(parents=True, exist_ok=True)
        log.replace(kept)
    else:
        log.unlink()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--arms", required=True, type=Path)
    p.add_argument("--clips-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--bitrate", type=int, default=600)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    p.add_argument("--clips", default=",".join(CLIPS))
    p.add_argument("--keep-logs", type=Path,
                   help="keep each partition log as DIR/<codec>-<rung>/<clip>.plog")
    args = p.parse_args()
    perf = perf_available()
    print(f"perf instructions: {'yes' if perf else 'no'}", flush=True)
    jobs = [{**arm, "clip": clip} for arm in json.loads(args.arms.read_text())
            for clip in args.clips.split(",")]
    work = args.out.parent / "nanocodec-replay-work"
    work.mkdir(parents=True)
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            results = []
            for r in pool.map(lambda j: one(j, args, work, perf), jobs):
                results.append(r)
                ins = {v: r[v]["instructions"] for v in ("fast", "search", "replay")}
                print(f"{r['codec']} {r['rung']} {r['arm']} {r['clip']}: identical "
                      f"{r['identical']} instructions {ins}", flush=True)
    finally:
        shutil.rmtree(work)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
