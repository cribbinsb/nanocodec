#!/usr/bin/env python3
"""Verify that each day of the history series ran under the benchmark's conditions, on the
day's own streams (coordinator for the owner, 2026-11-24: "the rehearsal must VERIFY the
conditions took effect, not just that the command ran").

    python3 tools/bench/history_verify.py RUN... [--bound 1.0]

Per run (one day), from its records, scores and streams:
  - buffer: every CBR point replayed by the report's own bucket (`report.replay`, the phase's
    initial fullness); the share of frames over the bound must be 0;
  - rate: the achieved rate over the target, at most 1.0 on average (the target is a ceiling);
  - no frame delay: every stream decodes to exactly its input's frame count (the scorer's V9),
    one output frame per input frame;
  - keys: the only key frame of a 150-frame stream with a 300-frame key interval is frame 0
    (`ffprobe` key flags).
Prints a table and exits non-zero when any day fails a check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402

RATE_CEILING = 1.0
#: A tiny share of frames over the bound is still a failure: the check is exact.
MAX_FRAMES_OVER = 0.0


def keyframes(stream: Path) -> list[int]:
    """The indices of the stream's key frames (ffprobe)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=key_frame", "-of", "csv=p=0", str(stream)],
                         capture_output=True, text=True).stdout.split()
    return [i for i, flag in enumerate(out) if flag.strip().startswith("1")]


def verify(run: Path, bound: float) -> dict:
    jobs = {j["id"]: j for j in json.loads((run / "jobs.json").read_text())}
    init_s = json.loads((run / "manifest.json").read_text())["phase"]["INIT_MS"] / 1000
    base = next((run / "series").iterdir())
    scores = {r["id"]: r for r in report.read_jsonl(base / "scores.jsonl") if "pooled" in r}
    encodes = {r["id"]: r for r in report.read_jsonl(base / "encodes.jsonl") if report.is_job(r)}
    over = frames = cbr = 0
    rates, bad_keys, bad_frames = [], [], []
    for job_id, s in scores.items():
        job = jobs.get(job_id)
        if job is None or not job["family"].startswith("nanocodec"):
            continue
        if len(s["frame_bytes"]) != job["frames"]:
            bad_frames.append(job_id)
        if job["mode"] == "cbr" and not job.get("anchor_extra"):
            num, den = (int(x) for x in job["fps"].split("/"))
            r = report.replay(s["frame_bytes"], job["kbps"], num / den, init_s)
            cbr += 1
            frames += r["frames"]
            # over one buffer: the replay's own count; over a wider bound: the points past it
            over += r["frames_over"] if bound == 1.0 else int(r["peak_buffers"] > bound)
            rates.append(sum(s["frame_bytes"]) * 8 / (job["frames"] * den / num) / 1000
                         / job["kbps"])
        stream = base / encodes.get(job_id, {}).get("stream", "")
        if stream.is_file():
            keys = keyframes(stream)
            if keys[:1] != [0] or len(keys) != 1:
                bad_keys.append(f"{job_id}: keys {keys[:5]}")
    return {"run": str(run), "cbr_points": cbr,
            "share_frames_over": over / frames if frames else 0.0,
            "mean_rate_over_target": sum(rates) / len(rates) if rates else None,
            "bad_frame_counts": bad_frames, "bad_keys": bad_keys[:10]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--bound", type=float, default=1.15,
                    help="the buffer bound the day ran at (BP-3.12: our default, 1.15)")
    a = ap.parse_args()
    failed = 0
    for run in a.runs:
        v = verify(run, a.bound)
        ok = (v["share_frames_over"] <= MAX_FRAMES_OVER and not v["bad_frame_counts"]
              and not v["bad_keys"] and (v["mean_rate_over_target"] or 0) <= RATE_CEILING)
        failed += not ok
        rate = v["mean_rate_over_target"]
        print(f"{'ok  ' if ok else 'FAIL'} {run.parent.name}: CBR points {v['cbr_points']}, "
              f"frames over the bound {v['share_frames_over']:.3%}, rate/target "
              f"{'-' if rate is None else f'{rate:.3f}'}, wrong frame counts "
              f"{len(v['bad_frame_counts'])}, wrong keys {len(v['bad_keys'])}"
              + (f" e.g. {v['bad_keys'][0]}" if v["bad_keys"] else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
