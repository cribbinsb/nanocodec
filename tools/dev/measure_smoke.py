#!/usr/bin/env python3
"""The dev measurement path end to end in a few minutes: run it before any long run.

    python3 tools/dev/measure_smoke.py [--commit COMMIT]

One staged-off screen of H.264 medium (candidate: hold on, SMOKE_PARAM) on 2 small train clips (60 frames), through
aws_measure with GPU scoring: a warm Graviton box (baked image), the encode
round, the stream transfer, the PC's GPU scoring service, the result and its phase timeline.
It fails loudly (non-zero exit, one line saying why) on any error, on any phase over its
budget (aws_measure's OVER BUDGET lines), or when the whole run passes SMOKE_BUDGET_S.
Its streams are identical to the anchor's, so it changes no decision anywhere.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SMOKE_CLIPS = ("texture_stockholm_360p", "bodycam_antare_0815")
SMOKE_FRAMES = 60
#: The candidate's one setting: a screen refuses a candidate identical to its anchor. Hold
#: leaves these clips' streams unchanged (no repeated frames), so the run decides nothing.
SMOKE_PARAM = "hold.enable=1"
#: Measured 2026-11-25: 404-423 s (two rounds: a fresh box from the baked image ~100 s, the
#: PC's GPU scoring ~100 s); the budget leaves ~15 % on that.
SMOKE_BUDGET_S = 480
SMOKE_USD = 0.10


def binary_dir(commit: str) -> str:
    # binary_for.sh prints the directory last (after any build-cache notes)
    return subprocess.run([str(REPO / "tools/dev/binary_for.sh"), commit], check=True,
                          capture_output=True, text=True).stdout.strip().splitlines()[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default="HEAD")
    a = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", a.commit], check=True,
                            capture_output=True, text=True).stdout.strip()
    binary = Path(binary_dir(commit)) / "nanocodec"
    with tempfile.TemporaryDirectory(prefix="nanocodec-smoke-") as tmp:
        out = Path(tmp) / "smoke.json"
        argv = [sys.executable, str(REPO / "tools/dev/aws_measure.py"),
                "--budget", str(SMOKE_USD), "--",
                "python3", "-m", "nceval.screen", "--codec", "h264", "--binary", str(binary),
                "--param", SMOKE_PARAM,
                "--clips", ",".join(SMOKE_CLIPS), "--no-staged", "--frames", str(SMOKE_FRAMES),
                "--json", str(out)]
        started = time.monotonic()
        run = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
        wall = time.monotonic() - started
        log = run.stdout + run.stderr
        print(log[-4000:])
        problems = []
        if run.returncode:
            problems.append(f"aws_measure exited {run.returncode}")
        problems += [line.strip() for line in log.splitlines() if "OVER BUDGET" in line]
        if wall > SMOKE_BUDGET_S:
            problems.append(f"the smoke run took {wall:.0f} s, over its {SMOKE_BUDGET_S} s budget")
        if not run.returncode and not (out.exists() and json.loads(out.read_text())):
            problems.append("no result JSON")
    if problems:
        for p in problems:
            print(f"SMOKE FAILED: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"smoke ok: {wall:.0f} s end to end")


if __name__ == "__main__":
    main()
