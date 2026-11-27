#!/usr/bin/env python3
"""Re-encode the dev baseline points once, centrally, after our benchmark conditions change.

    python3 tools/dev/rebaseline.py [--commit COMMIT] [--codecs h264,hevc,vvc,av1,vp9,av2]
                                    [--dry-run]

Why (owner, 2026-11-25): the dev screens run our encoder under the benchmark's conditions
(nceval.fairness.own_conditions, locked in eval/tests/test_conditions_lock.py). When those
change, every cached baseline point is stale, and the next agent to screen paid for all of
them mid-task (an hour that day). land.sh prints the command to run this whenever a landed
commit changes the lock.

What: for each codec's medium rung, one screen of the landed build against itself on the
day's rotating draw, every stage (L1, L2, L3) and the long windows (600 f with keys every
150, 900 f with keys every 300: --final-frames), so every point a later screen's anchor
needs is in the cache. One aws_measure --batch on a warm Graviton box with GPU scoring
(refused, never scored on a box, when the GPU cannot score; its own budget). Its results change no decision: candidate
and anchor are the same build.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from nceval import baseline_state  # noqa: E402  (eval/ is not an installed package)

CODECS = ("h264", "hevc", "vvc", "av1", "vp9", "av2")
#: The batch's spend cap (aws_measure --budget): six codecs' encode rounds on warm Graviton.
BUDGET_USD = 1.0
#: The candidate's setting (a screen refuses a candidate identical to its anchor): one the
#: benchmark conditions already set, so its streams are the anchor's byte for byte, and
#: scores by stream (nceval.cbr) score each once: the candidate costs its encodes only.
NEUTRAL_PARAM = "stream.encoder_info=0"


def commands(binary: Path, codecs: list[str], out: Path) -> list[list[str]]:
    return [["python3", "-m", "nceval.screen", "--codec", codec, "--binary", str(binary),
             "--param", NEUTRAL_PARAM,
             "--all-stages", "--final-frames", "--json", str(out / f"rebaseline-{codec}.json")]
            for codec in codecs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", a.commit], check=True,
                            capture_output=True, text=True).stdout.strip()
    binary = Path(subprocess.run([str(REPO / "tools/dev/binary_for.sh"), commit], check=True,
                                 capture_output=True, text=True).stdout.strip().splitlines()[-1]) / "nanocodec"
    with tempfile.TemporaryDirectory(prefix="nanocodec-rebaseline-") as tmp:
        batch = Path(tmp) / "batch.json"
        batch.write_text(json.dumps(commands(binary, a.codecs.split(","), Path(tmp))))
        argv = [sys.executable, str(REPO / "tools/dev/aws_measure.py"), "--batch", str(batch),
                "--budget", str(BUDGET_USD), *(["--dry-run"] if a.dry_run else [])]
        if a.dry_run:
            sys.exit(subprocess.call(argv, cwd=REPO))
        baseline_state.record(commit, baseline_state.RUNNING)
        status = subprocess.call(argv, cwd=REPO)
    baseline_state.record(commit, baseline_state.FAILED if status else baseline_state.OK,
                          f"aws_measure exit {status}; see this run's log" if status else "")
    if status:
        print(baseline_state.notice(), file=sys.stderr)
    sys.exit(status)


if __name__ == "__main__":
    main()
