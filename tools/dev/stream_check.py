#!/usr/bin/env python3
"""Whether two builds encode a chart family's streams bit-identically (chart_refresh.py's
`verify` step for --retime).

A refresh that changes a family's code but not its streams (an exact speed-up) only
needs the family re-timed: its quality points, cached by binary identity, stay valid
on its recorded build. This checks that claim on the streams the chart rests on:
the timing plan's fixed-QP encodes on the three timing clips (the speed axis: its
quantizers, each build's own `--preset`) and a CBR stream per timing clip at the
timing rung (the chart's quality mode, with the build's preset values), each
compared by SHA-256 between the old and the new build.

    python3 tools/dev/stream_check.py FAMILY OLD_BUILD_DIR NEW_BUILD_DIR
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLAN_FILE = REPO / "docs/results/bd-vs-speed-plan.json"
#: The chart's real-use encode length and key interval, and the CBR rung checked.
FRAMES = 150
INTRA_PERIOD = 300
CBR_KBPS = 600
#: Concurrent encodes (each one core, a few seconds).
JOBS = 4

Build = tuple[Path, Path]  # (scoring binary, presets directory)


def stream_sha(cmd: list[str], out: Path) -> str:
    run = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if run.returncode:
        raise SystemExit(f"encode failed: {' '.join(cmd[:4])} ...: {run.stderr[-300:]}")
    return hashlib.sha256(out.read_bytes()).hexdigest()


def encodes(family: str, build: Build, plan: dict, tmp: Path) -> list[tuple[list[str], Path]]:
    """The checked encodes of `family` on `build`: (command, the stream it writes)."""
    sys.path.insert(0, str(REPO / "eval"))
    from anchor_chart_timing import plan_family
    from nceval.config import load_paths
    from nceval.realuse import main_clips, own_family

    binary, presets = build
    clips = main_clips(load_paths())
    own = own_family(family, binary, presets)
    jobs = []
    for clip, entry in plan["points"][family].items():
        fixed_qp, cbr = tmp / f"{clip}-qp.bin", tmp / f"{clip}-cbr.bin"
        jobs.append((plan_family(own).command(clips[clip], FRAMES, INTRA_PERIOD, fixed_qp, None,
                                              qp=entry["qp"]), fixed_qp))
        jobs.append((own.command(clips[clip], FRAMES, INTRA_PERIOD, cbr, None, kbps=CBR_KBPS),
                     cbr))
    return jobs


def build_shas(family: str, build: Build, plan: dict) -> list[str]:
    with tempfile.TemporaryDirectory(dir=Path.home() / "nanocodec-data/work") as tmp:
        with ThreadPoolExecutor(JOBS) as pool:
            return list(pool.map(lambda job: stream_sha(*job),
                                 encodes(family, build, plan, Path(tmp))))


def identical(family: str, old: Build, new: Build, plan_file: Path = PLAN_FILE) -> bool:
    plan = json.loads(plan_file.read_text())
    return build_shas(family, old, plan) == build_shas(family, new, plan)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    family, old, new = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    same = identical(family, (old / "nanocodec", old / "presets"),
                     (new / "nanocodec", new / "presets"))
    print(f"{family}: {'identical' if same else 'CHANGED'}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
