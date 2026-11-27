#!/usr/bin/env python3
"""Whether two builds encode a chart family's streams bit-identically (chart_refresh.py's
`verify` step for --retime).

A refresh that changes a family's code but not its streams (an exact speed-up) only
needs the family re-timed: its quality points, cached by binary identity, stay valid
on its recorded build. This checks that claim on the streams the chart's speed axis
times: the family's timing encodes (nceval.chart_timing: its own CBR quality encodes on
the plan's timing clips, each build with its own preset values), compared by SHA-256
between the old and the new build.

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
#: Concurrent encodes (each one core, a few seconds).
JOBS = 4

Build = tuple[Path, Path]  # (scoring binary, presets directory)


def stream_sha(bench, point, out: Path) -> str:
    """One timing encode (its clip's frames served as its quality point reads them)."""
    with bench.frames_of(point.clip) as frames:
        cmd = bench.command(point, out, None, source=frames)
        run = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if run.returncode:
        raise SystemExit(f"encode failed: {' '.join(cmd[:4])} ...: {run.stderr[-300:]}")
    return hashlib.sha256(out.read_bytes()).hexdigest()


def encodes(family: str, build: Build, plan: dict, tmp: Path) -> list[tuple]:
    """The checked encodes of `family` on `build`: (bench, point, the stream it writes)."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.chart import chart_clips
    from nceval.chart_timing import plan_jobs
    from nceval.config import load_paths
    from nceval.gatepolicy import EVALUATION_GATE
    from nceval.realuse import Runner, own_family

    binary, presets = build
    runner = Runner(load_paths(), EVALUATION_GATE, workers=1, over_budget=True)
    jobs = []
    for bench, point in plan_jobs(runner, [own_family(family, binary, presets)],
                                  chart_clips(plan["clipset"]), plan):
        jobs.append((bench, point, tmp / f"{point.clip.name}-{point.kbps}.bin"))
    return jobs


def build_shas(family: str, build: Build, plan: dict) -> list[str]:
    """The checked streams' SHA-256s of `family` on `build`, cached beside the build's
    binary (a build directory never changes): every refresh compared each family's
    recorded build again, encoding the same streams each time."""
    key = hashlib.sha256(json.dumps([plan["rate_mode"], plan["frames"], plan["clipset"],
                                     plan["clips"], plan.get("low_clips")],
                                    sort_keys=True).encode()).hexdigest()[:16]
    cache = build[0].parent / f"stream-shas-{family}-{key}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    with tempfile.TemporaryDirectory(dir=Path.home() / "nanocodec-data/work") as tmp:
        with ThreadPoolExecutor(JOBS) as pool:
            shas = list(pool.map(lambda job: stream_sha(*job),
                                 encodes(family, build, plan, Path(tmp))))
    cache.write_text(json.dumps(shas))
    return shas


def identical(family: str, old: Build, new: Build, plan_file: Path = PLAN_FILE) -> bool:
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.chart_timing import load_plan

    plan = load_plan(plan_file)
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
