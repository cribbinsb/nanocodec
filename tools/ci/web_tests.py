#!/usr/bin/env python3
"""Run the page tests in headless Chromium (web/tests/*.py) in parallel, only
the ones a change can affect.

Called by tools/ci/browser_smoke.sh after it has built web/pkg once: every
test serves that same build (its own port, its own browser). Run one after
another they took 10-39 min under load; side by side they take about as long
as the slowest one.

Selection (`--since REV`, the changed paths since REV; no REV: all tests):
every test whose own file or whose listed page files changed runs. The whole
set runs when anything shared changed: the module and the crates it embeds,
the build, wasm.js, index.html, the shared site header, this runner. Any other changed
file under web/ runs the demo-page tests (they all load the whole page).
The product smoke always runs.

    python3 tools/ci/web_tests.py [--since REV] [--workers N] [--list | --needs-native]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR_DEFAULT = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data")) / "work/web-tests"
#: Side by side, each test's browser plus its codec workers.
DEFAULT_WORKERS = 6
#: The timing-sensitive tests (frame rates, skips, pauses measured against
#: thresholds) run in a second phase, this many at a time: six side by side at
#: load ~75 failed exact-source, live-skip and psnr-map (2026-11-15); two at a
#: time most pass. One that fails runs once more (the retries again two at a
#: time) before it counts.
TIMED_WORKERS = 2
#: A test still running after this long has hung: it fails (the slowest,
#: psnr-map, takes ~5 min beside the others under load).
TEST_TIMEOUT_S = 900
#: Lines of a failed test's log shown inline.
FAILURE_TAIL = 40

#: Changed paths (prefixes) that can affect every test.
SHARED = (
    "crates/", "Cargo.toml", "Cargo.lock", "presets/", "tools/web/build.sh",
    "tools/ci/browser_smoke.sh", "tools/ci/web_tests.py",
    "web/wasm.js", "web/site-header.css", "web/index.html",
)


@dataclass(frozen=True)
class PageTest:
    script: str
    #: Feature files only this test (and the others listing them) exercises.
    files: tuple[str, ...] = ()
    #: Whether it runs the demo page's code (index.html, its modules, workers):
    #: then any change to a page file no test lists runs it.
    demo: bool = True
    args: tuple[str, ...] = ()
    #: Needs NC_NANOCODEC_BIN (a native build of this tree).
    native: bool = False
    #: Measures speed against a threshold: runs in the TIMED_WORKERS phase.
    timed: bool = False


TESTS = (
    PageTest("browser-product-smoke.py"),
    PageTest("warmup-interrupt.py", ("web/warmup.js", "web/tests/warmup-interrupt.html")),
    PageTest("source-switch.py", ("web/scene.js", "web/clip-source.js")),
    PageTest("reference-psnr.py", ("web/reference.js", "web/streams.js", "web/streams-ref/")),
    PageTest("exact-source.py", ("web/exact-source.js", "web/clip-source.js"), timed=True),
    PageTest("clip-pauses.py", ("web/clip-source.js", "web/exact-source.js", "web/clips/"), timed=True),
    PageTest("live-skip.py", timed=True),
    PageTest("speed-hint.py", ("web/speed-hint.js",), timed=True),
    PageTest("results-pages.py", ("web/results/",), demo=False),
    PageTest("deep-links.py", ("web/results/", "web/deep-link.js")),
    PageTest("docs-pages.py", ("web/docs/",), demo=False),
    PageTest("analysis-smoke.py", ("web/analysis.js", "web/analysis-view.js", "web/view-zoom.js", "web/metrics-worker.js")),
    PageTest("psnr-map.py", ("web/error-map.js", "web/analysis-view.js", "web/metrics-worker.js"), timed=True),
    PageTest("capture-identity.py", ("web/mp4.js", "tools/web/source_frames.py", "web/tests/capture-identity.html"),
             args=("--clips", "3")),
    PageTest("svc-operating-points.py", native=True),
    PageTest("debug-report.py", ("web/debug-report.js", "tools/web/replay-debug-report.py"), native=True),
    PageTest("still-camera.py", ("web/debug-report.js",), native=True),
    PageTest("original-view.py", ("web/original-worker.js",), native=True),
    PageTest("auto-restart.py"),
    PageTest("view-layout.py", ("web/layout.js", "web/view-zoom.js", "web/view-settings.js", "web/analysis-view.js")),
)


def changed_paths(since: str) -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", since, "HEAD"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    return [p for p in out.splitlines() if p]


def select(paths: list[str] | None) -> list[PageTest]:
    """The tests the changed `paths` can affect (None: all of them)."""
    if paths is None or any(p.startswith(SHARED) for p in paths):
        return list(TESTS)
    chosen = {TESTS[0].script}
    for path in paths:
        mapped = False
        for test in TESTS:
            if path == f"web/tests/{test.script}" or path.startswith(test.files):
                chosen.add(test.script)
                mapped = True
        if not mapped and path.startswith("web/") and not path.startswith("web/tests/"):
            chosen.update(t.script for t in TESTS if t.demo)
    return [t for t in TESTS if t.script in chosen]


def run(test: PageTest, log_dir: Path) -> tuple[PageTest, int, float, Path]:
    log = log_dir / f"{test.script}.log"
    start = time.monotonic()
    with log.open("w") as out:
        try:
            status = subprocess.run([sys.executable, f"web/tests/{test.script}", *test.args], cwd=REPO,
                                    stdout=out, stderr=subprocess.STDOUT, timeout=TEST_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            print(f"timed out after {TEST_TIMEOUT_S} s", file=out)
            status = 1
    return test, status, time.monotonic() - start, log


def run_phase(tests: list[PageTest], workers: int, log_dir: Path, note: str = "") -> list[tuple[PageTest, Path]]:
    """Run `tests`, `workers` at a time; the (test, log) of each that failed."""
    failed = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for test, status, seconds, log in pool.map(lambda t: run(t, log_dir), tests):
            print(f"{'PASS' if status == 0 else 'FAIL'} {test.script} {seconds:.0f}s{note}", flush=True)
            if status != 0:
                failed.append((test, log))
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--since", default=os.environ.get("NC_WEB_TESTS_SINCE") or None,
                        help="run only the tests the changes since this revision can affect (NC_WEB_TESTS_SINCE)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("NC_WEB_TEST_WORKERS", DEFAULT_WORKERS)))
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR_DEFAULT)
    parser.add_argument("--list", action="store_true", help="print the selection and exit")
    parser.add_argument("--needs-native", action="store_true",
                        help="exit 0 if the selection includes a test that needs NC_NANOCODEC_BIN, else 1")
    args = parser.parse_args()

    tests = select(changed_paths(args.since) if args.since else None)
    if args.needs_native:
        sys.exit(0 if any(t.native for t in tests) else 1)
    print(f"page tests: {len(tests)} of {len(TESTS)}, {args.workers} at a time, then the timed ones {TIMED_WORKERS} at a time: "
          + " ".join(t.script.removesuffix(".py") for t in tests), flush=True)
    if args.list:
        return
    if any(t.native for t in tests) and not os.environ.get("NC_NANOCODEC_BIN"):
        raise SystemExit("set NC_NANOCODEC_BIN (a native build of this tree) for the native-replay tests")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    start, failed = time.monotonic(), []
    untimed = [t for t in tests if not t.timed]
    timed = [t for t in tests if t.timed]
    failed += run_phase(untimed, args.workers, args.log_dir)
    retry = run_phase(timed, TIMED_WORKERS, args.log_dir)
    failed += run_phase([test for test, _ in retry], TIMED_WORKERS, args.log_dir, " (retry)")
    for _, log in failed:
        print(f"== {log} (last {FAILURE_TAIL} lines)", file=sys.stderr)
        print("\n".join(log.read_text().splitlines()[-FAILURE_TAIL:]), file=sys.stderr)
    print(f"page tests: {len(tests) - len(failed)}/{len(tests)} passed in {time.monotonic() - start:.0f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
