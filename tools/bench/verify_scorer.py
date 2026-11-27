#!/usr/bin/env python3
"""Verify one build of the local GPU scorer against libvmaf 3.2.1 on CPU (V27): a separate
procedure, not part of any benchmark run (docs/benchmark-procedure.md D18, D19).

    systemd-run --user --scope -p MemoryMax=2G python3 tools/bench/verify_scorer.py RUN --series S

It scores a seeded sample of RUN's streams (1 %, at least 20, every codec) on this PC's GPU
service and on one cloud CPU box (`fleet.py score --ids`), compares them per frame, and on
agreement records the VMAFx build's SHA-256 in `bench/proven-scorers.json`. Benchmark scoring
(`fleet.py score --local`) refuses any build not recorded there. Run it once per VMAFx build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleet  # noqa: E402


def main() -> None:
    sys.argv = [sys.argv[0], "score", *sys.argv[1:], "--local", "--no-merge", "--tag", "verify",
                "--verify-scorer"]
    fleet.main()


if __name__ == "__main__":
    main()
