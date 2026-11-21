#!/usr/bin/env python3
"""Keep what the retention manifest and the NANOCODEC-KEEP.md markers keep; sweep only
what is declared regenerable (eval/nceval/retention.py).

    python3 tools/dev/retention.py check PATH...      # non-zero unless every PATH may go
    python3 tools/dev/retention.py status [--all]     # sweepable, live and UNKNOWN paths, sizes
    python3 tools/dev/retention.py sweep [--apply]    # remove the sweepable ones (lists without --apply)
    python3 tools/dev/retention.py mark               # write the markers of the kept directories
    python3 tools/dev/retention.py verify-release     # the release's durable stores, or fail

Nothing under ~/nanocodec-data is deleted unless `check` passes (docs/PROCEDURES.md
"Data retention"): a kept or marked path, a directory holding one, and an UNKNOWN
(undeclared) path are refused. Only an agent's own scratch (scratch/<agent>/) and build
target (targets/<agent>/) pass without a manifest entry. `sweep --apply` removes the
declared paths that are past the grace period and unused, one literal path at a time,
printing each with its bytes. `verify-release` is the release's pre-flight (opt-in until
after 0.6.0): it names each missing store and what it costs to regenerate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import retention  # noqa: E402

GB = 1e9
STATES = ("sweepable", "unknown", "live")


def show_status(show_all: bool) -> int:
    rows = retention.status()
    for state in STATES:
        group = sorted((r for r in rows if r.state == state), key=lambda r: -r.bytes)
        if state == "sweepable" or show_all:
            for r in group:
                print(f"{r.bytes / GB:7.2f} GB {r.age_hours:6.0f} h  {state:9}  {r.rel}"
                      f"  ({r.why})")
        print(f"== {state}: {len(group)} paths, {sum(r.bytes for r in group) / GB:.1f} GB")
    return 0


def check(paths: list[str]) -> int:
    refused = 0
    for raw in paths:
        path = Path(raw).expanduser().absolute()
        why = retention.check(path)
        if why is not None:
            print(f"refused: {why}", file=sys.stderr)
            refused += 1
        else:
            print(f"ok: {path} may be deleted")
    return 1 if refused else 0


def mark() -> int:
    for directory, how in retention.write_markers():
        print(f"{how:8} {directory / retention.MARKER}")
    return 0


def verify_release() -> int:
    problems = retention.verify_release()
    for line in problems:
        print(f"RELEASE PRE-FLIGHT: {line}", file=sys.stderr)
    if not problems:
        print("release pre-flight: every durable store the release needs is present")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--all", action="store_true", help="also list UNKNOWN and live paths")
    sub.add_parser("check").add_argument("paths", nargs="+")
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--apply", action="store_true", help="remove (default: list only)")
    sub.add_parser("mark")
    sub.add_parser("verify-release")
    args = parser.parse_args()
    if args.command == "status":
        return show_status(args.all)
    if args.command == "check":
        return check(args.paths)
    if args.command == "sweep":
        retention.sweep(apply=args.apply)
        return 0
    if args.command == "mark":
        return mark()
    return verify_release()


if __name__ == "__main__":
    sys.exit(main())
