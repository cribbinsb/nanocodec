#!/usr/bin/env python3
"""Refuse a landing whose nceval.train verdict tripped a guard without saying why
(used by land.sh; the rule is in eval/nceval/train_guards.py).

    python3 tools/dev/land_train_guards.py <integration-ref>     (in the candidate worktree)

Exit 0 if every candidate commit is clean, 1 listing the commits to fix.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval.train_guards import JUSTIFICATION_PREFIX, unjustified  # noqa: E402


def candidate_messages(base: str) -> list[tuple[str, str]]:
    """(short sha, message) of each commit on HEAD that `base` lacks."""
    shas = subprocess.run(["git", "rev-list", f"{base}..HEAD"], check=True,
                          capture_output=True, text=True).stdout.split()
    return [(sha[:9], subprocess.run(["git", "log", "-1", "--format=%B", sha], check=True,
                                     capture_output=True, text=True).stdout)
            for sha in shas]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    bad = [sha for sha, message in candidate_messages(argv[0]) if unjustified(message)]
    for sha in bad:
        print(f"{sha}: nceval.train guards tripped and no '{JUSTIFICATION_PREFIX} <why>' line",
              file=sys.stderr)
    if bad:
        print(f"Fix: look at what each guard flags (the screens' per-clip rows) and either drop "
              f"the change or write '{JUSTIFICATION_PREFIX} <why it is acceptable>' in the "
              f"commit message.", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
