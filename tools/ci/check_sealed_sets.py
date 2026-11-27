#!/usr/bin/env python3
"""Fail when code could read a release-only clip set outside a release measurement.

    python3 tools/ci/check_sealed_sets.py        (tools/ci/check.sh runs it)

The rule (nceval.sealing; docs/analysis/results-audit-2026-11-23.md finding 8):
a release-only held-out set is read by release and publish measurement only, never
by a tuning, screening or sweep tool. nceval.clips.load_clipset enforces it at run
time; this check keeps the ways around it out of the code. Over the tracked Python
and shell files:

  - only ALLOWED_NAMES name a release-only set (a tool given it by a literal name
    would be built around it);
  - only ALLOWED_ENV names the release switch (sealing.RELEASE_ENV): it is set by
    hand on a release command line, never by a tool;
  - only ALLOWED_EXCLUSION makes an `exclusion_only` load (families kept out of the
    pool and the learned-search data).

Each allowed file carries its reason.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "tools/ci"))
from check_names import tracked  # noqa: E402
from nceval.sealing import RELEASE_ENV, RELEASE_ONLY_SETS  # noqa: E402

THIS = "tools/ci/check_sealed_sets.py"
ALLOWED_NAMES = {
    "eval/nceval/sealing.py": "the registry of release-only sets",
    "tools/bench/preview.py": "names the set in the results page's dataset description only; "
                              "it reads reports, never clips",
    "eval/clipset/select_heldout.py": "selects the 2026-11-23 set",
    "eval/tests/test_sealing.py": "tests the seal",
    "eval/clipset/bench_raw.py": "release-measure tool: stages the sealed set's raw clips for a "
                                 "BP-3 run, refused without the release switch",
    "eval/clipset/history_select.py": "draws the BP-3 history subset from the manifest's names "
                                      "and stored grades only; the set's content is not read",
    "eval/tests/test_content_id.py": "reads only the manifest's names and source paths, to keep "
                                     "the PGO training set disjoint from the sealed set",
    THIS: "this check",
}
ALLOWED_ENV = {
    "eval/nceval/sealing.py": "defines the switch",
    "eval/tests/test_sealing.py": "tests the seal",
    "eval/clipset/bench_raw.py": "refuses the sealed set unless the switch is set on its "
                                 "command line",
    THIS: "this check",
}
ALLOWED_EXCLUSION = {
    "eval/nceval/sealing.py": "states the rule",
    "eval/nceval/clips.py": "defines the load",
    "eval/nceval/pool.py": "keeps sealed families out of the screening pool",
    "eval/nceval/nndata/splits.py": "keeps sealed families out of the learned-search data",
    "tools/dev/aws_heldout_chart.py": "keeps sealed families out of a held-out chart draw",
    "eval/tests/test_sealing.py": "tests the seal",
    THIS: "this check",
}
EXCLUSION = re.compile(r"\bexclusion_only\b")
ENV = re.compile(rf"\b(?:{RELEASE_ENV}|RELEASE_ENV)\b")
NAMES = re.compile("|".join(re.escape(n) for n in sorted(RELEASE_ONLY_SETS)))


def problems(root: Path = REPO) -> list[str]:
    """Every violation in `root`'s Python and shell files: git's list in a checkout, a
    walk of the tree on the build box's synced copy, which has no .git
    (check_names.tracked)."""
    found = []
    for path in (*tracked(".py", root), *tracked(".sh", root)):
        rel = path.relative_to(root).as_posix()
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        for pattern, allowed, what in ((NAMES, ALLOWED_NAMES, "names a release-only set"),
                                       (ENV, ALLOWED_ENV, f"names the release switch "
                                                          f"{RELEASE_ENV}"),
                                       (EXCLUSION, ALLOWED_EXCLUSION,
                                        "makes an exclusion_only clip set load")):
            if rel not in allowed:
                for number, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        found.append(f"{rel}:{number}: {what} (nceval.sealing): {line.strip()}")
    return found


def main() -> int:
    found = problems()
    for line in found:
        print(line)
    if found:
        print(f"{len(found)} sealed-set violation(s): only release and publish measurement may "
              "read a release-only set (docs/PROCEDURES.md \"Sealed held-out sets\")")
        return 1
    print("sealed sets: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
