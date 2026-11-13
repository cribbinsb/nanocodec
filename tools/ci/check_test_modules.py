#!/usr/bin/env python3
"""Fail if a crate's integration-test file would be silently skipped.

    python3 tools/ci/check_test_modules.py        (run by check.sh)

Crates link their integration tests as one binary (`autotests = false` and a
single `[[test]]` root that declares each `tests/*.rs` file as a module): a
binary per file spent most of the test build linking. With autotests off,
Cargo ignores a new `tests/foo.rs` that no root declares, so its tests would
never run; this check makes that an error instead.
"""

import re
import sys
import tomllib
from pathlib import Path

MOD_RE = re.compile(r"^\s*(?:pub\s+)?mod\s+([a-z0-9_]+)\s*;", re.MULTILINE)


def undeclared(crate_dir):
    """tests/*.rs files of a crate with autotests off that no [[test]] root declares."""
    manifest = tomllib.loads((crate_dir / "Cargo.toml").read_text())
    if manifest["package"].get("autotests", True):
        return []
    roots = {crate_dir / target["path"] for target in manifest.get("test", [])}
    declared = set()
    for root in roots:
        declared |= set(MOD_RE.findall(root.read_text()))
    return [path for path in sorted((crate_dir / "tests").glob("*.rs"))
            if path not in roots and path.stem not in declared]


def main():
    repo = Path(__file__).resolve().parents[2]
    missing = [path for crate in sorted((repo / "crates").iterdir()) if (crate / "Cargo.toml").is_file()
               for path in undeclared(crate)]
    for path in missing:
        print(f"{path.relative_to(repo)}: not a module of the crate's [[test]] root, so it never runs",
              file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
