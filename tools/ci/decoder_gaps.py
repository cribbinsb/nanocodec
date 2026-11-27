#!/usr/bin/env python3
"""Every decoder rejection path is in the compliance register.

Scans each decoder crate (crates/nc-{h264,hevc,vvc,av1,vp9,av2}/src) for the
names its decoder rejects a stream with (`DecodeError::Unsupported("...")`
and the few helpers that wrap it) and checks that each name appears, in
backticks, in docs/decoder-compliance.md: in that codec's feature table, whose
rows carry a status, or in its "Rejections that are not feature gaps" table.
A new rejection path without a register row fails the check, so the register
cannot silently fall behind the code (owner, 2026-11-27).

Encoder-side errors are not decoder gaps and are excluded by construction:
`ConfigError::Unsupported` is never matched, and the decoder-crate files that
hold only encoder or test-oracle code are listed in EXCLUDED_FILES with their
reason.

  tools/ci/decoder_gaps.py          check (exit 1 on any unregistered name)
  tools/ci/decoder_gaps.py --list   print every name with its first site
  tools/ci/decoder_gaps.py --summary  print the register's status counts per codec

Every feature row (a table row with a Status column) must carry one of
STATUSES, so a row cannot hide a gap behind free text.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "decoder-compliance.md"
CODECS = ("h264", "hevc", "vvc", "av1", "vp9", "av2")
# The statuses a feature row may have (the register's legend).
STATUSES = ("implemented", "in progress", "open", "proposed exception")

# Files in decoder crates whose `DecodeError::Unsupported` names are not
# rejections of a stream by the decoder.
EXCLUDED_FILES = {
    # The headers the AV2 encoder writes, read back as its tests' oracle; the
    # decoder parses every frame header with general/frame_header.
    "nc-av2/src/frame_header/read_back.rs": "AV2 encoder test oracle",
    # The AV2 encoder's frame-header writer validating its own input.
    "nc-av2/src/frame_header.rs": "AV2 encoder frame-header writer",
    # The subset sequence parse only routes a stream: when it refuses one, the
    # decoder takes the general path (general/decoder.rs, `subset_sequence`).
    "nc-av2/src/sequence.rs": "AV2 subset-path routing (falls back to the general decoder)",
    # Likewise the subset frame-tool check (general/decoder.rs `tile_params`).
    "nc-av2/src/general/subset.rs": "AV2 subset-path routing (falls back to the general decoder)",
}

# A string literal directly inside `DecodeError::Unsupported(`.
DIRECT = re.compile(r'DecodeError::Unsupported\(\s*"((?:[^"\\]|\\.)*)"', re.S)
# Calls of the crates' helpers that wrap `DecodeError::Unsupported(name)`:
# av1 decoder/subset.rs `unsupported(..)`, vp9 decoder/header.rs `expect(c, ..)`,
# vvc decoder_blocks/mod.rs's `(present, "name")` table.
HELPERS = [
    re.compile(r'(?<![.\w])unsupported\(\s*"((?:[^"\\]|\\.)*)"', re.S),
    re.compile(r'(?<![.\w])expect\(\s*[^;]*?,\s*"((?:[^"\\]|\\.)*)"\s*,?\s*\)', re.S),
]
# `DecodeError::Unsupported(<identifier>)`: a helper. Each must be one whose
# call sites HELPERS (or VVC_TABLE) reads; a new helper fails until it is added.
DYNAMIC = re.compile(r'DecodeError::Unsupported\(\s*([a-z_]\w*)\s*\)')
KNOWN_DYNAMIC = {
    ("nc-av1/src/decoder/subset.rs", "tool"),
    ("nc-vp9/src/decoder/header.rs", "what"),
    ("nc-vvc/src/decoder_blocks/mod.rs", "tool"),
}
VVC_TABLE = re.compile(r'let unsupported = \[(.*?)\];', re.S)
TABLE_ENTRY = re.compile(r'\([^()]*?,\s*"((?:[^"\\]|\\.)*)"\s*\)')


TEST_MODULE = re.compile(r"#\[cfg\(test\)\]\s*(?:pub(?:\([\w:]+\))?\s+)?mod\s+\w+\s*\{")


def strip_tests(text: str) -> str:
    """Drop the inline `#[cfg(test)] mod` (by convention the file's tail)."""
    cut = TEST_MODULE.search(text)
    return text if cut is None else text[: cut.start()]


def strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def rejection_names(codec: str) -> tuple[dict[str, str], list[str]]:
    """Each rejection name of a decoder crate with its first site, and errors."""
    names: dict[str, str] = {}
    errors: list[str] = []
    src = ROOT / "crates" / f"nc-{codec}" / "src"
    for path in sorted(src.rglob("*.rs")):
        rel = path.relative_to(ROOT / "crates").as_posix()
        if rel in EXCLUDED_FILES or path.name == "tests.rs" or "/tests/" in rel:
            continue
        text = strip_comments(strip_tests(path.read_text()))
        found = [m for pattern in [DIRECT, *HELPERS] for m in pattern.finditer(text)]
        for table in VVC_TABLE.finditer(text):
            found += TABLE_ENTRY.finditer(table.group(1))
        for m in found:
            line = text.count("\n", 0, m.start()) + 1
            names.setdefault(m.group(1), f"{rel}:{line}")
        for m in DYNAMIC.finditer(text):
            if (rel, m.group(1)) not in KNOWN_DYNAMIC:
                line = text.count("\n", 0, m.start()) + 1
                errors.append(
                    f"{rel}:{line}: DecodeError::Unsupported({m.group(1)}) is a helper "
                    "this script does not read; add its call pattern to HELPERS"
                )
    return names, errors


def registered(codec: str, register: str) -> set[str]:
    """The backticked names in the register's section for a codec."""
    heading = re.compile(rf"^## .*\[{codec}\]", re.M | re.I)
    start = heading.search(register)
    if not start:
        return set()
    end = re.search(r"^## ", register[start.end():], re.M)
    section = register[start.end(): start.end() + end.start()] if end else register[start.end():]
    return set(re.findall(r"`([^`]+)`", section))


def feature_rows(register: str) -> list[tuple[str, str, str]]:
    """(section, feature, status) of every row of a table with a Status column."""
    rows = []
    section, status_col = "", None
    for line in register.splitlines():
        if line.startswith("## "):
            section, status_col = line[3:].split(" [")[0].strip(), None
            continue
        if not line.startswith("|"):
            status_col = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if status_col is None:
            status_col = cells.index("Status") if "Status" in cells else -1
        elif status_col >= 0 and not set(line) <= set("|-: "):
            rows.append((section, cells[0], cells[status_col]))
    return rows


def status_failures(rows: list[tuple[str, str, str]]) -> list[str]:
    return [
        f"{section}: row '{feature[:60]}' has status '{status[:40]}', not one of {STATUSES}"
        for section, feature, status in rows
        if not status.startswith(STATUSES)
    ]


def summary(rows: list[tuple[str, str, str]]) -> str:
    """The register's summary table: feature rows per status per section."""
    sections = list(dict.fromkeys(section for section, _, _ in rows))
    lines = ["| Codec | " + " | ".join(STATUSES) + " |", "|---|" + "---|" * len(STATUSES)]
    for section in sections:
        counts = [
            sum(1 for s, _, status in rows if s == section and status.startswith(kind))
            for kind in STATUSES
        ]
        lines.append(f"| {section} | " + " | ".join(map(str, counts)) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print every rejection name")
    parser.add_argument("--summary", action="store_true", help="print status counts per codec")
    args = parser.parse_args()
    register = REGISTER.read_text()
    rows = feature_rows(register)
    if args.summary:
        print(summary(rows), end="")
        return 0
    failures = status_failures(rows)
    if summary(rows) not in register:
        failures.append(f"the summary table is stale; replace it with `{Path(__file__).name} --summary`")
    total = 0
    for codec in CODECS:
        names, errors = rejection_names(codec)
        failures += errors
        total += len(names)
        known = registered(codec, register)
        for name, site in sorted(names.items()):
            if args.list:
                print(f"{codec}\t{name}\t{site}")
            if name not in known:
                failures.append(f"{site}: rejection `{name}` has no row in {REGISTER.name} [{codec}]")
    for failure in failures:
        print(f"decoder_gaps: {failure}", file=sys.stderr)
    if failures:
        print(f"decoder_gaps: {len(failures)} problem(s); fix each in {REGISTER.relative_to(ROOT)}",
              file=sys.stderr)
        return 1
    if not args.list:
        print(f"decoder_gaps: all {total} decoder rejection names are in the register")
    return 0


if __name__ == "__main__":
    sys.exit(main())
