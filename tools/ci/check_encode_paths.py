#!/usr/bin/env python3
"""Every codec crate encodes only through the shared engine (owner, 2026-11-15).

A codec crate's one encode path is its `CodecBackend` implementation, driven
by `nc_encoder::Encoder`. A second encoder beside it (AV2 had a standalone
picture encoder until 2026-11-15) duplicates the engine's decisions, is
untested by the presets and lets tests pass on streams no user ever gets.
This check keeps one from coming back:

- each codec crate implements `CodecBackend` exactly once;
- nothing the crate exports (`pub use` names in `lib.rs`, and the public
  items of every module reachable through `pub mod`) is named like an
  encoder (`encode...`, `...Encoder`, `...encoder...`), except the backend
  itself.

Syntax writers (entropy coders, header and coefficient writers used by the
syntax oracles in `examples/`) are not encoders and stay allowed.

    python3 tools/ci/check_encode_paths.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODECS = ("nc-h264", "nc-hevc", "nc-vvc", "nc-av1", "nc-vp9", "nc-av2")
ENCODER_NAME = re.compile(r"encod", re.IGNORECASE)
PUB_MOD = re.compile(r"^\s*pub mod (\w+);", re.MULTILINE)
PUB_USE = re.compile(r"^\s*pub use ([^;]+);", re.MULTILINE)
PUB_ITEM = re.compile(r"^\s*pub (?:const )?(?:fn|struct|enum|trait|type) (\w+)", re.MULTILINE)
BACKEND_IMPL = re.compile(r"^impl\b[^{\n]*\bCodecBackend for\b", re.MULTILINE)
TEST_MODULE = "#[cfg(test)]\nmod tests"


def module_file(parent: Path, name: str) -> Path | None:
    """The file of `mod name;` declared in `parent` (lib.rs, mod.rs or a.rs)."""
    base = parent.parent if parent.name in ("lib.rs", "mod.rs") else parent.with_suffix("")
    for candidate in (base / f"{name}.rs", base / name / "mod.rs"):
        if candidate.exists():
            return candidate
    return None


def non_test(source: str) -> str:
    """`source` without its unit-test module."""
    at = source.find(TEST_MODULE)
    return source if at < 0 else source[:at]


def exported_names(lib: Path) -> list[tuple[Path, str]]:
    """(file, name) of every public item reachable from the crate root."""
    names: list[tuple[Path, str]] = []
    pending, seen = [lib], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        source = non_test(path.read_text())
        if path != lib:
            names += [(path, n) for n in PUB_ITEM.findall(source)]
        for use in PUB_USE.findall(source):
            names += [(path, n) for n in re.findall(r"\w+", use.split("::")[-1])]
        for module in PUB_MOD.findall(source):
            if (child := module_file(path, module)) is not None:
                pending.append(child)
    return names


def check(crate: str) -> list[str]:
    src = REPO / "crates" / crate / "src"
    problems = []
    impls = sum(len(BACKEND_IMPL.findall(p.read_text())) for p in src.rglob("*.rs"))
    if impls != 1:
        problems.append(f"{crate}: {impls} CodecBackend implementations (exactly 1 allowed)")
    for path, name in exported_names(src / "lib.rs"):
        if ENCODER_NAME.search(name) and not name.endswith("Backend"):
            rel = path.relative_to(REPO)
            problems.append(f"{rel}: public `{name}` looks like an encode path beside the backend")
    return problems


def main() -> int:
    problems = [p for crate in CODECS for p in check(crate)]
    for problem in problems:
        print(problem)
    if problems:
        print("codec crates encode only through their CodecBackend (CLAUDE.md, check_encode_paths.py)")
        return 1
    print(f"encode paths: {len(CODECS)} codec crates, one CodecBackend each, no other encoder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
