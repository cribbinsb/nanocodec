#!/usr/bin/env python3
"""Decide which landing checks a candidate needs from the paths it changes.

    python3 tools/dev/land_scope.py <integration-ref>     (in the candidate worktree)

land.sh calls this after rebasing, so `git diff <integration-ref> HEAD` is
exactly what landing would change. Each changed path falls in one class
(first matching rule wins; docs/PROCEDURES.md "Landing" has the table):

    class      paths                                        checks
    gate       Cargo.toml, Cargo.lock, rustfmt.toml,        full check.sh --quick on
               clippy.toml, .cargo/, rust-toolchain*,       the whole workspace + eval
               tools/ci/check.sh, check_layering.py,        tests + the cheap checks
               check_test_modules.py, browser_smoke.sh,
               tools/web/,
               tools/dev/land.sh, land_scope.py,
               land_guard.py, cargo-capped.sh, check_docs.py
    crate      crates/<name>/**                             fmt, clippy + tests of
                                                            <name> and every crate
                                                            that depends on it
    included   any file a crate embeds with include_str!/   as crate, for the
               include_bytes! (presets/*.toml today)        embedding crates; plus
                                                            eval tests for presets/
    presets    presets/** not embedded (candidates/)        TOML parse + eval tests
    eval       eval/**                                      eval tests
    web        web/**                                       browser module smoke tests
    fuzz       fuzz/**                                      rustfmt --check of the files
    python     other *.py (tools/**)                        byte-compile; eval tests if
                                                            an eval test names the file
    shell      other *.sh                                   bash -n
    docs       *.md, *.svg, *.json, *.csv, *.txt, *.png     relative links/images
               outside crates/, LICENSE, .gitignore,        resolve; SVG/JSON parse
               .gitattributes
    unknown    anything else                                full gate (the safe default)

Independently of the class, the landing conformance check
(tools/ci/landing_conformance.py: every shipped preset's streams against
the standard decoders and ours) runs when a change can alter a stream or
the check itself: the candidate reaches nc-cli (any encoder, codec, DSP or
embedded-preset change), changes a CONFORMANCE_FILES file, or needs the
full gate.

It prints that table for the candidate's files on stderr and the plan on
stdout, one `key value...` line per key, which land.sh reads:

    rust   all | none | <crate>...   clippy/test selection (dependents included)
    eval   0 | 1                     run the eval pytest suite
    web    0 | 1                     run the browser module smoke tests
    page   0 | 1                     also run the page tests in headless Chromium
                                     (the page itself changed: web/** or
                                     crates/nc-web/**; a codec or engine change
                                     gets the module's node tests only, and the
                                     page tests at the next demo publish)
    conformance 0 | 1                run the landing conformance check
    docs   <file>...                 files for tools/dev/check_docs.py
    python <file>...                 files to byte-compile
    shell  <file>...                 files for bash -n
    toml   <file>...                 files to parse as TOML
    fuzz   <file>...                 files for rustfmt --check
"""

import fnmatch
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Files that define the gate or the build of every crate: any change re-runs
# everything, so a change to the checks is checked by the full checks.
GATE_PATTERNS = (
    "Cargo.toml", "Cargo.lock", "rustfmt.toml", "clippy.toml", ".rustfmt.toml",
    ".cargo/*", "rust-toolchain", "rust-toolchain.toml",
    "tools/ci/check.sh", "tools/ci/check_layering.py", "tools/ci/check_test_modules.py",
    "tools/ci/browser_smoke.sh",
    "tools/web/*",
    "tools/dev/land.sh", "tools/dev/land_scope.py", "tools/dev/land_guard.py",
    "tools/dev/cargo-capped.sh",
    "tools/dev/check_docs.py",
)
# The landing conformance check and the harness code it decodes with.
CONFORMANCE_FILES = (
    "tools/ci/landing_conformance.py", "eval/nceval/conformance.py",
    "eval/nceval/gatepolicy.py", "eval/nceval/point.py", "eval/nceval/framepipe.py",
    "eval/nceval/framestats.py", "eval/nceval/yuv.py", "eval/nceval/jm_split.py",
    "eval/nceval/dav2d_crop.py",
)
# The crate every stream is encoded by: a change it depends on can alter streams.
ENCODER_CRATE = "nc-cli"
#: The browser module's own crate: a change here is a change to the page's API.
PAGE_CRATE_DIR = "crates/nc-web/"

DOC_SUFFIXES = {".md", ".svg", ".json", ".csv", ".txt", ".png", ".jpg", ".webp"}
DOC_NAMES = {"LICENSE", ".gitignore", ".gitattributes"}
INCLUDE_RE = re.compile(r'include_(?:str|bytes)!\(\s*"([^"]+)"\s*\)')


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def workspace_graph(root):
    """Map each workspace crate to its directory and its workspace dependents."""
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    meta = json.loads(subprocess.run(
        [cargo, "metadata", "--no-deps", "--format-version", "1", "--offline"],
        cwd=root, check=True, capture_output=True, text=True).stdout)
    dirs, dependents = {}, {}
    for pkg in meta["packages"]:
        dirs[pkg["name"]] = Path(pkg["manifest_path"]).parent.relative_to(root).as_posix()
        dependents.setdefault(pkg["name"], set())
    for pkg in meta["packages"]:
        for dep in pkg["dependencies"]:  # normal, dev and build dependencies alike
            if dep["name"] in dependents:
                dependents[dep["name"]].add(pkg["name"])
    return dirs, dependents


def embedded_files(root, dirs):
    """Repository files outside a crate that the crate's sources embed."""
    embedded = {}
    for crate, crate_dir in dirs.items():
        for source in (root / crate_dir).rglob("*.rs"):
            for target in INCLUDE_RE.findall(source.read_text(errors="replace")):
                path = (source.parent / target).resolve()
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if not rel.startswith(crate_dir + "/"):
                    embedded.setdefault(rel, set()).add(crate)
    return embedded


def closure(crates, dependents):
    """The crates plus every crate that depends on them, transitively."""
    todo, seen = list(crates), set(crates)
    while todo:
        for dependent in dependents[todo.pop()]:
            if dependent not in seen:
                seen.add(dependent)
                todo.append(dependent)
    return seen


def named_by_eval_tests(root, name):
    return any(name in test.read_text(errors="replace")
               for test in (root / "eval" / "tests").glob("test_*.py"))


def classify(path, root, dirs, embedded):
    """(class, crates) for one changed path; crates are the directly affected ones."""
    if any(fnmatch.fnmatchcase(path, pattern) for pattern in GATE_PATTERNS):
        return "gate", set()
    for crate, crate_dir in dirs.items():
        if path.startswith(crate_dir + "/"):
            return "crate", {crate}
    if path in embedded:
        return "included", embedded[path]
    suffix = Path(path).suffix
    top = path.split("/", 1)[0]
    if top == "presets":
        return "presets", set()
    if top == "eval":
        return "eval", set()
    if top == "web":
        return "web", set()
    if top == "fuzz":
        return "fuzz", set()
    if suffix == ".py":
        return "python", set()
    if suffix == ".sh":
        return "shell", set()
    if suffix in DOC_SUFFIXES or path in DOC_NAMES:
        return "docs", set()
    return "unknown", set()


def changes_page(kind, path):
    """Whether a changed path is the page itself (web/** or the browser module's
    crate), so the page tests in headless Chromium run at landing."""
    return kind == "web" or path.startswith(PAGE_CRATE_DIR)


def needs_conformance(changed, affected, full):
    """Whether the landing conformance check runs (module docstring)."""
    return full or ENCODER_CRATE in affected or any(p in CONFORMANCE_FILES for p in changed)


def plan(root, base):
    changed = [line for line in git("diff", "--name-only", "--no-renames", base, "HEAD").splitlines() if line]
    present = {p for p in changed if (root / p).exists()}
    dirs, dependents = workspace_graph(root)
    embedded = embedded_files(root, dirs)

    full, crates, eval_tests, web, page = False, set(), False, False, False
    lists = {"docs": [], "python": [], "shell": [], "toml": [], "fuzz": []}
    rows = []
    for path in changed:
        kind, direct = classify(path, root, dirs, embedded)
        rows.append((kind, path))
        crates |= direct
        if kind in ("gate", "unknown"):
            full = True
        elif kind in ("eval", "presets") or (kind == "included" and path.startswith("presets/")):
            eval_tests = True
        elif kind == "web":
            web = True
        elif kind == "python" and named_by_eval_tests(root, Path(path).name):
            eval_tests = True
        page = page or changes_page(kind, path)
        if path not in present:
            continue  # a deleted file has nothing left to parse; its users are checked above
        if kind == "docs":
            lists["docs"].append(path)
        elif kind == "python":
            lists["python"].append(path)
        elif kind == "shell":
            lists["shell"].append(path)
        elif kind == "fuzz" and path.endswith(".rs"):
            lists["fuzz"].append(path)
        if path.endswith(".toml") and kind in ("presets", "included"):
            lists["toml"].append(path)

    affected = closure(crates, dependents)
    if full or affected == set(dirs):
        rust = ["all"]
        eval_tests = eval_tests or full
    else:
        rust = sorted(affected) or ["none"]
    # The browser module embeds these crates: a change to any of them can break it.
    web = web or full or "nc-web" in affected

    for kind, path in rows:
        print(f"  {kind:<9} {path}", file=sys.stderr)
    conformance = needs_conformance(changed, affected, full)
    out = [("rust", rust), ("eval", [str(int(eval_tests))]), ("web", [str(int(web))]),
           ("page", [str(int(page or full))]), ("conformance", [str(int(conformance))])]
    out += [(key, files) for key, files in lists.items()]
    return "\n".join(" ".join([key, *values]) for key, values in out)


def rank(root, paths):
    """The train position of a candidate that changes `paths` (land.sh
    ranked_tickets): 0 docs only, 2 changes the page, 1 anything else."""
    dirs, _ = workspace_graph(root)
    embedded = embedded_files(root, dirs)
    kinds = [classify(path, root, dirs, embedded)[0] for path in paths]
    if paths and all(kind == "docs" for kind in kinds):
        return 0
    return 2 if any(changes_page(kind, path) for kind, path in zip(kinds, paths)) else 1


def main():
    if sys.argv[1:] == ["--rank"]:
        root = Path(git("rev-parse", "--show-toplevel").strip())
        print(rank(root, [line for line in sys.stdin.read().splitlines() if line]))
        return
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(git("rev-parse", "--show-toplevel").strip())
    print(plan(root, sys.argv[1]))


if __name__ == "__main__":
    main()
