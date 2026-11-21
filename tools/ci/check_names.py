#!/usr/bin/env python3
"""Fail when code would put nanocodec files outside our places or under another name.

    python3 tools/ci/check_names.py        (tools/ci/check.sh runs it)

Owner rule (2026-11-21): everything of ours lives under ~/nanocodec-data, the repo,
or a name starting `nanocodec`; never loose in $HOME, / or /tmp (682 leaked
nc-metrics-/nc-point-/nc-dec- directories in /tmp and stray ~/lens2..8, ~/rdcdf were
cleaned by hand that day). Checked over the tracked Python, shell and Rust files:

  - a Python temporary file or directory (tempfile.mkdtemp, TemporaryDirectory,
    NamedTemporaryFile, mkstemp) names a `dir=` or a `prefix=` starting `nanocodec`
    (nceval.tempdirs does both: use it);
  - a shell `mktemp` names a `nanocodec-` template, or `-p` a directory;
  - a Rust `temp_dir().join(...)` names a path starting `nanocodec`;
  - no argparse default is a relative output path (it lands in whatever directory
    the tool was started from, e.g. $HOME), no literal `/tmp/...` path, and no
    `Path.home() / "x"` outside `nanocodec*`.

ALLOWED lists the exceptions, each with its reason.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMP_CALLS = {"mkdtemp", "TemporaryDirectory", "NamedTemporaryFile", "mkstemp"}
PREFIX = "nanocodec"
#: (file, fragment of the line): why it may stay.
ALLOWED = {
    ("eval/nceval/gpuserve.py", 'STATE_DIR = Path("/tmp")'):
        "the VMAFx scoring servers (vmaf_serve.cpp, outside this repo) publish their "
        "state there; the pool is shared and live, so its path moves with the server",
    ("tools/dev/nc_remote_job.py", 'BOX = Path.home() / "ncbox"'):
        "runs on the build box (ours, remote): its job home there",
}
#: Test suites whose conftest puts every temporary path under nceval.tempdirs.root().
REDIRECTED_TESTS = ("eval/tests/",)
RELATIVE_OUTPUT = re.compile(
    r"""default=['"](?![/~$])[^'"\s]*(/|\.(json|jsonl|csv|md|html|png|svg|txt|log|ivf|yuv))['"]""")
TMP_LITERAL = re.compile(r"""['"]/tmp/(?!nanocodec)""")
#: A dot directory is a tool's own (~/.cargo/bin: the toolchain), read, never written.
HOME_JOIN = re.compile(r"""Path\.home\(\)\s*/\s*['"](?!nanocodec|\.)""")
RUST_TEMP = re.compile(r"""temp_dir\(\)\s*\.join\(\s*(?:format!\(\s*)?"(?!nanocodec)""")
SHELL_MKTEMP = re.compile(r"""\bmktemp\b(?![^\n]*(nanocodec-|\s-p\s))""")


#: Directories never walked when there is no git repository (the build box's synced
#: copy: nc-remote's rsync already leaves out what .gitignore lists, /.git and /target).
SKIPPED_DIRS = {".git", "target", "__pycache__", "node_modules"}


def tracked(suffix: str, root: Path = REPO) -> list[Path]:
    """The repository's files ending in `suffix`: git's list where there is a
    repository, else a walk of the tree (the build box's copy has no .git)."""
    if (root / ".git").exists():
        out = subprocess.run(["git", "-C", str(root), "ls-files", f"*{suffix}"],
                             capture_output=True, text=True, check=True).stdout.split()
        return [root / name for name in out]
    found = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIPPED_DIRS
                            and not os.path.islink(os.path.join(directory, d)))
        found += [Path(directory, f) for f in sorted(files) if f.endswith(suffix)]
    return found


def python_temp_problems(path: Path, text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    problems = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in TEMP_CALLS and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tempfile"):
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        prefix = keywords.get("prefix")
        named = isinstance(prefix, ast.Constant) and str(prefix.value).startswith(PREFIX) or \
            isinstance(prefix, ast.JoinedStr) and prefix.values and \
            isinstance(prefix.values[0], ast.Constant) and \
            str(prefix.values[0].value).startswith(PREFIX) or \
            isinstance(prefix, ast.Call) and "prefix" in ast.unparse(prefix.func)
        if "dir" not in keywords and not named:
            problems.append(f"{path.relative_to(REPO)}:{node.lineno}: tempfile.{node.func.attr} "
                            f"without dir= or a nanocodec prefix (use nceval.tempdirs)")
    return problems


def line_problems(path: Path, text: str, patterns: list[tuple[re.Pattern, str]]) -> list[str]:
    rel = str(path.relative_to(REPO))
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("#", "//")):
            continue
        if any(rel == f and fragment in line for f, fragment in ALLOWED):
            continue
        for pattern, why in patterns:
            if pattern.search(line):
                problems.append(f"{rel}:{number}: {why}: {line.strip()[:100]}")
    return problems


def main() -> int:
    problems = []
    for path in tracked(".py"):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(errors="replace")
        if not str(path.relative_to(REPO)).startswith(REDIRECTED_TESTS):
            problems += python_temp_problems(path, text)
        problems += line_problems(path, text, [
            (RELATIVE_OUTPUT, "a relative default path (lands in the start directory)"),
            (TMP_LITERAL, "a /tmp path not named nanocodec"),
            (HOME_JOIN, "a path in $HOME not named nanocodec")])
    for path in tracked(".sh"):
        problems += line_problems(path, path.read_text(errors="replace"),
                                  [(SHELL_MKTEMP, "mktemp without a nanocodec- template")])
    for path in tracked(".rs"):
        problems += line_problems(path, path.read_text(errors="replace"),
                                  [(RUST_TEMP, "a temp path not named nanocodec")])
    for problem in problems:
        print(problem)
    if problems:
        print(f"check_names: {len(problems)} problem(s): everything of ours lives under "
              f"~/nanocodec-data, the repo, or a name starting `nanocodec` (owner, 2026-11-21)")
        return 1
    print("check_names: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
