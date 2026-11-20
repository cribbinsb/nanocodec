"""Which commit a nanocodec binary was built from, and refusing a stale one.

Agents kept measuring old builds (a VP9 final check reused one, 2026-11-11).
`nanocodec --version` names the commit it was built from (nc-cli build.rs;
"-dirty": crates/ or presets/ had local changes; "-dirty.HASH": a build box's
copy whose local changes hash to HASH, tools/dev/build_stamp.py). Every screen, level run,
tune study and chart run prints it at start and records it in its results
(`build_info`); `--expect-commit SHA` or NC_EXPECT_COMMIT makes the run fail
at once when the binary is anything else, and a binary whose commit is
behind this worktree's HEAD in crates/ or presets/ gets a loud warning.
Leaf module: imports nothing of nceval.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
import sys
from pathlib import Path

EXPECT_ENV = "NC_EXPECT_COMMIT"
REPO = Path(__file__).resolve().parents[2]
WATCHED = ("crates", "presets")
DIRTY_SUFFIX = "-dirty"
UNKNOWN = "unknown"


@functools.lru_cache(maxsize=None)
def build_info(binary: Path, mtime_ns: int | None = None) -> dict:
    """{"binary", "commit", "dirty", "diff", "version"}; commit None if the binary does not say.

    `mtime_ns` keys the cache, so a rebuilt binary is asked again."""
    try:
        line = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        line = ""
    return {"binary": str(binary), **parse_version(line)}


def parse_version(line: str) -> dict:
    """{"commit", "dirty", "diff", "version"} of a `nanocodec --version` line.

    "diff" is the hash of a dirty build's local changes when its stamp names
    them (`COMMIT-dirty.HASH`), else None."""
    line = line.strip()
    word = line.split()[-1] if line.startswith("nanocodec ") else ""
    commit, dirty, diff = word.partition(DIRTY_SUFFIX)
    commit = commit or None
    return {"commit": None if commit == UNKNOWN else commit, "dirty": bool(dirty),
            "diff": diff.removeprefix(".") or None, "version": line or None}


def stamp_problem(info: dict, commit: str) -> str | None:
    """Why a build (build_info or parse_version) is not exactly `commit` (None: it is).

    `commit` may be abbreviated; the stamp is the full hash."""
    stamped = info["commit"]
    if not stamped or not stamped.startswith(commit):
        return f"its --version says {info['version']!r}, not commit {commit}"
    if info["dirty"]:
        return f"it was built from a dirty tree ({info['version']!r})"
    return None


def verify_built(binary: Path, commit: str) -> dict:
    """Fail loudly unless a fresh build of `commit` really is that commit.

    Builds from shared target dirs have come out as the previous commit's
    binary (cargo judged changed sources up to date), so every build the
    harness makes is checked before it is used."""
    info = info_of(binary)
    problem = stamp_problem(info, commit)
    if problem:
        raise SystemExit(f"build of {commit} at {binary} is wrong: {problem}; "
                         f"refusing to use it")
    return info


#: Prefix of a stream identity taken from a clean build's commit.
COMMIT_IDENTITY_PREFIX = "nanocodec@"


def stream_identity(binary: Path) -> str:
    """What a nanocodec binary's streams are keyed on in the measurement caches.

    A clean build of a commit codes the same streams whatever built it: the
    release, PGO, gate (no LTO) and per-agent target-dir builds differ in bytes
    but not in output (docs/PROCEDURES.md "Build and check"). Keying caches on
    the binary's bytes made every agent re-encode the same integration head's
    anchor curves with its own build. So a clean build is keyed on its commit,
    a dirty one whose stamp names its local changes (a build box's,
    tools/dev/build_stamp.py) on its commit and that diff hash; any other dirty
    or unstamped one keeps the SHA-256 of its bytes."""
    info = info_of(binary)
    if info["commit"] and not info["dirty"]:
        return COMMIT_IDENTITY_PREFIX + info["commit"]
    if info["commit"] and info["diff"]:
        return f"{COMMIT_IDENTITY_PREFIX}{info['commit']}{DIRTY_SUFFIX}.{info['diff']}"
    digest = hashlib.sha256()
    with Path(binary).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def info_of(binary: Path) -> dict:
    binary = Path(binary).resolve()
    try:
        stamp = binary.stat().st_mtime_ns
    except OSError:
        stamp = None
    return build_info(binary, stamp)


def check(binary: Path, role: str = "binary", expect: str | None = None,
          measured: bool = True) -> dict:
    """Print the binary's commit and return its build_info.

    For the build under test (`measured`): fail on a mismatch with `expect`
    (or NC_EXPECT_COMMIT), and warn loudly if it is behind HEAD in crates/
    or presets/. An anchor build is old on purpose: only printed."""
    info = info_of(binary)
    commit = info["commit"]
    shown = (commit[:12] if commit else "UNKNOWN (built before --version existed?)") + \
        (DIRTY_SUFFIX if info["dirty"] else "")
    print(f"{role}: {info['binary']} built from {shown}", flush=True)
    if not measured:
        return info
    expect = expect or os.environ.get(EXPECT_ENV)
    if expect:
        wanted = _resolve(expect) or expect
        if not commit or not (commit.startswith(wanted) or wanted.startswith(commit)):
            raise SystemExit(f"{role} {info['binary']} was built from {shown}, "
                             f"expected {expect}: refusing to measure a different build")
        if info["dirty"]:
            raise SystemExit(f"{role} {info['binary']} was built from a dirty tree "
                             f"({shown}), expected exactly {expect}")
    warning = staleness(commit)
    if warning:
        print(f"\n!!! WARNING: {role} {info['binary']}: {warning}\n", file=sys.stderr,
              flush=True)
    return info


def staleness(commit: str | None) -> str | None:
    """Why `commit` is not this worktree's code in crates/ or presets/ (None: it is)."""
    head = _git("rev-parse", "HEAD")
    if head is None or commit is None:
        return None if head is None else "its commit is unknown; it may be stale"
    if commit == head:
        return None
    if _git("cat-file", "-e", f"{commit}^{{commit}}") is None:
        return f"built from {commit[:12]}, which this repository does not have"
    changed = _git("diff", "--name-only", commit, head, "--", *WATCHED)
    if changed:
        files = changed.splitlines()
        return (f"built from {commit[:12]}, but HEAD {head[:12]} changes {len(files)} "
                f"file(s) under crates/ or presets/ (e.g. {files[0]}): it is STALE")
    return None


def _resolve(revision: str) -> str | None:
    return _git("rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}")


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
