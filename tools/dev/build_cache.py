#!/usr/bin/env python3
"""The landed-commit binary cache: a ready CLI binary per landed commit, so measuring
an existing commit never needs a cold compile (owner, 2026-11-18: agents spent 10+
minutes on fresh-target release builds just to measure a landed commit).

    build_cache.py store COMMIT GATE_DIR [--since EPOCH]   (land.sh --verify)
    build_cache.py for COMMIT                               (tools/dev/binary_for.sh)
    build_cache.py prune

The cache is the chart builds directory, in chart_refresh.py's layout:
~/nanocodec-data/charts/builds/<8-hex commit>/ holds `nanocodec` (the gate build:
release code generation without LTO, the same streams), `presets/` and `COMMIT`,
written last (complete). `store` copies the binaries the post-landing verifier has
just built (no extra compile): the CLI only when its `--version` stamp is exactly
COMMIT (clean), and `nc-checkasm` when it was built after EPOCH (it has no stamp).
A chart refresh of that commit then finds the build in place and only adds its
timed release binary.

`for` prints the directory of COMMIT's build (use it as NC_NANOCODEC_BIN): the
cached one; else a cached build whose sources (crates, presets, manifests) equal
COMMIT's (the same binary; its stamp names that build's commit); else it builds
COMMIT on the remote build box (nc-remote, the `binary-for` worktree) and stores
the result. Never a local compile.

`prune` (also after every store) keeps the newest KEEP_NEWEST builds this cache made
(marked CACHE_MARKER) and any build named in the integration branch's results
records (docs/results, web/results/data) or holding files the cache did not write
(a chart refresh adopted it). Builds made by chart_refresh or the release are never
touched here; chart_refresh removes its own.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "eval"))
from chart_refresh import (  # noqa: E402
    BUILD_INPUTS, BUILDS_ROOT, COMPLETE_MARKER, OLD_PRESETS_PATH, PRESETS_PATH, SCORING_BINARY,
    place)
from nceval import retention, tempdirs  # noqa: E402
from nceval.buildinfo import info_of, stamp_problem  # noqa: E402

REPO = HERE.parents[1]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
CACHE = Path.home() / BUILDS_ROOT
#: The commit naming of a build directory (chart_refresh: `rev-parse --short=8`).
SHORT = 8
CHECKASM_BINARY = "nc-checkasm"
#: Written by `store` beside COMMIT: this cache made the directory (prune may remove it).
CACHE_MARKER = "CACHED"
OWN_FILES = {SCORING_BINARY, CHECKASM_BINARY, "presets", COMPLETE_MARKER, CACHE_MARKER}
#: Cache-made builds kept regardless of use (~25 MB CLI + ~16 MB checkasm each).
KEEP_NEWEST = 20
INTEGRATION = os.environ.get("NC_LAND_TARGET", "variance-integration")
#: Results records whose naming of a build keeps it.
RECORDS = ("docs/results", "web/results/data")
#: Same-sources donors searched among the integration branch's last commits.
DONOR_DEPTH = 200
#: The worktree a missing build is checked out in and sent to the build box from.
REMOTE_TREE = DATA / "worktrees" / "binary-for"
NC_REMOTE = DATA / "bin" / "nc-remote"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def short(commit: str) -> str:
    return git("rev-parse", f"--short={SHORT}", f"{commit}^{{commit}}")


def complete(directory: Path) -> bool:
    return (directory / COMPLETE_MARKER).is_file() and (directory / SCORING_BINARY).is_file()


@contextmanager
def cache_lock():
    """One writer at a time (the two verifier parts may store the same head)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / ".cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def install_presets(commit: str, directory: Path) -> None:
    """COMMIT's presets into `directory`/presets (as chart_refresh.install_presets)."""
    moved = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{commit}:{PRESETS_PATH}"],
                           capture_output=True).returncode == 0
    source = PRESETS_PATH if moved else OLD_PRESETS_PATH
    archive = subprocess.run(["git", "-C", str(REPO), "archive", commit, source], check=True,
                             capture_output=True).stdout
    staging = Path(tempfile.mkdtemp(dir=directory, prefix=".presets-"))
    subprocess.run(["tar", "-x", f"--strip-components={source.count('/') + 1}", "-C",
                    str(staging)], input=archive, check=True)
    shutil.rmtree(directory / "presets", ignore_errors=True)
    staging.rename(directory / "presets")


def store(commit: str, gate_dir: Path, since: float | None) -> Path | None:
    """Copy COMMIT's freshly built binaries from `gate_dir` into its cache directory."""
    name = short(commit)
    directory = CACHE / name
    cli = gate_dir / SCORING_BINARY
    problem = stamp_problem(info_of(cli), name) if cli.is_file() else f"no {cli}"
    checkasm = gate_dir / CHECKASM_BINARY
    fresh_checkasm = (since is not None and checkasm.is_file()
                      and checkasm.stat().st_mtime >= since)
    with cache_lock():
        if problem is None and not complete(directory):
            directory.mkdir(parents=True, exist_ok=True)
            retention.ensure_entry_marker(CACHE)
            (directory / CACHE_MARKER).touch()
            place(cli, directory / SCORING_BINARY)
            install_presets(commit, directory)
            (directory / COMPLETE_MARKER).write_text(name + "\n")
            print(f"build cache: {directory} (from {cli})")
        elif problem is not None:
            print(f"build cache: CLI of {name} not stored: {cli}: {problem}")
        if fresh_checkasm and directory.is_dir() and not (directory / CHECKASM_BINARY).exists():
            place(checkasm, directory / CHECKASM_BINARY)
            print(f"build cache: {directory / CHECKASM_BINARY}")
        prune_locked()
    return directory if complete(directory) else None


def referenced(names: list[str]) -> set[str]:
    """The names any results record on the integration branch mentions."""
    found = set()
    for name in names:
        hit = subprocess.run(["git", "-C", str(REPO), "grep", "-q", "-F", name, INTEGRATION,
                              "--", *RECORDS], capture_output=True)
        if hit.returncode == 0:
            found.add(name)
    return found


def prune_locked() -> None:
    made = [d for d in CACHE.iterdir() if (d / CACHE_MARKER).is_file()]
    made.sort(key=lambda d: (d / CACHE_MARKER).stat().st_mtime, reverse=True)
    candidates = [d for d in made[KEEP_NEWEST:]
                  if {p.name for p in d.iterdir()} <= OWN_FILES]
    keep = referenced([d.name for d in candidates])
    for directory in candidates:
        if directory.name not in keep and retention.guard(directory, "build_cache") is None:
            print(f"build cache: pruned {directory}")
            shutil.rmtree(directory)


def prune() -> None:
    with cache_lock():
        prune_locked()


def same_sources_donor(commit: str) -> Path | None:
    """A complete cached build of a commit whose binary COMMIT would rebuild unchanged."""
    complete_dirs = {d.name: d for d in CACHE.iterdir() if complete(d)} if CACHE.is_dir() else {}
    if not complete_dirs:
        return None
    for candidate in git("rev-list", "--first-parent", f"-n{DONOR_DEPTH}", INTEGRATION).split():
        directory = complete_dirs.get(candidate[:SHORT])
        if directory is not None and subprocess.run(
                ["git", "-C", str(REPO), "diff", "--quiet", candidate, commit, "--",
                 *BUILD_INPUTS]).returncode == 0:
            return directory
    return None


def remote_build(commit: str) -> Path:
    """COMMIT's gate CLI built on the build box, then stored in the cache."""
    full = git("rev-parse", f"{commit}^{{commit}}")
    lock_path = DATA / "run" / "binary-for.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one checkout of the shared binary-for tree at a time
        if not (REMOTE_TREE / ".git").exists():
            git("worktree", "add", "--detach", str(REMOTE_TREE), full)
        subprocess.run(["git", "-C", str(REMOTE_TREE), "checkout", "-q", "--force", "--detach",
                        full], check=True)
        with tempdirs.temp_dir("binary-for") as fetched:
            print(f"build cache: {full[:SHORT]} not cached: building it on the build box",
                  file=sys.stderr)
            subprocess.run([str(NC_REMOTE), "run", "--worktree", str(REMOTE_TREE), "--name",
                            "binary-for", "--fetch", f"target/gate/{SCORING_BINARY}",
                            "--fetch-to", fetched, "--", "cargo", "build", "--profile", "gate",
                            "--quiet", "-p", "nanocodec-cli"],
                           check=True, stdout=sys.stderr)
            directory = store(full, Path(fetched) / "target" / "gate", None)
    if directory is None:
        raise SystemExit(f"build cache: the build box's build of {full[:SHORT]} was not stored")
    return directory


def binary_for(commit: str) -> Path:
    directory = CACHE / short(commit)
    if complete(directory):
        return directory
    donor = same_sources_donor(commit)
    if donor is not None:
        print(f"build cache: {short(commit)} builds the same binary as {donor.name}: using it",
              file=sys.stderr)
        return donor
    return remote_build(commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    stored = sub.add_parser("store", help="copy a just-built COMMIT's binaries into the cache")
    stored.add_argument("commit")
    stored.add_argument("gate_dir", type=Path, help="the target's gate profile directory")
    stored.add_argument("--since", type=float,
                        help="epoch: copy nc-checkasm only if built after it")
    found = sub.add_parser("for", help="print COMMIT's build directory (build on the box if missing)")
    found.add_argument("commit")
    sub.add_parser("prune", help="drop old cache-made builds")
    args = parser.parse_args()
    if args.command == "store":
        store(args.commit, args.gate_dir, args.since)
    elif args.command == "for":
        print(binary_for(args.commit))
    else:
        prune()


if __name__ == "__main__":
    main()
