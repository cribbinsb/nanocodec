"""The build stamp of a worktree without git: its commit and what differs from it.

A box build (nc-remote) has no `.git`, so nc-cli's build.rs cannot ask git
whether the tree is clean: it takes the commit from NC_BUILD_COMMIT and the
local changes from NC_BUILD_DIRTY. Before 2026-11-19 only the commit was
passed, so a build of uncommitted changes was stamped as a clean build of
HEAD, and the measurement caches (keyed on a clean build's commit,
nceval.buildinfo.stream_identity) gave one variant another variant's points.

`diff_identity` names the local changes: empty when the build inputs match
HEAD, else a hash of the diff against HEAD and of every untracked input file
(nc-remote's rsync sends those too). The binary's `--version` then reads
`<commit>-dirty.<hash>`, so two variants on one HEAD never share a stamp,
while two builds of the same variant (gate and release) do.
Standard library only.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

#: The paths whose contents reach the binary (crates embed presets/*.toml).
BUILD_INPUTS = ("crates", "presets", "Cargo.toml", "Cargo.lock", ".cargo",
                "rust-toolchain.toml", "rustfmt.toml")
#: Hex digits of the hash kept in the stamp.
DIFF_DIGITS = 16


def _git(worktree: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(worktree), *args], check=True,
                          capture_output=True).stdout


def diff_identity(worktree: Path) -> str:
    """'' if `worktree`'s build inputs match its HEAD, else DIFF_DIGITS hex digits
    of SHA-256 over the binary diff of tracked inputs against HEAD and each
    untracked (not ignored) input file's path and bytes."""
    worktree = Path(worktree)
    diff = _git(worktree, "diff", "HEAD", "--binary", "--no-ext-diff", "--", *BUILD_INPUTS)
    untracked = sorted(p for p in _git(worktree, "ls-files", "--others", "--exclude-standard",
                                       "-z", "--", *BUILD_INPUTS).split(b"\0") if p)
    if not diff and not untracked:
        return ""
    digest = hashlib.sha256(diff)
    for path in untracked:
        digest.update(b"\0untracked\0" + path + b"\0")
        digest.update((worktree / path.decode()).read_bytes())
    return digest.hexdigest()[:DIFF_DIGITS]
