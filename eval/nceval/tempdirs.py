"""Temporary files and directories: all under one place, named for nanocodec, swept.

Owner rule (2026-11-21): everything of ours is named starting `nanocodec` or lives
under ~/nanocodec-data or the repo, never loose in $HOME, / or /tmp. Every temporary
directory or file the eval and tools make goes under ROOT, `$TMPDIR/nanocodec-<user>/`
(one visible, removable place), named `nanocodec-<what>-<pid>-<random>`:

    with tempdirs.temp_dir("point") as path: ...      # tempfile.TemporaryDirectory
    path = tempdirs.mkdtemp("speed")                    # the caller removes it
    with tempdirs.named_file("index") as handle: ...  # tempfile.NamedTemporaryFile

Context managers remove them on a normal exit and on an exception; a process that is
killed (SIGKILL, or SIGTERM without a handler) cannot, so the first use in every
process sweeps ROOT: an entry older than SWEEP_HOURS whose owning pid (in its name)
is gone is removed (2026-11-21: 682 leaked nc-metrics-/nc-point-/nc-dec- directories
in /tmp, all over 3 h old).
"""

from __future__ import annotations

import functools
import getpass
import os
import shutil
import tempfile
import time
from pathlib import Path

PREFIX = "nanocodec-"
SWEEP_HOURS = 3


@functools.cache
def root() -> Path:
    """`$TMPDIR/nanocodec-<user>/`, made and swept on the first use in a process."""
    path = Path(tempfile.gettempdir()) / f"{PREFIX}{getpass.getuser()}"
    path.mkdir(mode=0o700, exist_ok=True)
    sweep(path)
    return path


def prefix(what: str) -> str:
    return f"{PREFIX}{what}-{os.getpid()}-"


def temp_dir(what: str) -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix=prefix(what), dir=root())


def mkdtemp(what: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix(what), dir=root()))


def named_file(what: str, **kwargs):
    return tempfile.NamedTemporaryFile(prefix=prefix(what), dir=root(), **kwargs)


def mkstemp(what: str, suffix: str = "") -> tuple[int, str]:
    return tempfile.mkstemp(prefix=prefix(what), suffix=suffix, dir=root())


def owner(name: str) -> int | None:
    """The pid in a `nanocodec-<what>-<pid>-<random>` name (None: not one of ours)."""
    parts = name.rsplit("-", 2)
    return int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None


def sweep(path: Path, hours: float = SWEEP_HOURS) -> int:
    """Remove entries of `path` older than `hours` whose owning process is gone;
    returns how many."""
    cutoff, removed = time.time() - 3600 * hours, 0
    for entry in path.iterdir():
        pid = owner(entry.name)
        try:
            old = entry.lstat().st_mtime < cutoff
        except OSError:
            continue
        if pid is None or not old or Path(f"/proc/{pid}").exists():
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
        removed += 1
    return removed
