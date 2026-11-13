"""Cache entries that concurrent agents share: atomic writes and per-entry locks.

Several agents' screens, sweeps and boards share one cache tree, and two of
them often need the same point (an anchor) at the same moment. Fixed
temporary names (`x.pending`, `x.tmp`, a stream named after its cache
entry) let them clobber each other: one requester's encode overwrote or
deleted the other's bitstream mid-gate, which showed up as a spurious
standard-decoder failure or a missing file. The rules here:

  - Every temporary file is private: a unique name in the entry's own
    directory (`sibling_temp`), so a rename into place is atomic and on the
    same filesystem.
  - Every cache write is whole or absent (`write_atomic`): readers never see
    a partial file, and two writers of the same entry cannot interleave.
  - Computing an entry holds an exclusive flock(2) on `<entry>.lock`
    (`entry_lock`). A second requester blocks on it, then finds the entry
    written and reuses it (`get_or_create`) instead of encoding again. The
    kernel releases the lock when its holder exits, even after kill -9, so a
    crashed run never wedges the cache. Lock files are empty and are kept:
    removing one while another process waits on it would let two holders in.

Take an entry lock BEFORE any CPU slots (nceval.slots) and never hold two
entry locks in one thread, so a waiter never sits on resources the holder
needs.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

LOCK_SUFFIX = ".lock"
#: Ends a private temporary that has no extension of its own (never matches `*.json`).
TEMP_SUFFIX = ".tmp"


def lock_path(entry: Path) -> Path:
    """The lock file guarding `entry` (a file or a directory)."""
    return entry.with_name(entry.name + LOCK_SUFFIX)


@contextmanager
def entry_lock(entry: Path) -> Iterator[None]:
    """Hold the exclusive lock on `entry` until the block ends (waits for it)."""
    path = lock_path(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each call opens its own file description, so threads of one process
    # exclude each other exactly as separate processes do.
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@contextmanager
def entry_lock_if_free(entry: Path) -> Iterator[bool]:
    """Try the lock on `entry` without waiting; yields whether it is held."""
    path = lock_path(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


@contextmanager
def sibling_temp(entry: Path, suffix: str = TEMP_SUFFIX) -> Iterator[Path]:
    """A private, not yet existing path beside `entry`; removed when the block ends.

    `suffix` ends the name, so tools that pick a container from the
    extension still see the stream's (`.ivf`, `.h264`). The caller renames
    it into place (`publish`) to keep it, or lets it go.
    """
    entry.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{entry.name}.", suffix=suffix, dir=entry.parent)
    os.close(handle)
    path = Path(name)
    path.unlink()  # tools that refuse to overwrite get a fresh name, not an empty file
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def publish(temporary: Path, entry: Path) -> None:
    """Move a finished private file into place atomically (same directory)."""
    os.replace(temporary, entry)


def write_atomic(entry: Path, text: str) -> None:
    """Write a cache file whole or not at all, safe against concurrent writers."""
    with sibling_temp(entry) as temporary:
        temporary.write_text(text)
        publish(temporary, entry)


def read_json(entry: Path, field: str | None = None):
    """A cache file's JSON (or one field of it); None when it is not written yet."""
    try:
        value = json.loads(entry.read_text())
    except FileNotFoundError:
        return None
    return value[field] if field else value


def get_or_create(entry: Path, read: Callable[[Path], T | None], create: Callable[[], T],
                  dump: Callable[[T], str]) -> T:
    """The cached value at `entry`, computed at most once however many ask at once.

    `read` returns None for a missing or unusable entry (e.g. a stale
    identity). The unlocked first read keeps cache hits lock-free; the second,
    under the lock, sees an entry another requester finished while we waited.
    """
    value = read(entry)
    if value is not None:
        return value
    with entry_lock(entry):
        value = read(entry)
        if value is None:
            value = create()
            write_atomic(entry, dump(value))
        return value


def identity_cached(entry: Path, identity: dict, compute: Callable[[], T]) -> T:
    """A JSON value stored with its `identity`, recomputed when the identity differs.

    The identity is round-tripped through JSON first, so tuples and lists
    compare equal to what a reader loads back.
    """
    identity = json.loads(json.dumps(identity, sort_keys=True))

    def read(path: Path):
        record = read_json(path)
        return record if record is not None and record["identity"] == identity else None
    record = get_or_create(entry, read, lambda: {"identity": identity, "value": compute()},
                           lambda value: json.dumps(value) + "\n")
    return record["value"]
