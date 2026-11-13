"""Hard bounds on the harness's disk usage.

The harness writes bitstreams and temporary YUVs under the data root. Several
sweeps run concurrently (one per agent, up to 14 workers each, each holding a
few 300-frame YUVs in flight), so the budget allows for that transient peak
while staying far below the disk's free space. Two limits are enforced before every curve, and the sweep stops (rather than
filling the disk) when either is crossed:

  - the work directory may not exceed WORK_BUDGET_BYTES;
  - the filesystem must keep at least MIN_FREE_BYTES free.

The work directory holds ~200k files, and walking it costs seconds of CPU
(mostly stat syscalls): measured 2026-11-11 as ~8 CPU-s per chart point,
79% of the measuring process's time, when every point walked it. Its size is
therefore shared through a small cache file at the work root and re-walked
at most every USAGE_TTL_SECONDS machine-wide (one process walks, under a
lock, while the others read the result). In that time the harness writes at
most a few hundred MB (bitstreams and results), far inside the gap between
the budget and the free-space floor.

Growth is bounded structurally too: only one fingerprint directory is kept
per (encoder, structure, clip) (see Sweep._run_curve), and decoder outputs
and encoder reconstructions are deleted as soon as they have been checked.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from pathlib import Path

GIB = 1 << 30
WORK_BUDGET_BYTES = 100 * GIB
MIN_FREE_BYTES = 200 * GIB
#: How long a measured work-directory size stands (see the module docstring).
USAGE_TTL_SECONDS = 60
USAGE_CACHE = ".disk-usage.json"


class DiskBudgetExceeded(RuntimeError):
    pass


def directory_bytes(root: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except FileNotFoundError:
                pass  # removed by a concurrent worker
    return total


def check_disk(work: Path) -> None:
    """Raise DiskBudgetExceeded if writing more under `work` would break a limit."""
    work.mkdir(parents=True, exist_ok=True)
    # An experiment with its own data root nested inside the main work
    # directory still counts against the main budget.
    work = _outermost_work_dir(work)
    free = shutil.disk_usage(work).free
    if free < MIN_FREE_BYTES:
        raise DiskBudgetExceeded(f"only {free / GIB:.0f} GiB free on {work} "
                                 f"(minimum {MIN_FREE_BYTES / GIB:.0f} GiB)")
    used = shared_work_bytes(work)
    if used > WORK_BUDGET_BYTES:
        raise DiskBudgetExceeded(f"{work} holds {used / GIB:.0f} GiB "
                                 f"(budget {WORK_BUDGET_BYTES / GIB:.0f} GiB); "
                                 f"prune stale results before sweeping")


def shared_work_bytes(work: Path) -> int:
    """`directory_bytes(work)`, re-walked at most every USAGE_TTL_SECONDS by any process."""
    cache = work / USAGE_CACHE
    stored = _fresh_usage(cache)
    if stored is not None:
        return stored
    with (work / (USAGE_CACHE + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one walker; the others then read its result
        stored = _fresh_usage(cache)
        if stored is not None:
            return stored
        used = directory_bytes(work)
        partial = cache.with_name(f"{cache.name}.{os.getpid()}")
        partial.write_text(json.dumps({"bytes": used, "time": time.time()}))
        partial.replace(cache)
        return used


def _fresh_usage(cache: Path) -> int | None:
    try:
        stored = json.loads(cache.read_text())
    except (FileNotFoundError, ValueError):
        return None
    return stored["bytes"] if time.time() - stored["time"] < USAGE_TTL_SECONDS else None


def _outermost_work_dir(work: Path) -> Path:
    outermost = work.resolve()
    for parent in outermost.parents:
        if parent.name == "work":
            outermost = parent
    return outermost
