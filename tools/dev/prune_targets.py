"""Remove stale cargo target directories under ~/nanocodec-data/targets (tools/dev/prune_targets.py).

Run by the coordinator's watchdog when free disk nears the 200 GiB floor
(2026-11-15: 156 finished agents' targets held 133 GB).

Stale = not touched (newest file mtime at the top two levels) for 6 hours and not kept by
the retention manifest (tools/dev/retention.toml: the shared tools' warm targets, which
must stay even when idle for hours). Prints the list with sizes first; removes only with
--yes, one literal path at a time.
"""
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import retention  # noqa: E402

ROOT = os.path.expanduser("~/nanocodec-data/targets")
#: An idle target this old belongs to a finished agent; rebuilding it costs minutes.
STALE_S = 6 * 3600


def newest(path: str) -> float:
    best = os.path.getmtime(path)
    try:
        for e in os.scandir(path):
            best = max(best, e.stat(follow_symlinks=False).st_mtime)
            if e.is_dir(follow_symlinks=False):
                for f in os.scandir(e.path):
                    best = max(best, f.stat(follow_symlinks=False).st_mtime)
    except OSError:
        pass
    return best


def size_gb(path: str) -> float:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total / 1e9


now = time.time()
victims = []
for name in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, name)
    if retention.guard(Path(path)) is not None or not os.path.isdir(path) or os.path.islink(path):
        continue
    if now - newest(path) > STALE_S:
        victims.append(path)
total = 0.0
for p in victims:
    gb = size_gb(p)
    total += gb
    print(f"{gb:6.1f} GB  {p}")
print(f"{len(victims)} dirs, {total:.0f} GB")
if "--yes" in sys.argv:
    for p in victims:
        shutil.rmtree(p)
        print("removed", p, flush=True)
