"""Dataset versions: immutable sets of shards with a manifest of content hashes.

A version lives in `~/nanocodec-data/nn-data/vN/` (tools/dev/retention.toml keeps it; not
under work/, whose budget the measurement caches use).
`release` moves a run's kept shards into a new version, moving each shard's source
patches into the version's patch store (one file per clip, start frame and superblock
size, shared by every rung and rate; nceval.nndata.shard.store_patches). It refuses an
existing version
(a released version is never edited: changes make vN+1), re-validates every shard, and
writes `manifest.json` (each shard's path and the sha256 of each of its files, plus the
version's own hash over them), so a trained model names exactly the data it saw.
`verify` re-hashes a version against its manifest.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import shutil
from pathlib import Path

from ..retention import ensure_entry_marker
from .shard import LABELS_FILE, PATCH_STORE, store_patches, unstore_patches, load as load_shard
from .validate import validate

ROOT = Path.home() / "nanocodec-data/nn-data"
MANIFEST = "manifest.json"
SUMMARY = "summary.json"
VERSION_NAME = re.compile(r"^v\d+(\.\d+)?$")


class VersionError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shard_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob(LABELS_FILE))


def shard_hashes(directory: Path) -> dict[str, str]:
    return {f.name: _sha256(f) for f in sorted(directory.iterdir()) if f.is_file()}


def manifest_of(root: Path) -> dict:
    shards = {str(d.relative_to(root)): shard_hashes(d) for d in shard_dirs(root)}
    patches = {str(f.relative_to(root)): _sha256(f)
               for f in sorted(root.rglob(f"{PATCH_STORE}/*")) if f.is_file()}
    digest = hashlib.sha256(json.dumps([shards, patches], sort_keys=True).encode()).hexdigest()
    return {"version": root.name, "shards": shards, "patches": patches, "sha256": digest}


def _release_clip(moves: list[tuple[Path, Path]], store: Path) -> str:
    """Validate and move one clip's shards and store their patches; a problem as text."""
    for source, target in moves:
        problems = validate(load_shard(source))
        if problems:
            return f"{source}: {problems}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)
        store_patches(target, store)
    return ""


def release(runs: list[Path], name: str, root: Path = ROOT, part: bool = False,
            workers: int = 8) -> Path:
    """A new version `name` from the kept shards of `runs` (each its `shards/`), with the
    runs' validator summaries merged into the version's `summary.json`. With `part`, a
    part of a version instead (any name): each rented box packs its own runs so only
    packed shards come back, and `assemble` makes the version of the parts."""
    if not part and not VERSION_NAME.match(name):
        raise VersionError(f"version names are vN or vN.M, not {name!r}")
    target = root / name
    if target.exists():
        raise VersionError(f"{target} exists: a released version is never edited; make the next one")
    sources = {}
    for run in runs:
        for source in shard_dirs(run / "shards"):
            key = source.relative_to(run / "shards")
            if key in sources:
                raise VersionError(f"{key} is in two runs")
            sources[key] = source
    if not sources:
        raise VersionError(f"no shards in {[str(r) for r in runs]}")
    ensure_entry_marker(root)
    staging = root / f".{name}.partial"
    if staging.exists():
        # Shards are moved into it: an interrupted release's staging holds real shards.
        raise VersionError(f"{staging} exists (an interrupted release): move its shards "
                           "back into their run, or remove it after checking")
    # Moved, not copied: a run's output is disposable once released, and a copy of a
    # whole version would need its size again on the disk (v0: ~60 GB). Shards of one
    # clip share a patch store file, so each clip's shards go through one worker.
    by_clip: dict[str, list[tuple[Path, Path]]] = {}
    for key, source in sorted(sources.items()):
        by_clip.setdefault(key.name, []).append((source, staging / key))
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for problems in pool.map(_release_clip, by_clip.values(),
                                 [staging / PATCH_STORE] * len(by_clip)):
            if problems:
                raise VersionError(problems)
    (staging / MANIFEST).write_text(json.dumps(manifest_of(staging), indent=1))
    (staging / SUMMARY).write_text(json.dumps(merged_summary(runs), indent=1))
    staging.rename(target)
    return target


def assemble(parts: list[Path], name: str, root: Path = ROOT) -> Path:
    """Version `name` of released `parts` (each moved in whole, its patch store with it),
    with one manifest and the parts' summaries merged."""
    if not VERSION_NAME.match(name):
        raise VersionError(f"version names are vN or vN.M, not {name!r}")
    target = root / name
    if target.exists():
        raise VersionError(f"{target} exists: a released version is never edited; make the next one")
    ensure_entry_marker(root)
    staging = root / f".{name}.partial"
    staging.mkdir(parents=True, exist_ok=True)
    for part in parts:
        if verify(part):
            raise VersionError(f"{part}: differs from its manifest")
        shutil.move(part, staging / part.name)
    (staging / MANIFEST).write_text(json.dumps(manifest_of(staging), indent=1))
    (staging / SUMMARY).write_text(json.dumps(merged_summary(
        [staging / p.name for p in parts]), indent=1))
    staging.rename(target)
    return target


def merged_summary(runs: list[Path]) -> dict:
    merged = {"kept": 0, "quarantine": 0, "problems": {}}
    for run in runs:
        path = run / SUMMARY
        if path.exists():
            summary = json.loads(path.read_text())
            merged["kept"] += summary["kept"]
            merged["quarantine"] += summary["quarantine"]
            merged["problems"].update(summary["problems"])
    return merged


def unstage(staging: Path, run: Path) -> int:
    """Put the shards an interrupted release moved into `staging` back into `run`
    (patches inline again); returns how many."""
    moved = shard_dirs(staging)
    for shard_dir in moved:
        unstore_patches(shard_dir)
        target = run / "shards" / shard_dir.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(shard_dir, target)
    shutil.rmtree(staging)
    return len(moved)


def verify(directory: Path) -> list[str]:
    """Differences between a version and its manifest (empty: intact)."""
    recorded = json.loads((directory / MANIFEST).read_text())
    current = manifest_of(directory)
    if recorded["sha256"] == current["sha256"]:
        return []
    differ = []
    for part in ("shards", "patches"):
        names = set(recorded.get(part, {})) | set(current[part])
        differ += [f"{part}/{name}" for name in sorted(names)
                   if recorded.get(part, {}).get(name) != current[part].get(name)]
    return differ
