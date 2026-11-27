"""Read one shard of the nn dataset (the only reader), and pack it for keeping.

A shard is one encode with `--label-dump`: `labels.json`, `meta.json` and the tables of
nceval.nndata.schema, either as the encoder wrote them (`<table>/<column>.npy`) or packed
into one compressed `tables.npz` (`pack`). In a released version the source patches
(`sb.patch`) live once per clip, start frame and superblock size in the version's patch
store (`store_patches`), since every rung and rate of a clip codes the same source; the
loader puts them back, so readers always see the full table. Loading checks the format version and every
column's dtype, row shape and row count against the schema, and raises ShardError on any
mismatch: a shard is used whole and as specified, or not at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schema import READABLE, SOURCE_COLUMNS, TABLES, TOOLS, required

LABELS_FILE, META_FILE, PACKED_FILE = "labels.json", "meta.json", "tables.npz"
#: The version's patch store, relative to the version (shards sit two levels below it).
PATCH_STORE = "patches"
#: meta.json key naming the shard's patch file in the store.
PATCH_KEY = "patch_store_file"


class ShardError(ValueError):
    """A shard that does not match the schema."""


@dataclass
class Shard:
    path: Path
    labels: dict
    meta: dict
    tables: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def __getitem__(self, table: str) -> dict[str, np.ndarray]:
        return self.tables[table]

    def rows(self, table: str) -> int:
        return len(next(iter(self.tables[table].values())))


def _read_columns(path: Path, only: set[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """The shard's columns (`only`: just these `table.column` keys)."""
    wanted = (lambda key: True) if only is None else (lambda key: key in only)
    packed = path / PACKED_FILE
    if packed.exists():
        with np.load(packed) as data:
            tables: dict[str, dict[str, np.ndarray]] = {t: {} for t in TABLES}
            for key in data.files:
                table, _, column = key.partition(".")
                if table not in tables:
                    raise ShardError(f"{packed}: unknown table in {key!r}")
                if wanted(key):
                    tables[table][column] = data[key]
            return tables
    return {t: {f.stem: np.load(f) for f in sorted((path / t).glob("*.npy"))
                if wanted(f"{t}.{f.stem}")}
            for t in TABLES if (path / t).is_dir()}


def declared(labels: dict, tools: int) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    """labels.json's columns: table -> column -> (dtype, row shape), tool rows resolved.
    A format-4 shard lists none: its columns are the format-4 set."""
    if "tables" not in labels:
        return {t: {c.name: (c.dtype, c.shape_for(tools)) for c in cols}
                for t, cols in required(labels["format_version"]).items()}
    return {table: {c["name"]: (c["dtype"], tuple(tools if d == TOOLS else d for d in c["shape"]))
                    for c in columns}
            for table, columns in labels["tables"].items()}


def check_columns(path: Path, tables: dict[str, dict[str, np.ndarray]], labels: dict,
                  tools: int) -> None:
    """The arrays are exactly the columns labels.json declares, with their dtypes and row
    shapes and one row count per table; the declaration holds every REQUIRED column with
    its dtype and shape."""
    declaration = declared(labels, tools)
    for table, columns in required(labels["format_version"]).items():
        for c in columns:
            if declaration.get(table, {}).get(c.name) != (c.dtype, c.shape_for(tools)):
                raise ShardError(f"{path}: {table}.{c.name} is declared "
                                 f"{declaration.get(table, {}).get(c.name)}, required "
                                 f"{(c.dtype, c.shape_for(tools))}")
    for table, columns in declaration.items():
        got = tables.get(table, {})
        if set(got) != set(columns):
            raise ShardError(f"{path}: {table} columns differ from labels.json: missing "
                             f"{sorted(set(columns) - set(got))}, extra {sorted(set(got) - set(columns))}")
        counts = {len(array) for array in got.values()}
        if len(counts) > 1:
            raise ShardError(f"{path}: {table} row counts differ: "
                             f"{ {name: len(array) for name, array in got.items()} }")
        for name, (dtype, shape) in columns.items():
            array = got[name]
            if array.dtype.str != dtype or array.shape[1:] != shape:
                raise ShardError(f"{path}: {table}.{name} is {array.dtype.str}{array.shape[1:]}, "
                                 f"declared {dtype}{shape}")


def load(path: Path, only: set[str] | None = None) -> Shard:
    """The shard at `path`, checked against the schema; with `only`, just those
    `table.column` keys (for summaries over many shards: the rest stay unread and
    unchecked)."""
    path = Path(path)
    try:
        labels = json.loads((path / LABELS_FILE).read_text())
    except FileNotFoundError as error:
        raise ShardError(f"{path}: not a shard ({LABELS_FILE} missing)") from error
    if labels.get("format_version") not in READABLE:
        raise ShardError(f"{path}: format version {labels.get('format_version')}, "
                         f"this reader reads {READABLE}")
    meta_path = path / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tables = _read_columns(path, only)
    tools = labels.get("tools")
    if not isinstance(tools, list) or not tools or tools[0] != "preset":
        raise ShardError(f"{path}: labels.json tools {tools!r}: the preset's slot comes first")
    if only is not None:
        return Shard(path=path, labels=labels, meta=meta, tables=tables)
    if PATCH_KEY in meta:
        tables["sb"].update(_stored_columns(path, meta, tables["sb"]))
    check_columns(path, tables, labels, len(tools))
    return Shard(path=path, labels=labels, meta=meta, tables=tables)


def pack(path: Path) -> Path:
    """Pack an unpacked shard's columns into one compressed `tables.npz` (about 3x
    smaller; first shard 29 -> 8.8 MB) and remove the column files. Checked first."""
    path = Path(path)
    shard = load(path)
    packed = path / PACKED_FILE
    # Written whole, then renamed: a run killed while packing leaves no partial file.
    partial = path / f".{PACKED_FILE}.partial.npz"
    np.savez_compressed(partial, **{f"{t}.{c}": a for t, cols in shard.tables.items()
                                    for c, a in cols.items()})
    partial.rename(packed)
    for table in TABLES:
        if (path / table).is_dir():
            for column in (path / table).glob("*.npy"):
                column.unlink()
            (path / table).rmdir()
    return packed


def _patch_file(path: Path, name: str) -> Path:
    return path.parents[1] / PATCH_STORE / name


def _patch_index(sb: dict[str, np.ndarray], side: int) -> tuple[np.ndarray, ...]:
    return sb["frame"].astype(np.int64), (sb["y"] // side).astype(np.int64), \
        (sb["x"] // side).astype(np.int64)


#: A v0 store file names the patch array so (later versions name it by its column).
LEGACY_PATCH_ARRAY = "patches"
#: meta.json key listing the sb columns held in the store (v0: only `patch`).
STORED_KEY = "stored_columns"


def _array_key(store: np.lib.npyio.NpzFile, column: str) -> str:
    return LEGACY_PATCH_ARRAY if column == "patch" and LEGACY_PATCH_ARRAY in store.files else column


def _stored_columns(path: Path, meta: dict, sb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The store's source columns of this shard's superblocks."""
    with np.load(_patch_file(path, meta[PATCH_KEY])) as store:
        side = int(store["superblock"])
        index = _patch_index(sb, side)
        return {c: store[_array_key(store, c)][index] for c in meta.get(STORED_KEY, ["patch"])}


def store_patches(path: Path, store: Path) -> None:
    """Move the packed shard's source columns (SOURCE_COLUMNS: the patch and the cells'
    source statistics, the same at every rung and rate of a clip) into the version's
    `store`: one file per clip, start frame, superblock size and coded size, indexed
    [frame, sb row, sb column]. A shard whose coded size changes, or whose values differ
    from those stored (a rung that prefilters or rescales its input), keeps its own."""
    shard = load(path)
    job = shard.meta["job"]
    frames = shard["frame"]
    side = 1 << int(frames["superblock_log2"][0])
    sizes = {(int(w), int(h)) for w, h in zip(frames["width"], frames["height"])}
    if len(sizes) != 1:
        # A stream that changed its coded size (adaptive resolution) codes other source
        # samples under the same superblock keys: its source columns stay in the shard.
        return
    (width, height), = sizes
    name = f"{job['clip']}@{job['start_frame']}~sb{side}~{width}x{height}.npz"
    sb = shard["sb"]
    columns = [c for c in SOURCE_COLUMNS if c in sb]
    index = _patch_index(sb, side)
    file = store / name
    count = int(job["frames"])  # the shard's frames are numbered from 0
    grid = (count, -(-height // side), -(-width // side))
    arrays: dict[str, np.ndarray] = {}
    if file.exists():
        with np.load(file) as old:
            arrays = {key: old[key] for key in old.files}
    for column in columns:
        key = LEGACY_PATCH_ARRAY if column == "patch" and LEGACY_PATCH_ARRAY in arrays else column
        values = arrays.setdefault(key, np.zeros((*grid, *sb[column].shape[1:]), sb[column].dtype))
        filled = arrays.setdefault(f"{key}.filled", np.zeros(grid, bool))
        known = filled[index]
        if not np.array_equal(values[index][known], sb[column][known]):
            # The same source position coded other samples (a rung that filters or
            # rescales its input before coding): this shard keeps its own.
            return
        values[index], filled[index] = sb[column], True
    arrays["superblock"] = np.int64(side)
    store.mkdir(parents=True, exist_ok=True)
    partial = store / f".{name}.partial.npz"
    np.savez_compressed(partial, **arrays)
    partial.rename(file)
    kept = {f"{t}.{c}": a for t, cols in shard.tables.items() for c, a in cols.items()
            if not (t == "sb" and c in columns)}
    partial = path / f".{PACKED_FILE}.partial.npz"
    np.savez_compressed(partial, **kept)
    partial.rename(path / PACKED_FILE)
    shard.meta[PATCH_KEY] = name
    shard.meta[STORED_KEY] = columns
    (path / META_FILE).write_text(json.dumps(shard.meta, indent=1))


def unstore_patches(path: Path) -> None:
    """Undo `store_patches` on one shard: its patches back inline, the store reference
    dropped (a release interrupted after moving it)."""
    shard = load(path)
    if PATCH_KEY not in shard.meta:
        return
    partial = path / f".{PACKED_FILE}.partial.npz"
    np.savez_compressed(partial, **{f"{t}.{c}": a for t, cols in shard.tables.items()
                                    for c, a in cols.items()})
    partial.rename(path / PACKED_FILE)
    del shard.meta[PATCH_KEY]
    shard.meta.pop(STORED_KEY, None)
    (path / META_FILE).write_text(json.dumps(shard.meta, indent=1))
