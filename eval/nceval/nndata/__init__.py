"""The nn dataset: labelled encoder decisions for learned models (the only way to read it).

The encoder writes shards (`nanocodec encode --label-dump`); this package plans and runs
the encodes (generate), checks every shard (validate), releases immutable versions with a
manifest (version), splits by whole sources with leakage checks (split), and reports
coverage and the dataset card (coverage). Layout: docs/analysis/nn-dataset-schema.md.

Readers use:

    from nceval import nndata
    shards = nndata.load_version("v0")                    # every shard, schema-checked
    sides = nndata.split_of(shards, seed="nn-lead-1")      # clip -> train / heldout
    weights = nndata.balance_weights(shard)               # per superblock row
"""

from __future__ import annotations

from pathlib import Path


from .schema import FORMAT_VERSION, SLOTS
from .shard import Shard, ShardError, load
from .splits import HELD_OUT, TRAIN, LeakageError, check_split, split, split_of
from .balance import EASY_STATIC_SHARE, EASY_ZERO_SAD, balance_weights, easy_rows
from .version import ROOT, shard_dirs, verify

__all__ = ["FORMAT_VERSION", "HELD_OUT", "SLOTS", "TRAIN", "LeakageError", "Shard",
           "ShardError", "balance_weights", "easy_rows", "load", "load_version", "split_of"]


def load_version(name: str, root: Path = ROOT, check: bool = True) -> list[Shard]:
    """Every shard of released version `name`; with `check`, the manifest must match."""
    directory = root / name
    if check:
        changed = verify(directory)
        if changed:
            raise ShardError(f"{directory}: shards differ from the manifest: {changed[:5]}")
    return [load(path) for path in shard_dirs(directory)]
