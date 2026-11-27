"""Reduce label-dump shards to the post-trial stop's training table (one row per stop point).

Where it fits: the stop network (crates/nc-encoder/src/stop_value.rs) scores a planned inter
node of the RD partition search once its planned candidate is coded; the dump records, at
exactly those nodes, the network's input bytes as the search computed them
(`node.stop_inputs`), every candidate's coded cost and deterministic work (each coded in
full), and which candidates the preset's gates and bounds admitted (`node.admitted`, before
any stop). stop_train.py reads the table written here.

Per row: the input bytes; the planned candidate's cost and work; the best admitted
alternative's cost and the admitted alternatives' summed work (the trials a stop skips);
whether the preset's stop skipped them (`node.eligible` lacks an admitted alternative); and
the clip, its family's train / held-out side (nceval.nndata.split_of, seed SPLIT_SEED),
class, codec, rate and frame. Rows without an admitted alternative are left out: no stop
can act there.

usage: stop_export.py OUT.npz RUN_DIR [RUN_DIR ...]
"""

import sys
from pathlib import Path

import numpy as np

from nceval.nndata import shard as shard_io
from nceval.nndata import split_of
from nceval.nndata.version import shard_dirs

SPLIT_SEED = "nn-leg-stop-1"
COLUMNS = {f"node.{c}" for c in ("has_stop_inputs", "stop_inputs", "planned", "admitted",
                                 "eligible", "cost", "cand_work", "frame", "log2")}


def rows_of(shard):
    node = shard["node"]
    has = node["has_stop_inputs"].astype(bool)
    planned = node["planned"].astype(np.int64)
    slots = node["cost"].shape[1]
    bits = 1 << np.arange(slots)
    admitted = (node["admitted"][:, None].astype(np.int64) & bits) != 0
    eligible = (node["eligible"][:, None].astype(np.int64) & bits) != 0
    alternative = admitted.copy()
    rows = np.flatnonzero(has & (planned < slots))
    alternative[rows, planned[rows]] = False
    rows = rows[alternative[rows].any(1)]
    cost = node["cost"][rows].astype(np.float64)
    work = node["cand_work"][rows].astype(np.float64)
    alt = alternative[rows]
    p = planned[rows]
    alt_cost = np.where(alt, np.nan_to_num(cost, nan=np.inf), np.inf).min(1)
    return {
        "inputs": node["stop_inputs"][rows],
        "planned_cost": cost[np.arange(len(rows)), p],
        "planned_work": work[np.arange(len(rows)), p],
        "alt_cost": alt_cost,
        "alt_work": np.where(alt, work, 0.0).sum(1),
        "stopped": (alt & ~eligible[rows]).any(1),
        "frame": node["frame"][rows].astype(np.int32),
        "log2": node["log2"][rows],
    }


def main():
    out, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    paths = [p for run in runs for p in shard_dirs(run / "shards")]
    shards = [shard_io.load(p, only=COLUMNS) for p in paths]
    sides = split_of(shards, SPLIT_SEED)
    parts, labels = [], {"clip": [], "side": [], "class": [], "codec": [], "rate": []}
    for s in shards:
        rows = rows_of(s)
        n = len(rows["inputs"])
        job = s.meta["job"]
        clip = job["clip_meta"]["name"]
        for key, value in (("clip", clip), ("side", sides[clip]),
                           ("class", job["clip_meta"]["content_class"]),
                           ("codec", job["codec"]), ("rate", job["rate"])):
            labels[key].extend([value] * n)
        parts.append(rows)
    table = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    table.update({k: np.asarray(v) for k, v in labels.items()})
    np.savez_compressed(out, **table)
    print(f"{len(shards)} shards, {len(table['inputs'])} stop points, "
          f"held out {np.mean(table['side'] == 'heldout'):.1%}, stopped {table['stopped'].mean():.1%}")


if __name__ == "__main__":
    main()
