"""Reduce label-dump shards to the zero-gate network's training table (one row per inter SB).

Where it fits: the zero-gate network (crates/nc-analysis/src/gate_net.rs) adds a learned term
to the motion search's zero-vector gates, which choose per superblock between no search and
the full hierarchical search. The dump records, per superblock, the network's input bytes as
the search computed them (`sb.gate_inputs`) and the search's zero-vector SAD and best total
SADs (`sb.gate_sads`). Coded with the gates off, every superblock is searched in full, so
`zero SAD - best 16x16 total` is what the full search gained over the zero vector there:
gate_train.py's label.

Per row: the input bytes; the zero SAD and the four best totals per pixel; the clip, its
family's train / held-out side (nceval.nndata.split_of, seed SPLIT_SEED), class and rate.

usage: gate_export.py OUT.npz RUN_DIR [RUN_DIR ...]
"""

import sys
from pathlib import Path

import numpy as np

from nceval.nndata import shard as shard_io
from nceval.nndata import split_of
from nceval.nndata.version import shard_dirs

SPLIT_SEED = "nn-leg-gate-1"
COLUMNS = {"sb.has_gate_inputs", "sb.gate_inputs", "sb.gate_sads", "sb.frame"}
#: Luma samples of an analysis superblock (64x64).
SB_PIXELS = 64 * 64


def main():
    out, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    paths = [p for run in runs for p in shard_dirs(run / "shards")]
    shards = [shard_io.load(p, only=COLUMNS) for p in paths]
    sides = split_of(shards, SPLIT_SEED)
    parts, labels = [], {"clip": [], "side": [], "class": [], "rate": []}
    for s in shards:
        sb = s["sb"]
        rows = np.flatnonzero(sb["has_gate_inputs"].astype(bool))
        job = s.meta["job"]
        clip = job["clip_meta"]["name"]
        for key, value in (("clip", clip), ("side", sides[clip]),
                           ("class", job["clip_meta"]["content_class"]), ("rate", job["rate"])):
            labels[key].extend([value] * len(rows))
        parts.append({"inputs": sb["gate_inputs"][rows],
                      "sads": sb["gate_sads"][rows].astype(np.float64) / SB_PIXELS,
                      "frame": sb["frame"][rows].astype(np.int32)})
    table = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    table.update({k: np.asarray(v) for k, v in labels.items()})
    np.savez_compressed(out, **table)
    gain = table["sads"][:, 0] - table["sads"][:, 2]
    print(f"{len(shards)} shards, {len(gain)} inter superblocks, held out "
          f"{np.mean(table['side'] == 'heldout'):.1%}, mean gain {gain.mean():.3f} SAD/px, "
          f"gain > 0.5/px on {np.mean(gain > 0.5):.1%}")


if __name__ == "__main__":
    main()
