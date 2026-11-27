"""Balance of the nn dataset's rows: which superblocks are easy (near-static) and the
sampling weights that keep a model from training mostly on them."""

from __future__ import annotations

import numpy as np

from .shard import Shard

#: A superblock counts as easy (trivially coded) when pre-analysis marks at least this
#: share of its 8x8 cells static, or it is in an inter frame whose source did not change
#: by more than EASY_ZERO_SAD per sample against the last frame.
EASY_STATIC_SHARE = 0.9
EASY_ZERO_SAD = 1.0


def easy_rows(shard: Shard) -> np.ndarray:
    """Per superblock row: whether it is an easy, near-static case (module constants)."""
    sb, frames = shard["sb"], shard["frame"]
    key_of = dict(zip(frames["index"].tolist(), frames["key"].tolist()))
    inter = np.array([not key_of[f] for f in sb["frame"].tolist()], bool)
    return inter & ((sb["static_share"] >= EASY_STATIC_SHARE) | (sb["zero_sad"] <= EASY_ZERO_SAD))


def balance_weights(shard: Shard, easy_weight: float | None = None) -> np.ndarray:
    """Per superblock row: a sampling weight that gives easy and hard rows equal total
    weight (or `easy_weight` per easy row, hard rows 1), so a model is not trained mostly
    on trivial blocks."""
    easy = easy_rows(shard)
    if easy_weight is None:
        n_easy, n_hard = int(easy.sum()), int((~easy).sum())
        easy_weight = n_hard / n_easy if n_easy and n_hard else 1.0
    return np.where(easy, easy_weight, 1.0)
