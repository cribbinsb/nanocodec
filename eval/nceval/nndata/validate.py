"""The validator every shard passes before it is used (nceval.nndata.generate runs it after
writing each shard; a shard that fails is quarantined, never silently used).

Each check returns the problems it finds, as text; a shard is valid when there are none.
The checks, beyond the loader's schema check (columns, dtypes, shapes, row counts):

- finite values: features finite; a superblock's cost finite exactly where it was priced
  (inside the picture, codecs with node trials); a node's cost finite in every slot the
  preset would have coded, never infinite;
- trees: each parent row comes before its children, in the same frame and superblock, one
  size up and covering them; each child hangs under a coded split; each searched split
  that codes children has some;
- choices: the chosen slot is eligible and the cheapest eligible one (the preset's rule);
- costs: a superblock's cost equals its root node's chosen cost and its preset tool
  trial's cost (the same choices on the same state; relative REL_TOLERANCE); every tool
  slot is priced where the superblock is;
- patches: residuals within the 8-bit difference range; the patch mean matches the
  superblock's luma mean (PATCH_MEAN_TOLERANCE, rounding of the averaged cells);
- times: every coded candidate took a nonzero time;
- provenance: the dump's stream was byte-identical to a plain encode (meta.json).
"""

from __future__ import annotations

import numpy as np

from .schema import SLOT_SPLIT, SLOTS
from .shard import Shard

#: Codecs without node trials: no superblock costs and no nodes (H.264).
UNPRICED_CODECS = frozenset({"h264"})
REL_TOLERANCE = 1e-5
#: The patch averages superblock samples into cells rounded to integers: its mean can
#: differ from the exact mean by at most half a level.
PATCH_MEAN_TOLERANCE = 0.51
MAX_RESIDUAL = 255
SB_FEATURES = ("luma_mean", "luma_variance", "gradient", "coherence", "block_variance",
               "static_share", "still", "zero_sad", "best_sad", "motion", "residual_energy")


def _eligible_mask(node: dict) -> np.ndarray:
    return (node["eligible"][:, None] >> np.arange(len(SLOTS))) & 1 == 1


def check_finite(shard: Shard) -> list[str]:
    sb, node, problems = shard["sb"], shard["node"], []
    for name in SB_FEATURES:
        if not np.isfinite(sb[name]).all():
            problems.append(f"sb.{name}: {int((~np.isfinite(sb[name])).sum())} non-finite values")
    priced = sb["inside"] == 1 if shard.labels.get("codec") not in UNPRICED_CODECS \
        else np.zeros(len(sb["inside"]), bool)
    wrong = np.isfinite(sb["cost"]) != priced
    if wrong.any():
        problems.append(f"sb.cost: {int(wrong.sum())} rows finite where unpriced or NaN where priced")
    if np.isinf(node["cost"]).any():
        problems.append("node.cost: infinite values")
    missing = _eligible_mask(node) & ~np.isfinite(node["cost"])
    if missing.any():
        problems.append(f"node.cost: {int(missing.sum())} eligible slots without a cost")
    return problems


def check_trees(shard: Shard) -> list[str]:
    node, problems = shard["node"], []
    parent = node["parent"]
    rows = np.arange(len(parent))
    has = parent >= 0
    if (parent[has] >= rows[has]).any() or (parent < -1).any():
        return ["node.parent: a parent row does not come before its child"]
    p = parent[has]
    for name in ("frame", "sb"):
        if (node[name][has] != node[name][p]).any():
            problems.append(f"node.parent: a child in another {name} than its parent")
    log2 = node["log2"].astype(int)
    if (log2[has] != log2[p] - 1).any():
        problems.append("node.parent: a child not one size below its parent")
    size = 1 << log2[p]
    inside = (node["x"][has] >= node["x"][p]) & (node["x"][has] < node["x"][p] + size) & \
        (node["y"][has] >= node["y"][p]) & (node["y"][has] < node["y"][p] + size)
    if not inside.all():
        problems.append("node.parent: a child outside its parent")
    if not np.isfinite(node["cost"][p, SLOT_SPLIT]).all():
        problems.append("node.parent: a child under a parent whose split was not coded")
    return problems


def check_choices(shard: Shard) -> list[str]:
    node = shard["node"]
    eligible = _eligible_mask(node)
    chosen = node["chosen"].astype(int)
    rows = np.arange(len(chosen))
    if (chosen >= len(SLOTS)).any() or not eligible[rows, np.minimum(chosen, len(SLOTS) - 1)].all():
        return ["node.chosen: a chosen slot the preset would not code"]
    # An abandoned slot holds its bound, a lower bound of its cost: never the cheapest.
    if "abandoned" in node:
        abandoned = (node["abandoned"][:, None] >> np.arange(len(SLOTS))) & 1 == 1
        eligible = eligible & ~abandoned
    costs = np.where(eligible, node["cost"], np.inf)
    worse = costs[rows, chosen] > costs.min(axis=1)
    return [f"node.chosen: {int(worse.sum())} nodes chose a costlier eligible slot"] if worse.any() else []


def check_costs(shard: Shard) -> list[str]:
    sb, node = shard["sb"], shard["node"]
    sb_log2 = int(shard["frame"]["superblock_log2"][0]) if shard.rows("frame") else 0
    root = (node["parent"] == -1) & (node["log2"] == sb_log2)
    rows = np.flatnonzero(root)
    chosen = node["cost"][rows, node["chosen"][rows]]
    expected = sb["cost"][node["sb"][rows]]
    problems = []
    if _differ(expected, chosen).any():
        problems.append(f"sb.cost: {int(_differ(expected, chosen).sum())} superblocks differ "
                        "from their root node's chosen cost")
    priced = np.isfinite(sb["cost"])
    if _differ(sb["cost"][priced], sb["tool_cost"][priced, 0]).any():
        problems.append("sb.tool_cost: the preset slot differs from the superblock's cost")
    if not np.isfinite(sb["tool_cost"][priced]).all():
        problems.append("sb.tool_cost: a priced superblock with an unpriced tool slot")
    return problems


def _differ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) > REL_TOLERANCE * np.maximum(np.abs(b), 1.0)


def check_patches(shard: Shard) -> list[str]:
    sb, problems = shard["sb"], []
    if np.abs(sb["residual"].astype(int)).max(initial=0) > MAX_RESIDUAL:
        problems.append("sb.residual: values beyond the 8-bit difference range")
    inside = sb["inside"] == 1
    means = sb["patch"][inside].reshape(int(inside.sum()), -1).mean(axis=1)
    off = np.abs(means - sb["luma_mean"][inside]) > PATCH_MEAN_TOLERANCE
    if off.any():
        problems.append(f"sb.patch: {int(off.sum())} patches whose mean is not the luma mean")
    return problems


def check_times(shard: Shard) -> list[str]:
    node = shard["node"]
    coded = np.isfinite(node["cost"])
    return ["node.ns: coded candidates with zero time"] if (node["ns"][coded] == 0).any() else []


def check_provenance(shard: Shard) -> list[str]:
    return [] if shard.meta.get("stream_identical") is True else \
        ["meta: the dump's stream is not recorded identical to a plain encode"]


CHECKS = (check_finite, check_trees, check_choices, check_costs, check_patches, check_times,
          check_provenance)


def validate(shard: Shard) -> list[str]:
    """Every problem of `shard` (empty: valid)."""
    return [problem for check in CHECKS for problem in check(shard)]
