"""One table of the inter planner's leaf-vs-split estimates and the coded costs, from nn-dataset
shards (docs/analysis/learned-estimator-2026-11-22.md section 1).

    python3 -m estimator.extract --version v0.3 --out FILE.npz [--cap 4000] [--workers 6]

One row per node of an inter frame that the planner weighed both ways (`plan_costs` finite) and
that the RD search coded both ways in full (cost of slots 0 and 1 finite, neither abandoned).
The planner's estimates are `plan_costs / 1000` (it compares `bias * leaf` with
`1000 * split`, so the leaf estimate carries the bias: exactly what the decision uses).

Inputs (FEATURES) are what the planner could hold when it plans the node's superblock:
- the estimate itself and the node: log2, the estimated log ratio, the estimates per sample
  over lambda, the planner's margin and the eleven trial bytes;
- the frame and the superblock's analysis statistics;
- the coded context, NNUE-style: for the left and above superblocks (coded before this one is
  planned) and the co-located superblock of the previous coded frame, how far the planner's
  estimate of each coded choice was from what it really coded (log coded / estimated, over
  the nodes planned leaf and planned split, all sizes and the node's size), their coded J per
  sample over lambda, and the co-located node's own calibration and estimated ratio.
  The calibration of a node uses only the planned slot's coded cost, which a planner-only
  encode also learns once it codes the plan.
"""

from __future__ import annotations

import argparse
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nceval import nndata
from nceval.nndata.schema import SLOT_LEAF, SLOT_SPLIT

#: The planner compares `bias * leaf` with `1000 * split` (partition_plan.rs plan_node).
PLAN_COST_SCALE = 1000.0
COLUMNS = {
    "frame.index", "frame.key", "frame.lambda", "frame.quantizer", "frame.noise_sigma",
    "frame.superblock_log2",
    "sb.frame", "sb.x", "sb.y", "sb.cost", "sb.luma_variance", "sb.zero_sad", "sb.best_sad",
    "sb.residual_energy", "sb.static_share",
    "node.frame", "node.x", "node.y", "node.log2", "node.sb", "node.planned", "node.chosen",
    "node.plan_margin", "node.plan_costs", "node.final_context", "node.search", "node.pre",
    "node.abandoned", "node.distortion", "node.bits", "node.cost", "node.has_inputs",
    "node.inputs",
}
CODECS = ("hevc", "av1", "vvc")
PRESETS = ("medium", "slow")
RATES = ("qp-high", "cbr")
#: Context features per neighbour superblock (left, above, co-located previous frame).
NEIGHBOUR_FEATURES = ("leaf_calib", "leaf_n", "split_calib", "split_n", "leaf_calib_size",
                      "split_calib_size", "cost_per_sample")
NEIGHBOURS = ("left", "above", "colo")
BASE_FEATURES = (
    "log2", "est_log_ratio", "leaf_est_per_sample", "split_est_per_sample", "log_margin",
    "log_lambda", "quantizer", "noise_sigma",
    "sb_luma_variance", "sb_zero_sad", "sb_best_sad", "sb_residual_energy", "sb_static_share",
    "final_context_nb", "final_context_colo",
    *(f"search{i}" for i in range(4)), *(f"pre{i}" for i in range(5)),
    *(f"input{i}" for i in range(11)),
)
CONTEXT_FEATURES = (
    *(f"{w}_{f}" for w in NEIGHBOURS for f in NEIGHBOUR_FEATURES),
    "colo_node_calib", "colo_node_est_log_ratio", "colo_node_planned",
)
FEATURES = (*BASE_FEATURES, *CONTEXT_FEATURES)
LABELS = ("cost0", "cost1", "dist0", "dist1", "bits0", "bits1", "chosen", "planned", "lambda")
#: Size slots of the per-(superblock, size) aggregates (log2 up to 7).
SIZES = 8


def _mean_by(index: np.ndarray, values: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` per group (NaN where empty) and the count."""
    count = np.bincount(index, minlength=groups).astype(np.float32)
    total = np.bincount(index, weights=values, minlength=groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total / count).astype(np.float32), count


def _calibration(node: dict, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per node, log(coded / estimated) of its planned leaf and of its planned split (NaN where
    it was not planned that way or either side is missing)."""
    cost = node["cost"]
    planned = node["planned"]
    with np.errstate(invalid="ignore", divide="ignore"):
        leaf = np.where(planned == SLOT_LEAF, np.log(cost[:, 0] / est[:, 0]), np.nan)
        split = np.where(planned == SLOT_SPLIT, np.log(cost[:, 1] / est[:, 1]), np.nan)
    return leaf, split


def _neighbour_table(node: dict, sb: dict, leaf: np.ndarray, split: np.ndarray,
                     lam_sb: np.ndarray, side: int) -> np.ndarray:
    """Per superblock row, the NEIGHBOUR_FEATURES of that superblock as a neighbour."""
    groups = len(sb["frame"])
    out = np.full((groups, len(NEIGHBOUR_FEATURES)), np.nan, np.float32)
    for column, calib in ((0, leaf), (2, split)):
        ok = np.isfinite(calib)
        mean, count = _mean_by(node["sb"][ok].astype(np.int64), calib[ok], groups)
        out[:, column], out[:, column + 1] = mean, count
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, 6] = sb["cost"] / (side * side) / lam_sb
    return out


def _sized(node: dict, calib: np.ndarray, groups: int) -> np.ndarray:
    """Per (superblock row, size), the mean calibration (flattened sb * SIZES + size)."""
    ok = np.isfinite(calib)
    index = node["sb"][ok].astype(np.int64) * SIZES + np.minimum(node["log2"][ok], SIZES - 1)
    return _mean_by(index, calib[ok], groups * SIZES)[0]


def _job_ids(shard: nndata.Shard) -> tuple[int, int, int]:
    job = shard.meta["job"]
    return CODECS.index(job["codec"]), PRESETS.index(job["preset"]), RATES.index(job["rate"])


def extract(path: Path, cap: int) -> dict[str, np.ndarray] | None:
    """The rows of one shard, at most `cap` drawn at random (seeded by the shard's path)."""
    shard = nndata.load(path, only=COLUMNS)
    frame, sb, node = shard["frame"], shard["sb"], shard["node"]
    if len(node.get("frame", [])) == 0 or shard.meta["job"]["codec"] not in CODECS:
        return None
    side = 1 << int(frame["superblock_log2"][0])
    frame_row = {int(index): row for row, index in enumerate(frame["index"])}
    node_frame = np.array([frame_row[int(i)] for i in node["frame"]])
    sb_frame = np.array([frame_row[int(i)] for i in sb["frame"]])
    lam = frame["lambda"][node_frame]
    est = node["plan_costs"].astype(np.float64) / PLAN_COST_SCALE
    leaf, split = _calibration(node, est)
    groups = len(sb["frame"])
    table = _neighbour_table(node, sb, leaf, split, frame["lambda"][sb_frame], side)
    leaf_sized, split_sized = _sized(node, leaf, groups), _sized(node, split, groups)

    cost = node["cost"]
    abandoned = node["abandoned"]
    keep = ((frame["key"][node_frame] == 0) & np.isfinite(est).all(1) & (est > 0).all(1)
            & np.isfinite(cost[:, :2]).all(1) & (cost[:, :2] > 0).all(1)
            & ((abandoned & 3) == 0))
    rows = np.flatnonzero(keep)
    if len(rows) == 0:
        return None
    rng = np.random.default_rng(zlib.crc32(str(path).encode()))
    if len(rows) > cap:
        rows = np.sort(rng.choice(rows, cap, replace=False))

    sb_at = {(int(f), int(x), int(y)): r
             for r, (f, x, y) in enumerate(zip(sb_frame, sb["x"], sb["y"]))}
    node_at = {(int(f), int(x), int(y), int(l)): r
               for r, (f, x, y, l) in enumerate(zip(node_frame, node["x"], node["y"],
                                                    node["log2"]))}
    log2 = node["log2"][rows].astype(np.int64)
    own_sb = node["sb"][rows].astype(np.int64)
    sx, sy, sf = sb["x"][own_sb].astype(int), sb["y"][own_sb].astype(int), node_frame[rows]
    where = {
        "left": [sb_at.get((f, x - side, y), -1) for f, x, y in zip(sf, sx, sy)],
        "above": [sb_at.get((f, x, y - side), -1) for f, x, y in zip(sf, sx, sy)],
        "colo": [sb_at.get((f - 1, x, y), -1) for f, x, y in zip(sf, sx, sy)],
    }
    context = []
    for name in NEIGHBOURS:
        at = np.array(where[name], np.int64)
        part = np.full((len(rows), len(NEIGHBOUR_FEATURES)), np.nan, np.float32)
        ok = at >= 0
        part[ok] = table[at[ok]]
        sized = at[ok] * SIZES + np.minimum(log2[ok], SIZES - 1)
        part[ok, 4], part[ok, 5] = leaf_sized[sized], split_sized[sized]
        context.append(part)
    colo_node = np.array([node_at.get((int(f) - 1, int(x), int(y), int(l)), -1)
                          for f, x, y, l in zip(sf, node["x"][rows], node["y"][rows], log2)])
    colo = np.full((len(rows), 3), np.nan, np.float32)
    ok = colo_node >= 0
    at = colo_node[ok]
    colo[ok, 0] = np.where(np.isfinite(leaf[at]), leaf[at], split[at])
    with np.errstate(invalid="ignore", divide="ignore"):
        colo[ok, 1] = np.log(est[at, 0] / est[at, 1])
    colo[ok, 2] = node["planned"][at]
    context.append(colo)

    area = (1 << (2 * log2)).astype(np.float64)
    lr = lam[rows].astype(np.float64)
    inputs = np.where(node["has_inputs"][rows, None] == 1, node["inputs"][rows].astype(np.float32),
                      np.nan)
    base = np.column_stack([
        log2, np.log(est[rows, 0] / est[rows, 1]),
        np.log(est[rows, 0] / area / lr), np.log(est[rows, 1] / area / lr),
        np.log(np.maximum(node["plan_margin"][rows], 1e-6)), np.log(lr),
        frame["quantizer"][node_frame[rows]], frame["noise_sigma"][node_frame[rows]],
        sb["luma_variance"][own_sb], sb["zero_sad"][own_sb], sb["best_sad"][own_sb],
        sb["residual_energy"][own_sb], sb["static_share"][own_sb],
        node["final_context"][rows].astype(np.float32),
        node["search"][rows], node["pre"][rows], inputs,
    ]).astype(np.float32)
    features = np.concatenate([base, *context], axis=1)
    labels = np.column_stack([
        cost[rows, 0], cost[rows, 1], node["distortion"][rows, 0], node["distortion"][rows, 1],
        node["bits"][rows, 0], node["bits"][rows, 1], node["chosen"][rows], node["planned"][rows],
        lr]).astype(np.float32)
    codec, preset, rate = _job_ids(shard)
    clip = shard.meta["job"]["clip_meta"]
    return {"features": features, "labels": labels,
            "job": np.tile(np.array([[codec, preset, rate]], np.int8), (len(rows), 1)),
            "clip": np.full(len(rows), clip["name"], object), "family": clip["family"],
            "clip_meta": clip, "twin": "rd_split=1" in str(path.parent.name)}


def _one(args: tuple[Path, int]) -> dict | None:
    return extract(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", default="v0.3")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="first N shards only (pipe-clean)")
    args = parser.parse_args()
    paths = nndata.shard_dirs(nndata.ROOT / args.version)
    if args.limit:
        paths = paths[:: max(1, len(paths) // args.limit)][: args.limit]
    parts, metas = [], {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, part in enumerate(pool.map(_one, [(p, args.cap) for p in paths], chunksize=4)):
            if part is not None:
                part["twin_flag"] = np.full(len(part["clip"]), part.pop("twin"), bool)
                meta = part.pop("clip_meta")
                metas[meta["name"]] = meta
                part.pop("family")
                parts.append(part)
            if i % 100 == 0:
                print(f"{i}/{len(paths)} shards, {sum(len(p['clip']) for p in parts)} rows",
                      flush=True)
    joined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    names = sorted(metas)
    clip_index = {n: i for i, n in enumerate(names)}
    np.savez(args.out, features=joined["features"], labels=joined["labels"], job=joined["job"],
             twin=joined["twin_flag"],
             clip=np.array([clip_index[c] for c in joined["clip"]], np.int32),
             clip_names=np.array(names), feature_names=np.array(FEATURES),
             label_names=np.array(LABELS),
             clip_meta=np.array([repr(metas[n]) for n in names]))
    print(f"{len(joined['clip'])} rows, {len(names)} clips -> {args.out}")


if __name__ == "__main__":
    main()
