"""Every causal input the encoder holds at a planned node, joined from one nn-dataset shard
(docs/analysis/distill-fast-2026-11-22.md section 2: the best model first, pruned later).

    python3 -m distill.features --root ~/nanocodec-data/nn-data/v0-parts --codec hevc \\
        --rungs medium,fast,faster --out DIR [--workers 32]

One row per node the planner weighed both ways (`has_inputs`) on an inter frame of a search
twin (every candidate coded), with the flip's value (the J the other choice of leaf or
quad split saves, per sample over lambda) as the label. Inputs, by group (GROUPS):

- `trial`: the eleven planner bytes the encoder records (`trial_inputs.rs`);
- `node`: size, the planned choice, the planner's margin, position inside the superblock;
- `depth_context`: the smallest coded leaf (log2, relative to the node) to the left, above,
  and over the node's area in the previous frame, from the shard's own coded trees;
- `neighbours`: the coded J per sample over lambda of the left, above and co-located
  previous superblocks; the co-located node's flip value and choice in the previous frame;
- `preanalysis`: the superblock's source and pre-analysis statistics;
- `motion`: the motion search's SADs, vector field statistics and search vector;
- `frame`: quantizer, log lambda, noise sigma, temporal layer, frames since the key;
- `rung`, `codec`: one-hot.

The superblock's luma and residual patches (32x32 each) are written once per superblock,
with each row's superblock index, for the convolutional variant.

Caveat (train / serve): the depth context and neighbour costs here are those of the search
twin's own coding. In a guided fast encode they are the guide's; the closed-loop check
(nceval.onpolicy) measures what that skew costs.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from distill.train_flip import TWIN_MARK, twin_dirs

SLOT_LEAF, SLOT_SPLIT, SLOT_NONE = 0, 1, 255
#: Shapes cut the node's shorter side by one level (binary) or two (ternary).
SHAPE_DEPTH = {2: 1, 3: 1, 4: 2, 5: 2}
#: Depth maps hold one value per 8x8 cell; 0 = unknown.
CELL_LOG2 = 3
UNKNOWN = 0
#: Where the context inputs come from (`--context`).
CONTEXT_TWIN, CONTEXT_PLAIN = "twin", "plain"
RUNGS = ("veryfast", "faster", "fast", "medium")
CODECS = ("h264", "hevc", "vvc", "av1", "vp9", "av2")
SB_STATS = ("luma_mean", "luma_variance", "gradient", "coherence", "block_variance",
            "static_share", "still")
SB_MOTION = ("zero_sad", "best_sad", "motion", "search_vector", "residual_energy")


def leaf_map(nodes: dict, rows: np.ndarray, cols: int, cell_rows: int) -> np.ndarray:
    """Per 8x8 cell, the log2 of the smallest coded leaf side over it (one frame's rows)."""
    out = np.full((cell_rows, cols), UNKNOWN, np.int8)
    for r in rows:
        chosen = int(nodes["chosen"][r])
        if chosen == SLOT_SPLIT:
            continue  # its quadrants have their own rows
        log2 = int(nodes["log2"][r])
        side = log2 - SHAPE_DEPTH.get(chosen, 0)
        x0, y0 = int(nodes["x"][r]) >> CELL_LOG2, int(nodes["y"][r]) >> CELL_LOG2
        n = 1 << max(0, log2 - CELL_LOG2)
        block = out[y0:y0 + n, x0:x0 + n]
        block[...] = np.where((block == UNKNOWN) | (block > side), side, block)
    return out


def relative(value: int, log2: int) -> int:
    return 0 if value == UNKNOWN else value - log2


def plain_twin(path: Path) -> Path:
    """The plain rung's shard of the same clip excerpt as search twin `path`."""
    job = path.parent.name.split(TWIN_MARK)[0]
    # The version's parts split one job's clips differently: look in every part.
    found = sorted(path.parent.parent.parent.glob(f"*/{job}/{path.name}"))
    if not found:
        raise FileNotFoundError(f"no plain shard {job}/{path.name} beside {path}")
    return found[0]


def shard_rows(path: Path, context: str = CONTEXT_TWIN) -> dict:
    """The rows of one shard (see the module doc), as numpy arrays. `context`: where the
    neighbours' costs come from (CONTEXT_TWIN: the twin's own coding; CONTEXT_PLAIN: the
    plain rung's coding of the same excerpt, what a deployed guide sees; v0 records no
    plain-rung inter depths, so the depth context and the previous decision are then
    left out until v0.1)."""
    from nceval import nndata
    shard = nndata.load(path)
    plain_sb = nndata.load(plain_twin(path), only={"sb.frame", "sb.x", "sb.y", "sb.cost"})["sb"] \
        if context == CONTEXT_PLAIN else None
    job, meta = shard.meta["job"], shard.meta["job"]["clip_meta"]
    frame, sb, node = shard["frame"], shard["sb"], shard["node"]
    findex = frame["index"].tolist()
    fpos = {f: i for i, f in enumerate(findex)}
    key = frame["key"]
    last_key = {}
    k = None
    for f in findex:
        if key[fpos[f]]:
            k = f
        last_key[f] = k
    width, height = int(frame["width"][0]), int(frame["height"][0])
    sb_log2 = int(frame["superblock_log2"][0])
    cols, crow = -(-width >> CELL_LOG2), -(-height >> CELL_LOG2)
    by_frame: dict[int, list[int]] = {}
    for r, f in enumerate(node["frame"].tolist()):
        by_frame.setdefault(f, []).append(r)
    maps = {f: leaf_map(node, np.array(rs), cols, crow) for f, rs in by_frame.items()}
    sb_at = {(int(f), int(x), int(y)): i for i, (f, x, y) in
             enumerate(zip(sb["frame"], sb["x"], sb["y"]))}
    node_at = {(int(f), int(x), int(y), int(l)): i for i, (f, x, y, l) in
               enumerate(zip(node["frame"], node["x"], node["y"], node["log2"]))}
    lam = frame["lambda"]
    cost = node["cost"]

    def flip_value(r: int) -> float:
        p = int(node["planned"][r])
        if p not in (SLOT_LEAF, SLOT_SPLIT):
            return np.nan
        here, other = cost[r, p], cost[r, 1 - p]
        area = float(1 << (2 * int(node["log2"][r])))
        return float((here - other) / (area * max(float(lam[fpos[int(node["frame"][r])]]), 1e-9)))

    cost_table, cost_at = (sb, sb_at) if plain_sb is None else (
        plain_sb, {(int(f), int(x), int(y)): i for i, (f, x, y) in
                   enumerate(zip(plain_sb["frame"], plain_sb["x"], plain_sb["y"]))})

    def sb_cost(f: int, x: int, y: int) -> float:
        i = cost_at.get((f, x, y))
        if i is None or not np.isfinite(cost_table["cost"][i]):
            return np.nan
        return float(cost_table["cost"][i] / ((1 << (2 * sb_log2)) * max(float(lam[fpos[f]]), 1e-9)))

    rows = [r for r in range(len(node["frame"]))
            if node["has_inputs"][r] and int(node["planned"][r]) in (SLOT_LEAF, SLOT_SPLIT)
            and not key[fpos[int(node["frame"][r])]]]
    out: dict[str, list] = {name: [] for name in ("label", "trial", "node", "depth_context",
                                                   "neighbours", "preanalysis", "motion",
                                                   "frame", "sb_row")}
    for r in rows:
        label = flip_value(r)
        if not np.isfinite(label):
            continue
        f, x, y, log2 = (int(node["frame"][r]), int(node["x"][r]), int(node["y"][r]),
                         int(node["log2"][r]))
        s = int(node["sb"][r])
        sbx, sby = int(sb["x"][s]), int(sb["y"][s])
        cx, cy = x >> CELL_LOG2, y >> CELL_LOG2
        n = 1 << max(0, log2 - CELL_LOG2)
        cur = maps[f]
        left = cur[cy:cy + n, cx - 1].min() if cx > 0 else UNKNOWN
        above = cur[cy - 1, cx:cx + n].min() if cy > 0 else UNKNOWN
        prev_f = f - 1 if (f - 1) in maps else None
        prev_area = maps[prev_f][cy:cy + n, cx:cx + n] if prev_f is not None else None
        co = int(prev_area[prev_area != UNKNOWN].min()) if prev_area is not None and \
            (prev_area != UNKNOWN).any() else UNKNOWN
        prev_node = node_at.get((f - 1, x, y, log2))
        sb_side = 1 << sb_log2
        out["label"].append(label)
        out["trial"].append(node["inputs"][r].astype(np.float32))
        out["node"].append([log2, int(node["planned"][r]),
                            np.log(max(float(node["plan_margin"][r]), 1.0)),
                            (x - sbx) / sb_side, (y - sby) / sb_side])
        out["depth_context"].append([relative(int(left), log2), relative(int(above), log2),
                                     relative(co, log2), float(left == UNKNOWN),
                                     float(above == UNKNOWN), float(co == UNKNOWN)])
        prev_value = flip_value(prev_node) if prev_node is not None else np.nan
        if plain_sb is not None:
            left = above = co = UNKNOWN
            prev_value, prev_node = np.nan, None
        out["neighbours"].append([
            sb_cost(f, sbx - sb_side, sby), sb_cost(f, sbx, sby - sb_side),
            sb_cost(f - 1, sbx, sby),
            0.0 if not np.isfinite(prev_value) else prev_value, float(np.isfinite(prev_value)),
            -1.0 if prev_node is None else float(node["chosen"][prev_node] == SLOT_SPLIT)])
        out["preanalysis"].append(np.concatenate([np.atleast_1d(sb[c][s]).astype(np.float32)
                                                  for c in SB_STATS]))
        out["motion"].append(np.concatenate([np.atleast_1d(sb[c][s]).astype(np.float32)
                                             for c in SB_MOTION]))
        i = fpos[f]
        out["frame"].append([float(frame["quantizer"][i]), np.log(max(float(lam[i]), 1e-9)),
                             float(frame["noise_sigma"][i]), float(frame["temporal_layer"][i]),
                             float(f - (last_key[f] if last_key[f] is not None else f))])
        out["sb_row"].append(s)
    arrays = {k: np.nan_to_num(np.asarray(v, np.float32)) if k not in ("sb_row",)
              else np.asarray(v, np.int32) for k, v in out.items()}
    count = len(out["label"])
    rung = np.zeros((count, len(RUNGS)), np.float32)
    if job["preset"] in RUNGS:
        rung[:, RUNGS.index(job["preset"])] = 1
    codec = np.zeros((count, len(CODECS)), np.float32)
    codec[:, CODECS.index(job["codec"])] = 1
    arrays["rung"], arrays["codec"] = rung, codec
    arrays["patch"] = sb["patch"].astype(np.uint8)
    arrays["residual"] = np.clip(sb["residual"], -128, 127).astype(np.int8)
    return {"arrays": arrays, "clip": meta, "preset": job["preset"], "rate": job["rate"]}


def write_one(args: tuple[Path, Path, str]) -> str:
    path, out, context = args
    target = out / f"{path.parent.parent.name}__{path.parent.name}__{path.name}.npz"
    if target.exists():
        return target.name
    rows = shard_rows(path, context)
    np.savez_compressed(target, **rows["arrays"],
                        meta=np.array(json.dumps({k: rows[k] for k in ("clip", "preset", "rate")})))
    return target.name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--codec", required=True)
    p.add_argument("--rungs", default="medium,fast,faster")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--context", choices=(CONTEXT_TWIN, CONTEXT_PLAIN), default=CONTEXT_TWIN)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = twin_dirs(args.root, args.codec, set(args.rungs.split(",")))
    assert all(TWIN_MARK in q.parent.name for q in paths)
    if args.context == CONTEXT_PLAIN:
        # Some twins have no plain job (AV1 medium qp-high): those excerpts are left out.
        def has_plain(q: Path) -> bool:
            try:
                plain_twin(q)
            except FileNotFoundError:
                return False
            return True
        kept = [q for q in paths if has_plain(q)]
        print(f"{len(paths) - len(kept)} twins without a plain shard left out", flush=True)
        paths = kept
    with ProcessPoolExecutor(args.workers) as pool:
        for i, name in enumerate(pool.map(write_one, [(q, args.out, args.context) for q in paths]), 1):
            if i % 100 == 0:
                print(f"{i}/{len(paths)} {name}", flush=True)
    print(f"{len(paths)} shards -> {args.out}")


if __name__ == "__main__":
    main()
