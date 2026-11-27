"""The distilled partition guide: a tiny integer network that tells a fast rung where to
change its plan's leaf / split choice to the one the slowest rung's search would code,
without a trial (docs/analysis/distill-fast-2026-11-22.md section 2).

    systemd-run --user --scope -p MemoryMax=8G nice -n 10 \\
        python3 -m distill.train_flip --version v0 --codec av1 --rungs medium,fast,faster \\
        --seed nn-fast-1 --out ~/nanocodec-data/work/nn-fast/models/flip-av1.json

Data: the nn dataset's ungated search twins of the fast rungs (nceval.nndata, the only
reader; `planner.rd_split=1` with the gates off, so every planned node also coded its
alternative). One row per node the planner weighed both ways (`has_inputs`), on inter
frames: its eleven planner inputs (`trial_inputs.rs`, the exact bytes the encoder computes)
and the coded J of the planned choice and of the other one (leaf against quad split).
Only the bytes the encoder recorded are used, never recomputed features (train / serve
parity; the missing-quadrant case is reported apart). The
flip's value is the J it saves, per sample in units of the frame's lambda.

The network is the nc-dsp `nn` kernels' arithmetic exactly (branch nn-kernel): inputs
0..127 padded to 32, int8 weights, int32 bias, `clamp(sum >> SHIFT, 0, 127)`, then a
linear int32 output. It predicts the flip's value (nn-lead's
quantise.QuantNet: float, then quantisation-aware with straight-through rounding; the
integer forward is intnet.forward). The rung's price is the value threshold above which
it flips, chosen on the training side; the held-out side (whole source groups,
nceval.nndata.split_of) reports:

- the share of the oracle's flip gain kept (sum of the J saved by the flips it makes,
  losses included, over the J the right flips would save), and the flip rate;
- the same for the rule baseline (flip where the planner's margin is under a threshold);
- per content class, the share kept and the worst class.

The record (`--out`, the network beside it as `.npz`, intnet's format) holds the threshold, the dataset version and
manifest hash, the seed, this script's path and git commit, and every number above.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INPUTS = 11
#: trial_inputs.rs: the byte that says whether the quadrant inputs were measured.
QUADRANTS_FLAG = 9
HIDDEN = (32, 8)
#: Each hidden layer's shift and the output scale (nn-lead's quantise.py defaults).
SHIFT = 6
OUTPUT_SCALE_LOG2 = 6
SLOT_LEAF, SLOT_SPLIT = 0, 1
EPOCHS = 30
#: Rows the network is fitted on (a uniform sample of the training side; thresholds and
#: reports use every row).
MAX_FIT_ROWS = 4_000_000
#: Share of |gain| below which the training target is left as it is (the rest clipped).
TARGET_CLIP_QUANTILE = 0.99
DEFAULT_LEARNED_TREE = Path.home() / "nanocodec-data/worktrees/nn-lead/eval/learned"


@dataclass
class Rows:
    x: np.ndarray        # (N, 11) uint8 planner inputs
    gain: np.ndarray     # (N,) J the flip saves, per sample over lambda (negative: loses)
    margin: np.ndarray   # (N,) the planner's margin
    clip: np.ndarray     # (N,) clip name (a short categorical array: few distinct names)
    cls: np.ndarray      # (N,) content class
    rung: np.ndarray     # (N,) preset


#: The columns the guide reads (nceval.nndata.load `only`).
COLUMNS = {"frame.index", "frame.key", "frame.lambda", "node.frame", "node.log2", "node.planned",
           "node.cost", "node.has_inputs", "node.inputs", "node.plan_margin"}
#: A search twin's directory names its extra settings (nceval.nndata.generate).
TWIN_MARK = "+planner.rd_split=1"


def twin_dirs(root: Path, codec: str, rungs: set[str]) -> list[Path]:
    """The search-twin shards of `codec` at `rungs` below `root`, by their job directory."""
    out = []
    for meta in sorted(root.rglob("meta.json")):
        job = meta.parent.parent.name
        if TWIN_MARK in job and job.split("-")[0] == codec and job.split("-")[1] in rungs:
            out.append(meta.parent)
    return out


def load_rows(nndata, paths: list[Path], codec: str, rungs: set[str]) -> tuple[Rows, list]:
    """The rows of every shard, read one at a time (only COLUMNS), and the clips' metadata."""
    parts, clips, vocab = [], [], {}
    for path in paths:
        shard = nndata.load(path, only=COLUMNS)
        clips.append(shard.meta["job"]["clip_meta"])
        job = shard.meta["job"]
        if job["codec"] != codec or job["preset"] not in rungs:
            continue
        if "planner.rd_split=1" not in " ".join(job.get("params", [])):
            continue  # only the search twins code both choices
        frame, node = shard["frame"], shard["node"]
        lam = dict(zip(frame["index"].tolist(), frame["lambda"].tolist()))
        key = dict(zip(frame["index"].tolist(), frame["key"].tolist()))
        planned = node["planned"].astype(int)
        cost = node["cost"]
        use = (node["has_inputs"] == 1) & np.isin(planned, (SLOT_LEAF, SLOT_SPLIT))
        use &= np.array([not key[f] for f in node["frame"].tolist()], bool)
        idx = np.nonzero(use)[0]
        here = cost[idx, planned[idx]]
        other = cost[idx, 1 - planned[idx]]
        ok = np.isfinite(here) & np.isfinite(other)
        idx, here, other = idx[ok], here[ok], other[ok]
        area = (1 << (2 * node["log2"][idx].astype(np.int64))).astype(np.float64)
        lam_row = np.array([lam[f] for f in node["frame"][idx].tolist()])
        meta = job["clip_meta"]
        n = len(idx)
        names = (meta["name"], meta.get("content_class", "?"), job["preset"])
        codes = [vocab.setdefault(name, len(vocab)) for name in names]
        parts.append(Rows(node["inputs"][idx].astype(np.uint8),
                          ((here - other) / (area * np.maximum(lam_row, 1e-9))).astype(np.float32),
                          node["plan_margin"][idx].astype(np.float32),
                          *(np.full(n, c, np.int32) for c in codes)))
    if not parts:
        raise SystemExit(f"no {codec} search-twin nodes for rungs {sorted(rungs)}")
    rows = Rows(*(np.concatenate([getattr(p, f) for p in parts])
                  for f in Rows.__dataclass_fields__))
    # Categorical codes back to names, as small object arrays only at the end.
    names = np.array(list(vocab), dtype=object)
    rows.clip, rows.cls, rows.rung = names[rows.clip], names[rows.cls], names[rows.rung]
    return rows, clips


def subset(rows: Rows, mask: np.ndarray) -> Rows:
    return Rows(*(getattr(rows, f)[mask] for f in Rows.__dataclass_fields__))


def kept_share(gain: np.ndarray, flip: np.ndarray) -> float:
    """The J the flips save (losses included) over what the right flips would save."""
    best = gain[gain > 0].sum()
    return float(gain[flip].sum() / best) if best > 0 else float("nan")


# --- the network --------------------------------------------------------------------
# nn-lead's quantisation-aware trainer and exact integer forward (eval/learned on branch
# nn-lead: quantise.QuantNet, intnet), shared rather than copied: one integer network
# format, one export path (export_weights.py) to the nc-dsp nn kernels.

def train(fit: Rows, seed: int, device: str, learned_tree: Path):
    """A value network: the flip's J saved per sample over lambda, float then QAT."""
    sys.path.insert(0, str(learned_tree))
    import torch
    import intnet
    import quantise
    import train_node_mlp
    torch.manual_seed(seed)
    x = torch.tensor(fit.x, dtype=torch.float32, device=device)
    # The tails are a few huge nodes (edges, scene changes): clipped so the fit follows
    # the bulk; the ranking, not the scale, decides.
    limit = np.quantile(np.abs(fit.gain), TARGET_CLIP_QUANTILE)
    y = torch.tensor(np.clip(fit.gain, -limit, limit), dtype=torch.float32, device=device)
    sizes = (fit.x.shape[1], *HIDDEN, 1)
    model = quantise.QuantNet(sizes, (SHIFT,) * len(HIDDEN), OUTPUT_SCALE_LOG2,
                              x.mean(0).cpu(), x.std(0).cpu()).to(device)
    quantise.train(model, x, y, EPOCHS, train_node_mlp.LEARNING_RATE, quantised=False)
    quantise.train(model, x, y, quantise.QAT_EPOCHS, quantise.QAT_LEARNING_RATE, quantised=True)
    return model.to_intnet(), intnet


def best_threshold(scores: np.ndarray, gain: np.ndarray) -> float:
    """The price: the output threshold that keeps the most of the training gain."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 1.0, 201)))
    kept = [kept_share(gain, scores > t) for t in candidates]
    return float(candidates[int(np.nanargmax(kept))])


def report(rows: Rows, flip: np.ndarray) -> dict:
    classes = {}
    for c in sorted(set(rows.cls.tolist())):
        m = rows.cls == c
        classes[c] = round(kept_share(rows.gain[m], flip[m]), 4)
    by_rung = {r: round(kept_share(rows.gain[rows.rung == r], flip[rows.rung == r]), 4)
               for r in sorted(set(rows.rung.tolist()))}
    # A node whose vector equals its parent's reuses the parent's statistics and has no
    # quadrant inputs (partition_plan.rs, distortion_reusing): its quadrant bytes are 0
    # with the flag byte off. Reported apart, since a model can misread them.
    quadrants = rows.x[:, QUADRANTS_FLAG] > 0
    by_quadrants = {name: {"kept": round(kept_share(rows.gain[m], flip[m]), 4),
                           "nodes": int(m.sum())}
                    for name, m in (("with", quadrants), ("without", ~quadrants))}
    worst = min(classes.items(), key=lambda kv: kv[1]) if classes else None
    return {"kept": round(kept_share(rows.gain, flip), 4), "flip_rate": round(float(flip.mean()), 4),
            "right_flip_rate": round(float((rows.gain > 0).mean()), 4), "nodes": len(rows.gain),
            "clips": len(set(rows.clip.tolist())), "by_rung": by_rung, "by_class": classes,
            "by_quadrant_inputs": by_quadrants,
            "worst_class": worst}


def git_commit() -> str:
    done = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return done.stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--version", required=True, help="nn dataset version (or a run directory)")
    p.add_argument("--codec", required=True)
    p.add_argument("--rungs", default="medium,fast,faster")
    p.add_argument("--seed", default="nn-fast-1")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--no-check", action="store_true", help="skip the manifest check")
    p.add_argument("--device", default="cuda")
    p.add_argument("--learned-tree", type=Path, default=DEFAULT_LEARNED_TREE,
                   help="nn-lead's eval/learned (quantise.py, intnet.py)")
    args = p.parse_args()
    from nceval import nndata

    path = Path(args.version)
    root = path if path.is_dir() else nndata.ROOT / args.version
    manifests = sorted(root.rglob("manifest.json"))
    manifest = hashlib.sha256(b"".join(m.read_bytes() for m in manifests)).hexdigest()
    if not path.is_dir() and not args.no_check and nndata.verify(root):
        raise SystemExit(f"{root}: shards differ from the manifest")
    rungs = set(args.rungs.split(","))
    paths = twin_dirs(root, args.codec, rungs)
    rows, clips = load_rows(nndata, paths, args.codec, rungs)
    sides = nndata.split(clips, args.seed)
    nndata.check_split(clips, sides)
    held = np.array([sides.get(c) == nndata.HELD_OUT for c in rows.clip.tolist()])
    fit, test = subset(rows, ~held), subset(rows, held)
    seed = int(hashlib.sha256(args.seed.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sample = rng.permutation(len(fit.gain))[:MAX_FIT_ROWS]
    net, intnet = train(subset(fit, np.sort(sample)), seed, args.device, args.learned_tree)
    fit_scores = intnet.forward(net, fit.x)
    threshold = best_threshold(fit_scores, fit.gain)
    margins = np.unique(np.quantile(fit.margin, np.linspace(0, 1, 101)))
    rule = float(margins[int(np.nanargmax([kept_share(fit.gain, fit.margin < m) for m in margins]))])
    result = {
        "script": "eval/distill/train_flip.py", "commit": git_commit(), "seed": args.seed,
        "dataset": {"version": args.version, "manifest_sha256": manifest,
                    "manifests": [str(m) for m in manifests], "shards": len(paths)},
        "codec": args.codec, "rungs": args.rungs.split(","),
        "network": {"sizes": list(net.sizes), "shifts": net.shifts,
                    "output_scale_log2": net.output_scale_log2, "threshold": threshold,
                    "macs_padded": net.multiply_accumulates(pad=True),
                    "file": str(args.out.with_suffix(".npz"))},
        "fit": report(fit, fit_scores > threshold),
        "heldout": report(test, intnet.forward(net, test.x) > threshold),
        "rule": {"margin_below": rule, "fit": report(fit, fit.margin < rule),
                 "heldout": report(test, test.margin < rule)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    intnet.save(args.out.with_suffix(".npz"), net)
    args.out.write_text(json.dumps(result, indent=1))
    h, r = result["heldout"], result["rule"]["heldout"]
    print(f"{args.codec} {args.rungs}: held-out {h['nodes']} nodes / {h['clips']} clips; "
          f"network keeps {h['kept']:.1%} at flip rate {h['flip_rate']:.1%} (right flips "
          f"{h['right_flip_rate']:.1%}); margin rule keeps {r['kept']:.1%}; by rung "
          f"{h['by_rung']}; worst class {h['worst_class']}")


if __name__ == "__main__":
    main()
