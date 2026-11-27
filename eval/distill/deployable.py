"""The deployable flip guide: one small decision tree per rung, on inputs the fast rung's
planner holds at the node, distilled for the closed-loop encoder A/B
(docs/analysis/distill-fast-2026-11-22.md section 2.3).

    python3 -m distill.deployable --features DIR --rung fast --seed nn-fast-1 \\
        --depths 6,8 --out TREE.json

Inputs (DEPLOYABLE): the eleven trial bytes (`trial_inputs.rs`: they already hold the node
size, the planned choice and the planner's margin), the node's position in its superblock,
and the frame's mode-decision lambda (`CodecBackend::lambda`). Nothing the fast rung does
not compute, so train and serve see the same values (plain-context feature tables).

Each depth is fitted as a classifier weighted by the J at stake, its leaves' flip decision
set by the threshold that keeps the most training gain; held-out source groups score it,
beside boosted trees on the same inputs and on every input (the ceiling of these inputs).
The tree is written with its thresholds on the raw values (lambda, not its log), so the
encoder compares exactly what training compared.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from distill.models import GROUPS, gbdt, kept_share, load, report, scalars, threshold

#: (name, table, column) of each deployable input, in the encoder's order.
DEPLOYABLE = tuple((f"trial{i}", "trial", i) for i in range(11)) + (
    ("x_in_sb", "node", 3), ("y_in_sb", "node", 4), ("log_lambda", "frame", 1))
LAMBDA_INPUT = len(DEPLOYABLE) - 1
MAX_ROWS = 3_000_000
MIN_LEAF_SHARE = 1e-4


def deployable_inputs(data: dict) -> np.ndarray:
    return np.stack([data[table][:, col] for _, table, col in DEPLOYABLE], 1).astype(np.float32)


def export_tree(tree, flip_leaf: np.ndarray) -> dict:
    """sklearn's tree as nodes: inner (input, threshold, left, right), leaves (flip)."""
    t = tree.tree_
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            nodes.append({"leaf": bool(flip_leaf[i])})
        else:
            feature, value = int(t.feature[i]), float(t.threshold[i])
            if feature == LAMBDA_INPUT:
                value = float(np.exp(value))  # compare lambda itself in the encoder
            nodes.append({"input": feature, "threshold": value,
                          "left": int(t.children_left[i]), "right": int(t.children_right[i])})
    return {"inputs": [n for n, _, _ in DEPLOYABLE], "lambda_input": LAMBDA_INPUT,
            "rule": "go left when input <= threshold", "nodes": nodes}


def main() -> None:
    from sklearn.tree import DecisionTreeClassifier
    from nceval import nndata
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--rung", required=True)
    p.add_argument("--seed", default="nn-fast-1")
    p.add_argument("--depths", default="6,8")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    data, clips = load(args.features)
    sides = nndata.split(clips, args.seed)
    nndata.check_split(clips, sides)
    held = np.array([sides[c["name"]] == nndata.HELD_OUT for c in clips])[data["clip"]]
    rung_rows = data["rung_name"] == args.rung
    gain = data["label"].astype(np.float64)
    y, w = (gain > 0).astype(np.int8), np.abs(gain)
    fit_idx, test_idx = np.nonzero(~held & rung_rows)[0], np.nonzero(held & rung_rows)[0]
    rng = np.random.default_rng(1)
    s = np.sort(rng.permutation(fit_idx)[:MAX_ROWS])
    x = deployable_inputs(data)
    rung, cls = data["rung_name"], data["cls"]
    result = {"script": "eval/distill/deployable.py", "seed": args.seed, "rung": args.rung,
              "features": str(args.features),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True, cwd=Path(__file__).parent).stdout.strip(),
              "rows": {"fit": len(fit_idx), "heldout": len(test_idx)}, "models": {}}

    def record(name, fit_scores, test_scores, extra=None):
        t = threshold(fit_scores, gain[fit_idx])
        result["models"][name] = {
            "threshold": t,
            "fit": report(gain[fit_idx], fit_scores > t, rung[fit_idx], cls[fit_idx]),
            "heldout": report(gain[test_idx], test_scores > t, rung[test_idx], cls[test_idx]),
            **(extra or {})}
        print(f"{name}: held-out kept {result['models'][name]['heldout']['kept']:.1%}",
              flush=True)
        return t

    for name, xs in (("gbdt_every_input", scalars(data, GROUPS)), ("gbdt_deployable", x)):
        _, ts = gbdt(xs[s], y[s], w[s], np.concatenate([xs[fit_idx], xs[test_idx]]), 1)
        record(name, ts[:len(fit_idx)], ts[len(fit_idx):])
    for depth in map(int, args.depths.split(",")):
        tree = DecisionTreeClassifier(max_depth=depth, min_weight_fraction_leaf=MIN_LEAF_SHARE,
                                      random_state=1)
        tree.fit(x[s], y[s], sample_weight=w[s])
        fit_p, test_p = tree.predict_proba(x[fit_idx])[:, 1], tree.predict_proba(x[test_idx])[:, 1]
        t = record(f"tree_depth_{depth}", fit_p, test_p)
        leaf_p = tree.tree_.value[:, 0, 1] / tree.tree_.value[:, 0].sum(1)
        result["models"][f"tree_depth_{depth}"]["tree"] = export_tree(tree, leaf_p > t)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
