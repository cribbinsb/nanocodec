"""How accurate the inter planner's leaf-vs-split estimate is against the coded costs, and how
much a learned correction of it gains (docs/analysis/learned-estimator-2026-11-22.md).

    python3 -m estimator.accuracy NODES.npz [--json OUT] [--seed nn-estimator-1]

Input: estimator.extract's table. The correction keeps today's estimate as its degenerate
case: the corrected log ratio is `log(leaf / split)_estimated + f(features)`, with f a
gradient-boosted regression of the estimate's error, starting at zero. It is fitted on the
training side of a split by whole source groups (nceval.nndata.split) and reported on the
held-out side.

Measures, per codec / rung / node size:
- per candidate: log(coded / estimated), its median (bias) and robust spread (IQR / 1.349),
  the leaf on every node, the split only where the plan split (a planned leaf's split
  estimate is the early exit's partial sum, a lower bound);
- the decision: agreement of the estimated and coded leaf-vs-split order (ties dropped), its
  Spearman correlation, and the regret, the J the estimate's choice loses against the
  cheaper coded candidate, as a share of the cheaper candidate's J summed over nodes.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from estimator.extract import BASE_FEATURES, CODECS, FEATURES, PRESETS
from nceval import nndata

#: Coded costs closer than this (relative) count as a tie and leave the order undecided.
TIE = 1e-6
#: The estimate's error is clipped here before fitting (log units; e^3 = 20x).
CLIP = 3.0
#: Inputs that depend on the split side's estimate, known only after the node's children are
#: planned: a correction without them shifts the parent bias before the children, so the
#: planner's exact early exit stays exact (split_selector.rs).
AFTER_CHILDREN = ("est_log_ratio", "split_est_per_sample", "log_margin",
                  *(f"input{i}" for i in range(11)))
#: What the planner itself holds at the node when it compares (the eleven trial bytes are
#: computed from its SplitEvidence, margin and choice): a correction from these alone needs no
#: new measurement in the encoder.
PLANNER_HELD = ("log2", "est_log_ratio", "leaf_est_per_sample", "split_est_per_sample",
                "log_margin", "log_lambda", *(f"input{i}" for i in range(11)))
#: Rungs as the tables group them: medium's search twin (its inter frames with the RD
#: search on), medium as shipped (VVC searches its inter frames), slow.
RUNGS = ("medium twin", "medium", "slow")


def rung_of(data: dict) -> np.ndarray:
    """Per row, its index in RUNGS."""
    slow = data["job"][:, 1] == PRESETS.index("slow")
    return np.where(slow, 2, np.where(data["twin"], 0, 1))


def per_candidate(log_err: np.ndarray) -> dict:
    """Bias and robust spread of log(coded / estimated)."""
    if len(log_err) == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(log_err, [25, 50, 75])
    return {"n": int(len(log_err)), "bias": float(q50), "spread": float((q75 - q25) / 1.349),
            "mean": float(log_err.mean())}


def decision(pred: np.ndarray, cost0: np.ndarray, cost1: np.ndarray) -> dict:
    """Agreement, rank correlation and regret of choosing the leaf where `pred` <= 0."""
    true = np.log(cost0 / cost1)
    decided = np.abs(true) > TIE
    leaf = pred <= 0
    picked = np.where(leaf, cost0, cost1)
    best = np.minimum(cost0, cost1)
    regret = picked - best
    return {
        "n": int(len(pred)),
        "agree": float(np.mean(leaf[decided] == (true[decided] <= 0))),
        "spearman": float(spearmanr(pred, true).statistic) if len(pred) > 2 else float("nan"),
        "regret_pct": float(100 * regret.sum() / best.sum()),
        "true_leaf_share": float(np.mean(true[decided] <= 0)),
        "pred_leaf_share": float(np.mean(leaf)),
    }


def load(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    out["names"] = list(out["feature_names"])
    out["label"] = {n: out["labels"][:, i] for i, n in enumerate(out["label_names"])}
    return out


def sides(data: dict, seed: str) -> np.ndarray:
    """True for held-out rows (whole source groups)."""
    metas = [ast.literal_eval(m) for m in data["clip_meta"]]
    side = nndata.split(metas, seed)
    held = np.array([side[m["name"]] == nndata.HELD_OUT for m in metas])
    return held[data["clip"]]


def baseline_tables(data: dict, rows: np.ndarray) -> list[dict]:
    """Per codec, rung (twin = medium's inter search) and size: today's estimator."""
    f = data["features"]
    col = {n: i for i, n in enumerate(data["names"])}
    lab = data["label"]
    log2 = f[:, col["log2"]].astype(int)
    area_lam = np.exp(2 * np.log(2) * log2) * lab["lambda"]
    leaf_err = np.log(lab["cost0"] / area_lam) - f[:, col["leaf_est_per_sample"]]
    split_err = np.log(lab["cost1"] / area_lam) - f[:, col["split_est_per_sample"]]
    planned_split = lab["planned"] == 1
    out = []
    job, rung_index = data["job"], rung_of(data)
    for c, codec in enumerate(CODECS):
        for p, rung in enumerate(RUNGS):
            for size in (None, 5, 6):
                m = rows & (job[:, 0] == c) & (rung_index == p)
                if size is not None:
                    m &= log2 == size
                if m.sum() < 100:
                    continue
                out.append({
                    "codec": codec, "rung": rung, "log2": size or "all",
                    "leaf": per_candidate(leaf_err[m]),
                    "split": per_candidate(split_err[m & planned_split]),
                    "decision": decision(f[m, col["est_log_ratio"]], lab["cost0"][m],
                                         lab["cost1"][m]),
                })
    return out


def fit_correction(data: dict, train: np.ndarray, names: tuple[str, ...],
                   small: bool = False) -> HistGradientBoostingRegressor:
    """f: the estimate's log-ratio error from `names`, starting from zero."""
    f = data["features"]
    col = [data["names"].index(n) for n in names]
    lab = data["label"]
    x = f[:, data["names"].index("est_log_ratio")]
    target = np.clip(np.log(lab["cost0"] / lab["cost1"]) - x, -CLIP, CLIP)
    size = {"max_iter": 64, "max_depth": 4} if small else {"max_iter": 300,
                                                            "max_leaf_nodes": 63}
    model = HistGradientBoostingRegressor(learning_rate=0.1, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0, **size)
    model.fit(f[train][:, col], target[train])
    return model


class DecisionCorrection:
    """f fitted for the decision itself: a logistic model of "the leaf codes cheaper" whose
    score starts at today's, `-k * log ratio` (k fitted alone first), with a correction from
    zero, each node weighted by what a wrong order there costs (|J leaf - J split| / lambda, in
    bits). The correction is boosted trees (LightGBM, the box's venv; MIT licence) or, with
    `linear`, an affine function of the inputs. `predict` returns f in log-ratio units, so
    `x + f <= 0` keeps the leaf."""

    def __init__(self, data: dict, train: np.ndarray, names: tuple[str, ...],
                 trees: int = 400, depth: int = -1, linear: bool = False):
        f = data["features"]
        lab = data["label"]
        x = f[train, data["names"].index("est_log_ratio")].astype(np.float64)
        leaf_wins = (lab["cost0"][train] <= lab["cost1"][train]).astype(np.float64)
        gap = np.abs(lab["cost0"][train] - lab["cost1"][train]) / lab["lambda"][train]
        weight = np.minimum(gap, np.percentile(gap, 99.9))
        self.k = self._slope(x, leaf_wins, weight)
        self.names = names
        self.col = [data["names"].index(n) for n in names]
        inputs = f[train][:, self.col].astype(np.float64)
        if linear:
            self.model = None
            self.weights = self._linear(inputs, leaf_wins, weight, -self.k * x)
            return
        import lightgbm

        leaves = 2**depth if depth > 0 else 63
        self.model = lightgbm.LGBMClassifier(n_estimators=trees, learning_rate=0.05,
                                             num_leaves=leaves, max_depth=depth,
                                             min_child_samples=200, reg_lambda=1.0,
                                             subsample=0.5, subsample_freq=1, verbose=-1)
        self.model.fit(inputs, leaf_wins, sample_weight=weight, init_score=-self.k * x)

    @staticmethod
    def _slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """The k of P(leaf) = sigmoid(-k x) minimising the weighted log loss."""
        from scipy.optimize import minimize_scalar

        def loss(k: float) -> float:
            # -log sigmoid(z) = logaddexp(0, -z), with z = -k x for the leaf, k x for the split.
            z = np.where(y > 0, -k * x, k * x)
            return float(np.sum(w * np.logaddexp(0, -z)))

        return float(minimize_scalar(loss, bounds=(0.05, 50), method="bounded").x)

    @staticmethod
    def _linear(z: np.ndarray, y: np.ndarray, w: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Intercept and weights of the score's affine correction (raw input units)."""
        from scipy.optimize import minimize

        mean, std = z.mean(0), z.std(0) + 1e-9
        zs = np.column_stack([np.ones(len(z)), (z - mean) / std])
        sign = np.where(y > 0, 1.0, -1.0)

        def loss(beta: np.ndarray) -> tuple[float, np.ndarray]:
            score = sign * (offset + zs @ beta)
            p = np.exp(-np.logaddexp(0, score))  # sigmoid(-score)
            return float(np.sum(w * np.logaddexp(0, -score))), -(zs.T @ (w * p * sign))

        beta = minimize(loss, np.zeros(zs.shape[1]), jac=True, method="L-BFGS-B").x
        raw = beta[1:] / std
        return np.concatenate([[beta[0] - np.sum(raw * mean)], raw])

    def score(self, features: np.ndarray) -> np.ndarray:
        z = features[:, self.col]
        if self.model is None:
            return self.weights[0] + z.astype(np.float64) @ self.weights[1:]
        return self.model.predict(z, raw_score=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return -self.score(features) / self.k

    def coefficients(self) -> dict:
        """The linear correction in log-ratio units: f = intercept + sum(w * input)."""
        f = -self.weights / self.k
        return {"intercept": float(f[0]), **{n: float(v) for n, v in zip(self.names, f[1:])}}


def corrected(data: dict, model, names: tuple[str, ...], rows: np.ndarray) -> np.ndarray:
    f = data["features"]
    x = f[rows, data["names"].index("est_log_ratio")]
    if isinstance(model, DecisionCorrection):
        return x + model.predict(f[rows])
    col = [data["names"].index(n) for n in names]
    return x + model.predict(f[rows][:, col])


def compare(data: dict, held: np.ndarray, models: dict) -> list[dict]:
    """Held-out decision metrics of today's estimate and each correction, per codec / rung."""
    lab, job, rung_index = data["label"], data["job"], rung_of(data)
    f = data["features"]
    log2 = f[:, data["names"].index("log2")].astype(int)
    out = []
    for c, codec in enumerate(CODECS):
        for p, rung in enumerate(RUNGS):
            for size in (None, 5, 6):
                m = held & (job[:, 0] == c) & (rung_index == p)
                if size is not None:
                    m &= log2 == size
                if m.sum() < 100:
                    continue
                row = {"codec": codec, "rung": rung, "log2": size or "all",
                       "today": decision(f[m, data["names"].index("est_log_ratio")],
                                         lab["cost0"][m], lab["cost1"][m])}
                for name, (model, names) in models.items():
                    row[name] = decision(corrected(data, model, names, m), lab["cost0"][m],
                                         lab["cost1"][m])
                out.append(row)
    return out


def importance(data: dict, model, names: tuple[str, ...], held: np.ndarray) -> dict:
    """Held-out permutation importance (drop in R^2 of the error) on a 200k-row sample."""
    from sklearn.inspection import permutation_importance

    rows = np.flatnonzero(held)
    rows = np.random.default_rng(0).choice(rows, min(len(rows), 200_000), replace=False)
    f = data["features"][rows]
    lab = data["label"]
    x = f[:, data["names"].index("est_log_ratio")]
    target = np.clip(np.log(lab["cost0"][rows] / lab["cost1"][rows]) - x, -CLIP, CLIP)
    col = [data["names"].index(n) for n in names]
    result = permutation_importance(model, f[:, col], target, n_repeats=2, random_state=0)
    order = np.argsort(-result.importances_mean)
    return {names[i]: float(result.importances_mean[i]) for i in order[:15]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("nodes", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--seed", default="nn-estimator-1")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--decision", action="store_true",
                        help="also the decision-loss corrections (needs lightgbm)")
    args = parser.parse_args()
    data = load(args.nodes)
    held = sides(data, args.seed)
    every = np.ones(len(held), bool)
    report = {"rows": int(len(held)), "held_out_rows": int(held.sum()),
              "baseline": baseline_tables(data, every)}
    print(json.dumps(report["baseline"][:3], indent=1), flush=True)
    if not args.baseline_only:
        early = tuple(n for n in FEATURES if n not in AFTER_CHILDREN)
        sets = {"base": BASE_FEATURES, "context": FEATURES, "planner_held": PLANNER_HELD,
                "early_context": early}
        models = {name: (fit_correction(data, ~held, names), names)
                  for name, names in sets.items()}
        models["planner_held_small"] = (fit_correction(data, ~held, PLANNER_HELD, small=True),
                                        PLANNER_HELD)
        if args.decision:
            for name in ("context", "early_context"):
                models[f"decision_{name}"] = (DecisionCorrection(data, ~held, sets[name]),
                                              sets[name])
        report["importance"] = importance(data, models["base"][0], BASE_FEATURES, held)
        report["held_out"] = compare(data, held, models)
    text = json.dumps(report, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text[-6000:])


if __name__ == "__main__":
    main()
