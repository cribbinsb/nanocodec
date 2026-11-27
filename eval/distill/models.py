"""The best model the data supports for the partition flip, before any cost limit
(docs/analysis/distill-fast-2026-11-22.md section 2), on distill.features tables.

    python3 -m distill.models --features DIR --seed nn-fast-1 --out RESULT.json \\
        [--models rule,trial_mlp,gbdt,wide_mlp,cnn] [--ablation]

Every model is a flip classifier weighted by the J at stake (|flip value|): flip where the
score clears a threshold chosen on the training side. Split: whole source groups
(nceval.nndata.split), the held-out side never seen by fitting or threshold choice. Score:
the share of the oracle's flip gain kept on the held-out side (the J saved by the flips
made, losses included, over the J every right flip would save), overall, per rung and per
content class. `--ablation` refits the boosted trees without each input group.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

GROUPS = ("trial", "node", "depth_context", "neighbours", "preanalysis", "motion", "frame",
          "rung", "codec")
#: Rows a neural model is fitted on (a uniform sample of the training side).
MAX_NN_ROWS = 1_500_000
MAX_CNN_ROWS = 400_000
#: Thresholds tried: these quantiles of the training scores.
THRESHOLD_QUANTILES = np.linspace(0.3, 0.999, 300)


def load(directory: Path) -> dict:
    tables, clips, patches, residuals = {g: [] for g in (*GROUPS, "label")}, [], [], []
    sb_offset, sb_rows, clip_index, rung, cls = 0, [], [], [], []
    for i, path in enumerate(sorted(directory.glob("*.npz"))):
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            n = len(z["label"])
            for g in (*GROUPS, "label"):
                tables[g].append(z[g])
            sb_rows.append(z["sb_row"] + sb_offset)
            patches.append(z["patch"])
            residuals.append(z["residual"])
            sb_offset += len(z["patch"])
        clips.append(meta["clip"])
        clip_index.append(np.full(n, i, np.int32))
        rung.append(np.full(n, meta["preset"], object))
        cls.append(np.full(n, meta["clip"].get("content_class", "?"), object))
    data = {g: np.concatenate(v) for g, v in tables.items()}
    data.update(sb_row=np.concatenate(sb_rows), patch=np.concatenate(patches),
                residual=np.concatenate(residuals), clip=np.concatenate(clip_index),
                rung_name=np.concatenate(rung), cls=np.concatenate(cls))
    return data, clips


def kept_share(gain: np.ndarray, flip: np.ndarray) -> float:
    best = gain[gain > 0].sum()
    return float(gain[flip].sum() / best) if best > 0 else float("nan")


def threshold(scores: np.ndarray, gain: np.ndarray) -> float:
    candidates = np.unique(np.quantile(scores, THRESHOLD_QUANTILES))
    kept = [kept_share(gain, scores > t) for t in candidates]
    return float(candidates[int(np.nanargmax(kept))])


#: Trial budgets of the gate form: shares of the nodes' area the search may code both ways.
GATE_AREAS = (0.05, 0.10, 0.20)


def gate_front(gain, scores, area) -> dict:
    """The model as a trial gate: the highest-scoring nodes, up to each share of the area,
    are coded both ways and take the better choice; the rest keep the plan."""
    order = np.argsort(-scores)
    cum = np.cumsum(area[order]) / area.sum()
    best = gain[gain > 0].sum()
    out = {}
    for share in GATE_AREAS:
        chosen = order[cum <= share]
        out[f"{share:.0%}"] = round(float(np.maximum(gain[chosen], 0).sum() / best), 4)
    return out


def report(gain, flip, rung, cls) -> dict:
    per = lambda keys: {k: round(kept_share(gain[keys == k], flip[keys == k]), 4)
                        for k in sorted(set(keys.tolist()))}
    classes = per(cls)
    return {"kept": round(kept_share(gain, flip), 4), "flip_rate": round(float(flip.mean()), 4),
            "by_rung": per(rung), "by_class": classes,
            "worst_class": min(classes.items(), key=lambda kv: kv[1])}


def scalars(data: dict, groups) -> np.ndarray:
    return np.concatenate([data[g] for g in groups], axis=1).astype(np.float32)


# --- models: each returns (fit scores, held-out scores) ---------------------------------

def gbdt(x_fit, y_fit, w_fit, x_test, seed):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=255,
                               min_child_samples=200, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=seed, verbose=-1)
    model.fit(x_fit, y_fit, sample_weight=w_fit)
    return model.predict_proba(x_fit)[:, 1], model.predict_proba(x_test)[:, 1]


def mlp(x_fit, y_fit, w_fit, x_test, seed, hidden, epochs=8):
    import torch
    torch.manual_seed(seed)
    mean, std = x_fit.mean(0), x_fit.std(0) + 1e-6
    layers, width = [], x_fit.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.ReLU()]
        width = h
    model = torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))
    _fit_torch(model, lambda idx: torch.tensor((x_fit[idx] - mean) / std),
                            len(x_fit), y_fit, w_fit, epochs, seed)
    with torch.no_grad():
        score = lambda x: torch.cat([model(torch.tensor((x[i:i + 65536] - mean) / std)).squeeze(1)
                                     for i in range(0, len(x), 65536)]).numpy()
        return score(x_fit), score(x_test)


def _fit_torch(model, batch_of, n, y, w, epochs, seed, batch=4096, lr=2e-3):
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    y_t, w_t = torch.tensor(y, dtype=torch.float32), torch.tensor(w / w.mean(), dtype=torch.float32)
    steps = epochs * (n // batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = order[s:s + batch].numpy()
            logit = model(batch_of(idx)).squeeze(-1)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logit, y_t[idx], reduction="none") * w_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()


def cnn(data, fit_idx, test_idx, x_all, y, w, seed, epochs=6):
    """Luma and residual patches of the node's superblock, a mask of the node, plus every
    scalar input."""
    import torch
    nn = torch.nn
    torch.manual_seed(seed)
    mean, std = x_all[fit_idx].mean(0), x_all[fit_idx].std(0) + 1e-6

    def images(rows):
        sbs = data["sb_row"][rows]
        luma = data["patch"][sbs].astype(np.float32) / 255.0
        res = data["residual"][sbs].astype(np.float32) / 64.0
        node = data["node"][rows]
        mask = np.zeros_like(luma)
        side = (2.0 ** node[:, 0]) / 2 ** 6 * 32  # node side in patch cells (64 -> 32)
        for k in range(len(rows)):
            x0, y0, s = int(node[k, 3] * 32), int(node[k, 4] * 32), max(1, int(side[k]))
            mask[k, y0:y0 + s, x0:x0 + s] = 1
        return torch.tensor(np.stack([luma, res, mask], 1))

    class Net(nn.Module):
        def __init__(self, scalars):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Linear(64 + scalars, 256), nn.ReLU(),
                                      nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, pair):
            img, sc = pair
            return self.head(torch.cat([self.conv(img).flatten(1), sc], 1))

    model = Net(x_all.shape[1])
    batch_of = lambda idx: (images(fit_idx[idx]),
                            torch.tensor((x_all[fit_idx[idx]] - mean) / std))
    _fit_torch(model, batch_of, len(fit_idx), y[fit_idx], w[fit_idx], epochs, seed, batch=1024)
    def score(rows):
        with torch.no_grad():
            return torch.cat([model((images(rows[i:i + 8192]),
                                     torch.tensor((x_all[rows[i:i + 8192]] - mean) / std))).squeeze(1)
                              for i in range(0, len(rows), 8192)]).numpy()
    return score, model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--seed", default="nn-fast-1")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--models", default="rule,trial_mlp,gbdt,wide_mlp,cnn")
    p.add_argument("--ablation", action="store_true")
    args = p.parse_args()
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    from nceval import nndata
    started = time.time()
    data, clips = load(args.features)
    print(f"loaded {len(data['label'])} rows from {len(clips)} shards "
          f"{time.time() - started:.0f}s", flush=True)
    sides = nndata.split(clips, args.seed)
    nndata.check_split(clips, sides)
    held = np.array([sides[c["name"]] == nndata.HELD_OUT for c in clips])[data["clip"]]
    gain = data["label"].astype(np.float64)
    y, w = (gain > 0).astype(np.float32), np.abs(gain).astype(np.float32)
    fit_idx, test_idx = np.nonzero(~held)[0], np.nonzero(held)[0]
    rng = np.random.default_rng(int.from_bytes(args.seed.encode()[:4], "little"))
    seed = int(rng.integers(1 << 30))
    result = {"script": "eval/distill/models.py", "seed": args.seed,
              "features": str(args.features), "rows": {"fit": len(fit_idx), "heldout": len(test_idx)},
              "clips": {"fit": int(sum(sides[c["name"]] != nndata.HELD_OUT for c in clips)),
                        "heldout": int(sum(sides[c["name"]] == nndata.HELD_OUT for c in clips))},
              "right_flip_rate": round(float(y[test_idx].mean()), 4), "models": {}}
    rung, cls = data["rung_name"], data["cls"]

    area = (2.0 ** (2 * data["node"][:, 0])).astype(np.float64)

    def record(name, fit_scores, test_scores, extra=None, fit_rows=None):
        fit_rows = fit_idx if fit_rows is None else fit_rows
        t = threshold(fit_scores, gain[fit_rows])
        extra = {**(extra or {}), "gate_heldout": gate_front(gain[test_idx], test_scores,
                                                             area[test_idx])}
        result["models"][name] = {"fit": report(gain[fit_rows], fit_scores > t, rung[fit_rows], cls[fit_rows]),
                                  "heldout": report(gain[test_idx], test_scores > t, rung[test_idx], cls[test_idx]),
                                  **extra}
        print(f"{name}: held-out kept {result['models'][name]['heldout']['kept']:.1%} "
              f"(fit {result['models'][name]['fit']['kept']:.1%}) "
              f"{time.time() - started:.0f}s", flush=True)
        args.out.write_text(json.dumps(result, indent=1))

    wanted = args.models.split(",")
    x_all = scalars(data, GROUPS)
    sample = lambda idx, k: np.sort(rng.permutation(idx)[:k])
    if "rule" in wanted:
        margin = data["node"][:, 2]  # log margin: flip where the planner nearly tied
        record("rule_margin", -margin[fit_idx], -margin[test_idx])
    if "trial_mlp" in wanted:
        s = sample(fit_idx, MAX_NN_ROWS)
        x = data["trial"]
        out = mlp(x[s], y[s], w[s], np.concatenate([x[fit_idx], x[test_idx]]), seed, (32, 8))
        record("trial_mlp_32_8", out[1][:len(fit_idx)], out[1][len(fit_idx):])
    if "gbdt" in wanted:
        s = sample(fit_idx, MAX_NN_ROWS)
        fs, ts = gbdt(x_all[s], y[s], w[s], np.concatenate([x_all[fit_idx], x_all[test_idx]]), seed)
        record("gbdt_all", ts[:len(fit_idx)], ts[len(fit_idx):])
        if args.ablation:
            for g in GROUPS:
                keep = [h for h in GROUPS if h != g]
                xa = scalars(data, keep)
                _, ts = gbdt(xa[s], y[s], w[s], np.concatenate([xa[fit_idx], xa[test_idx]]), seed)
                record(f"gbdt_without_{g}", ts[:len(fit_idx)], ts[len(fit_idx):])
            for g in GROUPS:
                xa = scalars(data, (g,))
                _, ts = gbdt(xa[s], y[s], w[s], np.concatenate([xa[fit_idx], xa[test_idx]]), seed)
                record(f"gbdt_only_{g}", ts[:len(fit_idx)], ts[len(fit_idx):])
    if "wide_mlp" in wanted:
        s = sample(fit_idx, MAX_NN_ROWS)
        out = mlp(x_all[s], y[s], w[s], np.concatenate([x_all[fit_idx], x_all[test_idx]]), seed,
                  (512, 256, 128))
        record("wide_mlp_512_256_128", out[1][:len(fit_idx)], out[1][len(fit_idx):])
    if "cnn" in wanted:
        s = sample(fit_idx, MAX_CNN_ROWS)
        score, _ = cnn(data, s, test_idx, x_all, y, w, seed)
        # The threshold from a training sample (scoring every row through the network
        # costs more than it tells).
        s2 = sample(fit_idx, MAX_CNN_ROWS)
        record("cnn_patches_plus_all", score(s2), score(test_idx), fit_rows=s2)


if __name__ == "__main__":
    main()
