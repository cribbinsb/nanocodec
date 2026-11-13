"""Surrogate model of a tune study's objectives, and a sampler that uses it.

Every finished trial of a study is a labelled example: parameters in, three
objectives (PSNR BD, VMAF BD, CPU ratio) and the class-guard constraint out.
A model fitted to them predicts an unmeasured setting in microseconds, with
an uncertainty, so a search can spend its expensive measurements where the
model expects the front to move.

Model (per objective and for the worst guard): the mean of a Gaussian
process (Matern 5/2, one length scale per parameter, white noise; it gives
the uncertainty) and gradient-boosted trees (robust to the step changes of
categorical and threshold parameters). Validation on the 2026-11-07 studies,
time split (fit on the first 60 % of trials, predict the rest), is in
docs/handoff/search.md; `python3 -m nceval.surrogate validate --study NAME`
repeats it on any study.

`SurrogateSampler` wraps Optuna's TPE sampler: every `every`-th trial (once
`MIN_TRAIN` trials have finished) takes the candidate with the best
optimistic random-weight Chebyshev score instead: draw a random weight w
over the three objectives, then minimise max_k w_k (mu_k - KAPPA sigma_k -
best_k) / range_k over random and front-neighbourhood candidates whose
optimistic guard is met. Other trials are TPE's, so the search never
depends on the model alone.
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import FrozenTrial, TrialState

#: Finished trials needed before the model may propose (fewer fit noise).
MIN_TRAIN = 20
#: Optimism of the acquisition, in predicted standard deviations.
KAPPA = 1.0
RANDOM_CANDIDATES = 3000
NEIGHBOUR_CANDIDATES = 1000
#: Relative step of a front-neighbourhood candidate, per unit of a domain.
NEIGHBOUR_STEP = 0.1
OBJECTIVES = 3


def _single_threaded() -> None:
    # The study shares the machine: a GP fit must not spread over every core.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


class Encoding:
    """Parameters <-> the unit cube; a categorical is placed by its numeric value."""

    def __init__(self, distributions: dict[str, optuna.distributions.BaseDistribution]):
        self.names = sorted(distributions)
        self.distributions = distributions
        self.low, self.high = [], []
        for name in self.names:
            d = distributions[name]
            values = d.choices if isinstance(d, CategoricalDistribution) else (d.low, d.high)
            self.low.append(min(values))
            self.high.append(max(values))
        self.low, self.high = np.array(self.low, float), np.array(self.high, float)

    def encode(self, params: dict[str, float]) -> np.ndarray:
        raw = np.array([params[n] for n in self.names], float)
        span = np.where(self.high > self.low, self.high - self.low, 1.0)
        return (raw - self.low) / span

    def decode(self, x: np.ndarray) -> dict[str, float]:
        out = {}
        for i, name in enumerate(self.names):
            value = self.low[i] + float(np.clip(x[i], 0, 1)) * (self.high[i] - self.low[i])
            d = self.distributions[name]
            if isinstance(d, CategoricalDistribution):
                value = min(d.choices, key=lambda c: abs(c - value))
            out[name] = value
        return out

    def random(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return np.array([self.encode(self.decode(x)) for x in rng.random((count, len(self.names)))])


class Model:
    """GP + boosted-trees mean, GP standard deviation, for one target."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        _single_threaded()
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        kernel = (ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(x.shape[1]),
                                            length_scale_bounds=(1e-2, 1e2))
                  + WhiteKernel(noise_level_bounds=(1e-6, 1e1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=0).fit(x, y)
        self.trees = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, random_state=0).fit(x, y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.gp.predict(x, return_std=True)
        return (mean + self.trees.predict(x)) / 2, std


def training_trials(trials: list[FrozenTrial]) -> list[FrozenTrial]:
    """Finished trials measured at the study's own fidelity (no early stops or copies)."""
    from . import ladder
    return [t for t in trials if t.state == TrialState.COMPLETE and t.values is not None
            and not t.user_attrs.get(ladder.PRUNED_ATTR)
            and ladder.DUPLICATE_ATTR not in t.user_attrs]


def fit(trials: list[FrozenTrial], encoding: Encoding) -> tuple[list[Model], Model]:
    x = np.array([encoding.encode(t.params) for t in trials])
    ys = np.array([t.values for t in trials])
    guard = np.array([max(t.user_attrs.get("constraint", [0.0])) for t in trials])
    return [Model(x, ys[:, k]) for k in range(OBJECTIVES)], Model(x, guard)


def propose(trials: list[FrozenTrial], encoding: Encoding, rng: np.random.Generator
            ) -> dict[str, float] | None:
    """The candidate with the best optimistic random-weight Chebyshev score."""
    usable = [t for t in training_trials(trials) if set(t.params) == set(encoding.names)]
    if len(usable) < MIN_TRAIN:
        return None
    models, guard_model = fit(usable, encoding)
    observed = np.array([t.values for t in usable])
    best, span = observed.min(axis=0), np.ptp(observed, axis=0) + 1e-9
    feasible = [t for t in usable if max(t.user_attrs.get("constraint", [0.0])) <= 0]
    seeds = np.array([encoding.encode(t.params) for t in (feasible or usable)])
    steps = rng.normal(0, NEIGHBOUR_STEP, (NEIGHBOUR_CANDIDATES, len(encoding.names)))
    near = seeds[rng.integers(len(seeds), size=NEIGHBOUR_CANDIDATES)] + steps
    candidates = np.vstack([encoding.random(rng, RANDOM_CANDIDATES),
                            [encoding.encode(encoding.decode(x)) for x in near]])
    guard_mean, guard_std = guard_model.predict(candidates)
    allowed = guard_mean - KAPPA * guard_std <= 0
    if not allowed.any():
        return None
    weight = rng.dirichlet(np.ones(OBJECTIVES))
    optimistic = np.column_stack([m - KAPPA * s for m, s in (model.predict(candidates)
                                                             for model in models)])
    score = np.max(weight * (optimistic - best) / span, axis=1)
    score[~allowed] = np.inf
    return encoding.decode(candidates[int(np.argmin(score))])


class SurrogateSampler(BaseSampler):
    """TPE, except that every `every`-th trial is the surrogate's proposal."""

    def __init__(self, tpe: TPESampler, every: int, seed: int | None = None):
        self._tpe = tpe
        self._every = every
        self._rng = np.random.default_rng(seed)
        self.proposed: set[int] = set()  # trial numbers the surrogate chose

    def _surrogate_turn(self, trial) -> bool:
        return trial.number % self._every == self._every - 1

    def infer_relative_search_space(self, study, trial):
        if self._surrogate_turn(trial):
            return optuna.search_space.intersection_search_space(
                study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        return self._tpe.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        if not self._surrogate_turn(trial):
            return self._tpe.sample_relative(study, trial, search_space)
        if search_space:
            proposal = propose(study.get_trials(deepcopy=False), Encoding(search_space), self._rng)
            if proposal is not None:
                self.proposed.add(trial.number)
                return proposal
        # No model yet: this parameter set falls to TPE's independent sampling.
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._tpe.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study, trial):
        self._tpe.before_trial(study, trial)

    def after_trial(self, study, trial, state, values):
        self._tpe.after_trial(study, trial, state, values)


# ------------------------------------------------------------------ validation


def validate(trials: list[FrozenTrial], split: float = 0.6) -> list[dict[str, float]]:
    """Time-split prediction error per objective: fit on the first trials, predict the rest."""
    from scipy.stats import spearmanr
    usable = training_trials(trials)
    usable.sort(key=lambda t: t.number)
    distributions = usable[0].distributions
    encoding = Encoding(distributions)
    usable = [t for t in usable if set(t.params) == set(encoding.names)]
    cut = int(len(usable) * split)
    train, test = usable[:cut], usable[cut:]
    if len(train) < 5 or len(test) < 5:
        raise SystemExit(f"too few finished trials to validate ({len(usable)})")
    models, _ = fit(train, encoding)
    x = np.array([encoding.encode(t.params) for t in test])
    truth = np.array([t.values for t in test])
    out = []
    for k, model in enumerate(models):
        mean, std = model.predict(x)
        error = mean - truth[:, k]
        out.append({"objective": k, "spearman": float(spearmanr(mean, truth[:, k])[0]),
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "spread": float(np.std(truth[:, k])),
                    "within_2sd": float(np.mean(np.abs(error) <= 2 * std)),
                    "train": len(train), "test": len(test)})
    return out


def main() -> None:
    from .tune import study_storage
    parser = argparse.ArgumentParser(prog="nceval.surrogate", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="time-split prediction error on a finished study")
    v.add_argument("--study", required=True)
    v.add_argument("--split", type=float, default=0.6)
    args = parser.parse_args()
    study = optuna.load_study(study_name=args.study, storage=study_storage(args.study))
    for row in validate(study.trials, args.split):
        print(f"objective {row['objective']}: spearman {row['spearman']:.3f}  rmse "
              f"{row['rmse']:.3f} (spread {row['spread']:.3f})  within 2 sd "
              f"{row['within_2sd']:.2f}  [{row['train']} fit, {row['test']} held out]")


if __name__ == "__main__":
    main()
