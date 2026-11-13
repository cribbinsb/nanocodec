"""The surrogate learns a study's objectives, proposes inside the space, and warm-starts."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import ladder, surrogate, tune

DISTRIBUTIONS = {"a": optuna.distributions.FloatDistribution(0.0, 4.0),
                 "b": optuna.distributions.CategoricalDistribution([0, 1, 2])}


def objectives(a, b):
    """A smooth trade-off: quality improves with a, CPU rises with a; b=2 costs quality."""
    return ((a - 3) ** 2 + 2 * (b == 2), (a - 2.5) ** 2, 1 + 0.2 * a)


def synthetic_trials(count, seed=0):
    rng = np.random.default_rng(seed)
    trials = []
    for number in range(count):
        params = {"a": float(rng.uniform(0, 4)), "b": int(rng.integers(3))}
        t = optuna.trial.create_trial(params=params, distributions=DISTRIBUTIONS,
                                      values=objectives(**params),
                                      user_attrs={"constraint": [params["a"] - 3.8, -1.0]})
        t.number = number
        trials.append(t)
    return trials


class SurrogateModel(unittest.TestCase):
    def test_validation_predicts_held_out_trials(self):
        rows = surrogate.validate(synthetic_trials(40))
        self.assertTrue(all(r["spearman"] > 0.9 for r in rows), rows)

    def test_proposal_is_inside_the_space_and_meets_the_guard(self):
        encoding = surrogate.Encoding(DISTRIBUTIONS)
        proposal = surrogate.propose(synthetic_trials(30), encoding, np.random.default_rng(1))
        self.assertIn(proposal["b"], (0, 1, 2))
        self.assertTrue(0 <= proposal["a"] <= 3.8 + 0.3)

    def test_too_few_or_early_stopped_trials_give_no_proposal(self):
        encoding = surrogate.Encoding(DISTRIBUTIONS)
        trials = synthetic_trials(surrogate.MIN_TRAIN)
        for t in trials[:2]:
            t.set_user_attr(ladder.PRUNED_ATTR, "30fx3")
        self.assertIsNone(surrogate.propose(trials, encoding, np.random.default_rng(1)))

    def test_sampler_mixes_tpe_and_proposals(self):
        sampler = surrogate.SurrogateSampler(optuna.samplers.TPESampler(seed=1), every=2, seed=1)
        study = optuna.create_study(directions=["minimize"] * 3, sampler=sampler)
        for t in synthetic_trials(surrogate.MIN_TRAIN):
            study.add_trial(t)
        study.optimize(lambda trial: objectives(trial.suggest_float("a", 0.0, 4.0),
                                                trial.suggest_categorical("b", [0, 1, 2])),
                       n_trials=4)
        self.assertTrue(sampler.proposed)
        self.assertTrue(all(n % 2 == 1 for n in sampler.proposed))


class WarmStart(unittest.TestCase):
    def test_copies_finished_trials_marked_and_kept_off_the_front(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(tune, "TUNING_DIR", Path(tmp)):
            old = optuna.create_study(study_name="old", storage=tune.study_storage("old"),
                                      directions=["minimize"] * 3)
            for t in synthetic_trials(6):
                old.add_trial(t)
            pruned = synthetic_trials(1, seed=9)[0]
            pruned.set_user_attr(ladder.PRUNED_ATTR, "30fx3")
            old.add_trial(pruned)
            new = optuna.create_study(study_name="new", storage=tune.study_storage("new"),
                                      directions=["minimize"] * 3)
            domains = {"a": {"low": 0.0, "high": 4.0, "log": False}, "b": {"choices": [0, 1, 2]},
                       "c": {"choices": [0, 1]}}
            specs = {"a": (1.0, 0.0, 4.0), "b": (0, 0, 2), "c": (1, 0, 1)}
            added = tune.warm_start(new, ["old"], ["a", "b", "c"], specs, domains)
            self.assertEqual(added, 6)
            self.assertTrue(all(t.params["c"] == 1 for t in new.trials))
            self.assertTrue(all(t.user_attrs[tune.WARM_START_ATTR].startswith("old#")
                                for t in new.trials))
            self.assertEqual(tune.feasible_front(new.trials), [])


if __name__ == "__main__":
    unittest.main()
