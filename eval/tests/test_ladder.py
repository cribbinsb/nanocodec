"""Multi-fidelity tune trials: rung parsing, dominance pruning and identity dedup."""

import argparse
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import ladder, tune, tune_realuse

SIGMAS = {"30fx3": (0.5, 0.01)}


class RungRules(unittest.TestCase):
    def test_parse_requires_measured_sigmas_and_the_study_fidelity_last(self):
        rungs = ladder.parse_ladder("30x3,60x4", 60, SIGMAS)
        self.assertEqual([r.label for r in rungs], ["30fx3", "60fx4"])
        self.assertEqual(rungs[0].qp_indices, (0, 1, 3))
        self.assertEqual((rungs[0].sigma, rungs[1].sigma), (0.5, 0.0))
        with self.assertRaisesRegex(ValueError, "no measured error"):
            ladder.parse_ladder("15x3,60x4", 60, SIGMAS)
        with self.assertRaisesRegex(ValueError, "last rung"):
            ladder.parse_ladder("30x3", 60, SIGMAS)
        with self.assertRaisesRegex(ValueError, "bad rung"):
            ladder.parse_ladder("30x2,60x4", 60, SIGMAS)

    def test_dominated_only_beyond_the_margin_in_every_objective(self):
        rung = ladder.Rung(30, (0, 1, 3), sigma=0.5, sigma_cpu=0.01)
        front = [(0.0, 0.0, 1.0)]
        self.assertTrue(ladder.clearly_dominated((2.0, 2.0, 1.05), front, rung))
        # Within the quality margin on one metric: may still be on the front.
        self.assertFalse(ladder.clearly_dominated((0.9, 2.0, 1.05), front, rung))
        # Cheaper: a speed point, never pruned for losing quality.
        self.assertFalse(ladder.clearly_dominated((5.0, 5.0, 0.9), front, rung))
        self.assertFalse(ladder.clearly_dominated((5.0, 5.0, 1.05), [], rung))

    def test_infeasible_only_beyond_the_margin(self):
        rung = ladder.Rung(30, (0, 1, 3), sigma=0.5)
        self.assertFalse(ladder.clearly_infeasible([0.9, -1.0], rung))
        self.assertTrue(ladder.clearly_infeasible([1.1, -1.0], rung))

    def test_stream_key_ignores_order_but_not_content(self):
        a = ladder.stream_key("bin", {"c1": [(30, "x"), (40, "y")], "c2": [(30, "z")]})
        b = ladder.stream_key("bin", {"c2": [(30, "z")], "c1": [(40, "y"), (30, "x")]})
        self.assertEqual(a, b)
        self.assertNotEqual(a, ladder.stream_key("bin", {"c1": [(30, "x"), (40, "q")],
                                                         "c2": [(30, "z")]}))
        self.assertNotEqual(a, ladder.stream_key("other", {"c1": [(30, "x"), (40, "y")],
                                                           "c2": [(30, "z")]}))


CLIPS = [SimpleNamespace(name=name, content_class=name.split("_")[0], clean_reference=None)
         for name in ("bodycam_bc1", "crowd_mot17_09", "lowlight_a", "pan_a", "static_cam22")]


def fake_evaluate(calls):
    """Curves whose rate grows 10 % per unit of `x` (x=0 is the default), cost likewise.

    Identical `x` gives identical streams, so repeats are duplicates.
    """
    def evaluate(binary, codec, overrides, work, quantizers=None, clips=None, frames=60,
                 fixed_grid=False, cost=None, cost_quantizers=None, **_):
        x = overrides.get("x", 0.0)
        calls.append((frames, x))
        if cost is None and overrides:
            # Costed at the anchor grid's third QP, not the rung subset's.
            assert cost_quantizers == {name: 30 for name in tune.COST_CLIPS}, cost_quantizers
        curves = {}
        for clip in CLIPS:
            qs = quantizers[clip.name] if quantizers else [20, 25, 30, 35]
            curves[clip.name] = tune.Curve(
                kbps=[1000 / 2 ** (i / 2) * (1 + 0.1 * x) for i, _ in enumerate(qs)],
                psnr=[40 - i for i, _ in enumerate(qs)], quantizers=list(qs),
                vmaf_v1=[90 - 5 * i for i, _ in enumerate(qs)],
                stream_sha256=[hashlib.sha256(f"{x}{clip.name}{q}{frames}".encode()).hexdigest()
                               for q in qs])
        p = 1.0 + 0.1 * x if cost is None else cost.p
        return tune.Evaluation(curves, p, 10.0)
    return evaluate


class LadderSearch(unittest.TestCase):
    def test_dominated_trials_stop_at_rung_one_and_repeats_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "nanocodec"
            binary.write_bytes(b"build")
            space = root / "space.json"
            space.write_text('{"x": {"choices": [0, 1, 2]}}')
            args = argparse.Namespace(codec="av1", binary=str(binary), params="x",
                                      search_space=str(space), study="ladder-test", seed=3,
                                      trials=9, workers=1, ladder="30x3,60x4", frames=60)
            calls = []
            with patch.object(tune, "TUNING_DIR", root), \
                 patch.object(tune, "specs_of", return_value={"x": (0, 0, 2)}), \
                 patch.object(tune, "proxy_clips", return_value=CLIPS), \
                 patch.object(tune, "clip_fingerprint", side_effect=lambda c: c.name), \
                 patch.object(tune, "evaluate", side_effect=fake_evaluate(calls)), \
                 patch.object(ladder, "RUNG_SIGMAS", SIGMAS), \
                 patch.object(tune, "print_front"):
                tune.search(args)
                study = tune.optuna.load_study(study_name="ladder-test",
                                               storage=tune.storage_of("ladder-test"))
            trials = [t for t in study.trials if t.state.name == "COMPLETE"]
            self.assertEqual(len(trials), 9)
            # Only the default reaches the last rung: x > 0 costs more and loses
            # quality by far more than the 30-frame margin.
            final = [t for t in trials if not t.user_attrs.get(ladder.PRUNED_ATTR)]
            self.assertEqual({t.params["x"] for t in final}, {0})
            # Calls: the anchor at 60 frames, its 30-frame rung, then the trials.
            self.assertEqual(calls[:2], [(60, 0.0), (30, 0.0)])
            trial_calls = calls[2:]
            self.assertEqual(sum(frames == 30 for frames, _ in trial_calls), 9)
            # A repeat of an earlier x is a duplicate: its rung-one streams
            # match, so it never measures the last rung again.
            distinct = {t.params["x"] for t in trials}
            duplicates = [t for t in trials if ladder.DUPLICATE_ATTR in t.user_attrs]
            self.assertEqual(len(duplicates), 9 - len(distinct))
            self.assertEqual([c for c in trial_calls if c[0] == 60], [(60, 0.0)])
            for t in duplicates:
                twin = study.trials[t.user_attrs[ladder.DUPLICATE_ATTR]]
                self.assertEqual(t.values, twin.values)
                self.assertEqual(t.params, twin.params)
            self.assertEqual([t.number for t in tune.feasible_front(study.trials)],
                             [min(t.number for t in final)])
            summary = tune.search_stats(study.trials)
            self.assertEqual(summary["trials"], 9)
            self.assertEqual(summary["duplicates"], len(duplicates))
            self.assertEqual(summary["pruned"], len(distinct) - 1)
            self.assertEqual(summary["front"], 1)


if __name__ == "__main__":
    unittest.main()


class AnchorResume(unittest.TestCase):
    def test_checkpointed_parallel_anchor_resumes_without_remeasuring(self):
        measured = []

        def measure_curve(binary, codec, clip, frames, overrides, work, qs, **_):
            measured.append(clip.name)
            return tune.Curve([100.0, 200.0, 400.0, 800.0], [30.0, 33.0, 36.0, 39.0],
                              quantizers=[40, 35, 30, 25], vmaf_v1=[50.0, 60.0, 70.0, 80.0])
        clips = [SimpleNamespace(name=c.name, content_class=c.content_class,
                                 clean_reference=None, yuv=c.name) for c in CLIPS]
        cost = tune.InstructionCost(1.0, 10.0)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(tune, "measure_curve", side_effect=measure_curve), \
                patch.object(tune, "evaluation_identity", return_value={"binary": "b"}), \
                patch.object(tune, "clip_fingerprint", side_effect=lambda c: c.name):
            work = Path(tmp) / "anchor"
            for _ in range(2):
                result = tune.evaluate(Path("bin"), tune.Codec.AV1, {}, work, clips=clips,
                                       cost=cost, clip_workers=4, checkpoint=True)
                self.assertEqual(sorted(result.curves), sorted(c.name for c in clips))
        self.assertEqual(sorted(measured), sorted(c.name for c in clips))


class RealUseSearch(unittest.TestCase):
    def test_realuse_objectives_constraints_and_cost(self):
        class FakeReal:
            def __init__(self, *args):
                self.labels = []

            def evaluate(self, overrides, fidelity):
                self.labels.append(fidelity)
                x = overrides["x"]
                summary = {"quality": {"classes": {"pan": 2.0 * x}},
                           "quality_low": {"classes": {"pan": 3.0 * x}},
                           "compliance": {"test_noncompliant": 0, "anchor_noncompliant": 0},
                           **{t: {"balanced": 0.0, "classes": {}} for t, _ in tune_realuse.GUARDS}}
                return tune_realuse.RealUseResult(x, 1.5 * x, 0.0, [3.0 * x - 3.0], summary, [])
        fake = FakeReal()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "nanocodec"
            binary.write_bytes(b"build")
            space = root / "space.json"
            space.write_text('{"x": {"choices": [0, 1, 2]}}')
            args = argparse.Namespace(codec="av1", binary=str(binary), params="x",
                                      search_space=str(space), study="real-test", seed=3,
                                      trials=4, workers=1, frames=60, objective="realuse")
            with patch.object(tune, "TUNING_DIR", root), \
                 patch.object(tune, "specs_of", return_value={"x": (0, 0, 2)}), \
                 patch.object(tune, "proxy_clips", return_value=CLIPS), \
                 patch.object(tune, "clip_fingerprint", side_effect=lambda c: c.name), \
                 patch.object(tune, "evaluate", side_effect=AssertionError("no fixed-QP curves")), \
                 patch.object(tune, "instruction_cost", side_effect=lambda b, c, overrides, w, **_:
                              tune.InstructionCost(1.1 if overrides else 1.0, 10.0)), \
                 patch.object(tune_realuse, "RealUseObjective", return_value=fake), \
                 patch.object(tune.jreal, "objective", return_value={"j_real": 0.0, "guards": []}), \
                 patch.object(tune, "print_front"):
                tune.search(args)
                study = tune.optuna.load_study(study_name="real-test",
                                               storage=tune.storage_of("real-test"))
        done = [t for t in study.trials if t.state.name == "COMPLETE"]
        self.assertEqual(len(done), 4)
        for t in done:
            x = t.params["x"]
            self.assertEqual(t.values[:2], [x, 1.5 * x])
            self.assertAlmostEqual(t.values[2], 1.1)
            self.assertEqual(t.user_attrs["constraint"], [3.0 * x - 3.0])
        self.assertEqual(study.user_attrs["objective"], "realuse")


class RealUseGuards(unittest.TestCase):
    def test_every_guard_is_a_constraint(self):
        objective = tune_realuse.RealUseObjective.__new__(tune_realuse.RealUseObjective)
        objective.class_guard = 3.0
        objective.flicker_objective = False
        summary = {"quality": {"classes": {"pan": 4.0, "static": -1.0}},
                   "quality_low": {"classes": {"pan": 1.0}},
                   "bd_worst_psnr_y": {"balanced": 2.5}, "bd_worst_vmaf_v1": {"balanced": None},
                   "d_flicker_psnr_y": {"balanced": 0.01},
                   "compliance": {"test_noncompliant": 2, "anchor_noncompliant": 1}}
        self.assertEqual([round(v, 6) for v in objective.constraints(summary)],
                         [1.0, -2.0, 0.5, -2.0, -0.04, 1.0])


class RealUseOptions(unittest.TestCase):
    def test_flicker_objective_replaces_the_flicker_guard(self):
        result = tune_realuse.RealUseResult(2.0, 4.0, -0.03, [], {}, [])
        self.assertEqual(result.objectives(0.9, False), (2.0, 4.0, 0.9))
        self.assertEqual(result.objectives(0.9, True), (3.0, -0.03, 0.9))
        objective = tune_realuse.RealUseObjective.__new__(tune_realuse.RealUseObjective)
        objective.class_guard, objective.flicker_objective = 3.0, True
        summary = {"quality": {"classes": {"pan": 1.0}}, "quality_low": {"classes": {"pan": 1.0}},
                   "bd_worst_psnr_y": {"balanced": 0.0}, "bd_worst_vmaf_v1": {"balanced": 0.0},
                   "d_flicker_psnr_y": {"balanced": 9.0},
                   "compliance": {"test_noncompliant": 0, "anchor_noncompliant": 0}}
        self.assertEqual(len(objective.constraints(summary)), 5)

    def test_points_are_keyed_by_parameters_not_trial(self):
        objective = tune_realuse.RealUseObjective.__new__(tune_realuse.RealUseObjective)
        objective.codec, objective.binary = tune.Codec.AV1, Path("bin")
        a = objective.family({"x": 1.0, "y": 2.0})
        self.assertEqual(a.name, objective.family({"y": 2.0, "x": 1.0}).name)
        self.assertNotEqual(a.name, objective.family({"x": 1.5, "y": 2.0}).name)
        self.assertEqual(objective.family({}).name, "nanocodec-av1@default")

    def test_real_ladder_uses_target_subsets(self):
        rungs = ladder.parse_ladder("30x2,60x4", 60, {"30fx2": (1.0, 0.0)},
                                    tune_realuse.TARGET_SUBSETS, (0, 1, 2, 3))
        self.assertEqual(rungs[0].qp_indices, tune_realuse.DEFAULT_TARGETS)
        with self.assertRaisesRegex(ValueError, "no measured error"):
            ladder.parse_ladder("30x2,60x4", 60, {}, tune_realuse.TARGET_SUBSETS, (0, 1, 2, 3))
