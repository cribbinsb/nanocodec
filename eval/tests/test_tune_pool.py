"""nceval.tune_pool's pure parts and jreal's single-point estimator."""

import math
import unittest
from types import SimpleNamespace

from nceval import jreal, singlepoint, tune_pool

SPECS = {"a.x": (2.0, 0.0, 64.0), "a.n": (3.0, 0.0, 3.0), "a.z": (0.0, 0.0, 10.0)}


class SinglePoint(unittest.TestCase):
    anchor = [{"kbps": 100.0, "psnr_yuv": 30.0}, {"kbps": 200.0, "psnr_yuv": 33.0},
              {"kbps": 400.0, "psnr_yuv": 36.0}]

    def test_on_the_curve_is_zero(self):
        self.assertAlmostEqual(jreal.point_rate_delta(self.anchor, self.anchor[1], "psnr_yuv"), 0)

    def test_same_quality_at_more_rate(self):
        point = {"kbps": 220.0, "psnr_yuv": 33.0}
        self.assertAlmostEqual(jreal.point_rate_delta(self.anchor, point, "psnr_yuv"), 10.0)

    def test_extrapolates_beyond_the_curve(self):
        point = {"kbps": 800.0, "psnr_yuv": 39.0}  # one more doubling: on the line
        self.assertAlmostEqual(jreal.point_rate_delta(self.anchor, point, "psnr_yuv"), 0.0)


class Space(unittest.TestCase):
    def test_trust_region_and_bounds(self):
        domains = tune_pool.read_space("a.x\na.n int\na.z 1 4 # bounded\n", SPECS, {"a.x": 0.1})
        self.assertEqual((domains["a.x"]["low"], domains["a.x"]["high"]), (0.05, 0.2))
        self.assertEqual((domains["a.n"]["low"], domains["a.n"]["high"]), (2.0, 3.0))
        self.assertEqual((domains["a.z"]["low"], domains["a.z"]["high"]), (1.0, 4.0))

    def test_out_of_range_and_too_many(self):
        with self.assertRaises(SystemExit):
            tune_pool.read_space("a.z 1 40\n", SPECS, {})
        names = [f"p{i}" for i in range(tune_pool.MAX_PARAMS + 1)]
        with self.assertRaises(SystemExit):
            tune_pool.read_space("\n".join(names), {n: (1.0, 0.0, 2.0) for n in names}, {})

    def test_widened_only_at_pressed_edges(self):
        domains = {"a.z": {"low": 2.0, "high": 4.0, "kind": "float", "limits": [0.0, 10.0]}}
        front = [SimpleNamespace(params={"a.z": 3.95})]
        out = tune_pool.widened(domains, front)["a.z"]
        self.assertEqual((out["low"], out["high"]), (2.0, 5.0))


class Estimates(unittest.TestCase):
    def test_rung_trend(self):
        rows = {f"c{i}": {"rung": i % 4, "quality": 0.5 * (i % 4)} for i in range(12)}
        change, t = singlepoint.rung_trend(rows)
        self.assertAlmostEqual(change, 1.5)
        self.assertTrue(math.isinf(t) or t > 10)

    def test_agreed_moves(self):
        domains = {"a.x": {"low": 0.0, "high": 10.0, "kind": "float"},
                   "a.n": {"low": 0.0, "high": 10.0, "kind": "int"}}
        moves = tune_pool.agreed_moves({"a.x": 5.0, "a.n": 5.0}, domains,
                                       [{"a.x": 7.0, "a.n": 6.0}, {"a.x": 6.0, "a.n": 4.0}])
        self.assertEqual(moves, {"a.x": 6.0})



class Sensitivity(unittest.TestCase):
    domain = {"low": 1.0, "high": 3.0, "kind": "float", "limits": [0.0, 8.0]}

    def test_two_values_each_side_and_switches_flip(self):
        self.assertEqual(tune_pool.sweep_values(2.0, self.domain), [1.0, 1.5, 2.5, 3.0])
        switch = {"low": 0.0, "high": 1.0, "kind": "int", "limits": [0.0, 1.0]}
        self.assertEqual(tune_pool.sweep_values(0.0, switch), [1.0])

    def test_live_and_widened_only_on_the_safe_side(self):
        results = [{"value": 1.0, "j": 3.0, "lo": 2.0, "hi": 4.0, "cpu": 1.0},
                   {"value": 3.0, "j": 0.1, "lo": -0.5, "hi": 0.6, "cpu": 1.0}]
        self.assertTrue(tune_pool.live(results))
        self.assertFalse(tune_pool.live(results[1:]))
        self.assertEqual(tune_pool.widened_bounds(2.0, self.domain, results), (1.0, 4.0))


    def test_seeds_combine_the_best_single_moves_within_the_cap(self):
        domains = {n: {"low": 0.0, "high": 10.0, "kind": "float"} for n in "abc"}
        results = {"_anchor_cost": 1.0,
                   "a=1": {"name": "a", "value": 1.0, "j": -2.0, "cpu": 1.0},
                   "a=2": {"name": "a", "value": 2.0, "j": -1.0, "cpu": 1.0},
                   "b=1": {"name": "b", "value": 1.0, "j": -3.0, "cpu": 1.2},
                   "c=1": {"name": "c", "value": 1.0, "j": +1.0, "cpu": 0.9}}
        self.assertEqual(tune_pool.sensitivity_seeds(results, domains, 1.01), [{"a": 1.0}])
        self.assertEqual(tune_pool.sensitivity_seeds(results, domains, None),
                         [{"b": 1.0}, {"b": 1.0, "a": 1.0}])


    def test_rate_and_structure_parameters_need_the_long_guard(self):
        for name in ("rate.cadence_scale", "layers.adaptive_max_static", "h264.keyframe_qp_offset",
                     "nonbase.rate.cadence_scale", "hevc.key_qp_offset_delta"):
            self.assertTrue(tune_pool.rate_or_structure(name), name)
        for name in ("decision.lambda_scale", "av1.inter_rounding_q7", "planner.bias_base"):
            self.assertFalse(tune_pool.rate_or_structure(name), name)


    def test_long_horizon_guard_judges_against_the_null_spread(self):
        noise = {"long_static_chirla_s024c2": {"quality": 3.3, "bd_worst_psnr_y": 24.0,
                                               "d_periodic_pulse_psnr_y": 0.7},
                 "long_crowd_mot20_02": {"quality": -1.6, "bd_worst_psnr_y": -1.7},
                 "long_pan_bdd_00c12bd0": {"quality": -3.7, "bd_worst_psnr_y": -6.0}}
        self.assertEqual(tune_pool.judge_long_horizon(noise)["guards"], [])
        loss = {clip: {"quality": 3.0} for clip in tune_pool.LONG_GUARD_CLIPS}
        self.assertTrue(tune_pool.judge_long_horizon(loss)["guards"])
        one_bad = {**{clip: {"quality": 0.0} for clip in tune_pool.LONG_GUARD_CLIPS},
                   "long_static_chirla_s024c2": {"quality": 8.8}}
        self.assertTrue(any("static" in g for g in tune_pool.judge_long_horizon(one_bad)["guards"]))
        with self.assertRaises(ValueError):
            tune_pool.judge_long_horizon({"long_crowd_mot20_02": {"quality": 0.0}})


if __name__ == "__main__":
    unittest.main()


def test_svc_base_guard_is_a_constraint_only_for_two_layer_rows():
    from nceval import jreal
    from nceval.singlepoint import base_constraint
    single = {"a": {"class": "pan", "quality": -1.0, "base_quality": None}}
    assert base_constraint(single) == []
    rows = {"a": {"class": "pan", "quality": -1.0, "base_quality": 3.0},
            "b": {"class": "static", "quality": -1.0, "base_quality": 1.0}}
    assert base_constraint(rows) == [2.0 - jreal.SVC_BASE_GUARD_PCT]
