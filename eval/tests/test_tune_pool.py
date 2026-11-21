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


def test_a_study_on_another_cpu_family_is_refused_unless_re_anchored(monkeypatch):
    import pytest
    from nceval import cloud
    monkeypatch.setattr(cloud, "cpu_family", lambda: "c7i")
    assert tune_pool.check_family(None, "study s", False) is False  # nothing recorded
    assert tune_pool.check_family("c7i", "study s", False) is False
    with pytest.raises(SystemExit, match="counted on c7a, this is c7i"):
        tune_pool.check_family("c7a", "study s", False)
    assert tune_pool.check_family("c7a", "study s", True) is True  # count the anchor again
    assert cloud.box_family("host:pc") is None and cloud.box_family("c7a") == "c7a"


def test_a_finalists_dcpu_weighs_arm_and_x86_equally_with_callgrind_standing_in():
    both = tune_pool.combined_dcpu(-3.0, {"arm": -4.0, "x86": -2.0, "targets": {}})
    assert both["dcpu"] == -3.0 and both["dcpu_x86"] == -2.0
    assert both["dcpu_x86_how"] == "zen5" and both["dcpu_callgrind"] == -3.0
    arm_only = tune_pool.combined_dcpu(-3.0, {"arm": -5.0, "x86": None})
    assert arm_only["dcpu"] == -4.0 and arm_only["dcpu_x86_how"] == "callgrind"
    none = tune_pool.combined_dcpu(-3.0, None)
    assert none["dcpu"] == -3.0 and none["dcpu_arm"] is None


def two_rung_study(slow_j, slow_cpu, k=0.5, explicit=False):
    """A fake between-rungs study: trial 0 the base rung, trial 1 the slower seed rung."""
    import optuna
    study = optuna.create_study(directions=["minimize", "minimize"])
    for attr, value in {"k": k, "k_explicit": explicit,
                        "seed_rungs": [{"rung": "fast", "trial": 0},
                                       {"rung": "medium", "trial": 1}]}.items():
        study.set_user_attr(attr, value)
    for j, cpu in ((0.0, 1.0), (slow_j, slow_cpu)):
        study.add_trial(optuna.trial.create_trial(
            values=[j, cpu], user_attrs={tune_pool.FIDELITY_ATTR: tune_pool.FULL}))
    return study


def test_a_between_rungs_study_takes_k_from_the_chord_and_judges_its_front_against_it(capsys):
    import math
    study = two_rung_study(-3.0, 1.5)  # the slower rung: 3 % better for 50 % more CPU
    assert tune_pool.study_k(study.user_attrs) == (0.5, "the default k (chord not measured yet)")
    tune_pool.set_chord(study)
    k, source = tune_pool.study_k(study.user_attrs)
    slope = 3.0 / (100 * math.log(1.5))
    assert math.isclose(k, slope) and source == "the chord fast -> medium"
    assert "k 0.07399 (the chord fast -> medium)" in capsys.readouterr().out
    chord = study.user_attrs["chord"]
    # half-way in log CPU the chord is at half the J; a move along it scores ~0 at its k
    assert math.isclose(tune_pool.chord_j(chord, math.sqrt(1.5)), -1.5)
    on_chord = -1.5
    assert abs(on_chord + k * 100 * math.log(math.sqrt(1.5))) < 1e-9
    assert 0.5 * 100 * (math.sqrt(1.5) - 1) + on_chord > 0  # the base k stopped it


def test_an_explicit_k_above_the_chord_is_kept_but_warned_and_a_rung_that_buys_nothing_has_none(
        capsys):
    study = two_rung_study(-3.0, 1.5, k=0.5, explicit=True)
    tune_pool.set_chord(study)
    assert tune_pool.study_k(study.user_attrs) == (0.5, "--k")
    out = capsys.readouterr().out
    assert "!!!" in out and "--k 0.5 is above the chord's slope 0.07399" in out
    flat = two_rung_study(+0.5, 1.5)
    tune_pool.set_chord(flat)
    assert flat.user_attrs["chord"]["k"] is None
    assert tune_pool.study_k(flat.user_attrs)[0] == 0.5
    assert "does not buy J with CPU" in capsys.readouterr().out


def test_a_study_from_a_binary_path_records_its_commit_and_finalists_need_rented_core_cpu(
        tmp_path, monkeypatch):
    import optuna
    import pytest
    from nceval import buildinfo
    binary = tmp_path / "nanocodec"
    binary.write_text("elf")
    (tmp_path / "COMMIT").write_text("f" * 40 + "\n")
    monkeypatch.setattr(buildinfo, "build_info", lambda b: {"commit": None, "dirty": False})
    assert tune_pool.binary_commit(binary) == "f" * 40  # the cached copy's COMMIT
    monkeypatch.setattr(buildinfo, "build_info", lambda b: {"commit": "a" * 40, "dirty": False})
    assert tune_pool.binary_commit(binary) == "a" * 40  # the stamp first
    monkeypatch.setattr(buildinfo, "build_info", lambda b: {"commit": "a" * 40, "dirty": True})
    assert tune_pool.binary_commit(binary) is None  # a dirty build is no commit's
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.set_user_attr("commit", "")
    monkeypatch.setattr(tune_pool, "finalist_todo", lambda s, c, k: {"3": None})
    monkeypatch.setattr(tune_pool, "study_k", lambda attrs: (0.1, "k"))
    with pytest.raises(SystemExit, match="#3 have no rented-core CPU"):
        tune_pool.require_box_cpu(study, 5, None, callgrind_only=False)
    tune_pool.require_box_cpu(study, 5, None, callgrind_only=True)
    study.set_user_attr("finalist_cpu", {"3": {"arm": 0.0, "x86": None}})
    tune_pool.require_box_cpu(study, 5, None, callgrind_only=False)  # 0.0 is a measurement
