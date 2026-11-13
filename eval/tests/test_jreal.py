"""J_real: temporal terms, robust aggregation, guards and decisions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import jreal  # noqa: E402

RATES = (150.0, 300.0, 600.0, 1200.0)


def curve(scale=1.0, quality_shift=0.0, flicker=0.3, pulse=1.0, worst_drop=1.0, periodic=None):
    """A plausible CBR curve; `scale` multiplies the achieved rates."""
    points = []
    for i, rate in enumerate(RATES):
        psnr = 32.0 + 3 * i + quality_shift
        points.append({"kbps": rate * scale, "psnr_yuv": psnr,
                       "vmaf_v1": 55.0 + 10 * i, "ms_ssim": 1 - 0.05 / (i + 1),
                       "worst_psnr_y": psnr - worst_drop, "worst_vmaf_v1": 50.0 + 10 * i,
                       "flicker_psnr_y": flicker, "first_key_pulse_psnr_y": pulse,
                       "periodic_key_pulse_psnr_y": periodic})
    return points


def test_temporal_stats_exclude_the_key_frame():
    psnr = [45.0] + [40.0, 39.0, 41.0, 38.0] * 25
    vmaf = [99.0] + [80.0] * 99 + [10.0]
    keys = [True] + [False] * 100
    stats = jreal.temporal_stats(psnr, vmaf, keys)
    # the 4-frame cadence averages out in every window; the last window
    # holds the one bad VMAF frame
    assert stats["worst_psnr_y"] == pytest.approx(39.5)
    assert stats["worst_vmaf_v1"] == pytest.approx((3 * 80.0 + 10.0) / 4)
    assert stats["worst_frame_psnr_y"] == 38.0  # 1% of 100 P frames: the single worst
    assert stats["worst_frame_vmaf_v1"] == 10.0
    assert stats["first_key_pulse_psnr_y"] == pytest.approx(45.0 - 39.5)
    assert stats["periodic_key_pulse_psnr_y"] is None  # one key: no periodic data
    # the pair (0, 1) touches the key frame and is skipped; the cadence steps
    # are 1, 2, 3, 2 dB
    assert stats["flicker_frame_psnr_y"] == pytest.approx(2.0)
    # every 8-frame window holds two whole cadence cycles: no pumping
    assert stats["flicker_psnr_y"] == pytest.approx(0.0)


def test_window_flicker_sees_slow_pumping_and_restarts_at_keys():
    window = jreal.FLICKER_WINDOW
    # 40 dB and 38 dB windows alternate: 2 dB between every pair of windows
    pumping = ([40.0] * window + [38.0] * window) * 3
    assert jreal.window_flicker(pumping, [False] * len(pumping)) == pytest.approx(2.0)
    # a key starts new windows; the step across it is not counted, nor a
    # trailing partial window
    psnr = [45.0] + [40.0] * window + [30.0] * window + [45.0] + [36.0] * (2 * window) + [1.0]
    keys = [True] + [False] * (2 * window) + [True] + [False] * (2 * window + 1)
    assert jreal.window_flicker(psnr, keys) == pytest.approx(10.0 / 2)


def test_worst_windows_see_a_run_of_bad_frames_not_a_layer_sawtooth():
    window = jreal.WORST_WINDOW
    keys = [True] + [False] * 100
    # L1T2: every other frame 2 dB coarser, by design
    sawtooth = [45.0] + [40.0, 38.0] * 50
    # the same mean, but one run of a whole window 4 dB down
    run = [45.0] + [39.0] * 60 + [35.0] * window + [39.0] * (40 - window)
    vmaf = [90.0] * 101
    assert jreal.temporal_stats(sawtooth, vmaf, keys)["worst_psnr_y"] == pytest.approx(39.0)
    assert jreal.temporal_stats(sawtooth, vmaf, keys)["worst_frame_psnr_y"] == 38.0
    assert jreal.temporal_stats(run, vmaf, keys)["worst_psnr_y"] == pytest.approx(35.0)


def test_compare_clip_reports_every_term():
    terms = jreal.compare_clip(curve(), curve(scale=0.9, flicker=0.5, pulse=-3.0))
    for metric in jreal.HEADLINE_METRICS:
        assert terms[f"bd_{metric}"] == pytest.approx(-10.0, abs=0.1)
        assert terms[f"low_{metric}"] == pytest.approx(-10.0, abs=0.1)
    assert terms["quality"] == pytest.approx(-10.0, abs=0.1)
    assert terms["d_flicker_psnr_y"] == pytest.approx(0.2)
    assert terms["d_first_pulse_psnr_y"] == pytest.approx(2.0)  # |pulse| compared
    assert terms["d_periodic_pulse_psnr_y"] is None


def test_periodic_key_pulse_is_the_jump_over_the_frames_before_each_later_key():
    psnr = ([45.0] + [38.0] * 59) * 2 + [40.0] + [38.0] * 29
    keys = ([True] + [False] * 59) * 2 + [True] + [False] * 29
    psnr[60] = 39.0  # the second key stands 1 dB over its neighbours, the third 2 dB
    stats = jreal.temporal_stats(psnr, [80.0] * len(psnr), keys)
    assert stats["first_key_pulse_psnr_y"] == pytest.approx(7.0)
    assert stats["periodic_key_pulse_psnr_y"] == pytest.approx(1.5)


def test_only_the_periodic_pulse_is_guarded():
    big_first = jreal.objective(summary_for(curve(scale=0.9, pulse=6.0)))
    assert big_first["decision"] == "better"
    big_periodic = jreal.objective(summary_for(curve(scale=0.9, periodic=2.0),
                                               anchor=curve(periodic=1.0)))
    assert big_periodic["decision"] == "worse"
    assert any("periodic" in g for g in big_periodic["guards"])


def test_aggregate_is_class_balanced_with_median_and_missing():
    rows = {"s1": {"class": "static", "quality": -30.0},
            "s2": {"class": "static", "quality": -30.0},
            "s3": {"class": "static", "quality": -30.0},
            "p1": {"class": "pan", "quality": 2.0},
            "c1": {"class": "crowd", "quality": None}}
    agg = jreal.aggregate(rows, "quality")
    assert agg["balanced"] == pytest.approx(-14.0)  # static and pan weigh the same
    assert agg["mean"] == pytest.approx(-22.0)
    assert agg["median"] == -30.0
    assert agg["worse"] == 1 and agg["missing"] == ["c1"]


def summary_for(test_curve, classes=("static", "pan"), anchor=None):
    rows = {f"{c}{i}": {"class": c, **jreal.compare_clip(anchor or curve(), test_curve)}
            for c in classes for i in range(2)}
    return jreal.summarise(rows)


def test_clear_gain_is_better():
    verdict = jreal.objective(summary_for(curve(scale=0.9)))
    assert verdict["decision"] == "better" and verdict["j_real"] == pytest.approx(-10.0, abs=0.1)


def test_cpu_term_can_turn_a_gain_into_a_loss():
    verdict = jreal.objective(summary_for(curve(scale=0.97)), dcpu_pct=10.0)
    assert verdict["decision"] == "worse"


def test_default_cpu_weight_keeps_a_quality_arm():
    # A Default quality arm is judged at k 0.1 (PROCEDURES "Preset roles"):
    # the same 3% gain for 10% CPU that k 0.6 rejects passes.
    summary = summary_for(curve(scale=0.97))
    verdict = jreal.objective(summary, dcpu_pct=10.0, k_cpu=0.1)
    assert verdict["decision"] == "better" and verdict["k_cpu"] == 0.1
    assert verdict["j_real"] == pytest.approx(
        jreal.objective(summary, dcpu_pct=10.0)["j_real"] - 5.0, abs=1e-9)


def test_mean_gain_with_worse_flicker_is_rejected():
    verdict = jreal.objective(summary_for(curve(scale=0.9, flicker=0.6)))
    assert verdict["decision"] == "worse"
    assert any("flicker" in g for g in verdict["guards"])


def test_mean_gain_with_worse_worst_frames_is_rejected():
    verdict = jreal.objective(summary_for(curve(scale=0.95, worst_drop=3.0)))
    assert verdict["decision"] == "worse"
    assert any("worst_psnr_y" in g for g in verdict["guards"])


def test_flicker_guard_only_moves_a_proxy_stage_on():
    verdict = jreal.objective(summary_for(curve(scale=0.9, flicker=0.6)), sigma=1.0)
    assert verdict["decision"] == "ambiguous"


def test_one_class_over_the_guard_is_rejected():
    rows = {"s": {"class": "static", **jreal.compare_clip(curve(), curve(scale=0.7))},
            "p": {"class": "pan", **jreal.compare_clip(curve(), curve(scale=1.05))}}
    verdict = jreal.objective(jreal.summarise(rows))
    assert verdict["j_real"] < 0 and verdict["decision"] == "worse"


def test_weaknesses_rank_the_worst_cells():
    rows = {"s": {"class": "static", **jreal.compare_clip(curve(), curve(scale=1.02))},
            "p": {"class": "pan", **jreal.compare_clip(curve(), curve(scale=1.10, flicker=0.4))}}
    cells = jreal.weaknesses(jreal.summarise(rows), rows)
    assert cells[0][0] == "pan"
    assert ("pan", "d_flicker_psnr_y", pytest.approx(0.1), "") in cells
    assert ("static", "quality", pytest.approx(2.0, abs=0.1), "") in cells


def with_buffer(points, fullness, overruns):
    return [dict(p, max_fullness=fullness, overruns=overruns) for p in points]


def test_buffer_compliance_is_reported_beside_bd_not_inside_it():
    anchor = with_buffer(curve(), 2.6, 40)  # the key frame ignores the 1 s buffer
    test = with_buffer(curve(scale=1.05), 0.9, 0)
    terms = jreal.compare_clip(anchor, test)
    assert terms["bd_psnr_yuv"] == pytest.approx(5.0, abs=0.1)  # BD unchanged
    assert (terms["anchor_overruns"], terms["anchor_noncompliant"]) == (160, 4)
    assert (terms["test_overruns"], terms["test_noncompliant"]) == (0, 0)


def test_losses_against_a_noncompliant_anchor_are_marked_and_ranked_last():
    rows = {"s": {"class": "static", **jreal.compare_clip(with_buffer(curve(), 2.6, 40),
                                                         with_buffer(curve(scale=1.3), 0.9, 0))},
            "p": {"class": "pan", **jreal.compare_clip(with_buffer(curve(), 0.9, 0),
                                                      with_buffer(curve(scale=1.05), 0.9, 0))}}
    summary = jreal.summarise(rows)
    assert summary["compliance"]["anchor_noncompliant_clips"] == ["s"]
    cells = jreal.weaknesses(summary, rows)
    assert cells[0][0] == "pan" and cells[0][3] == ""
    assert any(c[0] == "static" and c[3] == "anchor non-compliant on 1/1 clips" for c in cells)


def test_a_candidate_that_breaks_the_buffer_is_rejected():
    rows = {"p": {"class": "pan", **jreal.compare_clip(with_buffer(curve(), 0.9, 0),
                                                      with_buffer(curve(scale=0.8), 1.8, 10))}}
    verdict = jreal.objective(jreal.summarise(rows))
    assert verdict["decision"] == "worse" and any("buffer" in g for g in verdict["guards"])


def with_target(points, over):
    """`points` with targets their achieved rate lies `over` (fraction) above."""
    return [dict(p, target_kbps=p["kbps"] / (1 + over)) for p in points]


def test_ceiling_guard_relative_rejects_more_overshoot_than_the_anchor(monkeypatch):
    monkeypatch.setattr(jreal, "CEILING_MODE", jreal.CEILING_RELATIVE)
    anchor = with_target(with_buffer(curve(), 0.9, 0), 0.04)  # today's Default: 4% over
    same = with_target(with_buffer(curve(scale=0.98), 0.9, 0), 0.04)
    more = with_target(with_buffer(curve(scale=0.98), 0.9, 0), 0.06)
    passes = jreal.objective(jreal.summarise({"p": {"class": "pan",
                                                    **jreal.compare_clip(anchor, same)}}))
    fails = jreal.objective(jreal.summarise({"p": {"class": "pan",
                                                   **jreal.compare_clip(anchor, more)}}))
    assert not any("ceiling" in g for g in passes["guards"])
    assert fails["decision"] == "worse" and any("ceiling" in g for g in fails["guards"])


def test_ceiling_guard_absolute_rejects_any_point_over_target(monkeypatch):
    monkeypatch.setattr(jreal, "CEILING_MODE", jreal.CEILING_ABSOLUTE)
    anchor = with_target(with_buffer(curve(), 0.9, 0), 0.0)
    within = with_target(with_buffer(curve(scale=0.98), 0.9, 0), 0.005)  # inside 1%
    over = with_target(with_buffer(curve(scale=0.98), 0.9, 0), 0.02)
    terms = jreal.compare_clip(anchor, over)
    assert terms["test_over_target_points"] == 4
    assert terms["test_over_target_sum"] == pytest.approx(4 * 1.0)
    ok = jreal.objective(jreal.summarise({"p": {"class": "pan",
                                                **jreal.compare_clip(anchor, within)}}))
    assert not any("ceiling" in g for g in ok["guards"])
    bad = jreal.objective(jreal.summarise({"p": {"class": "pan", **terms}}))
    assert any(g.startswith("ceiling: 4 points") for g in bad["guards"])


def test_flicker_guard_allows_headroom_below_the_reference():
    anchor = [{"flicker_psnr_y": 0.1}, {"flicker_psnr_y": 0.2}]
    # the anchor (0.15) lies 0.25 dB below a 0.40 reference: a +0.2 delta is inside it
    assert jreal.flicker_over_headroom(0.2, anchor, 0.40) == pytest.approx(-0.05)
    # an anchor above the reference gets no headroom; no reference, none either
    assert jreal.flicker_over_headroom(0.2, anchor, 0.10) == pytest.approx(0.2)
    assert jreal.flicker_over_headroom(0.2, anchor, None) == pytest.approx(0.2)
    assert jreal.flicker_over_headroom(None, anchor, 0.40) is None


def two_layer(full, base):
    """A two-layer stream's curve: each point carries its base layer's (realuse.point_terms)."""
    return [{**f, "base": b} for f, b in zip(full, base)]


def test_base_layer_is_a_term_only_for_two_layer_streams():
    assert jreal.compare_clip(curve(), curve(scale=0.9))["base_quality"] is None
    terms = jreal.compare_clip(two_layer(curve(), curve()),
                               two_layer(curve(scale=0.9), curve(scale=1.1)))
    assert terms["quality"] == pytest.approx(-10.0, abs=0.1)
    assert terms["base_quality"] == pytest.approx(10.0, abs=0.1)


def test_full_layer_gain_bought_with_the_base_layer_is_rejected():
    anchor = two_layer(curve(), curve())
    verdict = jreal.objective(summary_for(two_layer(curve(scale=0.9), curve(scale=1.05)),
                                          anchor=anchor))
    assert verdict["decision"] == "worse"
    assert any("base_quality" in g for g in verdict["guards"])
    kept = jreal.objective(summary_for(two_layer(curve(scale=0.9), curve(scale=1.01)),
                                       anchor=anchor))
    assert kept["decision"] == "better"


def test_single_point_base_quality_prices_the_base_layer():
    anchor = two_layer(curve(), curve())
    point = two_layer(curve(scale=0.9), curve(scale=1.1))[1]
    assert jreal.point_base_quality(anchor, point) == pytest.approx(10.0, abs=0.5)
    assert jreal.point_base_quality(curve(), curve()[1]) is None
