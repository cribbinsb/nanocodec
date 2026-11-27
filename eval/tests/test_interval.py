"""nceval.interval: the level decisions, identical streams included."""

from nceval import interval


def test_identical_streams_are_neutral_at_l3_and_continue_before():
    # Identical points: J deltas 0 up to summation rounding (an L1 of 2026-11-22).
    rounding = {"mean": 8.1e-15, "lo": 4.6e-16, "hi": 1.5e-14}
    assert interval.decide(rounding, final=True) == interval.NEUTRAL
    assert interval.decide(rounding, final=False) == interval.CONTINUE


def test_a_tiny_real_change_is_not_identical():
    small = {"mean": 0.01, "lo": -0.02, "hi": 0.04}
    assert not interval.identical(small)
    assert interval.decide(small, final=True) == interval.AMBIGUOUS
    assert interval.decide({"mean": -0.4, "lo": -0.6, "hi": -0.2}, final=True) == interval.BETTER


def test_an_interval_that_crosses_zero_is_not_better_at_l3():
    # AV1 faster zero_gate_cell 32 (train 2026-11-23): J -0.08 % [-0.28, +0.12]
    # was labelled BETTER under the old +0.3 % upper bound.
    straddling = {"mean": -0.08, "lo": -0.28, "hi": 0.12}
    assert interval.decide(straddling, final=True) == interval.AMBIGUOUS
    touching = {"mean": -0.2, "lo": -0.4, "hi": 0.0}
    assert interval.decide(touching, final=True) == interval.AMBIGUOUS
    below = {"mean": -0.08, "lo": -0.15, "hi": -0.01}
    assert interval.decide(below, final=True) == interval.BETTER
