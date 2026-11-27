"""The key shares every acceptance screen measures (nceval.key_shares)."""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import interval, key_shares


def stats(mean, half=0.2, n=10):
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": n, "confidence": 0.9}


def test_shares_are_periodic_keys_in_frames_with_two_periodic_keys_at_least():
    short, long = key_shares.shares()
    assert (short.period, short.window) == (150, 600)  # keys at 0, 150, 300, 450
    assert (long.period, long.window) == (300, 900)  # keys at 0, 300, 600
    for share in (short, long):
        assert (share.window - 1) // share.period >= key_shares.MIN_PERIODIC_KEYS


def test_the_owner_minimum_cannot_be_left_out():
    with pytest.raises(ValueError, match="1/300"):
        key_shares.shares([150])
    assert [s.period for s in key_shares.parse_periods("300,150,600")] == [150, 300, 600]


def test_dcpu_is_given_per_share_or_once_for_every_share_with_a_warning():
    chosen = key_shares.shares()
    assert key_shares.parse_dcpu("150=2.5,300=1", chosen) == {150: 2.5, 300: 1.0}
    assert key_shares.parse_dcpu("1.5", chosen) == {150: 1.5, 300: 1.5}
    assert key_shares.parse_dcpu(None, chosen) == {150: None, 300: None}
    assert key_shares.uniform_dcpu_warning("1.5") and not key_shares.uniform_dcpu_warning(
        "150=1,300=1")
    with pytest.raises(ValueError, match="1/300"):
        key_shares.parse_dcpu("150=2", chosen)


def test_apply_dcpu_gives_the_short_levels_the_150_share():
    args = argparse.Namespace(dcpu="150=3,300=1", key_shares=None)
    key_shares.apply_dcpu(args)
    assert args.dcpu == 3.0 and args.dcpu_shares == {150: 3.0, 300: 1.0}


def test_a_loss_at_either_share_is_worse_whatever_the_mean():
    decision, combined, each = key_shares.judge({150: stats(-3.0), 300: stats(1.0)}, final=True)
    assert each == {150: interval.BETTER, 300: interval.WORSE}
    assert decision == interval.WORSE and combined["mean"] == pytest.approx(-1.0)


def test_the_verdict_is_on_the_equal_weight_mean_of_the_shares():
    decision, combined, _ = key_shares.judge({150: stats(-0.6), 300: stats(-0.2)}, final=True)
    assert combined["mean"] == pytest.approx(-0.4)
    assert combined["hi"] - combined["mean"] == pytest.approx((2 * 0.2 ** 2) ** 0.5 / 2)
    assert decision == interval.BETTER
    missing = key_shares.combined_stats({150: stats(-0.6), 300: {**stats(0), "mean": None}})
    assert missing["mean"] is None


def long_window(monkeypatch, loss_at, guards=()):
    """share_level on four clips with measure_level faked: each share's clips lose
    `loss_at(key_frames)` (J %), the first share's verdict carries `guards`; returns
    (the calls to measure_level, a function running share_level with a CPU charge)."""
    from nceval import levels
    from nceval.clips import Clip
    clip = lambda name, frames: Clip(name=name, yuv=Path(f"/x/{name}.yuv"), width=64,
                                     height=64, fps=25.0, frames=frames, source_kbps=1e4,
                                     content_class="crowd", split="train", family=name,
                                     start_frame=40)
    drawn = [clip("a", 900), clip("b", 900), clip("c", 600), clip("d", 300)]
    calls = []

    def measure_level(runner, families, clips, frames, level, estimator, args, key_frames=None):
        calls.append(([c.name for c in clips], frames, key_frames,
                      {c.start_frame for c in clips}))
        loss = loss_at(key_frames)
        return {"stage": level, "estimator": levels.SINGLE,
                "per_clip": {c.name: {"class": "crowd", "quality": loss + 0.01 * i,
                                      "quality_low": loss + 0.01 * i}
                             for i, c in enumerate(clips)}}
    monkeypatch.setattr(levels, "measure_level", measure_level)
    monkeypatch.setattr(levels, "add_periodic_pulse", lambda *a: None)
    monkeypatch.setattr(levels.jreal, "summarise", lambda rows: {})
    monkeypatch.setattr(levels.jreal, "objective",
                        lambda *a: {"guards": list(guards) if not calls[1:] else []})

    def run(dcpu_150):
        from nceval import decision_cpu
        measured = decision_cpu.Charge("d", {"f600-k150": dcpu_150, "f900-k300": 0.0}, "test")
        args = argparse.Namespace(frames="150", final_estimator=levels.SINGLE, k=0.5,
                                  dcpu=None, cpu=measured, key_shares=None)
        return levels.share_level(None, (None, None), drawn, "final", args)
    return calls, run


def test_the_long_window_level_measures_and_charges_every_share(monkeypatch):
    # the change gains at 1/150 (also with its CPU) and loses at 1/300
    calls, run = long_window(monkeypatch, lambda key: 0.4 if key == 300 else -0.4)
    result = run(dcpu_150=0.2)
    # frames, not seconds, at 25 fps; each share on the clips its window fits, from frame 0
    assert calls == [(["a", "b", "c"], 600, 150, {0}), (["a", "b"], 900, 300, {0})]
    assert result["share_results"][150]["dcpu"] == 0.2  # CPU charged at its own share
    assert result["shares"][150]["mean"] == pytest.approx(-0.4 + 0.01 + 0.5 * 0.2)
    assert result["share_decisions"] == {150: interval.BETTER, 300: interval.WORSE}
    assert result["decision"] == interval.WORSE and result["shares_skipped"] == []


def test_a_share_that_settles_the_level_worse_skips_the_rest(monkeypatch):
    from nceval import levels
    calls, run = long_window(monkeypatch, lambda key: -0.4)
    result = run(dcpu_150=2.0)  # 1/150 loses once its CPU is charged: WORSE already
    assert [c[2] for c in calls] == [150]
    assert result["decision"] == interval.WORSE and result["shares_skipped"] == [300]
    # a re-judge with a CPU that no longer loses cannot decide without the missing share
    stage = {**result, "stage": "final"}
    recorded = {"stages": [stage], "frames": "150",
                "levels": {"final": {"decision": interval.WORSE, "guards": []}}}
    from nceval import decision_cpu
    cheap = decision_cpu.Charge("d", {"f600-k150": 0.0, "f900-k300": 0.0}, "test")
    judged = levels.rejudge(recorded, cheap, 0.5)["levels"]["final"]
    assert judged["decision"] == interval.AMBIGUOUS and judged["shares_missing"] == [300]


def test_a_pulse_at_the_first_share_skips_the_rest(monkeypatch):
    pulse = "d_periodic_pulse_psnr_y (worst clip) +0.52 > +0.50"
    calls, run = long_window(monkeypatch, lambda key: -0.4, guards=[pulse])
    result = run(dcpu_150=0.0)  # BETTER on J, but the pulse fails the level (`run`)
    assert [c[2] for c in calls] == [150]
    assert result["shares_skipped"] == [300] and pulse in result["verdict"]["guards"]


def test_no_share_is_skipped_while_the_level_can_still_pass(monkeypatch):
    calls, run = long_window(monkeypatch, lambda key: -0.4)
    result = run(dcpu_150=0.0)
    assert [c[2] for c in calls] == [150, 300] and result["shares_skipped"] == []
    assert result["decision"] == interval.BETTER
