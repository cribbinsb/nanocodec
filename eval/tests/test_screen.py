"""The shared screen: staged decisions, cached points, paired signal, honest gaps."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import screen, stages, tune
from nceval.conformance import GateResult
from nceval.config import Paths
from nceval.metrics import Measurement
from nceval.point import PointResult

GRID = [26, 30, 34, 38]


class FakeEncoder:
    """Rate 1000/2^((q-26)/6) kbps, quality 50-q; the candidate scales the rate."""

    def __init__(self, factor):
        self.factor, self.calls = factor, []

    def encode(self, binary, codec, clip, q, frames, overrides, stream, metrics=None,
               own_decoder=None, standard=True):
        self.calls.append((clip.name, q, bool(overrides), standard))
        scale = self.factor if overrides else 1.0
        stream.write_text(str(1000 * scale / 2 ** ((q - 26) / 6)))
        quality = {"psnr_yuv": 50.0 - q, "vmaf_v1": 100.0 - 2 * q}
        return PointResult(GateResult(True, level="full" if standard else "own-decoder"),
                           "sha", Measurement(quality))

    def payload(self, stream, codec, frames):
        return float(stream.read_text()) * 1000 / 8 * frames / 30


@pytest.fixture
def world(tmp_path):
    clips = [tune.ClipRef(name, tmp_path / f"{name}.yuv", 16, 16, 30, 300, "static", 1e6)
             for name in tune.PROXY_CLIPS]
    paths = Paths(tmp_path / "data", tmp_path / "anchors", tmp_path / "bin", tmp_path / "work")
    with patch.object(tune, "proxy_clips", return_value=clips), \
         patch.object(screen, "load_paths", return_value=paths), \
         patch.object(stages, "load_paths", return_value=paths), \
         patch.object(screen, "measurement_fingerprint", return_value="m"), \
         patch.object(tune, "clip_fingerprint", side_effect=lambda c: c.name), \
         patch.object(tune, "calibrate_quantizers", return_value=GRID) as calibrate:
        yield tmp_path, calibrate


def run(tmp_path, factor, **changes):
    binary = tmp_path / "nanocodec"
    binary.write_bytes(b"encoder")
    fake = FakeEncoder(factor)
    args = dict(codec="h264", binary=str(binary), param=["decision.lambda_scale=1.2"],
                anchor_binary=None, anchor_param=[], clips="train", staged=True,
                frames="150", qps=None, gate="sample", workers=2, json=None, over_budget=False,
                all_stages=False)
    args.update(changes)
    with patch.object(tune, "encode_point", side_effect=fake.encode), \
         patch.object(screen, "payload_bytes", side_effect=fake.payload):
        result = screen.screen(argparse.Namespace(**args))
    return result, fake


def test_clear_win_stops_after_four_clips_and_three_qps(world):
    tmp_path, calibrate = world
    result, fake = run(tmp_path, 0.8)
    assert [s["stage"] for s in result["stages"]] == ["proxy-4"]
    assert result["decision"] == "better"
    assert len(fake.calls) == 4 * 3 * 2
    assert {q for _, q, _, _ in fake.calls} == {26, 30, 38}  # outer three of four
    assert result["stages"][0]["summary"]["psnr"]["mean"] == pytest.approx(-20, abs=0.5)
    assert calibrate.call_count == 4  # one anchor grid per clip


def test_clear_loss_stops_early(world):
    result, _ = run(world[0], 1.1)
    assert result["decision"] == "worse" and len(result["stages"]) == 1


def test_small_effect_escalates_to_full_train_reusing_points(world):
    tmp_path, _ = world
    result, fake = run(tmp_path, 0.995)
    assert [s["stage"] for s in result["stages"]] == ["proxy-4", "proxy-6", "train"]
    assert result["decision"] == "better"  # exact at the last stage
    # Stage 1 runs at 60 frames (its own points); later stages share 150-frame points.
    assert result["stages"][0]["frames"] == 60
    assert len(fake.calls) == 4 * 3 * 2 + len(tune.PROXY_CLIPS) * 4 * 2
    _, again = run(tmp_path, 0.995)
    assert again.calls == []  # all served from the cache


def test_matched_qp_signal_and_gate_sampling(world):
    result, fake = run(world[0], 0.8)
    row = next(iter(result["stages"][0]["per_clip"].values()))
    assert row["matched"]["bits_pct"] == pytest.approx(-20)
    assert row["matched"]["d_psnr"] == 0
    standard = [(c, q) for c, q, _, s in fake.calls if s]
    assert {q for _, q in standard} == {26}  # standard decoders on each curve's first QP


def test_unstaged_screen_measures_the_whole_fixed_grid_at_once(world):
    result, fake = run(world[0], 0.9, staged=False, qps="26,30,34,38,42")
    assert [s["stage"] for s in result["stages"]] == ["all"]
    assert len(fake.calls) == len(tune.PROXY_CLIPS) * 5 * 2


def test_identical_candidate_and_anchor_is_refused(world):
    with pytest.raises(SystemExit, match="identical"):
        run(world[0], 1.0, param=[])


def test_decision_rule():
    assert screen.decide({"psnr": -5.0, "vmaf_v1": -4.7}, 2.3) == "better"
    assert screen.decide({"psnr": -5.0, "vmaf_v1": -4.5}, 2.3) == "ambiguous"
    assert screen.decide({"psnr": -9.0, "vmaf_v1": 4.7}, 2.3) == "worse"
    assert screen.decide({"psnr": None, "vmaf_v1": -9.0}, 2.3) == "ambiguous"
    assert screen.decide({"psnr": 0.0, "vmaf_v1": 0.0}, 0.0) == "neutral"


VAL_NAMES = ("static_otw_homes390", "static_cam29", "static_london_bus", "pan_bdd_00cef86b",
             "pan_jaad_0101", "crowd_pp22_051", "crowd_mot20_06", "bodycam_airport",
             "handheld_mot17_10", "lowlight_pp22_049", "lowlight_bdd_00ce95b0")


@pytest.fixture
def val_world(world):
    tmp_path, _ = world
    clips = [tune.ClipRef(n, tmp_path / f"{n}.yuv", 16, 16, 30, 300, "static", 1e6, "val")
             for n in VAL_NAMES]
    noise = tune.ClipRef("static_london_bus_noise_low", tmp_path / "n.yuv", 16, 16, 30, 300,
                         "synthetic_noise", 1e6, "val", tmp_path / "static_london_bus.yuv")
    with patch.object(tune, "load_clips", return_value=clips + [noise]):
        yield tmp_path


def test_val_150_is_staged_undecoded_outer_three(val_world):
    result, fake = run(val_world, 0.8, clips="val", gate=None)
    assert result["gate"] == "off" and [s["stage"] for s in result["stages"]] == ["val-proxy-5"]
    assert len(fake.calls) == 5 * 3 * 2 and not any(standard for *_, standard in fake.calls)
    result, fake = run(val_world, 0.995, clips="val", gate=None,
                       param=["decision.lambda_scale=1.1"])
    assert [s["stage"] for s in result["stages"]] == ["val-proxy-5", "val"]
    anchor_calls = [c for c in fake.calls if not c[2]]
    assert len(fake.calls) == 11 * 3 + 6 * 3  # new candidate everywhere; anchor only new clips
    assert len(anchor_calls) == 6 * 3
    assert len(result["stages"][-1]["per_clip"]) == 11  # noise clips are not VAL-150


def test_sigmas_are_learned_from_complete_screens_only(world):
    tmp_path, _ = world
    path = stages.history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {"set": "val", "codec": "h264",
                "stages": {"val-proxy-5": {"psnr": 1.5, "vmaf_v1": -0.5},
                           "val": {"psnr": 0.5, "vmaf_v1": 0.5}}}
    early = {"set": "val", "codec": "h264", "stages": {"val-proxy-5": {"psnr": 9.0, "vmaf_v1": 9.0}}}
    path.write_text("\n".join(json.dumps(e) for e in [early] + [complete] * 19) + "\n")
    plan, note = stages.calibrated_plan("val", stages.VAL_STAGES)
    assert plan[0].sigma == 2.3 and "19/20 complete screens" in note  # still the prior
    with path.open("a") as out:
        out.write(json.dumps(complete) + "\n")
    plan, note = stages.calibrated_plan("val", stages.VAL_STAGES)
    assert plan[0].sigma == 1.0 and "learned (n=20)" in note
    assert [s.name for s in plan] == ["val-proxy-5", "val"]


def test_all_stages_runs_past_a_clear_decision_and_records_history(world):
    result, _ = run(world[0], 0.8, all_stages=True)
    assert [s["stage"] for s in result["stages"]] == ["proxy-4", "proxy-6", "train"]
    entry = json.loads(stages.history_path().read_text().splitlines()[-1])
    assert set(entry["stages"]) == {"proxy-4", "proxy-6", "train"}


def test_over_budget_whole_stage_runs_in_as_many_balanced_parts_as_needed(world):
    from nceval import budget
    calls = []
    real = screen.check_budget
    def priced(label, cpu, cuts, force):
        calls.append(label)
        return real(label, cpu, cuts, force)
    with patch.object(screen, "estimate_minutes", side_effect=lambda cpu: 55.0), \
         patch.object(screen, "check_budget", side_effect=priced), \
         patch.object(budget, "available_slots", return_value=1000.0):
        result, _ = run(world[0], 0.995)
    # 55 min -> 3 parts; each part is re-priced (still "55" here) and split again,
    # down to single clips, which are then priced and run.
    assert all("(1 clips)" in c for c in calls if c.startswith("screen stage train"))
    assert len(result["stages"][-1]["per_clip"]) == len(tune.PROXY_CLIPS)
    parts = stages.balanced_parts(tune.proxy_clips(), 3)
    assert sorted(len(p) for p in parts) == [6, 6, 6]


def test_refusal_still_writes_the_result_json(world, tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["screen", "--codec", "h264", "--binary", str(tmp_path / "b"),
                                      "--param", "x=1", "--json", str(out), "--mode", "cqp"])
    with patch.object(screen, "screen", side_effect=screen.OverBudget("refused: too big")):
        with pytest.raises(SystemExit):
            screen.main()
    assert json.loads(out.read_text()) == {"status": "refused", "reason": "refused: too big"}


def test_over_budget_proxy_stage_also_runs_in_parts(world):
    from nceval import budget
    calls = []
    real = screen.check_budget
    def priced(label, cpu, cuts, force):
        calls.append(label)
        return real(label, cpu, cuts, force)
    with patch.object(screen, "estimate_minutes", side_effect=lambda cpu: 45.0), \
         patch.object(screen, "check_budget", side_effect=priced), \
         patch.object(budget, "available_slots", return_value=1000.0):
        result, _ = run(world[0], 0.995)
    assert all("(1 clips)" in c for c in calls if c.startswith("screen stage proxy-6"))
    assert len(result["stages"][1]["per_clip"]) == 6
