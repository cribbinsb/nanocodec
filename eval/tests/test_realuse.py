"""Real-use screen and board: CBR ladder curves, staging and J_real decisions (no tools run)."""
import argparse
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import cbr, jreal, pool, realuse, screen, stages, tune  # noqa: E402
from nceval.clips import Clip  # noqa: E402
from nceval.config import Paths  # noqa: E402
from nceval.gatepolicy import PointCheck  # noqa: E402

CLASSES = ("bodycam", "crowd", "lowlight", "pan", "static")


def make_clips(tmp_path):
    names = list(tune.PROXY_CLIPS) + list(stages.VAL_PROXY)
    clips = {}
    for index, name in enumerate(names):
        yuv = tmp_path / f"{name}.yuv"
        yuv.write_bytes(b"x")
        clips[name] = Clip(name, yuv, 1280, 720, 30.0, 300, 10000.0,
                           name.split("_")[0] if name.split("_")[0] in CLASSES else
                           CLASSES[index % len(CLASSES)],
                           "val" if name in stages.VAL_PROXY else "train")
    return clips


class FakeMeasure:
    """Achieved rate = target * factor((family, clip) or family); PSNR rises 3 dB per doubling."""

    def __init__(self, factors, flicker=None):
        self.factors, self.flicker, self.calls = factors, flicker or {}, []
        self.checks = set()  # the gate checks the points asked for

    def __call__(self, bench, point):
        self.checks.add(point.check)
        self.calls.append((point.family.name, point.clip.name, point.kbps, bench.frames,
                           bench.intra))
        kbps = point.kbps * self.factors.get((point.family.name, point.clip.name),
                                             self.factors.get(point.family.name, 1.0))
        q = 30 + 3 * math.log2(point.kbps / 100)
        step = self.flicker.get(point.family.name, 0.3)
        # slow pumping: alternate flicker windows `step` dB apart
        psnr_y = [q + 5] + [q + (step if (i // jreal.FLICKER_WINDOW) % 2 else 0)
                            for i in range(bench.frames - 1)]
        return {"kbps": kbps, "target_kbps": point.kbps, "rate_error": 0.0, "overruns": 0, "max_fullness": 0.8, "psnr_yuv": q, "vmaf_v1": 2 * q,
                "ms_ssim": 1 - 10 ** (-q / 20),
                "frame_psnr_y": psnr_y, "frame_vmaf_v1": [2 * q] * bench.frames,
                "frame_key": [i % bench.intra == 0 for i in range(bench.frames)]}


@pytest.fixture
def world(tmp_path):
    clips = make_clips(tmp_path)
    paths = Paths(tmp_path / "data", tmp_path / "anchors", tmp_path / "bin", tmp_path / "work")
    with patch.object(realuse, "load_paths", return_value=paths), \
         patch.object(screen, "load_paths", return_value=paths), \
         patch.object(stages, "load_paths", return_value=paths), \
         patch.object(realuse, "main_clips", return_value=clips), \
         patch.object(cbr.Bench, "cached", return_value=False), \
         patch.object(realuse, "estimate_minutes", return_value=1.0), \
         patch.object(realuse, "check_budget", return_value=1.0):
        yield tmp_path


def run(tmp_path, measure, **changes):
    binary = tmp_path / "nanocodec"
    binary.write_bytes(b"encoder")
    args = dict(codec="h264", binary=str(binary), param=["decision.lambda_scale=1.2"],
                anchor_binary=None, anchor_param=[], clips="train", staged=True, frames="150",
                gate=None, workers=2, over_budget=False, all_stages=False, dcpu=None, k=jreal.K_CPU,
                max_level=3, rotation="test-day")
    args.update(changes)
    with patch.object(cbr.Bench, "measure", autospec=True, side_effect=measure):
        return realuse.screen(argparse.Namespace(**args))


def test_clear_win_stops_at_proxy_4_on_three_targets_and_60_frames(world):
    fake = FakeMeasure({"nanocodec-h264@candidate": 0.85})
    result = run(world, fake)
    assert [s["stage"] for s in result["stages"]] == ["proxy-4"]
    assert result["decision"] == "better"
    main = [c for c in fake.calls if c[4] == cbr.DEFAULT_INTRA_PERIOD]
    periodic = [c for c in fake.calls if c[4] != cbr.DEFAULT_INTRA_PERIOD]
    clips = {c for _, c, _, _, _ in main}
    assert clips == set(stages.PROXY_4)
    assert {f for _, _, _, f, _ in main} == {60}
    assert len(main) == 4 * 3 * 2  # clips x outer three ladder targets x sides
    # before passing: the periodic-key pass (2 s keys), one clip per class, two lowest rungs
    assert {i for *_, i in periodic} == {60} and len(periodic) == 3 * 2 * 2  # 3 classes
    assert result["stages"][0]["summary"]["d_periodic_pulse_psnr_y"]["scored"] == 3
    assert result["stages"][0]["verdict"]["j_real"] == pytest.approx(-15.0, abs=0.5)


POOL_CLASSES = ("bodycam", "crowd", "lowlight", "pan", "static", "texture")


@pytest.fixture
def pool_world(world):
    """A 90-clip pool (15 per stratum) behind `--clips adopt`."""
    clips = []
    for cls in POOL_CLASSES:
        for index in range(15):
            yuv = world / f"pool_{cls}{index}.yuv"
            yuv.write_bytes(b"x")
            clips.append(Clip(f"pool_{cls}{index}", yuv, 1280, 720, 30.0, 300, 10000.0, cls,
                              "train", family=f"{cls}{index}"))
    with patch.object(pool, "load_pool", return_value=clips):
        yield world, pool.draw(clips, "test-day", 150)


def main_calls(fake):
    return [c for c in fake.calls if c[4] == cbr.DEFAULT_INTRA_PERIOD]


def test_adopt_runs_the_three_levels_on_disjoint_draws_without_decoding(pool_world):
    world, draw = pool_world
    fake = FakeMeasure({"nanocodec-h264@candidate": 0.85})
    result = run(world, fake, clips="adopt")
    assert [s["stage"] for s in result["stages"]] == ["quick", "adopt", "final"]
    assert result["decision"] == "better"
    assert result["levels"]["final"]["hi"] < 0 and result["levels"]["quick"]["n"] == 24
    assert result["draw"]["rotation"] == "test-day"
    measured = {c for _, c, *_ in main_calls(fake)}
    assert measured == {c.name for level in draw.levels.values() for c in level}
    quick = {c.name for c in draw.levels["quick"]}
    # L1 is single-point: one candidate rung per clip on the anchor's full curve
    quick_calls = [c for c in main_calls(fake) if c[1] in quick]
    assert len([c for c in quick_calls if c[0] == "nanocodec-h264@candidate"]) == 24
    assert len([c for c in quick_calls if c[0] == "nanocodec-h264"]) == 24 * 4
    # evaluation decodes nothing (conformance is checked when a change lands)
    assert result["gate"] == "off" and fake.checks == {PointCheck.NONE}


def test_an_explicit_full_gate_checks_every_point(pool_world):
    world, _ = pool_world
    fake = FakeMeasure({"nanocodec-h264@candidate": 0.85})
    result = run(world, fake, clips="adopt", gate="full", max_level=1)
    assert fake.checks == {PointCheck.STANDARD}
    assert result["decision"].startswith("not adopted")


def test_a_confident_loss_stops_at_the_quick_level(pool_world):
    world, draw = pool_world
    fake = FakeMeasure({"nanocodec-h264@candidate": 1.3})
    result = run(world, fake, clips="adopt")
    assert [s["stage"] for s in result["stages"]] == ["quick"]
    assert result["decision"] == "worse" and result["levels"]["quick"]["lo"] > 0.5
    assert {c for _, c, *_ in main_calls(fake)} == {c.name for c in draw.levels["quick"]}


def test_one_bad_clip_is_ambiguous_not_worse(pool_world):
    world, draw = pool_world
    bad = draw.levels["quick"][0].name
    fake = FakeMeasure({("nanocodec-h264@candidate", bad): 1.5})
    result = run(world, fake, clips="adopt", max_level=2)
    assert [s["stage"] for s in result["stages"]] == ["quick", "adopt"]
    assert result["levels"]["quick"]["mean"] > 0 and result["levels"]["quick"]["lo"] <= 0.5
    assert result["decision"] == "not adopted (stopped after adopt)"


def test_mean_gain_with_more_flicker_is_rejected_on_the_whole_set(world):
    fake = FakeMeasure({"nanocodec-h264@candidate": 0.95},
                       flicker={"nanocodec-h264@candidate": 1.5})
    result = run(world, fake, clips="pan_jaad_0117,crowd_mot20_02", staged=False)
    verdict = result["stages"][-1]["verdict"]
    assert verdict["j_real"] < 0
    assert result["decision"] == "worse"
    assert any("flicker" in g for g in verdict["guards"])


def test_cpu_cost_is_folded_into_j_real(world):
    fake = FakeMeasure({"nanocodec-h264@candidate": 0.98})
    result = run(world, fake, clips="pan_jaad_0117", staged=False, dcpu=10.0)
    assert result["stages"][-1]["verdict"]["j_real"] == pytest.approx(-2.0 + 6.0, abs=0.3)
    assert result["decision"] == "worse"


def test_identical_sides_are_refused(world):
    with pytest.raises(SystemExit):
        run(world, FakeMeasure({}), param=[])


def overshooting(fake, family_part, rung=0):
    """`fake`, with every family containing `family_part` overshooting one rung by 50%."""
    def measure(bench, point):
        result = fake(bench, point)
        if family_part in point.family.name and \
                point.kbps == cbr.clip_targets(point.clip, realuse.LADDER_KBPS)[rung]:
            result["rate_error"] = 0.5
            result["kbps"] *= 1.5
        return result
    return measure


def test_a_rung_both_arms_overshoot_leaves_both_curves(world):
    fake = FakeMeasure({"nanocodec-h264@candidate": 0.9})
    result = run(world, overshooting(fake, "nanocodec-h264"), clips="pan_jaad_0117",
                 staged=False)
    row = result["stages"][-1]["per_clip"]["pan_jaad_0117"]
    assert row["points"] == [3, 3] and row["bd_psnr_yuv"] == pytest.approx(-10.0, abs=0.5)
    assert (row["rate_wins"], row["rate_losses"]) == (0, 0)


def test_an_overshooting_candidate_is_rejected_whatever_its_bd(world):
    # Like H.264 headroom off: the candidate lands far above its lowest target,
    # which on achieved-rate curves would look like extra quality for free.
    fake = FakeMeasure({"nanocodec-h264@candidate": 0.9})
    result = run(world, overshooting(fake, "@candidate"), clips="pan_jaad_0117",
                 staged=False)
    stage = result["stages"][-1]
    assert stage["per_clip"]["pan_jaad_0117"]["points"] == [3, 3]
    assert stage["summary"]["compliance"]["rate_losses"] == 1
    assert result["decision"] == "worse"
    assert any(g.startswith("rate:") for g in stage["verdict"]["guards"])


def test_an_overshooting_anchor_is_a_compliance_win(world):
    fake = FakeMeasure({"nanocodec-h264@candidate": 1.0})  # same quality: wins decide
    measure = overshooting(fake, "nanocodec-h264", rung=0)

    def anchor_only(bench, point):
        return (measure if "@candidate" not in point.family.name else fake)(bench, point)
    result = run(world, anchor_only, clips="pan_jaad_0117", staged=False)
    assert result["stages"][-1]["summary"]["compliance"]["rate_wins"] == 1
    assert result["decision"] == "better"


def test_dominated_points_leave_only_that_metric_curve():
    from nceval import jreal
    floor = [{"kbps": 400.0, "psnr_yuv": 30.0}, {"kbps": 420.0, "psnr_yuv": 30.0},
             {"kbps": 600.0, "psnr_yuv": 32.0}, {"kbps": 1200.0, "psnr_yuv": 35.0}]
    other = [dict(p, kbps=p["kbps"] * 0.9) for p in floor]
    assert jreal.curve_bd(floor, other, "psnr_yuv") == pytest.approx(-10.0, abs=0.01)


def test_own_family_turns_a_preset_into_params(tmp_path):
    family = realuse.own_family("nanocodec-hevc@hevc-fast", tmp_path / "nc")
    assert family.own and family.codec.value == "hevc" and not family.cqp
    assert family.extra[0] == "--param" and len(family.extra) % 2 == 0


def test_x264_cbr_anchor_runs_abr_with_a_one_second_vbv(tmp_path):
    clip = make_clips(tmp_path)["pan_jaad_0117"]
    family = cbr.Family("x264-veryfast", cbr.Codec.H264, Path("/x264"), kind="x264",
                        compares=(cbr.Codec.H264,))
    cmd = family.command(clip, 150, 300, Path("o.264"), None, kbps=600)
    assert cmd[cmd.index("--vbv-bufsize") + 1] == "600" and "zerolatency" in cmd
    assert cmd[cmd.index("--bframes") + 1] == "0"


def test_board_row_compares_with_bar_and_context_and_renders_a_page(world):
    tool = world / "tool"
    tool.write_bytes(b"anchor")
    fake = FakeMeasure({"nanocodec-h264": 1.05, "x264-veryfast": 0.9},
                       flicker={"nanocodec-h264": 0.8})
    args = argparse.Namespace(binary=str(world / "nanocodec"), rows="nanocodec-h264",
                              phase="all", frames=150, workers=2, over_budget=False,
                              presets_dir=str(realuse.PRESETS_DIR), clipset=None)
    (world / "nanocodec").write_bytes(b"encoder")
    with patch.object(cbr, "find_tool", return_value=tool), \
         patch.object(realuse, "OPENH264_HARNESS", tool), \
         patch.object(cbr.Bench, "measure", autospec=True, side_effect=fake):
        page = realuse.board(args)
    record = json.loads((realuse.board_dir(realuse.load_paths()) / "nanocodec-h264.json")
                        .read_text())
    bar = record["anchors"]["openh264-baseline"]
    assert bar["bar"] and not record["anchors"]["x264-veryfast"]["bar"]
    assert bar["summary"]["quality"]["balanced"] == pytest.approx(5.0, abs=0.3)
    assert bar["summary"]["d_flicker_psnr_y"]["balanced"] > 0.2
    text = page.read_text()
    assert "x264-veryfast (context)" in text and "Biggest weaknesses" in text
    assert "d_flicker_psnr_y" in text and "non-compliant points (ours / anchor)" in text
    assert "Stale rows" not in text
    # a row stored under the old J_real metrics (VMAF NEG) is listed, not shown
    old = dict(record, encoder="nanocodec-hevc", codec="hevc")
    del old["headline_metrics"]
    (realuse.board_dir(realuse.load_paths()) / "nanocodec-hevc.json").write_text(json.dumps(old))
    text = realuse.render_board(realuse.load_paths()).read_text()
    assert "Stale rows (stored under the old J_real metrics; re-run them): nanocodec-hevc." in text
    assert "| hevc |" not in text


def test_an_encoder_refusal_is_recorded_not_fatal(world):
    from nceval.framepipe import ProcessFailed
    fake = FakeMeasure({"nanocodec-h264@candidate": 0.9})

    def refuse(bench, point):
        if point.clip.name == "crowd_mot20_02" and "candidate" in point.family.name:
            raise ProcessFailed("encode failed (1): dimensions must be multiples of 8")
        return fake(bench, point)
    result = run(world, refuse, clips="pan_jaad_0117,crowd_mot20_02", staged=False)
    rows = result["stages"][-1]["per_clip"]
    assert rows["crowd_mot20_02"]["points"][1] == 0 and rows["crowd_mot20_02"]["quality"] is None
    assert rows["pan_jaad_0117"]["quality"] == pytest.approx(-10.0, abs=0.5)


def test_board_parts_only_cache_and_a_final_run_writes_the_row(world):
    tool = world / "tool"
    tool.write_bytes(b"anchor")
    (world / "nanocodec").write_bytes(b"encoder")
    fake = FakeMeasure({})
    args = argparse.Namespace(binary=str(world / "nanocodec"), rows="nanocodec-h264",
                              phase="all", frames=150, workers=2, over_budget=False,
                              presets_dir=str(realuse.PRESETS_DIR), clipset=None, part="1/2")
    with patch.object(cbr, "find_tool", return_value=tool), \
         patch.object(realuse, "OPENH264_HARNESS", tool), \
         patch.object(cbr.Bench, "measure", autospec=True, side_effect=fake):
        assert realuse.board(args) is None
    board = realuse.board_dir(realuse.load_paths())
    assert not (board / "nanocodec-h264.json").exists()
    ours = {c for f, c, *_ in fake.calls if f == "nanocodec-h264"}
    assert 0 < len(ours) < len(realuse.natural_val(realuse.load_paths()))


def test_rate_validity_allows_buffer_debt_over_a_short_window():
    five_seconds = 150 / 30.0
    funded_burst = {"rate_error": 0.16, "max_fullness": 1.2}  # london_bus at 150 frames
    assert realuse.rate_valid(funded_burst, five_seconds)  # allowance 10% + 1 s / 5 s = 30%
    assert not realuse.rate_valid(funded_burst, 60.0)  # the same over a minute: 11.7%
    floor = {"rate_error": 0.5, "max_fullness": 1.2}
    assert not realuse.rate_valid(floor, five_seconds)
    breaks_buffer = {"rate_error": 0.0, "max_fullness": 1.8}
    assert not realuse.rate_valid(breaks_buffer, five_seconds)
    loan = {"rate_error": 0.4, "loan_adjusted_rate_error": 0.05, "max_fullness": 1.0}
    assert realuse.rate_valid(loan, five_seconds)  # our key loan is excluded
