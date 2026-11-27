"""tools/bench/plan.py: the BP-3 ladder, NTSC rates, record commands and the sealed-set refusal."""
import json
import subprocess

import pytest
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
import plan  # noqa: E402


def test_sub_linear_ladder_matches_the_owners_examples():
    top = {(1920, 1080, "30/1"): 2205, (1920, 1080, "60/1"): 3118, (1280, 720, "60/1"): 1697,
           (640, 360, "30/1"): 424, (480, 270, "30/1"): 276, (1280, 720, "30/1"): 1200}
    for (w, h, fps), kbps in top.items():
        assert plan.scaled_kbps(1200, {"width": w, "height": h, "fps": fps}) == kbps


def test_ntsc_rates_are_normalised_for_every_encoder():
    assert plan.frame_rate("2997/100") == (30000, 1001)
    assert plan.frame_rate("2997/125") == (24000, 1001)
    assert plan.frame_rate("25/1") == (25, 1)
    assert plan.frame_rate(25.0) == (25, 1)
    assert plan.frame_rate(29.97) == (30000, 1001)


def test_jobs_come_from_the_record_and_respect_the_source_cap():
    clips = [{"name": "c", "width": 640, "height": 360, "fps": "30/1", "source_kbps": 700}]
    jobs = plan.plan(clips, ["x264-medium", "nanocodec-av1@medium"], "cbr",
                     list(plan.LADDER_720P30), 150, 300)
    ladder = [j for j in jobs if not j["anchor_extra"]]
    assert {j["kbps"] for j in ladder} == {106, 177, 283}  # 424 kbps is over half the 700 kbps source
    assert all(j["argv"][0].startswith("{BIN}/") for j in jobs)
    assert {j["decoder"] for j in jobs} == {"ffmpeg"}  # the standard decoder, ours included


def test_a_clip_without_a_source_rate_is_refused():
    clips = [{"name": "c", "width": 640, "height": 360, "fps": "30/1"}]
    with pytest.raises(SystemExit, match="source_kbps"):
        plan.plan(clips, ["x264-medium"], "cbr", list(plan.LADDER_720P30), 150, 300)


def test_fixed_qp_plans_six_quantizers_and_drop_rows_are_cbr_only():
    clips = [{"name": "c", "width": 640, "height": 360, "fps": "30/1", "source_kbps": 700}]
    jobs = plan.plan(clips, ["x264-medium", "libaom-av1-cpu10", "libvpx-vp9-cpu9+drop"],
                     "qp", [], 150, 300)
    by_family = {}
    for j in (j for j in jobs if not j["anchor_extra"]):
        by_family.setdefault(j["family"], []).append(j["qp"])
    assert by_family == {"x264-medium": [22, 27, 32, 37, 42, 47],
                         "libaom-av1-cpu10": [28, 34, 40, 46, 52, 59]}
    cbr = plan.plan(clips, ["libaom-av1-cpu10", "libvpx-vp9-cpu9+drop"], "cbr",
                    list(plan.LADDER_720P30), 150, 300)
    # BP-4.0: libaom's main rows run aomenc; its twins are deferred (D20)
    assert {j["family"] for j in cbr} == {"libaom-av1-cpu10", "libvpx-vp9-cpu9+drop"}
    drop = next(j for j in cbr if j["family"] == "libvpx-vp9-cpu9+drop")
    assert "--drop-frame=30" in drop["argv"] and "--max-q=52" in drop["argv"]
    main = next(j for j in cbr if j["family"] == "libaom-av1-cpu10")
    assert main["argv"][0].endswith("/aomenc") and "--max-q=63" in main["argv"]


def test_sealed_run_refuses_clips_seen_in_design(tmp_path):
    seen = json.loads((REPO / "eval/clipset/sealed_exclusions.json").read_text())["clips"][0]["name"]
    clips = tmp_path / "clips.json"
    clips.write_text(json.dumps([{"name": seen, "width": 640, "height": 360, "fps": "30/1",
                                  "source_kbps": 1000}]))
    run = subprocess.run([sys.executable, str(REPO / "tools/bench/plan.py"), "--clips", str(clips),
                          "--out", str(tmp_path / "jobs.json"), "--sealed"], capture_output=True, text=True)
    assert run.returncode != 0 and "refused" in run.stderr + run.stdout


def test_the_anchor_codes_points_past_each_end():
    """BP-3.4 and D11: x264 medium and its twins add 0.5x the bottom and 2x and 4x the top
    rung; a top point is capped at half the source rate, coded once at the cap, and dropped
    when the cap is under the top."""
    roomy = {"name": "r", "width": 1280, "height": 720, "fps": "30/1", "source_kbps": 10_000}
    mid = {**roomy, "name": "m", "source_kbps": 6_000}     # cap 3000: 2x 2400, 4x capped
    tight = {**roomy, "name": "t", "source_kbps": 3_000}   # cap 1500: both at the cap, once
    under = {**roomy, "name": "u", "source_kbps": 2_400}   # cap 1200: no high point
    jobs = plan.plan([roomy, mid, tight, under], ["x264-medium~psnr", "svtav1-m12"], "cbr",
                     list(plan.LADDER_720P30), 150, 300)
    extra = {(j["input"], j["kbps"]) for j in jobs if j["anchor_extra"]}
    assert extra == {("r", 150), ("r", 2400), ("r", 4800), ("m", 150), ("m", 2400),
                     ("m", 3000), ("t", 150), ("t", 1500), ("u", 150)}
    assert not any(j["anchor_extra"] for j in jobs if j["family"] == "svtav1-m12")
    qp = plan.plan([roomy], ["x264-medium"], "qp", [], 150, 300)
    assert sorted(j["qp"] for j in qp if j["anchor_extra"]) == [17, 51]


def test_the_dev_screens_code_the_benchmarks_rungs():
    """nceval's screens (cbr.clip_targets on realuse.LADDER_KBPS) and the benchmark's planner
    take their rungs and scaling from one definition (nceval.rate_ladder)."""
    from nceval.cbr import clip_targets
    from nceval.clips import Clip
    from nceval.realuse import LADDER_KBPS
    assert tuple(LADDER_KBPS) == tuple(plan.LADDER_720P30)
    for w, h, fps, source in ((1920, 1080, 29.97, 6000), (640, 360, 25.0, 900),
                              (1280, 720, 60.0, 1500)):
        clip = Clip(name="c", yuv=Path("c.yuv"), width=w, height=h, fps=fps, frames=150,
                    source_kbps=source, content_class="static", split="train")
        entry = {"width": w, "height": h, "fps": fps, "source_kbps": source}
        jobs = plan.plan([{**entry, "name": "c"}], ["x264-medium"], "cbr",
                         list(plan.LADDER_720P30), 150, 300)
        assert clip_targets(clip, LADDER_KBPS) == [j["kbps"] for j in jobs if not j["anchor_extra"]]


def test_each_clip_is_coded_once_at_its_longest_window():
    """D7/D8: a clip whose source allows the long window is coded at it and also serves the
    short window (its first frames); every other clip is coded at the short one. A key every
    150 frames in both."""
    short = {"name": "s", "width": 640, "height": 360, "fps": "30/1", "source_kbps": 2000,
             "frames_available": 300}
    long = {**short, "name": "l", "frames_available": 900}
    jobs = plan.plan([short, long], ["x264-medium"], "cbr", list(plan.LADDER_720P30),
                     plan.SHORT_FRAMES, plan.KEY_INTERVAL, long_frames=plan.LONG_FRAMES)
    by_clip = {}
    for j in jobs:
        by_clip.setdefault(j["input"], set()).add((j["frames"], tuple(j["windows"])))
    assert by_clip == {"s": {(150, (150,))}, "l": {(600, (150, 600))}}
    argv = next(j["argv"] for j in jobs if j["input"] == "l")
    assert argv[argv.index("--frames") + 1] == "600" and argv[argv.index("--keyint") + 1] == "150"
    assert len({j["id"] for j in jobs}) == len(jobs)  # one job per (family, clip, point)
