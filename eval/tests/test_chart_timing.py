"""nceval.chart_timing: a CBR chart's timing commands are its own quality commands.

Owner rule (2026-11-23): never combine measurements of different things. A chart's
speed axis times a subset of exactly the encodes its quality axis scores: the same rate
mode and buffer, clips, frames and key structure. These tests fail on any difference,
for every encoder kind the charts carry, and on the stored timing plan.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from nceval import chart, chart_timing
from nceval.cbr import BUFFER_MS, DEFAULT_INTRA_PERIOD, Bench, Codec, Family, clip_targets
from nceval.clips import Clip
from nceval.config import load_paths
from nceval.gatepolicy import EVALUATION_GATE
from nceval.realuse import DEFAULT_FRAMES, LADDER_KBPS, Runner

PLAN = Path(__file__).resolve().parents[2] / "docs/results/bd-vs-speed-plan.json"
#: One family of every encoder kind a CBR chart carries.
FAMILIES = (
    Family("x264-medium", Codec.H264, Path("/t/x264"), kind="x264", preset="medium"),
    Family("nanocodec-av1@fast", Codec.AV1, Path("/t/nanocodec"),
           ("--param", "a.b=1"), own=True, cqp=False),
    Family("svtav1-m13", Codec.AV1, Path("/t/SvtAv1EncApp"), kind="svt", preset="13"),
    Family("openh264-baseline", Codec.H264, Path("/t/openh264-harness"), kind="openh264"),
    Family("x265-medium", Codec.HEVC, Path("/t/x265"), kind="x265", preset="medium"),
    Family("libvpx-vp9-cpu7", Codec.VP9, Path("/t/vpxenc"), kind="vpx", preset="7"),
    Family("libaom-av1-cpu8", Codec.AV1, Path("/t/aomenc"), kind="aom", preset="8"),
)
#: Command-line forms of a fixed quantizer, in any of those encoders.
FIXED_QP_FORMS = ("--qp", "--end-usage=q", "--cq-level", "--min-q=", "--max-q=")


def clip(tmp_path, name, content_class, width=1280, height=720, fps=30.0):
    yuv = tmp_path / f"{name}.yuv"
    yuv.write_bytes(b"")
    return Clip(name=name, yuv=yuv, width=width, height=height, fps=fps, frames=300,
                source_kbps=20000.0, content_class=content_class, split="val")


@pytest.fixture
def clips(tmp_path):
    return [clip(tmp_path, "static_a", "static"), clip(tmp_path, "lowlight_b", "lowlight"),
            clip(tmp_path, "lowlight_a", "lowlight", fps=24.0),
            clip(tmp_path, "crowd_a", "crowd", width=1884), clip(tmp_path, "bodycam_a", "bodycam")]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_WORK", str(tmp_path / "work"))
    # the stand-in tools do not exist: their content identifies no command
    monkeypatch.setattr(Bench, "tool_hash", lambda self, family: "tool")
    return Runner(load_paths(), EVALUATION_GATE, workers=1, over_budget=True)


def quality_commands(runner, clips):
    """{(family, clip, kbps): quality encode command} as the chart's measure lists them."""
    return {(p.family.name, p.clip.name, p.kbps): bench.command(p, Path("{OUTPUT}"), None)
            for bench, p in chart_timing.quality_jobs(runner, list(FAMILIES), clips)}


def test_every_timing_command_is_one_of_the_charts_quality_commands(runner, clips):
    quality = quality_commands(runner, clips)
    jobs = chart_timing.timing_jobs(runner, list(FAMILIES), clips)
    assert len(jobs) == len(FAMILIES) * len(chart_timing.TIMING_SLOTS)
    for bench, point in jobs:
        entry = chart_timing.plan_entry(bench, point)
        command = quality[(point.family.name, point.clip.name, point.kbps)]
        assert entry["argv"] == chart_timing.placeholder_argv(command, point.clip)


def test_every_job_is_every_quality_encode_low_and_upper_rungs_included(runner, clips):
    quality = quality_commands(runner, clips)
    jobs = chart_timing.every_job(runner, list(FAMILIES), clips, ["static_a"])
    keys = {(p.family.name, p.clip.name, p.kbps) for _, p in jobs}
    assert set(quality) <= keys
    extra = [p for _, p in jobs if (p.family.name, p.clip.name, p.kbps) not in quality]
    low = [p for p in extra if p.kbps < min(clip_targets(p.clip, LADDER_KBPS))]
    upper = [p for p in extra if p.kbps > max(clip_targets(p.clip, LADDER_KBPS))]
    assert len(low) + len(upper) == len(extra)
    assert len(low) == 2 * len(FAMILIES) and {p.clip.name for p in low} == {"static_a"}
    assert {(p.family.name, p.clip.name, p.kbps) for p in upper} == {
        (f.name, c.name, k) for f in FAMILIES for c in clips for k in chart.upper_ladder(c)}


def test_every_job_is_the_charts_measure_job_list(runner, clips):
    low = [c for c in clips if c.name == "static_a"]
    timed = chart_timing.every_job(runner, list(FAMILIES), clips, ["static_a"])
    measured = chart.chart_jobs(runner, list(FAMILIES), clips, DEFAULT_FRAMES, None, low)
    command = lambda jobs: [b.command(p, Path("{OUTPUT}"), None) for b, p in jobs]  # noqa: E731
    assert command(timed) == command(measured)


def test_every_job_refuses_unknown_low_rung_clips(runner, clips):
    with pytest.raises(ValueError, match="low-rung"):
        chart_timing.every_job(runner, list(FAMILIES), clips, ["nope"])


def test_timing_is_cbr_on_the_charts_frames_and_key_structure(runner, clips):
    for bench, point in chart_timing.timing_jobs(runner, list(FAMILIES), clips):
        entry = chart_timing.plan_entry(bench, point)
        assert point.qp is None and point.kbps
        assert (entry["frames"], entry["intra_period"], entry["layers"]) == (
            DEFAULT_FRAMES, DEFAULT_INTRA_PERIOD, 1)
        assert not any(a.startswith(FIXED_QP_FORMS) for a in entry["argv"])
        assert any(a.endswith(str(DEFAULT_FRAMES)) for a in entry["argv"]), entry["argv"]


def test_our_timing_carries_the_quality_buffer(runner, clips):
    ours = [(b, p) for b, p in chart_timing.timing_jobs(runner, list(FAMILIES), clips)
            if p.family.own]
    for bench, point in ours:
        argv = chart_timing.plan_entry(bench, point)["argv"]
        assert argv[argv.index("--buffer-ms") + 1] == str(BUFFER_MS)
        assert argv[argv.index("--bitrate") + 1] == str(point.kbps)


def test_timing_clips_are_the_charts_own_clips_one_per_slot(clips):
    rungs = chart_timing.timing_rungs(clips)
    assert rungs == {"lowlight_a": 300, "crowd_a": 800, "bodycam_a": 1200}


def test_a_chart_whose_slot_class_has_no_rung_refuses(tmp_path, clips):
    # a source at 400 kbps: its ladder is capped at half that, so it has no 1200 rung
    tiny = replace(clip(tmp_path, "bodycam_slow", "bodycam"), source_kbps=400.0)
    with pytest.raises(ValueError, match="bodycam"):
        chart_timing.timing_rungs([*(c for c in clips if c.content_class != "bodycam"), tiny])


def test_a_chart_without_the_slot_classes_spreads_its_slots_over_its_clips(tmp_path):
    calls = [clip(tmp_path, f"call_{i:02d}", "videocall") for i in range(9)]
    rungs = chart_timing.timing_rungs(calls)
    assert rungs == {"call_00": 300, "call_03": 800, "call_06": 1200}


def test_the_stored_plan_is_cbr_on_the_charts_clips():
    """The checked-in plan (anchor_chart_timing.py plan --all-encodes): CBR, the chart's
    frames and key interval, every quality encode of the chart (all its clips and rungs,
    the upper rungs, the low rungs on its low-rung clips), each entry at its key's rate.
    The stored plan is the 0.6.0 chart's, made on the pre-BP-3.10 ladder (150/300/600/1200,
    pixel-rate scaled), so its inputs are checked against themselves, not today's ladder."""
    record = json.loads(PLAN.read_text())
    assert record["rate_mode"] == "cbr" and record["frames"] == DEFAULT_FRAMES
    assert record["all_encodes"]
    for name, point in record["points"].items():
        assert point["encodes"] == len(record["clips"]), name
        assert not any(a.startswith(FIXED_QP_FORMS) for a in point["example"]["argv"]), name
    try:
        plan = chart_timing.load_plan(PLAN)
    except FileNotFoundError:
        pytest.skip("the full plan is not on this machine (its record is checked above)")
    for name, entries in plan["points"].items():
        assert set(entries) == set(plan["clips"]), name
        for clip_name, entry in entries.items():
            argv = entry["argv"]
            assert entry["kbps"] == plan["clips"][clip_name], (name, clip_name)
            assert (entry["frames"], entry["intra_period"], entry["layers"]) == (
                DEFAULT_FRAMES, DEFAULT_INTRA_PERIOD, 1), (name, clip_name)
            assert not any(a.startswith(FIXED_QP_FORMS) for a in argv), (name, argv)
            assert any(str(entry["kbps"]) in a for a in argv), (name, argv)
            assert str(DEFAULT_INTRA_PERIOD) in " ".join(argv), (name, argv)


def test_planned_sizes_come_from_finished_quality_streams_only(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/dev"))
    from anchor_chart_timing import stream_sizes

    point = tmp_path / "clip_a" / "x264-medium"
    point.mkdir(parents=True)
    (point / "cbr300-aa.264").write_bytes(b"x" * 7)
    (point / "cbr300-aa.done").write_bytes(b"")
    (point / "cbr600-bb.264").write_bytes(b"x" * 9)  # not finished: no .done
    (point / "cbr300-aa.rate.csv").write_bytes(b"log")
    import hashlib
    assert stream_sizes(tmp_path) == {"clip_a/x264-medium/cbr300-aa": {
        "bytes": 7, "sha256": hashlib.sha256(b"x" * 7).hexdigest()}}


def test_an_every_encode_plan_splits_by_clip():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/dev"))
    from aws_timing import input_subset

    plan = {"anchor": "x264-medium", "clips": {"a@100": 100, "a@200": 200, "b@100": 100},
            "inputs": {"a@100": "a", "a@200": "a", "b@100": "b"},
            "points": {"x264-medium": {"a@100": 1, "a@200": 2, "b@100": 3},
                       "nanocodec-av1": {"a@100": 4, "a@200": 5, "b@100": 6}}}
    part = input_subset(plan, {"a"})
    assert part["clips"] == {"a@100": 100, "a@200": 200}
    assert part["inputs"] == {"a@100": "a", "a@200": "a"}
    assert part["points"]["nanocodec-av1"] == {"a@100": 4, "a@200": 5}
    with pytest.raises(SystemExit):
        input_subset(plan, {"c"})


def test_a_short_clip_is_staged_at_its_own_length():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/dev"))
    from anchor_chart_timing import input_frames

    plan = {"frames": 150, "inputs": {"a@1": "a", "b@1": "b"},
            "points": {"p": {"a@1": {"frames": 150}, "b@1": {"frames": 144}}}}
    assert (input_frames(plan, "a"), input_frames(plan, "b")) == (150, 144)
