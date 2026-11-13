"""tools/dev/chart_sanity.py: the ratio check and codec grouping (no callgrind runs)."""

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "tools" / "dev" / "chart_sanity.py"
spec = importlib.util.spec_from_file_location("chart_sanity", TOOL)
chart_sanity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chart_sanity)

# (Jetson fps, x86 fps): AV1 Fast mis-timed level with Faster on x86 (2026-11-12).
FPS = {"nanocodec-av1": (40.0, 100.0), "nanocodec-av1@av1-fast": (62.0, 155.0),
       "nanocodec-av1@av1-faster": (78.0, 195.0), "nanocodec-vvc": (25.0, 58.0)}


def test_groups_skip_single_preset_codecs():
    groups = chart_sanity.by_codec(FPS, None)
    assert list(groups) == ["av1"]
    assert chart_sanity.by_codec(FPS, {"vvc"}) == {}


def test_consistent_ratios_pass():
    assert chart_sanity.ratio_flags(FPS, chart_sanity.by_codec(FPS, None)) == []


def test_groups_run_slowest_to_fastest():
    fps = {**FPS, "nanocodec-av1@av1-fastest": (90.0, 225.0)}
    assert chart_sanity.by_codec(fps, {"av1"})["av1"] == [
        "nanocodec-av1", "nanocodec-av1@av1-fast", "nanocodec-av1@av1-faster",
        "nanocodec-av1@av1-fastest"]


def test_only_adjacent_presets_are_compared():
    # Default/Faster differ by 25% but each adjacent pair stays within the tolerance.
    fps = {"nanocodec-hevc": (48.0, 120.0), "nanocodec-hevc@hevc-fast": (72.0, 160.0),
           "nanocodec-hevc@hevc-faster": (90.0, 180.0)}
    assert chart_sanity.ratio_flags(fps, chart_sanity.by_codec(fps, None)) == []


def test_off_ratio_is_flagged():
    fps = {**FPS, "nanocodec-av1@av1-fast": (62.0, 210.0)}
    flags = chart_sanity.ratio_flags(fps, chart_sanity.by_codec(fps, None))
    assert len(flags) == 2 and all("nanocodec-av1@av1-fast " in flag for flag in flags)
