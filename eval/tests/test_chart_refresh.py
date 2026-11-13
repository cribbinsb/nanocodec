"""tools/dev/chart_refresh.py: timing merges and the build note (no builds or tools run)."""
import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "dev" / "chart_refresh.py"
spec = importlib.util.spec_from_file_location("chart_refresh", TOOL)
chart_refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chart_refresh)


def timing_record(tmp_path, point, point_ms, anchor_ms, time_ratio=None):
    """A timing file as `time` (Jetson) writes it, or with `time_ratio` (per clip,
    per anchor) as `time-host` (x86) writes it."""
    path = tmp_path / f"t-{point}.json"
    path.write_text(json.dumps({"point": point, "anchor": "x264-medium", "clips": {
        clip: {"ms_per_frame": {"point": point_ms[clip], "anchor": anchor_ms[clip]},
               "budget_fraction": 1.0,
               **({"time_ratio": time_ratio[clip]} if time_ratio else {})}
        for clip in point_ms}}))
    return path


@pytest.fixture
def stored(tmp_path):
    path = tmp_path / "timing.json"
    path.write_text(json.dumps({"anchor": "x264-medium", "points": {
        "x264-medium": {"ms_per_frame": {"a": 20.0, "b": 40.0}},
        "x265-medium": {"ms_per_frame": {"a": 80.0, "b": 90.0}},
        "nanocodec-h264": {"ms_per_frame": {"a": 1.0, "b": 1.0}}}}))
    return path


def test_x86_merge_scales_the_per_round_ratio_to_the_stored_anchor(tmp_path, stored):
    # the median per-round ratio, not the ratio of the separate medians (9/18, 18/36)
    record = timing_record(tmp_path, "nanocodec-h264", {"a": 9.0, "b": 18.0},
                           {"a": 18.0, "b": 36.0},
                           {"a": {"x264-medium": 0.45, "svtav1-m13": 1.5},
                            "b": {"x264-medium": 0.55, "svtav1-m13": 0.9}})
    chart_refresh.merge_timing(stored, [record], rescale=True)
    points = json.loads(stored.read_text())["points"]
    entry = points["nanocodec-h264"]
    assert entry["ms_per_frame"] == pytest.approx({"a": 9.0, "b": 22.0})
    assert entry["anchor_ms_per_frame"] == {"a": 20.0, "b": 40.0}
    assert entry["raw_ms_per_frame"] == {"a": 9.0, "b": 18.0}
    assert entry["same_session_time_ratio"] == {"svtav1-m13": {"a": 1.5, "b": 0.9}}
    assert points["x265-medium"]["ms_per_frame"] == {"a": 80.0, "b": 90.0}  # untouched


def test_jetson_merge_keeps_each_points_paired_anchor(tmp_path, stored):
    record = timing_record(tmp_path, "nanocodec-h264", {"a": 9.0, "b": 18.0},
                           {"a": 18.0, "b": 36.0})
    chart_refresh.merge_timing(stored, [record], rescale=False)
    entry = json.loads(stored.read_text())["points"]["nanocodec-h264"]
    assert entry["ms_per_frame"] == {"a": 9.0, "b": 18.0}
    assert entry["anchor_ms_per_frame"] == {"a": 18.0, "b": 36.0}


def test_build_note_names_the_families_on_other_builds(tmp_path):
    record = {"base": "aaaa1111", "families": {"nanocodec-h264": "bbbb2222",
                                               "nanocodec-h264@h264-fast": "bbbb2222",
                                               "nanocodec-av1": "aaaa1111"},
              "builds": {"aaaa1111": "aaaa1111", "bbbb2222": "bbbb2222 (landed as cccc3333)"},
              "tools": {"x264": "0.165"}, "x86_timing_note": ""}
    (tmp_path / "builds.json").write_text(json.dumps(record))
    args = argparse.Namespace(commit="HEAD", families="nanocodec-h264", retime="", workers=1,
                              dry_run=True, cpus="9", clipset=None, seed=False,
                              cloud_budget=None)
    with patch.object(chart_refresh, "BUILDS_FILE", str(tmp_path / "builds.json")), \
         patch.object(chart_refresh, "WORK", tmp_path / "work"), \
         patch.object(chart_refresh, "sources_equal", return_value=False):
        refresh = chart_refresh.Refresh(args)
    meta = refresh.meta()
    assert meta["build"] == "aaaa1111 (H.264, H.264 Fast: bbbb2222 (landed as cccc3333))"
    assert meta["x264"] == "0.165"
    assert refresh.stale == refresh.targets == ["nanocodec-h264"]


def test_jetson_build_is_stamped_and_refused_unless_it_is_the_commit():
    refresh = chart_refresh.Refresh.__new__(chart_refresh.Refresh)
    refresh.commit = chart_refresh.git("rev-parse", "--short=8", "HEAD")
    full = chart_refresh.git("rev-parse", "HEAD")
    with patch.object(chart_refresh, "jetson", lambda: "jetson"):
        command = refresh.jetson_build_command()
    assert f"NC_BUILD_COMMIT={full}" in command
    assert command.index(full + "$") < command.index("/COMMIT")  # checked before the marker
