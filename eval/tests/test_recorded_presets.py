"""nceval.encoders.nanocodec.recorded_values and own_family(recorded=True): a build
record's family names are those of their day, read in their build's own preset files."""
from pathlib import Path

import pytest

from nceval.encoders.nanocodec import FROZEN_BUILD_MARKER, preset_values, recorded_values
from nceval.realuse import own_family


def frozen_build_before_the_slow_rename(root: Path) -> Path:
    """A frozen build (presets/ beside COMMIT) from before 2026-11-21: av1/slow.toml is
    today's `slower`, and no file is today's `slow`."""
    (root / FROZEN_BUILD_MARKER).write_text("20d31869\n")
    presets = root / "presets"
    (presets / "av1").mkdir(parents=True)
    (presets / "av1" / "slow.toml").write_text("planner.depth = 3\n")
    return presets


def test_a_recorded_name_reads_its_builds_own_file(tmp_path):
    presets = frozen_build_before_the_slow_rename(tmp_path)
    assert recorded_values(presets, "av1", "slow") == {"planner.depth": 3.0}
    # today's names on that build: `slower` is its slow.toml, `slow` it does not have
    assert preset_values(presets, "av1", "slower") == {"planner.depth": 3.0}
    with pytest.raises(ValueError):
        preset_values(presets, "av1", "slow")


def test_a_recorded_family_is_the_one_its_points_were_measured_as(tmp_path):
    presets = frozen_build_before_the_slow_rename(tmp_path)
    family = own_family("nanocodec-av1@slow", Path("nanocodec"), presets, recorded=True)
    assert family.extra == ("--param", "planner.depth=3")
    assert own_family("nanocodec-av1", Path("nanocodec"), presets, recorded=True).extra == ()


def test_a_recorded_name_the_build_never_had_is_an_error(tmp_path):
    presets = frozen_build_before_the_slow_rename(tmp_path)
    with pytest.raises(ValueError):
        recorded_values(presets, "av1", "slower")


def history_frame(root: Path, files: dict[str, str]) -> Path:
    """A frozen history build: presets/ beside COMMIT (tools/dev/history_chart.py)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / FROZEN_BUILD_MARKER).write_text("commit\n")
    presets = root / "presets"
    for name, text in files.items():
        (presets / name).parent.mkdir(parents=True, exist_ok=True)
        (presets / name).write_text(text)
    return presets


def ladder_files(presets: Path, codec: str) -> dict[str, Path]:
    """Each of today's rung names the build has -> the file it codes with."""
    from nceval.encoders.nanocodec import preset_path
    stems = ("veryslow", "slower", "slow", "fast", "faster", "veryfast")
    return {s: preset_path(presets, codec, s) for s in stems if preset_path(presets, codec, s).is_file()}


def test_an_old_frames_slow_and_a_new_frames_slow_are_different_series(tmp_path):
    old = history_frame(tmp_path / "old", {"av1/slow.toml": "a = 1\n", "h264/fast.toml": "b = 1\n",
                                           "h264/faster.toml": "c = 1\n"})
    new = history_frame(tmp_path / "new", {"av1/slow.toml": "a = 2\n", "av1/slower.toml": "a = 1\n",
                                           "h264/fast.toml": "b = 2\n",
                                           "h264/faster.toml": "`fast` until 2026-11-21\nb = 1\n"})
    assert ladder_files(old, "av1") == {"slower": old / "av1/slow.toml"}
    assert set(ladder_files(new, "av1")) == {"slower", "slow"}
    # H.264: the old fast is today's faster; the old faster is retired (no series)
    assert ladder_files(old, "h264") == {"faster": old / "h264/fast.toml"}


def test_a_flat_frame_takes_the_renames_too(tmp_path):
    old = history_frame(tmp_path, {"h264-fast.toml": "b = 1\n", "h264-faster.toml": "c = 1\n",
                                   "vvc-fast.toml": "d = 1\n"})
    assert ladder_files(old, "h264") == {"faster": old / "h264-fast.toml"}
    assert ladder_files(old, "vvc") == {"faster": old / "vvc-fast.toml"}
