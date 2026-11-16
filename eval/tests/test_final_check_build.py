"""nceval.levels final-check: every arm's build is the commit asked for, and two
arms that cannot differ are never reported as a +0.00% result."""

import argparse
import contextlib
from pathlib import Path

import pytest

from nceval import buildinfo, levels

OLD, NEW = "a" * 40, "b" * 40


def stamped(path: Path, commit: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho 'nanocodec 0.1.0 {commit}'\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def fake_build(tmp_path, monkeypatch):
    """levels.build with git and cargo faked: cargo writes a binary stamped
    `cargo_stamp[0]` (a stale build when it is not the commit asked for)."""
    cargo_stamp, calls = [NEW], []

    def git(cwd, *args):
        calls.append(args[0])
        if args[0] == "checkout":
            (cwd / "presets").mkdir(parents=True, exist_ok=True)
        return NEW if args[0] == "rev-parse" else ""

    def cargo_build(tree, target):
        calls.append("cargo")
        return stamped(target / "release" / "nanocodec", cargo_stamp[0])

    monkeypatch.setattr(levels, "git", git)
    monkeypatch.setattr(levels, "cargo_build", cargo_build)
    monkeypatch.setattr(levels, "tree_lock", lambda tree: contextlib.nullcontext())
    buildinfo.build_info.cache_clear()
    return argparse.Namespace(data_root=tmp_path), cargo_stamp, calls


def test_a_stale_cached_build_is_rebuilt(fake_build):
    paths, _, calls = fake_build
    cached = stamped(paths.data_root / "builds" / NEW[:12] / "nanocodec", OLD)
    binary, commit = levels.build("HEAD", paths, "new")
    assert (binary, commit) == (cached, NEW) and "cargo" in calls
    assert buildinfo.info_of(binary)["commit"] == NEW


def test_a_build_that_is_not_the_commit_fails_loudly(fake_build):
    paths, cargo_stamp, _ = fake_build
    cargo_stamp[0] = OLD
    with pytest.raises(SystemExit, match="refusing to use it"):
        levels.build("HEAD", paths, "new")
    assert not (paths.data_root / "builds" / NEW[:12] / "nanocodec").exists()


def test_byte_identical_arms_with_different_code_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(levels, "same_code", lambda old, new: False)
    old = stamped(tmp_path / "old" / "nanocodec", OLD)
    new = stamped(tmp_path / "new" / "nanocodec", OLD)
    with pytest.raises(SystemExit, match="byte-identical"):
        levels.no_change_reason((old, OLD), (new, NEW))
    stamped(new, NEW)
    assert levels.no_change_reason((old, OLD), (new, NEW)) is None


def test_arms_with_identical_code_report_no_code_change(tmp_path, monkeypatch):
    monkeypatch.setattr(levels, "same_code", lambda old, new: True)
    binary = stamped(tmp_path / "nanocodec", OLD)
    assert "no code change" in levels.no_change_reason((binary, OLD), (binary, NEW))


def test_same_code_compares_the_crates_and_presets_trees():
    head = levels.git(Path(levels.__file__).parents[2], "rev-parse", "HEAD")
    assert levels.same_code(head, head)
    assert not levels.same_code("", head)


# --- each arm's own presets (2026-11-16: two bare binaries both read the repo's
# presets, and a preset change came out at J exactly 0.00 %) ----------------------------

def preset_dir(root: Path, value: str) -> Path:
    (root / "presets" / "av1").mkdir(parents=True)
    (root / "presets" / "av1" / "fast.toml").write_text(f"decision.noise_skip = {value}\n")
    return root / "presets"


def test_each_binary_reads_the_presets_beside_it(tmp_path):
    paths = argparse.Namespace(data_root=tmp_path)
    old, new = (stamped(tmp_path / arm / "nanocodec", "") for arm in ("old", "new"))
    preset_dir(old.parent, "1")
    preset_dir(new.parent, "2")
    args = [levels.preset_args(levels.presets_of(b, paths), levels.Codec("av1"), "fast")
            for b in (old, new)]
    assert args == [["decision.noise_skip=1"], ["decision.noise_skip=2"]]


def test_a_bare_dirty_binary_without_presets_fails_loudly(tmp_path):
    buildinfo.build_info.cache_clear()
    binary = stamped(tmp_path / "nanocodec", OLD + "-dirty")
    with pytest.raises(SystemExit, match="cannot tell which presets"):
        levels.presets_of(binary, argparse.Namespace(data_root=tmp_path))


def test_a_bare_clean_binary_gets_its_commits_presets_from_git(tmp_path):
    buildinfo.build_info.cache_clear()
    head = levels.git(Path(levels.__file__).parents[2], "rev-parse", "HEAD")
    binary = stamped(tmp_path / "bin" / "nanocodec", head)
    presets = levels.presets_of(binary, argparse.Namespace(data_root=tmp_path))
    assert presets == tmp_path / "builds" / head[:12] / "presets"
    assert (presets / "av1").is_dir()


def test_identical_presets_of_different_dirty_builds_are_refused(tmp_path):
    buildinfo.build_info.cache_clear()
    old = stamped(tmp_path / "old" / "nanocodec", OLD + "-dirty")
    new = stamped(tmp_path / "new" / "nanocodec", NEW + "-dirty")
    arms = ((old, preset_dir(old.parent, "1")), (new, preset_dir(new.parent, "1")))
    with pytest.raises(SystemExit, match="identical preset files"):
        levels.check_presets(*arms, allow_same=False)
    assert "only code differs" in levels.check_presets(*arms, allow_same=True)


def test_different_presets_pass_the_check(tmp_path):
    old = stamped(tmp_path / "old" / "nanocodec", OLD + "-dirty")
    new = stamped(tmp_path / "new" / "nanocodec", NEW + "-dirty")
    arms = ((old, preset_dir(old.parent, "1")), (new, preset_dir(new.parent, "2")))
    assert levels.check_presets(*arms, allow_same=False) is None


def test_one_binary_with_two_different_preset_dirs_is_a_preset_only_comparison(
        tmp_path, monkeypatch):
    monkeypatch.setattr(levels, "same_code", lambda old, new: False)
    old = stamped(tmp_path / "old" / "nanocodec", OLD + "-dirty")
    new = stamped(tmp_path / "new" / "nanocodec", OLD + "-dirty")  # byte-identical copy
    old_presets, new_presets = preset_dir(old.parent, "1"), preset_dir(new.parent, "2")
    differ = not levels.same_tree(old_presets, new_presets)
    assert levels.no_change_reason((old, ""), (new, ""), presets_differ=differ) is None
    assert levels.check_presets((old, old_presets), (new, new_presets), allow_same=False) is None


def test_one_binary_with_identical_preset_dirs_is_still_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(levels, "same_code", lambda old, new: False)
    old = stamped(tmp_path / "old" / "nanocodec", OLD + "-dirty")
    new = stamped(tmp_path / "new" / "nanocodec", OLD + "-dirty")
    differ = not levels.same_tree(preset_dir(old.parent, "1"), preset_dir(new.parent, "1"))
    with pytest.raises(SystemExit, match="byte-identical"):
        levels.no_change_reason((old, ""), (new, ""), presets_differ=differ)
