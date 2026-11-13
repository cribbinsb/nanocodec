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
