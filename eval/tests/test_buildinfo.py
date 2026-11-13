"""nceval.buildinfo: the binary's commit, --expect-commit refusals, stale-build warnings."""


import pytest

from nceval import buildinfo


def fake_binary(tmp_path, line):
    path = tmp_path / "nanocodec"
    path.write_text(f"#!/bin/sh\necho '{line}'\n")
    path.chmod(0o755)
    return path


def test_reads_commit_and_dirty_flag(tmp_path):
    info = buildinfo.info_of(fake_binary(tmp_path, "nanocodec 0.1.0 abc123-dirty"))
    assert (info["commit"], info["dirty"]) == ("abc123", True)


def test_an_old_binary_has_no_commit(tmp_path):
    binary = tmp_path / "old"
    binary.write_text("#!/bin/sh\nexit 2\n")
    binary.chmod(0o755)
    assert buildinfo.info_of(binary)["commit"] is None


def test_expect_commit_refuses_another_build(tmp_path, monkeypatch):
    monkeypatch.setattr(buildinfo, "staleness", lambda commit: None)
    monkeypatch.setattr(buildinfo, "_resolve", lambda revision: None)
    binary = fake_binary(tmp_path, "nanocodec 0.1.0 abc123")
    assert buildinfo.check(binary, expect="abc")["commit"] == "abc123"
    with pytest.raises(SystemExit, match="expected def"):
        buildinfo.check(binary, expect="def")
    monkeypatch.setenv(buildinfo.EXPECT_ENV, "def")
    with pytest.raises(SystemExit):
        buildinfo.check(binary)
    buildinfo.check(binary, measured=False)  # an anchor is old on purpose


def test_stale_build_is_flagged(monkeypatch):
    answers = {("rev-parse", "HEAD"): "head1", ("cat-file", "-e", "old1^{commit}"): ""}

    def git(*args):
        if args[0] == "diff":
            return "crates/nc-h264/src/lib.rs"
        return answers.get(args)
    monkeypatch.setattr(buildinfo, "_git", git)
    assert "STALE" in buildinfo.staleness("old1")
    assert buildinfo.staleness("head1") is None


def test_stamp_problem_needs_the_exact_clean_commit():
    line = "nanocodec 0.1.0 abc123def"
    assert buildinfo.stamp_problem(buildinfo.parse_version(line), "abc123") is None
    assert "not commit" in buildinfo.stamp_problem(buildinfo.parse_version(line), "fff")
    assert "not commit" in buildinfo.stamp_problem(
        buildinfo.parse_version("nanocodec 0.1.0 unknown"), "abc")
    assert "dirty" in buildinfo.stamp_problem(
        buildinfo.parse_version(line + "-dirty"), "abc123")


def test_verify_built_refuses_a_stale_build(tmp_path):
    binary = fake_binary(tmp_path, "nanocodec 0.1.0 old111")
    with pytest.raises(SystemExit, match="refusing to use it"):
        buildinfo.verify_built(binary, "new222")
