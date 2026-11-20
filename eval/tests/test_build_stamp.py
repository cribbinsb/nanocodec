"""tools/dev/build_stamp.py: a box build of local changes gets a distinct stamp."""
import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "dev" / "build_stamp.py"
spec = importlib.util.spec_from_file_location("build_stamp", SCRIPT)
build_stamp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_stamp)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def repo_with_a_crate(tmp_path):
    (tmp_path / "crates").mkdir()
    (tmp_path / "crates" / "lib.rs").write_text("fn a() {}\n")
    (tmp_path / "notes.md").write_text("notes\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_a_clean_tree_has_no_diff_identity(tmp_path):
    repo = repo_with_a_crate(tmp_path)
    assert build_stamp.diff_identity(repo) == ""
    (repo / "notes.md").write_text("other notes\n")  # not a build input
    assert build_stamp.diff_identity(repo) == ""


def test_each_variant_has_its_own_identity(tmp_path):
    repo = repo_with_a_crate(tmp_path)
    source = repo / "crates" / "lib.rs"
    source.write_text("fn b() {}\n")
    variant_b = build_stamp.diff_identity(repo)
    source.write_text("fn c() {}\n")
    variant_c = build_stamp.diff_identity(repo)
    assert variant_b and variant_c and variant_b != variant_c
    assert len(variant_b) == build_stamp.DIFF_DIGITS
    source.write_text("fn b() {}\n")
    assert build_stamp.diff_identity(repo) == variant_b


def test_untracked_sources_count(tmp_path):
    repo = repo_with_a_crate(tmp_path)
    new = repo / "crates" / "new.rs"
    new.write_text("fn n() {}\n")
    first = build_stamp.diff_identity(repo)
    new.write_text("fn m() {}\n")
    assert first and build_stamp.diff_identity(repo) not in ("", first)
