"""tools/ci/check_names.py: the same answer from git's file list and from a plain tree."""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "ci"))
import check_names  # noqa: E402


def tree(root: Path) -> None:
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "a.py").write_text("x = 1\n")
    (root / "tools" / "b.sh").write_text("echo\n")
    (root / "target").mkdir()
    (root / "target" / "built.py").write_text("x = 1\n")


def test_files_are_listed_with_git_and_without(tmp_path):
    plain, repo = tmp_path / "plain", tmp_path / "repo"
    tree(plain)
    shutil.copytree(plain, repo)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "tools"], check=True)
    names = lambda root: [p.relative_to(root).as_posix()
                          for p in check_names.tracked(".py", root)]
    assert names(repo) == ["tools/a.py"]  # git's list
    assert names(plain) == ["tools/a.py"]  # the walk: target/ left out, as the sync does
