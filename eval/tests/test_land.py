"""tools/dev/land.sh on a toy repository with stub checks (no lock, no queue).

Two branches landing at the same time both land (the compare-and-swap loser
rebases and retries); the revert guard only warns, in the commit message; a
stale integration worktree is moved to the new head.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = ("tools/dev/land.sh", "tools/dev/cargo-capped.sh")
#: The plan every candidate gets (land_scope.py format): no Rust, nothing to compile.
PLAN = "rust none\\ntouched\\npython\\nshell"
LAND_TIMEOUT_SECONDS = 25


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def toy(tmp_path):
    """A repository on `integ` with land.sh, stub scope and guard, and its env."""
    repo = tmp_path / "repo"
    git(tmp_path, "init", "-q", "-b", "integ", str(repo))
    for script in SCRIPTS:
        (repo / script).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / script, repo / script)
    # the scope stub takes a second, so concurrent landers overlap
    (repo / "tools/dev/land_scope.py").write_text(f'import time\ntime.sleep(1)\nprint("{PLAN}")\n')
    (repo / "tools/dev/land_guard.py").write_text(
        "import os, sys\n"
        "if os.path.exists('STALE'):\n"
        "    sys.exit(print('land: REFUSED: the candidate undoes 1 upstream change(s)', file=sys.stderr) or 1)\n")
    env = {**os.environ, "NC_LAND_TARGET": "integ", "NC_INTEGRATION_TREE": str(repo),
           "NC_DATA_ROOT": str(tmp_path / "data"), "CARGO_TARGET_DIR": str(tmp_path / "t"),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    git(repo, "add", "-A")
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, env=env, check=True)
    return tmp_path, repo, env


def branch(toy, name: str, file: str) -> Path:
    tmp, repo, env = toy
    tree = tmp / name
    git(repo, "worktree", "add", "-q", "-b", name, str(tree), "integ")
    (tree / file).write_text(name)
    git(tree, "add", "-A")
    subprocess.run(["git", "commit", "-qm", name], cwd=tree, env=env, check=True)
    return tree


def land(toy, trees: list[Path]) -> list[subprocess.CompletedProcess]:
    """land.sh in every tree at once."""
    _, _, env = toy
    procs = [subprocess.Popen(["bash", "tools/dev/land.sh"], cwd=tree, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) for tree in trees]
    try:
        return [subprocess.CompletedProcess(p.args, p.wait(LAND_TIMEOUT_SECONDS), p.stdout.read())
                for p in procs]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()


def test_concurrent_landers_both_land_and_the_stale_integration_tree_follows(toy):
    trees = [branch(toy, "a", "f_a"), branch(toy, "b", "f_b")]
    results = land(toy, trees)
    assert [r.returncode for r in results] == [0, 0], [r.stdout for r in results]
    _, repo, _ = toy
    assert set(git(repo, "log", "--format=%s", "integ").split()[:2]) == {"a", "b"}
    # the integration worktree (integ checked out) holds the new head's files, clean
    assert git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert (repo / "f_a").exists() and (repo / "f_b").exists()


def test_the_revert_guard_warns_in_the_commit_instead_of_refusing(toy):
    tree = branch(toy, "stale", "STALE")
    [result] = land(toy, [tree])
    assert result.returncode == 0, result.stdout
    assert "WARNING" in result.stdout
    _, repo, _ = toy
    assert "Land-warning: undoes 1 upstream change(s)" in git(repo, "log", "-1", "--format=%B", "integ")


def test_a_rebase_conflict_is_refused_and_leaves_the_branch_as_it_was(toy):
    trees = [branch(toy, "a", "same"), branch(toy, "b", "same")]
    assert land(toy, trees[:1])[0].returncode == 0
    head = git(trees[1], "rev-parse", "HEAD")
    [result] = land(toy, trees[1:])
    assert result.returncode == 1 and "rebase conflict" in result.stdout
    assert git(trees[1], "rev-parse", "HEAD") == head
