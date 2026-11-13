"""tools/dev/land.sh's merge train, on a toy repository with stub checks.

Three branches queue while another lander holds the lock; the next holder
lands them as one train. With one branch breaking the (stub) checks, the
train is bisected: the branches before it land, it is refused with its log,
and the one after it lands in the next train.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = ("tools/dev/land.sh", "tools/dev/cargo-capped.sh")
#: The plan every candidate gets: the stub check.sh only (land_scope.py format).
PLAN = "rust all\\neval 0\\nweb 0\\nconformance 0\\ndocs\\npython\\nshell\\ntoml\\nfuzz"
QUEUE_TIMEOUT_SECONDS = 60


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def toy(tmp_path):
    """A repository on `integ` with land.sh, stub scope/guard/checks, and its env."""
    if not shutil.which("flock"):
        pytest.skip("flock(1) not available")
    repo = tmp_path / "repo"
    git(tmp_path, "init", "-q", "-b", "integ", str(repo))
    for script in SCRIPTS:
        (repo / script).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / script, repo / script)
    (repo / "tools/dev/land_scope.py").write_text(f'print("{PLAN}")\n')
    (repo / "tools/dev/land_guard.py").write_text("import sys\nsys.exit(0)\n")
    check = repo / "tools/ci/check.sh"
    check.parent.mkdir(parents=True)
    check.write_text("#!/usr/bin/env bash\n[[ ! -e BROKEN ]]\n")
    check.chmod(0o755)
    env = {**os.environ, "NC_LAND_TARGET": "integ", "NC_INTEGRATION_TREE": str(repo),
           "NC_LANDER_TREE": str(tmp_path / "lander"), "NC_LAND_LOCK": str(tmp_path / "run/lock"),
           "NC_LAND_QUEUE": str(tmp_path / "run/queue"), "NC_LAND_TARGET_DIR": str(tmp_path / "t"),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    git(repo, "add", "-A")
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, env=env, check=True)
    (tmp_path / "run").mkdir()
    return tmp_path, repo, env


def branch(toy, name: str, file: str) -> Path:
    tmp, repo, env = toy
    tree = tmp / name
    git(repo, "worktree", "add", "-q", "-b", name, str(tree), "integ")
    (tree / file).write_text(name)
    git(tree, "add", "-A")
    subprocess.run(["git", "commit", "-qm", name], cwd=tree, env=env, check=True)
    return tree


def land_queued(toy, trees: list[Path]) -> list[subprocess.CompletedProcess]:
    """Run land.sh in every tree while a stand-in lander holds the lock, released once
    every one of them has queued its ticket (in order)."""
    tmp, _, env = toy
    release = hold_lock(tmp)
    procs = []
    for count, tree in enumerate(trees, 1):
        procs.append(subprocess.Popen(["bash", "tools/dev/land.sh"], cwd=tree, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        wait_for(lambda: len(list((tmp / "run/queue").glob("*.ticket"))) >= count)
    release()
    return [subprocess.CompletedProcess(p.args, p.wait(), p.stdout.read()) for p in procs]


def hold_lock(tmp: Path):
    """A stand-in lander holding the landing lock; returns the function that releases it."""
    flag = tmp / "run/release"
    holder = subprocess.Popen(["flock", str(tmp / "run/lock"), "bash", "-c",
                               f"until [ -e {flag} ]; do sleep 0.1; done"])

    def release() -> None:
        flag.touch()
        holder.wait()
    return release


def wait_for(condition) -> None:
    deadline = time.monotonic() + QUEUE_TIMEOUT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)


def test_queued_branches_land_as_one_train(toy):
    trees = [branch(toy, name, f"f_{name}") for name in ("a", "b", "c")]
    results = land_queued(toy, trees)
    assert [r.returncode for r in results] == [0, 0, 0], [r.stdout for r in results]
    assert sum("== train 3" in r.stdout for r in results) == 1
    _, repo, _ = toy
    assert git(repo, "log", "--format=%s", "integ").split()[:3] == ["c", "b", "a"]
    # each landed branch was moved onto the integration branch
    assert all(subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "integ"],
                              cwd=t).returncode == 0 for t in trees)


def test_a_failing_branch_is_bisected_out_and_the_rest_land(toy):
    trees = [branch(toy, "a", "f_a"), branch(toy, "b", "BROKEN"), branch(toy, "c", "f_c")]
    results = land_queued(toy, trees)
    assert [r.returncode for r in results] == [0, 1, 0], [r.stdout for r in results]
    assert "refused" in results[1].stdout
    assert any("bisect" in r.stdout for r in results)
    _, repo, _ = toy
    landed = git(repo, "log", "--format=%s", "integ").split()
    assert "b" not in landed and {"a", "c"} <= set(landed)


def blocked_waiters(lock: Path) -> int:
    inode = f":{lock.stat().st_ino} "
    return sum("->" in line and inode in line for line in Path("/proc/locks").read_text().splitlines())


def test_an_older_lander_waiting_for_the_lock_is_adopted_into_the_queue(toy):
    """A land.sh from before the queue (no ticket) blocked on the lock gets a ticket
    with its branch and HEAD (land.sh's adopt_waiters, run as the lock holder would)."""
    tmp, repo, env = toy
    lock = tmp / "run/lock"
    lock.touch()
    old_script = tmp / "old-land.sh"  # stands in for the older land.sh: it waits for the lock
    old_script.write_text('exec 9>>"$NC_LAND_LOCK"; flock 9\n')
    old = branch(toy, "old", "f_old")
    release = hold_lock(tmp)
    waiter = subprocess.Popen(["bash", str(old_script)], cwd=old, env=env)
    wait_for(lambda: blocked_waiters(lock) >= 1)
    queue = tmp / "run/queue"
    queue.mkdir()
    function = subprocess.run(["sed", "-n", "/^adopt_waiters()/,/^}/p", str(repo / SCRIPTS[0])],
                              capture_output=True, text=True, check=True).stdout
    adopted = subprocess.run(["bash", "-c", f'{function}\nlock={lock} queue={queue} adopt_waiters'],
                             capture_output=True, text=True)
    release()
    waiter.wait()
    tickets = list(queue.glob("*.ticket"))
    assert len(tickets) == 1, adopted.stdout + adopted.stderr
    name, sha, pid = tickets[0].read_text().split()
    assert (name, sha, int(pid)) == ("old", git(old, "rev-parse", "HEAD"), waiter.pid)
