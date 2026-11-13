"""land_guard refuses a stale-tree landing (2fae95f) and accepts the candidate's own edits.

True positives build a stale tree the way 2fae95f did: the agent's branch
forks, the integration branch moves, and the agent squashes with `git reset
--soft integration`, keeping its old tree. False positives replay, in
miniature, the landings the earlier hunk-matching guard refused on 2026-11-09
(each named in its test).
"""
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[2] / "tools" / "dev" / "land_guard.py"
LAND = GUARD.with_name("land.sh")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, files: dict, message: str):
    for name, text in files.items():
        if text is None:
            (repo / name).unlink()
        else:
            (repo / name).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)


def guard(repo):
    return subprocess.run([sys.executable, str(GUARD), "integration"], cwd=repo,
                          capture_output=True, text=True)


def new_repo(tmp_path, files: dict):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "integration")
    commit(repo, files, "base")
    return repo


def land(repo, upstream: list[tuple[dict, str]], mine: dict, stale: bool):
    """Fork `agent`, land `upstream` on integration, then commit `mine` on the agent branch.

    stale=False rebases the agent first (land.sh's path); stale=True squashes
    with `reset --soft` instead, so the agent's commit keeps its old tree.
    """
    git(repo, "checkout", "-qb", "agent")
    git(repo, "checkout", "-q", "integration")
    for files, message in upstream:
        commit(repo, files, message)
    git(repo, "checkout", "-q", "agent")
    if stale:
        for name, text in mine.items():
            (repo / name).write_text(text)
        git(repo, "add", "-A")
        git(repo, "reset", "--soft", "integration")
        git(repo, "commit", "-qm", "agent: squashed")
    else:
        git(repo, "rebase", "-q", "integration")
        commit(repo, mine, "agent: work")


def function(name: str, body: list[str]) -> str:
    """A Rust-like function that ends like every other (as real ones do: `record(..);`, `}`).

    Alike ends are what made the earlier hunk-matching guard find an upstream
    hunk's context again after an edit inside the hunk.
    """
    return (f"fn {name}() {{\n" + "".join(f"    {line}\n" for line in body)
            + "    record(report);\n}\n")


def lines(name: str, count: int) -> list[str]:
    return [f"let {name}_{i} = step({i});" for i in range(count)]


# -- true positives: a stale tree undoes upstream work ---------------------------------

def scenario(tmp_path):
    """An agent branch forks; integration then gains another agent's change."""
    repo = new_repo(tmp_path, {"a.txt": "one\ntwo\nthree\n", "b.txt": "base\n"})
    git(repo, "checkout", "-qb", "agent")
    commit(repo, {"b.txt": "base\nagent work\n"}, "agent: b")
    git(repo, "checkout", "-q", "integration")
    commit(repo, {"a.txt": "one\nTWO (other agent)\nthree\n"}, "other agent: a")
    git(repo, "checkout", "-q", "agent")
    return repo


def test_reset_soft_squash_onto_moved_integration_is_refused(tmp_path):
    repo = scenario(tmp_path)
    git(repo, "reset", "--soft", "integration")
    git(repo, "commit", "-qm", "agent: squashed")
    result = guard(repo)
    assert result.returncode == 1
    assert "a.txt: reverts" in result.stderr and "other agent: a" in result.stderr
    assert "b.txt" not in result.stderr  # the agent's own change is fine


def test_a_tree_stale_by_two_commits_is_refused(tmp_path):
    """The candidate's text predates both upstream edits of the same lines."""
    base = function("f", lines("a", 4)) + "\n" + function("g", lines("b", 4))
    first = base.replace("let a_1 = step(1);", "let a_1 = faster(1);")
    second = first.replace("let a_1 = faster(1);", "let a_1 = fastest(1);")
    repo = new_repo(tmp_path, {"lib.rs": base, "mine.rs": "x\n"})
    land(repo, [({"lib.rs": first}, "upstream: faster"), ({"lib.rs": second}, "upstream: fastest")],
         {"mine.rs": "x\ny\n"}, stale=True)
    result = guard(repo)
    assert result.returncode == 1
    assert "upstream: fastest" in result.stderr


def test_a_stale_tree_dropping_an_added_file_and_restoring_a_deleted_one_is_refused(tmp_path):
    old = function("old", lines("o", 5))
    repo = new_repo(tmp_path, {"old.rs": old, "mine.rs": "x\n"})
    git(repo, "checkout", "-qb", "agent")
    git(repo, "checkout", "-q", "integration")
    commit(repo, {"new.rs": function("new", lines("n", 5)), "old.rs": None},
           "upstream: replace old.rs by new.rs")
    git(repo, "checkout", "-q", "agent")
    commit(repo, {"mine.rs": "x\ny\n"}, "agent: work")
    git(repo, "reset", "--soft", "integration")
    git(repo, "commit", "-qm", "agent: squashed")
    result = guard(repo)
    assert result.returncode == 1
    assert "new.rs: reverts" in result.stderr and "old.rs: reverts" in result.stderr


def test_a_stale_tree_putting_back_deleted_lines_is_refused(tmp_path):
    base = function("f", lines("a", 3) + ["legacy_call();"] + lines("b", 3))
    repo = new_repo(tmp_path, {"lib.rs": base, "mine.rs": "x\n"})
    land(repo, [({"lib.rs": base.replace("    legacy_call();\n", "")}, "upstream: drop legacy")],
         {"mine.rs": "x\ny\n"}, stale=True)
    result = guard(repo)
    assert result.returncode == 1 and "upstream: drop legacy" in result.stderr


def test_a_stale_tree_is_refused_where_the_agent_also_edited_one_of_the_hunks(tmp_path):
    """2fae95f: the author's own edits overlapped some of the reverted commits' hunks."""
    parts = [function(name, lines(name, 4)) for name in "pqr"]
    base = "\n".join(parts)
    upstream = base.replace("step(1)", "tuned(1)")  # one hunk in each function
    repo = new_repo(tmp_path, {"lib.rs": base})
    mine = base.replace("let r_1 = step(1);", "let r_1 = mine(1);")
    land(repo, [({"lib.rs": upstream}, "upstream: tune")], {"lib.rs": mine}, stale=True)
    result = guard(repo)
    assert result.returncode == 1 and "upstream: tune" in result.stderr


# -- the candidate's own work passes ---------------------------------------------------

def test_a_proper_rebase_passes(tmp_path):
    repo = scenario(tmp_path)
    git(repo, "rebase", "-q", "integration")
    result = guard(repo)
    assert result.returncode == 0, result.stderr
    assert "b.txt" in result.stdout  # the file list is printed


def test_an_announced_revert_passes(tmp_path):
    repo = scenario(tmp_path)
    git(repo, "rebase", "-q", "integration")
    upstream = git(repo, "rev-parse", "integration")
    commit(repo, {"a.txt": "one\ntwo\nthree\n"}, f"revert the a.txt change\n\nReverts: {upstream[:9]}")
    assert guard(repo).returncode == 0


def test_an_upstream_change_already_reverted_upstream_is_not_flagged(tmp_path):
    repo = scenario(tmp_path)
    git(repo, "checkout", "-q", "integration")
    commit(repo, {"a.txt": "one\ntwo\nthree\n"}, "other agent: revert a")
    git(repo, "checkout", "-q", "agent")
    git(repo, "rebase", "-q", "integration")
    commit(repo, {"a.txt": "one\ntwo\nthree\nfour\n"}, "agent: extend a")
    assert guard(repo).returncode == 0


def body(name):
    # Test functions end alike, so the end of the file looks the same after any append.
    return f"def {name}():\n    setup()\n    run()\n    assert ok\n"


@pytest.mark.parametrize("ours_first", [False, True])
def test_appending_next_to_an_upstream_append_is_not_a_revert(tmp_path, ours_first):
    """realuse2: both sides appended at the end of the same file, conflict resolved by hand."""
    base = body("a") + "\n\n" + body("b")
    repo = new_repo(tmp_path, {"t.py": base})
    git(repo, "checkout", "-qb", "agent")
    commit(repo, {"t.py": base + "\n\n" + body("mine")}, "agent: append a test")
    git(repo, "checkout", "-q", "integration")
    theirs = "def theirs():\n    check(1)\n"  # ends unlike the others
    commit(repo, {"t.py": base + "\n\n" + theirs}, "other agent: append a test")
    git(repo, "checkout", "-q", "agent")
    subprocess.run(["git", "-C", str(repo), "rebase", "-q", "integration"], capture_output=True)
    tail = [body("mine"), theirs] if ours_first else [theirs, body("mine")]
    (repo / "t.py").write_text(base + "\n\n" + "\n\n".join(tail))
    git(repo, "add", "t.py")
    git(repo, "-c", "core.editor=true", "rebase", "--continue")
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "agent"
    result = guard(repo)
    assert result.returncode == 0, result.stderr


def test_editing_lines_an_upstream_commit_added_is_not_a_revert(tmp_path):
    """Our own follow-up edit to lines just landed upstream (f1f249e editing 2de4680)."""
    repo = scenario(tmp_path)
    git(repo, "rebase", "-q", "integration")
    commit(repo, {"a.txt": "one\nTWO (other agent, improved)\nthree\n"}, "agent: refine a")
    git(repo, "checkout", "-q", "integration")
    commit(repo, {"c.txt": "x\n"}, "unrelated")
    commit(repo, {"a.txt": "one\nTWO (other agent)\nthree\nfour\n"}, "other agent: append to a")
    git(repo, "checkout", "-q", "agent")
    subprocess.run(["git", "-C", str(repo), "rebase", "-q", "integration"], capture_output=True)
    (repo / "a.txt").write_text("one\nTWO (other agent, improved)\nthree\nfour\n")
    git(repo, "add", "a.txt")
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "core.editor=true", "rebase", "--continue"], capture_output=True)
    result = guard(repo)
    assert result.returncode == 0, result.stderr


def test_editing_inside_a_large_block_upstream_added_passes(tmp_path):
    """av1-dec: edits inside block_info.rs code 41db64cf had just added."""
    first, last = function("first", lines("f", 3)), function("last", lines("l", 3))
    added = function("transform_context", lines("t", 12))
    edited = added.replace("step(5)", "step(5).min(limit)").replace("step(6)", "clamp(6)")
    repo = new_repo(tmp_path, {"block_info.rs": first + "\n" + last})
    land(repo, [({"block_info.rs": first + "\n" + added + "\n" + last}, "upstream: tx context")],
         {"block_info.rs": first + "\n" + edited + "\n" + last}, stale=False)
    result = guard(repo)
    assert result.returncode == 0, result.stderr


def test_rewriting_a_function_upstream_added_passes(tmp_path):
    """av1-dec: predict_compound / predict_one_of_two from 1dbf0fc3 rewritten in place."""
    head = function("head", lines("h", 3))
    added = function("predict_compound", lines("c", 6)) + "\n" + function("predict_one", lines("o", 6))
    repo = new_repo(tmp_path, {"inter.rs": head})
    rewritten = function("predict_compound", ["blend(a, b);"]) + "\n" + function("predict_one", lines("o", 6))
    land(repo, [({"inter.rs": head + "\n" + added}, "upstream: compound")],
         {"inter.rs": head + "\n" + rewritten}, stale=False)
    result = guard(repo)
    assert result.returncode == 0, result.stderr


def test_extending_a_function_upstream_inserted_passes(tmp_path):
    """av2-next3: a check appended to the function 5b8cc452 added (check_codec_txfm.rs)."""
    first, last = function("check_hevc", lines("h", 4)), function("check_quant", lines("q", 4))
    added = function("check_av2", lines("d", 4))
    extended = added.replace("    record(report);\n}", "    record(report);\n"
                             "    let corner = corner_check();\n    record(report);\n}")
    repo = new_repo(tmp_path, {"check.rs": first + "\n" + last})
    land(repo, [({"check.rs": first + "\n" + added + "\n" + last}, "upstream: av2 checks")],
         {"check.rs": first + "\n" + extended + "\n" + last}, stale=False)
    result = guard(repo)
    assert result.returncode == 0, result.stderr


def test_moving_a_doc_comment_line_passes(tmp_path):
    """A doc-comment line upstream added moves (d23693c4's case)."""
    code = function("convert", lines("m", 4))
    doc = ["/// The engine's shared intra modes map to ours.", "/// Vertical and horizontal are exact."]
    repo = new_repo(tmp_path, {"modes.rs": code})
    land(repo, [({"modes.rs": "\n".join(doc) + "\n" + code}, "upstream: document convert")],
         {"modes.rs": doc[0] + "\n" + code.replace("    let m_0", f"    {doc[1].replace('///', '//')}\n    let m_0")},
         stale=False)
    result = guard(repo)
    assert result.returncode == 0, result.stderr


def test_dropping_part_of_an_upstream_commit_in_a_rewrite_passes(tmp_path):
    """6a74262c vs 3f5a7dd8: a rewrite restores one old condition of a many-hunk commit."""
    parts = [function(name, lines(name, 4) + [f"if {name}_ready {{ run(); }}"]) for name in "pqrs"]
    base = "\n".join(parts)
    upstream = base.replace("_ready {", "_ready && uniform {")
    mine = upstream.replace("if p_ready && uniform {", "if p_ready {").replace("let q_0", "let q_zero")
    repo = new_repo(tmp_path, {"deblock.rs": base})
    land(repo, [({"deblock.rs": upstream}, "upstream: uniform fast path")], {"deblock.rs": mine},
         stale=False)
    result = guard(repo)
    assert result.returncode == 0, result.stderr


def overlaps(repo, old, new):
    """Run land.sh's `overlaps` function in `repo` (exit 0 = overlap)."""
    function = subprocess.run(["sed", "-n", "/^overlaps()/,/^}/p", str(LAND)], capture_output=True,
                              text=True, check=True).stdout
    return subprocess.run(["bash", "-c", function + f'\noverlaps "{old}" "{new}" HEAD'],
                          cwd=repo).returncode == 0


def test_land_retry_needs_full_checks_only_when_new_upstream_commits_overlap(tmp_path):
    repo = scenario(tmp_path)          # integration changed a.txt; the agent changes b.txt
    git(repo, "rebase", "-q", "integration")
    checked = git(repo, "rev-parse", "integration")
    git(repo, "checkout", "-q", "integration")
    commit(repo, {"c.txt": "unrelated\n"}, "someone: c")
    disjoint = git(repo, "rev-parse", "HEAD")
    commit(repo, {"b.txt": "base\nupstream edit\n"}, "someone: b")
    overlapping = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "agent")
    git(repo, "rebase", "-q", disjoint)
    assert not overlaps(repo, checked, disjoint)  # only c.txt arrived: build check suffices
    subprocess.run(["git", "-C", str(repo), "rebase", "-q", overlapping], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "rebase", "--abort"], capture_output=True)
    assert overlaps(repo, checked, overlapping)  # b.txt arrived: full checks again
