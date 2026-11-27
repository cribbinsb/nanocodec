"""The retention manifest's gate (guard / check), status, sweep and release pre-flight."""

import os
import time
from pathlib import Path

import pytest

from nceval import retention

MANIFEST = """
grace_hours = 24

[[keep]]
path = "work/realuse/f150-ip300"
class = "durable"
reader = "main chart"
regen = "6.5 CPU-h"
evicted_by = ["chart_refresh"]
release = true
require_globs = ["*/x264-medium/*.json"]

[[keep]]
path = "worktrees/gh-pages"
class = "tool"
reader = "publish.sh"
regen = "git worktree add"

[[keep]]
path = "clips/*.json"
class = "durable"
reader = "clip sets"
regen = "selection scripts"

[[keep]]
path = "charts/builds"
class = "cache"
reader = "binary cache"
regen = "3 min a build"
evicted_by = ["build_cache"]
marker = true

[[regenerable]]
path = "work/realuse/f600-ip300"
reason = "screen cache"

[[regenerable]]
path = "work/old-agent"

[[regenerable]]
path = "work/fresh-agent"

[[regenerable]]
path = "stray.log"
"""

DAY = 86400


def age(path: Path, seconds: float) -> None:
    """Backdate `path` and everything under it by `seconds`."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))
    for dirpath, _, files in os.walk(path):
        for name in files:
            os.utime(os.path.join(dirpath, name), (stamp, stamp))
        os.utime(dirpath, (stamp, stamp))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    manifest = tmp_path / "retention.toml"
    manifest.write_text(MANIFEST)
    monkeypatch.setenv(retention.MANIFEST_ENV, str(manifest))
    root = tmp_path / "data"
    store = root / "work/realuse/f150-ip300/static_cam29/x264-medium"
    store.mkdir(parents=True)
    (store / "cbr300-abc.json").write_text("{}")
    (root / "work/realuse/f600-ip300/clip").mkdir(parents=True)
    (root / "work/realuse/f600-ip300/clip/p.json").write_text("{}" * 100)
    (root / "work/old-agent").mkdir(parents=True)
    (root / "work/old-agent/log.txt").write_text("x")
    (root / "work/fresh-agent").mkdir(parents=True)
    (root / "work/fresh-agent/log.txt").write_text("x")
    (root / "worktrees/gh-pages").mkdir(parents=True)
    (root / "clips").mkdir()
    (root / "clips/main.json").write_text("{}")
    (root / "stray.log").write_text("x")
    (root / "work/undeclared").mkdir()
    (root / "scratch/agent").mkdir(parents=True)
    (root / "charts/builds/abc").mkdir(parents=True)
    for old in ("work/realuse", "work/old-agent", "worktrees/gh-pages", "clips", "stray.log",
                "work/undeclared", "scratch", "charts"):
        age(root / old, 3 * DAY)
    return root


def test_guard_refuses_kept_paths_and_their_ancestors(tree):
    for kept in ("work/realuse/f150-ip300", "work/realuse", "work", "worktrees/gh-pages",
                 "clips/main.json", "clips", "."):
        assert retention.guard(tree / kept, root=tree) is not None, kept
    for free in ("work/realuse/f600-ip300", "work/old-agent", "stray.log", "clips/main"):
        assert retention.guard(tree / free, root=tree) is None, free


def test_check_refuses_unknown_and_passes_declared_scratch_and_targets(tree):
    assert "UNKNOWN" in retention.check(tree / "work/undeclared", root=tree)
    for free in ("work/old-agent", "work/realuse/f600-ip300/clip", "scratch/agent",
                 "targets/agent"):
        assert retention.check(tree / free, root=tree) is None, free
    assert retention.check(tree / "scratch", root=tree) is not None  # every agent's


def test_a_marker_keeps_everything_at_and_below_it_and_its_holders(tree):
    marked = tree / "work/old-agent"
    retention.ensure_marker(marked, "a reader", "a cost")
    assert "a cost" in (marked / retention.MARKER).read_text()
    for refused in ("work/old-agent", "work/old-agent/log.txt", "work"):
        assert retention.check(tree / refused, root=tree) is not None, refused
    assert "work/old-agent" not in {r.rel for r in retention.status(tree)}


def test_write_markers_marks_the_kept_directories(tree):
    done = dict(retention.write_markers(tree))
    assert done[tree / "work/realuse/f150-ip300"] == "written"
    assert (tree / "charts/builds" / retention.MARKER).is_file()
    assert tree / "worktrees/gh-pages" in done  # a tool entry, marked by default
    assert dict(retention.write_markers(tree))[tree / "charts/builds"] == "present"


def test_an_owner_evicts_inside_its_marked_cache_but_not_below_another_marker(tree):
    retention.write_markers(tree)
    build = tree / "charts/builds/abc"
    assert retention.guard(build, "build_cache", root=tree) is None
    assert retention.guard(build, "someone-else", root=tree) is not None
    retention.ensure_marker(build, "pinned by a release", "a PGO build")
    assert retention.guard(build, "build_cache", root=tree) is not None


def test_only_a_named_owner_evicts_inside_an_entry(tree):
    inside = tree / "work/realuse/f150-ip300/static_cam29"
    assert retention.guard(inside, root=tree) is not None
    assert retention.guard(inside, "someone-else", root=tree) is not None
    assert retention.guard(inside, "chart_refresh", root=tree) is None
    # the entry's root itself never goes, whoever asks
    assert retention.guard(tree / "work/realuse/f150-ip300", "chart_refresh", root=tree)


def test_paths_outside_the_data_root_are_not_judged(tree, tmp_path):
    assert retention.guard(tmp_path / "elsewhere", root=tree) is None


def test_refusal_says_what_it_costs(tree):
    text = retention.check(tree / "work", root=tree)
    assert "contains the kept work/realuse/f150-ip300" in text and "6.5 CPU-h" in text


def test_status_lists_unlisted_paths_and_keeps_recent_ones(tree):
    rows = {r.rel: r for r in retention.status(tree)}
    assert set(rows) == {"work/realuse/f600-ip300", "work/old-agent", "work/fresh-agent",
                         "stray.log", "work/undeclared", "scratch"}
    assert rows["work/old-agent"].state == "sweepable"
    assert rows["work/fresh-agent"].state == "live"
    assert rows["work/undeclared"].state == "unknown"
    assert rows["work/realuse/f600-ip300"].bytes > 0


def test_sweep_lists_without_apply_and_removes_only_crap_with_it(tree):
    said = []
    retention.sweep(apply=False, root=tree, say=said.append)
    assert (tree / "work/old-agent").exists() and any("would remove" in s for s in said)
    retention.sweep(apply=True, root=tree, say=said.append)
    for gone in ("work/old-agent", "work/realuse/f600-ip300", "stray.log"):
        assert not (tree / gone).exists(), gone
    for kept in ("work/realuse/f150-ip300/static_cam29/x264-medium/cbr300-abc.json",
                 "work/fresh-agent/log.txt", "worktrees/gh-pages", "clips/main.json",
                 "work/undeclared"):
        assert (tree / kept).exists(), kept


def test_verify_release_names_missing_or_empty_stores(tree):
    assert retention.verify_release(tree) == []
    (tree / "work/realuse/f150-ip300/static_cam29/x264-medium/cbr300-abc.json").unlink()
    problems = retention.verify_release(tree)
    assert len(problems) == 1 and "x264-medium" in problems[0] and "6.5 CPU-h" in problems[0]
    for path in sorted((tree / "work/realuse/f150-ip300").rglob("*"), reverse=True):
        path.rmdir()
    assert "missing or empty: work/realuse/f150-ip300" in retention.verify_release(tree)[0]


def test_the_shipped_manifest_loads_and_keeps_the_release_stores(monkeypatch):
    monkeypatch.delenv(retention.MANIFEST_ENV, raising=False)
    root = Path("/nanocodec-test-root")
    kept = {e.path for e in retention.manifest().entries}
    for needed in ("work/realuse/f150-ip300", "worktrees/gh-pages", "charts/builds",
                   "clips/layergate_fit", "run", "anchors/*", "conformance/*"):
        assert needed in kept, needed
    assert retention.guard(root / "work/realuse/f150-ip300", root=root) is not None


CAPPED = """
[[keep]]
path = "work/cache-a"
class = "cache"
reader = "a tool's capped cache"
regen = "seconds"
evicted_by = ["tool-a"]
cap_gb = 0.000003

[[keep]]
path = "work/cache-a/pinned"
class = "durable"
reader = "something kept inside it"
regen = "not regenerable"
"""


def test_a_capped_cache_loses_its_least_recently_used_entries_first(tmp_path, monkeypatch):
    """docs/aws-cost-guide.md, 2026-11-24: work/ over its budget blocked every GPU-scored
    quick; each owning tool now keeps its cache within the manifest's cap_gb."""
    manifest = tmp_path / "retention.toml"
    manifest.write_text(CAPPED)
    monkeypatch.setenv(retention.MANIFEST_ENV, str(manifest))
    root = tmp_path / "data"
    monkeypatch.setenv(retention.DATA_ROOT_ENV, str(root))
    cache = root / "work/cache-a"
    cache.mkdir(parents=True)
    entries = []
    for index, age_s in enumerate((300, 100, 200, 10)):  # 1 KiB each, used age_s ago
        entry = cache / f"clip{index}.zst"
        entry.write_bytes(b"x" * 1024)
        age(entry, age_s)
        entries.append(entry)
    (cache / "pinned").mkdir()
    (cache / "pinned/x").write_bytes(b"y" * 1024)
    age(cache / "pinned", 1000)  # the oldest, but kept by the manifest
    cap = retention.cap_bytes(cache, root)
    assert cap == int(0.000003 * (1 << 30))  # 3221 bytes: three of the four clips
    freed, removed = retention.prune_to_cap([*entries, cache / "pinned"], cap, "tool-a")
    assert (freed, removed) == (2048, 2)
    assert not entries[0].exists() and not entries[2].exists()  # the two oldest clips
    assert entries[1].exists() and entries[3].exists() and (cache / "pinned/x").exists()
    # another tool may not evict it
    entries[1].write_bytes(b"x" * 4096)
    assert retention.prune_to_cap([entries[1]], 0, "tool-b") == (0, 0) and entries[1].exists()


def test_a_capped_cache_spares_entries_in_use(tmp_path, monkeypatch):
    """2026-11-24: a prune removed a cloud point directory while another job was staging
    it as its seed; an entry used within min_idle_s stays, however old it was listed."""
    manifest = tmp_path / "retention.toml"
    manifest.write_text(CAPPED)
    monkeypatch.setenv(retention.MANIFEST_ENV, str(manifest))
    root = tmp_path / "data"
    monkeypatch.setenv(retention.DATA_ROOT_ENV, str(root))
    cache = root / "work/cache-a"
    cache.mkdir(parents=True)
    old, claimed = cache / "old.zst", cache / "claimed.zst"
    for path in (old, claimed):
        path.write_bytes(b"x" * 4096)
        age(path, 7200)
    os.utime(claimed)  # a staging job just claimed it
    freed, removed = retention.prune_to_cap([old, claimed], 0, "tool-a", min_idle_s=3600)
    assert (freed, removed) == (4096, 1) and claimed.exists() and not old.exists()
