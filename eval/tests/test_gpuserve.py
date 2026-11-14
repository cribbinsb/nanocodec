"""Switching VMAFx scoring pools is one way (nceval.gpuserve)."""

from pathlib import Path

import pytest

from nceval import gpuserve


def server(name: str, generation: int | None, retired: bool = False) -> gpuserve.Server:
    state = {} if generation is None else {"generation": generation}
    return gpuserve.Server(name, 1, "/x/vmaf", retired, state, None)


def test_only_older_generations_are_drained(tmp_path):
    tool = tmp_path / "vmaf"
    tool.write_bytes(b"")
    own = gpuserve.server_name(tool, 0)
    found = [server(own, 10), server("nc-old-unversioned", None), server("nc-old-9", 9),
             server("nc-same-10", 10), server("nc-newer-11", 11),
             server("nc-old-retired", 3, retired=True)]
    names = [s.name for s in gpuserve.drain_targets(found, tool, 10)]
    assert names == ["nc-old-unversioned", "nc-old-9"]


def test_a_newer_live_pool_is_found_and_never_drained(tmp_path):
    found = [server("nc-newer", 11), server("nc-newer-retired", 12, retired=True),
             server("nc-old", 9)]
    assert [s.name for s in gpuserve.newer_live_servers(found, 10)] == ["nc-newer"]
    assert gpuserve.newer_live_servers(found, 11) == []
    tool = tmp_path / "vmaf"
    tool.write_bytes(b"")
    assert "nc-newer" not in [s.name for s in gpuserve.drain_targets(found, tool, 10)]


def test_an_older_tree_fails_instead_of_draining(tmp_path, monkeypatch):
    tool = tmp_path / "vmaf"
    tool.write_bytes(b"")
    monkeypatch.setattr(gpuserve, "tool_generation", lambda _: 9)
    monkeypatch.setattr(gpuserve, "servers", lambda: [server("nc-live", 10)])
    drains = []
    monkeypatch.setattr(gpuserve.subprocess, "run", lambda *a, **k: drains.append(a))
    with pytest.raises(gpuserve.OlderThanLivePool, match="rebase"):
        gpuserve.drain_older_builds(tool)
    assert drains == []


def test_a_tool_without_generations_is_unversioned(tmp_path):
    assert gpuserve.tool_generation(Path(tmp_path / "missing")) == gpuserve.UNVERSIONED_GENERATION
