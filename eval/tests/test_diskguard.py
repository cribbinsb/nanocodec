"""nceval.diskguard: the work directory's size is shared and re-walked at most
every USAGE_TTL_SECONDS (walking ~200k files cost ~8 CPU-s per chart point)."""

import json

from nceval import diskguard


def test_shared_work_bytes_walks_once_within_the_ttl(tmp_path, monkeypatch):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    walks = []
    real_walk = diskguard.directory_bytes
    monkeypatch.setattr(diskguard, "directory_bytes",
                        lambda root: walks.append(root) or real_walk(root))
    first = diskguard.shared_work_bytes(tmp_path)
    (tmp_path / "b.bin").write_bytes(b"x" * 5)
    assert diskguard.shared_work_bytes(tmp_path) == first  # cached, not re-walked
    assert len(walks) == 1
    assert first >= 10


def test_shared_work_bytes_rewalks_a_stale_size(tmp_path, monkeypatch):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / diskguard.USAGE_CACHE).write_text(json.dumps({"bytes": 1, "time": 0}))
    assert diskguard.shared_work_bytes(tmp_path) >= 10
