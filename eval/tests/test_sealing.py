"""The release-only held-out set (nceval.sealing): refused outside a release
measurement, and its families sealed from the screening pool."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nceval import pool, sealing
from nceval.clips import load_clipset
from nceval.config import Paths

SEALED = "heldout_2026-11-23"
REPO = Path(__file__).resolve().parents[2]


def entry(name, source):
    return {"name": name, "yuv": f"/{name}.mp4", "width": 64, "height": 32, "fps": 30,
            "frames": 165, "source_kbps": 2000, "content_class": "static", "split": "train",
            "grade": "A", "source": source}


def write(clips_dir, name, entries):
    path = clips_dir / f"{name}.json"
    path.write_text(json.dumps({"name": name, "clips": entries}))
    return path


def test_release_only_set_is_refused_outside_a_release_measurement(tmp_path, monkeypatch):
    path = write(tmp_path, SEALED, [entry("ho_a", "/src/a.mp4")])
    monkeypatch.delenv(sealing.RELEASE_ENV, raising=False)
    with pytest.raises(sealing.SealedSetError):
        load_clipset(path, tmp_path)
    # a copy under another file name is still the set (the name inside decides)
    copy = tmp_path / "renamed.json"
    copy.write_text(path.read_text())
    with pytest.raises(sealing.SealedSetError):
        load_clipset(copy, tmp_path)
    assert [c.name for c in load_clipset(path, tmp_path, exclusion_only=True).clips] == ["ho_a"]
    monkeypatch.setenv(sealing.RELEASE_ENV, "1")
    assert [c.name for c in load_clipset(path, tmp_path).clips] == ["ho_a"]
    assert sealing.remote_env() == f"{sealing.RELEASE_ENV}=1 "


def test_release_only_families_stay_out_of_the_pool(tmp_path, monkeypatch):
    monkeypatch.delenv(sealing.RELEASE_ENV, raising=False)
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    write(clips_dir, SEALED, [entry("ho_a", "/src/a.mp4")])
    write(clips_dir, "pool", [entry("copy_of_a", "/src/a.mp4"), entry("b", "/src/b.mp4")])
    paths = Paths(data_root=tmp_path, anchors=tmp_path, nanocodec_bin=tmp_path)
    assert SEALED in pool.SEALED_SETS
    assert [c.name for c in pool.load_pool(paths)] == ["b"]


def test_the_ci_check_walks_a_tree_without_git(tmp_path):
    """The build box's synced copy has no .git: the check walks it."""
    sys.path.insert(0, str(REPO / "tools/ci"))
    import check_sealed_sets
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "tune_on_it.py").write_text(f'CLIPS = "{SEALED}.json"\n')
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "ignored.py").write_text(f'CLIPS = "{SEALED}.json"\n')
    found = check_sealed_sets.problems(tmp_path)
    assert len(found) == 1 and found[0].startswith("tools/tune_on_it.py:1:")


def test_the_ci_check_passes_on_the_tree():
    out = subprocess.run([sys.executable, str(REPO / "tools/ci/check_sealed_sets.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_the_release_raw_staging_refuses_without_the_switch(tmp_path):
    """bench_raw.py is allowed to name the sealed set only because it refuses it outside a
    release measurement (tools/ci/check_sealed_sets.py ALLOWED_NAMES)."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    env = {k: v for k, v in os.environ.items() if k != sealing.RELEASE_ENV}
    script = Path(__file__).resolve().parents[1] / "clipset" / "bench_raw.py"
    run = subprocess.run([sys.executable, str(script), "heldout", "--out", str(tmp_path)],
                         capture_output=True, text=True, env=env)
    assert run.returncode != 0 and "release-measure only" in run.stdout + run.stderr
    assert not any(tmp_path.iterdir())
