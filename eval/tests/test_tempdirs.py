"""nceval.tempdirs: our temporary paths are named for nanocodec, and leftovers are swept."""
import os
import time

from nceval import tempdirs


def test_temporary_paths_are_named_for_nanocodec_under_one_root():
    with tempdirs.temp_dir("point") as path:
        assert os.path.basename(path).startswith(f"nanocodec-point-{os.getpid()}-")
        assert os.path.dirname(path) == str(tempdirs.root())
        assert tempdirs.root().name.startswith("nanocodec-")
    assert not os.path.exists(path)


def test_the_sweep_removes_old_leftovers_of_dead_processes_only(tmp_path):
    old = time.time() - 4 * 3600
    dead = tmp_path / "nanocodec-metrics-4194999-abc"  # above pid_max: no such process
    alive = tmp_path / f"nanocodec-metrics-{os.getpid()}-abc"
    fresh = tmp_path / "nanocodec-point-4194999-new"
    foreign = tmp_path / "pytest-of-someone"
    for path in (dead, alive, fresh, foreign):
        path.mkdir()
    for path in (dead, alive, foreign):
        os.utime(path, (old, old))
    assert tempdirs.sweep(tmp_path) == 1
    assert not dead.exists() and alive.exists() and fresh.exists() and foreign.exists()
