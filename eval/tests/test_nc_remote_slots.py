"""tools/dev/nc_remote_job.py: verify's kept slots and the overflow signal."""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import nc_remote_job  # noqa: E402


def box(tmp_path, monkeypatch, slots=12):
    monkeypatch.setattr(nc_remote_job, "BOX", tmp_path)
    (tmp_path / "slots").mkdir()
    (tmp_path / "jobs").mkdir()
    for slot in range(slots):
        (tmp_path / "slots" / str(slot)).touch()


def test_verify_has_kept_slots_and_agents_use_the_rest(tmp_path, monkeypatch):
    box(tmp_path, monkeypatch)
    names = lambda job: [p.name for p in nc_remote_job.job_slots(job)]
    assert names("verifier-tests")[:2] == ["0", "1"] and len(names("verifier-x")) == 12
    assert names("agent") == [str(i) for i in range(2, 12)]


def test_free_counts_agents_slots_and_the_oldest_wait(tmp_path, monkeypatch, capsys):
    box(tmp_path, monkeypatch, slots=3)
    held = (tmp_path / "slots" / "2").open("w")
    fcntl.flock(held, fcntl.LOCK_EX)
    (tmp_path / "jobs" / "a").mkdir()
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 150))
    (tmp_path / "jobs" / "a" / "waiting.json").write_text(json.dumps({"since": since,
                                                                       "pid": os.getpid()}))
    nc_remote_job.print_free()
    free, waited = capsys.readouterr().out.split()
    assert free == "0" and 148 <= int(waited) <= 152  # slot 2 held; 0-1 are verify's


def test_a_record_whose_process_is_gone_or_predates_the_boot_holds_nothing(tmp_path, monkeypatch,
                                                                        capsys):
    box(tmp_path, monkeypatch, slots=3)
    held = (tmp_path / "slots" / "0").open("w")
    fcntl.flock(held, fcntl.LOCK_EX)  # verify's job holds slot 0 now
    stamp = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    for name, record in (
            ("verifier-tests", {"slot": "0", "pid": os.getpid(), "command": "verify",
                                "started": stamp(time.time())}),
            # baked into the image five days before this boot, on slot 0 too
            ("av1-writer", {"slot": "0", "pid": os.getpid(), "command": "cargo test",
                            "started": "2026-11-16T09:39:28Z"}),
            ("gone", {"slot": "1", "pid": 2 ** 22 + 1, "command": "x",
                      "started": stamp(time.time())})):
        (tmp_path / "jobs" / name).mkdir()
        (tmp_path / "jobs" / name / "running.json").write_text(json.dumps(record))
    (tmp_path / "jobs" / "av1-writer" / "waiting.json").write_text(
        json.dumps({"since": "2026-11-16T09:39:28Z"}))
    monkeypatch.setattr(nc_remote_job, "boot_time", lambda: time.time() - 3600)
    nc_remote_job.print_status()
    out = capsys.readouterr().out
    assert "slot 0: verifier-tests since" in out and "av1-writer" not in out
    assert "slot 1: free" in out and "queue: 0 waiting" in out
    assert not (tmp_path / "jobs" / "av1-writer" / "running.json").exists()
    assert not (tmp_path / "jobs" / "gone" / "running.json").exists()


def test_low_disk_evicts_the_least_recently_used_idle_jobs_only(tmp_path, monkeypatch):
    import collections
    box(tmp_path, monkeypatch, slots=2)
    now = time.time()
    for age, name in enumerate(["current", "fresh", "verifier-tests", "cpuab-x86_64",
                                "running", "old", "older"]):
        (tmp_path / "jobs" / name / "target").mkdir(parents=True)
        (tmp_path / "jobs" / name / nc_remote_job.LAST_USED).touch()
        os.utime(tmp_path / "jobs" / name / nc_remote_job.LAST_USED, (now - 3600 * age,) * 2)
    (tmp_path / "jobs" / "running" / "running.json").write_text(json.dumps(
        {"slot": "0", "pid": os.getpid(), "command": "x",
         "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    monkeypatch.setattr(nc_remote_job, "boot_time", lambda: now - 86400)
    removed = []
    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(nc_remote_job.shutil, "disk_usage",
                        lambda _: Usage(0, 0, (len(removed) * 30) << 30))
    monkeypatch.setattr(nc_remote_job.shutil, "rmtree", lambda d, **_: removed.append(d.name))
    nc_remote_job.make_room("current")
    assert removed == ["older", "old"]  # 0 -> 60 GiB free: two evictions, oldest first
    removed.clear()
    nc_remote_job.drop(["fresh", "running", "verifier-tests", "unknown"])
    assert removed == ["fresh"]
