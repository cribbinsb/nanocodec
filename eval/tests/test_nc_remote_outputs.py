"""nc-remote: a finished job's declared outputs (--fetch) are never lost with the box.

The box records them when the job ends (tools/dev/nc_remote_job.py UNFETCHED); `reap`
fetches them when the calling `run` has gone; the idle watchdog (nc_remote_idle.sh),
`stop`, replacement and draining do not end a box that holds them (they fetch first);
`stop --force` names what it loses; undeclared writes to the box's home are reported.
Fake EC2 and SSH from test_nc_remote_replace; the box side runs on a temporary tree."""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_nc_remote_replace as fakes  # noqa: E402
from test_nc_remote_replace import OLD, current_box, reap, world  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import nc_remote_job  # noqa: E402

nc_remote = fakes.nc_remote
IDLE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "dev" / "nc_remote_idle.sh"


# --- PC side -------------------------------------------------------------------------------

@pytest.fixture
def box_outputs(world, tmp_path, monkeypatch):
    """The fake world plus each box's unfetched markers, answered over the fake SSH, and
    a record of fetches and terminations in the order they happened."""
    monkeypatch.setattr(nc_remote, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(nc_remote, "ORPHANS_DIR", tmp_path / "orphans")
    markers: dict[str, list[dict]] = {}
    events: list[tuple] = []

    def ssh(box, command, **kwargs):
        text = kwargs.get("text")
        out = ""
        words = shlex.split(command)
        if words[-1] == "--unfetched":
            out = "".join(json.dumps(m) + "\n" for m in markers.get(box.instance_id, []))
        elif words[-3:-2] == ["--fetched"]:
            name, run_id = words[-2:]
            markers[box.instance_id] = [m for m in markers.get(box.instance_id, [])
                                        if (m["name"], m["run_id"]) != (name, run_id)]
        return subprocess.CompletedProcess(command, 0, out if text else out.encode(), "")

    def fetch(box, name, paths, fetch_to):
        events.append(("fetch", box.instance_id, name, tuple(paths), Path(fetch_to)))

    terminate = world.terminate

    def terminated(box):
        events.append(("terminate", box.instance_id))
        terminate(box)

    monkeypatch.setattr(nc_remote, "ssh", ssh)
    monkeypatch.setattr(nc_remote, "fetch", fetch)
    monkeypatch.setattr(world, "terminate", terminated)
    monkeypatch.setattr(nc_remote, "between_jobs", lambda box: (
        world.jobs.get(box.instance_id, 0) == 0 and not markers.get(box.instance_id)))
    world.markers, world.events = markers, events
    return world


def finished_job(world, box_id, name, fetch_to, caller_alive=False):
    """A run's record of where its outputs go, and the box's marker of its finished job.
    The claim stays held while the caller lives; a closed claim is a caller that has gone."""
    claim = nc_remote.OutputsClaim(name, fetch_to)
    world.markers.setdefault(box_id, []).append(
        {"name": name, "run_id": claim.run_id, "paths": ["remote-out/builds"], "status": 0,
         "ended": "2026-11-20T09:58:00Z"})
    if not caller_alive:
        claim.handle.close()  # the run was killed: its flock went with it
    return claim


def test_reap_fetches_the_outputs_of_a_run_that_has_gone(box_outputs, tmp_path):
    current_box(box_outputs, market=nc_remote.SPOT)
    claim = finished_job(box_outputs, OLD, "hist", tmp_path / "worktree")
    reap()
    assert box_outputs.events[0] == ("fetch", OLD, "hist", ("remote-out/builds",),
                                     tmp_path / "worktree")
    assert box_outputs.markers[OLD] == []           # the box's marker is dropped
    assert not claim.path.exists()                  # and the PC's record


def test_reap_leaves_a_live_runs_outputs_to_it_and_keeps_the_box(box_outputs, tmp_path):
    current_box(box_outputs)  # on-demand and idle: reap would end it
    live = finished_job(box_outputs, OLD, "hist", tmp_path / "worktree", caller_alive=True)
    reap()
    assert [e for e in box_outputs.events if e[0] == "fetch"] == []
    assert box_outputs.terminated == []             # outputs still on the box: kept up
    live.release(fetched=False)


def test_reap_fetches_then_ends_an_idle_box(box_outputs, tmp_path):
    current_box(box_outputs)  # on-demand while spot is allowed, no job: ended by reap
    finished_job(box_outputs, OLD, "hist", tmp_path / "worktree")
    reap()
    assert [e[0] for e in box_outputs.events] == ["fetch", "terminate"]


def test_an_unrecorded_runs_outputs_go_to_the_orphans(box_outputs, tmp_path):
    current_box(box_outputs, market=nc_remote.SPOT)
    claim = finished_job(box_outputs, OLD, "hist", tmp_path / "worktree")
    claim.path.unlink()
    reap()
    assert box_outputs.events[0][4] == nc_remote.ORPHANS_DIR / "hist" / claim.run_id


def test_a_draining_box_is_fetched_before_it_ends(box_outputs, tmp_path):
    current_box(box_outputs)
    box_outputs.jobs[OLD] = 1
    reap()  # replaced: OLD drains
    assert nc_remote.load_state().instance_id != OLD
    box_outputs.jobs[OLD] = 0
    finished_job(box_outputs, OLD, "hist", tmp_path / "worktree")  # its caller died
    reap()
    assert [e[0] for e in box_outputs.events] == ["fetch", "terminate"]
    assert box_outputs.terminated == [OLD]


def test_stop_fetches_first_and_refuses_while_a_run_still_owns_outputs(box_outputs, tmp_path):
    current_box(box_outputs, market=nc_remote.SPOT)
    held = finished_job(box_outputs, OLD, "live", tmp_path / "a", caller_alive=True)
    finished_job(box_outputs, OLD, "gone", tmp_path / "b")
    assert nc_remote.cmd_stop(SimpleNamespace(force=False)) == 0
    assert [e[2] for e in box_outputs.events if e[0] == "fetch"] == ["gone"]
    assert box_outputs.terminated == []             # "live"'s outputs are not on the PC yet
    held.release(fetched=False)                     # that run went away too
    assert nc_remote.cmd_stop(SimpleNamespace(force=False)) == 0
    assert [e[0] for e in box_outputs.events] == ["fetch", "fetch", "terminate"]


def test_stop_force_names_what_it_loses(box_outputs, tmp_path, capsys):
    current_box(box_outputs, market=nc_remote.SPOT)
    live = finished_job(box_outputs, OLD, "live", tmp_path / "a", caller_alive=True)
    assert nc_remote.cmd_stop(SimpleNamespace(force=True)) == 0
    assert box_outputs.terminated == [OLD]
    err = capsys.readouterr()
    assert "LOST" in err.out + err.err and "remote-out/builds" in err.out + err.err
    live.release(fetched=False)


# --- box side ------------------------------------------------------------------------------

@pytest.fixture
def box_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(nc_remote_job, "BOX", tmp_path / "ncbox")
    (tmp_path / "ncbox" / "jobs").mkdir(parents=True)
    return tmp_path / "ncbox"


def test_markers_are_listed_dropped_and_keep_the_job_from_eviction(box_tree):
    nc_remote_job.mark_unfetched("hist", "r1", ["remote-out/builds"], 0)
    (box_tree / "jobs" / "old").mkdir()
    assert [m["run_id"] for _, m in nc_remote_job.unfetched_markers()] == ["r1"]
    assert [d.name for d in nc_remote_job.evictable(set())] == ["old"]
    nc_remote_job.mark_fetched("hist", "r1")
    assert nc_remote_job.unfetched_markers() == []
    assert {d.name for d in nc_remote_job.evictable(set())} == {"hist", "old"}


def test_a_marker_from_before_boot_is_dropped(box_tree):
    nc_remote_job.mark_unfetched("baked", "r0", ["out"], 0)
    path = box_tree / "jobs" / "baked" / "unfetched" / "r0.json"
    os.utime(path, (nc_remote_job.boot_time() - 60,) * 2)
    assert nc_remote_job.unfetched_markers() == []
    assert not path.exists()


def run_idle_check(box: Path, idle_s: int) -> str:
    """nc_remote_idle.sh against `box`, its shutdown and syslog replaced by echoes."""
    script = IDLE_SCRIPT.read_text().replace("@IDLE_MINUTES@", "10") \
        .replace("/home/ubuntu/ncbox", str(box)).replace("shutdown -h now", "echo SHUTDOWN") \
        .replace("rm -rf -- /home/ubuntu/nanocodec-data/clips", ":").replace("logger -t", "echo")
    (box / "slots").mkdir(exist_ok=True)
    (box / "slots" / "0").touch()
    (box / "activity").touch()
    os.utime(box / "activity", (time.time() - idle_s,) * 2)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          check=True).stdout


def test_the_idle_check_refuses_while_outputs_are_unfetched(box_tree):
    nc_remote_job.mark_unfetched("hist", "r1", ["remote-out/builds"], 0)
    out = run_idle_check(box_tree, idle_s=3600)
    assert "SHUTDOWN" not in out and "unfetched outputs" in out
    nc_remote_job.mark_fetched("hist", "r1")
    assert "SHUTDOWN" in run_idle_check(box_tree, idle_s=3600)
    assert "SHUTDOWN" not in run_idle_check(box_tree, idle_s=60)  # not idle long enough


def test_user_data_carries_the_idle_check():
    user_data = (nc_remote.HERE / "nc_remote_box.sh").read_text() \
        .replace("@IDLE_SCRIPT@", nc_remote.idle_script())
    assert "unfetched" in user_data and "@IDLE_MINUTES@" not in nc_remote.idle_script()


def test_an_undeclared_home_write_is_reported(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    for path in ("ncbox/jobs/hist/src/remote-out/x", ".cargo/registry/y",
                 "nanocodec-data/bench/old", "declared/out"):
        (home / path).parent.mkdir(parents=True, exist_ok=True)
        (home / path).write_text("")
    started = time.time() - 1
    os.utime(home / "nanocodec-data/bench/old", (started - 100,) * 2)  # from before the job
    (home / "hist-build/abc").mkdir(parents=True)
    (home / "hist-build/abc/nanocodec").write_text("")
    (home / "nanocodec-data/bench/history-days").mkdir()
    (home / "nanocodec-data/bench/history-days/b").write_text("")
    monkeypatch.setenv("HOME", str(home))
    groups = nc_remote_job.undeclared_writes(started, ["~/declared"])
    assert groups == {"~/hist-build/abc": 1, "~/nanocodec-data/bench": 1}
    nc_remote_job.warn_undeclared(groups)
    err = capsys.readouterr().err
    assert "UNDECLARED WRITES: 2 files" in err and "~/hist-build/abc: 1 files" in err
