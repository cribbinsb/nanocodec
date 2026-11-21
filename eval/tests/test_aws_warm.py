"""tools/dev/aws_warm.py: the warm-box registry, on a fake session (nothing is rented)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_warm  # noqa: E402
from nceval import cloud  # noqa: E402

LAUNCHED = "2026-11-20T00:00:00Z"
SPEC = type("Spec", (), {"instance_type": "c7a.24xlarge"})()


class FakeSession:
    """A box: `states` maps instance id -> EC2 state name (absent: running)."""
    launched: list[str] = []
    cleaned: list[tuple[str, object]] = []
    states: dict[str, str] = {}

    def __init__(self, spec=None, record=None):
        self.spec, self.record = spec, record or {}
        self.instance_id = self.record.get("instance_id")
        self.ec2 = None

    @classmethod
    def attach(cls, record):
        return cls(record=record)

    def __enter__(self):
        self.instance_id = f"i-{len(FakeSession.launched)}"
        FakeSession.launched.append(self.instance_id)
        return self

    def detach(self):
        return {"instance_id": self.instance_id, "launched": aws_warm.utc_now().strftime(
            "%Y-%m-%dT%H:%M:%SZ"), "price": 0.6, "host": "h",
                "spec": {"instance_type": "c7a.24xlarge", "region": "eu-north-1"}}

    def describe(self):
        state = FakeSession.states.get(self.instance_id, "running")
        return {"State": {"Name": state},
                "StateTransitionReason": "User initiated (2026-11-21 07:40:00 GMT)"}

    def cleanup(self, ended_at=None):
        FakeSession.cleaned.append((self.instance_id, ended_at))


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(aws_warm, "REGISTRY", tmp_path / "warm")
    monkeypatch.setattr(aws_warm, "Session", FakeSession)
    monkeypatch.setattr(aws_warm, "instance_vcpus", lambda ec2, kind: 96)
    FakeSession.launched, FakeSession.cleaned, FakeSession.states = [], [], {}
    return tmp_path / "warm"


def test_jobs_share_a_box_up_to_its_cpu_then_a_second_is_launched(registry):
    setups = []
    for job in ("a", "b", "c", "d"):
        box, _, fresh = aws_warm.acquire(job, SPEC, setups.append)
    assert FakeSession.launched == ["i-0", "i-1"] and len(setups) == 2  # 96 vCPU: 3 jobs
    record = json.loads((registry / "i-0.json").read_text())
    assert set(record["jobs"]) == {"a", "b", "c"} and record["prepared"]
    aws_warm.release(aws_warm.Box(record), "b")
    box, _, fresh = aws_warm.acquire("e", SPEC, setups.append)
    assert box.instance_id == "i-0" and not fresh and len(setups) == 2


def test_a_box_that_stopped_itself_is_cleaned_up_to_when_it_stopped(registry):
    aws_warm.acquire("a", SPEC, lambda s: None)
    FakeSession.states["i-0"] = "terminated"
    with aws_warm.registry_lock():
        aws_warm.reap()
    [(instance, ended)] = FakeSession.cleaned
    assert instance == "i-0" and ended.isoformat().startswith("2026-11-21T07:40:00")
    assert not (registry / "i-0.json").exists()


def test_a_failed_setup_terminates_the_new_box(registry):
    def broken(session):
        raise RuntimeError("apt failed")
    with pytest.raises(RuntimeError):
        aws_warm.acquire("a", SPEC, broken)
    assert FakeSession.cleaned == [("i-0", None)] and not list(registry.glob("i-*.json"))


def test_a_box_near_its_lifetime_takes_no_new_job(registry, monkeypatch):
    aws_warm.acquire("a", SPEC, lambda s: None)
    record = json.loads((registry / "i-0.json").read_text())
    record["launched"] = LAUNCHED  # long ago: past its lifetime margin
    (registry / "i-0.json").write_text(json.dumps(record))
    box, _, fresh = aws_warm.acquire("b", SPEC, lambda s: None)
    assert fresh and box.instance_id == "i-1"


def test_the_idle_timer_cleans_up_and_shuts_down_after_its_idle_time():
    line = aws_warm.idle_watch_command("rm -rf -- ~/x", 20)
    assert "-ge 1200" in line and "rm -rf -- ~/x; sudo shutdown -h now" in line
    assert line.startswith("{ setsid nohup bash -c '") and line.endswith("& }")
    import subprocess
    checked = subprocess.run(["bash", "-n", "-c", f"true && {line}; echo $? > /dev/null"])
    assert checked.returncode == 0  # as run_detached runs it


def test_offload_adds_a_json_drops_local_and_exits_with_the_status(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cloud.subprocess, "call", lambda argv: calls.append(argv) or 0)
    with pytest.raises(SystemExit) as done:
        cloud.offload("nceval.quick", ["--codec", "h264", "--local"], None, tmp_path / "q.json")
    assert done.value.code == 0
    argv = calls[0]
    assert argv[2:6] == ["--", "python3", "-m", "nceval.quick"]
    assert "--local" not in argv and argv[-2:] == ["--json", str(tmp_path / "q.json")]


def test_on_the_box_and_with_local_a_command_runs_where_it_is(monkeypatch):
    monkeypatch.delenv(cloud.ON_BOX_ENV, raising=False)
    monkeypatch.delenv(cloud.BUILD_BOX_ENV, raising=False)
    assert not cloud.runs_here(False) and cloud.runs_here(True)
    monkeypatch.setenv(cloud.ON_BOX_ENV, "1")
    assert cloud.runs_here(False)


def test_a_tree_behind_the_integration_branch_in_the_cloud_tools_is_refused(monkeypatch):
    monkeypatch.delenv(cloud.ON_BOX_ENV, raising=False)
    monkeypatch.setattr(cloud, "REPO", Path(__file__).resolve().parents[2])
    if not (cloud.REPO / ".git").exists():
        pytest.skip("no git checkout (the build box's copy)")
    monkeypatch.delenv(cloud.STALE_ENV, raising=False)
    behind = type("Done", (), {"returncode": 0, "stdout": "abc1234 aws: fix\n"})()
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **k: behind)
    with pytest.raises(SystemExit, match="rebase variance-integration"):
        cloud.check_fresh()
    monkeypatch.setenv(cloud.STALE_ENV, "1")
    cloud.check_fresh()  # allowed on request


def test_a_checkpoint_skips_other_jobs_temp_files_and_vanished_ones(tmp_path):
    """Two jobs share a warm box's point cache: one's checkpoint must not fail on the
    other's in-progress files (dot temporaries, or a file gone between list and tar)."""
    import subprocess
    import tarfile
    import aws_session
    cache = tmp_path / "cache"
    (cache / "f150").mkdir(parents=True)
    (cache / "f150" / "mine.json").write_text("{}")
    (cache / "f150" / ".cbr150-x.json.qlpmzmbo.times.txt").write_text("partial")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # `find` lists one more file, which the other job renames away before tar reads it
    (fake_bin / "find").write_text("#!/bin/bash\n/usr/bin/find \"$@\"; echo ./f150/gone.json\n")
    (fake_bin / "find").chmod(0o755)
    out = tmp_path / "out.tar"
    with out.open("wb") as sink:
        done = subprocess.run(["bash", "-c", aws_session.checkpoint_command(str(cache), ".cp")],
                              stdout=sink, env={"PATH": f"{fake_bin}:/usr/bin:/bin"})
    assert done.returncode == 0
    with tarfile.open(out) as archive:
        assert archive.getnames() == ["./f150/mine.json"]
    assert (cache / ".cp").exists()  # the checkpoint moved on


def test_a_failed_point_download_keeps_the_finished_result(tmp_path):
    import subprocess
    import aws_measure

    class TwoJobBox:
        """A warm box whose point checkpoint fails (another job's files) after the job
        wrote its result."""
        def __init__(self):
            self.commands = []

        def ssh(self, command, check=True, quiet=False):
            self.commands.append(command)
            out = '{"status": "done"}' if "result.json" in command else ""
            return subprocess.CompletedProcess(command, 0 if out else 1, out, "")

        def download_new(self, remote, local, marker):
            raise RuntimeError("checkpoint download failed")
    layout = aws_measure.warm_layout("job-a")
    box = TwoJobBox()
    aws_measure.bring_back(box, tmp_path, [f"{layout.out}/result.json"], layout)
    assert (tmp_path / "result.json").read_text() == '{"status": "done"}'
    assert any(c.startswith("rm -rf -- ") and "job-a" in c for c in box.commands)


def test_queued_jobs_pack_onto_a_launch_in_progress(registry, monkeypatch):
    registry.mkdir(parents=True, exist_ok=True)
    marker = registry / "launching-1-x.json"  # pid 1 is alive: another job's launch
    marker.write_text(json.dumps({"owner": 1, "jobs": {"x": 1}}))
    monkeypatch.setattr(aws_warm.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def launched_meanwhile():  # the launcher turns its marker into a ready box
        state = json.loads(marker.read_text())
        marker.unlink()
        (registry / "i-7.json").write_text(json.dumps({
            "instance_id": "i-7", "launched": aws_warm.utc_now().strftime(
                "%Y-%m-%dT%H:%M:%SZ"), "price": 0.6, "host": "h", "vcpus": 96, "owner": 1,
            "prepared": True, "jobs": state["jobs"],
            "spec": {"instance_type": "c7a.24xlarge", "region": "eu-north-1"}}))
    original = aws_warm.reap

    def reap():
        calls["n"] += 1
        if calls["n"] == 2:
            launched_meanwhile()
        original()
    monkeypatch.setattr(aws_warm, "reap", reap)
    box, _, fresh = aws_warm.acquire("a", SPEC, lambda s: None)
    assert box.instance_id == "i-7" and not fresh and FakeSession.launched == []
    assert set(box.jobs) == {"x", "a"}


def test_the_spend_cap_queues_a_job_instead_of_launching(registry, monkeypatch, capsys):
    aws_warm.acquire("a", SPEC, lambda s: None)  # one box at the fake's $0.6/h
    monkeypatch.setattr(aws_warm, "NOMINAL_JOBS", 0)
    record = json.loads((registry / "i-0.json").read_text())
    record["jobs"] = {"p": 1, "q": 1, "r": 1}  # full
    (registry / "i-0.json").write_text(json.dumps(record))

    def stop(_seconds):
        raise TimeoutError
    monkeypatch.setattr(aws_warm.time, "sleep", stop)
    with pytest.raises(TimeoutError):
        aws_warm.acquire("b", SPEC, lambda s: None, spend_cap=1.2)
    assert "WAITING: the warm boxes cost $0.62/h" in capsys.readouterr().out
    assert FakeSession.launched == ["i-0"]


def test_a_job_joins_a_box_being_set_up_that_has_room(registry, monkeypatch):
    registry.mkdir(parents=True, exist_ok=True)
    record = {"instance_id": "i-9", "launched": aws_warm.utc_now().strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "price": 0.6, "host": "h", "vcpus": 96, "owner": 1,
        "prepared": False, "jobs": {"first": 1}, "spec": {"instance_type": "c7a.24xlarge",
                                                          "region": "eu-west-1"}}
    (registry / "i-9.json").write_text(json.dumps(record))
    monkeypatch.setattr(aws_warm, "WAIT_SECONDS", 0.01)
    ready = {"after": 2}

    def sleep(seconds):  # the box's own setup finishes while the job waits
        ready["after"] -= 1
        if ready["after"] == 0:
            done = json.loads((registry / "i-9.json").read_text())
            done["prepared"] = True
            (registry / "i-9.json").write_text(json.dumps(done))
    monkeypatch.setattr(aws_warm.time, "sleep", sleep)
    box, _, fresh = aws_warm.acquire("second", SPEC, lambda s: None)
    assert box.instance_id == "i-9" and not fresh and FakeSession.launched == []
    assert set(json.loads((registry / "i-9.json").read_text())["jobs"]) == {"first", "second"}


def test_a_job_pinned_to_a_family_uses_only_its_boxes_and_asks_for_its_types(registry):
    aws_warm.acquire("a", SPEC, lambda s: None)  # a c7a box with room
    box, _, fresh = aws_warm.acquire("b", SPEC, lambda s: None, family="c7i")
    assert fresh and FakeSession.launched == ["i-0", "i-1"]  # not onto the c7a box
    box, _, fresh = aws_warm.acquire("c", SPEC, lambda s: None, family="c7a")
    assert box.instance_id == "i-0" and not fresh
    assert aws_warm.warm_spec("eu-north-1", 10, "c7i").instance_type == "c7i.24xlarge"
    assert all(t.startswith("c7i.") for t in aws_warm.warm_spec("eu-north-1", 10, "c7i")
               .fallback_types)
    assert aws_warm.family_types("c8a") == ("c8a.24xlarge", "c8a.16xlarge", "c8a.12xlarge")


def pricey_box(registry, instance_id="i-9", price=1.8, region="eu-west-1"):
    registry.mkdir(parents=True, exist_ok=True)
    (registry / f"{instance_id}.json").write_text(json.dumps({
        "instance_id": instance_id, "launched": aws_warm.utc_now().strftime(
            "%Y-%m-%dT%H:%M:%SZ"), "price": price, "host": "h", "vcpus": 96, "owner": 1,
        "prepared": True, "jobs": {}, "spec": {"instance_type": "c7a.24xlarge",
                                               "region": region}}))


def test_a_pricey_spilled_box_drains_and_jobs_go_to_the_cheapest_box(registry):
    pricey_box(registry)  # $1.80/h in eu-west-1, empty
    box, _, fresh = aws_warm.acquire("a", SPEC, lambda s: None)
    assert fresh and box.instance_id == "i-0"  # a new (home) box, not the pricey one
    assert json.loads((registry / "i-9.json").read_text())["draining"]
    pricey_box(registry, "i-8", price=0.55, region="eu-north-1")  # cheaper than i-0's 0.6
    box, _, _ = aws_warm.acquire("b", SPEC, lambda s: None)
    assert box.instance_id == "i-8"


def test_no_draining_while_the_home_region_was_full_at_the_last_launch(registry):
    pricey_box(registry)
    aws_warm.note_launch("eu-west-1")  # the last launch spilled: home had no room
    box, _, fresh = aws_warm.acquire("a", SPEC, lambda s: None)
    assert box.instance_id == "i-9" and not fresh


def test_a_started_job_pins_its_tools_and_its_slices_are_not_checked_again(monkeypatch):
    monkeypatch.delenv(cloud.ON_BOX_ENV, raising=False)
    monkeypatch.delenv(cloud.STALE_ENV, raising=False)
    monkeypatch.delenv(cloud.PINNED_ENV, raising=False)
    monkeypatch.setattr(cloud, "REPO", Path(__file__).resolve().parents[2])
    if not (cloud.REPO / ".git").exists():
        pytest.skip("no git checkout (the build box's copy)")
    state = {"behind": ""}
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **k: type(
        "Done", (), {"returncode": 0, "stdout": state["behind"]})())
    cloud.check_fresh()  # the start: fresh, so pinned
    assert cloud.os.environ[cloud.PINNED_ENV] == cloud.tools_version()
    state["behind"] = "abc1234 aws: a landing meanwhile\n"
    cloud.check_fresh()  # a later slice from the same tree: not refused
    monkeypatch.setattr(cloud, "tools_version", lambda: "changed")  # the tree's tools moved
    with pytest.raises(SystemExit, match="behind variance-integration"):
        cloud.check_fresh()


def test_a_study_resumed_at_its_recorded_tool_version_is_not_checked(monkeypatch):
    from nceval import tune_pool
    monkeypatch.delenv(cloud.PINNED_ENV, raising=False)
    monkeypatch.delenv(cloud.ON_BOX_ENV, raising=False)
    monkeypatch.delenv(cloud.STALE_ENV, raising=False)
    monkeypatch.setattr(cloud, "tools_version", lambda: "v1")
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **k: type(
        "Done", (), {"returncode": 0, "stdout": "abc1234 aws: fix\n"})())
    if not (cloud.REPO / ".git").exists():
        pytest.skip("no git checkout (the build box's copy)")
    tune_pool.check_started("v1")  # the same tree and tools the study started with
    monkeypatch.delenv(cloud.PINNED_ENV)
    with pytest.raises(SystemExit):
        tune_pool.check_started("v0")  # tools changed since: checked, and behind
