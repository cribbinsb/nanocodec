"""tools/dev/nc_remote.py: make-before-break replacement of the build box. An
on-demand box with jobs on it, once spot is allowed again, gets a spot
successor; new runs go there; the old box drains and is ended only when no run
uses it (its job and its fetch done). Fake EC2, fake box slots, PC records in a
temporary directory."""
import datetime
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "dev" / "nc_remote.py"
spec = importlib.util.spec_from_file_location("nc_remote", SCRIPT)
nc_remote = importlib.util.module_from_spec(spec)
sys.modules["nc_remote"] = nc_remote  # dataclasses look their module up by name
spec.loader.exec_module(nc_remote)

NOW = nc_remote.parse("2026-11-20T10:00:00Z")
OLD = "i-old-on-demand"


class FakeImages:
    def describe_images(self, **_):
        return {"Images": []}


class FakeAws:
    """EC2 as nc_remote uses it, plus each box's running jobs (`jobs`)."""

    def __init__(self, spot_capacity=True):
        self.ec2 = FakeImages()
        self.spot_capacity = spot_capacity
        self.states, self.jobs = {}, {}
        self.launches, self.terminated = [], []

    ours = staticmethod(nc_remote.Aws.ours)

    def regional(self, _region):
        return self  # one fake EC2 for every region

    def launch(self, plan, on_demand_fallback=True):
        self.launches.append(plan.market)
        market = plan.market
        if market == nc_remote.SPOT and not self.spot_capacity:
            if not on_demand_fallback:
                raise nc_remote.NoCapacity("no spot capacity")
            market = nc_remote.ON_DEMAND
        instance_id = f"i-new-{len(self.launches)}"
        self.states[instance_id] = "running"
        return nc_remote.Box(instance_id=instance_id, instance_type=plan.types[0],
                             zone="eu-north-1a", price=0.3, launched=nc_remote.iso(NOW),
                             expected_end=nc_remote.iso(NOW + datetime.timedelta(hours=12)),
                             host="198.51.100.2", market=market)

    def state(self, instance_id):
        return self.states.get(instance_id, "gone")

    def security_group(self, _expected_end):
        return "sg-fake"  # the SSH rule (tests/test_nc_remote_ssh_rule.py)

    def reclaimed(self, _instance_id):
        return False

    def end_time(self, _instance_id):
        return NOW

    def terminate(self, box):
        assert self.jobs.get(box.instance_id, 0) == 0, f"{box.instance_id}: killed a running job"
        self.states[box.instance_id] = "terminated"
        self.terminated.append(box.instance_id)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """PC records under tmp_path; SSH, readiness and the ledger faked."""
    for name, path in (("STATE_DIR", tmp_path), ("STATE", tmp_path / "state.json"),
                       ("HISTORY", tmp_path / "history.json"), ("LOCK", tmp_path / "lock"),
                       ("REPLACING", tmp_path / "replacing"), ("USERS_DIR", tmp_path / "users"),
                       ("OUTPUTS_DIR", tmp_path / "outputs"), ("ORPHANS_DIR", tmp_path / "orphans")):
        monkeypatch.setattr(nc_remote, name, path)
    aws = FakeAws()
    ledger = []
    monkeypatch.setattr(nc_remote, "Aws", lambda: aws)
    monkeypatch.setattr(nc_remote, "utc_now", lambda: NOW)
    monkeypatch.setattr(nc_remote, "ssh", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(nc_remote, "between_jobs",
                        lambda box: aws.jobs.get(box.instance_id, 0) == 0)
    monkeypatch.setattr(nc_remote, "wait_for_ssh", lambda box: None)
    monkeypatch.setattr(nc_remote, "provision", lambda box: None)
    monkeypatch.setattr(nc_remote, "ledger_row", lambda *a, **k: ledger.append(a[:2]))
    aws.ledger = ledger
    return aws


def current_box(aws, market=nc_remote.ON_DEMAND, age=datetime.timedelta(hours=1)):
    launched = NOW - age
    box = nc_remote.Box(instance_id=OLD, instance_type="c6i.24xlarge", zone="eu-north-1c",
                        price=2.912, launched=nc_remote.iso(launched),
                        expected_end=nc_remote.iso(launched + datetime.timedelta(hours=12.25)),
                        host="198.51.100.1", market=market)
    aws.states[OLD] = "running"
    nc_remote.save_state(box)
    nc_remote.record(box)
    return box


def reap():
    assert nc_remote.cmd_reap(None) == 0


def test_on_demand_with_a_job_is_replaced_by_spot_and_drains(world):
    current_box(world)
    first_run = nc_remote.ensure_box(world)  # a run on the old box, job running
    assert first_run.box.instance_id == OLD
    world.jobs[OLD] = 1

    reap()  # spot allowed (no reclaims): a spot successor, made current once ready
    assert world.launches == [nc_remote.SPOT]
    new = nc_remote.load_state()
    assert new.instance_id != OLD and new.market == nc_remote.SPOT
    assert OLD not in world.terminated
    assert {b.instance_id for b in nc_remote.unended()} == {OLD, new.instance_id}
    assert nc_remote.role(nc_remote.unended()[0], new) == "draining"

    with nc_remote.ensure_box(world) as box:  # every new run goes to the new box
        assert box.instance_id == new.instance_id

    reap()  # job still running: the old box stays
    assert OLD not in world.terminated
    world.jobs[OLD] = 0
    reap()  # job done, but its run is still fetching: still up
    assert OLD not in world.terminated

    first_run.__exit__(None, None, None)  # the run has its results
    reap()
    assert world.terminated == [OLD]
    assert [b.instance_id for b in nc_remote.unended()] == [new.instance_id]
    assert nc_remote.load_state().instance_id == new.instance_id
    assert ("started" in [e for _, e in world.ledger]) and \
        (any(r.startswith(OLD) for r, e in world.ledger if e == "ended"))
    assert world.launches == [nc_remote.SPOT]  # nothing else was launched


def test_an_idle_on_demand_box_is_ended_without_a_successor(world):
    current_box(world)
    reap()
    assert world.terminated == [OLD]
    assert world.launches == []
    assert nc_remote.load_state() is None


def test_no_spot_capacity_keeps_the_on_demand_box(world):
    current_box(world)
    world.jobs[OLD] = 1
    world.spot_capacity = False
    reap()
    assert world.launches == [nc_remote.SPOT] * len(nc_remote.SPILL_REGIONS)  # no on-demand box
    assert nc_remote.load_state().instance_id == OLD
    assert world.terminated == []


def test_a_spot_box_is_not_replaced_until_near_its_hard_end(world):
    current_box(world, market=nc_remote.SPOT)
    world.jobs[OLD] = 1
    reap()
    assert world.launches == []


def test_a_busy_box_near_its_hard_end_is_replaced(world):
    current_box(world, market=nc_remote.SPOT, age=datetime.timedelta(hours=11, minutes=30))
    world.jobs[OLD] = 1
    reap()
    assert world.launches == [nc_remote.SPOT]
    assert nc_remote.load_state().instance_id != OLD
    assert OLD not in world.terminated


def test_a_successor_that_never_gets_ready_is_ended(world, monkeypatch):
    current_box(world)
    world.jobs[OLD] = 1

    def not_ready(box):
        raise RuntimeError("SSH not ready")
    monkeypatch.setattr(nc_remote, "wait_for_ssh", not_ready)
    with pytest.raises(RuntimeError):
        reap()
    assert world.terminated == ["i-new-1"]
    assert nc_remote.load_state().instance_id == OLD


def test_a_starting_successor_is_not_taken_for_a_draining_box(world):
    current_box(world)
    starting = FakeAws.launch(world, nc_remote.market_now())
    nc_remote.record(starting)
    with nc_remote.Replacing() as replacing:
        replacing.name(starting.instance_id)
        assert nc_remote.role(starting, nc_remote.load_state()) == "starting"
        with nc_remote.Replacing() as second:  # one replacement at a time
            assert second is None
        reap()  # the old box has no job: ended; the starting one is left alone
    assert world.terminated == [OLD]
    assert nc_remote.starting_id() == ""


def test_a_run_that_has_gone_no_longer_holds_its_box(world):
    box = current_box(world)
    nc_remote.USERS_DIR.mkdir()
    (nc_remote.USERS_DIR / f"{OLD}.999999").write_text("")  # its flock died with it
    assert not nc_remote.box_in_use(box)
    assert list(nc_remote.USERS_DIR.iterdir()) == []
    with nc_remote.ensure_box(world):
        assert nc_remote.box_in_use(box)
    assert not nc_remote.box_in_use(box)
