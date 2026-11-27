"""tools/bench/fleet.py's driver: many boxes as threads of one process (owner, 2026-11-24: one
process per box held about 6 GB of the PC), each box's output in its own log."""
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
sys.path.insert(0, str(REPO / "eval"))
import fleet  # noqa: E402


def jobs(clips: dict[str, int], seconds: float = 20.0) -> list[dict]:
    return [{"id": f"{c}|{n}", "input": c, "expected_s": seconds}
            for c, count in clips.items() for n in range(count)]


def test_units_stay_on_one_clip_and_prefer_the_box_s_clips():
    q = fleet.WorkQueue(jobs({"a": 20, "b": 5}))
    unit = q.take("box0", set())
    assert {j["input"] for j in unit} == {"a"} and len(unit) == 6  # 120 s of 20 s jobs
    assert {j["input"] for j in q.take("box1", {"b"})} == {"b"}
    assert {j["input"] for j in q.take("box0", {"a"})} == {"a"}


def test_a_lost_box_gives_back_only_its_unsettled_jobs():
    q = fleet.WorkQueue(jobs({"a": 10}))
    unit = q.take("box0", set())
    q.settle("box0", {unit[0]["id"], unit[1]["id"]})
    assert q.release("box0") == 4
    assert q.pending_count() == 8  # four back beside the four never taken
    assert not q.finished()


def test_settled_needs_a_later_reference_and_the_stream(tmp_path):
    box = tmp_path / "box0-0"
    (box / "streams").mkdir(parents=True)
    (box / "streams" / "j0").write_bytes(b"x")
    (box / "streams" / "j1").write_bytes(b"x")
    records = [
        {"id": "canary-start|j0", "canary": "start", "cpu_s": 1.0, "seq": 0},
        {"reference": "x264-medium", "cpu_s": 1.0, "seq": 0},
        {"reference": "nanocodec-av1@medium", "cpu_s": 2.0, "seq": 0},
        {"reference": "nanocodec-av1@medium", "cpu_s": 2.0, "seq": 3},
        {"id": "j0", "stream_sha256": "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881", "stream": "streams/j0", "seq": 0, "cpu_s": 1.0},
        {"id": "j1", "stream_sha256": "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881", "stream": "streams/j1", "seq": 1, "cpu_s": 1.0},
        {"id": "j2", "stream_sha256": "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881", "stream": "streams/j2", "seq": 2, "cpu_s": 1.0},
        {"reference": "x264-medium", "cpu_s": 1.0, "seq": 3},
        {"id": "j3", "stream_sha256": "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881", "stream": "streams/j3", "seq": 3, "cpu_s": 1.0},
    ]
    (box / "encodes.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n{partial")
    assert fleet.settled_ids(box, keep_streams=True) == {"j0", "j1"}  # j2: no stream on the PC
    assert fleet.settled_ids(box, keep_streams=False) == {"j0", "j1", "j2"}  # j3: unbracketed


def test_each_box_prints_to_its_own_log(tmp_path):
    logs = {i: tmp_path / f"box{i}.log" for i in range(5)}

    def work(index, stop):
        for _ in range(3):
            print(f"box {index} line")
            time.sleep(0.01)
        if index == 3:
            raise RuntimeError("refused: V7 on box 3")

    assert fleet.drive(logs, work) == [3]
    for index, log in logs.items():
        lines = log.read_text().strip().splitlines()
        assert [l for l in lines if l.startswith("box ") and l.endswith("line")] == \
            [f"box {index} line"] * 3
        assert lines[-1] == (fleet.BOX_FAILED + "refused: V7 on box 3" if index == 3
                             else fleet.BOX_DONE)


def test_a_signal_stops_every_box(tmp_path):
    logs = {i: tmp_path / f"box{i}.log" for i in range(3)}

    def work(index, stop):
        if not stop.wait(10):
            raise AssertionError("not stopped")
        raise RuntimeError("stopped by the driver")

    threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    started = time.monotonic()
    assert fleet.drive(logs, work) == [0, 1, 2]
    assert time.monotonic() - started < 5
    assert all(log.read_text().strip().splitlines()[-1] ==
               fleet.BOX_FAILED + "stopped by the driver" for log in logs.values())


def test_graviton_types_take_the_arm_image():
    assert fleet.image_arch("c8g.2xlarge") == "arm64"
    assert fleet.image_arch("c7gn.xlarge") == "arm64"
    assert fleet.image_arch("c8i.2xlarge") == fleet.image_arch("c7a.24xlarge") == "amd64"


def test_a_quarter_of_the_slots_carry_the_packing_test_load():
    loaded = [i for i in range(12) if fleet.loaded_slot(f"box{i}-0", 0.25)]
    assert loaded == [0, 4, 8]
    assert not fleet.loaded_slot("box0-1", 0.0)
    assert "--background background.json" in fleet.serve_command("c8i", True, 0, True)


def test_coded_jobs_waiting_for_references_do_not_stop_the_feed():
    """A box whose coded jobs await their next references still gets new units (2026-11-24:
    counting them as held work starved boxes whose jobs were slow, and they stalled)."""
    import datetime
    import types
    q = fleet.WorkQueue(jobs({"a": 40}, seconds=15.0))
    fake = types.SimpleNamespace(a=types.SimpleNamespace(lifetime=60), queue=q)
    s = types.SimpleNamespace(launched=datetime.datetime.now(datetime.timezone.utc))
    feed = fleet.BoxFeed(fake, s, "box0-0", "a")
    sent = []
    feed.ensure_clip = lambda job: None
    feed.write = lambda path, text: sent.append(path)
    feed.send(q.take("box0-0", set()))
    feed.send(q.take("box0-0", set()))
    coded = {j["id"] for j in q.held("box0-0")}  # 16 coded (240 s), none settled yet
    feed.top_up(recorded=coded)  # wants INBOX_SECONDS (120 s) of uncoded work: one more unit
    assert len([p for p in sent if "unit-" in p]) == 3


def test_the_benchmark_cap_counts_every_live_box_of_its_purpose():
    """The fleets' share of the quota (coordinator, 2026-11-24: at most 1000 of 1200 vCPU)."""
    import aws_session

    class Pages:
        def paginate(self, Filters):
            assert {"Name": "tag:Purpose", "Values": ["bench", "bench-*"]} in Filters
            return [{"Reservations": [{"Instances": [
                {"CpuOptions": {"CoreCount": 48, "ThreadsPerCore": 2}},
                {"CpuOptions": {"CoreCount": 4, "ThreadsPerCore": 2}}]}]}]

    class Ec2:
        def get_paginator(self, name):
            return Pages()

    assert aws_session.purpose_vcpus(Ec2(), "bench") == 104
    assert fleet.BENCH_VCPU_CAP == 1000


def test_a_replacement_scoring_box_skips_what_the_slot_already_scored(tmp_path):
    for d, ids in (("scorearm13", ["a", "b"]), ("scorearm13-1", ["c"]), ("scorearm113", ["x"])):
        (tmp_path / d).mkdir()
        (tmp_path / d / "scores.jsonl").write_text(
            "\n".join(json.dumps({"id": i}) for i in ids) + "\n{partial")
    assert fleet.pass_scored(tmp_path, "arm1", 3) == {"a", "b", "c"}


def test_scoring_defaults_to_graviton_and_packed_encodes_to_x86():
    import types
    score = types.SimpleNamespace(type=None, step="score", packed=0)
    packed = types.SimpleNamespace(type=None, step="run", packed=48)
    timing = types.SimpleNamespace(type=None, step="run", packed=0)
    assert fleet.resolve_types(score) == fleet.ARM_SCORE_TYPES == ("c8g.8xlarge",)
    assert fleet.resolve_types(packed) == fleet.UNTIMED_TYPES
    assert fleet.resolve_types(timing) == (fleet.TIMING_TYPE,)
    assert fleet.image_arch(fleet.ARM_SCORE_TYPES[0]) == "arm64"


def test_an_arm_scoring_box_gets_the_aarch64_tools(tmp_path, monkeypatch):
    tools = tmp_path / "aarch64"
    (tools / "bin").mkdir(parents=True)
    model = tools / "share" / "vmaf" / "model" / "vmaf_v1.0.16"
    model.mkdir(parents=True)
    for t in fleet.ARM_SCORE_TOOLS:
        (tools / "bin" / t).write_text("x")
    (model / fleet.VMAF_MODEL_NAME).write_text("{}")
    (tmp_path / "nanocodec").write_text("x")
    monkeypatch.setattr(fleet, "ARM_TOOLS", tools)
    monkeypatch.setattr(fleet, "ARM_NANOCODEC", tmp_path / "nanocodec")
    run, staged = tmp_path / "run", tmp_path / "staged"
    (run / "stage" / "bin").mkdir(parents=True)
    staged.mkdir()
    fleet.link_score_tools(run, staged, "arm64", None)
    assert sorted(p.name for p in (staged / "bin").iterdir()) == sorted(
        [*fleet.ARM_SCORE_TOOLS, "nanocodec", fleet.VMAF_MODEL_NAME])
    fleet.link_score_tools(run, staged, "amd64", None)
    assert (staged / "bin").resolve() == (run / "stage" / "bin").resolve()


def test_the_rehearsal_takes_whole_clips_largest_and_smallest():
    jobs = [{"id": f"{c}|{n}", "input": c, "w": w, "h": h}
            for c, w, h in (("a", 640, 360), ("b", 1280, 720), ("c", 1920, 1080))
            for n in range(3)]
    jobs.append({"id": "vvenc|b|1", "input": "b", "w": 1280, "h": 720, "expected_s": 850.0})
    picked = fleet.rehearsal_jobs(jobs, 2)
    assert {j["input"] for j in picked} == {"a", "b", "c"}
    assert "vvenc|b|1" in {j["id"] for j in picked}  # the long tail is rehearsed too


def test_a_short_queue_is_shared_over_every_slot():
    q = fleet.WorkQueue(jobs({"a": 21}), slots=11)
    sizes = [len(q.take(f"box{i}-0", set()) or []) for i in range(11)]
    assert all(sizes) and max(sizes) == 2  # every box gets work at once
    assert sum(sizes) + q.pending_count() == 21


def test_a_full_launch_needs_a_rehearsal_and_an_approval_on_the_same_tools(tmp_path):
    """RCA R1: the gate refuses until both records exist for this exact tools version."""
    log = tmp_path / "gate.jsonl"
    assert "rehearsal" in fleet.gate_refusal("abc", log) and "approved" in fleet.gate_refusal("abc", log)
    log.write_text(json.dumps({"kind": "rehearsal", "tools": "abc"}) + "\n")
    assert "approved" in fleet.gate_refusal("abc", log)
    with open(log, "a") as f:
        f.write(json.dumps({"kind": "approval", "tools": "old"}) + "\n")
    assert "approved" in fleet.gate_refusal("abc", log)  # another version's approval does not count
    with open(log, "a") as f:
        f.write(json.dumps({"kind": "approval", "tools": "abc"}) + "\n")
    assert fleet.gate_refusal("abc", log) is None
    assert fleet.gate_refusal("def", log) is not None


def test_the_tools_version_ignores_commits_outside_the_tools(tmp_path, monkeypatch):
    """A rehearsal stays valid across a commit that does not touch TOOLS_PATHS."""
    import subprocess
    git = lambda *args: subprocess.run(["git", "-C", str(tmp_path), *args], check=True,  # noqa: E731
                                       capture_output=True)
    git("init", "-q")
    for path in fleet.TOOLS_PATHS:  # a directory gets a file in it, a file is written
        target = tmp_path / path if Path(path).suffix else tmp_path / path / "x.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("1\n")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "tools")
    monkeypatch.setattr(fleet, "REPO", tmp_path)
    before = fleet.tools_version()
    (tmp_path / "README.md").write_text("doc\n")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "doc")
    assert fleet.tools_version() == before
    (tmp_path / "tools" / "bench" / "x.py").write_text("2\n")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "tool")
    assert fleet.tools_version() != before


def test_plan_spend_counts_only_the_benchmarks_own_rows():
    """RCA R6: the ledger total is the whole account; the plan's spend is its own rows."""
    ledger = "\n".join([
        "| 2026-11-24T07:00:00Z | i-1 | c8i.2xlarge / eu-north-1 | bench (started) | — | — | cap $0.30 |",
        "| 2026-11-24T07:30:00Z | i-1 | c8i.2xlarge / eu-north-1 | bench (ended) | a | b | $0.05 |",
        "| 2026-11-24T07:40:00Z | i-2 | c7i.24xlarge / eu-north-1 | measure-warm (ended) | a | b | $2.00 |",
        "| 2026-11-24T08:00:00Z | i-3 | c8g.8xlarge / eu-north-1 | bench (recovered after the driver died) | a | terminated | $0.10 |",
        "| 2026-11-23T23:00:00Z | i-4 | c8a.2xlarge / eu-west-1 | bench (ended) | a | b | $9.00 |",
        "**Spent so far: $711.15 of $750.**"])
    assert abs(fleet.purpose_spend(ledger, "bench", "2026-11-24T06:54") - 0.15) < 1e-9


def test_a_box_rate_ignores_its_setup():
    """RCA R5: a box that started its work 2 minutes ago (after 8 minutes of setup) and has
    finished 5 more jobs since its first is at 150 per hour, not the 30 a 10-minute window gives."""
    now = 10_000.0
    records = [{"end_unix": now - 120 + 20 * i} for i in range(6)]
    assert round(fleet.group_rate(records, now)) == 150
    assert fleet.group_rate([{"end_unix": now - 700}], now) == 0.0  # silent for the window


def test_missing_and_status_count_box_records_before_the_merge(tmp_path):
    """RCA R5: valid records still in a box's directory count for `missing` too."""
    run = tmp_path / "run"
    box = run / "series" / "c8i" / "box0-0"
    box.mkdir(parents=True)
    (run / "jobs.json").write_text(json.dumps([{"id": "j0"}, {"id": "j1"}]))
    records = [{"id": "canary-start|j0", "canary": "start", "cpu_s": 1.0, "seq": 0},
               {"reference": "x264-medium", "cpu_s": 1.0, "seq": 0},
               {"reference": "nanocodec-av1@medium", "cpu_s": 2.0, "seq": 0},
               {"id": "j0", "stream_sha256": "h", "stream": "streams/j0", "seq": 0, "cpu_s": 1.0},
               {"reference": "x264-medium", "cpu_s": 1.0, "seq": 1},
               {"reference": "nanocodec-av1@medium", "cpu_s": 2.0, "seq": 1}]
    (box / "encodes.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
    assert fleet.valid_ids([run], "c8i") == {"j0"}
    assert [j["id"] for j in fleet.missing([run], "c8i")] == ["j1"]


def test_placement_scores_shrink_the_request_to_what_the_account_may_explore():
    """RCA R7: AWS caps the capacity a placement-score request may explore; preflight halves it."""
    from botocore.exceptions import ClientError

    class Ec2:
        def __init__(self):
            self.asked = []

        def get_spot_placement_scores(self, TargetCapacity, **_):
            self.asked.append(TargetCapacity)
            if TargetCapacity > 30:
                raise ClientError({"Error": {"Code": "TargetCapacityLimitExceeded"}}, "x")
            return {"SpotPlacementScores": []}

    ec2 = Ec2()
    assert fleet.placement_scores(ec2, "c8i.2xlarge", 100, "eu-north-1")["boxes"] == 25
    assert ec2.asked == [100, 50, 25]


def test_the_queue_hands_out_the_longest_jobs_first_and_alone():
    """RCA R10: descending expected duration; a job past UNIT_SECONDS is its own unit."""
    work = [{"id": f"s{i}", "input": "a", "expected_s": 10.0} for i in range(30)]
    work += [{"id": "vvenc", "input": "b", "expected_s": 850.0},
             {"id": "x265", "input": "a", "expected_s": 300.0}]
    q = fleet.WorkQueue(work, slots=4)
    assert [j["id"] for j in q.take("box0-0", set())] == ["vvenc"]
    assert [j["id"] for j in q.take("box1-0", set())] == ["x265"]
    assert all(j["expected_s"] == 10.0 for j in q.take("box2-0", {"a"}))


def test_estimates_and_fleet_size_come_from_expected_seconds():
    """RCA R10: a job's own measured time, else its family's per-pixel rate; boxes from the
    expected seconds over the target, not from the job count."""
    work = [{"id": "f|a|1", "family": "f", "w": 100, "h": 100, "frames": 10},
            {"id": "f|b|1", "family": "f", "w": 200, "h": 100, "frames": 10},
            {"id": "g|a|1", "family": "g", "w": 100, "h": 100, "frames": 10}]
    fleet.expected_seconds(work, {"f|a|1": 50.0, "g|a|1": 7.0})
    assert [j["expected_s"] for j in work] == [50.0, 100.0, 7.0]
    assert fleet.fleet_size(work, target_minutes=1, vcpu_cap=1000, box_vcpus=8) == 3
    many = [{"id": str(i), "expected_s": 6.0} for i in range(1000)]
    assert fleet.fleet_size(many, target_minutes=30, vcpu_cap=1000, box_vcpus=8) == 5


def test_canaries_are_short_jobs_of_the_first_clip_not_the_queue_head():
    """Review #1: with longest first, the head is the longest job; canaries must stay short."""
    planned = [{"id": "vvenc", "input": "a", "expected_s": 850.0},
               {"id": "x264-uf", "input": "a", "expected_s": 0.5},
               {"id": "x264-vf", "input": "a", "expected_s": 2.5},
               {"id": "av1-m", "input": "a", "expected_s": 4.0},
               {"id": "hevc-s", "input": "a", "expected_s": 30.0},
               {"id": "other", "input": "b", "expected_s": 3.0}]
    assert [j["id"] for j in fleet.canary_jobs(planned, "a", 3)] == ["x264-vf", "av1-m", "x264-uf"]


def test_the_default_fleet_budget_follows_the_sized_fleet(tmp_path, monkeypatch):
    """Review #9: `--boxes 0` sizes the fleet first; the budget is then 1.5 x boxes x per-box,
    never $0."""
    import types
    run = tmp_path / "run"
    run.mkdir()
    (run / "jobs.json").write_text(json.dumps(
        [{"id": str(i), "input": "a", "family": "f", "w": 1280, "h": 720, "frames": 150,
          "expected_s": 600.0} for i in range(30)]))
    seen = {}
    monkeypatch.setattr(fleet, "require_capped_scope", lambda: None)
    monkeypatch.setattr(fleet, "check_gate", lambda a: None)
    monkeypatch.setattr(fleet, "check_free_disk", lambda p: None)
    monkeypatch.setattr(fleet, "measured_durations", lambda b: {str(i): 600.0 for i in range(30)})
    monkeypatch.setattr(fleet, "Fleet", lambda a, todo, stop, planned=None:
                        (_ for _ in ()).throw(SystemExit(seen.update(a=a))))
    a = types.SimpleNamespace(run=run, series="c8i", boxes=0, target_minutes=30, vcpu_cap=1000,
                              types=("c8i.2xlarge",), fleet_budget=None, budget=0.3)
    try:
        fleet.run(a)
    except SystemExit:
        pass
    assert seen["a"].boxes > 0 and seen["a"].fleet_budget == 1.5 * seen["a"].boxes * 0.3


def fake_fleet(n_jobs: int = 10):
    import threading
    import types
    q = fleet.WorkQueue(jobs({"a": n_jobs}))
    return types.SimpleNamespace(
        stop=threading.Event(), queue=q, may_launch=lambda: True, zones=["z1", "z2"],
        started="150000", zone_fallback=False, failure=None,
        a=types.SimpleNamespace(fleet_budget=10.0))


def test_the_slot_replaces_a_box_lost_to_any_ssh_or_api_error(monkeypatch):
    """Review #2 and #14: transient errors of every kind are a lost box; the slot replaces it and
    settles the work; box names carry the driver's start so a restart never reuses one."""
    import subprocess
    from botocore.exceptions import ClientError
    f = fake_fleet()
    errors = iter([RuntimeError("checkpoint download failed"),
                   subprocess.CalledProcessError(255, "ssh"),
                   ClientError({"Error": {"Code": "RequestLimitExceeded"}}, "x"),
                   fleet.NoCapacity("no zone could launch")])
    names = []

    def serve(fl, name, zone):
        names.append(name)
        error = next(errors, None)
        if error:
            raise error
        unit = fl.queue.take(name, set())
        while unit:
            fl.queue.settle(name, {j["id"] for j in unit})
            unit = fl.queue.take(name, set())
        return "done"

    monkeypatch.setattr(fleet, "serve_box", serve)
    fleet.slot(f, 0)
    assert f.queue.pending_count() == 0 and len(f.queue.settled) == 10
    assert names == [f"box0-150000-{i}" for i in range(5)]
    assert f.zone_fallback  # the NoCapacity opened the avoided zone


def test_the_slot_gives_up_only_after_boxes_in_a_row_settle_nothing(monkeypatch):
    """Review #12: reclaims after useful work do not count; MAX_ATTEMPTS barren boxes do."""
    f = fake_fleet(100)
    calls = []

    def serve(fl, name, zone):
        calls.append(name)
        if len(calls) % 2:  # every other box settles one unit before its reclaim
            unit = fl.queue.take(name, set())
            fl.queue.settle(name, {unit[0]["id"]})
        raise fleet.BoxLost("reclaimed")

    monkeypatch.setattr(fleet, "serve_box", serve)
    monkeypatch.setattr(fleet, "MAX_ATTEMPTS", 2)
    fleet.slot(f, 0)
    assert len(calls) > 2 * fleet.MAX_ATTEMPTS  # productive reclaims kept the slot going


def test_a_non_timing_refusal_stops_the_fleet(monkeypatch):
    f = fake_fleet()
    monkeypatch.setattr(fleet, "serve_box", lambda fl, n, z: (_ for _ in ()).throw(
        fleet.BoxRefused("V9: encoder exit 1")))
    try:
        fleet.slot(f, 0)
    except fleet.BoxRefused:
        pass
    assert f.stop.is_set() and "V9" in f.failure


def test_a_stream_settles_only_when_it_arrived_whole(tmp_path):
    """Review #10: size and SHA-256 equal the record's; a partial copy does not settle."""
    import hashlib
    box = tmp_path / "box0-1-0"
    (box / "streams").mkdir(parents=True)
    data = b"stream bytes"
    rec = {"stream": "streams/s", "bytes": len(data),
           "stream_sha256": hashlib.sha256(data).hexdigest()}
    (box / "streams" / "s").write_bytes(data[:5])
    assert not fleet.stream_arrived(box, rec)
    (box / "streams" / "s").write_bytes(data)
    assert fleet.stream_arrived(box, rec) and (box / ".verified" / "s").exists()


def test_a_poison_job_is_retried_once_then_quarantined():
    """Review #5: a failing job goes back once; after QUARANTINE_AFTER boxes it is listed and
    the rest of the run goes on."""
    q = fleet.WorkQueue(jobs({"a": 2}, seconds=200.0))
    first = q.take("box0-1-0", set())[0]
    q.fail("box0-1-0", first["id"], "V9: exit 1")
    assert q.pending_count() == 2
    taken = [q.take("box1-1-0", set())[0], q.take("box1-1-0", set())[0]]
    assert first["id"] in {j["id"] for j in taken}
    q.fail("box1-1-0", first["id"], "V9: exit 1")
    assert first["id"] in q.quarantined and q.pending_count() == 0
    q.settle("box1-1-0", {j["id"] for j in taken if j["id"] != first["id"]})
    assert q.finished()


def test_a_failed_job_is_recorded_and_the_box_goes_on(tmp_path, monkeypatch):
    """Review #5, box side: a V9 job failure is a record, not the end of the box."""
    import types
    calls = []

    def fake_encode(job, cpu, group, box_, out, checked=True):
        calls.append(job["id"])
        if job["id"] == "j|0":
            raise box.JobFailure("V9: j|0 exit -9")
        name = job["id"].replace("|", "_")
        (box_["streams"] / name).write_bytes(b"s")
        return {"id": job["id"], "stream": f"streams/{name}", "stream_sha256": "h",
                "major_faults": 0, "cpu_s": 1.0, "wall_s": 1.0}
    import sys
    sys.path.insert(0, str(REPO / "tools" / "bench"))
    import box
    monkeypatch.setattr(box, "encode_job", fake_encode)
    monkeypatch.setattr(box, "spot_notice", lambda: None)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "unit-00000.json").write_text(json.dumps([{"id": "j|0", "argv": ["e"]},
                                                        {"id": "j|1", "argv": ["e"]}]))
    (inbox / "END").write_text("")
    a = types.SimpleNamespace(cpu=1, out=tmp_path / "out", inbox=inbox, discard_streams=False)
    streams = tmp_path / "out" / "streams"
    streams.mkdir(parents=True)
    with open(tmp_path / "out" / "encodes.jsonl", "w") as log:
        box.Server(a, {"streams": streams}, tmp_path, log, [], []).serve()
    lines = [json.loads(x) for x in (tmp_path / "out" / "encodes.jsonl").read_text().splitlines()]
    assert any(r.get("failed", "").startswith("V9") for r in lines)
    assert any(r.get("id") == "j|1" and "stream_sha256" in r for r in lines)


def test_a_run_of_only_long_jobs_uses_its_references_as_canaries():
    """2026-11-24: a fill-in of six 1080p x265 jobs (~8 min each) spent its boxes' lifetime on
    9 canary encodes; canaries are never longer than CANARY_MAX_S."""
    planned = [{"id": f"x265|{i}", "input": "a", "expected_s": 480.0} for i in range(6)]
    refs = [{"id": "ref-x264"}, {"id": "ref-av1"}]
    assert fleet.canary_jobs(planned, "a", 3, refs) == refs


def test_jobs_a_box_settled_go_back_when_its_end_canaries_refuse_it():
    """Review #4 (seen in the BP-3.12 rehearsal: a box's end canaries drifted 2.4 % after its 36
    jobs had settled): the queue takes them back, so the fleet codes them again."""
    q = fleet.WorkQueue(jobs({"a": 4}))
    unit = q.take("box0-1-0", set())
    q.settle("box0-1-0", {j["id"] for j in unit})
    assert q.finished()
    assert q.unsettle({j["id"] for j in unit}) == len(unit)
    assert not q.finished() and q.pending_count() == len(unit)


def test_a_history_run_times_its_references_on_the_benchmark_build(tmp_path):
    """The history rehearsal: 17 of 19 days' builds refused the record's reference command
    (unknown params, no CBR); a run carrying the benchmark's build as nanocodec-ref runs its
    references on it."""
    shard = [{"input": "clip", "w": 1280, "h": 720, "fps": "30/1", "frames": 150}]
    plain = fleet.references(shard, tmp_path)
    assert any(j["argv"][0] == "{BIN}/nanocodec" for j in plain)
    (tmp_path / fleet.REFERENCE_BINARY).write_text("x")
    fixed = fleet.references(shard, tmp_path)
    assert all(j["argv"][0] != "{BIN}/nanocodec" for j in fixed)
    assert any(j["argv"][0] == "{BIN}/nanocodec-ref" for j in fixed)


def test_no_box_launches_past_the_accounts_stop_line(monkeypatch):
    """Owner, 2026-11-24 ($800 cap, stop at $790): a box whose worst case would take the
    account past the line is not launched, whatever the fleet's own budget allows."""
    import types
    fake = types.SimpleNamespace(lock=threading.Lock(), spent=0.0, running=0,
                                 a=types.SimpleNamespace(budget=1.0, fleet_budget=100.0))
    monkeypatch.setattr(fleet, "account_spent", lambda: fleet.ACCOUNT_STOP_USD - 1.5)
    assert fleet.Fleet.may_launch(fake) and fake.running == 1
    assert not fleet.Fleet.may_launch(fake) and fake.running == 1


def test_the_preflight_counts_each_streams_bytes():
    """The disk check counts what the PC will hold: a CBR stream at its target (with the
    margin), a Fixed QP stream at the measured bytes per pixel."""
    cbr = {"mode": "cbr", "kbps": 800, "frames": 150, "fps": "30/1"}
    qp = {"mode": "qp", "w": 1920, "h": 1080, "frames": 130, "fps": "30/1"}
    assert fleet.expected_stream_bytes(cbr) == pytest.approx(800e3 / 8 * 5 * fleet.CBR_STREAM_MARGIN)
    assert fleet.expected_stream_bytes(qp) == pytest.approx(1920 * 1080 * 130 * fleet.QP_BYTES_PER_PIXEL)


def test_box_stage_replaces_a_dangling_link(tmp_path):
    """A box stage's link left pointing into a removed worktree is made again (2026-11-25: a
    V27 scoring box failed on FileExistsError)."""
    run = tmp_path / "run"
    (run / "stage" / "clips").mkdir(parents=True)
    (run / "stage" / "score.py").write_text("current")
    target = run / "boxstage" / "score0"
    target.mkdir(parents=True)
    (target / "score.py").symlink_to(tmp_path / "removed-worktree" / "score.py")
    staged = fleet.box_stage(run, "score0")
    assert (staged / "score.py").read_text() == "current"
