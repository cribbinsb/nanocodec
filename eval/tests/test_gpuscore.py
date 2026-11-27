"""nceval.gpuscore and aws_measure's GPU path: the guards, the fallback, the encode rounds
and a reclaim mid-round (nothing is rented, no GPU is used)."""
import time
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_measure  # noqa: E402
from nceval import gpuscore  # noqa: E402


def test_the_memory_gate_needs_the_free_memory_it_asks_for():
    assert gpuscore.gate_reason(gpuscore.START_FREE_MIB, lambda: 11788) is None
    assert "under 8192" in gpuscore.gate_reason(gpuscore.START_FREE_MIB, lambda: 8000)
    assert gpuscore.gate_reason(gpuscore.BATCH_FREE_MIB, lambda: 8000) is None
    assert gpuscore.gate_reason(gpuscore.START_FREE_MIB, lambda: None).startswith("GPU memory")


def test_the_gate_has_no_fixed_hours():
    """Coordinator, 2026-11-24: no quiet hours; the free-memory gate decides at 02:19 too."""
    assert not hasattr(gpuscore, "in_quiet_hours")
    assert gpuscore.gate_reason(0, lambda: 32000) is None


def test_the_batch_gate_waits_for_memory_then_gives_up(monkeypatch):
    free = iter([1000, 1000, 5000])
    slept = []
    assert gpuscore.wait_for_gpu(4096, lambda: next(free), wait_s=60, poll_s=1,
                                 sleep=slept.append) is None
    assert slept == [1, 1]
    assert "under 4096" in gpuscore.wait_for_gpu(4096, lambda: 1000, wait_s=0, poll_s=1,
                                                 sleep=slept.append)


def rounds(tmp_path, run_round, gpu_check=lambda: None):
    return gpuscore.EncodeRounds(["python3", "-m", "nceval.quick", "--json", "x"],
                                 tmp_path / "home" / "prebuilt", tmp_path / "work",
                                 run_round, gpu_check)


def test_a_guard_failing_before_a_batch_falls_back_to_cpu_scoring(tmp_path):
    encode = rounds(tmp_path, lambda argv: 0, gpu_check=lambda: "1000 MiB free")
    with pytest.raises(SystemExit) as stop:
        encode.ensure([1], lambda jobs: jobs)
    assert stop.value.code == gpuscore.FALLBACK_EXIT


def test_a_failed_round_falls_back_to_cpu_scoring(tmp_path):
    encode = rounds(tmp_path, lambda argv: 1)
    with pytest.raises(SystemExit) as stop:
        encode.ensure([1], lambda jobs: jobs)
    assert stop.value.code == gpuscore.FALLBACK_EXIT


def test_one_round_per_batch_seeded_with_the_used_results(tmp_path):
    work = tmp_path / "work"
    used = work / "realuse" / "f150-ip300" / "clip" / "fam" / "cbr300-abc.json"
    used.parent.mkdir(parents=True)
    used.write_text("{}")
    missing = {"a", "b", "c"}
    calls = []

    def run_round(argv):
        calls.append(argv)
        seed = Path(argv[argv.index("--seed") + 1])
        assert (seed / used.relative_to(work)).read_text() == "{}"
        missing.difference_update({"a", "b"})  # the box brought back two of three
        return 0
    encode = rounds(tmp_path, run_round)
    encode.note(used)
    encode.ensure(["a", "b", "c"], lambda jobs: [j for j in jobs if j in missing])
    encode.ensure(["a", "b"], lambda jobs: [j for j in jobs if j in missing])  # no new round
    assert len(calls) == 1 and encode.rounds == 1 and encode.local_encodes == 1
    assert calls[0][calls[0].index("--") + 1:] == encode.command


def complete(prebuilt: Path, key: str, suffix: str = ".h264") -> None:
    base = prebuilt / key
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (suffix, ".recon.txt", ".times.txt"):
        Path(f"{base}{ext}").write_text(ext)
    Path(f"{base}.done").touch()


def test_a_reclaim_mid_round_loses_only_the_points_in_flight(tmp_path):
    """The box finished two points, was writing a third (no .done yet) when it was
    reclaimed: the pull brings the two complete ones into the PC's prebuilt directory,
    the third does not count as there, and the retry's box is told to skip the two."""
    back, target = tmp_path / "back" / aws_measure.PREBUILT / "0", tmp_path / "prebuilt"
    complete(back, "clip/fam/cbr300-a")
    complete(back, "clip/fam/cbr600-b")
    Path(f"{back}/clip/fam/cbr900-c.h264").parent.mkdir(parents=True, exist_ok=True)
    Path(f"{back}/clip/fam/cbr900-c.h264").write_text("half")  # in flight
    have = tmp_path / "have.txt"
    have.write_text("")
    pull = aws_measure.Pull([target], [have])
    assert pull.merge(tmp_path / "back") == 2
    assert gpuscore.done_keys(target) == ["clip/fam/cbr300-a", "clip/fam/cbr600-b"]
    pull.refresh_have()
    assert have.read_text().split() == ["clip/fam/cbr300-a", "clip/fam/cbr600-b"]
    assert pull.remote(aws_measure.warm_layout("job")).endswith("/prebuilt")


def test_a_reclaimed_round_s_retry_skips_what_came_back(tmp_path, monkeypatch):
    target, have = tmp_path / "prebuilt", tmp_path / "have.txt"
    have.write_text("")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.txt").write_text("")
    monkeypatch.setattr(aws_measure, "CLOUD_WORK", tmp_path / "cloud")
    runs = [type("R", (), {"clips": [], "remote_json": "r.json"})()]
    seen = []

    def warm_session(*args):
        seen.append(have.read_text().split())
        if len(seen) == 1:
            complete(target, "clip/fam/cbr300-a")  # pulled while it ran
            raise aws_measure.RemoteFailure("the instance is terminated: "
                                            "Server.SpotInstanceTermination", [])
        return {"stage_minutes": {}}
    monkeypatch.setattr(aws_measure, "warm_session", warm_session)
    aws_measure.rent(None, True, aws_measure.ARM, (), 0, None, stage, tmp_path, [], {}, runs,
                     aws_measure.warm_layout("job"), {}, aws_measure.Pull([target], [have]))
    assert seen == [[], ["clip/fam/cbr300-a"]]


def test_the_box_runs_the_command_encode_only_in_its_own_cache():
    layout = aws_measure.warm_layout("job")
    env = aws_measure.encode_env(layout, 1)
    assert f"{gpuscore.ENCODE_ONLY_ENV}=1" in env
    assert f"NC_WORK={layout.root}/{aws_measure.ENCODE_WORK}/1" in env  # its own cache
    setup = " && ".join(aws_measure.encode_setup(layout, 2))
    assert f"{layout.root}/seed/1/." in setup and f"{layout.root}/have-1.txt" in setup


def test_a_full_service_scores_on_a_box_instead(tmp_path, monkeypatch):
    monkeypatch.setattr(gpuscore, "queue_dir", lambda: tmp_path / "queue")
    monkeypatch.setattr(gpuscore, "QUEUE_WAIT_MINUTES", 0)
    holders = [gpuscore.GpuQueue() for _ in range(gpuscore.SERVICE_COMMANDS)]
    assert all(h.acquire(wait_s=0) is None for h in holders)
    commands = [["python3", "-m", "nceval.quick", "--json", "x"]]
    assert aws_measure.gpu_score(None, commands)[0].startswith("queue wait")
    holders[0].release()
    monkeypatch.setattr(gpuscore, "gate_reason", lambda minimum: "GPU memory: 7000 MiB free")
    assert aws_measure.gpu_score(None, commands) == ["GPU memory: 7000 MiB free"]
    for holder in holders[1:]:
        holder.release()


def test_have_keys_reads_the_pc_s_list(tmp_path):
    (tmp_path / gpuscore.HAVE_FILE).write_text("a/b/c\nd/e/f\n")
    assert gpuscore.have_keys(tmp_path) == {"a/b/c", "d/e/f"}
    assert json.dumps(sorted(gpuscore.have_keys(tmp_path / "none"))) == "[]"


def test_every_runner_of_a_command_shares_one_rounds_instance(tmp_path, monkeypatch):
    monkeypatch.setenv(gpuscore.ROUND_COMMAND_ENV, json.dumps(["python3", "-m", "nceval.quick"]))
    monkeypatch.setenv(gpuscore.PREBUILT_ENV, str(tmp_path / "prebuilt"))
    monkeypatch.delenv(gpuscore.ENCODE_ONLY_ENV, raising=False)
    first = gpuscore.EncodeRounds.from_env(tmp_path / "work")
    assert first is not None and gpuscore.EncodeRounds.from_env(tmp_path / "work") is first
    monkeypatch.setenv(gpuscore.ENCODE_ONLY_ENV, "1")  # the box: no rounds
    assert gpuscore.EncodeRounds.from_env(tmp_path / "work") is None


def test_the_disk_budget_is_a_guard_before_every_round(tmp_path, monkeypatch):
    """2026-11-24: a quick's round encoded its points, then the run died on the work
    directory's budget with no fallback. The budget is now checked before each batch
    (so before its round) and fails over to the CPU path."""
    def over(work):
        raise gpuscore.DiskBudgetExceeded("work holds 111 GiB (budget 100 GiB)")
    monkeypatch.setattr(gpuscore, "check_disk", over)
    rounds_run = []
    encode = gpuscore.EncodeRounds(["python3", "-m", "nceval.quick"], tmp_path / "prebuilt",
                                   tmp_path / "work", rounds_run.append,
                                   gpu_check=None)
    monkeypatch.setattr(gpuscore, "wait_for_gpu", lambda: None)
    with pytest.raises(SystemExit) as stop:
        encode.ensure([1], lambda jobs: jobs)
    assert stop.value.code == gpuscore.FALLBACK_EXIT and rounds_run == []
    assert "111 GiB" in gpuscore.local_reason(tmp_path, lambda: None)


def test_a_budget_hit_while_scoring_falls_back_instead_of_failing(tmp_path, monkeypatch):
    from nceval import realuse
    runner = realuse.Runner.__new__(realuse.Runner)
    runner.cached_only, runner.skip_missing, runner.skipped = False, False, 0
    runner.encode_only, runner.workers = False, 1
    runner.rounds = gpuscore.EncodeRounds(["x"], tmp_path / "p", tmp_path, lambda a: 0,
                                          gpu_check=lambda: None)

    def full(job):
        raise gpuscore.DiskBudgetExceeded("full")
    runner._measure_one = full
    point = type("P", (), {"clip": type("C", (), {"name": "c"})()})()
    bench = type("B", (), {"cached": lambda self, p: True, "has_prebuilt": lambda self, p: True})()
    with pytest.raises(SystemExit) as stop:
        runner.run_jobs([(bench, point)])
    assert stop.value.code == gpuscore.FALLBACK_EXIT
    runner.rounds = None  # outside GPU scoring the budget still fails the run
    with pytest.raises(gpuscore.DiskBudgetExceeded):
        runner.run_jobs([(bench, point)])


def test_a_retry_of_the_same_command_reuses_its_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(aws_measure, "STAGE_ROOT", tmp_path)
    command = aws_measure.Command.parse(["python3", "-m", "nceval.quick", "--rotation", "d",
                                         "--json", "x"])
    home = aws_measure.prebuilt_home(command)
    assert aws_measure.prebuilt_home(aws_measure.Command.parse(command.argv())) == home
    other = aws_measure.Command.parse(["python3", "-m", "nceval.quick", "--rotation", "e",
                                       "--json", "x"])
    assert aws_measure.prebuilt_home(other) != home
    complete(home / aws_measure.PREBUILT, "clip/fam/cbr300-a")
    assert aws_measure.prune_prebuilt(tmp_path) == 0  # fresh: kept for the retry
    assert aws_measure.prune_prebuilt(tmp_path, now=time.time() + 25 * 3600) == 1
    assert not home.exists()


def entry_env(tmp_path, index):
    return {gpuscore.PREBUILT_ENV: str(tmp_path / f"home{index}" / "prebuilt"),
            gpuscore.ROUND_COMMAND_ENV: json.dumps(["python3", "-m", "nceval.quick"])}


def test_a_batch_runs_side_by_side_and_records_each_status(tmp_path):
    spec = [{"argv": [sys.executable, "-c", f"import sys; sys.exit({code})"],
             "env": entry_env(tmp_path, code), "status": str(tmp_path / f"s{code}")}
            for code in (0, gpuscore.FALLBACK_EXIT)]
    (tmp_path / "batch.json").write_text(json.dumps(spec))
    gpuscore.run_batch(tmp_path / "batch.json", lambda argv: 0)
    assert (tmp_path / "s0").read_text().strip() == "0"
    assert (tmp_path / f"s{gpuscore.FALLBACK_EXIT}").read_text().strip() == "75"


def test_which_commands_take_the_gpu_path():
    quick = ["python3", "-m", "nceval.quick", "--json", "x"]
    screen = ["python3", "-m", "nceval.screen", "--codec", "h264", "--json", "x"]
    fixedqp = ["python3", "-m", "nceval.fixedqp", "gap", "--json", "x"]
    assert aws_measure.gpu_scored([quick], "auto", False)
    assert aws_measure.gpu_scored([screen], "auto", False)
    assert aws_measure.gpu_scored([screen], "gpu", False)
    assert not aws_measure.gpu_scored([fixedqp], "gpu", False)
    assert aws_measure.gpu_scored([screen, screen], "auto", True) == aws_measure.GPU_DEFAULT_BATCH


def test_points_of_one_clip_window_in_flight_share_its_frames(tmp_path):
    from nceval.cbr import Bench
    made = []

    class Frames:
        def __init__(self):
            self.closed = False
            made.append(self)

        def close(self, raise_errors=True):
            self.closed = True
    bench = Bench(None, 150, 300, tmp_path, prebuilt=tmp_path / "prebuilt")
    bench.frames_of = lambda clip: Frames()
    clip = type("C", (), {"name": "c", "yuv": Path("/c.yuv"), "start_frame": 7})()
    with bench.scoring_frames(clip) as first, bench.scoring_frames(clip) as second:
        assert first is second and not first.closed
    assert first.closed and len(made) == 1
    with bench.scoring_frames(clip) as third:  # a later point: a new one
        assert third is not first
    assert len(made) == 2 and not bench._shared_frames


def test_the_service_admits_its_places_then_queues_then_gives_up(tmp_path):
    """SERVICE_COMMANDS commands score at once; the next waits for a place, and takes
    one as soon as a command ends; it gives up (CPU scoring) only after its wait."""
    places = [gpuscore.GpuQueue(tmp_path, places=2) for _ in range(2)]
    assert all(p.acquire(wait_s=0) is None for p in places)
    waiting, slept = gpuscore.GpuQueue(tmp_path, places=2), []

    def sleep(seconds):
        slept.append(seconds)
        places[1].release()  # a command ends while the next waits
    assert waiting.acquire(wait_s=600, sleep=sleep) is None
    assert slept == [gpuscore.QUEUE_POLL_SECONDS]
    late = gpuscore.GpuQueue(tmp_path, places=2)
    reason = late.acquire(wait_s=0)
    assert reason.startswith("all 2 places busy") and "pid" in reason
    waiting.release()
    places[0].release()


def test_each_command_gets_an_equal_share_of_the_workers():
    assert gpuscore.worker_share() * gpuscore.SERVICE_COMMANDS >= gpuscore.LOCAL_SLOTS
    assert gpuscore.worker_share() >= 1


def test_fixedqp_ab_takes_the_gpu_path_but_gap_and_cost_stay_on_boxes():
    ab = ["python3", "-m", "nceval.fixedqp", "ab", "--codec", "av1", "--json", "x"]
    gap = ["python3", "-m", "nceval.fixedqp", "gap", "--codec", "av1", "--json", "x"]
    assert aws_measure.gpu_scored([ab], "gpu", False)
    assert not aws_measure.gpu_scored([gap], "gpu", False)
    assert not aws_measure.gpu_scored([gap], "auto", False)
    assert aws_measure.gpu_scored([ab], "auto", False) == \
        (("nceval.fixedqp", "ab") in aws_measure.GPU_DEFAULT_COMMANDS)


def test_an_encode_round_of_our_encoder_may_go_to_an_arm_box_whatever_its_module():
    command = aws_measure.Command.parse(["python3", "-m", "nceval.fixedqp", "gap", "--json", "x"])
    run = aws_measure.Plan(command, [], [])
    assert "reference encoders" in aws_measure.arm_reason([run], [])
    run.encode_only = True
    reason = aws_measure.arm_reason([run], [])
    assert reason is None or "reference encoders" not in reason


def test_the_box_s_rate_probes_come_here_without_its_cpu_times(tmp_path):
    """fixedqp chooses its quantizers on rate probes: the box's are kept here so the
    choice is not coded again, but not its CPU seconds (its own load)."""
    source, work = tmp_path / "rate-probes", tmp_path / "work"
    probe = source / "realuse" / "f150-ip300" / "clip" / "fam" / "rate-qp30-abc.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(json.dumps({"kbps": 512.5, "cpu_s": 3.2}))
    mine = work / "realuse" / "f150-ip300" / "clip" / "fam" / "rate-qp32-abc.json"
    mine.parent.mkdir(parents=True)
    mine.write_text(json.dumps({"kbps": 400.0, "cpu_s": 1.0}))
    (source / mine.relative_to(work)).write_text(json.dumps({"kbps": 999.0}))
    assert gpuscore.import_rate_probes(source, work) == 1
    assert json.loads((work / probe.relative_to(source)).read_text()) == {"kbps": 512.5}
    assert json.loads(mine.read_text())["kbps"] == 400.0  # kept


def test_a_prefetch_round_respects_the_guards_and_imports_probes(tmp_path):
    calls = []
    encode = rounds(tmp_path, lambda argv: calls.append(argv) or 0)
    encode.prefetch()
    assert len(calls) == 1 and encode.rounds == 1
    blocked = rounds(tmp_path, calls.append, gpu_check=lambda: "1000 MiB free")
    with pytest.raises(SystemExit) as stop:
        blocked.prefetch()
    assert stop.value.code == gpuscore.FALLBACK_EXIT and len(calls) == 1


def test_a_command_starts_only_with_memory_headroom_in_the_slice():
    gib = 2 ** 30
    assert gpuscore.memory_reason(1, lambda: (int(6.33 * gib), 10 * gib)) is None
    assert "GiB needed" in gpuscore.memory_reason(1, lambda: (int(9.5 * gib), 10 * gib))
    # a batch needs room for the commands it runs at once, not all of them
    big = gpuscore.memory_reason(26, lambda: (1 * gib, 10 * gib))
    assert (big is None) == (gpuscore.COMMAND_MEMORY_BYTES * gpuscore.BATCH_PARALLEL <= 9 * gib)
    assert gpuscore.memory_reason(1, lambda: None) is None  # no slice yet: nothing in it


def test_a_full_slice_sends_the_command_to_cpu_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(gpuscore, "queue_dir", lambda: tmp_path / "queue")
    monkeypatch.setattr(gpuscore, "memory_reason", lambda n: "no room")
    monkeypatch.setattr(gpuscore, "local_reason", lambda *a: None)
    args = type("A", (), {"gpu_wait": 0})()
    assert aws_measure.gpu_score(args, [["python3", "-m", "nceval.quick", "--json", "x"]]) == \
        ["service memory: no room"]
    assert gpuscore.GpuQueue(tmp_path / "queue").acquire(wait_s=0) is None  # place released


def test_a_batch_runs_at_most_batch_parallel_commands_at_once(tmp_path):
    marker = tmp_path / "running"
    marker.mkdir()
    code = ("import os, sys, time; d = sys.argv[1]; n = os.path.join(d, str(os.getpid())); "
            "open(n, 'w').close(); time.sleep(0.3); "
            "c = len(os.listdir(d)); os.remove(n); sys.exit(min(c, 50))")
    spec = [{"argv": [sys.executable, "-c", code, str(marker)], "env": entry_env(tmp_path, i),
             "status": str(tmp_path / f"s{i}")} for i in range(gpuscore.BATCH_PARALLEL + 3)]
    (tmp_path / "batch.json").write_text(json.dumps(spec))
    gpuscore.run_batch(tmp_path / "batch.json", lambda argv: 0)
    peaks = [int((tmp_path / f"s{i}").read_text()) for i in range(len(spec))]
    assert max(peaks) <= gpuscore.BATCH_PARALLEL


def test_a_batch_s_first_round_packs_every_command_into_one_round(tmp_path):
    """2026-11-24: a 26-quick batch whose commands each ran their own round launched 14
    boxes ($1.19). The batch's first round now carries all 26 commands."""
    rounds_run = []

    def call(argv):
        rounds_run.append(json.loads(Path(argv[argv.index("--encode-rounds") + 1]).read_text()))
        return 0
    spec = [{"argv": [sys.executable, "-c", "pass"], "env": entry_env(tmp_path, i),
             "status": str(tmp_path / f"s{i}")} for i in range(26)]
    (tmp_path / "batch.json").write_text(json.dumps(spec))
    gpuscore.run_batch(tmp_path / "batch.json", call)
    assert len(rounds_run) == 1 and len(rounds_run[0]) == 26
    assert {r["prebuilt"] for r in rounds_run[0]} == \
        {e["env"][gpuscore.PREBUILT_ENV] for e in spec}


def test_requests_waiting_together_go_to_one_round(tmp_path):
    calls = []
    broker = gpuscore.RoundBroker(tmp_path, tmp_path / "rounds.jsonl", None,
                                  lambda argv: calls.append(argv) or 0, gather_s=60)
    for index in range(5):
        (tmp_path / f"req-{index}.json").write_text(json.dumps({"command": [str(index)]}))
    assert broker.serve_once(running=6) == 0  # one more command may still ask
    assert broker.serve_once(running=5) == 5  # every running command is waiting
    assert len(calls) == 1
    assert sorted(p.name for p in tmp_path.glob("req-*.status")) == \
        [f"req-{i}.json.status" for i in range(5)]
    (tmp_path / "req-9.json").write_text("{}")
    assert broker.serve_once(running=3, now=time.time() + 61) == 1  # waited long enough


def test_a_round_past_the_batch_budget_is_refused(tmp_path):
    log = tmp_path / "rounds.jsonl"
    log.write_text(json.dumps({"estimated_usd": 0.2}) + "\n" +
                   json.dumps({"estimated_usd": 0.15}) + "\n")
    calls = []
    broker = gpuscore.RoundBroker(tmp_path, log, 0.3, calls.append)
    assert broker.run_round([{"command": ["x"]}]) == 1 and calls == []
    assert gpuscore.rounds_spent(log) == pytest.approx(0.35)


def test_a_command_s_round_goes_through_the_broker(tmp_path, monkeypatch):
    broker = tmp_path / "broker"
    broker.mkdir()
    monkeypatch.setenv(gpuscore.BROKER_ENV, str(broker))

    def answer(seconds):
        for request in broker.glob("req-*.json"):
            Path(f"{request}.status").write_text("0\n")
            request.unlink()
    monkeypatch.setattr(gpuscore.time, "sleep", answer)
    calls = []
    encode = rounds(tmp_path, calls.append)
    encode.ensure([1], lambda jobs: jobs if encode.rounds == 0 else [])
    assert encode.rounds == 1 and calls == []  # the broker ran it, not this process


def test_a_26_command_round_is_one_box_job(tmp_path, monkeypatch):
    """Fake session: the 26 requests of a batch's first round are staged, rented and
    pulled as ONE warm-box job."""
    rents = []
    monkeypatch.setattr(aws_measure, "STAGE_ROOT", tmp_path / "stage")
    monkeypatch.setattr(aws_measure, "plan", lambda command, index, layout: aws_measure.Plan(
        command, [], [], layout=layout))
    monkeypatch.setattr(aws_measure, "choose_machine", lambda *a: aws_measure.ARM)
    monkeypatch.setattr(aws_measure, "stage", lambda stage_dir, *a: stage_dir.mkdir(parents=True))
    monkeypatch.setattr(aws_measure, "local_decodes", lambda run: {})
    monkeypatch.setattr(aws_measure, "rent", lambda *a: rents.append(a) or {
        "estimated_usd": 0.02, "stage_minutes": {"run": 1.0}, "warm_box": "i-1"})
    requests = []
    for i in range(26):
        (tmp_path / f"seed{i}").mkdir()
        (tmp_path / f"have{i}").write_text("")
        requests.append({"command": ["python3", "-m", "nceval.quick", "--json", f"o{i}"],
                         "prebuilt": str(tmp_path / f"p{i}"), "seed": str(tmp_path / f"seed{i}"),
                         "have": str(tmp_path / f"have{i}")})
    args = type("A", (), {"arch": "auto", "tree_commit": None, "dry_run": False,
                          "run_minutes": 35, "rounds_log": str(tmp_path / "rounds.jsonl")})()
    cloud = aws_measure.encode_rounds(args, requests)
    assert len(rents) == 1 and cloud["commands"] == 26
    runs, pull = rents[0][10], rents[0][13]
    assert len(runs) == 26 and len(pull.targets) == 26
    assert gpuscore.rounds_spent(tmp_path / "rounds.jsonl") == pytest.approx(0.02)


def test_the_box_is_released_after_the_grace_and_its_idle_tail_counted(tmp_path):
    """2026-11-24: the box's 10-min idle tail (~$0.03) cost more than a 6-quick batch's
    round ($0.021); the broker stops it after the grace unless another job is on it."""
    log, now, calls = tmp_path / "rounds.jsonl", [1000.0], []

    def call(argv):
        calls.append(argv)
        if "--encode-rounds" in argv:
            with log.open("a") as out:
                out.write(json.dumps({"warm_box": "i-1", "spot_usd_per_hour": 0.18,
                                      "estimated_usd": 0.021}) + "\n")
        return 0
    broker = gpuscore.RoundBroker(tmp_path, log, None, call, grace_s=120, clock=lambda: now[0])
    broker.run_round([{"command": ["x"]}])
    now[0] += 60
    assert not broker.release()  # within the grace: a follow-up round may still come
    now[0] += 61
    assert broker.release()
    assert calls[-1][-2:] == ["stop-if-idle", "i-1"]
    tail = json.loads(log.read_text().splitlines()[-1])
    assert tail["idle_tail_s"] == 121 and tail["estimated_usd"] == pytest.approx(0.18 * 121 / 3600, abs=1e-4)
    assert gpuscore.rounds_spent(log) == pytest.approx(0.021 + tail["estimated_usd"])
    assert not broker.release(force=True)  # released once


def test_a_box_another_job_uses_is_left_to_its_idle_timer(tmp_path):
    log = tmp_path / "rounds.jsonl"

    def call(argv):
        if "--encode-rounds" in argv:
            log.write_text(json.dumps({"warm_box": "i-2", "spot_usd_per_hour": 0.18,
                                       "estimated_usd": 0.02}) + "\n")
            return 0
        return 1  # stop-if-idle: a job is on it
    broker = gpuscore.RoundBroker(tmp_path, log, None, call)
    broker.run_round([{"command": ["x"]}])
    assert not broker.release(force=True)
    assert len(log.read_text().splitlines()) == 1  # no tail charged to this batch


def test_stop_if_idle_stops_only_a_box_with_no_job(tmp_path, monkeypatch):
    import aws_warm
    monkeypatch.setattr(aws_warm, "REGISTRY", tmp_path / "warm")
    (tmp_path / "warm").mkdir()
    stopped = []
    monkeypatch.setattr(aws_warm.Box, "session",
                        lambda self: type("S", (), {"cleanup": lambda s: stopped.append(
                            self.instance_id)})())
    for name, jobs in (("i-free", {}), ("i-busy", {"other-job": 1})):
        aws_warm.path_of(name).write_text(json.dumps(
            {"instance_id": name, "prepared": True, "jobs": jobs, "vcpus": 32}))
    assert aws_warm.stop_if_idle("i-free") and stopped == ["i-free"]
    assert not aws_warm.path_of("i-free").exists()
    assert not aws_warm.stop_if_idle("i-busy") and stopped == ["i-free"]
    assert not aws_warm.stop_if_idle("i-gone")


def test_a_fallback_is_loud_with_its_cost_and_refused_past_the_budget(monkeypatch, capsys):
    monkeypatch.setattr(aws_measure, "planned_rounds_usd", lambda commands: 0.02)
    monkeypatch.setattr(aws_measure, "planned_cpu_usd", lambda commands: 0.21)
    commands = [["python3", "-m", "nceval.quick", "--json", f"o{i}"] for i in range(6)]
    args = argparse.Namespace(budget=4.0)
    aws_measure.announce_fallback(args, commands, "disk floor: 195 GiB free")
    err = capsys.readouterr().err
    assert "scored on CPU because: disk floor" in err and "(+$0.190)" in err
    with pytest.raises(SystemExit, match="passes --budget"):
        aws_measure.announce_fallback(argparse.Namespace(budget=0.1), commands, "queue wait")


def test_a_fallen_back_command_writes_why(tmp_path, monkeypatch):
    monkeypatch.setenv(gpuscore.FALLBACK_REASON_ENV, str(tmp_path / "why.txt"))
    with pytest.raises(SystemExit):
        raise gpuscore.GpuUnavailable("disk floor: 195 GiB free")
    assert (tmp_path / "why.txt").read_text().strip() == "disk floor: 195 GiB free"


def test_ledger_rows_say_where_they_were_scored(monkeypatch, tmp_path):
    from nceval import levers
    seen = []
    monkeypatch.setattr(levers, "append", lambda rows: seen.extend(rows) or len(rows))
    monkeypatch.setenv(levers.SCORED_ON_ENV, aws_measure.SCORED_GPU)
    levers.record([{"id": "x", "source": "quick.json"}], "nceval.quick")
    assert seen[0]["source"] == f"quick.json; scored on {aws_measure.SCORED_GPU}"


def test_the_box_is_held_while_commands_still_score(tmp_path):
    """2026-11-25: a staged screen's box stopped 2 min into a 9.5-min scoring pass, and its
    next level's round waited ~10 min for a fresh box. While a command runs, the broker
    keeps the box (renewing its idle timer) and releases it only once none runs."""
    log, now, calls = tmp_path / "rounds.jsonl", [1000.0], []

    def call(argv):
        calls.append(argv)
        if "--encode-rounds" in argv:
            with log.open("a") as out:
                out.write(json.dumps({"warm_box": "i-3", "spot_usd_per_hour": 0.18,
                                      "estimated_usd": 0.02}) + "\n")
        return 0
    broker = gpuscore.RoundBroker(tmp_path, log, None, call, grace_s=120, clock=lambda: now[0])
    broker.run_round([{"command": ["x"]}])
    now[0] += 600  # past the grace, but a command still scores
    assert not broker.release(running=1)
    assert broker.hold(running=1)
    assert calls[-1][-2:] == ["hold", "i-3"]
    now[0] += 10
    assert not broker.hold(running=1)  # renewed at most every HOLD_EVERY_SECONDS
    assert not broker.hold(running=0)  # nothing runs: nothing to hold for
    assert broker.release(running=0)
    assert calls[-1][-2:] == ["stop-if-idle", "i-3"]
