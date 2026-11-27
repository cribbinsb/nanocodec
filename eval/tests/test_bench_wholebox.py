"""BP-4 timing on whole machines: halfload.py's slots and family passes, wholebox.py's plan and
capacity rule, and report.py's check-set validity (D13, D14, V6, V24)."""
import json
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
sys.path.insert(0, str(REPO / "tools" / "dev"))
import box  # noqa: E402
import halfload  # noqa: E402
import report  # noqa: E402
import wholebox  # noqa: E402

LSCPU = "\n".join(["# CPU,Core,Socket"] + [f"{c},{c % 4},0" for c in range(8)])  # 4 cores, SMT 2


def test_slots_are_every_second_physical_core_from_core_one():
    cores = halfload.physical_cores(LSCPU)
    assert cores == [0, 1, 2, 3]                      # one thread per core, siblings dropped
    assert halfload.half_slots(cores) == (0, [1, 3])  # core 0 for the harness


def job(name, seconds):
    return {"id": f"fam|{name}", "family": "fam", "expected_s": seconds, "argv": ["x"]}


class FakeEncoder:
    """Records every encode; `refuse` names jobs whose first timed encode fails V7."""

    def __init__(self, refuse=(), always=()):
        self.lock, self.calls, self.refuse, self.always = threading.Lock(), [], set(refuse), set(always)

    def __call__(self, j, cpu, timed):
        with self.lock:
            self.calls.append((j["id"], cpu, timed))
            if timed and (j["id"] in self.always or j["id"] in self.refuse):
                self.refuse.discard(j["id"])
                raise box.TimingRefusal("V7: steal")
        return {"id": j["id"], "cpu_s": j["expected_s"], "stream_sha256": j["id"]}


def run_pass(jobs, encoder, stopped=lambda: False, slots=(1, 3)):
    written = []
    outcome = halfload.FamilyPass("fam", jobs, list(slots), encoder, written.append, stopped).run()
    return outcome, written


def test_a_pass_codes_every_timed_job_once_longest_first_with_same_family_filler():
    jobs = [job(n, s) for n, s in (("a", 1), ("b", 3), ("c", 2), ("d", 0.5), ("e", 0.1))]
    enc = FakeEncoder()
    outcome, written = run_pass(jobs, enc)
    timed = [c[0] for c in enc.calls if c[2]]
    assert outcome == "done" and sorted(timed) == sorted(j["id"] for j in jobs)
    assert timed[:2] in (["fam|b", "fam|c"], ["fam|c", "fam|b"])  # the two longest start first
    assert all(c[0].startswith("filler") for c in enc.calls if not c[2])
    assert {r["id"] for r in written} == {j["id"] for j in jobs}
    assert all(1 <= r["load"] <= 2 for r in written)


def test_a_timing_refusal_is_coded_again_once_and_too_many_refuse_the_family():
    jobs = [job(n, 1) for n in "abcd"]
    outcome, written = run_pass(jobs, FakeEncoder(refuse={"fam|a"}))
    assert outcome == "done" and any("retry" in r for r in written)
    outcome, written = run_pass(jobs, FakeEncoder(always={"fam|a"}))
    assert outcome == "refused"  # one of four refused twice: over MAX_REFUSED_SHARE
    assert any("timing_refused" in r for r in written)


def test_a_mixed_pass_has_no_filler_and_ends_each_family_with_its_last_job():
    jobs = [{**job(n, s), "id": f"{f}|{n}", "family": f}
            for f, n, s in (("f1", "a", 3), ("f2", "b", 2), ("f1", "c", 1), ("f2", "d", 0.5))]
    enc, written = FakeEncoder(), []
    outcome = halfload.FamilyPass("mixed", jobs, [1, 3], enc, written.append, lambda: False,
                                  filler=False).run()
    assert outcome == "done"
    assert all(timed for _, _, timed in enc.calls)  # no filler
    assert sorted(c[0] for c in enc.calls) == sorted(j["id"] for j in jobs)
    ends = [r["family_done"] for r in written if "family_done" in r]
    assert sorted(ends) == ["f1", "f2"]
    for fam in ("f1", "f2"):  # a family ends after its last job's record
        last = max(i for i, r in enumerate(written) if r.get("id", "").startswith(fam))
        assert written.index({"family_done": fam}) > last


def test_a_stop_interrupts_the_pass():
    jobs = [job(n, 1) for n in "abcdef"]
    calls = []
    outcome, _ = run_pass(jobs, FakeEncoder(), stopped=lambda: calls.append(1) or len(calls) > 2)
    assert outcome == "interrupted"


def family(name, n, seconds):
    return [{"id": f"{name}|{i}", "family": name, "expected_s": seconds} for i in range(n)]


def test_families_go_whole_to_the_least_loaded_machine_longest_first():
    fams = wholebox.families_of(family("big", 10, 10) + family("mid", 10, 6) +
                                family("small", 10, 5))
    assert wholebox.assign(fams, 2, slots=5) == [["big"], ["mid", "small"]]
    plan = wholebox.timing_plan(family("big", 10, 10) + family("mid", 10, 6), "c8g", 2, 0.41,
                                slots=5)
    assert plan["instance_type"] == "c8g.metal-24xl"
    walls = sorted(m["expected_s"] for m in plan["machines"])
    assert walls == [round(600 + 1.11 * 12), round(600 + 1.11 * 20)]
    assert plan["expected_cost_usd"] == round(sum(walls) / 3600 * 0.41, 2)


def test_bp4_times_only_on_bare_metal_and_only_the_planned_type():
    assert wholebox.machine_type("c8i") == "c8i.metal-48xl"
    with pytest.raises(SystemExit):
        wholebox.machine_type("c7a")


def test_spill_regions_within_the_price_factor_or_the_owner_cap():
    # owner 2026-11-26: any region up to BENCH_SPOT_CAP_USD, or 1.5x home where that is higher
    cap = wholebox.BENCH_SPOT_CAP_USD
    prices = {"eu-north-1": 0.40, "eu-west-1": 0.59, "eu-central-1": cap - 0.01,
              "us-east-1": cap + 0.01}
    regions = wholebox.capacity_regions(lambda r, t: prices.get(r), "c8g.metal-24xl")
    assert regions == ["eu-north-1", "eu-west-1", "eu-central-1"]


def test_capacity_is_sought_for_hours_then_refused(monkeypatch):
    import aws_session

    class NoRoom:
        def __init__(self, spec, stop, clients):
            pass

        def __enter__(self):
            raise aws_session.NoCapacity("no zone")

    monkeypatch.setattr(aws_session, "Session", NoRoom)
    now = [0.0]
    waits = []

    def sleep(s):
        waits.append(s)
        now[0] += s
        return False
    spec = lambda region: type("S", (), {"instance_type": "c8g.metal-24xl"})()  # noqa: E731
    with pytest.raises(aws_session.NoCapacity, match=f"{wholebox.CAPACITY_WAIT_S // 60} min"):
        wholebox.open_session(spec, ["eu-north-1"], threading.Event(), None,
                              clock=lambda: now[0], sleep=sleep)
    rounds = wholebox.CAPACITY_WAIT_S // wholebox.CAPACITY_RETRY_S
    assert waits == [wholebox.CAPACITY_RETRY_S] * rounds  # a round every minute, for hours


def test_a_region_over_the_session_budget_is_left_out_not_the_search_ended(monkeypatch):
    import aws_session
    tried = []

    class Priced:
        def __init__(self, spec, stop, clients):
            self.region = spec.region

        def __enter__(self):
            tried.append(self.region)
            if self.region == "eu-central-1":
                raise aws_session.BudgetExceeded("over budget")
            if len(tried) < 4:
                raise aws_session.NoCapacity("no zone")
            return self.region

    monkeypatch.setattr(aws_session, "Session", Priced)
    now = [0.0]

    def sleep(s):
        now[0] += s
        return False
    spec = lambda region: type("S", (), {"instance_type": "c8i.metal-48xl",  # noqa: E731
                                         "region": region})()
    got = wholebox.open_session(spec, ["eu-north-1", "eu-central-1"], threading.Event(), None,
                                clock=lambda: now[0], sleep=sleep)
    assert got == "eu-north-1"
    assert tried == ["eu-north-1", "eu-central-1", "eu-north-1", "eu-north-1"]
    tried.clear()  # every region over the budget: refused at once, not after 10 min
    with pytest.raises(aws_session.BudgetExceeded, match="every capacity region"):
        wholebox.open_session(spec, ["eu-central-1"], threading.Event(), None,
                              clock=lambda: now[0], sleep=sleep)


def records(box_name, start=1.0, end=1.0, settled=("fam",), end_set=True):
    """A machine's records: check set at its start, a family, check set at its end."""
    def check(where, level):
        out = [{"box": box_name, "check": where, "id": f"check-{where}-{r}|x", "cpu_s": level,
                "stream_sha256": "c"} for r in range(3)]
        return out + [{"box": box_name, "check_done": where}]
    out = [{"box": box_name, "box_info": {"layout": "half", "cpu_model": "m"}}]
    out += check("start", start)
    out.append({"box": box_name, "id": "fam|1", "family": "fam", "cpu_s": 2.0,
                "stream_sha256": "s"})
    out += [{"box": box_name, "family_done": f} for f in settled]
    return out + (check("end", end) if end_set else [])


def test_a_family_counts_once_finished_after_a_passing_start_set():
    ok, refused = report.half_records("c8g", records("m1") + records("m2"))
    assert {r["box"] for r in ok} == {"m1", "m2"} and not refused
    assert all(r["cpu_s_normalised"] == r["cpu_s"] for r in ok)  # never rescaled
    # m2's end set 3 % slow: its times still count; the drift is reported (D13)
    report.CHECK_SET_DRIFT.clear()
    ok, refused = report.half_records("c8g", records("m1") + records("m2", end=1.03) +
                                      records("m3"))
    assert {r["box"] for r in ok} == {"m1", "m2", "m3"} and not refused
    assert report.CHECK_SET_DRIFT["c8g/m2"] > 0.02
    # m2's start set 3 % slow: its times are refused (V6)
    ok, refused = report.half_records("c8g", records("m1") + records("m2", start=1.03) +
                                      records("m3"))
    assert {r["box"] for r in ok} == {"m1", "m3"} and "m2" in refused
    # a family with no family_done marker (an interrupted pass) does not count (V24)
    ok, _ = report.half_records("c8g", records("m1", settled=()))
    assert ok == []
    # a machine reclaimed after a family, before its end set: the family counts and settles
    lost = records("m4", end_set=False)
    ok, refused = report.half_records("c8g", records("m1") + lost)
    assert {r["box"] for r in ok} == {"m1", "m4"} and not refused
    assert wholebox.settled_families(lost) == {"fam"}


def test_each_tool_warms_up_with_its_shortest_job():
    machine = halfload.Machine.__new__(halfload.Machine)
    machine.warmed, machine.slots, warmed = set(), [1], []
    machine.encode = lambda job, cpu, timed: warmed.append(job["id"])
    jobs = [{"id": "a-long", "argv": ["a"], "expected_s": 90.0},
            {"id": "b-only", "argv": ["b"], "expected_s": 50.0},
            {"id": "a-short", "argv": ["a"], "expected_s": 2.0}]
    machine.warm(jobs)
    assert warmed == ["warmup|a-short", "warmup|b-only"]


def test_check_sets_only_at_a_machines_start_and_end():
    """Owner, 2026-11-25 (D13): whole machines do not drift; a check set when a machine starts
    and one at its end, none between families."""
    machine = halfload.Machine.__new__(halfload.Machine)
    events = []
    machine.run_pass = lambda family, jobs, extra: events.append(family) or "done"
    machine.check_set = lambda where, checks: events.append(where) or True
    machine.write = lambda r: None
    machine.serve([{"family": f, "jobs": []} for f in ("f1", "f2", "f3")], {})
    assert events == ["start", "f1", "f2", "f3", "end"]


def test_a_reclaim_loses_only_the_family_in_progress():
    def check(box_name, where):
        return [{"box": box_name, "check": where, "id": f"check-{where}-0|x", "cpu_s": 1.0,
                 "stream_sha256": "c"}, {"box": box_name, "check_done": where}]

    def fam(box_name, name, done=True):
        out = [{"box": box_name, "id": f"{name}|1", "family": name, "cpu_s": 2.0,
                "stream_sha256": "s"}]
        return out + ([{"box": box_name, "family_done": name}] if done else [])
    head = [{"box": "m", "box_info": {"layout": "half", "cpu_model": "m"}}]
    lost = head + check("m", "start") + fam("m", "a") + fam("m", "b", done=False)
    assert wholebox.settled_families(lost) == {"a"}       # b goes to the replacement
    ok, refused = report.half_records("c8g", lost + head + check("r", "start") +
                                      fam("r", "b") + check("r", "end"))
    assert {(r["box"], r["family"]) for r in ok} == {("m", "a"), ("r", "b")} and not refused
    # no family settles before the machine's start set is complete
    assert wholebox.settled_families(head + fam("m", "a")) == set()


def test_a_start_set_off_the_other_machines_refuses_the_machine(tmp_path):
    import fleet

    def write(name, level):
        (tmp_path / name).mkdir()
        rows = [{"check": "start", "id": f"check-start-{r}|{job}", "cpu_s": level * t,
                 "stream_sha256": "c"} for r in range(3) for job, t in (("x", 1.0), ("y", 4.0))]
        (tmp_path / name / "encodes.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows + [{"check_done": "start"}]) + "\n")
    write("box0-1-0", 1.0)
    assert fleet.judge_start_set("box0-1-0", tmp_path) is None  # alone: the report judges it
    write("box1-1-0", 1.002)
    write("box2-1-0", 1.03)
    assert fleet.judge_start_set("box1-1-0", tmp_path) is None
    assert "V6" in fleet.judge_start_set("box2-1-0", tmp_path)
