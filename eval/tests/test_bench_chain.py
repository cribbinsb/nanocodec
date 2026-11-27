"""BP-4 machine chains (owner, 2026-11-26): a run is claimed atomically before a machine
serves it, a stale claim (its driver gone) is taken over, and a run's remaining families are
the planned ones no machine of it has settled."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import fleet  # noqa: E402


def test_a_run_is_claimed_once_and_released(tmp_path):
    run = tmp_path / "bp4-x-c8g"
    assert fleet.claim_run(run, "c8g", "box0")
    assert fleet.claim_run(run, "c8g", "box0")  # this driver already holds it
    holder = fleet.run_claim(run, "c8g")
    assert holder["pid"] == os.getpid() and holder["box"] == "box0"
    assert not fleet.claimed_elsewhere(run, "c8g")
    fleet.release_run(run, "c8g")
    assert fleet.run_claim(run, "c8g") is None


def test_a_live_drivers_claim_is_respected_and_a_dead_ones_taken_over(tmp_path):
    run = tmp_path / "bp4-y-c8g"
    path = run / "series" / "c8g" / fleet.CLAIM_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"pid": os.getppid(), "box": "other"}))  # alive
    assert fleet.claimed_elsewhere(run, "c8g")
    assert not fleet.claim_run(run, "c8g", "box1")
    path.write_text(json.dumps({"pid": 2 ** 22 + 12345, "box": "gone"}))  # no such process
    assert not fleet.claimed_elsewhere(run, "c8g")
    assert fleet.claim_run(run, "c8g", "box1")
    assert fleet.run_claim(run, "c8g")["box"] == "box1"


def test_remaining_families_leave_out_the_settled_ones(tmp_path, monkeypatch):
    import wholebox
    run = tmp_path / "bp4-z-c8i"
    (run / "series" / "c8i" / "box0-1").mkdir(parents=True)
    (run / wholebox.PLAN_FILE).write_text(json.dumps({
        "machines": [{"index": 0, "families": ["f1", "f2", "f3"]}], "jobs": {"a": 1.0}}))
    (run / "jobs.json").write_text(json.dumps([{"id": "a", "family": "f1"}]))
    (run / "series" / "c8i" / "box0-1" / "encodes.jsonl").write_text("")
    monkeypatch.setattr(wholebox, "settled_families", lambda records: {"f2"})
    plan, jobs, families = fleet.run_remaining(run, "c8i")
    assert families == ["f1", "f3"] and jobs["a"]["expected_s"] == 1.0


def test_the_next_run_of_a_chain_moves_the_machines_own_stop_on():
    command = fleet.next_stage_command("bench-bp4-hist-c8i")
    assert "shutdown -c" in command and f"shutdown -h +{fleet.MACHINE_EXTEND_MINUTES}" in command
    assert "bench-bp4-hist-c8i/clips" in command and "out-bench-bp4-hist-c8i" in command


def test_an_interrupted_halfload_never_chains(monkeypatch, tmp_path):
    """A spot notice or STOP ends halfload with an `interrupted` record: the machine is lost,
    so serve_run raises BoxLost and the chain never takes another run on it."""
    import types
    import pytest
    run = tmp_path / "bp4-r-c8g"
    local = run / "series" / "c8g" / "box0"
    local.mkdir(parents=True)
    (local / "encodes.jsonl").write_text(json.dumps({"interrupted": "mixed"}) + "\n")

    class Box:
        def start_detached(self, *a, **k): pass
        def download_new(self, *a, **k): return 0
        def ssh(self, *a, **k): return types.SimpleNamespace(returncode=0, stdout="0\n")
        def state(self): return "running"
    a = types.SimpleNamespace(series="c8g", chaos=False)
    import threading
    monkeypatch.setattr(fleet.wholebox if hasattr(fleet, "wholebox") else __import__("wholebox"),
                        "POLL_S", 0, raising=False)
    with pytest.raises(fleet.BoxLost):
        fleet.serve_run(a, Box(), run, tmp_path, "bench", "box0", threading.Event(), "none")


def test_regions_near_the_bid_are_tried_last():
    import wholebox
    prices = {"eu-north-1": 0.41, "eu-west-1": 1.92, "eu-central-1": 1.10, "us-west-2": 1.30}
    regions = wholebox.capacity_regions(lambda r, t: prices.get(r), "c8g.metal-24xl")
    assert regions[0] == "eu-north-1" and regions[-1] == "eu-west-1"
    assert regions.index("eu-central-1") < regions.index("us-west-2")


def test_halfload_clock_rule_matches_the_report():
    import halfload
    import report
    assert (halfload.CLOCK_MIN_SHARE, halfload.CLOCK_MIN_WALL_S) == \
        (report.MIN_CLOCK_SHARE, report.MIN_CLOCK_WALL_S)


def test_a_chained_run_starts_with_a_marker_naming_the_run_with_the_start_set():
    import wholebox
    marker = json.loads(fleet.chain_marker("box0-1", "bp4-vc1k-refs-c8i"))
    assert marker == {"check_done": "start", "box": "box0-1", "chained_from": "bp4-vc1k-refs-c8i"}
    records = [marker, {"family_done": "x264-medium"}]
    assert wholebox.settled_families(records) == {"x264-medium"}
    assert fleet.chain_marker("b", "r") in fleet.next_stage_command("bench-r", fleet.chain_marker("b", "r")) \
        or "chained_from" in fleet.next_stage_command("bench-r", fleet.chain_marker("b", "r"))


def test_a_points_time_stands_without_its_family_finishing_on_its_machine():
    """D13 as amended 2026-11-26 21:28 (owner): a point coded on a machine whose start check
    set passed keeps its time even though that machine never finished the family."""
    import report
    box = "box0-1"
    records = [{"box": {"layout": "half"}, "box_info": {"layout": "half"}},
               {"check": "start", "box": box, "id": "check-start-0|x264-medium|c|cbr500",
                "family": "x264-medium", "cpu_s": 1.0, "stream_sha256": "s0"},
               {"check_done": "start", "box": box},
               {"id": "fam|clip|cbr500", "family": "fam", "box": box, "cpu_s": 2.0,
                "wall_s": 2.0, "stream_sha256": "s1"}]  # no family_done for "fam"
    valid, refused = report.half_records("c8i", records)
    assert refused == {} and [r["id"] for r in valid] == ["fam|clip|cbr500"]


def test_a_point_without_a_valid_time_counts_for_quality_but_not_speed(tmp_path):
    """Owner, 2026-11-26: a point with a valid stream and score but no valid time is a point of
    its curve (quality, BD), out of its family's speed, and listed; V13 does not refuse it."""
    import report
    run = tmp_path / "bp4-x-refs-c8i"
    d = run / "series" / "c8i"
    d.mkdir(parents=True)
    box = "box0-1"
    recs = [{"box_info": {"layout": "half"}},
            {"check": "start", "box": box, "id": "check-start-0|x264-medium|c|cbr500",
             "family": "x264-medium", "cpu_s": 1.0, "stream_sha256": "s0"},
            {"check_done": "start", "box": box},
            {"id": "f|c|cbr500", "family": "f", "box": box, "cpu_s": 2.0, "wall_s": 2.0,
             "stream_sha256": "sa"},
            {"id": "f|c|cbr800", "family": "f", "box": box, "cpu_s": 2.0, "wall_s": 2.0,
             "stream_sha256": "sb"},
            {"id": "f|c|cbr800", "family": "f", "box": box, "timing_refused": "V7: slow clock"}]
    (d / "encodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    scores = [{"id": i, "series": "c8i", "stream_sha256": h, "pooled": {}, "per_frame": {}}
              for i, h in (("f|c|cbr500", "sa"), ("f|c|cbr800", "sb"))]
    (d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    jobs = {i: {"id": i, "family": "f", "w": 16, "h": 16, "frames": 1}
            for i in ("f|c|cbr500", "f|c|cbr800")}
    report.UNTIMED_POINTS.clear()
    points = report.load_series(run, jobs)["c8i"]
    assert set(points) == set(jobs)
    assert points["f|c|cbr800"]["untimed"] and not points["f|c|cbr500"].get("untimed")
    assert report.UNTIMED_POINTS == [{"series": "c8i", "id": "f|c|cbr800",
                                      "reason": "V7: slow clock"}]
