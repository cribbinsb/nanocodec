"""tools/dev/heldout_fleet.py: the lane planner, relaunches of a lost or unlaunched
session, and the render's refusal to measure (nothing is rented)."""
import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_heldout_chart  # noqa: E402
import heldout_fleet  # noqa: E402
from nceval.config import load_paths  # noqa: E402
from nceval.gatepolicy import EVALUATION_GATE  # noqa: E402
from nceval.realuse import MissingPoints, Runner  # noqa: E402


def test_lanes_hold_about_the_lane_minutes_and_every_clip_once():
    minutes = {f"c{i}": 2.0 for i in range(20)}  # 40 box-min of work
    lanes = heldout_fleet.plan_lanes(minutes, [], lane_minutes=10, boxes=8)
    assert len(lanes) == 4 and sorted(n for lane in lanes for n in lane) == sorted(minutes)
    assert all(sum(minutes[n] for n in lane) == 10 for lane in lanes)


def test_more_work_than_the_boxes_hold_makes_bigger_lanes_not_more():
    minutes = {f"c{i}": 2.0 for i in range(20)}
    assert len(heldout_fleet.plan_lanes(minutes, [], lane_minutes=10, boxes=2)) == 2


def test_the_clips_others_wait_on_get_lanes_of_their_own_first():
    """The rest's lane that starts behind them on the same box takes less work."""
    minutes = {"slow": 9.0, "a": 4.0, "b": 4.0, "c": 1.0, "first1": 0.5, "first2": 0.5}
    lanes = heldout_fleet.plan_lanes(minutes, ["first2", "first1"], lane_minutes=10, boxes=2)
    assert lanes[0] == ["first1", "first2"]
    load = lambda lane: sum(minutes[n] for n in lane)  # noqa: E731
    assert len(lanes) == 3 and load(lanes[2]) < load(lanes[1])  # lanes[2] starts behind


def test_a_named_clip_list_comes_inline_or_from_a_file(tmp_path):
    names = tmp_path / "lane.clips"
    names.write_text("a\nb\n")
    assert aws_heldout_chart.clip_names("a,b") == ["a", "b"]
    assert aws_heldout_chart.clip_names(f"@{names}") == ["a", "b"]


def fleet(tmp_path, lanes):
    args = argparse.Namespace(state=str(tmp_path / "fleet.json"), boxes=2, budget=3.0,
                              instance_type="c7a.24xlarge", purpose="test", cloud_results=None,
                              options=["--clipset", "set.json"])
    return heldout_fleet.Fleet(args, [heldout_fleet.ClipWork(n, [], [], 1.0) for lane in lanes
                                      for n in lane], lanes)


def test_a_lost_session_is_relaunched_on_its_own_lane_and_an_unlaunched_one_waits(tmp_path):
    """A lost lane's clips go up again at once (only its own); a session that found no
    room exits unlaunched and is tried again after LAUNCH_RETRY_S."""
    exits = {("1", "run"): [1, 0], ("1", "score-cloud"): [0],
             ("2", "run"): [aws_heldout_chart.EXIT_NO_LAUNCH, 0], ("2", "score-cloud"): [0]}
    calls = []

    def run(command, **_):
        lane = Path(command[command.index("--clip-list") + 1][1:]).stem.split("-")[1]
        calls.append((lane, command[2], Path(command[command.index("--clip-list") + 1][1:])
                      .read_text().split()))
        return subprocess.CompletedProcess(command, exits[(lane, command[2])].pop(0))

    run_fleet = fleet(tmp_path, [["a", "b"], ["c"]])
    with patch.object(heldout_fleet.subprocess, "run", run), \
            patch.object(heldout_fleet, "LAUNCH_RETRY_S", 0):
        run_fleet.measure()
    assert [c for c in calls if c[0] == "1"] == [("1", "run", ["a", "b"])] * 2 + \
        [("1", "score-cloud", ["a", "b"])]
    assert [c for c in calls if c[0] == "2"] == [("2", "run", ["c"])] * 2 + \
        [("2", "score-cloud", ["c"])]
    lanes = {lane.index: lane for lane in run_fleet.lanes}
    assert lanes[1].relaunched == 1 and lanes[2].relaunched == 0
    assert all(lane.status == "done" for lane in run_fleet.lanes)
    assert '"status": "done"' in Path(run_fleet.args.state).read_text()


def test_a_lane_lost_too_often_fails_the_fleet_loudly(tmp_path):
    run_fleet = fleet(tmp_path, [["a"]])
    with patch.object(heldout_fleet.subprocess, "run",
                      lambda command, **_: subprocess.CompletedProcess(command, 1)), \
            pytest.raises(SystemExit, match="lanes failed: 1"):
        run_fleet.measure()
    assert run_fleet.lanes[0].relaunched == heldout_fleet.RELAUNCHES + 1


class FakeBench:
    def cached(self, point):
        return point.cached


def point(family: str, clip: str, cached: bool):
    return SimpleNamespace(family=SimpleNamespace(name=family), clip=SimpleNamespace(name=clip),
                           label="300k", cached=cached)


def test_a_cached_only_runner_refuses_to_measure_and_lists_the_missing_points():
    """The chart render measures nothing (owner, 2026-11-21: the PC carries no compute);
    only the families it may measure here (the hardware encoders) are run."""
    runner = Runner(load_paths(), EVALUATION_GATE, 1, True, cached_only=True,
                    may_measure=frozenset({"nvenc-h264"}))
    jobs = [(FakeBench(), point("x265-medium~ssim", "clipA", False)),
            (FakeBench(), point("x264-medium", "clipA", True)),
            (FakeBench(), point("nvenc-h264", "clipA", False))]
    with pytest.raises(MissingPoints, match="1 points are not in the store") as refused:
        runner.run_jobs(jobs)
    assert "x265-medium~ssim clipA 300k" in str(refused.value)
    measured = []
    runner._measure_one = lambda job: measured.append(job[1].family.name)
    runner.run_jobs(jobs[1:])
    assert measured == ["x264-medium", "nvenc-h264"]
