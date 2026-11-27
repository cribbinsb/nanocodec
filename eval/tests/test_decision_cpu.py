"""A decision's CPU is measured on its own encodes (nceval.decision_cpu): any difference
between the commands the quality path codes and the ones the CPU path times fails here."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
from nceval import decision_cpu, levels, pool, realuse, singlepoint
from nceval.cbr import Bench, Family
from nceval.clips import Clip
from nceval.codecs import Codec
from nceval.config import Paths
from nceval.gatepolicy import EVALUATION_GATE

CLASSES = ("crowd", "static", "pan")


def clip(name, cls, frames=900, start=0):
    return Clip(name=name, yuv=Path(f"/clips/{name}.mkv"), width=1280, height=720, fps=30.0,
                frames=frames, source_kbps=20000.0, content_class=cls, split="pool",
                start_frame=start)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(Bench, "tool_hash", lambda self, family: "build")
    paths = Paths(data_root=tmp_path, anchors=tmp_path, nanocodec_bin=tmp_path)
    return realuse.Runner(paths, EVALUATION_GATE, 1, over_budget=True)


def families(code_change=False):
    anchor = Family("nanocodec-hevc", Codec.HEVC, Path("/b/old/nanocodec"),
                    ("--preset", "fast"), own=True, cqp=False)
    test = Family("nanocodec-hevc@candidate", Codec.HEVC,
                  Path("/b/new/nanocodec" if code_change else "/b/old/nanocodec"),
                  ("--preset", "fast", "--param", "a.b=1"), own=True, cqp=False)
    return anchor, test


def draw():
    made = lambda level, n, start: [clip(f"{level}{i}", CLASSES[i % 3], 900 if i % 2 else 300,
                                         start) for i in range(n)]
    return pool.Draw("2026-11-23", 150, {pool.QUICK: made("q", 6, 40),
                                         pool.ADOPT: made("a", 6, 10),
                                         pool.FINAL: made("f", 6, 0)})


def screen_args(**over):
    base = dict(frames="150", rotation="2026-11-23", max_level=3, levels=None, k=0.2,
                all_stages=False, final_estimator=levels.SINGLE, final_frames=None,
                key_shares=None, cpu=None, dcpu=None)
    return argparse.Namespace(**{**base, **over})


def measured_points(runner, monkeypatch, call, test_name):
    """The candidate's points the quality path asks the runner for while `call` runs
    (each measurement returns nothing, so the run stops early at a missing point)."""
    asked = []
    original = realuse.Runner.points

    def record(self, families, *a, **kw):
        asked.extend(j for j in original(self, families, *a, **kw)
                     if j[1].family.name == test_name)
        return []
    monkeypatch.setattr(realuse.Runner, "points", record)
    try:
        call()
    except (ValueError, KeyError, ZeroDivisionError, TypeError):
        pass
    monkeypatch.setattr(realuse.Runner, "points", original)
    return [(b.frames, b.intra, p.clip.name, p.clip.start_frame, p.kbps) for b, p in asked]


def planned_points(runner, test, chosen, args):
    return [(b.frames, b.intra, p.clip.name, p.clip.start_frame, p.kbps)
            for b, p in levels.quality_jobs(runner, test, draw(), chosen, args)]


@pytest.mark.parametrize("level,estimator", [(pool.QUICK, levels.SINGLE),
                                             (pool.FINAL, levels.SINGLE),
                                             (pool.FINAL, levels.LADDER)])
def test_the_cpu_plan_lists_the_encodes_a_level_measures(runner, monkeypatch, level, estimator):
    anchor, test = families()
    args = screen_args(final_estimator=estimator)
    clips = draw().levels[level]
    measured = measured_points(runner, monkeypatch, lambda: levels.measure_level(
        runner, (anchor, test), clips, 150, level, levels.estimator_of(level, args), args),
        test.name)
    assert measured and sorted(measured) == sorted(planned_points(runner, test, (level,), args))


@pytest.mark.parametrize("estimator", [levels.SINGLE, levels.LADDER])
def test_the_cpu_plan_lists_every_key_share_of_the_long_window(runner, monkeypatch, estimator):
    anchor, test = families()
    args = screen_args(final_estimator=estimator, final_frames=600)
    measured = []
    for share in (levels.key_shares.KeyShare(150), levels.key_shares.KeyShare(300)):
        clips = levels.long_clips(draw().levels[pool.FINAL], share.window)
        measured += measured_points(runner, monkeypatch, lambda: levels.measure_level(
            runner, (anchor, test), clips, share.window, pool.FINAL, estimator, args,
            key_frames=share.period), test.name)
    planned = planned_points(runner, test, (pool.FINAL,), args)
    assert sorted(measured) == sorted(planned)
    assert {(f, k) for f, k, *_ in planned} == {(600, 150), (900, 300)}
    assert all(start == 0 for _, _, _, start, _ in planned)  # long windows from frame 0


def test_a_cpu_job_is_the_quality_command_with_placeholders(runner):
    anchor, test = families(code_change=True)
    jobs = levels.quality_jobs(runner, test, draw(), (pool.QUICK,), screen_args())
    for bench, point in jobs:
        job = decision_cpu.job_of(bench, point)
        filled = [{decision_cpu.BIN: str(test.tool), decision_cpu.INPUT: str(point.clip.yuv),
                   decision_cpu.OUTPUT: "/out"}.get(a, a) for a in job.argv]
        assert [*filled, *test.extra] == bench.command(point, Path("/out"), None)
        assert "--bitrate" in job.argv and "--buffer-ms" in job.argv and "--qp" not in job.argv
        assert (job.kbps, job.frames, job.start_frame) == (point.kbps, 150, point.clip.start_frame)


def plan(runner, args=None, code_change=False):
    anchor, test = families(code_change)
    args = args or screen_args()
    jobs = levels.quality_jobs(runner, test, draw(), (pool.QUICK, pool.FINAL), args)
    return decision_cpu.plan_entry("screen", "hevc", "fast", "2026-11-23", jobs, anchor, test,
                                   "HEAD", "HEAD" if not code_change else "HEAD~1")


def test_the_subset_is_one_clip_per_class_and_group_from_the_decisions_own_jobs(runner):
    entry = plan(runner)
    jobs = [decision_cpu.CpuJob.from_record(j) for j in entry["jobs"]]
    groups = {j.group for j in jobs}
    assert groups == {decision_cpu.ONE_KEY}  # L1 and the 150-frame L3: one key
    for group in groups:
        classes = [j.content_class for j in jobs if j.group == group]
        assert sorted(classes) == sorted(set(classes))
    quality = {(b.frames, b.intra, p.clip.name, p.kbps) for b, p in levels.quality_jobs(
        runner, families()[1], draw(), (pool.QUICK, pool.FINAL), screen_args())}
    assert {(j.frames, j.intra, j.clip, j.kbps) for j in jobs} <= quality


def summary_for(entry, change=1.0, targets=("graniterapids", "graviton4", "zen5")):
    groups = sorted({decision_cpu.CpuJob.from_record(j).group for j in entry["jobs"]})
    rows = [{"pair": f"{entry['name']}@{g}", "rung": "hevc:fast", "target": t,
             "change_pct": [change, change - 0.3, change + 0.3]} for g in groups for t in targets]
    return {"rows": rows, "decisions": {entry["name"]: {"digest": entry["digest"],
                                                        "rotation": entry["rotation"],
                                                        "groups": groups}}}


def test_a_charge_is_refused_unless_it_measured_this_decisions_encodes(runner):
    entry = plan(runner)
    found = decision_cpu.charge(summary_for(entry), "screen", "s.json")
    assert decision_cpu.verify(found, entry["digest"], "t") is found
    assert found.dcpu(150, 300) == pytest.approx(1.0)
    with pytest.raises(decision_cpu.CpuMismatch):  # another group: not measured
        found.dcpu(600, 150)
    other = plan(runner, screen_args(final_estimator=levels.LADDER))
    with pytest.raises(decision_cpu.CpuMismatch):  # other encodes (the L3 ladder)
        decision_cpu.verify(found, other["digest"], "t")
    with pytest.raises(decision_cpu.CpuMismatch):  # an aws_cpu_ab run of its own clips
        decision_cpu.charge({"rows": summary_for(entry)["rows"]}, "screen", "s.json")
    with pytest.raises(decision_cpu.CpuMismatch):  # no Arm target
        decision_cpu.charge(summary_for(entry, targets=("graniterapids", "zen5")), "screen",
                            "s.json")
    assert decision_cpu.parse(found.text()).groups == pytest.approx(found.groups)


def test_the_digest_covers_both_sides_arguments(runner):
    entry = plan(runner)
    jobs = [decision_cpu.CpuJob.from_record(j) for j in entry["jobs"]]
    assert decision_cpu.digest(jobs, entry["base"]["extra"], entry["cand"]["extra"]) == \
        entry["digest"]
    assert decision_cpu.digest(jobs, entry["base"]["extra"], [*entry["cand"]["extra"], "x"]) \
        != entry["digest"]


def test_aws_cpu_ab_times_the_plans_commands_on_both_sides(runner):
    import aws_cpu_ab
    entry = plan(runner, code_change=True)
    jobs = aws_cpu_ab.decision_jobs({"decisions": [entry]})
    assert len(jobs) == len(entry["jobs"])
    for job, record in zip(jobs, entry["jobs"]):
        cpu_job = decision_cpu.CpuJob.from_record(record)
        pair, rung, _ = job["id"].split("|")
        assert pair == f"screen@{cpu_job.group}" and rung == "hevc:fast"
        for side in ("base", "cand"):
            argv = job["variants"][side]
            assert argv[0] == f"bin/{aws_cpu_ab.build_key(entry[side]['build'])}/nanocodec"
            assert argv[argv.index("--input") + 1] == \
                f"clips/{aws_cpu_ab.decision_clip_file(cpu_job)}"
            assert argv[len(cpu_job.argv):] == entry[side]["extra"]
            body = [a for a in cpu_job.argv[1:] if a not in (decision_cpu.INPUT,
                                                              decision_cpu.OUTPUT)]
            assert [a for a in argv[1:len(cpu_job.argv)]
                    if not a.startswith("clips/") and a != "/dev/null"] == body
    assert set(aws_cpu_ab.decisions_record({"decisions": [entry]})["screen"]["groups"]) == \
        {decision_cpu.CpuJob.from_record(j).group for j in entry["jobs"]}


def test_single_points_and_the_plan_share_one_rung_assignment():
    clips = [clip(f"c{i}", CLASSES[i % 3]) for i in range(5)]
    rungs = singlepoint.assign_rungs(clips)
    ladder = singlepoint.rung_ladder(rungs)
    assert [len(ladder(c)) for c in clips] == [1] * 5


def test_plan_files_round_trip(runner, tmp_path):
    entry = plan(runner)
    out = tmp_path / "plan.json"
    decision_cpu.write_plan(out, [entry])
    assert json.loads(out.read_text())["decisions"][0]["digest"] == entry["digest"]
    assert replace(decision_cpu.CpuJob.from_record(entry["jobs"][0])).record() == \
        entry["jobs"][0]


def test_steering_counts_the_trials_own_single_points(runner, monkeypatch, tmp_path):
    """The tuner's callgrind steering (nceval.steering_cpu) runs a declared subset of the
    very commands its trials' single points code, with each setting's own arguments."""
    from nceval import steering_cpu
    anchor, test = families()
    clips = draw().levels[pool.ADOPT]
    rungs = singlepoint.assign_rungs(clips)
    jobs = steering_cpu.jobs_of(runner, anchor, clips, rungs, 150)
    trial = measured_points(runner, monkeypatch, lambda: singlepoint.single_points(
        runner, anchor, test, clips, rungs, 150), test.name)
    assert {(j.frames, j.intra, j.clip, j.start_frame, j.kbps) for j in jobs} <= set(trial)
    ran = []

    class Served:
        def __init__(self, clip, frames, scratch):
            self.clip = clip

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def reader(self):
            return f"/served/{self.clip.name}@{self.clip.start_frame}"
    monkeypatch.setattr(steering_cpu, "ClipFrames", Served)
    monkeypatch.setattr(steering_cpu, "run", lambda argv: ran.append(argv))
    monkeypatch.setattr(steering_cpu, "_callgrind_total", lambda path: 1000)
    monkeypatch.setattr(steering_cpu, "load_paths", lambda: runner.paths)
    counts = steering_cpu.instructions(test.tool, test.extra, jobs, {c.name: c for c in clips})
    assert len(counts) == len(jobs) and steering_cpu.ratio(counts, counts) == 1.0
    for argv in ran:
        assert argv[:2] == ["valgrind", "--tool=callgrind"]
        command = argv[3:]
        assert command[0] == str(test.tool) and command[-len(test.extra):] == list(test.extra)
        assert "--bitrate" in command and "--qp" not in command
