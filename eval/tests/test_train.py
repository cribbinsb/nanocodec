"""The candidate queue and its checks (nceval.train), with a fake measurement."""
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import levers, levers_ideas, train
from nceval.config import Paths

CLASSES = ("static", "lowlight", "crowd", "pan")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv(train.QUEUE_ENV, str(tmp_path / "queue.jsonl"))
    monkeypatch.setenv(levers.SPOOL_ENV, str(tmp_path / "spool.jsonl"))
    monkeypatch.setattr(levers, "LEDGER", tmp_path / "ledger.jsonl")
    paths = Paths(tmp_path / "data", tmp_path / "anchors", tmp_path / "bin", tmp_path / "work")
    monkeypatch.setattr(train, "load_paths", lambda: paths)
    monkeypatch.setattr(train, "plan_minutes", lambda arms, base, cloud: "  (plan)")
    presets = tmp_path / "presets"
    (presets / "h264").mkdir(parents=True)
    (presets / "h264" / "fast.toml").write_text("# fast\nquant.trellis = 0\na.x = 1  # why\n")
    for name in ("idea-a", "idea-b", "idea-c"):
        levers.append([levers_ideas.make_idea(idea=name, name=name, outcome="held")])
    base = train.Base(tmp_path / "nanocodec", "abc123", presets, "r1")
    return tmp_path, base


def quick_report(tmp_path, name, params, l1=-0.5, classes=None, pulse=0.0, verdict="CONTINUE"):
    report = {"status": "done", "codec": "h264", "preset": "fast", "params": params,
              "verdict": verdict, "reasons": [], "rotation": "r1",
              "l1": {"mean": l1, "lo": l1 - 0.5, "hi": l1 + 0.5, "n": 24, "quality": l1,
                     "classes": classes or {}},
              "probe": {"mean": l1, "lo": l1 - 1, "hi": l1 + 1, "n": 6, "quality": l1,
                        "classes": classes or {}, "pulse": pulse, "pulse_max": pulse}}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report))
    return path


def queue(tmp_path, idea, params, mechanism="subpel", **stage1):
    pairs = [f"{k}={v}" for k, v in params.items()]
    args = argparse.Namespace(codec="h264", preset="fast", param=pairs, idea=idea,
                              mechanism=mechanism, dcpu=-1.0, dcpu_jetson=None, force=False,
                              quick_json=str(quick_report(tmp_path, idea, params, **stage1)))
    return train.add(args)


def level(decision, mean=-0.5, guards=()):
    return {"decision": decision, "mean": mean, "lo": mean - 0.3, "hi": mean + 0.3, "n": 8,
            "confidence": 0.9, "guards": list(guards)}


def result(arm, decision, per_clip, guards=()):
    rows = {f"clip{i}": {"class": CLASSES[i % 4], "quality": j, "quality_low": j}
            for i, j in enumerate(per_clip)}
    return {"mode": "real", "codec": arm.codec, "binary": "/nonexistent/nanocodec",
            "anchor_binary": "/nonexistent/nanocodec", "params": train.checkrun.pairs(arm.params),
            "anchor_params": [], "draw": {"rotation": "r1", "frames": 150},
            "levels": {"final": level(decision, sum(per_clip) / len(per_clip), guards)},
            "stages": [{"stage": "final", "per_clip": rows, "summary": {}, "verdict": {}}]}


def fake(scenario):
    """measure(): `scenario(member ids)` gives (L3 decision, final-check decision,
    per-clip J, guards) of each arm."""
    calls = []

    def measure(arms, base):
        calls.append([sorted(c.id for c in arm.members) for arm in arms])
        out = {}
        for arm in arms:
            l3, fc, clips, guards = scenario({c.idea for c in arm.members})
            out[arm.key] = {train.L3: result(arm, l3, clips), "cost": 0.1,
                            train.FINAL_CHECK: result(arm, fc, clips, guards)}
        return out
    measure.calls = calls
    return measure


def test_add_and_list_show_stage1_numbers(world, capsys):
    tmp_path, _ = world
    candidate = queue(tmp_path, "idea-a", {"a.x": 2.0}, l1=-0.7)
    [loaded] = train.load_queue()
    assert loaded.id == candidate.id and loaded.status == train.QUEUED
    assert loaded.stage1_j(0.6) == pytest.approx(-0.7 - 0.6)
    train.listing(argparse.Namespace(all=False, codec=None))
    out = capsys.readouterr().out
    assert "h264@fast" in out and "L1 J -0.70%" in out and candidate.id in out


def test_add_refuses_a_kill_an_unknown_idea_and_a_mismatched_report(world):
    tmp_path, _ = world
    args = argparse.Namespace(codec="h264", preset="fast", param=["a.x=2"], idea="idea-a",
                              mechanism="m", dcpu=None, dcpu_jetson=None, force=False,
                              quick_json=str(quick_report(tmp_path, "k", {"a.x": 2.0},
                                                          verdict="KILL")))
    with pytest.raises(SystemExit, match="KILL"):
        train.add(args)
    args.quick_json = str(quick_report(tmp_path, "ok", {"a.x": 2.0}))
    with pytest.raises(SystemExit, match="no idea"):
        train.add(argparse.Namespace(**{**vars(args), "idea": "nope"}))
    with pytest.raises(SystemExit, match="not this candidate"):
        train.add(argparse.Namespace(**{**vars(args), "param": ["a.x=3"]}))


def test_mechanisms_are_independent_unless_they_share_a_parameter(world):
    tmp_path, _ = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0}, mechanism="subpel")
    b = queue(tmp_path, "idea-b", {"b.y": 1.0}, mechanism="partition")
    c = queue(tmp_path, "idea-c", {"c.z": 1.0}, mechanism="subpel")
    groups = {tuple(sorted(x.id for x in g)) for g in train.units([a, b, c])}
    assert groups == {tuple(sorted((a.id, c.id))), (b.id,)}
    d = train.Candidate(**{**vars(b), "id": "d", "mechanism": "other", "params": {"a.x": 3.0}})
    joined = {tuple(sorted(x.id for x in g)) for g in train.units([a, b, d])}
    assert joined == {tuple(sorted((a.id, "d"))), (b.id,)}  # d sets a's parameter


def test_candidates_setting_the_same_parameter_are_alternatives_not_a_bundle(world):
    tmp_path, _ = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0}, l1=-0.2)
    b = queue(tmp_path, "idea-b", {"a.x": 2.0, "b.y": 2.0}, l1=-0.9)  # a's gate + more
    c = queue(tmp_path, "idea-c", {"c.z": 1.0}, l1=-0.1)
    members, alternatives = train.compose([a, b, c], 0.6)
    assert [m.id for m in members] == [b.id, c.id] and [x.id for x in alternatives] == [a.id]


def test_alternatives_run_alone_and_only_the_best_net_j_passes(world, capsys):
    tmp_path, base = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0})
    b = queue(tmp_path, "idea-b", {"a.x": 2.0, "b.y": 2.0})

    def scenario(ideas):
        assert len(ideas) == 1, "alternatives must never be bundled"
        return ("better", "better", [-0.4] * 4 if "idea-a" in ideas else [-0.8] * 4, [])
    measure = fake(scenario)
    outcomes = train.run_units(train.prepare([a, b], cloud=True), base, measure, cloud=True,
                               rounds=3)
    status = {c.id: c.status for c in train.load_queue()}
    assert status == {a.id: train.QUEUED, b.id: train.PASSED}
    assert [o["passed"] is not None for o in outcomes] == [False, True]
    assert "ALTERNATIVE NOT CHOSEN" in capsys.readouterr().out


def test_the_frozen_preset_rewrites_in_place_and_appends(world):
    _, base = world
    text = (base.presets / "h264" / "fast.toml").read_text()
    out = train.write_preset(text, {"a.x": 1.3333334, "new.p": 2.0}, "note")
    assert "a.x = 1.3333334  # why" in out and out.rstrip().endswith("new.p = 2")
    assert "quant.trellis = 0" in out


def test_a_passing_unit_is_marked_recorded_and_printed(world, capsys):
    tmp_path, base = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0})
    measure = fake(lambda ideas: ("better", "ambiguous", [-0.5, -0.4, -0.6, -0.3], []))
    outcome = train.run_unit([a], base, measure, cloud=True, rounds=3)
    assert outcome["passed"]
    assert train.load_queue()[0].status == train.PASSED
    out = capsys.readouterr().out
    assert "+a.x = 2" in out and "h264 fast: a.x" in out and "idea-a (subpel)" in out
    frozen = Path(outcome["passed"]["directory"])
    assert "a.x = 2" in (frozen / "presets/h264/fast.toml").read_text()
    idea = next(i for i in levers_ideas.load_ideas() if i["idea"] == "idea-a")
    assert idea["rows"] and "PASS" in idea["reason"]
    assert {r["level"] for r in levers.load()} == {"L3"}


def test_local_failure_ejects_the_member_matching_the_failing_class(world, capsys):
    tmp_path, base = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0}, classes={"lowlight": 0.2})
    b = queue(tmp_path, "idea-b", {"b.y": 1.0}, classes={"lowlight": 4.0})
    guard = "class lowlight quality +3.50 > +3.00"
    measure = fake(lambda ideas: ("worse", "worse", [0.5, 0.4, 0.6, 0.3], [guard])
                   if "idea-b" in ideas else ("better", "better", [-0.5] * 4, []))
    outcome = train.run_unit([a, b], base, measure, cloud=False, rounds=3)
    status = {c.id: c.status for c in train.load_queue()}
    assert status == {a.id: train.PASSED, b.id: train.EJECTED}
    assert len(measure.calls) == 2 and outcome["passed"]
    rejected = next(i for i in levers_ideas.load_ideas() if i["idea"] == "idea-b")
    assert rejected["outcome"] == "rejected" and "lowlight" in rejected["reason"]


def test_an_ambiguous_failure_bisects_and_defers_the_weaker_half(world, capsys):
    tmp_path, base = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0}, l1=-0.9)
    b = queue(tmp_path, "idea-b", {"b.y": 1.0}, l1=-0.8)
    measure = fake(lambda ideas: ("ambiguous", "ambiguous", [0.1] * 4, [])
                   if len(ideas) > 1 else ("better", "better", [-0.5] * 4, []))
    outcome = train.run_unit([a, b], base, measure, cloud=False, rounds=3)
    status = {c.id: c.status for c in train.load_queue()}
    assert status == {a.id: train.PASSED, b.id: train.QUEUED}  # b stays for the next run
    assert outcome["deferred"] == [b.id]


def test_cloud_round_ejects_by_leave_one_out_and_lands_the_verified_arm(world, capsys):
    tmp_path, base = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0})
    b = queue(tmp_path, "idea-b", {"b.y": 1.0})

    def scenario(ideas):
        if ideas == {"idea-a", "idea-b"}:
            return "ambiguous", "worse", [0.6, 0.9, 0.7, 1.0, 0.8, 0.75, 0.65, 0.85], []
        if ideas == {"idea-a"}:  # leaving b out: b's marginal is clearly worse
            return "better", "better", [-0.5, -0.4, -0.6, -0.3, -0.45, -0.55, -0.5, -0.4], []
        return "ambiguous", "ambiguous", [0.7, 1.0, 0.8, 1.1, 0.9, 0.85, 0.75, 0.95], []
    measure = fake(scenario)
    outcome = train.run_unit([a, b], base, measure, cloud=True, rounds=3)
    assert len(measure.calls) == 1 and len(measure.calls[0]) == 3  # one round, 3 arms
    status = {c.id: c.status for c in train.load_queue()}
    assert status == {a.id: train.PASSED, b.id: train.EJECTED}
    assert outcome["passed"]["arm"].endswith(f"loo:{b.id} [{a.id}]")


def test_screen_commands_run_any_preset_as_params(world):
    tmp_path, base = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0})
    [arm] = train.arms_of([a], cloud=True)
    train.freeze(arm, base.presets, tmp_path / "frozen")
    fc = train.screen_command(arm, base, train.FINAL_CHECK, tmp_path / "fc.json")
    l3 = train.screen_command(arm, base, train.L3, tmp_path / "l3.json")
    assert fc[fc.index("--levels") + 1] == "final" and "--final-frames" in fc
    assert "--dcpu" not in fc and l3[l3.index("--dcpu") + 1] == "-1.0000"
    assert "a.x=2" in l3 and "quant.trellis=0" in l3 and "a.x=1" in l3  # test and anchor


def test_a_rounds_screens_are_packed_and_one_targets_anchor_counted_once(world):
    tmp_path, _ = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0})
    b = queue(tmp_path, "idea-b", {"b.y": 1.0})
    arms = train.arms_of([a, b], cloud=True)  # full + two leave-one-out
    sizes = {train.FINAL_CHECK: (400, 100), train.L3: (300, 100)}
    boxes = train.pack(arms, sizes)
    first = boxes[0]
    assert first.points == 400 + 300 + 200 * (len(first.screens) // 2)
    assert sum(len(b.screens) for b in boxes) == 6
    assert all(train.box_minutes(b.points) <= train.PACK_MINUTES or len(b.screens) == 2
               for b in boxes)


def test_the_long_window_l3_is_a_final_check_row_not_a_duplicate_l3():
    assert levers.level_name("final", 600) == "final-check"
    assert levers.level_name("final", 150) == "L3" and levers.level_name("quick", 150) == "L1"


def test_a_failed_box_fails_only_its_units_after_one_retry(world, monkeypatch, capsys):
    tmp_path, base = world
    a = queue(tmp_path, "idea-a", {"a.x": 2.0}, mechanism="m1")
    b = queue(tmp_path, "idea-b", {"b.y": 1.0}, mechanism="m2")
    c = queue(tmp_path, "idea-c", {"c.z": 1.0}, mechanism="m3")
    units_ = [u for g in train.units([a, b, c]) for u in train.prepare(g, cloud=True)]
    arms = [arm for u in units_ for arm in u.arms]
    for arm in arms:
        train.freeze(arm, base.presets, tmp_path / "frozen")
    monkeypatch.setattr(train, "pack", lambda arms_, sizes: [
        train.Box(screens=[(arm, check) for check in (train.FINAL_CHECK, train.L3)])
        for arm in arms_])
    monkeypatch.setattr(train, "screen_sizes", lambda base_: {})
    runs = []

    def run_logged(command, log, limit):
        batch = json.loads(Path(command[command.index("--batch") + 1]).read_text())
        runs.append(log.name)
        for screen in batch:
            out = Path(screen[screen.index("--json") + 1])
            arm = next(x for x in arms if x.directory == out.parent)
            if b in arm.members:
                raise RuntimeError("FAILED (exit 1): rsync to the warm box failed")
            out.write_text(json.dumps({**result(arm, "better", [-0.5] * 4), "status": "done",
                                       "cloud": {"estimated_usd": 0.1}}))
    monkeypatch.setattr(train, "run_logged", run_logged)
    monkeypatch.setattr(train, "plan_minutes", lambda *_: "")
    measure = lambda arms_, base_: train.measure_arms(arms_, base_, True, 4.0, 8)
    outcomes = train.run_units(units_, base, measure, cloud=True, rounds=3)
    judged = {o["members"][0]: o["judged"] for o in outcomes}
    assert judged == {a.id: True, b.id: False, c.id: True}
    status = {x.id: x.status for x in train.load_queue()}
    assert status == {a.id: train.PASSED, b.id: train.QUEUED, c.id: train.PASSED}
    assert sum(1 for name in runs if "retry" in name) == 1  # the failed box, once more
