"""tools/dev/aws_cpu_ab.py: CPU A/B jobs, staging, one box on a fake session that runs
the box's commands here (nothing is rented), and the summary."""
import contextlib
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_cpu_ab  # noqa: E402
import cpu_ab_box  # noqa: E402
import cpu_ab_shard  # noqa: E402
import jperf  # noqa: E402

WIDTH, HEIGHT = 16, 8
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A data root with a two-clip set and a fake encoder (prints its version, counts
    nothing: a script, so the box runner measures only its CPU and wall time)."""
    root = tmp_path / "data"
    (root / "clips/quick").mkdir(parents=True)
    clips = []
    for name in ("a", "b"):
        (root / f"clips/quick/{name}.yuv").write_bytes(bytes(range(256)) * (FRAME_BYTES * 4 // 256))
        clips.append({"name": name, "yuv": f"clips/quick/{name}.yuv", "width": WIDTH,
                      "height": HEIGHT, "fps": 30.0, "frames": 4})
    (root / "clips/quick.json").write_text(json.dumps({"name": "quick", "clips": clips}))
    monkeypatch.setattr(aws_cpu_ab, "DATA", root)
    monkeypatch.setattr(aws_cpu_ab, "CLIPSET", root / "clips/quick.json")
    monkeypatch.setattr(aws_cpu_ab, "CLIP_CACHE", root / "work/cpu-ab/clips")
    encoder = tmp_path / "nanocodec"
    encoder.write_text("#!/bin/sh\n[ \"$1\" = --version ] && echo 'nanocodec 0.0 abc'\nexit 0\n")
    encoder.chmod(0o755)
    return root, encoder


def pair(encoder: Path, frames: int = 2) -> aws_cpu_ab.Pair:
    return aws_cpu_ab.Pair("lever", aws_cpu_ab.Side(str(encoder)),
                           aws_cpu_ab.Side(str(encoder), ("a.b=1",)), ("hevc:fast", "av1:faster"),
                           ("a", "b"), frames, ("--bitrate", "600"))


def test_jobs_name_the_box_paths_and_only_the_candidate_takes_its_params(data):
    _, encoder = data
    jobs = aws_cpu_ab.jobs_for([pair(encoder)], aws_cpu_ab.load_clips({"a", "b"}))
    assert [j["id"] for j in jobs] == ["lever|hevc:fast|a", "lever|hevc:fast|b",
                                       "lever|av1:faster|a", "lever|av1:faster|b"]
    base, cand = jobs[0]["variants"]["base"], jobs[0]["variants"]["cand"]
    key = aws_cpu_ab.build_key(str(encoder))
    assert base[0] == cand[0] == f"bin/{key}/nanocodec"
    assert base[base.index("--input") + 1] == "clips/a-2f.yuv"
    assert cand[len(base):] == ["--param", "a.b=1"] and "--param" not in base


def test_shards_group_the_jobs_by_clip_and_drop_empty_boxes():
    jobs = [{"id": str(i), "variants": {"base": ["bin", "--input", f"clips/{'ab'[i % 2]}.yuv"]}}
            for i in range(5)]
    parts = aws_cpu_ab.shards(jobs, 2)
    assert [[j["id"] for j in p] for p in parts] == [["0", "2", "4"], ["1", "3"]]
    assert len(aws_cpu_ab.shards(jobs[:1], 3)) == 1


def test_a_binary_of_the_wrong_architecture_is_refused(tmp_path):
    arm = tmp_path / "arm"
    arm.write_bytes(b"\x7fELF" + bytes(14) + (0xB7).to_bytes(2, "little") + bytes(8))
    assert aws_cpu_ab.build_for(str(arm), "aarch64") == arm.resolve()
    with pytest.raises(SystemExit, match="built for aarch64"):
        aws_cpu_ab.build_for(str(arm), "x86_64")


def test_the_round_order_alternates():
    assert cpu_ab_box.round_order(["base", "cand"], 0) == ["base", "cand"]
    assert cpu_ab_box.round_order(["base", "cand"], 1) == ["cand", "base"]


class LocalSession:
    """A session whose box is a directory here: its commands run in bash with HOME there."""

    def __init__(self, spec, home: Path):
        self.spec, self.home, self.commands = spec, home, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.commands.append("terminated")

    def upload(self, local_dir, remote_dir):
        shutil.copytree(local_dir, self.home / remote_dir.removeprefix("~/"), symlinks=False)

    def run_detached(self, command, log, minutes, **_):
        self.commands.append(command)
        subprocess.run(["bash", "-c", command], check=True, env={**os.environ, "HOME": str(self.home)},
                       stdout=subprocess.DEVNULL)

    def download(self, remote_dir, local_dir):
        shutil.copytree(self.home / remote_dir.removeprefix("~/"), local_dir, dirs_exist_ok=True)

    @contextlib.contextmanager
    def pulling(self, remote_dir, local_dir, marker):
        self.commands.append(f"pulling {remote_dir}")
        try:
            yield
        finally:
            self.download(remote_dir, local_dir)


def test_one_box_runs_readiness_the_pipe_cleaner_and_the_rounds(data, tmp_path):
    _, encoder = data
    clips = aws_cpu_ab.load_clips({"a", "b"})
    jobs = aws_cpu_ab.jobs_for([pair(encoder)], clips)
    stage = tmp_path / "stage"
    frames = pair(encoder).frames
    files = {aws_cpu_ab.clip_file(n, frames): aws_cpu_ab.cached_clip(clips[n], frames)
             for n in clips}
    aws_cpu_ab.stage(stage, jobs, {aws_cpu_ab.build_key(str(encoder)): encoder}, files)
    sessions = []
    out = tmp_path / "out" / "graviton4-1"
    # vCPU 0 here: the build machines' CPU 1 may be offline
    target = dataclasses.replace(cpu_ab_shard.TARGETS["graviton4"], timed_cpu=0)
    cpu_ab_shard.run_shard(stage, target, out, 3, False, "test",
                           lambda spec: sessions.append(LocalSession(spec, tmp_path / "box"))
                           or sessions[-1])
    session = sessions[0]
    assert session.spec.instance_type == "c8g.2xlarge" and not session.spec.spill
    assert session.commands[-1] == "terminated"
    assert "--limit 1" in session.commands[1] and session.commands[2].startswith("pulling")
    assert "--rounds 3" in session.commands[3] and "--skip done.json" in session.commands[3]
    # the clip went up cut to the pair's frames, compressed, and was unpacked there
    assert (tmp_path / "box/cpuab/clips/a-2f.yuv").stat().st_size == 2 * FRAME_BYTES
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == len(jobs) * 3 * 2
    assert {r["variant"] for r in runs} == {"base", "cand"}
    assert json.loads((out / "session.json").read_text())["jobs"] == len(jobs)
    assert "nanocodec 0.0 abc" in (out / "versions.txt").read_text()


def fake_runs(path: Path, ratios: dict[str, list[float]], counted: bool = True) -> None:
    """runs.jsonl for one rung with the given cand/base cycle ratio per clip and round."""
    path.mkdir(parents=True)
    lines = []
    for clip, per_round in ratios.items():
        for index, ratio in enumerate(per_round):
            for variant, cycles in (("base", 1e9), ("cand", 1e9 * ratio)):
                counts = {"cycles": int(cycles), "instructions": int(2 * cycles)} if counted \
                    else {"cycles": None, "instructions": None}
                lines.append(json.dumps({"job": f"lever|hevc:fast|{clip}", "variant": variant,
                                         "round": index, "counts": counts, "user_s": cycles / 1e9,
                                         "system_s": 0.0, "wall_s": cycles / 1e9}))
    (path / "runs.jsonl").write_text("\n".join(lines) + "\n")


def test_the_summary_averages_clips_with_an_interval_from_the_rounds(tmp_path):
    fake_runs(tmp_path / "graviton2-1", {"a": [0.98, 0.99, 0.97], "b": [1.0, 1.01, 0.99]})
    fake_runs(tmp_path / "zen5-1", {"a": [0.98, 0.99, 0.97]}, counted=False)
    rows = {r["target"]: r for r in aws_cpu_ab.summarize(tmp_path)["rows"]}
    arm = rows["graviton2"]
    centre, low, high = arm["change_pct"]
    expected = 100 * math.expm1(sum(map(math.log, (0.98, 0.99, 0.97, 1.0, 1.01, 0.99))) / 6)
    assert arm["metric"] == "cycles" and centre == pytest.approx(expected, abs=1e-4)
    assert low < centre < high and arm["instructions_pct"][0] == pytest.approx(centre, abs=0.01)
    assert arm["per_clip_pct"]["a"] == pytest.approx(-2.0, abs=0.01)
    assert arm["round_sd_pct"] == pytest.approx(1.0, abs=0.02)
    # no counters: the encode's CPU time, and no instruction change
    assert rows["zen5"]["metric"] == "user_s" and rows["zen5"]["instructions_pct"] is None


def test_the_t_quantile_falls_to_the_normal():
    assert aws_cpu_ab.t975(4) == 2.776 and aws_cpu_ab.t975(11) == 2.228
    assert aws_cpu_ab.t975(500) == aws_cpu_ab.NORMAL_975 and aws_cpu_ab.t975(0) == math.inf


def test_the_probe_keeps_the_events_that_count_together(monkeypatch):
    """A PMU with two counters: a third event multiplexes them all, so it is dropped."""
    def fake_measure(argv, core, events):
        counted = len(events) <= 2
        return jperf.Run({n: 1 if counted else None for n in events}, 1.0, 0.0, 1.0)
    monkeypatch.setattr(cpu_ab_box, "measure", fake_measure)
    events = cpu_ab_box.EVENTS["aarch64"]
    probe = cpu_ab_box.probe_events(["enc"], 1, events)
    assert probe == {"base": ["cycles", "instructions"], "extras": [], "slots": 0}
    assert cpu_ab_box.schedule(probe, 3) == ["cycles", "instructions"]


def test_the_probe_encodes_a_few_frames_only():
    argv = ["nanocodec", "encode", "--frames", "900", "--start-frame", "30"]
    assert cpu_ab_box.short_encode(argv) == ["nanocodec", "encode", "--frames",
                                             str(cpu_ab_box.PROBE_FRAMES), "--start-frame", "30"]
    assert cpu_ab_box.short_encode(["enc"]) == ["enc"]


def test_the_extra_events_take_the_free_counters_in_turn():
    probe = {"base": ["cycles", "instructions"], "extras": ["x", "y", "z"], "slots": 2}
    assert [cpu_ab_box.schedule(probe, r)[2:] for r in range(3)] == [["x", "y"], ["z", "x"],
                                                                      ["y", "z"]]


def test_aarch64_builds_cross_link_and_x86_builds_do_not():
    assert "aarch64-linux-gnu-gcc" in aws_cpu_ab.build_command("aarch64")
    assert "--target aarch64-unknown-linux-gnu" in aws_cpu_ab.build_command("aarch64")
    assert "LINKER" not in aws_cpu_ab.build_command("x86_64")


def test_a_plan_entry_takes_its_own_rate_and_clips(tmp_path):
    plan = tmp_path / "pairs.json"
    plan.write_text(json.dumps([{"name": "p", "base": {"build": "B"},
                                 "cand": {"build": "B", "params": ["x=1"]},
                                 "rungs": ["vp9:fast"], "frames": 90, "qp": 32}]))
    args = type("Args", (), {"plan": plan, "qp": None, "bitrate": 600, "clips": "a,b",
                             "frames": 150})()
    (pair,) = aws_cpu_ab.pairs_from_args(args)
    assert pair.rate == ("--qp", "32") and pair.frames == 90 and pair.clips == ("a", "b")
    assert pair.cand.argv_tail() == ["--param", "x=1"] and pair.base.argv_tail() == []


def test_a_held_build_lock_is_reported_and_given_up_on(tmp_path, monkeypatch, capsys):
    import fcntl
    import pytest
    path = tmp_path / "aarch64.lock"
    path.write_text("pid 1 building abc since 12:30:00\n")
    other = path.open("a+")
    fcntl.flock(other, fcntl.LOCK_EX)
    monkeypatch.setattr(aws_cpu_ab, "PROGRESS_SECONDS", 0)
    monkeypatch.setattr(aws_cpu_ab.time, "sleep", lambda s: None)
    with path.open("a+") as lock, pytest.raises(SystemExit, match="pid 1 building abc"):
        aws_cpu_ab.wait_for_lock(lock, "the aarch64 build lock", limit_s=0.05)
    assert "waiting for the aarch64 build lock (pid 1 building abc" in capsys.readouterr().out


def test_a_stage_that_does_not_end_is_killed_loudly():
    import pytest
    with pytest.raises(SystemExit, match="no end after"):
        aws_cpu_ab.run_watched(["sleep", "30"], "building x", limit_s=0.5)
    with pytest.raises(SystemExit, match="failed \\(exit 3\\)"):
        aws_cpu_ab.run_watched(["sh", "-c", "exit 3"], "building x", limit_s=10)


def test_builds_of_different_commits_run_side_by_side_and_one_commit_builds_once(
        tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(aws_cpu_ab, "BUILD_CACHE", tmp_path)
    (tmp_path / "x86_64").mkdir()
    made, lock = [], threading.Lock()

    def fake_build(commit, machine, name, binary):
        with lock:
            made.append((commit, name))
        time.sleep(0.5)
        binary.write_text("elf")
    monkeypatch.setattr(aws_cpu_ab, "build_on_box", fake_build)
    build = lambda commits: list(ThreadPoolExecutor(4).map(
        lambda c: aws_cpu_ab.release_build(c, "x86_64"), commits))
    started = time.monotonic()
    build(["a" * 40, "b" * 40, "c" * 40])
    assert time.monotonic() - started < 1.4  # three builds at once, not 1.5 s in a row
    assert len({name for _, name in made}) == 3
    made.clear()
    paths = build(["d" * 40, "d" * 40])  # one commit asked twice at once: one build
    assert len(made) == 1 and paths[0] == paths[1]


def test_prune_keeps_the_newest_builds_and_recent_clips_and_runs(tmp_path, monkeypatch):
    import os
    import time
    monkeypatch.setattr(aws_cpu_ab, "WORK", tmp_path)
    monkeypatch.setattr(aws_cpu_ab, "BUILD_CACHE", tmp_path / "builds")
    monkeypatch.setattr(aws_cpu_ab, "CLIP_CACHE", tmp_path / "clips")
    monkeypatch.setattr(aws_cpu_ab, "KEEP_BUILDS", 2)
    now = time.time()
    for age in range(4):  # four x86 builds, used 0..3 hours ago
        home = tmp_path / "builds/x86_64" / f"c{age}"
        home.mkdir(parents=True)
        (home / "nanocodec").write_bytes(b"e" * 10)
        (home / aws_cpu_ab.COMMIT_MARKER).write_text("c\n")
        os.utime(home / aws_cpu_ab.COMMIT_MARKER, (now - 3600 * age,) * 2)
    (tmp_path / "clips").mkdir()
    for name, days in (("old.zst", 5), ("new.zst", 1)):
        (tmp_path / "clips" / name).write_bytes(b"z")
        os.utime(tmp_path / "clips" / name, (now - 86400 * days,) * 2)
    aws_cpu_ab.prune(now)
    assert sorted(p.name for p in (tmp_path / "builds/x86_64").iterdir()) == ["c0", "c1"]
    assert [p.name for p in (tmp_path / "clips").iterdir()] == ["new.zst"]


def test_key_shares_measure_each_pair_at_every_share_and_print_the_screens_dcpu(data):
    _, encoder = data
    pairs = aws_cpu_ab.at_key_shares([pair(encoder)], "150,300")
    assert [(p.name, p.frames, p.intra_period) for p in pairs] == [
        ("lever@k150", 600, 150), ("lever@k300", 900, 300)]
    job = aws_cpu_ab.jobs_for(pairs[1:], aws_cpu_ab.load_clips({"a", "b"}))[0]
    argv = job["variants"]["base"]
    assert argv[argv.index("--intra-period") + 1] == "300" and "--bitrate" in argv  # CBR
    row = lambda name, change: {"pair": name, "rung": "hevc:fast", "target": "zen5",
                                "change_pct": (change, change - 1, change + 1)}
    summary = {"rows": [row("lever@k150", 3.0), row("lever@k300", 1.5), row("plain", 9.0)]}
    assert aws_cpu_ab.share_dcpu(summary) == {("lever", "hevc:fast", "zen5"): "150=3.00,300=1.50"}
