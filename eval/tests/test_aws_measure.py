"""tools/dev/aws_measure.py: one nceval command on a fake session (nothing is rented)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_measure  # noqa: E402
import aws_session  # noqa: E402
from nceval import levers  # noqa: E402
from nceval.clips import Clip  # noqa: E402

SCREEN = ["python3", "-m", "nceval.screen", "--codec", "h264"]


def test_the_command_keeps_its_options_and_drops_the_workers():
    command = aws_measure.Command.parse(["--", *SCREEN, "--param", "a=1", "--workers", "8",
                                         "--json=out.json"])
    assert (command.module, command.sub) == ("nceval.screen", None)
    assert command.get("--json") == "out.json" and command.get("--workers") is None
    command.set("--json", "/remote/out.json")
    assert command.argv() == [*SCREEN, "--param", "a=1", "--json", "/remote/out.json"]


def test_a_command_it_cannot_run_is_refused():
    with pytest.raises(SystemExit, match="not supported"):
        aws_measure.Command.parse(["python3", "-m", "nceval.sweep", "--workers", "8"])
    with pytest.raises(SystemExit, match="nceval.MODULE"):
        aws_measure.Command.parse(["nceval.screen"])
    sub = aws_measure.Command.parse(["python3", "-m", "nceval.levels", "final-check", "--old",
                                     "a"])
    assert sub.sub == "final-check" and sub.get("--old") == "a"


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A data root with two clips, a clip set file, libvmaf and its model, and a build."""
    root = tmp_path / "data"
    for name in ("clips/main/a.yuv", "clips/pool/b.mp4"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"\0" * 96)
    (root / "clips/main.json").write_text('{"name": "main", "clips": []}')
    model = root / "anchors/x86_64" / aws_measure.VMAF_V1_MODEL
    model.parent.mkdir(parents=True)
    model.write_text("{}")
    (root / "anchors/x86_64/bin").mkdir()
    (root / "anchors/x86_64/bin/vmaf").write_text("#!/bin/sh\n")
    for tool in aws_measure.BOX_DECODERS:  # every box carries the anchor decoders
        (root / "anchors/x86_64/bin" / tool).write_text("#!/bin/sh\n")
    build = tmp_path / "build"
    (build / "presets").mkdir(parents=True)
    binary = build / "nanocodec"
    binary.write_text("#!/bin/sh\necho nanocodec 0.5.0 0123abcd\n")
    binary.chmod(0o755)
    monkeypatch.setenv("NC_DATA_ROOT", str(root))
    monkeypatch.setenv("NC_ANCHORS", str(root / "anchors/x86_64"))
    monkeypatch.setenv("NC_WORK", str(root / "work"))
    monkeypatch.setattr(aws_measure, "DATA", root)
    monkeypatch.setattr(aws_measure, "CLOUD_WORK", root / "work/cloud-measure")
    monkeypatch.setattr(aws_measure, "STAGE_ROOT", root / "work/stage")
    monkeypatch.setattr(aws_measure, "archive_tree", lambda path, commit=None: path.write_bytes(b"tar"))
    clips = [Clip("a", root / "clips/main/a.yuv", 64, 48, 30.0, 150, 900.0, "static", "train"),
             Clip("b", root / "clips/pool/b.mp4", 64, 48, 30.0, 150, 900.0, "pan", "train")]
    monkeypatch.setattr(aws_measure, "command_clips", lambda command, paths, rotation: clips)
    monkeypatch.setattr(aws_measure, "local_decodes", lambda run: {"b": "f" * 64})
    return root, binary


def test_the_plan_moves_every_local_path_to_the_box(data, tmp_path):
    root, binary = data
    out = tmp_path / "out.json"
    run = aws_measure.plan(aws_measure.Command.parse(
        [*SCREEN, "--binary", str(binary), "--param", "a=1", "--json", str(out)]))
    argv = run.command.argv()
    assert str(tmp_path) not in " ".join(argv)
    assert run.command.get("--binary") == f"{aws_measure.REMOTE_HOME}/arms/0/nanocodec"
    assert run.command.get("--json") == aws_measure.REMOTE_JSON
    assert run.command.get("--rotation")  # pinned: a run across midnight keeps its draw
    assert run.arms[0].presets == binary.parent / "presets"
    assert aws_measure.localise(json.dumps(argv), run).count(str(binary)) == 1


def test_a_command_without_json_is_refused(data):
    _, binary = data
    with pytest.raises(SystemExit, match="--json"):
        aws_measure.plan(aws_measure.Command.parse([*SCREEN, "--binary", str(binary)]))


class FakeSession:
    """Records what a session is asked to do; the run leaves one point, a result naming
    the box's paths and one lever row."""

    def __init__(self, spec, decoded: str = "f" * 64, fail_run: bool = False):
        self.spec, self.decoded, self.fail_run = spec, decoded, fail_run
        self.commands: list[str] = []
        self.uploaded: list[str] = []
        self.launched, self.zone, self.price = None, "eu-north-1a", 0.5

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.commands.append("terminated")

    def cost(self, price, hours):
        return price * hours

    def upload(self, local_dir, remote_dir):
        self.uploaded = sorted(str(p.relative_to(local_dir)) for p in local_dir.rglob("*")
                               if p.is_file())

    def run_detached(self, command, log, minutes, **_):
        self.commands.append(command)
        if self.fail_run and "-m nceval.screen" in command:
            raise RuntimeError("remote run failed")

    # the real pull loop over the fake's downloads
    pulling = aws_session.Session.pulling

    def spot_notice(self):
        return False

    def download_new(self, remote, local, marker, names=None):
        point = local / "f150-ip300/a/nanocodec-h264/cbr300-0123.json"
        point.parent.mkdir(parents=True, exist_ok=True)
        point.write_text(json.dumps({"psnr_yuv": 40.0, "ms_ssim": 0.95, "kbps": 300.0}))
        if names is not None:
            names.append(point)
        return 1

    def ssh(self, command, check=True, quiet=False):
        if command.startswith("cat") and "prepare.log" in command:
            out = f"{self.decoded}  b\n"
        elif aws_measure.REMOTE_JSON in command and not self.fail_run:
            out = json.dumps({"status": "done", "binary": f"{aws_measure.REMOTE_HOME}/arms/0/"
                              "nanocodec", "decision": "better"})
        elif aws_measure.REMOTE_SPOOL in command and not self.fail_run:
            out = json.dumps(levers.make_row(codec="h264", base="medium", rate="cbr",
                                             overrides={"a": 1.0}, level="L1", j=-1.0,
                                             source=aws_measure.REMOTE_JSON)) + "\n"
        else:
            out = ""
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0 if out else 1, out, "")


def measure_args(command: list[str], **extra):
    return type("Args", (), {"command": command, "dry_run": False, "region": "eu-north-1",
                             "instance_type": "c7i.12xlarge", "lifetime": 60, "budget": 4.0,
                             "no_fallback": False, "run_minutes": 35.0, **extra})()


def test_a_run_comes_back_localised_and_leaves_nothing_behind(data, tmp_path):
    root, binary = data
    out = tmp_path / "out.json"
    local_point = root / "work/realuse/f150-ip300/a/nanocodec-h264/cbr300-0123.json"
    local_point.parent.mkdir(parents=True)
    local_point.write_text(json.dumps({"psnr_yuv": 40.0, "ms_ssim": 0.9500004, "kbps": 300.0}))
    sessions = []
    result = aws_measure.measure(
        measure_args([*SCREEN, "--binary", str(binary), "--json", str(out)]),
        lambda spec: sessions.append(FakeSession(spec)) or sessions[-1])
    session = sessions[0]
    assert result["binary"] == str(binary) and result["cloud"]["instance_type"] == "c7i.12xlarge"
    assert json.loads(out.read_text())["decision"] == "better"
    # only the run's clips, the clip set files, libvmaf, the build: never the local cache
    assert "nanocodec-data/clips/main/a.yuv" in session.uploaded
    assert "nanocodec-data/clips/main.json" in session.uploaded
    # every anchor decoder goes up, and the job's readiness fails fast without one
    assert "nanocodec-data/anchors/x86_64/bin/dav1d" in session.uploaded
    assert any("missing anchor decoder dav1d" in c for c in session.commands)
    assert "arms/0/nanocodec" in session.uploaded and not any(
        "realuse" in p for p in session.uploaded)
    assert session.spec.fallback_types[0] == "c7i.24xlarge"
    run = next(c for c in session.commands if "-m nceval.screen" in c)
    assert "NC_METRICS_BACKEND=libvmaf-cpu" in run and "--workers" in run
    # the clips are deleted on the box before it is terminated
    assert session.commands.index(aws_measure.cleanup_command()) < \
        session.commands.index("terminated")
    # the cloud point is kept apart from the local cache, which is untouched
    kept = aws_measure.CLOUD_WORK / "realuse/f150-ip300/a/nanocodec-h264/cbr300-0123.json"
    assert json.loads(kept.read_text())["ms_ssim"] == 0.95
    assert json.loads(local_point.read_text())["ms_ssim"] == 0.9500004
    rows = levers.read_rows(levers.spool_path())
    assert rows[0]["source"].startswith(str(out)) and "aws" in rows[0]["source"]
    assert not any(aws_measure.STAGE_ROOT.iterdir())  # staging and downloads removed


def test_a_decode_that_differs_on_the_box_stops_the_run(data, tmp_path):
    _, binary = data
    sessions = []
    with pytest.raises(SystemExit, match="decodes it differently"):
        aws_measure.measure(
            measure_args([*SCREEN, "--binary", str(binary), "--json", str(tmp_path / "o")]),
            lambda spec: sessions.append(FakeSession(spec, decoded="0" * 64)) or sessions[-1])
    assert not any("-m nceval.screen" in c for c in sessions[0].commands)
    assert aws_measure.cleanup_command() in sessions[0].commands


def test_a_failed_run_still_brings_back_its_points_and_cleans_the_box(data, tmp_path):
    _, binary = data
    sessions = []
    with pytest.raises(RuntimeError, match="remote run failed"):
        aws_measure.measure(
            measure_args([*SCREEN, "--binary", str(binary), "--json", str(tmp_path / "o")]),
            lambda spec: sessions.append(FakeSession(spec, fail_run=True)) or sessions[-1])
    assert aws_measure.cleanup_command() in sessions[0].commands
    assert not any(aws_measure.STAGE_ROOT.iterdir())


def test_parity_counts_the_fields_that_differ(tmp_path):
    cloud, local = tmp_path / "cloud", tmp_path / "local"
    for root, ms_ssim in ((cloud, 0.95), (local, 0.9500008)):
        point = root / "f150-ip300/a/fam/cbr300-1.json"
        point.parent.mkdir(parents=True)
        point.write_text(json.dumps({"psnr_yuv": 40.0, "ms_ssim": ms_ssim,
                                     "frame_vmaf_v1": [90.0, 91.0]}))
    report = aws_measure.parity(cloud, local)
    assert (report["compared"], report["identical"]) == (1, 0)
    assert set(report["fields"]) == {"ms_ssim"}
    assert report["fields"]["ms_ssim"]["max_abs"] == pytest.approx(8e-7)


def test_a_reclaimed_box_is_replaced_once(data, tmp_path):
    _, binary = data
    sessions = []

    class Reclaimed(FakeSession):
        def run_detached(self, command, log, minutes, **_):
            if "-m nceval.screen" in command:
                raise aws_measure.RemoteFailure("the instance is terminated: Spot", [])

    def factory(spec):
        sessions.append((Reclaimed if not sessions else FakeSession)(spec))
        return sessions[-1]
    result = aws_measure.measure(
        measure_args([*SCREEN, "--binary", str(binary), "--json", str(tmp_path / "o")]),
        factory)
    assert len(sessions) == 2 and result["decision"] == "better"


def test_the_run_command_ends_on_its_status_for_the_watch(tmp_path):
    """aws_session appends `; echo $? > LOG.status`: the run must leave that reachable."""
    command = aws_measure.Command("nceval.screen", None, [])
    line = aws_measure.run_command(type("Run", (), {"command": command, "layout": aws_measure.Layout()})())
    line = line.replace(f"cd {aws_measure.REMOTE_REPO}/eval", f"cd {tmp_path}").replace(
        "python3 -m nceval.screen", "false")
    done = subprocess.run(["bash", "-c", f"{line}; echo status $?"], capture_output=True,
                          text=True)
    assert "box CPUs busy" in done.stdout and done.stdout.strip().endswith("status 1")


def test_auto_sizes_the_box_to_the_points_and_never_past_96_vcpus():
    assert aws_measure.auto_types(20)[0] == "c7i.8xlarge"  # too few for any: the smallest
    assert aws_measure.auto_types(170)[0] == "c7i.16xlarge"  # a final check: 64 vCPU
    assert aws_measure.auto_types(540)[0] == "c7i.24xlarge"  # an L3: 96
    assert aws_measure.auto_types(5000)[:2] == ("c7i.24xlarge", "c6i.24xlarge")
    # the guide's pair only (docs/aws-cost-guide.md): no c7a, no m family
    assert all(t.split(".")[0] in ("c7i", "c6i") for n in (20, 170, 540, 5000)
               for t in aws_measure.auto_types(n))
    assert all("48xl" not in t for n in (20, 170, 5000) for t in aws_measure.auto_types(n))


def test_a_batch_runs_its_commands_side_by_side_and_ends_on_their_status():
    runs = [aws_measure.Plan(aws_measure.Command("nceval.screen", None, ["--json", f"/o/{i}"]),
                             [], []) for i in range(2)]
    command = aws_measure.run_command(runs[0], runs)
    assert command.count("& p") == 2 and "wait $p0 || status=1; wait $p1 || status=1;" in command
    assert command.rstrip().endswith("(exit $status)")
    assert "/eval && { " in command  # every command runs in the tree, not just the first


class HalfBatchSession(FakeSession):
    """A batch whose first command finished and whose second failed."""

    def run_detached(self, command, log, minutes, **_):
        self.commands.append(command)
        if "-m nceval.screen" in command:
            raise aws_measure.RemoteFailure("exit status 1", ["Traceback ..."])

    def ssh(self, command, check=True, quiet=False):
        if "result-0.json" in command:
            self.commands.append(command)
            out = json.dumps({"status": "done", "decision": "better"})
            return subprocess.CompletedProcess(command, 0, out, "")
        if "result-1.json" in command:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 1, "", "")
        return super().ssh(command, check, quiet)


def test_a_batch_brings_back_the_results_of_the_commands_that_succeeded(data, tmp_path):
    _, binary = data
    outs = [tmp_path / "a.json", tmp_path / "b.json"]
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([[*SCREEN, "--binary", str(binary), "--json", str(o)]
                                 for o in outs]))
    with pytest.raises(SystemExit, match="1 of 2 command"):
        aws_measure.measure(measure_args([], batch=str(batch)),
                            lambda spec: HalfBatchSession(spec))
    assert json.loads(outs[0].read_text())["decision"] == "better" and not outs[1].exists()


def test_finished_jobs_staging_is_swept_and_running_jobs_kept(tmp_path):
    import os
    dead = tmp_path / "stage-20261121-105121-4194999"  # above pid_max: no such process
    gone = tmp_path / "back-20261121-105121-4194999"
    running = tmp_path / f"stage-20261121-150000-{os.getpid()}"
    for path in (dead, gone, running):
        (path / "jobs").mkdir(parents=True)
        (path / "jobs" / "repo.tar").write_bytes(b"x" * 1000)
    (dead / "clip.yuv").symlink_to(tmp_path / "elsewhere.yuv")  # a link, not its target
    (tmp_path / "elsewhere.yuv").write_bytes(b"y" * 10)
    assert aws_measure.sweep_staging(tmp_path) == 2000
    assert not dead.exists() and not gone.exists() and running.exists()
    assert (tmp_path / "elsewhere.yuv").exists()


class CappedSession(FakeSession):
    """A run that finishes points while it runs (one a pull: the box writes them as it
    goes) and is then stopped at the run cap; the box is cleaned after it."""

    def __init__(self, spec):
        super().__init__(spec)
        self.pulls: list[str] = []

    def run_detached(self, command, log, minutes, **_):
        self.commands.append(command)
        if "-m nceval.screen" in command:
            time.sleep(0.3)  # several pulls while it runs
            raise aws_measure.RemoteFailure(f"no result in {minutes} min", ["level 1 ..."])

    def download_new(self, remote, local, marker, names=None):
        self.pulls.append(marker)
        self.commands.append("pull")
        if len(self.pulls) > 3:  # one failed pull is logged, never raised
            raise RuntimeError("checkpoint download failed")
        point = local / f"f150-ip300/a/nanocodec-h264/cbr300-{len(self.pulls):04d}.json"
        point.parent.mkdir(parents=True, exist_ok=True)
        point.write_text(json.dumps({"psnr_yuv": 40.0, "ms_ssim": 0.95, "kbps": 300.0}))
        if names is not None:
            names.append(point)
        return 1


def test_a_run_stopped_at_the_cap_keeps_every_point_it_pulled_back(data, tmp_path,
                                                                     monkeypatch):
    _, binary = data
    monkeypatch.setattr(aws_measure, "PULL_SECONDS", 0.05)
    sessions = []
    with pytest.raises(SystemExit, match="stopped at the run cap .*partial results in"):
        aws_measure.measure(
            measure_args([*SCREEN, "--binary", str(binary), "--json", str(tmp_path / "o")]),
            lambda spec: sessions.append(CappedSession(spec)) or sessions[-1])
    session = sessions[0]
    kept = sorted(p.name for p in (aws_measure.CLOUD_WORK / "realuse").rglob("*.json"))
    assert kept == ["cbr300-0001.json", "cbr300-0002.json", "cbr300-0003.json"]
    # pulls while it ran, the last before the box was cleaned and terminated
    last_pull = len(session.commands) - 1 - session.commands[::-1].index("pull")
    assert len(session.pulls) >= 3
    assert last_pull < session.commands.index(aws_measure.cleanup_command()) \
        < session.commands.index("terminated")
    assert not any(aws_measure.STAGE_ROOT.iterdir())


def test_a_capped_failure_is_told_apart_from_a_failed_command():
    assert aws_measure.capped("no result in 35.0 min\n  remote log tail:")
    assert not aws_measure.capped("exit status 1") and not aws_measure.capped(None)


def test_arm_boxes_only_for_our_encoder_with_clean_commits(monkeypatch):
    """--arch auto: aarch64 when the job allows it, x86_64 otherwise; --arch aarch64 fails loudly."""
    run = type("R", (), {"command": aws_measure.Command.parse(["--", *SCREEN])})()
    gap = type("R", (), {"command": aws_measure.Command.parse(
        ["--", "python3", "-m", "nceval.fixedqp", "gap"])})()
    assert aws_measure.choose_machine(aws_measure.X86, [gap], []) == aws_measure.X86
    assert aws_measure.choose_machine("auto", [gap], []) == aws_measure.X86
    with pytest.raises(SystemExit, match="reference encoders"):
        aws_measure.choose_machine(aws_measure.ARM, [gap], [])
    monkeypatch.setattr(aws_measure, "arm_reason", lambda runs, arms: None)
    assert aws_measure.choose_machine("auto", [run], []) == aws_measure.ARM
    assert aws_measure.auto_types(5000, aws_measure.AUTO_SIZES_ARM)[0] == "c8g.16xlarge"
    assert aws_measure.auto_types(20, aws_measure.AUTO_SIZES_ARM)[0] == "c8g.8xlarge"


def test_only_a_clean_commit_names_an_aarch64_build(monkeypatch):
    commit = "d51e02c3f1cf90bd527e296497d568940b1d6daf"
    for version, expected in ((f"nanocodec 0.6.0 {commit}", commit),
                              (f"nanocodec 0.6.0 {commit}-dirty.ab12", None),
                              ("nanocodec 0.6.0 unknown", None)):
        monkeypatch.setattr(aws_measure, "version_of", lambda binary, v=version: v)
        assert aws_measure.commit_of(Path("nanocodec")) == expected


def test_auto_refuses_a_dirty_build_instead_of_a_quiet_x86_box(tmp_path, monkeypatch):
    """2026-11-25: a dirty candidate silently took a 48-vCPU x86 box in place of a 96-vCPU
    Graviton; --arch auto now refuses it, and only --arch x86_64 takes it to x86."""
    arm_anchors = tmp_path / aws_measure.ARM
    for tool in ("vmaf", *aws_measure.BOX_DECODERS):
        (arm_anchors / "bin" / tool).parent.mkdir(parents=True, exist_ok=True)
        (arm_anchors / "bin" / tool).write_text("")
    (arm_anchors / aws_measure.VMAF_V1_MODEL).parent.mkdir(parents=True, exist_ok=True)
    (arm_anchors / aws_measure.VMAF_V1_MODEL).write_text("")
    monkeypatch.setenv("NC_ANCHORS", str(tmp_path / aws_measure.X86))
    monkeypatch.setattr(aws_measure, "version_of", lambda binary: "nanocodec 0.6.0 unknown")
    run = type("R", (), {"command": aws_measure.Command.parse(["--", *SCREEN])})()
    dirty = type("A", (), {"binary": Path("nanocodec")})()
    with pytest.raises(SystemExit, match="binary_for.sh"):
        aws_measure.choose_machine("auto", [run], [dirty])
    assert aws_measure.choose_machine(aws_measure.X86, [run], [dirty]) == aws_measure.X86


def test_a_retry_is_seeded_with_the_points_pulled_from_the_lost_box(tmp_path, monkeypatch):
    cache = tmp_path / "cloud"
    monkeypatch.setattr(aws_measure, "CLOUD_WORK", cache)
    old = cache / aws_measure.WORK_SUBDIR / "f150-ip300" / "clip_a"
    old.mkdir(parents=True)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.txt").write_text("")
    assert aws_measure.reseed(stage, {"clip_a"}) == 1  # staged before: linked once
    # the lost box's pulls made a new window for clip_a and points of another clip
    (cache / aws_measure.WORK_SUBDIR / "f900-ip300" / "clip_a").mkdir(parents=True)
    (cache / aws_measure.WORK_SUBDIR / "f900-ip300" / "clip_b").mkdir(parents=True)
    assert aws_measure.reseed(stage, {"clip_a"}) == 1
    assert (stage / "seed/f900-ip300/clip_a").resolve() == \
        (cache / aws_measure.WORK_SUBDIR / "f900-ip300" / "clip_a").resolve()
    assert not (stage / "seed/f900-ip300/clip_b").exists()
    assert len((stage / "manifest.txt").read_text().splitlines()) == 2


def test_a_reclaimed_warm_box_s_retry_re_runs_only_the_missing_points(tmp_path, monkeypatch):
    """The warm box is reclaimed after finishing clip_a's points (pulled back while it ran);
    the job's next warm box gets them as its seed and its prepare copies them into its cache
    without overwriting its own, so only the in-flight points are coded again."""
    cache = tmp_path / "cloud"
    monkeypatch.setattr(aws_measure, "CLOUD_WORK", cache)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.txt").write_text("")
    runs = [type("R", (), {"clips": [type("C", (), {"name": "clip_a"})()],
                           "remote_json": "r.json"})()]
    seen = []

    def warm_session(args, stage_dir, *rest):
        seen.append(sorted(p.name for p in (stage_dir / "seed").rglob("*")))
        if len(seen) == 1:  # the pulls brought clip_a back, then the reclaim
            (cache / aws_measure.WORK_SUBDIR / "f150-ip300" / "clip_a").mkdir(parents=True)
            raise aws_measure.RemoteFailure("the instance is terminated: "
                                            "Server.SpotInstanceTermination", [])
        return {"stage_minutes": {}}
    monkeypatch.setattr(aws_measure, "warm_session", warm_session)
    cloud = aws_measure.rent(None, True, aws_measure.X86, (), 0, None, stage, tmp_path, [], {},
                             runs, aws_measure.warm_layout("job"), {})
    assert cloud == {"stage_minutes": {}}
    assert seen == [[], ["clip_a", "f150-ip300"]]
    [step] = aws_measure.seed_steps(warm=True)  # copied in without overwriting the box's own
    assert "cp -rn" in step and step.startswith(f"if [ -d {aws_measure.REMOTE_SEED} ]")


def test_a_dirty_tree_is_refused_at_once_and_a_pinned_commit_is_archived(tmp_path, monkeypatch):
    """nceval.cloud.pinned_tree refuses uncommitted eval/, tools/ or presets before
    anything is rented; a job's pinned commit is what the box gets, whatever the tree."""
    from nceval import cloud
    repo = tmp_path / "repo"
    (repo / "eval").mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,  # noqa: E731
                                    capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (repo / "eval" / "a.py").write_text("1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")
    monkeypatch.setattr(cloud, "REPO", repo)
    pinned = cloud.pinned_tree()
    assert pinned == git("rev-parse", "HEAD")
    (repo / "eval" / "a.py").write_text("2\n")
    with pytest.raises(SystemExit, match="commit eval/"):
        cloud.pinned_tree()
    monkeypatch.setattr(aws_measure, "REPO", repo)
    aws_measure.archive_tree(tmp_path / "t.tar", pinned)  # the dirty tree does not matter
    listed = subprocess.run(["tar", "-xOf", str(tmp_path / "t.tar"), "eval/a.py"],
                            capture_output=True, text=True, check=True).stdout
    assert listed == "1\n"


def test_a_seed_directory_pruned_meanwhile_is_left_out_not_fatal(tmp_path):
    """2026-11-24: a staging job died on a seed directory the cache's prune had removed."""
    live = tmp_path / "live"
    live.mkdir()
    assert aws_measure.claim_seed(live) and time.time() - live.stat().st_mtime < 5
    assert not aws_measure.claim_seed(tmp_path / "gone")


def test_fixedqp_ab_may_go_to_arm_but_gap_stays_x86():
    """2026-11-24 parity on c8g.8xlarge and c6i.24xlarge: the same fixedqp ab, same streams,
    J within 1.0e-5 pp ($0.009 against $0.026); gap codes reference encoders (fixedqp cost
    does not run through aws_measure at all)."""
    parse = lambda *argv: aws_measure.Command.parse(["--", "python3", "-m", *argv])  # noqa: E731
    assert aws_measure.arm_command(parse("nceval.fixedqp", "ab", "--codec", "h264"))
    assert not aws_measure.arm_command(parse("nceval.fixedqp", "gap"))
    assert aws_measure.arm_command(parse("nceval.screen", "--codec", "av1"))


def l3_screen(i):
    return ["python3", "-m", "nceval.screen", "--codec", "av1", "--clips", "adopt", "--levels",
            "final", "--final-estimator", "ladder", "--final-frames", "600", "--json", f"o{i}"]


def fake_runs(commands, points, monkeypatch):
    """Plans of `commands` whose job_points are `points` each (no clips needed)."""
    runs = [aws_measure.Plan(aws_measure.Command.parse(c), [], []) for c in commands]
    monkeypatch.setattr(aws_measure, "job_points",
                        lambda rs: sum(points for _ in rs))
    return runs


def test_a_batch_over_the_run_cap_is_split_across_boxes_never_launched_to_time_out(
        monkeypatch):
    """2026-11-24: 8 L3-ladder screens (3,856 planned points) on one c8g.12xlarge hit the
    35-min run cap with none of their results back. Planned first, they go to sub-batches
    that each fit on the largest Arm box."""
    commands = [l3_screen(i) for i in range(8)]
    runs = fake_runs(commands, 482, monkeypatch)
    assert aws_measure.run_minutes(runs, 48) > 35  # the incident, planned
    launched = []
    monkeypatch.setattr(aws_measure, "run_sub_batches",
                        lambda args, groups, factory: launched.append(groups) or {"results": []})
    args = argparse.Namespace(run_minutes=35.0, instance_type="auto")
    result, _ = aws_measure.fit_to_run_cap(args, runs, commands, ("c8g.12xlarge",),
                                           aws_measure.ARM, None)
    assert result == {"results": []} and len(launched) == 1
    groups = launched[0]
    assert sorted(c[-1] for g in groups for c in g) == sorted(c[-1] for c in commands)
    limit = 35.0 * aws_measure.RUN_FILL
    for group in groups:
        planned = aws_measure.run_minutes(
            fake_runs(group, 482, monkeypatch), max(v for v, _ in aws_measure.AUTO_SIZES_ARM))
        assert planned <= limit


def test_a_command_that_cannot_fit_alone_is_refused_with_nothing_launched(monkeypatch):
    commands = [l3_screen(0), l3_screen(1)]
    runs = fake_runs(commands, 4000, monkeypatch)  # one alone is ~62 min on 64 vCPU
    args = argparse.Namespace(run_minutes=35.0, instance_type="auto")
    with pytest.raises(SystemExit, match="Nothing launched"):
        aws_measure.fit_to_run_cap(args, runs, commands, ("c8g.12xlarge",), aws_measure.ARM,
                                   None)


def test_a_run_that_fits_keeps_its_box_or_asks_for_the_largest(monkeypatch):
    quick = [["python3", "-m", "nceval.quick", "--codec", "h264", "--json", "q"]]
    runs = fake_runs(quick, 158, monkeypatch)
    args = argparse.Namespace(run_minutes=35.0, instance_type="auto")
    assert aws_measure.fit_to_run_cap(args, runs, quick, ("c8g.8xlarge",), aws_measure.ARM,
                                      None) == (None, ("c8g.8xlarge",))
    big = [l3_screen(0), l3_screen(1)]
    runs = fake_runs(big, 482, monkeypatch)  # ~32 min on 32 vCPU, ~15 on 64
    result, types = aws_measure.fit_to_run_cap(args, runs, big, ("c8g.8xlarge",),
                                               aws_measure.ARM, None)
    assert result is None and types == max(aws_measure.AUTO_SIZES_ARM)[1]


def test_a_final_check_alone_is_planned_at_its_own_floor_not_refused(monkeypatch):
    final = [["python3", "-m", "nceval.levels", "final-check", "--old", "a", "--new", "b",
              "--json", "f"]]
    runs = fake_runs(final, 482, monkeypatch)
    assert aws_measure.run_minutes(runs, 48) == aws_measure.COMMAND_FLOOR_MINUTES["final"]


def test_a_phase_over_its_budget_is_named_loudly(capsys):
    """Owner, 2026-11-25: inefficiencies are found by the tools, not by reading narration."""
    budgets = aws_measure.phase_budgets(0)
    over = aws_measure.report_phases({"launch+setup": 420.0, "sync": 60.0}, budgets)
    assert over == ["launch+setup"]
    out = capsys.readouterr()
    text = out.out + out.err
    assert "OVER BUDGET: launch+setup took 420 s" in text and "sync" not in text.split("OVER")[1]


def test_a_box_from_the_baked_image_skips_the_package_install():
    import measure_image
    first = aws_measure.box_setup()[0]
    assert first == measure_image.install_unless_baked()
    assert f"{measure_image.MARKER_DIR}/{measure_image.setup_hash()}" in first
    assert "apt-get" in first  # a stock image still installs


def test_a_phase_without_a_budget_never_divides_by_zero():
    assert aws_measure.report_phases({"PC scoring and waits": 48.0},
                                     {"PC scoring and waits": 0.0}) == []


def test_stage_links_under_another_link_are_skipped_and_never_leave_the_stage(tmp_path):
    """2026-11-25: a batch staged a clip set's directory and one of its files; creating the
    file's link followed the directory link into the data root and failed."""
    data = tmp_path / "data" / "set"
    data.mkdir(parents=True)
    (data / "a.yuv").write_bytes(b"x")
    other = tmp_path / "other.bin"
    other.write_bytes(b"y")
    stage = tmp_path / "stage"
    links = {stage / "clips" / "set": data, stage / "clips" / "set" / "a.yuv": data / "a.yuv",
             stage / "bin" / "tool": other}
    aws_measure.make_links(stage, links)
    aws_measure.make_links(stage, links)  # idempotent
    assert (stage / "clips" / "set" / "a.yuv").read_bytes() == b"x"
    assert sorted(p.name for p in data.iterdir()) == ["a.yuv"]  # nothing written into data
    assert (stage / "bin" / "tool").read_bytes() == b"y"
