"""tools/dev/aws_measure.py: one nceval command on a fake session (nothing is rented)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_measure  # noqa: E402
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
    monkeypatch.setattr(aws_measure, "archive_tree", lambda path: path.write_bytes(b"tar"))
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

    def download_new(self, remote, local, marker):
        point = local / "f150-ip300/a/nanocodec-h264/cbr300-0123.json"
        point.parent.mkdir(parents=True, exist_ok=True)
        point.write_text(json.dumps({"psnr_yuv": 40.0, "ms_ssim": 0.95, "kbps": 300.0}))
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
                             "instance_type": "c7a.48xlarge", "lifetime": 60, "budget": 4.0,
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
    assert result["binary"] == str(binary) and result["cloud"]["instance_type"] == "c7a.48xlarge"
    assert json.loads(out.read_text())["decision"] == "better"
    # only the run's clips, the clip set files, libvmaf, the build: never the local cache
    assert "nanocodec-data/clips/main/a.yuv" in session.uploaded
    assert "nanocodec-data/clips/main.json" in session.uploaded
    assert "arms/0/nanocodec" in session.uploaded and not any(
        "realuse" in p for p in session.uploaded)
    assert session.spec.fallback_types[0] == "c7a.24xlarge"
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
    assert aws_measure.auto_types(20)[0] == "c7a.8xlarge"  # too few for any: the smallest
    assert aws_measure.auto_types(170)[0] == "c7a.16xlarge"  # a final check: 64 vCPU
    assert aws_measure.auto_types(540)[0] == "c7a.24xlarge"  # an L3: 96
    assert aws_measure.auto_types(5000)[:2] == ("c7a.24xlarge", "c7i.24xlarge")
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
