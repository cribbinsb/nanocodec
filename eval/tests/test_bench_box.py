"""tools/bench/box.py: the per-encode checks that need no root or timing box."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
sys.path.insert(0, str(REPO / "eval"))
import box  # noqa: E402
import pinning  # noqa: E402

GOOD = {"steal": 0.0, "wall_s": 1.0, "cpu_s": 0.99, "major_faults": 0, "working_threads": 3,
        "cpu": 1, "cpus_allowed": ["1"]}


def test_a_clean_encode_passes_the_timing_checks():
    box.check_timing(GOOD)


@pytest.mark.parametrize("change", [{"steal": 0.01}, {"wall_s": 1.2}, {"major_faults": 1},
                                    {"cpus_allowed": ["0-1"]}])
def test_each_timing_check_refuses(change):
    with pytest.raises(box.Refusal):
        box.check_timing({**GOOD, **change})


def test_placeholders_are_filled_from_the_box_paths():
    job = {"argv": ["{BIN}/x264", "--input", "{INPUT}", "-o", "{OUTPUT}", "{CFG}/a.cfg"]}
    paths = {"BIN": Path("/b"), "CFG": Path("/c"), "INPUT": Path("/i.yuv"), "OUTPUT": Path("/o")}
    assert box.concrete(job, paths) == ["/b/x264", "--input", "/i.yuv", "-o", "/o", "/c/a.cfg"]


def test_the_sampler_sees_a_second_working_thread():
    code = ("import threading\ndef spin():\n    x = 0\n    for i in range(3_000_000): x += i\n"
            "t = threading.Thread(target=spin); t.start(); spin(); t.join()")
    proc = subprocess.Popen([sys.executable, "-c", code])
    sampler = box.ThreadSampler(proc.pid, 0)
    sampler.start()
    proc.wait()
    sampler.done.set()
    sampler.join()
    assert sampler.working() >= 2


def test_a_file_off_tmpfs_is_refused(tmp_path):
    clip = tmp_path / "c.yuv"
    clip.write_bytes(b"\0" * 64)
    if box.mount_type(clip) == "tmpfs":
        pytest.skip("the test directory is on tmpfs")
    with pytest.raises(box.Refusal, match="V5"):
        box.check_resident(clip)


def test_cpu_lists_parse():
    assert pinning.cpus("1,3") == [1, 3] and pinning.cpus("0-2") == [0, 1, 2]


def test_residency_counts_whole_pages(tmp_path, monkeypatch):
    clip = tmp_path / "c.yuv"
    clip.write_bytes(b"\0" * 5000)  # not a page multiple: fincore reports 8192
    monkeypatch.setattr(box, "mount_type", lambda path: "tmpfs")
    monkeypatch.setattr(box.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": "8192\n"})())
    box.check_resident(clip)


def test_a_queue_box_serves_units_and_stops_on_a_spot_notice(tmp_path, monkeypatch):
    """BP-3.11: units from the inbox, references every REFERENCE_EVERY jobs and before the box
    stops; on a notice it takes no more work but references the jobs it coded."""
    import json
    import types
    coded = []

    def fake_encode(job, cpu, group, box_, out, checked=True):
        coded.append(job["id"])
        name = job["id"].replace("|", "_")
        (box_["streams"] / name).write_bytes(b"s")
        return {"id": job["id"], "stream": f"streams/{name}", "stream_sha256": "h",
                "major_faults": 0, "cpu_s": 1.0, "wall_s": 1.0}

    monkeypatch.setattr(box, "encode_job", fake_encode)
    monkeypatch.setattr(box, "REFERENCE_EVERY", 3)
    notices = iter([None] * 5 + ["terminate"] * 100)
    monkeypatch.setattr(box, "spot_notice", lambda: next(notices))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    job = lambda i: {"id": f"j|{i}", "argv": ["enc"]}  # noqa: E731
    (inbox / "unit-00000.json").write_text(json.dumps([job(i) for i in range(4)]))
    (inbox / "unit-00001.json").write_text(json.dumps([job(i) for i in range(4, 8)]))
    a = types.SimpleNamespace(cpu=1, out=tmp_path / "out", inbox=inbox, discard_streams=False)
    streams = tmp_path / "out" / "streams"
    streams.mkdir(parents=True)
    log_path = tmp_path / "out" / "encodes.jsonl"
    with open(log_path, "w") as log:
        box.Server(a, {"streams": streams}, tmp_path, log, [],
                   [{"id": "ref", "family": "x264-medium", "argv": ["ref"]}]).serve()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    jobs_done = [r["id"] for r in records if r.get("id", "").startswith("j|") and "seq" in r]
    assert jobs_done == [f"j|{i}" for i in range(5)]  # the notice came before the sixth job
    refs = [r["seq"] for r in records if "reference" in r]
    assert refs == [0, 3, 5]  # at the start, every 3 jobs, and before stopping
    assert any("interrupted" in r for r in records)
    assert (inbox / "unit-00001.taken").exists()  # taken: its jobs go back via the driver


def test_the_packing_test_loads_only_idle_cores(monkeypatch):
    """The background load never shares the timing CPU's or the harness's physical core."""
    import pinning
    topology = {0: "0,4", 1: "1,5", 2: "2,6", 3: "3,7", 4: "0,4", 5: "1,5", 6: "2,6", 7: "3,7"}
    monkeypatch.setattr(pinning, "siblings", lambda cpu: set(pinning.cpus(topology[cpu])))
    monkeypatch.setattr(pinning.Path, "read_text", lambda self: "0-7")
    assert pinning.idle_core_cpus(1, 0) == [2, 3, 6, 7]


def test_references_follow_a_long_encode_and_a_driver_stop(tmp_path, monkeypatch):
    """BP-3.11: a reference pair after REFERENCE_MAX_S of encode time (here one long encode),
    and a driver STOP closes the box like a spot notice: references, then exit."""
    import json
    import types

    def fake_encode(job, cpu, group, box_, out, checked=True):
        name = job["id"].replace("|", "_")
        (box_["streams"] / name).write_bytes(b"s")
        return {"id": job["id"], "stream": f"streams/{name}", "stream_sha256": "h",
                "major_faults": 0, "cpu_s": 1.0, "wall_s": 400.0 if job["id"] == "j|1" else 1.0}

    monkeypatch.setattr(box, "encode_job", fake_encode)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "unit-00000.json").write_text(json.dumps([{"id": f"j|{i}", "argv": ["enc"]}
                                                        for i in range(4)]))
    calls = iter(range(100))
    monkeypatch.setattr(box, "spot_notice", lambda: None)

    def stop_after_unit():
        if next(calls) >= 5:
            (inbox / "STOP").write_text("")
    a = types.SimpleNamespace(cpu=1, out=tmp_path / "out", inbox=inbox, discard_streams=False)
    streams = tmp_path / "out" / "streams"
    streams.mkdir(parents=True)
    log_path = tmp_path / "out" / "encodes.jsonl"
    server = None
    with open(log_path, "w") as log:
        server = box.Server(a, {"streams": streams}, tmp_path, log, [],
                            [{"id": "ref", "family": "x264-medium", "argv": ["ref"]}])
        original = server.check_notice

        def check():
            stop_after_unit()
            original()
        server.check_notice = check
        server.serve()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    refs = [r["seq"] for r in records if "reference" in r]
    assert refs[:2] == [0, 2]  # at the start, then right after the 400 s encode (seq 1)
    assert refs[-1] == 4 and any(r.get("interrupted") == "stopped by the driver" for r in records)


def test_the_coded_frame_list_moves_into_the_record(tmp_path):
    """T16: openh264's '+drop' sidecar is read and removed; without one there is no list."""
    stream = tmp_path / "s.264"
    assert box.take_coded_frames(stream) is None
    box.coded_frames_path(stream).write_text("0\n1\n3\n")
    assert box.take_coded_frames(stream) == [0, 1, 3]
    assert not box.coded_frames_path(stream).exists()
