"""tools/dev/nc_remote.py occupancy: the job records' summary and their copy from the box."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "dev" / "nc_remote.py"
spec = importlib.util.spec_from_file_location("nc_remote", SCRIPT)
nc_remote = importlib.util.module_from_spec(spec)
sys.modules["nc_remote"] = nc_remote  # dataclasses look their module up by name
spec.loader.exec_module(nc_remote)


def job(name, started, ended, wait=0.0, pool="main"):
    return {"name": name, "started": started, "ended": ended, "wait_s": wait, "pool": pool}


def test_summary_weights_concurrency_by_time_and_leaves_verify_out():
    records = [job("a", 0, 50), job("b", 25, 75, wait=10), job("verifier-x", 0, 100),
               job("c", 10, 20, pool="overflow")]
    text = nc_remote.occupancy_summary(records, 0, 100)
    assert "4 jobs; wait p50 0 s" in text and "max 10 s; 1 waited at all; 1 on the overflow" in text
    # 0-25: one; 25-50: two; 50-75: one; 75-100: none
    assert "0: 25.0 %, 1: 50.0 %, 2: 25.0 %" in text


def test_collect_copies_only_whole_new_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(nc_remote, "OCCUPANCY", tmp_path / "occupancy.jsonl")
    monkeypatch.setattr(nc_remote, "OCCUPANCY_OFFSETS", tmp_path / "offsets.json")
    line = json.dumps(job("a", 1, 2)).encode() + b"\n"
    remote = line + line + b'{"name": "half'
    seen = []

    def ssh(box, command, **_):
        start = int(command.split("+")[1].split()[0]) - 1
        seen.append(start)
        return subprocess.CompletedProcess(command, 0, remote[start:], b"")

    monkeypatch.setattr(nc_remote, "ssh", ssh)
    box = nc_remote.Box(instance_id="i-1", instance_type="c7i.24xlarge", zone="z", price=0.5,
                        launched="2026-11-24T00:00:00Z", expected_end="2026-11-24T12:00:00Z")
    nc_remote.collect_occupancy(box)
    nc_remote.collect_occupancy(box)
    rows = [json.loads(r) for r in (tmp_path / "occupancy.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["instance_type"] == "c7i.24xlarge"
    assert seen == [0, 2 * len(line)]  # the half-written line is read again next time
