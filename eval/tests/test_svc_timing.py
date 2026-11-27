"""The SVC timing box runner (tools/dev/svc_timing_box.py): a reclaimed box's records survive,
and a restart with them codes only the runs still missing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[2] / "tools" / "dev" / "svc_timing_box.py"


def run_box(tmp_path: Path, out: Path) -> None:
    subprocess.run([sys.executable, "-I", str(RUNNER), "--plan", str(tmp_path / "plan.json"),
                    "--bin", str(tmp_path / "bin"), "--clips", str(tmp_path), "--out", str(out),
                    "--rounds", "2", "--cpu", "0"], check=True, capture_output=True)


def test_a_restarted_box_runs_only_the_missing_encodes(tmp_path):
    (tmp_path / "bin").mkdir()
    calls = tmp_path / "calls.txt"
    encoder = tmp_path / "bin" / "enc"
    encoder.write_text(f"#!/bin/sh\necho \"$1\" >> {calls}\nprintf abc > \"$2\"\n")
    encoder.chmod(0o755)
    entry = {"argv": ["{BIN}/enc", "{INPUT}", "{OUTPUT}"]}
    (tmp_path / "plan.json").write_text(json.dumps(
        {"points": {"a": {"c1": entry, "c2": entry}, "b": {"c1": entry}}}))
    out = tmp_path / "runs.jsonl"
    # the lost box got two runs home, and a third line was cut by the reclaim
    out.write_text(json.dumps({"point": "a", "clip": "c1", "round": 0, "cpu_seconds": 1.0}) + "\n"
                   + json.dumps({"point": "b", "clip": "c1", "round": 1, "cpu_seconds": 1.0})
                   + "\n" + '{"point": "a", "cl')
    run_box(tmp_path, out)
    runs = [json.loads(line) for line in out.read_text().splitlines()[3:]]
    done = {(r["point"], r["clip"], r["round"]) for r in runs}
    assert done == {("a", "c2", 0), ("b", "c1", 0), ("a", "c1", 1), ("a", "c2", 1)}
    assert len(calls.read_text().splitlines()) == 4
    assert all(r["bytes"] == 3 for r in runs)
