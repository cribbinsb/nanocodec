"""The central dev rebaseline's last outcome, so a missing or failed one is never silent.

tools/dev/rebaseline.py re-encodes every dev baseline point once after our benchmark
conditions change. On 2026-11-25 it failed twice unnoticed (a non-zero exit in a systemd
unit nobody read), and agents re-encoded their anchors cold, about 110 CPU-min a codec.
It now records each run here; every screen prints one line when the record does not show
a completed run under today's conditions (`notice`), and land.sh prints the same line.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .cbr import conditions_tag
from .config import load_paths

RUNNING, OK, FAILED = "running", "ok", "failed"
RERUN = "systemd-run --user --scope -p MemoryMax=2G python3 tools/dev/rebaseline.py"


def state_file() -> Path:
    return load_paths().data_root / "results" / "rebaseline-state.json"


def record(commit: str, status: str, detail: str = "") -> None:
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"commit": commit, "status": status, "detail": detail,
                                "conditions": conditions_tag(),
                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


def notice() -> str | None:
    """One warning line when the dev baselines are not known to be warm, else None."""
    path = state_file()
    state = json.loads(path.read_text()) if path.exists() else None
    if state is None:
        why = "no rebaseline has been recorded"
    elif state["conditions"] != conditions_tag():
        why = f"the last rebaseline ({state['time']}) ran under other benchmark conditions"
    elif state["status"] != OK:
        why = (f"the last rebaseline ({state['commit'][:10]}, {state['time']}) is "
               f"{state['status']}{': ' + state['detail'] if state['detail'] else ''}")
    else:
        return None
    return (f"!!! dev baselines may be cold: {why}; anchors re-encode here until "
            f"`{RERUN}` completes")
