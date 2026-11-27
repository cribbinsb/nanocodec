#!/usr/bin/env python3
"""The history series' clips of the "ours" set: the one definition every history tool reads.

`eval/clipset/history-set.json` holds the ten clips drawn on 2026-11-23 (`clips`, dataset
"ours") and the extension of 2026-11-25 (`ours_extension`: the clips added so that the subset
tracks the full set, with the rule that chose them and its measured error).

    python3 tools/bench/history_clips.py          # prints the names, one per line
"""

from __future__ import annotations

import json
from pathlib import Path

HISTORY_SET = Path(__file__).resolve().parents[2] / "eval" / "clipset" / "history-set.json"
DATASET = "ours"


def base_clips() -> list[str]:
    """The ten clips of the first draw."""
    record = json.loads(HISTORY_SET.read_text())
    return sorted(c["name"] for c in record["clips"] if c["dataset"] == DATASET)


def added_clips() -> list[str]:
    """The extension's clips, in the order they were added."""
    record = json.loads(HISTORY_SET.read_text())
    return [c["name"] for c in record.get("ours_extension", {}).get("clips", [])]


def history_clips() -> list[str]:
    """Every history clip of the "ours" set, sorted."""
    return sorted(base_clips() + added_clips())


if __name__ == "__main__":
    print("\n".join(history_clips()))
