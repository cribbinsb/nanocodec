#!/usr/bin/env python3
"""Replay stored real-use screens through nceval.earlystop.

For every distinct stage in the given screen JSONs (`nceval.screen --json`
outputs, mode real), feeds its clips one at a time in the order a screen
measures them (StageScope.order) to `certain_worse`, and checks that every
early stop agrees with the stage's verdict over all its clips (recomputed
with today's jreal.objective). Prints the stops, any disagreement, and the
share of clips the stops would have saved. Run it after changing an early
stopping rule or QUALITY_FLOOR_PCT:

    find ~/nanocodec-data/work -name '*.json' | xargs grep -l '"mode": "real"' \\
        | python3 tools/dev/earlystop_replay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from nceval import jreal  # noqa: E402
from nceval.earlystop import StageScope, certain_worse  # noqa: E402

#: Points per arm assumed for a clip not yet measured: the full ladder, the
#: most a stage measures (the conservative choice for the count guards).
LADDER_RUNGS = 4


def stages(paths: list[str]):
    seen = set()
    for path in paths:
        try:
            screen = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        if screen.get("status") != "done" or screen.get("mode") != "real":
            continue
        for stage in screen.get("stages", []):
            key = json.dumps([stage["stage"], stage["per_clip"]], sort_keys=True)
            if key not in seen:
                seen.add(key)
                yield path, stage


def main() -> int:
    paths = [line.strip() for line in sys.stdin if line.strip()]
    count = stopped = disagreements = clips_total = clips_saved = 0
    for path, stage in stages(paths):
        rows = stage["per_clip"]
        verdict = stage["verdict"]
        dcpu, k_cpu = verdict.get("dcpu_pct"), verdict.get("k_cpu", jreal.K_CPU)
        final = jreal.objective(jreal.summarise(rows), dcpu, stage["sigma"], k_cpu)["decision"]
        scope = StageScope({c: r["class"] for c, r in rows.items()},
                           {c: LADDER_RUNGS for c in rows})
        count += 1
        clips_total += len(rows)
        seen: dict[str, dict] = {}
        for index, clip in enumerate(scope.order(), 1):
            seen[clip] = rows[clip]
            if index == len(rows):
                break
            reason = certain_worse(seen, scope, stage["sigma"], dcpu, k_cpu)
            if reason:
                stopped += 1
                clips_saved += len(rows) - index
                agrees = final == "worse"
                disagreements += not agrees
                print(f"{'ok ' if agrees else 'BAD'} {stage['stage']:12s} {index}/{len(rows)} "
                      f"final {final:9s} {reason}  [{path}]")
                break
    print(f"{count} stages, {stopped} stopped early, {disagreements} disagreements; "
          f"{clips_saved}/{clips_total} clips saved ({100 * clips_saved / max(1, clips_total):.0f}%)")
    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
