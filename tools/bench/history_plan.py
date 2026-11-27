#!/usr/bin/env python3
"""The BP-3 history series' plans: one run per day, our encoder of that day on the "ours"
history clips under the benchmark's conditions (coordinator for the owner, 2026-11-24).

    python3 tools/bench/history_plan.py --days DAYS.json --cbr CBR.json --qp QP.json \\
        --src-dir SRC --out OUT [--bound record|1.15]

Inputs: the day list (day, commit, codecs), the "ours" history-subset plans of today's record
(`plan.py` output: our medium families give each point's command, x264 medium and its tune
twins are the zero line), and each day's source archive (its preset files).

Per day, each of our families is the day's own preset (today's rung names; a rung the day
did not have is absent), and the record's command is adapted to the day only where a
condition's switch differs; every adaptation is written to the day's `conditions.json`:
  - CBR from 2026-11-09 only: before it there was no buffer re-code ceiling (11-06, 11-07;
    11-08's had other semantics) and no key cap, so those days have a Fixed QP point only;
  - the buffer bound (`--bound`, default the record's one buffer, BP-3.13): `--strict-buffer`
    from 11-22, where the flag arrived; from 11-09 to 11-21 `rate.recode_overflow=1.0` (a
    fullness ceiling since 2e8d2a9b5), the key cap as the record sets it; `--bound 1.15` gives
    our default (no flag from 11-22, the ceiling at 1.15 before);
  - `hold.enable` before 11-11 and `stream.encoder_info` before 11-23 are dropped (the hold
    and the information SEI did not exist; their absence is the condition); before 11-23
    `hold.code_skip` is dropped and hold runs off, so every frame is still coded;
  - `--temporal-layers` before 11-08 is dropped (single layer was the only mode).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from nceval.encoders.nanocodec import preset_cli_args, preset_path  # noqa: E402

#: Today's single-layer rung names, as the record's families use them.
RUNGS = ("veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
CBR_FROM, HOLD_FROM, INFO_FROM, LAYERS_FROM, STRICT_FLAG_FROM = (
    "2026-11-09", "2026-11-11", "2026-11-23", "2026-11-08", "2026-11-22")
#: All-skip output for held frames (`hold.code_skip`, 96d1626d6). Before it, a held frame was
#: not coded at all, so those days run hold off to keep every frame coded (D21).
CODE_SKIP_FROM = "2026-11-23"


def drop_param(argv: list[str], name: str) -> list[str]:
    """`argv` without `--param name=...`."""
    out, skip = [], False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg == "--param" and i + 1 < len(argv) and argv[i + 1].startswith(f"{name}="):
            skip = True
            continue
        out.append(arg)
    return out


def drop_flag(argv: list[str], flag: str, takes_value: bool) -> list[str]:
    if flag not in argv:
        return argv
    i = argv.index(flag)
    return argv[:i] + argv[i + 1 + takes_value:]


def adapt(argv: list[str], day: str, mode: str, bound: str) -> tuple[list[str], list[str]]:
    """The record's command as the day can run it, and the adaptations made."""
    notes = []
    if mode == "cbr":
        if day >= STRICT_FLAG_FROM:
            if bound == "1.15" and "--strict-buffer" in argv:
                argv = drop_flag(argv, "--strict-buffer", False)
                notes.append("--strict-buffer dropped: 1.15 buffers is the default bound")
            if bound == "1.15" and any(a.startswith("rate.key_cap_buffer=") for a in argv):
                argv = drop_param(argv, "rate.key_cap_buffer")
                notes.append("rate.key_cap_buffer dropped: 1.0 is the default (e84ef02c)")
        else:
            argv = drop_flag(argv, "--strict-buffer", False)
            ceiling = "1.0" if bound == "record" else bound
            argv = argv + ["--param", f"rate.recode_overflow={ceiling}"]
            notes.append(f"bound set by rate.recode_overflow={ceiling} (--strict-buffer arrived "
                         f"{STRICT_FLAG_FROM})")
    if day < CODE_SKIP_FROM and any(a.startswith("hold.code_skip=") for a in argv):
        argv = drop_param(argv, "hold.code_skip")
        argv = [("hold.enable=0" if a.startswith("hold.enable=") else a) for a in argv]
        notes.append(f"hold off: all-skip output (hold.code_skip) arrived {CODE_SKIP_FROM}, so "
                     "earlier days code every frame without hold")
    if day < HOLD_FROM and any(a.startswith("hold.enable=") for a in argv):
        argv = drop_param(argv, "hold.enable")
        notes.append(f"hold.enable dropped: no hold before {HOLD_FROM}")
    if day < INFO_FROM and any(a.startswith("stream.encoder_info=") for a in argv):
        argv = drop_param(argv, "stream.encoder_info")
        notes.append(f"stream.encoder_info dropped: no information SEI before {INFO_FROM}")
    if day < LAYERS_FROM and "--temporal-layers" in argv:
        argv = drop_flag(argv, "--temporal-layers", True)
        notes.append(f"--temporal-layers dropped: single layer only before {LAYERS_FROM}")
    return argv, notes


def day_presets(archive: Path, workdir: Path) -> Path:
    """The day's presets directory, extracted from its source archive."""
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(workdir), "--wildcards",
                    "*presets/*"], check=False, capture_output=True)
    for candidate in ("crates/nc-encoder/presets", "presets"):
        found = list(workdir.glob(f"**/{candidate}"))
        if found:
            return found[0]
    return workdir / "presets"


#: The strict series' family suffix: ours at one buffer (`--strict-buffer` or its ceiling),
#: beside the default bound (fairness review, 2026-11-25: both bounds measured, side by side).
STRICT_SUFFIX = "~strict"


def plan_day(day: dict, templates: list[dict], zero: list[dict], src: Path,
             bound: str, both: bool = False) -> tuple[list[dict], dict]:
    jobs, rungs, notes = [], {}, {}
    with tempfile.TemporaryDirectory(prefix="nanocodec-hist-") as tmp:
        presets = day_presets(src / f"{day['commit']}.tar.gz", Path(tmp))
        for codec in day["codecs"].split():
            have = [r for r in RUNGS if r == "medium" or preset_path(presets, codec, r).exists()]
            rungs[codec] = have
            for rung in have:
                args = list(preset_cli_args(codec, rung, presets))
                for t in templates:
                    if t["family"] != f"nanocodec-{codec}@medium":
                        continue
                    if t["mode"] == "cbr" and day["day"] < CBR_FROM:
                        continue
                    base = t["argv"]
                    if "--preset" in base:  # today's medium names no preset; be safe
                        base = drop_flag(base, "--preset", True)
                    bounds = [bound] + (["record"] if both and t["mode"] == "cbr"
                                        and bound != "record" else [])
                    for b in bounds:
                        argv, why = adapt(base + args, day["day"], t["mode"], b)
                        family = f"nanocodec-{codec}@{rung}" + (
                            STRICT_SUFFIX if b == "record" and both else "")
                        point = t["id"].split("|", 1)[1]
                        jobs.append({**t, "family": family, "id": f"{family}|{point}",
                                     "argv": argv})
                        for note in why:
                            notes[note] = notes.get(note, 0) + 1
    jobs += zero
    return jobs, {"day": day["day"], "commit": day["full"], "rungs": rungs,
                  "cbr": day["day"] >= CBR_FROM, "bound": bound,
                  "strict_series": both, "adaptations": notes,
                  "jobs": len(jobs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=Path, required=True)
    ap.add_argument("--cbr", type=Path, required=True)
    ap.add_argument("--qp", type=Path, help="the Fixed QP plan (BP-3; BP-4 measures CBR only)")
    ap.add_argument("--src-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bound", choices=("record", "1.15"), default="record",
                    help="the CBR buffer bound: the record's one buffer (BP-3.13: as the "
                         "benchmark ran), or 1.15 buffers (our default)")
    ap.add_argument("--no-zero", action="store_true",
                    help="leave the zero line out: BP-4 reads it from the main series' runs, "
                         "which code x264 medium on every history clip (D15)")
    ap.add_argument("--both-bounds", action="store_true",
                    help=f"also code CBR at one buffer, as families ending {STRICT_SUFFIX}")
    a = ap.parse_args()
    plans = json.loads(a.cbr.read_text()) + (json.loads(a.qp.read_text()) if a.qp else [])
    templates = [j for j in plans if j["family"].endswith("@medium")
                 and j["family"].startswith("nanocodec-")]
    zero = [] if a.no_zero else [j for j in plans if j["family"].startswith("x264-medium")]
    for day in json.loads(a.days.read_text()):
        jobs, conditions = plan_day(day, templates, zero, a.src_dir, a.bound, a.both_bounds)
        out = a.out / day["day"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "jobs.json").write_text(json.dumps(jobs))
        (out / "conditions.json").write_text(json.dumps(conditions, indent=1))
        print(f"{day['day']} {day['commit']}: {len(jobs)} jobs, CBR {conditions['cbr']}, "
              f"rungs {sum(map(len, conditions['rungs'].values()))}")


if __name__ == "__main__":
    main()
