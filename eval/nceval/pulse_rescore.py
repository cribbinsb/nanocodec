"""Re-score a stored screen or final-check's periodic-key pulse with the key-attributed guard.

    python3 -m nceval.pulse_rescore RESULT.json [RESULT.json ...]

For results measured before the guard counted only the part of a pulse change the key
itself caused (jreal.key_attributed_pulse, 2026-11-21): the L3 pulse passes are
rebuilt from the result's own builds, parameters and draw (as nceval.levels runs
them), read from the point caches (this machine's and the cloud one), and judged
again. A clip with any point not cached is left out and listed: nothing is encoded.
Each result prints the raw and the key-attributed balanced and worst-clip pulse, the
keys' and the P frames' changes, and the verdict of each measure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from . import jreal, levels, pool
from .cbr import Family
from .codecs import Codec
from .config import load_paths
from .gatepolicy import EVALUATION_GATE, GatePolicy
from .realuse import (LONG_PULSE_FRAMES, LONG_PULSE_KEY_SECONDS, PERIODIC_KEY_SECONDS,
                      PERIODIC_RUNGS, Runner, one_per_class, param_args)

CLOUD_WORK = "cloud-measure"


def arms(result: dict) -> list[tuple[str, Family, Family]]:
    """(codec, anchor, test) of a screen result or of each codec of a final-check report."""
    if "codecs" in result:  # levels.final_check
        paths = load_paths()
        old, new = Path(result["old"]), Path(result["new"])
        if not old.is_file():
            old = paths.data_root / "builds" / result["old_commit"][:12] / "nanocodec"
        if not new.is_file():
            new = paths.data_root / "builds" / result["new_commit"][:12] / "nanocodec"
        presets = result.get("presets") or {}
        out = []
        for name in result["codecs"]:
            codec = Codec(name)
            def args(binary, key):
                directory = Path(presets[key]) if presets.get(key) else None
                stem = result.get("preset")
                if directory and stem and not (directory / name / f"{stem}.toml").exists() \
                        and (directory / name / f"{stem}er.toml").exists():
                    stem += "er"  # 2026-11-21: the old `slow` rung is `slower`
                return levels.preset_args(directory, codec, stem)
            out.append((name, Family(f"nanocodec-{name}", codec, old, param_args(args(old, "old")),
                                     own=True, cqp=False),
                        Family(f"nanocodec-{name}@candidate", codec, new,
                               param_args(args(new, "new")), own=True, cqp=False)))
        return out
    codec = Codec(result["codec"])
    anchor = Family(f"nanocodec-{codec.value}", codec, Path(result["anchor_binary"]),
                    param_args(result.get("anchor_params") or []), own=True, cqp=False)
    test = Family(f"nanocodec-{codec.value}@candidate", codec, Path(result["binary"]),
                  param_args(result.get("params") or []), own=True, cqp=False)
    return [(codec.value, anchor, test)]


def l3_clips(result: dict, codec: str) -> tuple[list, int]:
    """The clips of the result's L3 (long window when it was one) and its frame count."""
    level = (result.get("codecs") or {}).get(codec, result)
    draw = level.get("draw") or {}
    frames = int(draw.get("frames") or levels.DEFAULT_FRAMES)
    drawn = pool.draw(pool.load_pool(load_paths(), levels.usable), draw["rotation"], frames)
    return drawn.levels[pool.FINAL], frames


def cached(runner: Runner, families: list[Family], clip, frames: int, seconds: float,
           rungs) -> bool:
    jobs = runner.points(families, [clip], frames, rungs, seconds)
    return bool(jobs) and all(bench.cached(point) for bench, point in jobs)


def rescore(result: dict) -> list[dict]:
    out = []
    paths = load_paths()
    runners = [Runner(paths, GatePolicy(EVALUATION_GATE.value), 1, over_budget=True),
               Runner(dataclasses.replace(paths, work_root=paths.work / CLOUD_WORK),
                      GatePolicy(EVALUATION_GATE.value), 1, over_budget=True)]
    for codec, anchor, test in arms(result):
        clips, frames = l3_clips(result, codec)
        level = (result.get("codecs") or {}).get(codec, result)
        level_frames = (level.get("stages") or [{}])[-1].get("frames") or frames
        long_clips = levels.long_clips(clips, levels.LONG_FRAMES)
        # a long-window L3 measured on the full ladder (its single points reshaped the
        # curve): its pulse is in every long clip's own curve too
        main = [(long_clips, levels.LONG_FRAMES, levels.LONG_KEY_SECONDS, jreal.PULSE_TERMS[0],
                 None)] if int(level_frames) >= levels.LONG_FRAMES else []
        passes = main + [(one_per_class(clips), frames, PERIODIC_KEY_SECONDS,
                          jreal.PULSE_TERMS[0], PERIODIC_RUNGS),
                         (one_per_class([dataclasses.replace(c, start_frame=0) for c in clips
                                         if c.frames >= LONG_PULSE_FRAMES]),
                          LONG_PULSE_FRAMES, LONG_PULSE_KEY_SECONDS, jreal.PULSE_TERMS[1],
                          PERIODIC_RUNGS)]
        rows: dict[str, dict] = {}
        missing = []
        for pass_clips, pass_frames, seconds, term, rungs in passes:
            for clip in pass_clips:
                runner = next((r for r in runners if cached(r, [anchor, test], clip,
                                                              pass_frames, seconds, rungs)), None)
                if runner is None:
                    missing.append(f"{clip.name}@{seconds:g}s")
                    continue
                curves = runner.measure("rescore", [anchor, test], [clip], pass_frames,
                                        rungs, seconds, budgeted=False)
                a, t, _, _ = jreal.pair_rungs(curves[(anchor.name, clip.name)],
                                              curves[(test.name, clip.name)])
                terms = jreal.compare_clip(a, t, min_points=2)
                names = jreal.pulse_companions(term)
                row = rows.setdefault(clip.name, {"class": clip.content_class})
                row[names["raw"]] = terms["d_periodic_pulse_psnr_y"]
                row[names["key"]] = terms["d_periodic_key_psnr_y"]
                row[names["pre_key"]] = terms["d_periodic_pre_key_psnr_y"]
                row[term] = jreal.key_attributed_pulse(row[names["raw"]], row[names["key"]])
        out.append({"codec": codec, "rows": rows, "missing": missing,
                    "summary": {term: jreal.aggregate(rows, term) for term in
                                (*jreal.PULSE_TERMS, *(n for term in jreal.PULSE_TERMS
                                                       for n in jreal.pulse_companions(term).values()))}})
    return out


def verdict(summary: dict, raw: bool) -> str:
    """WORSE if a pulse term passes a pulse limit (raw: the old measure)."""
    for term in jreal.PULSE_TERMS:
        name = jreal.pulse_companions(term)["raw"] if raw else term
        cell = summary.get(name) or {}
        if (cell.get("balanced") or 0) > jreal.PULSE_GUARD_DB or \
                (cell.get("max") or 0) > jreal.PULSE_WORST_CLIP_DB:
            return "WORSE"
    return "ok"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python3 -m nceval.pulse_rescore", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    args = p.parse_args(argv)
    fmt = lambda v: "  n/a" if v is None else f"{v:+.2f}"
    for path in args.results:
        try:
            scored_all = rescore(json.loads(Path(path).read_text()))
        except (OSError, KeyError, ValueError) as error:
            print(f"{Path(path).parent.name}/{Path(path).name}: cannot rebuild ({error})")
            continue
        for scored in scored_all:
            s = scored["summary"]
            cells = []
            for term in jreal.PULSE_TERMS:
                names = jreal.pulse_companions(term)
                window = "10s" if "long" in term else "2s"
                cells.append(f"{window}: raw {fmt(s[names['raw']]['balanced'])}/"
                             f"{fmt(s[names['raw']]['max'])} key-attr {fmt(s[term]['balanced'])}/"
                             f"{fmt(s[term]['max'])} dkey {fmt(s[names['key']]['balanced'])} "
                             f"dP {fmt(s[names['pre_key']]['balanced'])}")
            print(f"{Path(path).parent.name}/{Path(path).name} {scored['codec']}: "
                  f"raw {verdict(s, True)} -> key-attributed {verdict(s, False)}; "
                  + "; ".join(cells)
                  + (f"; not cached: {len(scored['missing'])}" if scored["missing"] else ""))


if __name__ == "__main__":
    main()
