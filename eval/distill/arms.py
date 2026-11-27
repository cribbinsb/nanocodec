"""Decision-transplant oracle arms: a fast rung given the slowest rung's settings for one
decision type at a time (docs/analysis/distill-fast-2026-11-22.md).

A distilled guide teaches a fast rung's per-superblock decisions what the slowest rung
decides. Its ceiling, per decision type, is the fast rung coding with the slowest rung's
mechanism for that decision: the decisions a perfect guide would predict, in the fast
rung's own context (its references, its other tools). Each arm is the fast rung's
parameters with every parameter of one decision group set to the slowest rung's
effective value (its file's, else the engine default). Light-superblock and non-base
copies of a moved parameter (`light.X`, `nonbase.X`) take the same value, since the
slowest rung has no light superblocks.

    python3 -m distill.arms --binary BIN --codec av1 --rung fast       # print the arms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nceval import levers
from nceval.encoders.nanocodec import preset_values

# Each codec's slowest rung (crates/nc-encoder/src/presets.rs, presets/<codec>/).
SLOWEST = {"av1": "slower", "hevc": "veryslow", "vvc": "veryslow"}
# The fast end each codec offers.
FAST_RUNGS = {"av1": ("medium", "fast", "faster", "veryfast"),
              "hevc": ("medium", "fast", "faster"),
              "vvc": ("medium", "fast", "faster")}
COPY_PREFIXES = ("light.", "nonbase.")

# Decision groups by parameter name (prefixes; first match wins). Key-frame settings are
# their own group: a fast rung's key is one frame in hundreds.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("other", ("hevc.inter_rounding", "hevc.intra_rounding", "av1.inter_rounding")),
    ("key", ("planner.key_", "planner.intra_plan", "av1.keyframe_", "hevc.key_",
             "quant.trellis_key", "rate.key_")),
    ("partition", ("planner.",)),
    ("skip", ("av1.pre_tx_skip", "av1.skip_bias", "av1.chroma_skip", "hevc.early_merge",
              "hevc.merge_", "vvc.merge", "decision.skip_", "decision.static_zero",
              "hevc.pre_transform_zero", "vvc.zero", "vvc.pre", "decision.winner_walk_keeps_skip")),
    ("transform", ("av1.inter_tx_split", "av1.transform_rd", "av1.partial_forward",
                   "av1.pre_tx_luma", "hevc.inter_tu_split", "hevc.max_transform",
                   "hevc.fast_forward", "hevc.partial_forward", "hevc.tu16", "vvc.tx64",
                   "vvc.fast", "vvc.unit", "av1.residual_rd")),
    ("mode", ("decision.", "adapt.", "analysis.", "av1.inter_intra", "av1.near",
              "hevc.intra", "hevc.coarse", "hevc.inter_intra", "vvc.angular", "vvc.isp",
              "vvc.coarse", "vvc.inter", "vvc.temporal", "vvc.rd_candidates")),
)
# The partition group's RD search (as against the planner's own thresholds).
SEARCH_PARAMS = ("planner.rd_split", "planner.split_selector", "planner.rect_rd")
# The groups whose sum is "every superblock decision" (not filters, rate, references).
DECISION_GROUPS = ("partition", "skip", "transform", "mode")


def base_name(name: str) -> str:
    for prefix in COPY_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def group_of(name: str) -> str:
    plain = base_name(name)
    for group, prefixes in GROUPS:
        if plain.startswith(prefixes):
            return group
    return "other"


def arms(binary: Path, codec: str, rung: str) -> dict[str, dict[str, float]]:
    """{arm: full parameter set}: `base` (the rung) and one arm per decision group the
    slowest rung differs in, plus `all` (every decision group)."""
    presets = binary.parent / "presets"
    defaults = levers.codec_defaults(binary, codec)
    if defaults is None:
        raise SystemExit(f"{binary}: cannot list its parameters")
    fast = preset_values(presets, codec, rung)
    slow = preset_values(presets, codec, SLOWEST[codec])
    effective_slow = lambda name: slow.get(base_name(name), defaults[base_name(name)])
    names = set(fast) | {n for n in slow if not n.startswith(COPY_PREFIXES)}
    result = {"base": dict(fast)}
    moved_all: dict[str, float] = {}
    for group in DECISION_GROUPS:
        moved = {n: effective_slow(n) for n in names if group_of(n) == group}
        changed = {n: v for n, v in moved.items()
                   if fast.get(n, defaults.get(base_name(n))) != v}
        if changed:
            result[group] = {**fast, **moved}
            moved_all.update(moved)
    result["all"] = {**fast, **moved_all}
    # The partition group split into the RD search and the planner's rules: how much of
    # the ceiling a threshold change alone gives, with no search and no model.
    partition = changes(result.get("partition", fast), fast)
    search = {n: v for n, v in partition.items() if base_name(n).startswith(SEARCH_PARAMS)}
    if search and len(search) < len(partition):
        result["partition_search"] = {**fast, **search}
        result["partition_rules"] = {**fast, **{n: v for n, v in partition.items()
                                                if n not in search}}
    return result


def changes(arm: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {n: v for n, v in arm.items() if base.get(n) != v}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--codec", required=True, choices=sorted(SLOWEST))
    p.add_argument("--rung", required=True)
    args = p.parse_args()
    result = arms(args.binary.resolve(), args.codec, args.rung)
    print(json.dumps({a: changes(v, result["base"]) for a, v in result.items()}, indent=1,
                     sort_keys=True))


if __name__ == "__main__":
    main()
