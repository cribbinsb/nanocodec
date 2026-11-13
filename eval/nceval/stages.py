"""Staged evaluation plans shared by the screens and the real-use board: leaf module.

Which clips each stage measures, at how many frames and quantizers (or
ladder targets), the sigma each proxy stage's estimate carries, how a
stage decides, how sigmas are re-learned from screen history, and how an
over-budget stage splits into class-balanced parts. nceval.screen,
nceval.realuse, nceval.cbr and nceval.tune all use these; keeping them here
(importing nothing of nceval but its config) keeps the evaluation modules
free of import cycles (tests/test_imports.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .config import load_paths

# Class-balanced TRAIN proxy: three clips per class, distinct sources. Any
# even size codes in every codec (padded and cropped inside the encoder).
# The texture class (2026-11-09) is pristine fine-texture content (AOM CTC /
# Xiph sources, 130 frames, <= 720p): the other classes are pre-compressed
# camera footage, on which HEVC hid a large loss to x265 on fine texture.
TRAIN_PROXY = (
    "bodycam_bc1", "bodycam_antare_0815", "handheld_mot17_11",
    "crowd_pp22_027", "crowd_mot17_09", "crowd_mot20_02",
    "lowlight_pp22_004", "lowlight_bdd_00c497ae", "lowlight_city_night",
    "pan_bdd_00c12bd0", "pan_jaad_0117", "pan_jaad_0223",
    "static_meva_hospital", "static_otw_homes432", "static_cam22",
    "texture_aerial3200_720p", "texture_tunnelflag_720p", "texture_stockholm_360p",
)
METRICS = ("psnr", "vmaf_v1")
MARGIN_SIGMAS = 2.0
OUTER_THREE = (0, 1, 3)  # of four QPs in ascending order: drop the third
ALL_FOUR = (0, 1, 2, 3)
PROXY_4 = ("crowd_mot20_02", "crowd_pp22_027", "pan_jaad_0117", "static_cam22")
PROXY_6 = PROXY_4 + ("bodycam_antare_0815", "pan_jaad_0223")
# A stage's sigma is re-derived from screen history once this many completed
# (all-stage) screens of its set exist; until then its prior is used.
MIN_HISTORY = 20


@dataclass(frozen=True)
class Stage:
    name: str
    clips: tuple[str, ...] | None  # None: the whole TRAIN proxy
    qp_indices: tuple[int, ...]
    sigma: float  # percent; 0 for the exact target; the prior until history exists
    frames: int | None = None  # cap on the screen's frame count for this stage


# Stage 1 at 60 frames: on four settings (h264 x2, hevc, av1) the 4-clip
# outer-3 BD at 60 vs 150 frames differed by 0.50 % RMS (max 0.86 %, same
# sign every time), added in quadrature to the 2.3 % proxy error.
FRAMES_STAGE_1 = 60
STAGES = (Stage("proxy-4", PROXY_4, OUTER_THREE, 2.4, FRAMES_STAGE_1),
          Stage("proxy-6", PROXY_6, ALL_FOUR, 1.0),
          Stage("train", None, ALL_FOUR, 0.0))
# VAL confirmation (VAL-150): one clip per content class, then all natural VAL
# clips, outer three QPs throughout. No history covers VAL yet, so the proxy's
# sigma is PROVISIONAL, borrowed from the 4-clip TRAIN proxy; re-derive it once
# staged VAL results have accumulated in the cache.
VAL_PROXY = ("static_otw_homes390", "pan_bdd_00cef86b", "crowd_pp22_051", "bodycam_airport",
             "lowlight_pp22_049")
VAL_STAGES = (Stage("val-proxy-5", VAL_PROXY, OUTER_THREE, 2.3),
              Stage("val", None, OUTER_THREE, 0.0))


def cache_root() -> Path:
    return load_paths().data_root / "screen-cache"


def history_path() -> Path:
    return cache_root() / "history.jsonl"


def learned_sigmas(clip_set: str, plan: tuple[Stage, ...], path: Path | None = None,
                   metrics: tuple[str, ...] = METRICS) -> dict[str, tuple[float, int]]:
    """RMSE of each proxy stage's mean against the final stage's, from history.

    Only screens that ran every stage count (early-stopped screens would bias
    the estimate towards ambiguous candidates); `--all-stages` produces them.
    Returns {stage: (rmse_percent, samples)}.
    """
    final = plan[-1].name
    errors: dict[str, list[float]] = {s.name: [] for s in plan[:-1]}
    try:
        lines = (path or history_path()).read_text().splitlines()
    except FileNotFoundError:
        return {}
    for line in lines:
        entry = json.loads(line)
        if entry["set"] != clip_set or final not in entry["stages"]:
            continue
        for stage, values in errors.items():
            proxy = entry["stages"].get(stage)
            if proxy is None:
                continue
            for metric in metrics:
                p, t = proxy.get(metric), entry["stages"][final].get(metric)
                if p is not None and t is not None:
                    values.append(p - t)
    return {stage: ((sum(e * e for e in values) / len(values)) ** 0.5, len(values))
            for stage, values in errors.items() if values}


def calibrated_plan(clip_set: str, plan: tuple[Stage, ...], path: Path | None = None,
                    metrics: tuple[str, ...] = METRICS) -> tuple[tuple[Stage, ...], str]:
    """The plan with sigmas re-derived where enough history exists, and a note."""
    learned = learned_sigmas(clip_set, plan, path, metrics)
    stages, notes = [], []
    for stage in plan:
        rmse, n = learned.get(stage.name, (None, 0))
        if rmse is not None and n >= MIN_HISTORY * len(metrics):
            stage = replace(stage, sigma=round(rmse, 2))
            notes.append(f"{stage.name} sigma {rmse:.2f}% learned (n={n // len(metrics)})")
        elif stage.sigma:
            notes.append(f"{stage.name} sigma {stage.sigma:.1f}% prior "
                         f"({n // len(metrics)}/{MIN_HISTORY} complete screens)")
        stages.append(stage)
    return tuple(stages), "; ".join(notes)


def decide(means: dict[str, float | None], sigma: float) -> str:
    """better / worse / ambiguous for one stage (see the module docstring)."""
    if any(v is None for v in means.values()):
        return "ambiguous"
    margin = MARGIN_SIGMAS * sigma
    if all(v + margin < 0 for v in means.values()):
        return "better"
    if any(v - margin > 0 for v in means.values()):
        return "worse"
    return "ambiguous" if sigma > 0 else "neutral"


# A part is sized to fill at most this fraction of the limit: the estimate
# moves with contention while a part runs.
PART_FILL = 0.8


def balanced_parts(clips: list, count: int) -> list[list]:
    """`count` class-balanced parts (dealt round-robin within each class)."""
    ordered = sorted(clips, key=lambda c: (c.content_class, c.name))
    return [ordered[index::count] for index in range(count) if ordered[index::count]]
