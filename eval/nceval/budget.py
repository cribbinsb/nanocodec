"""Wall-time budget for evaluation runs (PROCEDURES: no run longer than 30 min).

Before a screen, sweep or remeasure starts, its points are priced with a
simple measured cost model and divided by the CPU slots this process may
can expect right now (nceval.slots.available_slots: its fair share among the
processes currently holding or queued for slots, capped by the idle cores),
so contention is priced in. A run estimated over `MAX_MINUTES` is
refused with suggested cuts unless `--over-budget` is given, which is logged
to ~/nanocodec-data/run/over-budget.log.

Cost model: `CPU_SECONDS_PER_MEGAPIXEL_FRAME` per megapixel-frame of a
gated, natively scored point. Measured per process on streamed 720p
150-frame points (2026-11-11, quiet machine, full gate, H.264/HEVC/AV1):
encode 1-2, VMAF v1 3.1, own decoder 0.3-0.6, ffmpeg 0.4, JM/HM/aomdec
0.7-1.9, Python PSNR and hashes 2.9 CPU-s, ~10 CPU-s without MS-SSIM
(0.072 per megapixel-frame); the constant adds 1.5x for a loaded machine
(SMT siblings busy). The 2026-11-08 fit (0.18) priced slot waits as CPU:
one process could then run only one point at a time. Rescoring a kept stream
(decode + metrics, no encode) costs about `RESCORE_FRACTION` of a point.
Contention changes during a run, so the estimate is a guide, not a promise.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

from .slots import available_slots

MAX_MINUTES = 30.0
TARGET_MINUTES = 10.0
CPU_SECONDS_PER_MEGAPIXEL_FRAME = 0.11
RESCORE_FRACTION = 0.4


class OverBudget(SystemExit):
    """Raised (exits) when a run is estimated over MAX_MINUTES."""


def point_cpu_seconds(width: int, height: int, frames: int, rescore: bool = False) -> float:
    cost = width * height / 1e6 * frames * CPU_SECONDS_PER_MEGAPIXEL_FRAME
    return cost * RESCORE_FRACTION if rescore else cost


def estimate_minutes(cpu_seconds: float, slots: float | None = None) -> float:
    return cpu_seconds / (slots or available_slots()) / 60.0


def check(label: str, cpu_seconds: float, cuts: str, force: bool) -> float:
    """Print the estimate; refuse (with `cuts`) above MAX_MINUTES unless forced."""
    slots = available_slots()
    minutes = estimate_minutes(cpu_seconds, slots)
    print(f"budget: {label}: ~{minutes:.1f} min wall ({cpu_seconds / 60:.0f} CPU-min over "
          f"{slots:.1f} available slots)", flush=True)
    if minutes <= MAX_MINUTES:
        if minutes > TARGET_MINUTES:
            print(f"budget: over the {TARGET_MINUTES:g}-min target; consider: {cuts}", flush=True)
        return minutes
    if not force:
        raise OverBudget(f"refused: {label} is estimated at {minutes:.0f} min, over the "
                         f"{MAX_MINUTES:g}-min limit. Cut it: {cuts}. "
                         f"(--over-budget forces it and is logged.)")
    log = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data")) / "run" / "over-budget.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        out.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} "
                  f"{label}: {minutes:.0f} min forced\n")
    print(f"budget: OVER BUDGET ({minutes:.0f} min) forced with --over-budget; logged", flush=True)
    return minutes
