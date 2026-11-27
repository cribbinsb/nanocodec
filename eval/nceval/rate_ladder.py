"""The CBR rate ladder: one definition for the benchmark (tools/bench/plan.py) and the dev screens.

BP-3.10 (docs/benchmark-procedure.md section 5, owner 2026-11-23): 300 / 500 / 800 / 1200 kbps
at 720p30, scaled per clip sub-linearly, target = rung x (pixels / 1280x720)^0.75 x
(fps / 30)^0.5, rounded to whole kbps; a target above half the clip's source bitrate is not
coded. The dev screens (nceval.realuse, cbr.clip_targets, nceval.levels) and the benchmark's
job planner both read it from here, so the two can never use different rates.
"""

from __future__ import annotations

from .clips import MAX_FRACTION_OF_SOURCE_BITRATE

#: The rungs, named by their 720p30 rate (kbps).
LADDER_720P30_KBPS = (300, 500, 800, 1200)
#: The owner's real-time judgement: the bits needed grow less than proportionally with picture
#: size and frame rate (no outside source is claimed).
PIXEL_EXPONENT, FPS_EXPONENT = 0.75, 0.5
REFERENCE_PIXELS, REFERENCE_FPS = 1280 * 720, 30
#: A rung above this share of the clip's source bitrate is not coded (the 50 % rule).
SOURCE_RATE_SHARE = MAX_FRACTION_OF_SOURCE_BITRATE
#: A rate within this of N*1000/1001 is that NTSC rate, for every encoder alike: VVenC refuses a
#: non-standard rate such as 2997/100 ("Detected non-standard Frame Rate", tested 2026-11-23),
#: and the clip sets store NTSC rates rounded (29.97).
NTSC_TOLERANCE = 0.005


def frame_rate(fps: str | float) -> tuple[int, int]:
    """A clip's rate as a fraction: "num/den" from a manifest, or a clip set's float rate
    (an NTSC rate within NTSC_TOLERANCE becomes N*1000/1001)."""
    if "/" in str(fps):
        num, den = map(int, str(fps).split("/"))
    else:
        num, den = round(float(fps) * 1000), 1000
    rate = num / den
    ntsc = round(rate * 1001 / 1000) * 1000
    if ntsc and abs(rate - ntsc / 1001) < NTSC_TOLERANCE and den != 1:
        return ntsc, 1001
    if rate == round(rate):
        return round(rate), 1
    return num, den


def scale(width: int, height: int, fps: str | float) -> float:
    """The factor from a 720p30 rate to the clip's own (PIXEL_EXPONENT, FPS_EXPONENT)."""
    num, den = frame_rate(fps)
    return (width * height / REFERENCE_PIXELS) ** PIXEL_EXPONENT \
        * (num / den / REFERENCE_FPS) ** FPS_EXPONENT


def scaled_kbps(rung: float, width: int, height: int, fps: str | float) -> int:
    """The clip's target for a 720p30-named rung, in whole kbps."""
    return max(1, round(rung * scale(width, height, fps)))


def allowed(kbps: float, source_kbps: float) -> bool:
    """Whether a target is within the clip's cap (SOURCE_RATE_SHARE of its source bitrate)."""
    return kbps <= SOURCE_RATE_SHARE * source_kbps
