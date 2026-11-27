"""Key shares: how much of a stream is key frames, judged at more than one share.

Owner rule (2026-11-22): key-frame distance is dealt with properly in cost
estimates, screening and the decision of how much quality and CPU go into the
key. Nothing is tuned to one assumption: every acceptance screen measures
quality AND CPU at each share in KEY_SHARES (at least 1 key in 150 frames and
1 key in 300), CPU at the same share and in the same rate mode as its quality.
Why: on 2026-11-22 a key RD search was accepted on screens at 1 key in 300
frames and cost +4-7 % CPU on the chart's 1 key in 150 (reverted, 8b77a4e89).

A share is periodic keys, never the first key alone: the first key follows other
rate-control rules (`rate.key_level_finer` acts only on periodic keys; a screen at
150 f / 1 key and 300 f / 1 key found that arm exactly inert, coordinator
2026-11-22). So share N is measured on a window from frame 0 with a key every N
frames, long enough for at least MIN_PERIODIC_KEYS periodic keys after the first
(`window`: 600 f at 150, 900 f at 300). The period is in frames, not seconds, so
a 25 fps clip has the same share as a 30 fps one. The first key is still inside
the window (1 of the 4 keys at 150, 1 of 3 at 300); it is not scored apart.

Judging (`judge`): each share's per-clip J + k dCPU, with the rung's front slope k
and that share's CPU change; the change may not lose at any share (a share
decided WORSE is WORSE), and the verdict is on the shares' equal-weight mean.

This module is the one place that defines the shares; docs/PROCEDURES.md
"Key shares" describes the rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import interval

#: Frames per key of the shares every acceptance screen measures (owner minimum).
REQUIRED_PERIODS = (150, 300)
KEY_SHARES = REQUIRED_PERIODS
#: Periodic keys (after the first) a share's window holds at least.
MIN_PERIODIC_KEYS = 2
#: The shortest window (the long-window protocol's 600 frames, levels.LONG_FRAMES).
MIN_WINDOW = 600


@dataclass(frozen=True)
class KeyShare:
    """1 key in `period` frames: a window from frame 0 with a key every `period`."""
    period: int

    @property
    def window(self) -> int:
        """Frames coded: MIN_PERIODIC_KEYS periodic keys and the frames after the last."""
        return max(MIN_WINDOW, (MIN_PERIODIC_KEYS + 1) * self.period)

    @property
    def label(self) -> str:
        return f"1/{self.period}"

    def describe(self) -> str:
        return f"1 key in {self.period} f ({self.window} f from frame 0, key every {self.period} f)"


def shares(periods: tuple[int, ...] | list[int] | None = None) -> list[KeyShare]:
    """The shares to measure: `periods` (default KEY_SHARES), which must include the
    owner's minimum REQUIRED_PERIODS."""
    chosen = sorted(set(periods or KEY_SHARES))
    missing = [p for p in REQUIRED_PERIODS if p not in chosen]
    if missing:
        raise ValueError(f"key shares {chosen} leave out the required "
                         f"{', '.join(f'1/{p}' for p in missing)} (owner minimum: "
                         f"{', '.join(f'1/{p}' for p in REQUIRED_PERIODS)})")
    if any(p < 1 for p in chosen):
        raise ValueError(f"key periods must be positive: {chosen}")
    return [KeyShare(p) for p in chosen]


def parse_periods(text: str | None) -> list[KeyShare]:
    """`--key-shares 150,300` (None: KEY_SHARES)."""
    return shares([int(v) for v in text.split(",")] if text else None)


def parse_dcpu(text: str | None, chosen: list[KeyShare]) -> dict[int, float | None]:
    """A CPU change per share from `--dcpu`: `150=1.2,300=0.7` (every share named),
    or one number for every share (accepted, but `uniform_dcpu_warning` says so), or
    None (not measured)."""
    if text is None:
        return {s.period: None for s in chosen}
    text = str(text)
    if "=" not in text:
        value = float(text)
        return {s.period: value for s in chosen}
    given = {}
    for part in text.split(","):
        period, _, value = part.partition("=")
        given[int(period)] = float(value)
    missing = [s.period for s in chosen if s.period not in given]
    if missing:
        raise ValueError(f"--dcpu has no CPU change for share(s) "
                         f"{', '.join(f'1/{p}' for p in missing)}: give PERIOD=PCT for each")
    return {s.period: given[s.period] for s in chosen}


def uniform_dcpu_warning(text: str | None) -> str | None:
    """A loud note when one CPU number stands for every share (it was measured at one)."""
    if text is not None and "=" not in str(text):
        return (f"WARNING: one dCPU ({text} %) used at every key share; CPU should be "
                f"measured at each (e.g. --dcpu 150=X,300=Y from aws_cpu_ab.py --key-shares)")
    return None


def short_window_dcpu(per_share: dict[int, float | None]) -> float | None:
    """The CPU change for the one-key 150-frame levels (L1, L2, L3): the 1/150 share's,
    which has their key fraction."""
    return per_share.get(REQUIRED_PERIODS[0])


def apply_dcpu(args) -> None:
    """Read `args.dcpu` (text, parse_dcpu) at `args.key_shares` into `args.dcpu_shares`
    and set `args.dcpu` to the short-window value (short_window_dcpu); warns when one
    number stands for every share."""
    text = args.dcpu
    args.dcpu_shares = parse_dcpu(text, parse_periods(getattr(args, "key_shares", None)))
    args.dcpu = short_window_dcpu(args.dcpu_shares)
    warning = uniform_dcpu_warning(text)
    if warning:
        print(warning, flush=True)


def combine_platforms(x86: dict[int, float | None], jetson: dict[int, float | None],
                      combined) -> dict[int, float | None]:
    """Per share, `combined(x86, jetson)` (checkrun.combined_dcpu)."""
    return {p: combined(x86.get(p), jetson.get(p)) for p in x86}


def combined_stats(per_share: dict[int, dict]) -> dict:
    """The shares' equal-weight mean of J + k dCPU, with an interval: the mean of the
    shares' means and half-widths combined as independent (the shares' clips differ)."""
    stats = [s for s in per_share.values() if s.get("mean") is not None]
    result = {"n": sum(s.get("n", 0) for s in per_share.values()),
              "confidence": interval.CONFIDENCE, "mean": None, "lo": None, "hi": None,
              "shares": len(per_share)}
    if len(stats) != len(per_share) or not stats:
        return result
    mean = sum(s["mean"] for s in stats) / len(stats)
    below = math.sqrt(sum((s["mean"] - s["lo"]) ** 2 for s in stats)) / len(stats)
    above = math.sqrt(sum((s["hi"] - s["mean"]) ** 2 for s in stats)) / len(stats)
    result.update(mean=mean, lo=mean - below, hi=mean + above)
    return result


def judge(per_share: dict[int, dict], final: bool) -> tuple[str, dict, dict[int, str]]:
    """(decision, combined stats, each share's decision): WORSE when any share is
    WORSE; otherwise the decision on the equal-weight mean (combined_stats)."""
    decisions = {p: interval.decide(s, final) for p, s in per_share.items()}
    combined = combined_stats(per_share)
    if interval.WORSE in decisions.values():
        return interval.WORSE, combined, decisions
    return interval.decide(combined, final), combined, decisions


def describe(per_share: dict[int, dict], decisions: dict[int, str], combined: dict) -> str:
    """One line: each share's J + k dCPU and decision, then their mean."""
    parts = [f"1/{p}: {interval.describe(s)} {decisions[p].upper()}"
             for p, s in sorted(per_share.items())]
    return "; ".join(parts) + f"; mean of shares: {interval.describe(combined)}"
