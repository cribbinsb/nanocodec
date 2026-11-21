"""What the quick screen (nceval.quick) and the bundle run (nceval.train) share.

The three-stage checks (docs/PROCEDURES.md "Three-stage checks") judge a
preset-level parameter set on one binary: the head build from the
landed-commit cache (tools/dev/binary_for.sh), anchor = the preset as it is,
candidate = the preset plus the change. This module holds what both stages
need: that binary, the preset's values and CPU charge k, the combined CPU
estimate, the stage plan's price, and a deadline that aborts a stage loudly
instead of letting it hang.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import jreal, levers, singlepoint
from .budget import MAX_MINUTES, estimate_minutes
from .cbr import Family, clip_targets
from .clips import Clip
from .codecs import Codec
from .encoders.nanocodec import MEDIUM, param_text, preset_values
from .realuse import LADDER_KBPS, Runner, param_args

REPO = Path(__file__).resolve().parents[2]
BUILD_CACHE = REPO / "tools" / "dev" / "build_cache.py"
#: The CPU charge k of each preset's J (PROCEDURES "Preset roles"): Default is the
#: quality point (0.1), and so is Slow (its gated RD search was screened at the
#: Default's k, docs/handoff/presets.md "The ladder"); Slower is judged on J at k 0
#: inside its budget, every speed rung (fast, faster, veryfast) at the engine k
#: (jreal.K_CPU, 0.6).
PRESET_K = {MEDIUM: 0.1, "slow": 0.1, "slower": 0.0, "veryslow": 0.0}
#: Exit status of a stage aborted by its deadline.
DEADLINE_EXIT = 3


def preset_k(preset: str) -> float:
    return PRESET_K.get(preset, jreal.K_CPU)


def landed_binary(commit: str = "HEAD") -> Path:
    """The landed-commit cache's CLI binary of `commit` (never a local compile;
    the build box builds it when the cache has none). Its presets/ is beside it."""
    resolved = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{commit}^{{commit}}"],
                              capture_output=True, text=True, check=True).stdout.strip()
    out = subprocess.run([sys.executable, str(BUILD_CACHE), "for", resolved],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"no build of {resolved[:12]}: {out.stderr.strip()[-400:]}")
    binary = Path(out.stdout.strip().splitlines()[-1]) / "nanocodec"
    if not binary.is_file():
        raise SystemExit(f"build cache gave {binary}, which is not a file")
    return binary


def presets_beside(binary: Path) -> Path:
    presets = binary.parent / "presets"
    if not presets.is_dir():
        raise SystemExit(f"{binary}: no presets/ beside it (use a landed-cache build)")
    return presets


def preset_params(presets: Path, codec: str, preset: str) -> dict[str, float]:
    """The preset's parameter values ({} for medium, the engine defaults)."""
    return preset_values(presets, Codec(codec), preset)


def pairs(values: dict[str, float]) -> list[str]:
    return [f"{name}={param_text(value)}" for name, value in sorted(values.items())]


def check_known(binary: Path, codec: str, params: dict[str, float]) -> None:
    """Refuse parameter names the binary does not register (a typo or unlanded code)."""
    known = levers.codec_defaults(binary, codec)
    if known is None:
        raise SystemExit(f"{binary}: cannot list its parameters (`nanocodec params`)")
    unknown = [n for n in params if n.removeprefix("nonbase.").removeprefix("key.") not in known]
    if unknown:
        raise SystemExit(f"{codec}: {binary} has no parameter {', '.join(unknown)} (land the "
                         f"code default-off first)")


def combined_dcpu(x86: float | None, jetson: float | None) -> float | None:
    """One CPU change for J: ARM and x86 weighted equally (levers.combined_cpu)."""
    return levers.combined_cpu({"x86_pct": x86, "jetson_pct": jetson})


def families(codec: str, binary: Path, anchor: dict[str, float],
             test: dict[str, float]) -> tuple[Family, Family]:
    """Anchor and candidate on one binary, in CBR, named as nceval.screen names them."""
    name = f"nanocodec-{codec}"
    return (Family(name, Codec(codec), binary, param_args(pairs(anchor)), own=True, cqp=False),
            Family(f"{name}@candidate", Codec(codec), binary, param_args(pairs(test)), own=True,
                   cqp=False))


def rungs(indices) -> Callable[[Clip], list[int]]:
    """A ladder: the clip's legal targets at these indices (those it has), as
    Runner.measure's `ladder` (clip_targets keeps every target of a capped clip)."""
    def ladder(clip: Clip) -> list[int]:
        legal = clip_targets(clip, LADDER_KBPS)
        return [legal[i] for i in indices if i < len(legal)]
    return ladder


def single_rung(clips: list[Clip]) -> Callable[[Clip], list[int]]:
    """The single-point levels' one target per clip (singlepoint.assign_rungs)."""
    assigned = singlepoint.assign_rungs(clips)
    return lambda clip: [clip_targets(clip, LADDER_KBPS)[assigned[clip.name]]]


def price(runner: Runner, anchor: Family, test: Family, clips: list[Clip], frames: int,
          anchor_ladder=None, test_ladder=None,
          key_seconds: float | None = None) -> tuple[float, int]:
    """(CPU-seconds, points) not yet cached for one measurement: the anchor on
    `anchor_ladder` (default the full ladder), the candidate on `test_ladder`
    (default the anchor's)."""
    jobs = (runner.points([anchor], clips, frames, None, key_seconds, ladder=anchor_ladder)
            + runner.points([test], clips, frames, None, key_seconds,
                            ladder=test_ladder or anchor_ladder))
    return runner.price(jobs), sum(1 for bench, p in jobs if not bench.cached(p))


def minutes(cpu_seconds: float) -> float:
    return estimate_minutes(cpu_seconds)


class Deadline:
    """Abort the whole process, loudly, when a stage runs past `limit_minutes`.

    A stage that hangs (a stuck point, a lost slot) must never surface as a
    silent timeout: the watchdog prints what ran out and exits non-zero."""

    def __init__(self, label: str, limit_minutes: float = MAX_MINUTES,
                 on_abort=lambda: None):
        self.label, self.limit = label, limit_minutes
        self.on_abort = on_abort
        self.started = time.monotonic()
        self._done = threading.Event()

    def __enter__(self) -> "Deadline":
        threading.Thread(target=self._watch, daemon=True).start()
        return self

    def __exit__(self, *_) -> None:
        self._done.set()

    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.started) / 60

    def _watch(self) -> None:
        if self._done.wait(self.limit * 60):
            return
        print(f"\n!!! {self.label}: still running after {self.limit:g} min: ABORTED "
              f"(stall or overload; nothing was decided)\n", file=sys.stderr, flush=True)
        self.on_abort()
        os._exit(DEADLINE_EXIT)
