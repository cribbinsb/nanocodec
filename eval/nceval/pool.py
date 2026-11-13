"""The rotating clip pool and its daily draws: the screens' clips (anti-overfitting).

Every adoption decision used to be taken on the same 11 clips (proxy-6 and
val-proxy-5); on 15 never-used clips one day's "flat" AV1 Default was 1.3%
worse and an H.264 -3% gain was -1.0% (2026-11-11). So no screen picks its
clips any more: they are drawn from a large pool, stratified by class, by a
rotation key that changes every day, and the start frame of every clip
longer than the screen rotates with it.

Pool: the natural (no synthetic noise), non-TEST clips of POOL_SETS (raw YUV
sets plus `pool`, the high-bitrate encodes of eval/clipset/build_pool.py),
minus any clip whose source family is also a sealed clip's (main's TEST,
adv12: both stay for milestones). A family is the source a clip was cut
from (nceval.clips): a draw holds at most one clip per family, and a family
is in at most one level of a day.

Levels (nceval.realuse runs them in order; docs/PROCEDURES.md):

  L1 quick   LEVEL_SIZES["quick"] families, every stratum, 3 rungs: kills
             clear losers in a few minutes.
  L2 adopt   a disjoint draw of LEVEL_SIZES["adopt"], full ladder.
  L3 final   every other family of the pool (one clip each), full ladder,
             in parts under 30 min: required before a preset/Default change
             lands, and after every bundle (`realuse final-check`).

The rotation key is the UTC date unless NC_POOL_ROTATION or --rotation sets
one (to reproduce a recorded run); every result records it with each clip's
start frame. Anchor and candidate always share the draw.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .clips import Clip, load_clipset
from .config import Paths

#: Clip sets the pool takes its clips from (hires: 1080p/5MP copies of TRAIN
#: sources; quick/smoke: duplicates).
POOL_SETS = ("main", "broad_independent_v1", "layergate_fit", "layergate_fit2", "long", "pool")
#: Sealed: neither these clips nor their source families enter the pool, so no
#: screen or tuning study draws them. adv12 is kept for milestones; `independent`
#: (eval/clipset/select_independent.py: 150 never-used sources, 2026-11-13) is the
#: independent test's fresh set and the candidate main chart.
SEALED_SETS = ("adv12", "independent")
SEALED_SPLIT = "test"
#: Source quality (owner, 2026-11-11): our encoder sees clean camera frames,
#: so a clip of a poorly compressed source mostly tests transcoding. Each
#: clip's source is graded by eval/clipset/grade.py (bits per pixel,
#: blockiness, noise): A clean is used at any size, B mediocre only
#: downscaled by MIN_POOR_DOWNSCALE or more with at least BPP_FLOOR source
#: bits per stored pixel left, C poor never.
BPP_FLOOR = 0.15
MIN_POOR_DOWNSCALE = 2.0
#: Draw strata: the five camera classes and everything else.
STRATA = {"crowd": ("crowd",), "lowlight": ("lowlight",), "pan": ("pan",),
          "static": ("static",), "bodycam": ("bodycam",), "other": ("texture", "drone")}
QUICK, ADOPT, FINAL = "quick", "adopt", "final"
LEVELS = (QUICK, ADOPT, FINAL)
#: Families per level (None: the rest of the pool). Measured 2026-11-11 on a
#: loaded machine (H.264, 8 workers): 12 / 24 / 111 clips took 1.1 / 4.6 /
#: 24.5 min. The pool (~150 families) is the limit, so L1 and L2 are doubled
#: (~2 and ~9 min) to stop losers sooner, and L3 keeps the other ~75
#: (~17 min); a bigger pool (eval/clipset/pool.json) lengthens L3 towards 30.
LEVEL_SIZES = {QUICK: 24, ADOPT: 48, FINAL: None}
ROTATION_ENV = "NC_POOL_ROTATION"
#: L1's chosen clip sets (nceval.clipselect: the clips whose mean best predicts
#: the pool's J on measured responses), used one per rotation key in turn.
#: While it lists none, L1 is drawn like the other levels.
L1_SETS = Path(__file__).resolve().parents[1] / "clipset" / "l1_sets.json"


def rotation_key(override: str | None = None) -> str:
    """Today's rotation key (UTC date), unless one is given to reproduce a run."""
    return override or os.environ.get(ROTATION_ENV) or \
        datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def stratum(clip: Clip) -> str:
    return next(name for name, classes in STRATA.items() if clip.content_class in classes)


def load_pool(paths: Paths, usable: Callable[[Clip], bool] = lambda clip: True) -> list[Clip]:
    """Every pool clip (see the module docstring) that is `usable`, by name.

    The screens pass "has enough ladder rungs under its 50% cap for a BD"."""
    sealed_families, chosen = set(), {}
    sets = {name: load_clipset(paths.clips / f"{name}.json", paths.data_root).clips
            for name in (*POOL_SETS, *SEALED_SETS) if (paths.clips / f"{name}.json").exists()}
    for name, clips in sets.items():
        sealed_families |= {family_key(c.family) for c in clips
                            if name in SEALED_SETS or c.split == SEALED_SPLIT}
    for name in POOL_SETS:
        for clip in sets.get(name, ()):
            if clip.split == SEALED_SPLIT or clip.clean_reference is not None or \
                    family_key(clip.family) in sealed_families or not _in_strata(clip) or \
                    not clean_source(clip) or not usable(clip):
                continue
            chosen.setdefault(clip.name, replace(clip, family=family_key(clip.family)))
    return sorted(chosen.values(), key=lambda c: c.name)


def family_key(family: str) -> str:
    """One key per source video, wherever its copy lives (a dataset is often
    under both /mldata/tracking_original and /mldata/downloaded_datasets)."""
    return Path(family).stem.lower().removeprefix("bdd_")


def clean_source(clip: Clip) -> bool:
    """The source-quality rule (see BPP_FLOOR); ungraded clips are not used."""
    if clip.grade == "A":
        return True
    return clip.grade == "B" and clip.downscale >= MIN_POOR_DOWNSCALE and \
        (clip.source_bpp or 0.0) >= BPP_FLOOR


def _in_strata(clip: Clip) -> bool:
    return any(clip.content_class in classes for classes in STRATA.values())


@dataclass(frozen=True)
class Draw:
    """One day's levels: disjoint families, each clip at its rotated start."""
    key: str
    frames: int
    levels: dict[str, list[Clip]]

    def record(self) -> dict:
        """What a result JSON keeps to reproduce the draw."""
        return {"rotation": self.key, "frames": self.frames,
                "levels": {level: [{"clip": c.name, "start_frame": c.start_frame,
                                    "class": c.content_class, "family": c.family}
                                   for c in clips]
                           for level, clips in self.levels.items()}}


def chosen_quick(pool: list[Clip], key: str, path: Path = L1_SETS) -> list[Clip] | None:
    """L1's chosen set for `key` (the sets in turn, by the key's hash), or None when
    no set is recorded or one of its clips has left the pool."""
    sets = json.loads(path.read_text())["sets"] if path.exists() else []
    if not sets:
        return None
    by_name = {c.name: c for c in pool}
    names = sets[_seed("quick", key) % len(sets)]
    return [by_name[n] for n in names] if all(n in by_name for n in names) else None


def draw(pool: list[Clip], key: str, frames: int,
         sizes: dict[str, int | None] = LEVEL_SIZES, l1_sets: Path = L1_SETS) -> Draw:
    """The levels of rotation `key`: stratified, family-disjoint, rotated starts.

    Families are shuffled within each stratum by the key and dealt
    round-robin over their datasets; each level takes families from the
    strata in turn (so every stratum is in every level
    that has room), one clip per family (the family's forms rotate too).
    When L1 has chosen sets (chosen_quick), L1 is the key's set and the
    other levels are dealt from the remaining families.
    """
    quick = chosen_quick(pool, key, l1_sets)
    rng = random.Random(_seed("draw", key))
    families: dict[str, dict[str, list[Clip]]] = {s: {} for s in STRATA}
    for clip in pool:
        families[stratum(clip)].setdefault(clip.family, []).append(clip)
    queues = {}
    for name, members in families.items():
        order = sorted(members)
        rng.shuffle(order)
        order = spread_datasets(order, {f: members[f][0].dataset for f in order})
        queues[name] = [(family, rng.choice(sorted(members[family], key=lambda c: c.name)))
                        for family in order]
    used = {c.family for c in quick or ()}
    levels: dict[str, list[Clip]] = {}
    for level, size in sizes.items():
        if level == QUICK and quick is not None:
            levels[level] = [rotated(c, key, frames) for c in quick]
            continue
        chosen: list[Clip] = []
        while size is None or len(chosen) < size:
            took = False
            for name in STRATA:
                queue = queues[name]
                while queue and queue[0][0] in used:
                    queue.pop(0)
                if queue and (size is None or len(chosen) < size):
                    family, clip = queue.pop(0)
                    used.add(family)
                    chosen.append(rotated(clip, key, frames))
                    took = True
            if not took:
                break
        levels[level] = chosen
    return Draw(key, frames, levels)


def spread_datasets(order: list[str], dataset_of: dict[str, str]) -> list[str]:
    """`order` dealt round-robin over datasets, so no dataset dominates a level."""
    by_dataset: dict[str, list[str]] = {}
    for family in order:
        by_dataset.setdefault(dataset_of[family], []).append(family)
    queues = list(by_dataset.values())
    spread = []
    while any(queues):
        for queue in queues:
            if queue:
                spread.append(queue.pop(0))
    return spread


def rotated(clip: Clip, key: str, frames: int) -> Clip:
    """`clip` from its start frame for `key`: any frame that leaves `frames` to measure."""
    room = clip.frames - frames
    if room <= 0:
        return clip
    start = random.Random(_seed("start", key, clip.name)).randrange(room + 1)
    return replace(clip, start_frame=start)


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\0".join(parts).encode()).digest()[:8], "big")


def pool_summary(pool: list[Clip]) -> str:
    """Pool size by stratum and family count (for the docs and --help)."""
    lines = []
    for name in STRATA:
        members = [c for c in pool if stratum(c) == name]
        lines.append(f"{name}: {len(members)} clips, {len({c.family for c in members})} families")
    return "; ".join(lines)
