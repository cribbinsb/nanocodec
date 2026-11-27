"""Source-grouped train / held-out splits, with the leakage checks.

- **Groups are whole sources.** Every shard belongs to a source group: its dataset for
  camera datasets (one camera network or recording campaign: bdd100k_mot, jaad, otw, ...),
  or its source family for catalogues of unrelated sequences (CATALOGUES: xiph, svt, ...).
  Groups that share a family or a clip file hash are merged, so the same source under
  another name, size or copy can never sit on both sides.
- **The held-out side is whole groups**, drawn by a seed until it holds about
  `held_out_share` of the families, content class by content class so every class is
  represented on both sides where it can be.
- **Sealed sources are refused.** A shard whose family belongs to a sealed set (adv12,
  the independent set, VC-1K's measurement set, main's TEST split, the CTC sequences;
  nceval.pool) raises LeakageError: they never enter the dataset at all.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from ..clips import load_clipset
from ..config import load_paths
from ..pool import SEALED_FAMILIES, SEALED_SETS, SEALED_SPLIT, family_key
from .shard import Shard

#: Datasets that are catalogues of unrelated sequences: grouped per source family.
CATALOGUES = ("xiph", "svt", "pristine", "uvg_vcm", "video", "raw_movies")
HELD_OUT_SHARE = 0.2
TRAIN, HELD_OUT = "train", "heldout"


class LeakageError(ValueError):
    pass


def sealed_families() -> set[str]:
    """The family of every sealed clip (nceval.pool's rule)."""
    paths, families = load_paths(), set(SEALED_FAMILIES)
    for path in paths.clips.glob("*.json"):
        try:
            clips = load_clipset(path, paths.data_root, exclusion_only=True).clips
        except (KeyError, TypeError, ValueError):
            continue  # not a clip set file
        families |= {family_key(c.family) for c in clips
                     if path.stem in SEALED_SETS or c.split == SEALED_SPLIT}
    return families


def group_key(clip: dict) -> str:
    dataset = clip.get("dataset") or ""
    if not dataset or dataset.split("/")[0] in CATALOGUES:
        return f"family:{clip['family']}"
    return f"dataset:{dataset}"


def check_sealed(clips: Iterable[dict], sealed: set[str]) -> None:
    leaked = sorted({c["name"] for c in clips if family_key(c["family"]) in sealed})
    if leaked:
        raise LeakageError(f"sealed sources in the data: {leaked}")


def groups(clips: Iterable[dict]) -> dict[str, str]:
    """Clip name -> source group, merging groups linked by a family or a file hash."""
    clips = list(clips)
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        while parent.setdefault(key, key) != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for clip in clips:
        for link in (f"family:{clip['family']}", f"sha:{clip['sha256']}"):
            parent[find(link)] = find(group_key(clip))
    return {clip["name"]: find(group_key(clip)) for clip in clips}


def split(clips: Iterable[dict], seed: str, held_out_share: float = HELD_OUT_SHARE,
          sealed: set[str] | None = None) -> dict[str, str]:
    """Clip name -> TRAIN or HELD_OUT, by whole source groups (module docs)."""
    clips = {c["name"]: c for c in clips}.values()
    check_sealed(clips, sealed_families() if sealed is None else sealed)
    of = groups(clips)
    members: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        members[of[clip["name"]]].append(clip)
    by_class: dict[str, list[str]] = defaultdict(list)
    for group, inside in members.items():
        main_class = max({c["content_class"] for c in inside},
                         key=lambda k: sum(c["content_class"] == k for c in inside))
        by_class[main_class].append(group)
    rng = random.Random(seed)
    held = set()
    for content_class in sorted(by_class):
        candidates = sorted(by_class[content_class])
        rng.shuffle(candidates)
        families = {g: len({c["family"] for c in members[g]}) for g in candidates}
        total, taken = sum(families.values()), 0
        # Whole groups until the share is reached; a group that would overshoot it is
        # passed over (one at least twice the share is never held out), and one group of
        # the class always stays in training.
        target = held_out_share * total
        for group in candidates[:-1]:
            size = families[group]
            if taken + size > target and (taken > 0 or size > 2 * target):
                continue
            held.add(group)
            taken += size
    return {name: HELD_OUT if group in held else TRAIN for name, group in of.items()}


def check_split(clips: Iterable[dict], sides: dict[str, str]) -> None:
    """No group, family or file on both sides."""
    clips = list(clips)
    for key in ("family", "sha256"):
        seen: dict[str, set[str]] = defaultdict(set)
        for clip in clips:
            seen[clip[key]].add(sides[clip["name"]])
        crossed = sorted(k for k, s in seen.items() if len(s) > 1)
        if crossed:
            raise LeakageError(f"{key} on both sides: {crossed[:5]}")


def split_of(shards: list[Shard], seed: str) -> dict[str, str]:
    """Clip name -> TRAIN / HELD_OUT by whole source groups (module docs)."""
    clips = [s.meta["job"]["clip_meta"] for s in shards]
    sides = split(clips, seed)
    check_split(clips, sides)
    return sides
