"""nceval.tune_spsa's sampling: stratified draws, one sample per clip and share, disjoint
halves and SPRT rounds that never reuse a sample."""

import random

from nceval import tune_spsa
from nceval.tune_spsa import Sample


def candidates():
    out = []
    for stratum in ("crowd", "static", "pan"):
        for i in range(6):
            for period in (150, 300):
                for start in (0, 900):
                    out.append((Sample(f"{stratum}{i}", period, start), stratum))
    return out


def test_draw_is_stratified_and_one_per_clip_and_share():
    chosen = tune_spsa.draw(candidates(), 9, random.Random(1))
    assert len(chosen) == 9
    assert len({(s.clip, s.period) for s in chosen}) == 9
    strata = [s.clip.rstrip("0123456789") for s in chosen]
    assert {strata.count(x) for x in set(strata)} == {3}


def test_draw_leaves_out_excluded_samples():
    first = tune_spsa.draw(candidates(), 12, random.Random(2))
    second = tune_spsa.draw(candidates(), 12, random.Random(3), {s.key for s in first})
    assert not {s.key for s in first} & {s.key for s in second}


def test_halves_split_families():
    families = [f"family{i}" for i in range(200)]
    halves = [tune_spsa.half_of(f, "seed") for f in families]
    assert 70 < halves.count(tune_spsa.STEER) < 130
    assert halves == [tune_spsa.half_of(f, "seed") for f in families]


def test_replace_option():
    assert tune_spsa.replace_option(["--a", "1", "--iterations", "3", "--b"], "--iterations",
                                    "6") == ["--a", "1", "--b", "--iterations", "6"]
