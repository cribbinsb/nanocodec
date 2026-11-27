"""nceval.spsa on synthetic objectives with known noise: SPSA recovers a real optimum through
noise the size of the encoder's, and the SPRT adopts a real gain and rejects noise."""

import random
import statistics

from nceval import spsa

#: Per-clip J spread of a 600-frame trial (docs/handoff/stop-network.md): about 0.9 %, so a
#: 12-clip evaluation has a standard error of about 0.26 %.
CLIP_SD = 0.9
CLIPS = 12


def noisy(true_value, rng, clips=CLIPS):
    """A mean over `clips` clips of a true value plus per-clip noise."""
    return true_value + statistics.fmean(rng.gauss(0, CLIP_SD) for _ in range(clips))


def run_spsa(objective, u0, iterations, seed, c=0.2, first_step=0.3, first_gradient=1.0):
    rng = random.Random(seed)
    gains = spsa.Gains.plan(iterations, c, first_step, first_gradient)
    u = list(u0)
    for k in range(iterations):
        delta = spsa.direction(rng, len(u))
        plus, minus = spsa.sides(u, delta, gains.perturbation(k))
        # Common random numbers: both sides see the same clips (the same noise draw).
        clip_rng = random.Random(rng.random())
        shared = statistics.fmean(clip_rng.gauss(0, CLIP_SD) for _ in range(CLIPS))
        own = lambda: rng.gauss(0, CLIP_SD * 0.5 / CLIPS ** 0.5)
        u = spsa.update(u, delta, objective(plus) + shared + own(),
                        objective(minus) + shared + own(), k, gains)
    return u


def test_spsa_finds_a_real_optimum_through_noise():
    target = [0.5, -0.3]
    # A quadratic bowl: 1.5 % of J worse at a normalised distance of 1.
    objective = lambda u: 1.5 * sum((x - t) ** 2 for x, t in zip(u, target))
    finals = [run_spsa(objective, [-0.5, 0.5], 40, seed) for seed in range(20)]
    errors = [max(abs(x - t) for x, t in zip(u, target)) for u in finals]
    assert statistics.median(errors) < 0.25


def test_spsa_stays_near_the_start_on_a_flat_objective():
    finals = [run_spsa(lambda u: 0.0, [0.0], 40, seed) for seed in range(20)]
    # No signal: the walk is the noise's alone (median about 0.37 of the half-range here):
    # SPSA does not tell a flat objective from a shallow one, which is why adoption is the
    # SPRT's on unused clips, never the walk's end point.
    assert statistics.median(abs(u[0]) for u in finals) < 0.5


def test_spsa_steps_follow_the_gradient():
    gains = spsa.Gains(a=0.1, c=0.1, stability=0.0)
    u = spsa.update([0.0, 0.0], [1, -1], 1.0, 0.0, 0, gains)
    assert u[0] < 0 < u[1]


def sprt_run(true_gain, seed, delta=0.5, max_samples=400):
    rng = random.Random(seed)
    test = spsa.Sprt(delta=delta)
    while len(test.samples) < max_samples:
        test.add([rng.gauss(-true_gain, CLIP_SD) for _ in range(4)])
        if test.decision():
            return test.decision(), len(test.samples)
    return None, len(test.samples)


def test_sprt_adopts_a_real_gain():
    outcomes = [sprt_run(0.5, seed) for seed in range(100)]
    adopted = sum(d == "adopt" for d, _ in outcomes)
    assert adopted >= 90
    assert statistics.median(n for _, n in outcomes) <= 80


def test_sprt_rejects_noise_and_losses():
    for gain in (0.0, -0.3):
        outcomes = [sprt_run(gain, seed) for seed in range(100)]
        assert sum(d == "adopt" for d, _ in outcomes) <= 8
        assert sum(d == "reject" for d, _ in outcomes) >= 85


def test_sprt_waits_for_samples():
    test = spsa.Sprt(delta=0.5)
    test.add([-5.0] * (spsa.MIN_SAMPLES - 1))
    assert test.decision() is None
    test.add([-5.0])
    assert test.decision() == "adopt"


def test_units_round_trip():
    assert spsa.to_value(spsa.to_unit(0.25, -1.0, 3.0), -1.0, 3.0) == 0.25
    assert spsa.to_unit(9.0, -1.0, 3.0) == 1.0
