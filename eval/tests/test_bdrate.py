"""Sanity tests for BD-rate: known shifts must come back exactly."""

import math

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nceval.bdrate import bd_rate  # noqa: E402

RATES = [150.0, 220.0, 320.0, 470.0, 680.0, 1000.0]
# A plausible concave PSNR curve.
QUALITY = [30.0 + 4.0 * math.log2(r / 150.0) for r in RATES]


def test_identical_curves_give_zero():
    result = bd_rate(RATES, QUALITY, RATES, QUALITY)
    assert result.reliable
    assert abs(result.bd_rate_percent) < 1e-9


def test_uniform_rate_scaling_is_recovered():
    # Test codec reaches the same quality with 70% of the bits: -30%.
    test_rates = [r * 0.7 for r in RATES]
    result = bd_rate(RATES, QUALITY, test_rates, QUALITY)
    assert abs(result.bd_rate_percent - (-30.0)) < 1e-6


def test_disjoint_curves_are_flagged():
    shifted = [q + 100.0 for q in QUALITY]
    result = bd_rate(RATES, QUALITY, RATES, shifted)
    assert result.bd_rate_percent is None
    assert not result.reliable


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")


def test_low_half_measures_only_the_low_rate_region():
    # Test saves 20% at low rates and nothing at high rates.
    rates = [100.0, 200.0, 400.0, 800.0, 1600.0]
    quality = [30.0, 33.0, 36.0, 39.0, 42.0]
    test_rates = [100.0 * 0.8, 200.0 * 0.8, 400.0 * 0.9, 800.0, 1600.0]
    full = bd_rate(rates, quality, test_rates, quality)
    low = bd_rate(rates, quality, test_rates, quality, low_half=True)
    assert low.reliable and full.reliable
    assert low.bd_rate_percent < full.bd_rate_percent < 0
    assert low.bd_rate_percent == pytest.approx(-17.5, abs=2.5)


def test_low_half_equals_full_for_a_uniform_gain():
    rates = [100.0, 200.0, 400.0, 800.0]
    quality = [30.0, 33.0, 36.0, 39.0]
    test = [r * 0.9 for r in rates]
    low = bd_rate(rates, quality, test, quality, low_half=True).bd_rate_percent
    assert low == pytest.approx(-10.0, abs=0.01)
