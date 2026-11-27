"""tools/dev/aws_choose.py: the pick and the estimate, on fixed rows (nothing is queried)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_choose  # noqa: E402
from aws_choose import Row  # noqa: E402

TYPES = ("c7i.24xlarge", "c6i.24xlarge")


def rows(home_free: int, spill_price: float = 1.58) -> list[Row]:
    return [Row("c7i.24xlarge", "eu-north-1", 96, 0.56, home_free),
            Row("c6i.24xlarge", "eu-north-1", 96, None, home_free),
            Row("c7i.24xlarge", "eu-central-1", 96, spill_price, 512)]


def test_home_first_in_the_guide_order():
    chosen, why = aws_choose.pick(rows(200), TYPES, boxes=1)
    assert chosen.instance_type == "c7i.24xlarge" and chosen.region == "eu-north-1"
    assert why == "home region"


def test_expensive_spill_waits_instead():
    """2026-11-24: eu-central-1 c7i.24xlarge spot $1.58/h against $0.56/h at home."""
    chosen, why = aws_choose.pick(rows(50), TYPES, boxes=1)
    assert chosen is None and "2.8x home" in why and "wait" in why


def test_cheap_spill_is_taken():
    chosen, why = aws_choose.pick(rows(50, spill_price=0.70), TYPES, boxes=1)
    assert chosen.region == "eu-central-1" and why.startswith("spill: 1.25x")


def test_estimate_counts_volume_and_ipv4():
    cost = aws_choose.estimate(0.5, hours=2, boxes=3, root_gb=0)
    assert cost == pytest.approx(3 * 2 * (0.5 + aws_choose.PUBLIC_IPV4_USD_PER_HOUR))
