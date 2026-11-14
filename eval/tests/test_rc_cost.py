"""nceval.rc_cost: the pairings and their class-balanced BD-J."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nceval import rc_cost


def _curve(rate_scale: float) -> list[dict]:
    return [{"kbps": k * rate_scale, "psnr_yuv": 30.0 + i, "vmaf_v1": 60.0 + 5 * i,
             "ms_ssim": 0.9 + 0.02 * i} for i, k in enumerate((100, 200, 400, 800))]


def test_halving_the_rate_everywhere_is_minus_fifty():
    clips = [SimpleNamespace(name="a", content_class="static"),
             SimpleNamespace(name="b", content_class="pan")]
    curves = {(f, m, c.name): _curve(1.0 if f == "x" else 0.5)
              for f in ("x", "y") for m in ("cbr", "qp") for c in clips}
    result = rc_cost.pairing(curves, clips, ("x", "qp"), ("y", "cbr"))
    assert result["bd_j"] == pytest.approx(-50.0, abs=0.01)
    assert result["clips"] == 2


def test_every_encoder_gets_a_rate_control_cost_line():
    lines = rc_cost.pairings(["nanocodec-av1"])
    costs = {test[0] for line, _, test in lines if line == "rc_cost"}
    assert costs == {"nanocodec-av1", rc_cost.ANCHOR, rc_cost.TUNED_ANCHOR}
    assert all(a[0] == t[0] and (a[1], t[1]) == ("qp", "cbr")
               for line, a, t in lines if line == "rc_cost")
