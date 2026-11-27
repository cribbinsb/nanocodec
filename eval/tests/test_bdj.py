"""nceval.bdj's curve rules (BP-3 section 9, BP-3.14): two points make a curve, any shared
quality range gives a BD, and a point without a metric is in no curve of that metric."""
import math

import pytest

from nceval import bdj


def curve(rates, saving=0.0, drop=None):
    """Points on one quality-against-log-rate line, coded at `saving` less rate; `drop`
    names (index, metric) pairs whose value is missing."""
    points = []
    for i, kbps in enumerate(rates):
        q = math.log(kbps)
        values = {"psnr_yuv": 30 + 4 * q, "vmaf_v1": 10 * q, "ms_ssim": 1 - 0.5 / kbps}
        for index, metric in drop or ():
            if index == i:
                values[metric] = None
        points.append((kbps * (1 - saving), values))
    return points


def test_two_points_make_a_curve_and_one_does_not():
    """Owner, 2026-11-25, BP-3.14: the minimum is 2 points after the envelope."""
    assert bdj.MIN_POINTS == 2
    anchor = curve([300, 500, 800, 1200])
    assert bdj.bd_j(curve([400, 900], saving=0.3), anchor) == pytest.approx(-30, abs=0.5)
    assert bdj.bd_j(curve([400], saving=0.3), anchor) is None
    with pytest.raises(bdj.CurveRefusal, match="1 rising point of 1 coded"):
        bdj.bd(curve([400]), anchor, "psnr_yuv")


def test_a_thin_overlap_gives_a_bd_with_its_fraction():
    """BP-3.14: no minimum overlap; the fraction is returned for the report to show."""
    anchor = curve([300, 500, 800, 1200])
    result = bdj.bd(curve([1100, 2000, 4000], saving=0.2), anchor, "psnr_yuv")
    assert result.bd_rate_percent == pytest.approx(-20, abs=0.5)
    assert 0 < result.overlap_fraction < bdj.THIN_OVERLAP_FRACTION
    assert bdj.bd_j(curve([1100, 2000, 4000], saving=0.2), anchor) == pytest.approx(-20, abs=0.5)


def test_ranges_that_do_not_touch_give_no_bd_and_say_which_side():
    anchor = curve([300, 500, 800, 1200])
    with pytest.raises(bdj.DisjointRanges, match="test curve lies above"):
        bdj.bd(curve([1300, 2000]), anchor, "vmaf_v1")
    with pytest.raises(bdj.DisjointRanges, match="test curve lies below"):
        bdj.bd(curve([100, 250]), anchor, "vmaf_v1")
    assert bdj.bd_j(curve([1300, 2000]), anchor) is None


def test_a_point_without_a_metric_is_out_of_that_metrics_curve_only():
    """The stored "ours" CBR runs hold points whose pooled metric is None: no crash, and the
    point still serves the metrics it has."""
    anchor = curve([300, 500, 800, 1200])
    test = curve([300, 500, 800, 1200], saving=0.1, drop=[(1, "vmaf_v1")])
    assert bdj.has_metric(test, "vmaf_v1") == [0, 2, 3]
    assert bdj.envelope(test, "vmaf_v1") == [0, 2, 3]
    assert bdj.envelope(test, "psnr_yuv") == [0, 1, 2, 3]
    assert len(bdj.rd_curve(test, "vmaf_v1")[0]) == 3
    assert len(bdj.rd_curve(test, "vmaf_v1", use_envelope=False)[0]) == 3
    assert bdj.bd_j(test, anchor) == pytest.approx(-10, abs=0.5)
    lone = curve([300, 500], drop=[(0, "ms_ssim")])
    with pytest.raises(bdj.CurveRefusal, match=r"1 rising point of 2 coded \(1 without the metric\)"):
        bdj.rd_curve(lone, "ms_ssim")


def test_the_envelope_leaves_no_fall_to_refuse():
    """A fall is refused only without the envelope: with it the falling point is dominated."""
    falling = [(300, {"psnr_yuv": 35.0}), (500, {"psnr_yuv": 36.0}), (800, {"psnr_yuv": 34.0}),
               (1200, {"psnr_yuv": 38.0})]
    assert bdj.rd_curve(falling, "psnr_yuv")[0] == [300, 500, 1200]
    with pytest.raises(bdj.CurveRefusal, match="falls"):
        bdj.rd_curve(falling, "psnr_yuv", use_envelope=False)
