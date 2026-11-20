"""Cached CBR points measured under the old 100 dB PSNR cap (cbr.under_psnr_cap)."""

from nceval import cbr
from nceval.framepipe import PSNR_CAP_DB


def legacy(luma: list[float], chroma: list[float]) -> dict:
    """An old real-use result: per-frame Y and 6:1:1 PSNR, U = V = `chroma`, no cap field."""
    yuv = [(6 * y + 2 * c) / 8 for y, c in zip(luma, chroma)]
    return {"psnr_yuv": sum(yuv) / len(yuv), "frame_psnr_y": luma, "frame_psnr_yuv": yuv,
            "gate_level": "unchecked"}


def test_an_ordinary_point_is_kept_unchanged():
    old = legacy([38.0, 41.5], [42.0, 49.9])
    kept = cbr.under_psnr_cap(old)
    assert kept == {**old, "psnr_cap_db": PSNR_CAP_DB}


def test_a_grayscale_point_is_re_measured():
    # flat chroma coded exactly scored the old 100 dB cap: 40 dB above the new one
    assert cbr.under_psnr_cap(legacy([38.0, 39.0], [100.0, 100.0])) is None
    # one frame over the threshold is enough
    assert cbr.under_psnr_cap(legacy([38.0, 39.0], [45.0, cbr.RECAP_MAX_CHROMA_DB + 1])) is None


def test_near_lossless_luma_and_points_without_per_frame_luma_are_re_measured():
    assert cbr.under_psnr_cap(legacy([61.0, 40.0], [45.0, 45.0])) is None
    classic = legacy([38.0], [42.0])
    del classic["frame_psnr_y"]
    assert cbr.under_psnr_cap(classic) is None


def test_a_base_layer_must_pass_too():
    point = {**legacy([38.0], [42.0]), "base": legacy([35.0], [100.0])}
    assert cbr.under_psnr_cap(point) is None
    point["base"] = legacy([35.0], [40.0])
    assert cbr.under_psnr_cap(point)["base"]["psnr_cap_db"] == PSNR_CAP_DB


def test_a_point_of_the_current_cap_is_returned_as_is():
    current = {**legacy([38.0], [70.0]), "psnr_cap_db": PSNR_CAP_DB}
    assert cbr.under_psnr_cap(current) is current
    assert cbr.under_psnr_cap(None) is None
