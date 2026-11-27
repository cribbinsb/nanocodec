"""nceval.fixedqp site and the Fixed QP views: kept references, one clip set, the zero line, the metric."""
import pytest

from nceval import site
from nceval.ctc.local import X264_TUNED_QP, x264_variant
from nceval.fixedqp import common_clips, keep_references


def record(families: dict, clips=("a", "b")) -> dict:
    return {"clipset": "set.json", "anchor": "x264-medium", "frames": 150, "points": 4,
            "clips": [{"clip": c} for c in clips], "families": families}


def test_the_references_rows_are_kept_and_ours_never_carried_over():
    new = record({"nanocodec-av1": {"j": -40.0}})
    old = record({"nanocodec-av1": {"j": -30.0}, "nanocodec-av1@old": {"j": -1.0},
                  "svtav1-m8": {"j": -45.0}})
    keep_references(new, old)
    assert new["families"] == {"nanocodec-av1": {"j": -40.0}, "svtav1-m8": {"j": -45.0}}


def test_another_view_is_refused():
    with pytest.raises(SystemExit):
        keep_references(record({}), record({"svtav1-m8": {}}, clips=("a", "c")))


def scored(j):
    return {"class": "c", "quality": j, "quality_low": j}


def test_every_point_of_a_view_is_on_one_clip_set():
    """The CBR charts' rule (nceval.chart.common_clips): a clip one test has no BD-J on
    is left out for every test, and listed with the tests that lacked it."""
    pairs = {"nanocodec-av1 vs x264-medium~psnr": {"per_clip": {"a": scored(-1.0),
                                                                "b": scored(-2.0)}},
             "libvpx-vp9-cpu9 vs x264-medium~psnr": {"per_clip": {"a": scored(3.0),
                                                                  "b": scored(None)}}}
    used, excluded = common_clips(pairs, ["a", "b", "c"])
    assert used == ["a"]
    assert excluded == {"b": ["libvpx-vp9-cpu9"],
                        "c": ["libvpx-vp9-cpu9", "nanocodec-av1"]}


def test_the_zero_line_is_x264_tuned_for_psnr():
    """Results audit 2026-11-23, finding 5: every Fixed QP view's zero line is x264 medium
    --tune zerolatency,psnr (the CBR charts' TUNED_ZERO), not its psychovisual default."""
    assert X264_TUNED_QP == {"x264-medium+psnr": "psnr-qp"}
    assert site.LENS_FAMILIES["x264-medium+psnr"] == site.TUNED_ZERO == "x264-medium~psnr"
    job = {"id": "aom-ld/S/x264-medium/q22", "encoder": "x264-medium", "kind": "qp",
           "point": 22, "bitstream": "streams/aom-ld/S/x264-medium/q22.264",
           "encode": ["bin/x264", "--tune", "zerolatency", "--qp", "22", "-o",
                      "streams/aom-ld/S/x264-medium/q22.264"]}
    [tuned] = x264_variant(job, "psnr-qp")
    assert tuned["encoder"] == "x264-medium+psnr"
    assert tuned["encode"][1:3] == ["--tune", "zerolatency,psnr"]


def test_the_fixed_qp_views_open_on_psnr_yuv_with_j_beside_it():
    """The standards' PSNR-YUV (6:1:1, nceval.metrics) is every Fixed QP view's first,
    default metric, J the second, luma PSNR shown; the lens adds the AOM Weighted."""
    assert site.QP_METRICS[:3] == ("psnr", "j", "psnr_y")
    assert site.LENS_METRICS[:4] == ("psnr", "j", "psnr_y", "weighted")
    explorer = (site.REPO / "web/results/explorer.js").read_text()
    assert "const DEFAULT_METRIC = { cbr: 'j', qp: 'psnr' };" in explorer
    page = (site.REPO / "web/results/fixed-qp.js").read_text()
    assert page.index("psnr_yuv: 'PSNR-YUV") < page.index("j: 'J (")


def test_a_lens_column_on_another_build_is_named_in_the_caption():
    view = {"build_commit": "207745d7 for every nanocodec row", "speed": {},
            "column_builds": {"constant": "feb16365"}}
    quality = site.view_measured(view, {})["quality"]
    assert "for every preset" in quality
    assert "the Constant QP column of the Fixed QP page at" in quality and "feb16365" in quality
