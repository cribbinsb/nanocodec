"""nceval.ctc lens-variants / lens-rows: a release's re-measured lens rows of ours."""
import json
from pathlib import Path

import pytest

from nceval.ctc import lensrows

REPO = Path(__file__).resolve().parents[2]
SITE_LENS = REPO / "web/results/data/fixed-qp-lens.json"


def compare_rows(runs, anchor="x264-medium", j=-10.0):
    return [{"test": f"{r['encoder']}+{r['label']}", "anchor": anchor, "n": 12, "fps": 123.4567,
             "Weighted": -5.04, "J": j, "PSNR-YUV": -1.0, "VMAF-v1": -2.0, "MS-SSIM": -3.0}
            for r in runs]


def test_a_column_with_the_profiles_parameters_reuses_its_curve():
    data = {"rows": [{"test": "nanocodec-hevc", "column": c, "anchor": "x264-medium"}
                     for c in ("strict", "profile", "cbr_settings")]
            + [{"test": "nanocodec-av1", "column": "strict", "anchor": "x264-medium"}]}
    runs = lensrows.variants(data, ["nanocodec-hevc", "nanocodec-av1", "nanocodec-vp9@new"])
    labels = {(r["encoder"], r["label"]) for r in runs}
    assert labels == {("nanocodec-hevc", "rel-profile"), ("nanocodec-hevc", "rel-cbr_settings"),
                      ("nanocodec-av1", "rel-profile"), ("nanocodec-av1", "rel-strict"),
                      ("nanocodec-vp9@new", "rel-profile")}
    strict = next(r for r in runs if r["label"] == "rel-strict")
    assert strict["params"] == ["rate.cadence_static_scaling=0"]


def test_every_row_of_ours_in_the_site_data_is_re_measured_and_the_references_kept():
    data = json.loads(SITE_LENS.read_text())
    ours = sorted({r["test"] for r in data["rows"] if r["test"].startswith("nanocodec-")})
    references = [dict(r) for r in data["rows"] if not r["test"].startswith("nanocodec-")]
    written = lensrows.apply(data, compare_rows(lensrows.variants(data, ours)), "abcd1234")
    assert len(written) == sum(r["test"].startswith("nanocodec-") for r in data["rows"])
    assert all(r["j"] == -10.0 and r["fps"] == 123.457 and r["weighted"] == -5.0
               for r in written)
    assert [r for r in data["rows"] if not r["test"].startswith("nanocodec-")] == references
    assert data["build_commit"].startswith("abcd1234")


def test_a_row_left_unmeasured_stops_it():
    data = json.loads(SITE_LENS.read_text())
    runs = lensrows.variants(data, ["nanocodec-av1"])
    with pytest.raises(SystemExit):
        lensrows.apply(data, compare_rows(runs), "abcd1234")


def test_the_csv_keeps_its_line_endings_and_other_rows():
    text = "test,column,j\r\nx264-medium,profile,0.0\r\nnanocodec-av1,profile,-1.0\r\n"
    out = lensrows.rewrite_csv(text, [{"test": "nanocodec-av1", "column": "profile", "j": -2.0},
                                      {"test": "nanocodec-av9", "column": "profile", "j": -3.0}])
    assert out == ("test,column,j\r\nx264-medium,profile,0.0\r\nnanocodec-av1,profile,-2.0\r\n"
                   "nanocodec-av9,profile,-3.0\r\n")


def reported_data():
    return {"rows": [{"test": "x264-medium", "column": "profile", "anchor": "x264-medium", "n": 12,
                      "fps": 78.1, "weighted": 0.0, "j": 0.0, "psnr_yuv": 0.0, "vmaf": 0.0,
                      "ms_ssim": 0.0},
                     {"test": "nanocodec-av1", "column": "profile", "anchor": "x264-medium",
                      "n": 12, "fps": 81.0, "weighted": -5.0, "j": -10.0, "psnr_yuv": -1.0,
                      "vmaf": -2.0, "ms_ssim": -3.0},
                     {"test": "hm-ldp", "column": "profile", "anchor": "x264-medium", "n": 12,
                      "fps": 0.353, "weighted": -47.8, "j": -48.6, "psnr_yuv": -48.3,
                      "vmaf": -42.4, "ms_ssim": -48.9}]}


def reported_rows(j=-10.0):
    figures = {"Weighted": -5.04, "J": j, "PSNR-YUV": -1.0, "VMAF-v1": -2.0, "MS-SSIM": -3.0}
    return [{"test": "x264-medium", "anchor": "x264-medium", "n": 12, "fps": 99.0,
             **{k: 0.0 for k in figures}, "XPSNR-YUV": 0.0},
            {"test": "nanocodec-av1+rel-profile", "anchor": "x264-medium", "n": 12, "fps": 99.0,
             **figures, "XPSNR-YUV": -17.04, "XPSNR-YUV n": 11}]


def test_reported_only_adds_xpsnr_where_the_rerun_reproduces_and_keeps_the_rest():
    data = reported_data()
    written = lensrows.apply_reported(data, reported_rows())
    assert [r["test"] for r in written] == ["x264-medium", "nanocodec-av1"]
    rows = {r["test"]: r for r in data["rows"]}
    assert rows["nanocodec-av1"]["xpsnr"] == -17.0 and rows["nanocodec-av1"]["fps"] == 81.0
    assert rows["nanocodec-av1"]["xpsnr_clips"] == 11
    assert rows["x264-medium"]["xpsnr"] == 0.0
    assert "xpsnr" not in rows["hm-ldp"]  # no re-run: left alone


def test_reported_only_refuses_a_rerun_that_did_not_reproduce():
    data = reported_data()
    with pytest.raises(SystemExit):
        lensrows.apply_reported(data, reported_rows(j=-9.8))
    assert all("xpsnr" not in r for r in data["rows"])


def test_the_csv_gains_the_reported_column():
    text = "test,column,j\nx264-medium,profile,0.0\nhm-ldp,profile,-48.6\n"
    out = lensrows.rewrite_csv(text, [{"test": "x264-medium", "column": "profile", "j": 0.0,
                                       "xpsnr": 0.0, "xpsnr_clips": 12}])
    assert out == ("test,column,j,xpsnr,xpsnr_clips\nx264-medium,profile,0.0,0.0,12\n"
                   "hm-ldp,profile,-48.6,,\n")


def test_a_remeasured_reference_takes_the_rerun_figures():
    data = reported_data()
    rerun = [{"test": "hm-ldp", "anchor": "x264-medium", "n": 12, "fps": 1.0, "Weighted": -47.8,
              "J": -48.6, "PSNR-YUV": -48.0, "VMAF-v1": -42.4, "MS-SSIM": -48.9,
              "XPSNR-YUV": -52.7}]
    with pytest.raises(SystemExit):
        lensrows.apply_reported(reported_data(), rerun)
    lensrows.apply_reported(data, rerun, frozenset({"hm-ldp"}))
    hm = next(r for r in data["rows"] if r["test"] == "hm-ldp")
    assert hm["psnr_yuv"] == -48.0 and hm["xpsnr"] == -52.7 and hm["fps"] == 0.353


def test_a_figure_one_rounding_unit_away_still_reproduces():
    data = reported_data()
    lensrows.apply_reported(data, reported_rows(j=-9.95))
    av1 = next(r for r in data["rows"] if r["test"] == "nanocodec-av1")
    assert av1["j"] == -10.0 and av1["xpsnr"] == -17.0
