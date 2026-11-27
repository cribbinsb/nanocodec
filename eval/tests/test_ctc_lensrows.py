"""nceval.ctc lens-variants / lens-rows: a release's re-measured lens rows of ours."""
import json
from pathlib import Path

import pytest

from nceval.ctc import lensrows

REPO = Path(__file__).resolve().parents[2]
SITE_LENS = REPO / "web/results/data/fixed-qp-lens.json"


def site_anchor():
    """The site lens data's zero line (rows from a compare against another are ignored)."""
    return json.loads(SITE_LENS.read_text())["rows"][0]["anchor"]


def compare_rows(runs, anchor=None, j=-10.0):
    anchor = anchor or site_anchor()
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
                      ("nanocodec-vp9@new", "rel-profile")} | {
        (f, "rel-constant") for f in ("nanocodec-hevc", "nanocodec-av1", "nanocodec-vp9@new")}
    strict = next(r for r in runs if r["label"] == "rel-strict")
    assert strict["params"] == ["rate.cadence_static_scaling=0"]


def test_every_row_of_ours_in_the_site_data_is_re_measured_and_the_references_kept():
    data = json.loads(SITE_LENS.read_text())
    ours = sorted({r["test"] for r in data["rows"] if r["test"].startswith("nanocodec-")})
    references = [dict(r) for r in data["rows"] if not r["test"].startswith("nanocodec-")]
    written = lensrows.apply(data, compare_rows(lensrows.variants(data, ours)), "abcd1234")
    assert len(written) == sum(r["test"].startswith("nanocodec-") for r in data["rows"])
    assert {(t, "constant") for t in ours} <= {(r["test"], r["column"]) for r in written}
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


def test_the_constant_qp_column_is_the_one_flat_switch_for_every_codec():
    """Constant QP (results audit 2026-11-23, finding 5): ours with every picture and block
    at the given quantizer, the engine's rate.constant_qp (crates/nc-encoder constant_qp.rs),
    the same for every codec; a release codes it for every preset beside the profile."""
    for codec in ("h264", "hevc", "av1", "vp9", "vvc", "av2"):
        assert lensrows.column_params(f"nanocodec-{codec}@fast", "constant") == (
            "rate.constant_qp=1",)
    assert lensrows.RELEASE_COLUMNS == ("profile", "constant")
    assert lensrows.label("nanocodec-av1", "constant") == "rel-constant"


def rebase_rows(anchor="x264-medium+psnr"):
    figures = {"Weighted": -4.0, "J": -8.0, "PSNR-YUV": -6.0, "PSNR-Y": -7.0, "VMAF-v1": -9.0,
               "MS-SSIM": -5.0, "XPSNR-YUV": -3.0, "XPSNR-YUV n": 12}
    return [{"test": t, "anchor": anchor, "n": 11, "fps": 5.0, **figures}
            for t in ("x264-medium", "nanocodec-av1+rel-profile", "hm-ldp")]


def test_a_new_zero_line_rebases_every_row_and_gains_its_own():
    data = reported_data()
    lensrows.rebase(data, rebase_rows(), "x264-medium+psnr")
    rows = {r["test"]: r for r in data["rows"]}
    assert all(r["anchor"] == "x264-medium+psnr" for r in data["rows"])
    assert rows["hm-ldp"]["psnr_yuv"] == -6.0 and rows["hm-ldp"]["psnr_y"] == -7.0
    assert rows["hm-ldp"]["fps"] == 0.353  # the timing is the row's own, not the re-run's
    assert rows["x264-medium"]["j"] == -8.0  # the old zero line is now a row like any
    zero = rows["x264-medium+psnr"]
    assert zero["j"] == zero["psnr_yuv"] == zero["psnr_y"] == 0.0 and zero["fps"] is None


def test_a_rebase_missing_a_row_stops_it():
    data = reported_data()
    with pytest.raises(SystemExit):
        lensrows.rebase(data, rebase_rows()[:2], "x264-medium+psnr")
    assert all(r["anchor"] == "x264-medium" for r in data["rows"])


def test_a_column_on_another_build_is_recorded_for_the_captions():
    data = json.loads(SITE_LENS.read_text())
    ours = sorted({r["test"] for r in data["rows"] if r["test"].startswith("nanocodec-")})
    lensrows.apply(data, compare_rows(lensrows.variants(data, ours)), "207745d7",
                   {"constant": "feb16365"})
    assert data["column_builds"] == {"constant": "feb16365"}
    assert data["build_commit"].split()[0] == "207745d7"
    assert "column constant at feb16365" in data["build_commit"]
    with pytest.raises(SystemExit):
        lensrows.apply(data, compare_rows(lensrows.variants(data, ours)), "x", {"nope": "y"})


def test_a_rebase_compare_listing_the_references_adds_no_column_to_them():
    """A re-base compares every curve, references included: only our families gain the
    release columns (constant), never a reference."""
    data = json.loads(SITE_LENS.read_text())
    ours = sorted({r["test"] for r in data["rows"] if r["test"].startswith("nanocodec-")})
    references = [{"test": r["test"], "anchor": site_anchor(), "n": 12, "fps": 1.0,
                   "Weighted": 1.0, "J": 1.0, "PSNR-YUV": 1.0, "VMAF-v1": 1.0, "MS-SSIM": 1.0}
                  for r in data["rows"] if not r["test"].startswith("nanocodec-")]
    written = lensrows.apply(data, compare_rows(lensrows.variants(data, ours)) + references,
                             "abcd1234")
    assert all(r["test"].startswith("nanocodec-") for r in written)


def test_a_rebase_without_xpsnr_leaves_the_zero_row_without_it():
    data = reported_data()
    rows = [{k: v for k, v in r.items() if k != "XPSNR-YUV"} for r in rebase_rows()]
    lensrows.rebase(data, rows, "x264-medium+psnr")
    zero = next(r for r in data["rows"] if r["test"] == "x264-medium+psnr")
    assert all(r["xpsnr"] is None for r in data["rows"]) and zero["xpsnr_clips"] is None
