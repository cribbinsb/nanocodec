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
