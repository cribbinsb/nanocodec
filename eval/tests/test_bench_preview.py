"""tools/bench/preview.py: a benchmark report as the results pages' data, figures unchanged."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
import preview as bench_site  # noqa: E402

SERIES = "c8i"
BINARIES = {"nanocodec": "aa", "x264": "bb", "x265": "cc"}
COMMIT = "0123456789abcdef0123456789abcdef01234567"
#: A report's provenance: ours at COMMIT and x264 checked; x265 ran but is not recorded.
ENCODERS = {
    "nanocodec": {"commit": COMMIT, "version": f"nanocodec 9.9.9 {COMMIT}", "sha256": "aa",
                  "build": "PGO", "store": "x86_64-opt-bp4"},
    "x264": {"ref": "v1", "commit": COMMIT, "lto": "yes", "pgo": "yes", "sha256": "bb",
             "store": "x86_64-opt-bp4"},
}


def bd(mean):
    return {m: {"mean": mean, "low": mean - 1, "high": mean + 1} for m in bench_site.METRICS}


def row(mean, own_mean=None, drop=False, clips=3):
    return {"clips": clips, "own_clips": clips + 1, "missing": {"c": "3 points"},
            "headline_set": "own clips (+drop row)" if drop else "common set",
            "bd": bd(mean), "bd_own_set": bd(mean if own_mean is None else own_mean),
            "speed_mpix_per_cpu_s": 50.0, "speed_raw_mpix_per_cpu_s": 51.0, "speed_clips": clips,
            "compliance": None,
            "exclusion_check": {"missing_clips": 1, "ours_bd_j_on_missing": float("nan")}}


def write_report(reports, dataset, families, view_clips=10, common_clips=8, series=SERIES):
    path = bench_site.report_path(reports, dataset, series)
    path.parent.mkdir(parents=True, exist_ok=True)
    zero = {"j": "x264-medium", "vmaf_v1": "x264-medium", "psnr_yuv": "x264-medium~psnr",
            "ms_ssim": "x264-medium~ssim"}
    path.write_text(json.dumps({
        "manifests": [{"procedure_commit": "abc", "phase": {"INIT_MS": 1000},
                       "phase_name": "interim", "binaries": BINARIES}],
        "provenance": {"encoders": ENCODERS, "procedure": {"report": "BP-4.0"}},
        "series": {series: {"view_clips": view_clips, "common_clips": common_clips,
                            "families": families, "zero_line": zero, "front": ["x264-medium"]}},
        "speed_provisional": False, "envelope": True, "envelope_dropped": {},
        "v6_failures": [], "refused_boxes": {}, "quality_series": "c8a",
        "timing_series": series}))


FAMILIES = {"x264-medium": row(0.0), "x264-medium~psnr": row(-5.0), "nanocodec-av1@fast": row(-30.0, -28.0),
            "libvpx-vp9-cpu7+drop": row(4.0, drop=True), "openh264-baseline": row(20.0),
            "openh264-baseline~aq": row(20.0)}


def test_a_report_becomes_a_view_with_its_common_set_figures(tmp_path):
    write_report(tmp_path, "ours", FAMILIES)
    data = bench_site.site_data(tmp_path)
    (view,) = data["views"]
    ours = view["values"]["nanocodec-av1@fast"]
    assert ours["bd"]["j"] == {"mean": -30.0, "low": -31.0, "high": -29.0}
    assert (ours["clips"], ours["own_clips"], ours["lost"], ours["own_only"]) == (3, 4, 1, False)
    assert ours["exclusion_check"]["ours_bd_j_on_missing"] is None  # NaN is not JSON
    assert (view["machine"], view["procedure"]) == (SERIES, "BP-4.0")
    json.dumps(data, allow_nan=False)


def test_a_drop_row_is_over_its_own_clips(tmp_path):
    write_report(tmp_path, "ours", FAMILIES | {"libvpx-vp9-cpu7+drop": row(4.0, own_mean=6.0, drop=True)})
    (view,) = bench_site.site_data(tmp_path)["views"]
    drop = view["values"]["libvpx-vp9-cpu7+drop"]
    assert (drop["bd"]["j"]["mean"], drop["clips"], drop["own_only"]) == (6.0, 4, True)


def test_both_machines_are_views_and_missing_reports_are_listed_not_invented(tmp_path):
    write_report(tmp_path, "ours", FAMILIES)
    write_report(tmp_path, "vc1k-out-of-sample", FAMILIES, series="c8g")
    data = bench_site.site_data(tmp_path)
    assert {v["id"] for v in data["views"]} == {"ours-c8i", "vc1k-c8g"}
    assert {m["id"] for m in data["missing"]} == {
        "vc1k-c8i", "aomctc-c8i", "ours-c8g", "aomctc-c8g",
        "fast-c8i", "fast-c8g", "fast-mac-m4", "fast-jetson-a78ae"}
    assert list(data["machines"])[:2] == ["c8i", "c8g"]  # x86 first: the pages open on it


def test_the_slow_platforms_have_only_the_fast_subset(tmp_path):
    write_report(tmp_path, "fast", FAMILIES, series="mac-m4")
    data = bench_site.site_data(tmp_path)
    assert [v["id"] for v in data["views"]] == ["fast-mac-m4"]
    assert not any(m["machine"] == "mac-m4" and m["dataset"] != "fast" for m in data["missing"])
    assert data["datasets"]["fast"]["machines"] == ["c8i", "c8g", "mac-m4", "jetson-a78ae"]
    assert data["datasets"]["ours"]["machines"] == ["c8i", "c8g"]
    assert list(bench_site.machines(bench_site.AWS_MACHINES)) == ["c8i", "c8g"]  # history, SVC


def test_identically_measured_twins_are_shown_once(tmp_path):
    write_report(tmp_path, "ours", FAMILIES)
    (view,) = bench_site.site_data(tmp_path)["views"]
    assert view["same_streams"] == {"openh264-baseline~aq": "openh264-baseline"}
    assert "openh264-baseline~aq" not in view["values"] and "x264-medium~psnr" in view["values"]


@pytest.mark.parametrize("family, owner, codec, tuned, drop", [
    ("nanocodec-vvc@veryslow", "nanocodec", "vvc", False, False),
    ("x265-fast~ssim", "x265", "hevc", True, False),
    ("openh264-baseline~cabac", "openh264", "h264", False, False),
    ("libaom-av1-cpu10+drop", "libaom", "av1", False, True),
    ("svtav1-m12", "SVT-AV1", "av1", False, False),
])
def test_families_are_described(family, owner, codec, tuned, drop):
    p = bench_site.describe(family)
    assert (p["owner"], p["codec"], p["metric_tuned"], p["drop"]) == (owner, codec, tuned, drop)


def test_the_preview_directory_is_self_contained(tmp_path):
    write_report(tmp_path / "reports", "ours", FAMILIES)
    index = bench_site.write_site(bench_site.site_data(tmp_path / "reports"), tmp_path / "out")
    assert index.is_file() and (index.parent / "data" / bench_site.DATA_NAME).is_file()
    assert all((index.parent / name).is_file() for name in bench_site.SHARED_ASSETS)


def test_the_build_is_the_reports_checked_provenance():
    build = bench_site.build_identity(ENCODERS, set(BINARIES))
    assert build["nanocodec"]["commit"] == COMMIT[:bench_site.COMMIT_SHOWN]
    assert build["nanocodec"]["version"] == "nanocodec 9.9.9" and build["nanocodec"]["build"] == "PGO"
    versions = {r["name"]: r["version"] for r in build["references"]}
    assert versions == {"x264": "v1", "x265": None}  # x265 ran unrecorded; libvpx did not run
    assert build["record"] == "x86_64-opt-bp4"


def test_a_report_without_provenance_is_not_recorded():
    none = bench_site.build_identity({}, set(BINARIES))
    assert none["nanocodec"] is None and all(r["version"] is None for r in none["references"])


HISTORY_FAMILIES = {
    "x264-medium": row(0.0), "x264-medium~psnr": row(-5.0), "libvpx-vp9-cpu7+drop": row(4.0, drop=True),
    "nanocodec-av1@fast~2026-11-10": row(-12.0), "nanocodec-av1@fast~2026-11-09": row(-10.0),
    "nanocodec-av1@slow~2026-11-09": row(-20.0), "nanocodec-av1@fast~2026-11-09~k600": row(-11.0),
    "nanocodec-av1@fast~2026-11-26~k600": row(-15.0),
    "nanocodec-av1@medium~2026-11-10": {"clips": 0, "missing": {}},  # no well-formed curve
}


def test_history_frames_are_the_reports_days_per_key_interval(tmp_path):
    write_report(tmp_path, bench_site.preview_history.REPORT_NAME, HISTORY_FAMILIES)
    days = tmp_path / "days.json"
    days.write_text(json.dumps([{"day": d, "commit": c * 9, "codecs": "av1"}
                                for d, c in (("2026-11-09", "a"), ("2026-11-10", "b"))]))
    data = bench_site.preview_history.history_data(
        tmp_path, days, tmp_path / "no-captions.json", list(bench_site.AWS_MACHINES), {},
        bench_site.describe, bench_site.finite, bench_site.report_path)
    assert data["missing"] == [str(bench_site.report_path(tmp_path, "history", "c8g"))]
    view = data["views"][SERIES]
    k150, k600 = view["series"]["k150"], view["series"]["k600"]
    assert [f["id"] for f in k150] == ["2026-11-09", "2026-11-10"]  # oldest first
    assert k150[0]["rungs"] == {"av1": ["fast", "slow"]} and k150[0]["commit"] == "a" * 9
    assert k150[0]["view"]["values"]["av1@fast"] == {"j": -10.0, "low": -11.0, "high": -9.0,
                                                    "speed": 50.0, "clips": 3}
    assert set(k150[1]["view"]["values"]) == {"av1@fast"}  # no curve, no point
    assert [f["id"] for f in k600] == ["2026-11-09", "2026-11-26"]
    # a day the days file does not list is the report's own build
    assert k600[1]["commit"] == COMMIT[:9] and k600[1]["label"] == "2026-11-26"
    assert k600[1]["caption"][0] == bench_site.preview_history.BENCHMARK_CAPTION
    assert set(view["references"]) == {"x264-medium"}  # no +drop row, no metric-tuned twin
    assert view["references"]["x264-medium"]["clips"] == 3  # the common set
    json.dumps(data, allow_nan=False)


def test_a_listed_day_after_the_benchmark_build_keeps_its_own_frame(tmp_path):
    families = HISTORY_FAMILIES | {"nanocodec-av1@fast": row(-14.0),
                                   "nanocodec-av1@fast~2026-11-27": row(-16.0)}
    write_report(tmp_path, bench_site.preview_history.REPORT_NAME, families)
    days = tmp_path / "days.json"
    days.write_text(json.dumps([{"day": d, "commit": c * 9, "codecs": "av1"}
                                for d, c in (("2026-11-09", "a"), ("2026-11-27", "c"))]))
    data = bench_site.preview_history.history_data(
        tmp_path, days, tmp_path / "no-captions.json", list(bench_site.AWS_MACHINES), {},
        bench_site.describe, bench_site.finite, bench_site.report_path)
    k150 = {f["id"]: f for f in data["views"][SERIES]["series"]["k150"]}
    # the benchmark build's families sit on its own (unlisted) day, not the newest one
    assert k150["2026-11-26"]["view"]["values"]["av1@fast"]["j"] == -14.0
    assert k150["2026-11-27"]["view"]["values"]["av1@fast"]["j"] == -16.0
    assert k150["2026-11-27"]["commit"] == "c" * 9


def test_a_days_compliance_is_the_worst_of_our_families():
    rows = {"nanocodec-av1@fast": {"compliance": {"own_rate": {"worst_peak_buffers": 1.04}, "mean_rate_over_target": 0.99}},
            "nanocodec-av1@slow": {"compliance": {"own_rate": {"worst_peak_buffers": 1.0}, "mean_rate_over_target": 1.05}},
            "x264-medium": {"compliance": {"own_rate": {"worst_peak_buffers": 2.0}}}}
    figures = {r["label"]: r["value"] for r in bench_site.preview_history.compliance(rows)}
    assert figures["worst buffer peak at its own rate"] == "1.040 buffers"
    assert figures["achieved / target rate, highest family mean"] == "1.050"


def test_a_dataset_shows_its_headline_report_with_the_other_view_beside_it(tmp_path):
    write_report(tmp_path, "vc1k-out-of-sample", FAMILIES)
    write_report(tmp_path, "vc1k", FAMILIES | {"nanocodec-av1@fast": row(-29.0)}, common_clips=9)
    (view,) = bench_site.site_data(tmp_path)["views"]
    assert view["dataset"] == "vc1k" and view["source"].endswith("vc1k-out-of-sample-c8i.json")
    assert view["values"]["nanocodec-av1@fast"]["bd"]["j"]["mean"] == -30.0
    beside = view["beside"]
    assert beside["common_clips"] == 9 and beside["values"]["nanocodec-av1@fast"]["bd"]["j"]["mean"] == -29.0
    (tuned,) = view["in_sample"]  # the in-sample record's VC-1K set
    assert len(tuned["clips"]) == 9


def test_the_limitations_are_the_procedures_section_16():
    items = bench_site.limitations()
    assert items[0]["text"].startswith("Machines:")
    nested = next(i for i in items if i["text"].startswith("Loaded speed"))
    assert nested["items"] and not any("**" in i["text"] or "`" in i["text"] for i in items)


def test_a_history_failing_its_tracking_check_is_shown_with_the_check(tmp_path):
    full = {"x264-medium": row(0.0), "x264-fast": row(10.0), "nanocodec-av1@fast": row(-30.0)}
    write_report(tmp_path, "ours", full)
    # on the subset ours reads 3 points worse than the references do: outside the 1-point limit
    write_report(tmp_path, "history", HISTORY_FAMILIES | {"x264-fast": row(10.0),
                                                         "nanocodec-av1@fast": row(-27.0)})
    data = bench_site.preview_history.history_data(
        tmp_path, tmp_path / "days.json", tmp_path / "no-captions.json", [SERIES], {},
        bench_site.describe, bench_site.finite, bench_site.report_path)
    check = data["views"][SERIES]["tracking_check"]
    assert not check["passed"]
    assert check["median_offset"] == 0.0 and set(check["outside"]) == {"nanocodec-av1@fast"}
    write_report(tmp_path, "history", HISTORY_FAMILIES | {"x264-fast": row(10.0),
                                                         "nanocodec-av1@fast": row(-29.5)})
    data = bench_site.preview_history.history_data(
        tmp_path, tmp_path / "days.json", tmp_path / "no-captions.json", [SERIES], {},
        bench_site.describe, bench_site.finite, bench_site.report_path)
    assert data["views"][SERIES]["tracking_check"]["passed"]
