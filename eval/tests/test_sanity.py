"""Check logic of the sanity report (nceval.sanity), on synthetic data."""

import json

import numpy as np
import pytest

from nceval.sanity.activity import (Activity, ToolLimits, compare_activity,
                                    histogram_divergence, mean_activity, subpel_phase_shares)
from nceval.sanity.avc_tools import parse_mb_type_log
from nceval.sanity.curves import Curve, Point
from nceval.sanity.decay import decay_finding, gop_trends
from nceval.sanity.flags import Flag, Severity, severity_by_excess, sort_flags
from nceval.sanity.harness import check_cbr, check_clip_caps, command_mismatches, curve_problems
from nceval.sanity.headers import parse_trace
from nceval.sanity.ordering import ordered_pairs, region_ratios
from nceval.sweep import SweepConfig

RATES = [200.0, 300.0, 450.0, 700.0, 1000.0]
QUALITY = [30.0, 31.5, 33.0, 34.5, 36.0]
LIMITS = ToolLimits(skip_share_ratio=2.0, share_gap=0.25, never_active_share=0.01)


# --- ordering ----------------------------------------------------------------

def test_region_ratios_equal_curves_give_one():
    ratios = region_ratios(RATES, QUALITY, RATES, QUALITY)
    assert [r.region for r in ratios] == ["low", "high"]
    assert all(r.bits_ratio == pytest.approx(1.0) for r in ratios)


def test_region_ratios_double_rate_everywhere():
    ratios = region_ratios([2 * r for r in RATES], QUALITY, RATES, QUALITY)
    assert all(r.bits_ratio == pytest.approx(2.0) for r in ratios)


def test_region_ratios_loss_confined_to_high_rates():
    worse_high = [r * (1.0 if q < 33 else 1.5) for r, q in zip(RATES, QUALITY)]
    low, high = region_ratios(worse_high, QUALITY, RATES, QUALITY)
    assert high.bits_ratio > low.bits_ratio > 0.99


def test_region_ratios_unusable_curves():
    assert region_ratios(RATES[:2], QUALITY[:2], RATES, QUALITY) == []
    assert region_ratios(RATES, [30, 32, 31, 33, 34], RATES, QUALITY) == []
    assert region_ratios(RATES, [40, 41, 42, 43, 44], RATES, QUALITY) == []  # no overlap


def test_ordered_pairs_cover_every_later_codec():
    assert ordered_pairs(("av1", "hevc", "h264")) == [
        ("av1", "hevc"), ("av1", "h264"), ("hevc", "h264")]


# --- decay -------------------------------------------------------------------

def _frames(n, slope, start=40.0, key_every=None):
    types = ["I" if i == 0 or (key_every and i % key_every == 0) else "P" for i in range(n)]
    values = [start + slope * i for i in range(n)]
    return types, values


def test_steady_decline_is_flagged():
    (trend,) = gop_trends(*_frames(150, -0.05))
    assert trend.slope == pytest.approx(-0.05)
    assert trend.sustained(-0.02)
    excess = decay_finding(trend, None, limit=-0.02, margin=0.01)
    assert excess == pytest.approx(0.05 * 149, rel=1e-6)


def test_flat_or_shared_decline_is_not_flagged():
    (flat,) = gop_trends(*_frames(150, 0.0))
    assert decay_finding(flat, None, -0.02, 0.01) is None
    (ours,) = gop_trends(*_frames(150, -0.05))
    (ref,) = gop_trends(*_frames(150, -0.045))
    assert decay_finding(ours, ref, -0.02, 0.01) is None  # the content gets harder for both


def test_dip_and_recovery_is_not_sustained():
    types, _ = _frames(150, 0.0)
    values = [40 - 0.1 * i if i < 75 else 32.5 + 0.02 * (i - 75) for i in range(150)]
    (trend,) = gop_trends(types, values)
    assert not trend.sustained(-0.02)
    assert decay_finding(trend, None, -0.02, 0.01) is None


def test_key_frame_settling_is_not_decay():
    types, _ = _frames(150, 0.0)
    values = [48 - 0.05 * i if i < 75 else 44.3 - 0.001 * (i - 75) for i in range(150)]
    (trend,) = gop_trends(types, values)
    assert trend.slope < -0.02
    assert decay_finding(trend, None, -0.02, 0.01) is None


def test_gops_split_at_key_frames_and_short_chains_skipped():
    trends = gop_trends(*_frames(100, -0.05, key_every=80))
    assert [t.start for t in trends] == [0]  # the 20-frame second GOP is too short


# --- harness -----------------------------------------------------------------

def _curve(points, failures=()):
    curve = Curve("enc", "clip", failures=list(failures))
    for i, (q, kbps, psnr, vmaf) in enumerate(points):
        curve.points.append(Point(i, q, 150, kbps, {"psnr_yuv": psnr, "vmaf_v1": vmaf}))
    return curve


def test_healthy_curve_has_no_problems():
    curve = _curve([(40, 200, 30, 50), (35, 400, 32, 60), (30, 800, 34, 70)])
    assert curve_problems(curve, ("psnr_yuv", "vmaf_v1")) == []


def test_curve_problems_found():
    few = _curve([(40, 200, 30, 50), (35, 400, 32, 60)])
    assert any("only 2" in p for p in curve_problems(few, ("psnr_yuv",)))
    bumpy = _curve([(40, 200, 30, 50), (35, 400, 29, 60), (30, 800, 34, 70)])
    assert any("psnr_yuv does not rise" in p for p in curve_problems(bumpy, ("psnr_yuv",)))
    saturated = _curve([(40, 200, 40, 98.5), (35, 400, 41, 99), (30, 800, 42, 99.5)])
    assert any("saturated" in p for p in curve_problems(saturated, ("vmaf_v1",)))
    failed = _curve([(40, 200, 30, 50)] * 3, failures=[(30, "md5 mismatch")])
    assert any("conformance" in p for p in curve_problems(failed, ()))
    assert curve_problems(Curve("enc", "clip"), ()) == ["missing: no encodes stored"]


def test_command_mismatches():
    command = ["enc", "--rtc", "1", "--preset", "12", "--rc", "1"]
    assert command_mismatches(command, {"--rtc": "1", "--preset": "12"}) == []
    assert command_mismatches(command, {"--rc": "0", "--lp": "1"}) == [
        "--rc: expected 0, got 1", "--lp: expected 1, got missing"]


def test_clip_caps(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps({"clips": [{"name": "a", "source_kbps": 900}]}))
    (tmp_path / "bad.json").write_text(json.dumps({"clips": [{"name": "b"}, {"name": "c",
                                                                            "source_kbps": 0}]}))
    (flag,) = check_clip_caps(tmp_path)
    assert flag.subject == "clip set bad.json" and flag.evidence["clips"] == ["b", "c"]


def _cbr(directory, name, family, target, kbps):
    directory.mkdir(parents=True, exist_ok=True)
    data = {"family": family, "clip": "clip", "kbps": kbps}
    if target:
        data["target_kbps"] = target
    (directory / name).write_text(json.dumps(data))


def test_cbr_miss_severity_and_floor(tmp_path):
    base = tmp_path / "f150" / "clip"
    _cbr(base / "ours", "cbr600-a.json", "ours", 600, 610)       # within limit
    _cbr(base / "ours", "cbr200-a.json", "ours", 200, 420)       # below the QP-51 floor
    _cbr(base / "ours", "qp51-a.json", "ours", None, 425)
    _cbr(base / "anchor", "cbr350-a.json", "anchor", 350, 480)   # no floor known: HIGH below 3x the limit
    flags = {f.subject: f for f in check_cbr(tmp_path, 0.15)}
    assert set(flags) == {"CBR ours/clip target 200 kbps", "CBR anchor/clip target 350 kbps"}
    assert flags["CBR ours/clip target 200 kbps"].severity is Severity.MEDIUM
    assert flags["CBR anchor/clip target 350 kbps"].severity is Severity.HIGH


# --- tool activity -----------------------------------------------------------

def test_subpel_phase_shares():
    rows = np.array([0, 8, 4, 2, 1])
    cols = np.array([0, 0, 0, 0, 0])
    assert subpel_phase_shares(rows, cols, 8) == {
        "integer": 0.4, "half": 0.2, "quarter": 0.2, "eighth": 0.2}
    assert subpel_phase_shares(np.array([0]), np.array([3]), 4) == {"quarter": 1.0}


def test_histogram_divergence():
    assert histogram_divergence({"a": 1.0}, {"a": 1.0}) == 0.0
    assert histogram_divergence({"a": 1.0}, {"b": 1.0}) == 1.0


def _activity(skip=0.5, cdef=0.8, integer=0.6, kind="P"):
    return Activity(frames=1, frame_type=kind, skip_share=skip, intra_share=0.02,
                    histograms={"subpel_phases": {"integer": integer, "quarter": 1 - integer}},
                    filters={"cdef_active_share": cdef})


def test_cdef_never_active_is_flagged():
    flags = compare_activity(_activity(cdef=0.0), _activity(cdef=0.9), LIMITS, "s",
                             ("cdef_active_share",))
    assert [f.finding.split(" ")[0] for f in flags] == ["cdef_active_share"]
    assert flags[0].severity is Severity.HIGH


def test_matching_activity_is_quiet_and_divergence_flags():
    assert compare_activity(_activity(), _activity(), LIMITS, "s", ("cdef_active_share",)) == []
    flags = compare_activity(_activity(skip=0.1), _activity(skip=0.7), LIMITS, "s", ())
    assert any("skip_share" in f.finding for f in flags)
    flags = compare_activity(_activity(integer=1.0), None, LIMITS, "s", ())
    assert any("sub-pel" in f.finding for f in flags)


def test_mean_activity_by_frame_type():
    means = mean_activity([_activity(skip=0.2), _activity(skip=0.4), _activity(kind="I")])
    assert means["P"].frames == 2 and means["P"].skip_share == pytest.approx(0.3)
    assert means["I"].frames == 1


def test_parse_mb_type_log_skips_header_rows():
    lines = ["[h264 @ 0x1] New frame, type: P",
             "[h264 @ 0x1]      0                       128",
             "[h264 @ 0x1]    0 S  >  i  >+ ",
             "[h264 @ 0x1]   16 I  S  S  S  "]
    (frame,) = parse_mb_type_log(lines)
    assert [c[0] for c in frame] == ["S", ">", "i", ">", "I", "S", "S", "S"]
    assert frame[3][1] == "+"


def test_parse_trace_headers():
    lines = ["[trace_headers @ 0x1] Sequence Header",
             "[trace_headers @ 0x1] 3   enable_cdef        1 = 1",
             "[trace_headers @ 0x1] Frame Header",
             "[trace_headers @ 0x1] 24  show_existing_frame 0 = 0",
             "[trace_headers @ 0x1] 57  loop_filter_level[0] 000011 = 3",
             "[trace_headers @ 0x1] Frame Header",
             "[trace_headers @ 0x1] 24  show_existing_frame 1 = 1"]
    trace = parse_trace(lines)
    assert trace.params == {"enable_cdef": 1}
    assert trace.frames == [{"show_existing_frame": 0, "loop_filter_level[0]": 3}]


# --- flags and structures ----------------------------------------------------

def test_flags_sort_most_severe_first():
    flags = [Flag("b", Severity.MEDIUM, "x", ""), Flag("a", Severity.CRITICAL, "y", ""),
             Flag("a", Severity.MEDIUM, "w", "")]
    assert [(f.check, f.subject) for f in sort_flags(flags)] == [("a", "y"), ("a", "w"),
                                                                ("b", "x")]
    assert severity_by_excess(0.01, 0.1, 0.3) is Severity.MEDIUM
    assert severity_by_excess(0.5, 0.1, 0.3) is Severity.CRITICAL


def test_single_keyframe_structure_spans_the_clip():
    from nceval.clips import Clip
    from pathlib import Path
    clip = Clip("c", Path("/c.yuv"), 64, 64, 25.0, 1500, 1000.0, "static", "train")
    cfg = SweepConfig(intra_period_seconds=0, single_keyframe=True)
    assert cfg.structure_tag == "ip1key-tl1-vmaf-v1n"
    assert cfg.structure_for(clip).intra_period_frames == 1500
    assert SweepConfig(intra_period_seconds=10).structure_for(clip).intra_period_frames == 250


def test_overlapping_sets_report_a_flag_once():
    from nceval.sanity.checks import unique_flags
    a = Flag("outliers", Severity.HIGH, "flagged/clip: x vs y", "worse")
    b = Flag("outliers", Severity.HIGH, "short/clip: x vs y", "worse")
    assert unique_flags([a, b]) == [a]


def test_sampling_identity_only_marks_changed_sampling():
    from nceval.sweep import _coarse_grid, sampling_identity
    # Every curve now carries the cap headroom (targets top out at 90% of the cap).
    assert sampling_identity((10, 51), 6) == "cap-headroom=0.9"
    assert sampling_identity((40, 255), 6) == "cap-headroom=0.9,coarse-grid-step=27"
    assert sampling_identity((10, 51), 5) == "cap-headroom=0.9,points=5"
    assert sampling_identity((10, 51), 6, include_xpsnr=False) == "no-xpsnr,cap-headroom=0.9"
    assert len(_coarse_grid((40, 255))) == 9
    assert _coarse_grid((10, 51)) == [10, 16, 22, 28, 34, 40, 46, 51]


class _FrameStore:
    """Just enough of CurveStore for keyframe_flag."""

    def __init__(self, frames):
        self._frames = frames

    def frames(self, point):
        return self._frames[point]


def test_keyframe_flag_large_key_without_quality():
    from nceval.sanity.curves import FrameRow
    from nceval.sanity.tools import keyframe_flag
    ours = [FrameRow(0, 70000, "I", 50.0, None)] + [FrameRow(i, 500, "P", 46, None)
                                                  for i in range(1, 10)]
    ref = [FrameRow(0, 21000, "I", 49.8, None)] + [FrameRow(i, 400, "P", 46, None)
                                                 for i in range(1, 10)]
    store = _FrameStore({"ours": ours, "ref": ref})
    flag = keyframe_flag(store, "ours", "ref", 2.0, "s")
    assert flag is not None and flag.severity is Severity.HIGH
    assert keyframe_flag(store, "ref", "ours", 2.0, "s") is None


def test_cbr_uses_loan_adjusted_error_when_present(tmp_path):
    base = tmp_path / "f150" / "clip" / "ours"
    base.mkdir(parents=True)
    (base / "cbr600-a.json").write_text(json.dumps(
        {"family": "ours", "clip": "clip", "kbps": 740, "target_kbps": 600,
         "loan_adjusted_rate_error": 0.01}))
    (base / "cbr350-a.json").write_text(json.dumps(
        {"family": "ours", "clip": "clip", "kbps": 500, "target_kbps": 350,
         "loan_adjusted_rate_error": 0.30}))
    flags = check_cbr(tmp_path, 0.15)
    assert [f.subject for f in flags] == ["CBR ours/clip target 350 kbps"]
    assert flags[0].evidence["raw_rate_error"] > 0.4


def test_stress_matrix_seeds_cover_every_combination():
    from nceval.codecs import Codec
    from nceval.stress_matrix import combinations, features_for
    for codec in (Codec.H264, Codec.HEVC, Codec.AV1):
        combos = combinations(codec)
        seen = {features_for(codec, seed) for seed in range(1, len(combos) + 1)}
        assert seen == set(combos)
    assert any(f.loop_filter for f in combinations(Codec.AV1))
    assert any(f.filter_intra for f in combinations(Codec.AV1))
    assert not any(f.filter_intra for f in combinations(Codec.HEVC))
    assert not any(f.loop_filter for f in combinations(Codec.H264))
    assert any(f.cavlc for f in combinations(Codec.H264))
    assert not any(f.cavlc for f in combinations(Codec.HEVC))


def test_stress_case_flags_and_lost_frame():
    from nceval.codecs import Codec
    from nceval.stress_matrix import FRAMES, StressCase, combinations, features_for
    for seed in range(1, len(combinations(Codec.AV1)) + 1):
        case = StressCase(Codec.AV1, 200, 120, 30.0, seed)
        flags = case.encoder_flags()
        assert ("--qp" in flags) != ("--bitrate" in flags)
        structure = features_for(Codec.AV1, seed).structure
        if structure.endswith("ltr-loss"):
            assert case.coded_frames == FRAMES - 1 and 0 < case.lost_frame < FRAMES - 1
            assert "--simulate-loss" in flags
        else:
            assert case.coded_frames == FRAMES and "--simulate-loss" not in flags
        assert ("--temporal-layers" in flags) == structure.startswith("layers")
        assert ("--rate-change" in flags) == (features_for(Codec.AV1, seed).rate == "cbr-step")
        assert ("--intra-refresh" in flags) == structure.startswith("refresh")


def test_cbr_targets_are_ceilings_for_our_encoder(tmp_path):
    _cbr(tmp_path / "f150" / "clip" / "nanocodec-av1", "cbr400-a.json", "nanocodec-av1", 400, 70)
    _cbr(tmp_path / "f150" / "clip" / "svtav1-m12", "cbr400-a.json", "svtav1-m12", 400, 200)
    _cbr(tmp_path / "f150" / "clip" / "nanocodec-h264", "cbr400-a.json", "nanocodec-h264", 400, 480)
    subjects = {f.subject for f in check_cbr(tmp_path, 0.15)}
    assert subjects == {"CBR svtav1-m12/clip target 400 kbps", "CBR nanocodec-h264/clip target 400 kbps"}
