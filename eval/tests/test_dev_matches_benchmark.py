"""Dev conditions match the benchmark's (owner, 2026-11-24): how ours is run, the ladder and the
metrics come from one shared definition each; the intended differences are listed."""
from pathlib import Path

import pytest

from nceval import cbr, fairness
from nceval.clips import Clip
from nceval.codecs import Codec


def clip(tmp_path):
    yuv = tmp_path / "c.yuv"
    yuv.write_bytes(b"")
    return Clip(name="c", yuv=yuv, width=1280, height=720, fps=30.0, frames=150,
                source_kbps=6000, content_class="static", split="train")


def record_tail(mode, value):
    values = {"BIN": "B", "INPUT": "I", "OUTPUT": "O", "W": 1280, "H": 720, "FPS_NUM": 30,
              "FPS_DEN": 1, "N": 150, "KEYINT": 300, ("KBPS" if mode == "cbr" else "QP"): value}
    argv = fairness.command("nanocodec-av1@medium", mode, values)
    return argv[argv.index("--bitrate" if mode == "cbr" else "--qp"):]


def test_our_cbr_and_fixed_qp_conditions_are_the_benchmark_records(tmp_path, monkeypatch):
    monkeypatch.delenv(fairness.CONDITIONS_ENV, raising=False)
    ours = cbr.Family("nanocodec-av1", Codec.AV1, Path("/t/nanocodec"), own=True)
    for mode, kwargs, value in (("cbr", {"kbps": 300}, 300), ("qp", {"qp": 140}, 140)):
        argv = ours.command(clip(tmp_path), 150, 300, Path("O"), None, **kwargs)
        tail = argv[argv.index("--bitrate" if mode == "cbr" else "--qp"):]
        assert tail == record_tail(mode, value), mode


def test_the_shipped_defaults_are_an_explicit_escape_with_their_own_point_identity(
        tmp_path, monkeypatch):
    ours = cbr.Family("nanocodec-av1", Codec.AV1, Path("/t/nanocodec"), own=True)
    monkeypatch.setenv(fairness.CONDITIONS_ENV, fairness.SHIPPED_DEFAULTS)
    assert ours.own_conditions() == fairness.SHIPPED_DEFAULTS
    argv = ours.command(clip(tmp_path), 150, 300, Path("O"), None, kbps=300)
    assert "--strict-buffer" not in argv and "rate.first_key_lookahead_off=1" not in argv
    # benchmark-condition points carry the record's flags in their identity; shipped-default
    # points keep the identities they had before 2026-11-24
    assert cbr.conditions_tag().startswith("conditions=")


def test_the_ladder_and_metrics_are_the_benchmarks():
    """The rate ladder (one module) and the scored VMAF model."""
    import sys
    from nceval import metrics, rate_ladder, realuse
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
    import plan
    assert tuple(realuse.LADDER_KBPS) == tuple(plan.LADDER_720P30) == \
        rate_ladder.LADDER_720P30_KBPS
    score = (Path(__file__).resolve().parents[2] / "tools" / "bench" / "score.py").read_text()
    assert metrics.VMAF_V1_MODEL.name in score


#: Differences between dev and the benchmark that are intended, each with its reason
#: (docs/PROCEDURES.md "Three-stage checks"); anything else must be the same.
INTENDED_DIFFERENCES = {
    "scored": "dev scores the encoder's reconstruction; decoding is checked separately",
    "zero_line": "dev compares a candidate against its own base, not x264 medium",
    "long_windows": "dev adds the 600/900-frame key-share windows the benchmark defers",
    "cpu": "dev's CPU is a paired change on its own encodes, not an absolute speed",
    "class_balance": "dev averages per class first: its draws are stratified by class",
    "single_point": "L1/L3 code one rung per clip (no curve): their score is the point quality",
}


def test_the_intended_differences_are_the_documented_ones():
    procedures = (Path(__file__).resolve().parents[2] / "docs" / "PROCEDURES.md").read_text()
    for phrase in ("scores the encoder's reconstruction", "its own base",
                   "600/900-frame key-share windows", "paired change on its own encodes",
                   "class-balanced", "one rung per clip"):
        assert phrase in procedures, phrase


def test_the_pre_bp3_charts_keep_the_shipped_defaults_and_their_point_identities(tmp_path):
    """The published charts were measured at the shipped defaults: their families say so and
    their points keep the identities they had (no conditions part)."""
    from nceval import site
    family = site.chart_family("nanocodec-av1")
    assert family.own_conditions() == fairness.SHIPPED_DEFAULTS
    argv = family.command(clip(tmp_path), 150, 300, Path("O"), None, kbps=600)
    assert "--strict-buffer" not in argv


def curve_points(rates, psnr, vmaf, ssim):
    return [{"kbps": r, "target_kbps": r, "psnr_yuv": a, "vmaf_v1": b, "ms_ssim": c,
             "worst_psnr_yuv": a - 2, "worst_vmaf_v1": b - 5, "flicker_psnr_y": 0.3,
             "first_key_pulse_psnr_y": 0.5}
            for r, a, b, c in zip(rates, psnr, vmaf, ssim)]


def test_the_decision_score_is_the_benchmarks_bd_j_from_the_one_function():
    """nceval's per-clip decision term and the benchmark's report both come from nceval.bdj:
    the screens' bd_j equals bdj.bd_j, which the report's clip_bd is built from; the earlier
    J_real stays a printed secondary score."""
    import sys
    from nceval import bdj, interval, jreal
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
    import report
    anchor = curve_points([300, 500, 800, 1200], [33, 35, 37, 39], [60, 70, 78, 85],
                          [0.95, 0.96, 0.97, 0.98])
    test = curve_points([290, 480, 770, 1150], [33, 35, 37, 39], [60, 70, 78, 85],
                        [0.95, 0.96, 0.97, 0.98])
    terms = jreal.compare_clip(anchor, test)
    assert terms["bd_j"] == bdj.bd_j(jreal.bdj_curve(test), jreal.bdj_curve(anchor))
    assert report.MIN_POINTS == bdj.MIN_POINTS and report.METRICS == bdj.METRICS
    row = {"class": "static", **terms}
    assert interval.clip_j(row, None, 0.0) == terms["bd_j"]
    assert interval.clip_j(row, None, 0.0, "j_real") == (terms["quality"] + terms["quality_low"]) / 2
    # BP-3.14 (owner, 2026-11-25): the screens take the benchmark's 2-point minimum with it
    two = curve_points([300, 800], [33, 37], [60, 78], [0.95, 0.97])
    assert jreal.compare_clip(two, two, min_points=2)["bd_j"] == pytest.approx(0.0)
    one = curve_points([300], [33], [60], [0.95])
    assert jreal.compare_clip(one, one, min_points=2)["bd_j"] is None


def test_fixed_qp_ab_codes_the_benchmarks_quantizers():
    """nceval.fixedqp ab and tools/bench/plan.py take our Fixed QP points from one place."""
    import sys
    from nceval.ctc.conditions import ours_points
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
    import plan
    for codec in Codec:
        assert plan.quantizers("nanocodec", codec.value) == ours_points(codec)
    fixedqp_source = (Path(__file__).resolve().parents[1] / "nceval" / "fixedqp.py").read_text()
    assert "list(ours_points(codec))" in fixedqp_source
