"""The CPU charge k of a preset change: its codec's front slope (nceval.front_k)."""
import argparse
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import checkrun, front_k, screen


def day():
    """A codec with four rungs on its front and one rung off it."""
    values = {"x@slow": (-50.0, 50.0), "x": (-48.0, 100.0), "x@fast": (-44.0, 200.0),
              "x@faster": (-40.0, 300.0), "x@slower": (-49.0, 40.0)}
    return {"date": "2026-01-01", "commit": "abc", front_k.TIMING_MODE_KEY: "cbr",
            "values": {k: {"j": j, "fps_x86": fps} for k, (j, fps) in values.items()},
            "codec_fronts": {"x": ["x@slow", "x", "x@fast", "x@faster"]}}


@pytest.fixture
def cbr_table(tmp_path, monkeypatch):
    """A checked-in table generated from a CBR-timed frame of codec x."""
    path = tmp_path / "front_k.json"
    path.write_text(json.dumps(front_k.generate_from(day(), "history.json")))
    monkeypatch.setattr(front_k, "TABLE", path)
    return path


def test_a_chord_scores_zero_along_itself_in_the_rungs_own_units():
    slow, fast = {"j": -50.0, "fps": 50.0}, {"j": -44.0, "fps": 200.0}
    k = front_k.chord_k(slow, fast, -48.0)
    # Moving from fast to slow: the chart gains 6 points, 6 / 0.52 % at a rung of J -48 %,
    # for 100 ln 4 % more CPU.
    assert k == pytest.approx(6 / 0.52 / (100 * math.log(4)))
    assert -6 / 0.52 + k * 100 * math.log(4) == pytest.approx(0)


def test_interior_rungs_take_their_neighbours_chord_and_ends_their_end_chord():
    table = front_k.slopes(day())["x"]
    assert table["medium"]["chord"] == ["x@slow", "x@fast"]
    assert table["slow"]["chord"] == ["x@slow", "x"]
    assert table["faster"]["chord"] == ["x@fast", "x@faster"]
    # off the front and slowest: the front's slow end chord
    assert table["slower"]["chord"] == ["x@slow", "x"] and not table["slower"]["on_front"]
    assert table["medium"]["toward_slower"] == pytest.approx(
        front_k.chord_k({"j": -50.0, "fps": 50.0}, {"j": -48.0, "fps": 100.0}, -48.0), abs=1e-4)


def test_rung_k_names_its_source_and_a_missing_rung_fails_loudly():
    table = front_k.generate_from(day(), "history.json")
    k, source = front_k.rung_k("x", "medium", table)
    assert k == table["rungs"]["x"]["medium"]["k"] and "2026-01-01" in source
    with pytest.raises(front_k.NoSlope):
        front_k.rung_k("x", "veryfast", table)


def test_an_explicit_k_wins_and_is_warned_about(capsys):
    k, source = front_k.resolve("h264", "medium", 0.6)
    assert k == 0.6 and "explicit" in source
    assert "!!!" in capsys.readouterr().err


def test_a_slope_from_another_timing_mode_than_the_screens_cbr_is_refused(tmp_path):
    fixed = {**day(), front_k.TIMING_MODE_KEY: None}
    with pytest.raises(front_k.NoSlope, match="fixed QP"):
        front_k.generate_from(fixed, "history.json")
    earlier = {**day(), "date": "2025-12-31", front_k.TIMING_MODE_KEY: None}
    chained = {**day(), front_k.CHAINED_KEY: "2025-12-31"}
    with pytest.raises(front_k.NoSlope, match="chained to 2025-12-31"):
        front_k.generate_from(chained, "history.json", [earlier, chained])
    timed = {**earlier, front_k.TIMING_MODE_KEY: "cbr"}
    assert front_k.generate_from(chained, "history.json", [timed, chained])["rungs"]
    old = tmp_path / "old.json"
    old.write_text(json.dumps({**front_k.generate_from(day(), "h"),
                               front_k.TIMING_MODE_KEY: "fixed-qp"}))
    with pytest.raises(front_k.NoSlope, match="fixed-qp timing"):
        front_k.load(old)
    # the checked-in table (2026-11-23: the main chart, timed on its own CBR encodes) serves
    assert front_k.load()[front_k.TIMING_MODE_KEY] == "cbr"


def test_the_screen_charges_cpu_only_with_a_rung_or_k(cbr_table):
    args = lambda **kw: argparse.Namespace(**{"codec": "x", "preset": None, "k": None,
                                              "cpu_summary": None, "cpu_charge": None, **kw})
    assert screen.cpu_charge(args()) == (0.0, "no CPU measured: no CPU charge")
    with pytest.raises(SystemExit):
        screen.cpu_charge(args(cpu_charge="d:f150-k300=2"))
    assert screen.cpu_charge(args(cpu_charge="d:f150-k300=2", preset="medium")) == \
        front_k.rung_k("x", "medium")


def test_a_chart_csv_gives_slopes_only_when_its_timing_is_cbr(tmp_path):
    csv_path = tmp_path / "x86.csv"
    csv_path.write_text("family,bd_j_pct,fps_x86\n"
                        "x264-medium,0.0,100\n"
                        "nanocodec-hevc@slow,-40.0,20\n"
                        "nanocodec-hevc,-35.0,50\n"
                        "nanocodec-hevc@fast,-30.0,120\n"
                        "nanocodec-hevc@faster,-20.0,110\n")
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"rate_mode": "cbr"}))
    day = front_k.chart_day(csv_path, timing, "2026-11-23", "abc")
    assert sorted(day["codec_fronts"]["hevc"]) == ["hevc", "hevc@fast", "hevc@slow"]
    table = front_k.generate_from(day, "x86.csv")
    assert table["rungs"]["hevc"]["medium"]["chord"] == ["hevc@slow", "hevc@fast"]
    timing.write_text(json.dumps({"rung_kbps": 600}))
    with pytest.raises(front_k.NoSlope):
        front_k.generate_from(front_k.chart_day(csv_path, timing, "d", "c"), "x86.csv")


def report_row(j, speed, cbr=True):
    return {"bd": {"j": {"mean": j}}, "speed_mpix_per_cpu_s": speed,
            "compliance": {"overruns": 0} if cbr else None}


def test_a_benchmark_report_gives_slopes_only_when_cbr_with_checked_speed(tmp_path):
    families = {"nanocodec-av1@slow": report_row(-45.0, 50.0),
                "nanocodec-av1@medium": report_row(-43.0, 100.0),
                "nanocodec-av1@fast": report_row(-40.0, 200.0),
                "nanocodec-av1@fast+drop": report_row(-60.0, 900.0),
                "x264-medium": report_row(0.0, 300.0)}
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"speed_provisional": False,
                                "series": {"c8i": {"families": families}}}))
    day = front_k.report_day(path, "c8i", "2026-11-24", "abc")
    assert sorted(day["values"]) == ["av1", "av1@fast", "av1@slow"]
    table = front_k.generate_from(day, "r")
    assert table["rungs"]["av1"]["medium"]["chord"] == ["av1@slow", "av1@fast"]
    path.write_text(json.dumps({"speed_provisional": False, "series": {"c8i": {"families": {
        k: report_row(v["bd"]["j"]["mean"], v["speed_mpix_per_cpu_s"], cbr=False)
        for k, v in families.items()}}}}))
    with pytest.raises(front_k.NoSlope, match="qp"):
        front_k.generate_from(front_k.report_day(path, "c8i", "2026-11-24", "abc"), "r")
    path.write_text(json.dumps({"speed_provisional": True, "series": {}}))
    with pytest.raises(front_k.NoSlope, match="provisional"):
        front_k.report_day(path, "c8i", "2026-11-24", "abc")
