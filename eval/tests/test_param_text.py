"""Preset values reach the encoder exactly (encoders.nanocodec.param_text)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import levels, realuse
from nceval.encoders.nanocodec import param_text


def test_values_round_trip_and_short_ones_keep_their_form():
    for value in (1.3333334, 0.104975485, 277.3997809563062, 0.000301923024994135):
        assert float(param_text(value)) == value
    assert param_text(1.3333334) == "1.3333334"
    assert [param_text(v) for v in (2.0, 0.25, 1.3)] == ["2", "0.25", "1.3"]


def test_final_check_and_boards_pass_the_shipped_value(tmp_path):
    (tmp_path / "h264").mkdir()
    (tmp_path / "h264" / "fast.toml").write_text("analysis.one_position_sad_per_pixel = "
                                                 "1.3333334\nquant.trellis = 0\n")
    assert levels.preset_args(tmp_path, "h264", "fast") == [
        "analysis.one_position_sad_per_pixel=1.3333334", "quant.trellis=0"]
    family = realuse.own_family("nanocodec-h264@fast", tmp_path / "nanocodec", tmp_path)
    assert "analysis.one_position_sad_per_pixel=1.3333334" in family.extra
