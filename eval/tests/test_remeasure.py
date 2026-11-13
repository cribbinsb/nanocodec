"""Re-measuring stored curves: grid lookup, preset fallback, kept streams."""
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval.clips import Clip
from nceval.codecs import Codec
from nceval.remeasure import Remeasure, Source, stored_points
from nceval.results import EncodeRecord, ResultsDb
from nceval.gatepolicy import GatePolicy, PointCheck
from nceval.sweep import SweepConfig


def database(root: Path, rows) -> Path:
    path = root / "results" / "r.sqlite"
    db = ResultsDb(path)
    for encoder, clip, q, passed in rows:
        db.store(EncodeRecord(encoder, "ip10s-tl1", clip, q, 30, 10, 100, passed, "",
                              platform.machine(), f"fp-{encoder}", "c"), {})
    db.conn.close()
    return path


def test_first_source_with_the_curve_wins_and_presets_use_their_base(tmp_path):
    first = Source(database(tmp_path / "a", [("nanocodec-av1", "clip", 100, True),
                                             ("nanocodec-av1", "clip", 140, False)]), "ip10s-tl1")
    second = Source(database(tmp_path / "b", [("svtav1-m12", "clip", 30, True)]), "ip10s-tl1")
    assert stored_points([first, second], "svtav1-m12", "clip")[0] == second
    source, points = stored_points([first, second], "nanocodec-av1@av1-fast", "clip")
    assert source == first and [p[0] for p in points] == [100]  # failed points are not reused
    assert stored_points([first, second], "openh264-baseline", "clip") is None
    assert first.work == tmp_path / "a" / "work"


def test_anchor_streams_are_rescored_never_reencoded(tmp_path):
    source = Source(database(tmp_path, [("svtav1-m12", "clip", 30, True),
                                        ("svtav1-m12", "clip", 40, True)]), "ip10s-tl1")
    kept = source.work / "svtav1-m12" / "ip10s-tl1" / "clip" / "fp-svtav1-m12"
    kept.mkdir(parents=True)
    (kept / "q030-30f.ivf").write_bytes(b"x" * 100)  # q40's stream is gone: dropped
    sweep = MagicMock()
    sweep.fingerprint.return_value = "new"
    sweep.paths.work = tmp_path / "work"  # where the curve lock lives
    sweep._check_and_measure.return_value = True
    encoder = SimpleNamespace(name="svtav1-m12", codec=Codec.AV1, verify_recon=lambda: False)
    clip = Clip("clip", tmp_path / "clip.yuv", 16, 16, 30, 30, 2000, "static", "val")
    runner = Remeasure(sweep, SweepConfig(10), [source])
    with patch("nceval.remeasure.payload_bytes", return_value=1000):
        assert runner.curve(encoder, clip).startswith("done svtav1-m12/clip")
    sweep._measure_point.assert_not_called()
    (call,) = sweep._check_and_measure.call_args_list
    assert call.args[3] == 30 and call.args[7] == kept / "q030-30f.ivf"
    assert call.args[9] is PointCheck.NONE  # decoded once, for the metrics only


def test_our_encoders_reencode_the_stored_grid_with_sampled_standard_decodes(tmp_path):
    source = Source(database(tmp_path, [("nanocodec-h264", "clip", q, True) for q in (26, 30, 34, 38)]),
                    "ip10s-tl1")
    sweep = MagicMock()
    sweep.fingerprint.return_value = "new"
    sweep.paths.work = tmp_path / "work"
    sweep._measure_point.return_value = True
    encoder = SimpleNamespace(name="nanocodec-h264", codec=Codec.H264, verify_recon=lambda: True)
    clip = Clip("clip", tmp_path / "clip.yuv", 16, 16, 30, 30, 2000, "static", "val")
    Remeasure(sweep, SweepConfig(10, gate=GatePolicy.SAMPLE), [source]).curve(encoder, clip)
    quantizers = [c.args[3] for c in sweep._measure_point.call_args_list]
    checks = [c.args[7] for c in sweep._measure_point.call_args_list]
    assert quantizers == [26, 30, 34, 38]
    # first point + one rotating QP get the standard decoders, our decoder the rest
    assert checks[0] is PointCheck.STANDARD and checks.count(PointCheck.STANDARD) == 2
    assert checks.count(PointCheck.OWN) == 2
