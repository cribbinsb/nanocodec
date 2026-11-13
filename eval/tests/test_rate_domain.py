"""A rate ceiling must constrain both new tuning and historical reports."""
from pathlib import Path
from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval.clips import Clip, ClipSet
from nceval.rate_domain import bounded_quantizers
from nceval import report


def test_calibration_uses_real_clip_ceiling():
    rate = lambda q: 12000 * 2 ** (-q / 6)
    selected = bounded_quantizers(rate, (10, 51), [150, 260, 455, 800], 500)
    assert len(selected) == 4
    assert all(rate(q) <= 500 for q in selected)
    assert rate(selected[0]) <= 450  # headroom for candidate rate changes


def test_reports_exclude_historical_points_above_source_cap():
    clip = Clip('c', Path('/x'), 1280, 720, 30, 300, 2000, 'static', 'val')
    db = MagicMock()
    db.rd_points.side_effect = [
        [(100, 30), (200, 33), (400, 36), (800, 40), (1500, 50)],
        [(100, 30), (200, 33), (400, 36), (800, 40), (1500, 41)],
    ]
    rows = report.compute(db, ClipSet('set', (clip,)), 'anchor', 'psnr_yuv',
                          'ip10s-tl1', 'test', ['candidate'])
    assert abs(rows[0].result.bd_rate_percent) < 1e-8


def test_noise_best_quality_cannot_come_from_above_cap():
    clip = Clip('noise', Path('/x'), 1280, 720, 30, 300, 1000, 'synthetic_noise', 'val')
    db = MagicMock()
    db.rd_points.return_value = [(100, 50), (400, 70), (800, 99)]
    rows = report.noise_summary(db, ClipSet('set', (clip,)), 'vmaf_neg', 'ip10s-tl1',
                                'test', ['candidate'], lambda _: 1)
    assert rows[0].best_quality == 70
    assert rows[0].best_kbps == 400


def test_tuning_retries_overshoot_without_scoring_it(tmp_path):
    from unittest.mock import patch
    from nceval import tune
    from fake_point import FakeEncodePoint
    clip = tune.ClipRef('c', Path('/raw'), 16, 16, 1, 4, 'static', 200)
    rates = {40: 120, 41: 90, 42: 80, 43: 70, 45: 50}
    encode = FakeEncodePoint(stream_bytes=lambda q: bytes(rates[q] * 500), psnr=float)
    with patch.object(tune, 'encode_point', side_effect=encode), patch.object(tune, 'file_sha256', return_value='binary'), \
         patch.object(tune, 'check_disk'):
        curve = tune.measure_curve(Path('/bin'), tune.Codec.H264, clip, 4, {},
                                   tmp_path, [40, 41, 43, 45])
    assert len(curve.kbps) == 4
    assert all(r <= 100 for r in curve.kbps)
    assert 40 not in curve.quantizers and 40.0 not in curve.psnr
    assert len(encode.calls) == 5  # including the gated, excluded rate probe
    assert not list(tmp_path.iterdir())


def test_source_bitrate_metadata_changes_curve_identity(tmp_path):
    from dataclasses import replace
    from nceval.fingerprint import clip_fingerprint
    yuv = tmp_path / 'clip.yuv'
    yuv.write_bytes(bytes(384))
    clip = Clip('c', yuv, 16, 16, 30, 1, 1000, 'static', 'train')
    assert clip_fingerprint(clip) != clip_fingerprint(replace(clip, source_kbps=2000))


def test_sweep_overshoot_is_gated_but_has_no_quality_metrics(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from nceval.conformance import GateResult
    from nceval.sweep import Sweep, SweepConfig
    from nceval.codecs import Codec
    clip = Clip('c', Path('/raw'), 16, 16, 30, 1, 2000, 'static', 'val')
    stream = tmp_path / 'q.264'
    stream.write_bytes(bytes(100))
    db = MagicMock()
    encoder = SimpleNamespace(name='test', codec=Codec.H264, verify_recon=lambda: True,
                              own_decoder=lambda paths: None)
    with patch('nceval.sweep.gate_stream', return_value=GateResult(True, level='full')) as verify, \
         patch('nceval.sweep._recon_hashes', return_value=['md5']), \
         patch('nceval.sweep.MetricSession') as metrics:
        result = Sweep(SimpleNamespace(), db, 'commit')._check_and_measure(
            encoder, clip, SweepConfig(10), 26, 1, tmp_path, 'fingerprint', stream, 1200)
    assert result
    verify.assert_called_once()
    assert verify.call_args.kwargs['expected_hashes'] == ['md5']
    metrics.assert_not_called()
    assert db.store.call_args.args[1] == {}
    assert db.store.call_args.args[0].passed
    assert db.store.call_args.args[0].gate_level == 'full'


def test_tuning_reference_cache_uses_complete_clip_identity(tmp_path):
    from unittest.mock import patch
    from nceval import tune
    yuv = tmp_path / 'clip.yuv'
    yuv.write_bytes(bytes([23]) * 384 + bytes([42]) * 384)
    clip = tune.ClipRef('c', yuv, 16, 16, 30, 2, 'static', 1000, 'train')
    with patch.object(tune, 'TUNING_DIR', tmp_path / 'cache'):
        ref = tune.trimmed_reference(clip, 1)
    assert ref.read_bytes() == bytes([23]) * 384
    assert tune.clip_fingerprint(clip) in ref.name


def test_gate_level_is_stored_and_reported(tmp_path):
    from nceval.report import format_gate_levels
    from nceval.results import EncodeRecord, ResultsDb
    db = ResultsDb(tmp_path / 'r.sqlite')
    for q, level in [(20, 'full'), (24, 'own-decoder'), (28, 'own-decoder')]:
        db.store(EncodeRecord('enc', 's', 'clip', q, 1, 10, 100, True, '', 'p', 'fp', 'c',
                              gate_level=level), {})
    assert db.gate_levels() == {'enc': {'full': 1, 'own-decoder': 2}}
    assert 'enc: full 1, own-decoder 2' in format_gate_levels(db)
    db.conn.close()


def test_curve_with_an_over_cap_point_is_current(tmp_path):
    import platform as host
    from nceval.framestats import FrameInfo
    from nceval.results import EncodeRecord, ResultsDb
    db = ResultsDb(tmp_path / 'r.sqlite')
    for q, scored in [(20, False), (24, True), (28, True), (32, True), (36, True)]:
        db.store(EncodeRecord('enc', 's', 'clip', q, 2, 10, 100, True, '', host.machine(), 'fp', 'c'),
                 {'psnr_yuv': 40.0} if scored else {}, [FrameInfo(10, 'P')] * 2 if scored else None)
    assert db.curve_is_current('enc', 's', 'clip', 2, host.machine(), 'fp')
    db.conn.close()
