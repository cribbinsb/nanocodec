"""Speed samples must be decoder-gated without timing reconstruction I/O."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import speed
from nceval.clips import Clip
from nceval.codecs import Codec
from nceval.config import Paths
from nceval.encoders.base import CodingStructure, EncoderAdapter


class Adapter(EncoderAdapter):
    codec = Codec.H264
    quantizer_range = (0, 51)

    def __init__(self, name, recon):
        self.name = name
        self.recon = recon

    def verify_recon(self):
        return self.recon

    def command(self, job, paths):
        args = ['fake-encoder', '--name', self.name, '--output', str(job.bitstream)]
        if job.recon is not None:
            args += ['--recon', str(job.recon)]
        return args


def setup_measurement(tmp_path, monkeypatch, *, corrupt=False, failed=False, perf=False,
                      cap=1000, corrupt_perf=False):
    paths = Paths(tmp_path, tmp_path / 'anchors', tmp_path / 'bin')
    clip = Clip(name='pattern', yuv=tmp_path / 'in.yuv', width=16, height=8,
                fps=30, frames=4, source_kbps=cap, content_class='static', split='train')
    commands, gates = [], []
    seen = set()

    def run(command):
        commands.append(command)
        out = Path(command[command.index('--output') + 1])
        data = b'valid-stream' * 20
        name = command[command.index('--name') + 1]
        if corrupt and name in seen:
            data += b'changed'
        seen.add(name)
        out.write_bytes(data)
        if '--recon' in command:
            Path(command[command.index('--recon') + 1]).write_bytes(b'reconstruction')
        return 0.5, 0.25

    def gate(stream, codec, fmt, frames, work, actual_paths, recon=None):
        assert actual_paths == paths and frames == 4
        assert work.is_relative_to(paths.work)
        if recon is not None:
            assert recon.is_relative_to(paths.work) and recon.is_file()
        gates.append(recon is not None)
        return SimpleNamespace(passed=not failed, reason='decoder mismatch' if failed else '')

    def counters(command):
        run(command)
        if corrupt_perf:
            output = Path(command[command.index('--output') + 1])
            output.write_bytes(output.read_bytes() + b'corrupt-perf-output')
        # The encoder can run even when perf reports no usable counters.
        return None, None

    monkeypatch.setattr(speed, '_timed_run', run)
    monkeypatch.setattr(speed, '_perf_usable', lambda: perf)
    monkeypatch.setattr(speed, '_perf_counters', counters)
    monkeypatch.setattr(speed, 'check_stream', gate)
    monkeypatch.setattr(speed, 'check_disk', lambda work: work.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(speed, '_fast_tmp', lambda: str(tmp_path))
    return paths, clip, commands, gates


@pytest.mark.parametrize('perf', [False, True])
def test_all_speed_streams_are_gated_and_only_warmup_writes_recon(tmp_path, monkeypatch, perf):
    paths, clip, commands, gates = setup_measurement(tmp_path, monkeypatch, perf=perf)
    encoders = [Adapter('nanocodec-h264@candidates/example', True), Adapter('anchor', False)]
    result = speed.measure(encoders, clip, CodingStructure(300),
                           {e.name: (40, 100) for e in encoders}, 4, 3, 5, paths)
    expected = 2 * (speed.WARMUP_RUNS + speed.MIN_TIMED_RUNS + int(perf))
    assert len(commands) == expected
    assert len(gates) == len(encoders)
    assert sum('--recon' in command for command in commands) == speed.WARMUP_RUNS
    assert sum(gates) == speed.WARMUP_RUNS
    assert all(r.ms_per_frame == 125 and r.cpu_ms_per_frame == 62.5 for r in result)
    assert all(0 < r.kbps < clip.max_test_kbps for r in result)
    assert list(paths.work.iterdir()) == []


@pytest.mark.parametrize('mode,error', [('failed', 'conformance failed'),
                                        ('corrupt', 'differs from decoder-verified warmup'),
                                        ('cap', 'exceeds source cap')])
def test_bad_speed_samples_cannot_be_reported(tmp_path, monkeypatch, mode, error):
    paths, clip, _, _ = setup_measurement(tmp_path, monkeypatch, failed=mode == 'failed',
                                         corrupt=mode == 'corrupt', cap=1 if mode == 'cap' else 1000)
    with pytest.raises((RuntimeError, ValueError), match=error):
        speed.measure([Adapter('ours', True)], clip, CodingStructure(300),
                      {'ours': (40, 100)}, 4, 3, 5, paths)
    assert list(paths.work.iterdir()) == []


def test_speed_quantizer_selection_ignores_historical_over_cap_points(tmp_path, monkeypatch):
    _, clip, _, _ = setup_measurement(tmp_path, monkeypatch)
    encoder = Adapter('ours', True)
    db = SimpleNamespace(swept_points=lambda *args: [(20, 600), (30, 300), (51, 0)])
    assert speed.choose_quantizers(db, [encoder], 'ip10s-tl1', clip, 'x86_64', 600) == \
        {'ours': (30, 300)}
    db.swept_points = lambda *args: [(20, 600)]
    with pytest.raises(LookupError, match='no legal sweep results'):
        speed.choose_quantizers(db, [encoder], 'ip10s-tl1', clip, 'x86_64', 600)


@pytest.mark.parametrize('perf', [False, True])
def test_reference_timed_bytes_must_match_decoder_verified_warmup(tmp_path, monkeypatch, perf):
    paths, clip, _, gates = setup_measurement(tmp_path, monkeypatch, corrupt=True, perf=perf)
    with pytest.raises(RuntimeError, match='differs from decoder-verified warmup'):
        speed.measure([Adapter('reference', False)], clip, CodingStructure(300),
                      {'reference': (40, 100)}, 4, 3, 5, paths)
    assert gates == [False]
    assert list(paths.work.iterdir()) == []


@pytest.mark.parametrize('recon', [False, True])
def test_perf_output_is_certified_even_without_counters(tmp_path, monkeypatch, recon):
    paths, clip, _, gates = setup_measurement(tmp_path, monkeypatch, perf=True,
                                             corrupt_perf=True)
    with pytest.raises(RuntimeError, match='differs from decoder-verified warmup'):
        speed.measure([Adapter('encoder', recon)], clip, CodingStructure(300),
                      {'encoder': (40, 100)}, 4, 3, 5, paths)
    assert gates == [recon]
    assert list(paths.work.iterdir()) == []
