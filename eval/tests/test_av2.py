"""AV2 adapter uses final AVM quantizer controls and bounded low-delay settings."""
from pathlib import Path
from unittest.mock import patch

import pytest
from nceval.clips import Clip
from nceval.codecs import Codec
from nceval.encoders.anchors import AvmLowDelay
from nceval.encoders.base import CodingStructure, EncodeJob
from nceval.sweep import payload_bytes


def test_final_avm_quantizers_and_low_delay():
    clip = Clip('synthetic', Path('/in.yuv'), 16, 16, 30, 3, 1000, 'synthetic_noise', 'train')
    encoder = AvmLowDelay(9)
    with patch('nceval.encoders.anchors.find_tool', return_value=Path('/avmenc')):
        cmd = encoder.command(EncodeJob(clip, 140, CodingStructure(2), 3, Path('/out.ivf')), None)
        assert '--qp=140' in cmd and '--min-qp=140' in cmd and '--max-qp=140' in cmd
        assert '--min-q=140' not in cmd
        assert '--lag-in-frames=0' in cmd and '--auto-alt-ref=0' in cmd
        assert '--enable-fwd-kf=0' in cmd and '--threads=1' in cmd
        with pytest.raises(ValueError, match='temporal layers'):
            encoder.command(EncodeJob(clip, 140, CodingStructure(2, 2), 3, Path('/out.ivf')), None)
        with pytest.raises(ValueError, match='quantizer'):
            encoder.command(EncodeJob(clip, 256, CodingStructure(2), 3, Path('/out.ivf')), None)


def test_av2_payload_excludes_ivf_overhead(tmp_path):
    stream = tmp_path / 'av2.ivf'
    frame = (22).to_bytes(4, 'little') + bytes(8 + 22)
    stream.write_bytes(bytes(32) + 2 * frame)
    assert payload_bytes(stream, Codec.AV2, 2) == 44


def test_av2_ivf_statistics_do_not_require_ffprobe(tmp_path):
    import struct
    from nceval.framestats import probe_frames
    header = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV02', 16, 16, 30, 1, 2, 0)
    packets = [bytes([1, 2 << 2, 1, 4 << 2]), bytes([1, 7 << 2])]
    data = header + b''.join(struct.pack('<IQ', len(p), i) + p for i, p in enumerate(packets))
    stream = tmp_path / 'frames.ivf'
    stream.write_bytes(data)
    with patch('nceval.framestats.subprocess.run', side_effect=AssertionError('ffprobe not needed')):
        frames = probe_frames(stream, None)
    assert [(f.bytes, f.frame_type) for f in frames] == [(4, 'I'), (2, 'P')]
    stream.write_bytes(data[:-1])
    with pytest.raises(ValueError, match='truncated'):
        probe_frames(stream, None)
    stream.write_bytes(header + struct.pack('<IQ', 2, 0) + bytes([127, 4 << 2]))
    with pytest.raises(ValueError, match='OBU extent'):
        probe_frames(stream, None)


@pytest.mark.parametrize('reference', ['avm', 'vpx'])
def test_new_reference_fingerprints_track_actual_binaries(tmp_path, monkeypatch, reference):
    from nceval import fingerprint as fp
    from nceval.encoders import anchors
    names = ['avmenc', 'avmdec', 'dav2d', 'vpxenc', 'vpxdec', 'ffmpeg']
    for name in names:
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(fp, 'measurement_fingerprint', lambda *args: 'metrics')
    monkeypatch.setattr(fp, 'find_tool', lambda name, paths: tmp_path / name)
    monkeypatch.setattr(anchors, 'find_tool', lambda name, paths: tmp_path / name)
    adapter = anchors.AvmLowDelay(9) if reference == 'avm' else anchors.LibvpxRealtime(8)
    structure = CodingStructure(300)
    first = fp.encoder_fingerprint(adapter, structure, None)
    encoder = 'avmenc' if reference == 'avm' else 'vpxenc'
    (tmp_path / encoder).write_text('rebuilt encoder')
    second = fp.encoder_fingerprint(adapter, structure, None)
    assert first != second
    decoder = 'dav2d' if reference == 'avm' else 'vpxdec'
    (tmp_path / decoder).write_text('rebuilt decoder')
    assert second != fp.encoder_fingerprint(adapter, structure, None)
