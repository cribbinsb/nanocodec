#!/usr/bin/env python3
"""Bounded AV02 CLI CQP/CBR gate; raw I420 remains in anonymous pipes/memory."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'eval'))
from nceval.codecs import Codec
from nceval.config import load_paths
from nceval.conformance import gate_stream
from nceval.slots import cpu_slots, low_priority
from nceval.yuv import YuvFormat, frame_digest


def picture_types(data: bytes) -> list[int]:
    assert data[:4] == b'DKIF' and data[8:12] == b'AV02'
    result = []
    offset = 32
    while offset < len(data):
        size, timestamp = struct.unpack_from('<IQ', data, offset)
        offset += 12
        packet = data[offset:offset + size]
        assert len(packet) == size and timestamp == len(result)
        offset += size
        position = 0
        pictures = []
        while position < size:
            length = shift = 0
            while True:
                byte = packet[position]
                position += 1
                length |= (byte & 127) << shift
                if byte < 128:
                    break
                shift += 7
                assert shift < 64
            assert length > 0 and position + length <= size
            kind = (packet[position] >> 2) & 31
            if kind in (4, 7):
                pictures.append(kind)
            position += length
        assert len(pictures) == 1
        result.extend(pictures)
    assert offset == len(data)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--param', action='append', default=[])
    args = parser.parse_args()
    width, height, frames = 72, 40, 9
    frame_bytes = width * height * 3 // 2
    source = b''.join(bytes((i * 13 + i // width * 7 + (f // 2) * 19 * ((i // 8) % 2)) % 256
                            for i in range(frame_bytes)) for f in range(frames))
    cases = [('lossless-periodic', 3, ['--qp', '0']),
             ('lossy-all-key', 1, ['--qp', '160']),
             ('cbr-periodic', 3, ['--bitrate', '100', '--qp', '140', '--buffer-ms', '300'])]
    with cpu_slots(1), tempfile.TemporaryDirectory(prefix='nc-av2-cli-') as temporary:
        for name, period, rate_args in cases:
            path = Path(temporary) / f'{name}.ivf'
            rate_log = Path(temporary) / f'{name}.csv'
            command = [str(args.cli), 'encode', '--codec', 'av2', '--input', '/dev/stdin',
                       '--output', str(path), '--recon', '/dev/stdout', '--width', str(width),
                       '--height', str(height), '--fps', '30', '--frames', str(frames),
                       '--intra-period', str(period), *rate_args]
            for override in args.param:
                command += ['--param', override]
            if name.startswith('cbr'):
                command += ['--rate-log', str(rate_log)]
            encoded = subprocess.run(low_priority(command), input=source, capture_output=True,
                                     check=True, timeout=120)
            recon = encoded.stdout
            assert len(recon) == len(source), encoded.stderr.decode(errors='replace')
            if name.startswith('lossless'):
                assert recon == source
            types = picture_types(path.read_bytes())
            assert types == [4 if i % period == 0 else 7 for i in range(frames)], types
            decoded = subprocess.run(low_priority([str(args.cli), 'decode', '--codec', 'av2',
                                     '--input', str(path), '--output', '/dev/stdout']),
                                     capture_output=True, check=True, timeout=120).stdout
            assert decoded == recon
            expected = [frame_digest(recon[i:i + frame_bytes])
                        for i in range(0, len(recon), frame_bytes)]
            gate = gate_stream(path, Codec.AV2, YuvFormat(width, height), frames, load_paths(),
                               expected_hashes=expected, serial=True, hold_slots=False)
            quantizers = []
            if rate_log.exists():
                rows = list(csv.DictReader(rate_log.open()))
                assert len(rows) == frames
                quantizers = [int(row['qp']) for row in rows]
                assert len(set(quantizers)) > 1, 'CBR did not adjust quantizers'
            print(f'{name}: frames={frames} bytes={path.stat().st_size} types={types} '
                  f'quantizers={quantizers} gate={gate.level} passed={gate.passed} '
                  f'reason={gate.reason}', flush=True)
            assert gate.passed, gate.reason


if __name__ == '__main__':
    main()
