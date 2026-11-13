#!/usr/bin/env python3
"""Four streamed, fully gated CBR control points; no quality scoring or raw copies."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'eval'))
from nceval.cbr import Family
from nceval.cbr_stats import replay_buffer
from nceval.clips import load_clipset
from nceval.codecs import Codec
from nceval.config import load_paths
from nceval.metrics import StreamSpec
from nceval.point import run_point
from nceval.yuv import YuvFormat

parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
paths = load_paths()
args.output.mkdir(parents=True, exist_ok=True)
clips = load_clipset(paths.clips / 'quick360.json', paths.data_root).clips
report = {'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
          'frames': 90, 'buffer_ms': 1000, 'frame_drop_enabled': False,
          'quality_scored': False, 'points': []}
for codec in (Codec.VP9, Codec.AV2):
    for prefix, target in [('static_', 250), ('bodycam_', 100)]:
        clip = next(c for c in clips if c.name.startswith(prefix))
        stem = args.output / f'{codec.value}-{clip.name}'
        log, stream = stem.with_suffix('.csv'), stem.with_suffix('.ivf')
        family = Family(f'nanocodec-{codec.value}', codec, args.binary,
                        ('--rate-log', str(log)), own=True)
        spec = StreamSpec(clip.yuv, YuvFormat(clip.width, clip.height), 90, clip.fps, paths)
        result = run_point(lambda recon: family.command(clip, 90, 300, stream, recon, kbps=target),
                           stream, codec, spec, paths, own_decoder=args.binary, serial_gate=True)
        assert result.gate.passed, result.gate.reason
        with log.open() as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 90
        bits = [int(r['bytes']) * 8 for r in rows]
        actual = sum(bits) * clip.fps / 90 / 1000
        point = {'codec': codec.value, 'clip': clip.name, 'target_kbps': target,
                 'actual_kbps': actual, 'rate_error_percent': 100 * (actual / target - 1),
                 'source_cap_kbps': clip.max_test_kbps, 'within_cap': clip.allows_kbps(actual),
                 'quantizer_min': min(int(r['planned_qp']) for r in rows),
                 'quantizer_max': max(int(r['planned_qp']) for r in rows),
                 'dropped_frames': sum(r['type'] == 'Dropped' for r in rows),
                 'controller_buffer_overflows': int(rows[-1]['buffer_overflows']),
                 'strict_buffer_replay': asdict(replay_buffer(bits, target * 1000, clip.fps, 1000)),
                 'gate': result.gate.level, 'stream_sha256': hashlib.sha256(stream.read_bytes()).hexdigest()}
        report['points'].append(point)
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(point), flush=True)
