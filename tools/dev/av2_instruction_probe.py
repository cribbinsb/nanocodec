#!/usr/bin/env python3
"""Bounded, gated AV2 instruction counts: one key plus two P frames per case."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from vp9_efficiency_probe import (Codec, YuvFormat, command, cpu_slots, gate_stream,
                                 invoke, load_paths, sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pre-skip-energy', type=float, default=0.0)
    args = parser.parse_args()
    args.cli = args.cli.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = load_paths()
    manifest = json.loads((paths.data_root / 'clips/quick360.json').read_text())
    selected = [('static_chirla_s000c3_360p', 144), ('pan_jaad_0117_360p', 208)]
    common = ('av2.cdf_adaptation=1', 'av2.motion_search=0', 'av2.residual_skip_rd=0')
    cases = [('zero', ()), ('skip', ('av2.residual_skip_rd=1', f'av2.pre_tx_skip_energy={args.pre_skip_energy}')),
             ('motion', ('av2.residual_skip_rd=1', 'av2.motion_search=1',
                         'av2.search_lambda_scale=0.135', 'av2.early_sse_per_lambda=0.025',
                         'av2.subpel_steps=1', f'av2.pre_tx_skip_energy={args.pre_skip_energy}'))]
    metadata = {'binary': str(args.cli), 'sha256': hashlib.sha256(args.cli.read_bytes()).hexdigest(),
                'clips': selected, 'common': common, 'cases': cases, 'frames': 3,
                'measurement': 'callgrind instruction counts; key-subtracted mean of two P frames; no wall-clock speed claim'}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    for name, qindex in selected:
        clip = next(c for c in manifest['clips'] if c['name'] == name)
        key_count = None
        for variant, extra, frames in [('key', (), 1), *((n, p, 3) for n, p in cases)]:
            stream = args.output / f'{name}-{variant}.ivf'
            report = args.output / f'{name}-{variant}.callgrind'
            with cpu_slots(1):
                invoke(['valgrind', '--tool=callgrind', f'--callgrind-out-file={report}',
                        *command(args.cli, 'native', clip, paths.data_root / clip['yuv'],
                                 qindex, frames, stream, codec='av2', params=(*common, *extra))])
                gate = gate_stream(stream, Codec.AV2, YuvFormat(clip['width'], clip['height']),
                                   frames, paths, own_decoder=args.cli, serial=True, hold_slots=False)
            if not gate.passed:
                raise RuntimeError(f'{stream.name}: {gate.reason}')
            count = next(int(line.split()[1]) for line in report.read_text().splitlines()
                         if line.startswith('summary:'))
            report.unlink()
            if variant == 'key':
                key_count = count
            row = {'clip': name, 'qindex': qindex, 'case': variant, 'frames': frames,
                   'instructions': count, 'mean_p_instructions': (count-key_count)/2 if frames == 3 else None,
                   'payload_bytes': sum(sizes(stream, fourcc=b'AV02')), 'gate': gate.level}
            rows.append(row)
            (args.output / 'instructions.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
