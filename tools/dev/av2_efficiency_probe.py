#!/usr/bin/env python3
"""Compare AV2 zero-motion and finalist RD on three existing 16-frame clips."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vp9_efficiency_probe import (NAMES, Codec, PsnrMetric, StreamSpec, YuvFormat,
                                 command, load_paths, run_point, sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', nargs='+', default=NAMES)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--param', action='append', default=[],
                        help='extra NAME=VALUE override for RD variant only')
    parser.add_argument('--common-param', action='append', default=[],
                        help='NAME=VALUE override applied to both variants')
    parser.add_argument('--qmap', type=json.loads, default=None,
                        help='JSON mapping clip names to selected quantizer lists')
    parser.add_argument('--variants', nargs='+', choices=('zero', 'rd'), default=['zero', 'rd'])
    args = parser.parse_args()
    args.cli = args.cli.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = load_paths()
    manifest = json.loads((paths.data_root / 'clips/quick360.json').read_text())
    clips = [next(c for c in manifest['clips'] if c['name'] == name) for name in args.clips]
    qmap = args.qmap or {name: ([144, 160] if name.startswith("static_") else [208, 224])
                            for name in args.clips}
    from nceval import budget
    cost = sum(budget.point_cpu_seconds(c['width'], c['height'], 16)
               * len(qmap.get(c['name'], [])) * len(args.variants) for c in clips)
    budget.check('AV2 diagnostic', cost, 'reduce the selected quantizers or variants', False)
    metadata = {'binary': str(args.cli), 'sha256': hashlib.sha256(args.cli.read_bytes()).hexdigest(),
                'clips': clips, 'frames': 16, 'qindices': qmap, 'gop': 300,
                'common_params': args.common_param,
                'rd_params': ['av2.motion_search=1', *args.param],
                'measurement': 'streamed PSNR plus full native/AVM/dav2d pixel gate; no speed claims'}
    metadata_path = args.output / 'metadata.json'
    quality_path = args.output / 'quality.json'
    rows = (json.loads(quality_path.read_text()) if metadata_path.exists() and quality_path.exists()
            and json.loads(metadata_path.read_text()) == metadata else [])
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    for clip in clips:
        source = paths.data_root / clip['yuv']
        spec = StreamSpec(source, YuvFormat(clip['width'], clip['height']), 16, clip['fps'], paths)
        for qindex in qmap[clip['name']]:
            for variant, params in [('zero', tuple(args.common_param)),
                                    ('rd', (*args.common_param, 'av2.motion_search=1', *args.param))]:
                if variant not in args.variants:
                    continue
                output = args.output / f'{clip["name"]}-{variant}-q{qindex}.ivf'
                existing = next((r for r in rows if r['clip'] == clip['name']
                                 and r['variant'] == variant and r['qindex'] == qindex), None)
                if existing and output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == existing.get('bitstream_sha256'):
                    print(json.dumps(existing), flush=True)
                    continue
                point = run_point(
                    lambda recon: command(args.cli, 'native', clip, source, qindex, 16,
                                          output, recon, codec='av2', params=params),
                    output, Codec.AV2, spec, paths, [PsnrMetric(spec)],
                    own_decoder=args.cli, serial_gate=True)
                if not point.gate.passed:
                    raise RuntimeError(f'{output.name}: {point.gate.reason}')
                metric = point.measurement
                lengths = sizes(output, fourcc=b'AV02')
                row = {'clip': clip['name'], 'variant': variant, 'qindex': qindex,
                       'frames': len(lengths), 'payload_bytes': sum(lengths),
                       'key_bytes': lengths[0], 'p_bytes': sum(lengths[1:]),
                       'kbps': sum(lengths) * 8 * clip['fps'] / 16 / 1000,
                       'source_kbps': clip['source_kbps'],
                       'rate_cap_kbps': 0.5 * clip['source_kbps'],
                       'within_rate_cap': sum(lengths) * 8 * clip['fps'] / 16 / 1000 <= 0.5 * clip['source_kbps'],
                       'gate': point.gate.level,
                       'bitstream_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), **metric.summary,
                       'psnr_y_key': metric.per_frame[0]['psnr_y'],
                       'psnr_y_p': sum(f['psnr_y'] for f in metric.per_frame[1:]) / 15}
                rows = [r for r in rows if not (r['clip'] == clip['name'] and r['variant'] == variant and r['qindex'] == qindex)]
                rows.append(row)
                (args.output / 'quality.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
