#!/usr/bin/env python3
"""Bounded VP9 diagnostic: three existing clips, two QPs, sixteen frames.

Uses shared streamed point/metric/gate code, never copies raw video. Optional
callgrind probes use only one key and one key+P per encoder on a chosen clip.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'eval'))
from nceval.codecs import Codec
from nceval.config import load_paths
from nceval.conformance import gate_stream
from nceval.metrics import StreamSpec, PsnrMetric, MetricSession
from nceval.point import run_point
from nceval.slots import cpu_slots, low_priority
from nceval.yuv import YuvFormat

NAMES = ('static_chirla_s000c3_360p', 'crowd_mot17_09_360p', 'pan_jaad_0117_360p')


def command(binary, kind, clip, source, qindex, frames, output, recon=None, codec="vp9", params=()):
    if kind == 'native':
        result = [str(binary), 'encode', '--codec', codec, '--input', str(source),
                  '--width', str(clip['width']), '--height', str(clip['height']),
                  '--fps', str(clip['fps']), '--frames', str(frames), '--qp', str(qindex),
                  '--intra-period', '300', '--output', str(output)]
        if recon:
            result += ['--recon', str(recon)]
        for param in params:
            result += ["--param", param]
        return result
    q = qindex // 4
    return [str(binary), '--codec=vp9', '--disable-warning-prompt', '--rt', '--cpu-used=8', '--threads=1',
            '--row-mt=0', '--tile-columns=0', '--tile-rows=0', '--passes=1',
            '--lag-in-frames=0', '--auto-alt-ref=0', '--aq-mode=0', '--end-usage=q',
            f'--cq-level={q}', f'--min-q={q}', f'--max-q={q}', '--kf-min-dist=300',
            '--kf-max-dist=300', '--test-decode=off', '--ivf',
            f'--width={clip["width"]}', f'--height={clip["height"]}', '--fps=30/1',
            f'--limit={frames}', '-o', str(output), str(source)]


def invoke(cmd):
    result = subprocess.run(low_priority(cmd), stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return result.stderr


def sizes(path, fourcc=b'VP90'):
    data = path.read_bytes()
    assert data[:4] == b'DKIF' and data[8:12] == fourcc
    offset, result, previous_timestamp = 32, [], -1
    while offset < len(data):
        length, timestamp = struct.unpack_from('<IQ', data, offset)
        assert timestamp > previous_timestamp
        previous_timestamp = timestamp
        result.append(length)
        offset += 12 + length
    assert offset == len(data)
    return result


def quality(args, paths, clips, vpx):
    result_path = args.output / 'quality.json'
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    for clip in clips:
        source = paths.data_root / clip['yuv']
        fmt = YuvFormat(clip['width'], clip['height'])
        spec = StreamSpec(source, fmt, 16, clip['fps'], paths)
        for qindex in args.qindices:
            for kind, binary in [('native', args.cli), ('libvpx-rt8', vpx)]:
                if kind not in args.encoders: continue
                output = args.output / f'{clip["name"]}-{kind}-q{qindex}.ivf'
                existing = next((r for r in rows if (r['clip'],r['encoder'],r['qindex']) == (clip['name'],kind,qindex)),None)
                if existing and output.exists() and existing.get('bitstream_sha256') == hashlib.sha256(output.read_bytes()).hexdigest():
                    print('RESUME '+output.name,flush=True)
                    continue
                if kind == 'native':
                    point = run_point(lambda recon: command(binary, kind, clip, source, qindex, 16, output, recon, params=args.param),
                                      output, Codec.VP9, spec, paths, [PsnrMetric(spec)],
                                      own_decoder=args.cli, serial_gate=True)
                    gate, metric = point.gate, point.measurement
                else:
                    with cpu_slots(1):
                        invoke(command(binary, kind, clip, source, qindex, 16, output))
                        with MetricSession(spec, [PsnrMetric(spec)]) as session:
                            gate = gate_stream(output, Codec.VP9, fmt, 16, paths,
                                               extra_sinks=session.sinks, serial=True, hold_slots=False)
                            metric = session.result(16) if gate.passed else None
                if not gate.passed:
                    raise RuntimeError(f'{output.name}: {gate.reason}')
                lengths = sizes(output)
                row = {'clip': clip['name'], 'encoder': kind, 'qindex': qindex,
                       'frames': len(lengths), 'payload_bytes': sum(lengths),
                       'key_bytes': lengths[0], 'p_bytes': sum(lengths[1:]),
                       'kbps': sum(lengths) * 8 * clip['fps'] / 16 / 1000,
                       'gate': gate.level, **metric.summary,
                       'psnr_y_key': metric.per_frame[0]['psnr_y'],
                       'psnr_y_p': sum(f['psnr_y'] for f in metric.per_frame[1:]) / 15}
                row['source_kbps'] = clip['source_kbps']
                row['cap_kbps'] = 0.5 * clip['source_kbps']
                row['in_domain'] = row['kbps'] <= row['cap_kbps']
                row['bitstream_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
                if existing: rows.remove(existing)
                rows.append(row)
                (args.output / 'quality.json').write_text(json.dumps(rows, indent=2)+'\n')
                print(json.dumps(row), flush=True)


def profile(args, paths, clips, vpx):
    clip = next(c for c in clips if c['name'] == args.profile)
    source = paths.data_root / clip['yuv']
    fmt = YuvFormat(clip['width'], clip['height'])
    results = []
    for kind, binary in [('native', args.cli), ('libvpx-rt8', vpx)]:
        for frames in (1, 2):
            output = args.output / f'cost-{kind}-{frames}.ivf'
            report = args.output / f'callgrind-{kind}-{frames}'
            with cpu_slots(1):
                invoke(['valgrind', '--tool=callgrind', f'--callgrind-out-file={report}',
                        *command(binary, kind, clip, source, 160, frames, output, params=args.param)])
                gate = gate_stream(output, Codec.VP9, fmt, frames, paths,
                                   own_decoder=args.cli if kind == 'native' else None,
                                   serial=True, hold_slots=False)
            if not gate.passed:
                raise RuntimeError(f'profile gate: {gate.reason}')
            count = next(int(line.split()[1]) for line in report.read_text().splitlines()
                         if line.startswith('summary:'))
            if frames == 2:
                annotation = subprocess.check_output(['callgrind_annotate', '--inclusive=yes',
                                                      '--threshold=90', str(report)], text=True)
                (args.output / f'functions-{kind}.txt').write_text(annotation)
            report.unlink()
            row = {'clip': clip['name'], 'encoder': kind, 'frames': frames,
                   'instructions': count, 'gate': gate.level}
            results.append(row)
            (args.output / 'instructions.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=NAMES)
    parser.add_argument('--param', action='append', default=[], help='native encoder key=value override')
    parser.add_argument('--clips', choices=NAMES, nargs='+', default=NAMES)
    parser.add_argument('--qindices', type=int, nargs='+', default=(96,160))
    parser.add_argument('--encoders', choices=('native','libvpx-rt8'), nargs='+', default=('native','libvpx-rt8'))
    args = parser.parse_args()
    args.cli = args.cli.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = load_paths()
    manifest = json.loads((paths.data_root / 'clips/quick360.json').read_text())
    clips = [next(c for c in manifest['clips'] if c['name'] == name) for name in args.clips]
    from nceval import budget
    if args.profile:
        cost = sum(budget.point_cpu_seconds(c['width'], c['height'], 6) * 100
                   for c in clips if c['name'] == args.profile)
    else:
        cost = sum(budget.point_cpu_seconds(c['width'], c['height'], 16)
                   * len(args.qindices) * len(args.encoders) for c in clips)
    budget.check('VP9 diagnostic', cost, 'reduce clips or quantizers', False)
    vpx = Path(shutil.which('vpxenc')).resolve()
    metadata = {'params': args.param, 'native_binary': str(args.cli), 'native_sha256': hashlib.sha256(args.cli.read_bytes()).hexdigest(),
                'libvpx_binary': str(vpx), 'libvpx_sha256': hashlib.sha256(vpx.read_bytes()).hexdigest(),
                'clips': clips, 'frames': 16, 'qindices': args.qindices, 'gop': 300,
                'reference_command': command(vpx,'libvpx-rt8',clips[0],Path('SOURCE'),96,16,Path('OUTPUT'))}
    meta_path = args.output / ('profile-metadata.json' if args.profile else 'metadata.json')
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for key in ('native_sha256','libvpx_sha256','clips'):
            if old[key] != metadata[key]: raise ValueError('Probe identity changed; use another output directory')
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    (profile if args.profile else quality)(args, paths, clips, vpx)


if __name__ == '__main__':
    main()
