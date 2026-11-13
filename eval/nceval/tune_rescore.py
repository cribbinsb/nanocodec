"""Rescore certified retained tuning streams with pinned VMAF v1, without encoding.

Example: python -m nceval.tune_rescore --codec hevc --binary ENCODER \
  --work-dir DATA/tuning/work/STUDY-TRIAL --output DATA/shortlist/TRIAL.v1.json
The screening study and its retained artifacts are read only. Output is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

from . import conformance, metrics, tune
from .cachefile import write_atomic
from .codecs import Codec
from .config import Paths, load_paths
from .diskguard import check_disk


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda: source.read(1 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_number(a, b) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def _fresh_clip_fingerprint(clip) -> str:
    # Certificate validation must not reuse the tuner's process-local memo.
    fingerprint = getattr(tune.clip_fingerprint, '__wrapped__', tune.clip_fingerprint)
    return fingerprint(clip)


def retained_points(binary: Path, codec: Codec, work: Path) -> tuple[list[dict], dict]:
    """Validate all identities/certificates before decoding or computing metrics."""
    expected = tune.evaluation_identity(binary, tune.SCREEN_METRIC_PROFILE)
    clips = {c.name: c for c in tune.load_clips('train')}
    checkpoints = sorted((work / 'curve-checkpoints').glob('*.json'))
    _require(bool(checkpoints), 'no completed screening curve checkpoints')
    points = []
    inputs = {}
    used = set()
    for checkpoint in checkpoints:
        saved = json.loads(checkpoint.read_text())
        key, curve = saved['identity'], tune.Curve(**saved['curve'])
        name = checkpoint.stem
        _require(name in clips, f'{name}: not an eligible TRAIN clip')
        clip = clips[name]
        reference = replace(clip, yuv=clip.clean_reference) if clip.clean_reference else clip
        _require(key['evaluation'] == expected, f'{name}: screening binary/tool/protocol identity changed')
        _require(key['metric_profile'] == tune.SCREEN_METRIC_PROFILE and key['keep_streams'], f'{name}: not a retained MS-SSIM screening curve')
        _require(key['clip'] == _fresh_clip_fingerprint(clip), f'{name}: source fingerprint changed')
        _require(key['reference'] == _fresh_clip_fingerprint(reference), f'{name}: reference fingerprint changed')
        _require(key['frames'] == tune.PROXY_FRAMES and 0 < key['frames'] <= clip.frames, f'{name}: frame protocol changed')
        tune.curve_record(curve, tune.SCREEN_METRIC_PROFILE)
        _require(len(curve.kbps) >= tune.MIN_POINTS and len(curve.quantizers) == len(curve.kbps) == len(curve.psnr), f'{name}: incomplete curve')
        _require(len(set(curve.quantizers)) == len(curve.quantizers), f'{name}: duplicate quantizers')
        inputs[str(checkpoint.relative_to(work))] = _sha(checkpoint)
        for index, q in enumerate(curve.quantizers):
            stream = work / 'retained' / f'{name}_q{q}{tune.SUFFIX[codec]}'
            manifest_path = stream.with_suffix(stream.suffix + '.json')
            _require(stream.is_file() and manifest_path.is_file(), f'{name} q{q}: missing retained stream/certificate')
            manifest = json.loads(manifest_path.read_text())
            _require(manifest['binary_sha256'] == expected['binary'] and manifest['overrides'] == key['overrides'], f'{name} q{q}: binary/parameter mismatch')
            _require(manifest['clip'] == name and manifest['clip_fingerprint'] == key['clip'] and manifest['reference_fingerprint'] == key['reference'], f'{name} q{q}: source/reference mismatch')
            _require(manifest['metric_profile'] == tune.SCREEN_METRIC_PROFILE, f'{name} q{q}: wrong metric profile')
            _require((manifest['frames'], manifest['width'], manifest['height'], manifest['fps'], manifest['quantizer'], manifest['intra_period']) ==
                     (key['frames'], clip.width, clip.height, clip.fps, q, round(clip.fps * tune.INTRA_PERIOD_SECONDS)), f'{name} q{q}: geometry/frame/QP/GOP mismatch')
            _require(manifest['conformance'] == 'two decoders and encoder reconstruction passed', f'{name} q{q}: missing original conformance gate')
            _require(_sha(stream) == manifest['stream_sha256'], f'{name} q{q}: retained stream hash mismatch')
            rate = tune.payload_bytes(stream, codec, key['frames']) * 8 / 1000 / (key['frames'] / clip.fps)
            _require(rate <= clip.max_test_kbps and _same_number(manifest['source_cap_kbps'], clip.max_test_kbps), f'{name} q{q}: source cap changed/exceeded')
            _require(_same_number(rate, manifest['kbps']) and _same_number(rate, curve.kbps[index]), f'{name} q{q}: rate mismatch')
            _require(_same_number(manifest['psnr_yuv'], curve.psnr[index]) and _same_number(manifest['ms_ssim'], curve.ms_ssim[index]), f'{name} q{q}: screening score mismatch')
            inputs[str(manifest_path.relative_to(work))] = _sha(manifest_path)
            used.add(manifest_path.resolve())
            points.append({'clip': clip, 'reference': reference, 'stream': stream, 'manifest': manifest,
                           'manifest_sha256': inputs[str(manifest_path.relative_to(work))]})
    all_manifests = {p.resolve() for p in (work / 'retained').glob('*' + tune.SUFFIX[codec] + '.json')}
    _require(all_manifests == used, 'retained directory contains incomplete or uncheckpointed points')
    return points, {'screening_evaluation': expected, 'input_manifest_hashes': inputs}


def decode_certified(point: dict, codec: Codec, decoded: Path, paths: Paths) -> None:
    """One standard decode must equal the previously two-decoder-gated encoder YUV."""
    decoder = conformance.DECODERS[codec][0]
    error = conformance.decode_to_file(decoder, point['stream'], decoded, codec,
                                       point['clip'].fmt, paths)
    _require(not error, f"{point['clip'].name}: decoder failed: {error}")
    expected_bytes = point['manifest']['frames'] * point['clip'].fmt.frame_size
    _require(decoded.is_file() and decoded.stat().st_size == expected_bytes, 'decoded frame count/geometry mismatch')
    _require(_sha(decoded) == point['manifest']['recon_sha256'], 'decoded reconstruction hash mismatch')


def rescore(binary: Path, codec: Codec, work: Path, output: Path, paths: Paths | None = None) -> dict:
    work, output = work.resolve(), output.resolve()
    _require(not output.is_relative_to(work), 'V1 output must be outside the read-only screening work directory')
    paths = paths or load_paths()
    points, provenance = retained_points(binary, codec, work)
    identity = {'codec': codec.value, 'work_dir': str(work), **provenance,
                'metric_profile': 'vmaf-v1-retained-stream-rescore',
                'vmaf_protocol': metrics.vmaf_v1_protocol(paths, subsample=1)}
    result = (json.loads(output.read_text()) if output.exists() else
              {'identity': identity, 'points': {}, 'curves': {},
               'gate': 'original two independent decoders+encoder recon, then one fresh decode with exact saved encoder-recon SHA',
               'screening_study_modified': False})
    _require(result['identity'] == identity, 'V1 rescore cache identity changed')
    planned = {f"{p['clip'].name}:q{p['manifest']['quantizer']}" for p in points}
    _require(set(result['points']) <= planned, 'V1 cache contains unplanned points')
    def save():
        write_atomic(output, json.dumps(result, indent=2) + '\n')
    save()
    for point in points:
        clip, manifest = point['clip'], point['manifest']
        label = f"{clip.name}:q{manifest['quantizer']}"
        if label in result['points']:
            cached = result['points'][label]
            _require(cached['manifest_sha256'] == point['manifest_sha256'] and
                     cached['stream_sha256'] == manifest['stream_sha256'] and
                     cached['decoded_encoder_recon_sha256'] == manifest['recon_sha256'] and
                     math.isfinite(cached['vmaf_v1']) and
                     len(cached['per_frame']) == manifest['frames'] and
                     all('vmaf_v1' in f for f in cached['per_frame']),
                     f'{label}: incomplete/stale V1 point cache')
            continue
        paths.work.mkdir(parents=True, exist_ok=True)
        check_disk(paths.work)
        with tempfile.TemporaryDirectory(prefix='nc-retained-v1-', dir=paths.work) as tmp:
            decoded = Path(tmp) / 'decoded.yuv'
            decode_certified(point, codec, decoded, paths)
            reference = (point['reference'].yuv if manifest['frames'] == point['reference'].frames else
                         tune.trimmed_reference(point['reference'], manifest['frames']))
            measured = metrics.vmaf_v1(reference, decoded, clip.fmt, paths, subsample=1)
            _require(len(measured.per_frame) == manifest['frames'], f'{label}: incomplete V1 per-frame metrics')
            result['points'][label] = {'clip': clip.name, 'quantizer': manifest['quantizer'],
                'kbps': manifest['kbps'], 'psnr': manifest['psnr_yuv'],
                'vmaf_v1': float(measured.summary['vmaf_v1']), 'per_frame': measured.per_frame,
                'manifest_sha256': point['manifest_sha256'], 'stream_sha256': manifest['stream_sha256'],
                'decoded_encoder_recon_sha256': manifest['recon_sha256']}
            save()
            print('RESCORED', label, result['points'][label]['vmaf_v1'], flush=True)
    for name in sorted({p['clip'].name for p in points}):
        scored = sorted((v for v in result['points'].values() if v['clip'] == name), key=lambda v: v['quantizer'])
        curve = tune.Curve(kbps=[v['kbps'] for v in scored], psnr=[v['psnr'] for v in scored],
                           quantizers=[v['quantizer'] for v in scored], vmaf_v1=[v['vmaf_v1'] for v in scored])
        result['curves'][name] = tune.curve_record(curve)
    result['complete'] = True
    save()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--codec', choices=[c.value for c in Codec], required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rescore(args.binary, Codec(args.codec), args.work_dir, args.output)


if __name__ == '__main__':
    main()
