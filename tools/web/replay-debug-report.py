#!/usr/bin/env python3
"""Reproduce a browser demo debug report natively and check every number in it.

The page's "Debug report" button (web/debug-report.js) saves its whole state as
Markdown with one JSON block. This tool reads that block and, independently of
the page:

- recomputes each view's shown bitrate, PSNR-YUV, encode and decode times from
  the report's own window with the page's formulas;
- for a pre-encoded view, checks the stream file's SHA-256 and every reported
  access unit's hash and size against the file;
- for a live view of a sample clip (the frames are regenerated with
  tools/web/make-reference-streams.py `write_source`, the demo's own source
  frames), encodes the same frames with the native CLI at the view's
  configuration (live rate changes become --rate-change) and checks every
  reported access unit's hash and size: the WASM and native encoders are the
  same code, so the streams must be identical;
- decodes the stream (the native one, a pre-encoded file, or a stream embedded
  in the report) with an independent decoder (ffmpeg; AV2 with nanocodec's
  own) and scores every reported frame against its source frame: the page's
  PSNR-YUV and, with analysis on, PSNR-Y/Cb/Cr must agree within PSNR_TOLERANCE_DB.

It prints one row per checked number and exits non-zero on any mismatch. A
webcam, local file or test scene cannot be regenerated: their statistics and
any embedded streams are still checked. Scratch files live in a temporary
directory removed on exit; no decoded video is kept.

  python3 tools/web/replay-debug-report.py REPORT.md [--nanocodec BIN]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / 'web'
DATA = Path(os.environ.get('NC_DATA_ROOT', Path.home() / 'nanocodec-data'))
SCHEMA = 'nanocodec-debug-report/1'
SOURCE_FPS = 30
PSNR_TOLERANCE_DB = 0.01
# Shown numbers are rounded: bitrate to 1 kbps, PSNR and times to 0.1.
SHOWN_DECIMALS = {'bitrate': 0, 'psnr': 1, 'encode': 1, 'decode': 1}
MAX_PSNR_DB = 100.0
PLANE_WEIGHTS = (6, 1, 1)
PLANES = ('Y', 'Cb', 'Cr')
IVF_CODECS = {'av1': b'AV01', 'vp9': b'VP90', 'av2': b'AV02'}
IVF_HEADER, IVF_FRAME_HEADER = 32, 12
# The CLI's --preset for the page's preset names (web/options.js cliPreset).
PRESET_SUFFIX = {'fast': 'fast', 'faster': 'faster', 'fastest': 'fastest', 'svc': 'svc', 'svc-fast': 'svc-fast',
                 'svc3': 'svc3', 'svc3-fast': 'svc3-fast'}
# A replay longer than this many frames is not attempted (30 minutes at 30 fps).
MAX_REPLAY_FRAMES = 54000


@dataclass
class Check:
    view: str
    item: str
    page: object
    independent: object
    ok: bool
    note: str = ''


def parse_report(path: Path) -> dict:
    text = path.read_text()
    start = text.rfind('```json')
    if start < 0:
        raise SystemExit(f'{path}: no machine-readable state')
    body = text[start + len('```json'):text.index('\n```', start + len('```json'))]
    report = json.loads(body)
    if report.get('schema') != SCHEMA:
        raise SystemExit(f'{path}: schema {report.get("schema")}, this tool reads {SCHEMA}')
    return report


def generator():
    spec = importlib.util.spec_from_file_location('make_reference_streams', REPO / 'tools/web/make-reference-streams.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['make_reference_streams'] = module
    spec.loader.exec_module(module)
    return module


def frame_bytes(width: int, height: int) -> int:
    return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)


def ivf_packets(data: bytes) -> list[bytes]:
    packets, at = [], IVF_HEADER
    while at + IVF_FRAME_HEADER <= len(data):
        size = int.from_bytes(data[at:at + 4], 'little')
        at += IVF_FRAME_HEADER
        packets.append(data[at:at + size])
        at += size
    return packets


def ivf_file(codec: str, width: int, height: int, packets: list[bytes]) -> bytes:
    header = bytearray(IVF_HEADER)
    header[0:4] = b'DKIF'
    header[6:8] = IVF_HEADER.to_bytes(2, 'little')
    header[8:12] = IVF_CODECS[codec]
    header[12:14] = width.to_bytes(2, 'little')
    header[14:16] = height.to_bytes(2, 'little')
    header[16:20] = (1).to_bytes(4, 'little')
    header[20:24] = (1).to_bytes(4, 'little')
    header[24:28] = len(packets).to_bytes(4, 'little')
    out = bytearray(header)
    for index, packet in enumerate(packets):
        out += len(packet).to_bytes(4, 'little') + index.to_bytes(8, 'little') + packet
    return bytes(out)


def decode(codec: str, stream: Path, width: int, height: int, nanocodec: Path | None) -> tuple[np.ndarray, str]:
    """Every frame of `stream` as packed I420 rows, and the decoder used."""
    if codec == 'av2':
        if not nanocodec:
            raise SystemExit('AV2 needs --nanocodec to decode')
        raw = subprocess.run([str(nanocodec), 'decode', '--codec', 'av2', '--input', str(stream), '--output', '-'],
                             capture_output=True, check=True).stdout
        decoder = 'nanocodec (not independent: no other AV2 decoder reads raw output here)'
    else:
        raw = subprocess.run([str(DATA / 'anchors/x86_64/bin/ffmpeg-decode'), '-v', 'error', '-i', str(stream),
                              '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], capture_output=True, check=True).stdout
        decoder = 'ffmpeg'
    return np.frombuffer(raw, np.uint8).reshape(-1, frame_bytes(width, height)), decoder


def plane_psnr(decoded: np.ndarray, source: np.ndarray, width: int, height: int) -> list[float]:
    luma, chroma = width * height, ((width + 1) // 2) * ((height + 1) // 2)
    result = []
    for start, end in ((0, luma), (luma, luma + chroma), (luma + chroma, luma + 2 * chroma)):
        mse = ((decoded[start:end].astype(np.int64) - source[start:end].astype(np.int64)) ** 2).mean()
        result.append(MAX_PSNR_DB if mse == 0 else min(MAX_PSNR_DB, 10 * np.log10(255 ** 2 / mse)))
    return result


def yuv(planes: list[float]) -> float:
    return sum(w * p for w, p in zip(PLANE_WEIGHTS, planes)) / sum(PLANE_WEIGHTS)


def window_checks(view: dict) -> list[Check]:
    """The panel's bitrate, PSNR-YUV and times from the report's window, as the page computes them."""
    frames, fps, shown = view['window']['frames'], view['window']['fps'], view['shown']['metrics']
    if not frames:
        return []
    seconds = (frames[-1]['timestamp'] - frames[0]['timestamp'] + 1000 / fps) / 1000
    mean = lambda key: (lambda values: sum(values) / len(values) if values else None)([f[key] for f in frames if f[key] is not None])
    values = {'bitrate': sum(f['bytes'] for f in frames) * 8 / seconds / 1000, 'psnr': mean('psnr'),
              'encode': mean('encodeMs'), 'decode': mean('decodeMs')}
    checks = []
    for key, value in values.items():
        text = shown.get(key)
        if value is None:
            checks.append(Check(view['letter'], f'shown {key}', text, 'n/a', text in ('n/a', '—')))
            continue
        expected = f'{value:.{SHOWN_DECIMALS[key]}f}'
        checks.append(Check(view['letter'], f'shown {key}', text, expected, text == expected, view['window']['bitrateFormula'] if key == 'bitrate' else ''))
    return checks


def unit_checks(letter: str, frames: list[dict], units: list[bytes], label: str) -> list[Check]:
    """Each reported coded frame (repeats code nothing) against the last of `units`, in order."""
    coded = [f for f in frames if not f['repeat']]
    if len(units) < len(coded):
        return [Check(letter, f'{label} access units', len(coded), len(units), False, 'fewer units than reported frames')]
    checks = []
    for frame, unit in zip(coded, units[len(units) - len(coded):]):
        digest = hashlib.sha256(unit).hexdigest()
        checks.append(Check(letter, f'tick {frame["tick"]} {label} unit', f'{frame["bytes"]} B {frame["sha256"][:12]}',
                            f'{len(unit)} B {digest[:12]}', digest == frame['sha256'] and len(unit) == frame['bytes']))
    return checks


def psnr_checks(letter: str, frames: list[dict], pictures: list[np.ndarray], source_of, width: int, height: int, decoder: str) -> list[Check]:
    """Each reported frame's PSNR against the decoded picture it showed (`pictures`, the last ones, coded frames only)."""
    coded = [f for f in frames if not f['repeat']]
    pictures = pictures[len(pictures) - len(coded):]
    checks, shown = [], None
    index = -1
    for frame in frames:
        if not frame['repeat']:
            index += 1
            shown = pictures[index]
        source = source_of(frame['tick'])
        if shown is None or source is None or frame['psnr'] is None:
            continue
        planes = plane_psnr(shown, source, width, height)
        checks.append(Check(letter, f'tick {frame["tick"]} PSNR-YUV', round(frame['psnr'], 4), round(yuv(planes), 4),
                            abs(frame['psnr'] - yuv(planes)) <= PSNR_TOLERANCE_DB, decoder))
        for name, page, offline in zip(PLANES, frame.get('planes') or [], planes):
            checks.append(Check(letter, f'tick {frame["tick"]} PSNR-{name}', round(page, 4), round(offline, 4), abs(page - offline) <= PSNR_TOLERANCE_DB))
    return checks


def cli_command(nanocodec: Path, config: dict, frames: int, colour: str | None, rate_changes: list[str], source: Path, out: Path) -> list[str]:
    """The native encode of the page's configuration (web/options.js cliCommand, plus colour and rate changes)."""
    args = [str(nanocodec), 'encode', '--codec', config['codec'], '--input', str(source), '--width', str(config['width']),
            '--height', str(config['height']), '--fps', str(config['fps']), '--frames', str(frames),
            '--intra-period', str(config['keyInterval']), '--qp', str(config['qp']), '--output', str(out)]
    if config['rateMode'] == 'cbr':
        args += ['--bitrate', str(config['targetKbps']), '--peak-bitrate', str(config['peakKbps']), '--buffer-ms', str(config['bufferMs'])]
    if config['preset'] in PRESET_SUFFIX:
        args += ['--preset', f'{config["codec"]}-{PRESET_SUFFIX[config["preset"]]}']
    for line in (config.get('overrides') or '').split('\n'):
        setting = line.split('#')[0].strip()
        if setting:
            args += ['--param', '='.join(part.strip() for part in setting.split('='))]
    if config.get('filmGrain'):
        args.append('--film-grain')
    if colour and config['codec'] != 'av2':
        family, _, full = colour.partition('-')
        args += ['--colour', family, '--range', 'full' if full else 'limited']
    for change in rate_changes:
        args += ['--rate-change', change]
    return args


class Replay:
    def __init__(self, report: dict, scratch: Path, nanocodec: Path | None):
        self.report, self.scratch, self.nanocodec = report, scratch, nanocodec
        self.source = report['source']
        self.sources: dict[tuple[int, int, int], np.ndarray] = {}
        self.make = generator() if self.source['reproducible'] else None

    def source_frames(self, config: dict) -> np.ndarray | None:
        """A view's source frames (one per coded tick of a loop): the clip rendition the page
        says the view coded (`config['rendition']`), placed at the view's size."""
        if not self.make:
            return None
        width, height, fps, file = config['width'], config['height'], config['fps'], config['rendition']
        key = (width, height, fps, file)
        if key not in self.sources:
            path = self.scratch / f'source-{width}x{height}-{fps}.yuv'
            entry = self.make.Entry(self.source['clip'], '', width, height, fps, 0)
            self.make.write_source(entry, path, WEB / 'clips' / file)
            self.sources[key] = np.fromfile(path, np.uint8).reshape(-1, frame_bytes(width, height))
            path.unlink()
        return self.sources[key]

    def source_of(self, config: dict):
        frames = self.source_frames(config)
        loop, step = self.source.get('loopLength'), SOURCE_FPS // config['fps']
        if frames is None or not loop:
            return lambda tick: None
        return lambda tick: frames[(tick % loop) // step]

    def renditions(self) -> list[Check]:
        """Each clip file the views coded: the repository's file has the hash the page measured."""
        checks = []
        for rendition in self.source.get('renditions') or []:
            digest = hashlib.sha256((WEB / 'clips' / rendition['file']).read_bytes()).hexdigest()
            checks.append(Check('-', f'{rendition["file"]} SHA-256', rendition['sha256'][:16], digest[:16], digest == rendition['sha256']))
        return checks

    def view(self, view: dict) -> list[Check]:
        checks = window_checks(view)
        config, letter, frames = view['config'], view['letter'], view['frames']
        if self.make:
            # A view codes its own size's rendition, as the pre-encoded streams were made (clips.js clipUrl).
            own = self.make.clip_file(self.source['clip'], config['height']).name
            checks.append(Check(letter, 'source rendition', config['rendition'], own, config['rendition'] == own,
                                '' if config['rendition'] == own else 'the view coded another rendition than one of its size'))
        if config['source'] == 'reference':
            checks += self.reference(view)
        elif self.make and self.source['path'].startswith('exact'):
            checks += self.live(view)
        else:
            checks.append(Check(letter, 'native re-encode', '—', '—', True, self.source.get('reason') or 'the clip was sampled while it played'))
        if view.get('stream', {}).get('complete'):
            checks += self.embedded(view)
        if not frames:
            checks.append(Check(letter, 'frames', 0, '—', False, 'no coded frames in the report'))
        return checks

    def reference(self, view: dict) -> list[Check]:
        letter, config = view['letter'], view['config']
        path = WEB / 'streams-ref' / view['streamFile']
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        checks = [Check(letter, 'stream file SHA-256', view['streamSha256'][:16], digest[:16], digest == view['streamSha256'])]
        packets = ivf_packets(data)
        step, loop = SOURCE_FPS // config['fps'], self.source.get('loopLength')
        for frame in view['frames']:
            packet = packets[(frame['tick'] % loop) // step]
            unit = hashlib.sha256(packet).hexdigest()
            checks.append(Check(letter, f'tick {frame["tick"]} stream unit', frame['sha256'][:12], unit[:12], unit == frame['sha256']))
        pictures, decoder = decode(config['codec'], path, config['width'], config['height'], self.nanocodec)
        source_of = self.source_of(config)
        for frame in view['frames']:
            source = source_of(frame['tick'])
            if source is None or frame['psnr'] is None:
                continue
            planes = plane_psnr(pictures[(frame['tick'] % loop) // step], source, config['width'], config['height'])
            checks.append(Check(letter, f'tick {frame["tick"]} PSNR-YUV', round(frame['psnr'], 4), round(yuv(planes), 4),
                                abs(frame['psnr'] - yuv(planes)) <= PSNR_TOLERANCE_DB, decoder))
            for name, page, offline in zip(PLANES, frame.get('planes') or [], planes):
                checks.append(Check(letter, f'tick {frame["tick"]} PSNR-{name}', round(page, 4), round(offline, 4), abs(page - offline) <= PSNR_TOLERANCE_DB))
        return checks

    def live(self, view: dict) -> list[Check]:
        """Re-encode the session from its first tick natively and compare units and PSNR."""
        letter, config, frames = view['letter'], view['config'], view['frames']
        step = SOURCE_FPS // config['fps']
        unreproducible = [i for i in view['interventions'] if i['kind'] == 'keyframe' or (i['kind'] == 'operating-point' and i['operatingPoint'] == 0)]
        if unreproducible:
            return [Check(letter, 'native re-encode', '—', '—', True, f'key frames were requested at ticks {[i["tick"] for i in unreproducible]}: not replayed')]
        if not self.nanocodec:
            return [Check(letter, 'native re-encode', '—', '—', False, 'no native nanocodec (--nanocodec or NC_NANOCODEC_BIN)')]
        count = frames[-1]['tick'] // step + 1
        if count > MAX_REPLAY_FRAMES:
            return [Check(letter, 'native re-encode', '—', '—', True, f'{count} frames since the start: too long to replay')]
        source = self.source_frames(config)
        loop_frames = self.source['loopLength'] // step
        input_path = self.scratch / f'{letter}-input.yuv'
        with input_path.open('wb') as out:
            for index in range(count):
                out.write(source[index % loop_frames].tobytes())
        rate_changes = [f'{i["tick"] // step}:{i["targetKbps"]}:{i["peakKbps"]}:{i["bufferMs"]}' for i in view['interventions'] if i['kind'] == 'rate']
        colour = next((f['colour'] for f in frames if f.get('colour')), None)
        out_path = self.scratch / f'{letter}.stream'
        command = cli_command(self.nanocodec, config, count, colour, rate_changes, input_path, out_path)
        subprocess.run(command, capture_output=True, check=True)
        input_path.unlink()
        data = out_path.read_bytes()
        units = ivf_packets(data) if config['codec'] in IVF_CODECS else self.split_tail(data, frames)
        checks = unit_checks(letter, frames, units, 'native')
        pictures, decoder = decode(config['codec'], out_path, config['width'], config['height'], self.nanocodec)
        out_path.unlink()
        checks += psnr_checks(letter, frames, list(pictures), self.source_of(config), config['width'], config['height'], decoder)
        return checks

    @staticmethod
    def split_tail(data: bytes, frames: list[dict]) -> list[bytes]:
        """An Annex-B stream's last access units, cut at the reported sizes (the window ends the session)."""
        sizes = [f['bytes'] for f in frames if not f['repeat']]
        at, units = len(data) - sum(sizes), []
        for size in sizes:
            units.append(data[at:at + size])
            at += size
        return units

    def embedded(self, view: dict) -> list[Check]:
        """The access units embedded in the report: hashes against the frames, then decoded and scored."""
        letter, config, stream = view['letter'], view['config'], view['stream']
        units = [base64.b64decode(u['data']) for u in stream['units']]
        frames = [f for f in view['frames'] if f['tick'] >= stream['firstTick']]
        checks = unit_checks(letter, frames, units, 'embedded')
        path = self.scratch / f'{letter}-embedded'
        path.write_bytes(ivf_file(config['codec'], config['width'], config['height'], units) if config['codec'] in IVF_CODECS else b''.join(units))
        pictures, decoder = decode(config['codec'], path, config['width'], config['height'], self.nanocodec)
        path.unlink()
        return checks + psnr_checks(letter, frames, list(pictures), self.source_of(config), config['width'], config['height'], decoder)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('report', type=Path)
    parser.add_argument('--nanocodec', type=Path, default=os.environ.get('NC_NANOCODEC_BIN'))
    args = parser.parse_args()
    report = parse_report(args.report)
    print(f'Report from demo commit {report["build"]["commit"]}{" (local changes)" if report["build"]["dirty"] else ""}; '
          f'source {report["source"]["kind"]} {report["source"].get("clip") or ""}')
    with tempfile.TemporaryDirectory(prefix='nc-replay-', dir=DATA / 'work' if (DATA / 'work').is_dir() else None) as scratch:
        replay = Replay(report, Path(scratch), args.nanocodec)
        checks = replay.renditions() + [check for view in report['views'] for check in replay.view(view)]
    for check in checks:
        print(f'{"ok " if check.ok else "BAD"} {check.view} {check.item:<28} page {check.page!s:<22} independent {check.independent!s:<22} {check.note}')
    bad = [c for c in checks if not c.ok]
    print(json.dumps({'status': 'fail' if bad else 'pass', 'checks': len(checks), 'mismatches': len(bad)}))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
