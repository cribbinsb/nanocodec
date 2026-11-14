#!/usr/bin/env python3
"""The page's PSNR numbers equal an independent offline computation, at equal
and at different view sizes.

Headless Chromium runs each SCENARIO with analysis on, once on a normal page
and once on a slowed one (the main thread stalled 30 ms every 170 ms; a slow
page once scored frames against the wrong source frame, 2026-11-14). For every
frame either view shows, the page's PSNR of each plane (the PSNR map's frame
values, error-map.js), its PSNR-YUV (6:1:1) and the page's own PSNR-YUV number
(nc-web `psnr`) must equal, within TOLERANCE_DB, the PSNR computed here: the
same access units decoded natively (ffmpeg; a pre-encoded view from its stream
file, a live view from the packets the page's encoder made) against the
demo's own source frame of that tick at the view's size, from the clip
rendition for that size (tools/web/make-reference-streams.py `write_source`,
as the streams were made; the page gives each view that rendition,
exact-source.js ViewSources, and names it in the statistics). On a sample of
frames every 16x16 luma block of the map is checked too. A live view's packets
must be byte-identical in the normal and slowed runs (exact sources: the
machine's speed changes nothing).

Scenarios: the BWC SOC FoodMarket 720p stream beside a matched live encode;
a 640x360 pre-encoded stream beside a live 854x480 view (two renditions); a
live 1280x720 view beside a live 640x360 one (two renditions).

  python3 web/tests/psnr-map.py
"""
import base64
import functools
import hashlib
import http.server
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
DATA = Path.home() / 'nanocodec-data'
SOURCE_FPS = 30
TOLERANCE_DB = 0.01
MEASURE_S = 8
MIN_FRAMES = 20
# Every BLOCK_SAMPLE_EVERY-th shown frame keeps its whole luma map for the block check.
BLOCK_SAMPLE_EVERY = 8
MAP_BLOCK = 16
MAX_PSNR_DB = 100.0
PLANE_WEIGHTS = (6, 1, 1)
IVF_FOURCC = {'av1': b'AV01', 'vp9': b'VP90'}
STALL = 'setInterval(() => { const t = performance.now(); while (performance.now() - t < 30); }, 170);'
COMPARED = "+(document.querySelector('#session-stats').textContent.match(/(\\d+) frames compared/) || [0, 0])[1]"
MANIFEST = {e['id']: e for e in json.loads((WEB / 'streams-ref/manifest.json').read_text())['streams']}

# Decoders are made in view order (A then B) at every start and analysis switch
# (pipeline.js), so decode worker n shows view n % 2. A comparison starts with its
# capture worker: everything recorded is reset then. Each view's coded packets are
# kept in order (a live view's for the native decode), and each shown frame with a
# PSNR map is recorded.
HOOK = '''window.__m = { shown: [], packets: [[], []] };
const Original = Worker; let decoders = 0;
window.Worker = class extends Original {
  constructor(url, ...rest) {
    super(url, ...rest);
    const script = String(url).split('/').pop();
    if (script === 'capture-worker.js') { window.__m = { shown: [], packets: [[], []] }; decoders = 0; }
    if (script !== 'decode-worker.js') return;
    const view = decoders++ % 2;
    this.addEventListener('message', ({ data: d }) => {
      if (d.type !== 'frame' || d.held) return;
      const packets = window.__m.packets[view];
      for (const c of [...(d.earlier || []), d]) if (!c.repeat) packets.push({ tick: Math.round(c.timestamp * 30 / 1000), data: c.packet.slice() });
      if (!d.errorMap) return;
      const keep = window.__m.shown.length % EVERY === 0;
      window.__m.shown.push({ view, tick: Math.round(d.timestamp * 30 / 1000), psnr: d.psnr, frame: d.errorMap.frame,
        decoded: packets.length - 1, repeat: !!d.repeat, luma: keep ? Array.from(d.errorMap.coarse.planes[0]) : null });
    });
  }
};
window.__packets = view => window.__m.packets[view].map(p => { let text = '';
  for (let i = 0; i < p.data.length; i += 32768) text += String.fromCharCode(...p.data.subarray(i, i + 32768));
  return { tick: p.tick, data: btoa(text) }; });'''


@dataclass(frozen=True)
class View:
    """A pre-encoded stream (`stream`), or a live encode of `codec` at a size and rate."""
    stream: str | None = None
    codec: str = ''
    width: int = 0
    height: int = 0
    fps: int = 0

    def __post_init__(self):
        if self.stream:
            entry = MANIFEST[self.stream]
            for name in ('codec', 'width', 'height', 'fps'):
                object.__setattr__(self, name, entry[name])


@dataclass(frozen=True)
class Scenario:
    name: str
    clip: str
    views: tuple[View, View]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def choose_stream(page, view: int, stream: str, clip: str):
    page.locator('.source-picker').nth(view).select_option('reference')
    page.wait_for_function(f"document.querySelectorAll('.stream-picker')[{view}].options.length > 10", timeout=30000)
    page.locator('.stream-picker').nth(view).select_option(stream)
    page.wait_for_function(f"document.querySelector('#clip').value === '{clip}'", timeout=30000)


def match_bwc(page):
    choose_stream(page, 0, 'netflix-food-market-bwc-soc-h264-1280x720-30fps-1000k', 'netflix-food-market')
    page.locator('.match').nth(0).click()


def reference_beside_480p(page):
    choose_stream(page, 0, 'xiph-johnny-openh264-640x360-15fps-150k', 'xiph-johnny')


def live_720p_beside_360p(page):
    page.locator('.resolution').nth(0).select_option('1280x720')
    page.locator('.resolution').nth(1).select_option('640x360')


SCENARIOS = [
    (Scenario('BWC SOC 720p stream beside a matched live encode', 'netflix-food-market',
              (View(stream='netflix-food-market-bwc-soc-h264-1280x720-30fps-1000k'), View(codec='h264', width=1280, height=720, fps=30))), match_bwc),
    (Scenario('640x360 stream beside a live 854x480 view', 'xiph-johnny',
              (View(stream='xiph-johnny-openh264-640x360-15fps-150k'), View(codec='av1', width=854, height=480, fps=15))), reference_beside_480p),
    (Scenario('live 1280x720 beside live 640x360', 'netflix-bar-scene',
              (View(codec='h264', width=1280, height=720, fps=15), View(codec='av1', width=640, height=360, fps=15))), live_720p_beside_360p),
]


def page_run(port: int, setup, slowed: bool) -> dict:
    """The shown frames' page PSNR numbers, the renditions named, and both views' packets, from one run."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1400, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(HOOK.replace('EVERY', str(BLOCK_SAMPLE_EVERY)) + (STALL if slowed else ''))
        page.goto(f'http://127.0.0.1:{port}/')
        page.wait_for_function(f'{COMPARED} >= 1', timeout=120000)
        setup(page)
        page.wait_for_timeout(500)
        page.wait_for_function(f'{COMPARED} >= 10', timeout=120000)
        page.locator('.analysis-toggle').click()
        page.wait_for_function("[...document.querySelectorAll('.analysis-stats')].every(s => s.querySelector('table'))", timeout=60000)
        time.sleep(MEASURE_S)
        page.locator('#pause').click()
        page.wait_for_function("document.querySelector('#runtime-label').textContent === 'Paused'", timeout=30000)
        time.sleep(1)
        renditions = page.evaluate("""[...document.querySelectorAll('.analysis-stats')].map(s =>
          [...s.querySelectorAll('tr')].find(r => r.textContent.startsWith('Source rendition'))?.querySelector('td').textContent)""")
        data = {'shown': page.evaluate('window.__m.shown'), 'renditions': renditions,
                'packets': [[(p['tick'], base64.b64decode(p['data'])) for p in page.evaluate(f'window.__packets({v})')] for v in (0, 1)]}
        browser.close()
    assert not errors, errors
    return data


def generator():
    spec = importlib.util.spec_from_file_location('generator', REPO / 'tools/web/make-reference-streams.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['generator'] = module
    spec.loader.exec_module(module)
    return module


def frame_bytes(view: View) -> int:
    return view.width * view.height * 3 // 2


def decode(stream: Path, view: View) -> np.ndarray:
    """Every frame of `stream`, decoded natively, as packed I420 rows."""
    raw = subprocess.run([str(DATA / 'anchors/x86_64/bin/ffmpeg-decode'), '-v', 'error', '-i', str(stream),
                          '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, frame_bytes(view))


def container(view: View, units: list[bytes]) -> bytes:
    """A live view's units as a file ffmpeg reads: Annex-B as they are, AV1 / VP9 in IVF."""
    if view.codec not in IVF_FOURCC:
        return b''.join(units)
    header = (b'DKIF' + (0).to_bytes(2, 'little') + (32).to_bytes(2, 'little') + IVF_FOURCC[view.codec]
              + view.width.to_bytes(2, 'little') + view.height.to_bytes(2, 'little') + (1).to_bytes(4, 'little') * 2
              + len(units).to_bytes(4, 'little') + bytes(4))
    return header + b''.join(len(u).to_bytes(4, 'little') + i.to_bytes(8, 'little') + u for i, u in enumerate(units))


def plane_psnr(decoded: np.ndarray, source: np.ndarray, view: View) -> list[float]:
    luma = view.width * view.height
    bounds = ((0, luma), (luma, luma + luma // 4), (luma + luma // 4, luma * 3 // 2))
    result = []
    for start, end in bounds:
        mse = ((decoded[start:end].astype(np.int64) - source[start:end].astype(np.int64)) ** 2).mean()
        result.append(MAX_PSNR_DB if mse == 0 else min(MAX_PSNR_DB, 10 * np.log10(255 ** 2 / mse)))
    return result


def block_psnr(decoded: np.ndarray, source: np.ndarray, view: View) -> np.ndarray:
    """PSNR of every 16x16 luma block, row by row (edge blocks clipped)."""
    w, h = view.width, view.height
    error = ((decoded[:w * h].astype(np.int64) - source[:w * h].astype(np.int64)) ** 2).reshape(h, w)
    rows, columns = -(-h // MAP_BLOCK), -(-w // MAP_BLOCK)
    values = np.empty(rows * columns)
    for r in range(rows):
        for c in range(columns):
            block = error[r * MAP_BLOCK:(r + 1) * MAP_BLOCK, c * MAP_BLOCK:(c + 1) * MAP_BLOCK]
            sse = block.sum()
            values[r * columns + c] = MAX_PSNR_DB if sse == 0 else min(MAX_PSNR_DB, 10 * np.log10(255 ** 2 * block.size / sse))
    return values


class Offline:
    """Source frames and native decodes of one scenario, cached per view."""

    def __init__(self, scenario: Scenario, scratch: Path):
        self.scenario, self.scratch, self.make = scenario, scratch, generator()
        self.sources, self.streams = {}, {}

    def source(self, view: View) -> np.ndarray:
        if view not in self.sources:
            path = self.scratch / 'source.yuv'
            self.make.write_source(self.make.Entry(self.scenario.clip, '', view.width, view.height, view.fps, 0), path)
            self.sources[view] = np.fromfile(path, np.uint8).reshape(-1, frame_bytes(view))
            path.unlink()
        return self.sources[view]

    def rendition(self, view: View) -> str:
        return self.make.clip_file(self.scenario.clip, view.height).name

    def stream(self, view: View) -> np.ndarray:
        if view not in self.streams:
            self.streams[view] = decode(WEB / 'streams-ref' / MANIFEST[view.stream]['file'], view)
        return self.streams[view]

    def live(self, view: View, units: list[bytes]) -> np.ndarray:
        path = self.scratch / 'live.stream'
        path.write_bytes(container(view, units))
        pictures = decode(path, view)
        path.unlink()
        assert len(pictures) == len(units), f'{len(pictures)} frames decoded of {len(units)} units'
        return pictures


def check(run: dict, label: str, scenario: Scenario, offline: Offline) -> dict:
    """Compare one run's page numbers with the offline values."""
    for i, view in enumerate(scenario.views):
        assert run['renditions'][i] == offline.rendition(view), f'{label}: view {"AB"[i]} coded {run["renditions"][i]}, expected {offline.rendition(view)}'
    pictures = [None if view.stream else offline.live(view, [data for _, data in run['packets'][i]]) for i, view in enumerate(scenario.views)]
    wrong, checked, largest = [], 0, 0.0
    for frame in run['shown']:
        view = scenario.views[frame['view']]
        source_frames, step = offline.source(view), SOURCE_FPS // view.fps
        local = (frame['tick'] % (len(source_frames) * step)) // step
        decoded = offline.stream(view)[local] if view.stream else pictures[frame['view']][frame['decoded']]
        planes = plane_psnr(decoded, source_frames[local], view)
        yuv = sum(w * v for w, v in zip(PLANE_WEIGHTS, planes)) / sum(PLANE_WEIGHTS)
        page_yuv = sum(w * v for w, v in zip(PLANE_WEIGHTS, frame['frame'])) / sum(PLANE_WEIGHTS)
        values = list(zip(['Y', 'Cb', 'Cr', 'YUV map', 'YUV page'], [*frame['frame'], page_yuv, frame['psnr']], [*planes, yuv, yuv]))
        if frame['luma'] is not None:
            blocks = block_psnr(decoded, source_frames[local], view)
            worst = int(np.argmax(np.abs(np.array(frame['luma']) - blocks)))
            values.append((f'block {worst}', frame['luma'][worst], float(blocks[worst])))
        checked += 1
        largest = max(largest, *(abs(page - expected) for _, page, expected in values))
        wrong += [(label, 'AB'[frame['view']], frame['tick'], name, round(page, 4), round(expected, 4))
                  for name, page, expected in values if abs(page - expected) > TOLERANCE_DB]
    assert not wrong, f'{scenario.name}: {len(wrong)} values differ from the offline PSNR: {wrong[:8]}'
    for i in (0, 1):
        shown = sum(f['view'] == i for f in run['shown'])
        assert shown >= MIN_FRAMES, f'{label}: view {"AB"[i]} showed only {shown} frames'
    return {'frames': checked, 'largest_difference_db': round(largest, 6)}


def same_streams(runs: dict, scenario: Scenario):
    """A live view's packets are identical in the normal and slowed runs (over the ticks both coded)."""
    for i, view in enumerate(scenario.views):
        if view.stream:
            continue
        normal, slowed = (dict(runs[label]['packets'][i]) for label in ('normal', 'slowed'))
        common = sorted(set(normal) & set(slowed))
        assert len(common) >= MIN_FRAMES, f'{scenario.name}: only {len(common)} common ticks'
        differ = [t for t in common if hashlib.sha256(normal[t]).digest() != hashlib.sha256(slowed[t]).digest()]
        assert not differ, f'{scenario.name}: view {"AB"[i]} packets differ between the normal and slowed runs at ticks {differ[:5]}'


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = {}
    try:
        for scenario, setup in SCENARIOS:
            runs = {'normal': page_run(server.server_port, setup, False), 'slowed': page_run(server.server_port, setup, True)}
            same_streams(runs, scenario)
            with tempfile.TemporaryDirectory(dir=DATA / 'work') as directory:
                offline = Offline(scenario, Path(directory))
                results[scenario.name] = {label: check(run, label, scenario, offline) for label, run in runs.items()}
    finally:
        server.shutdown()
    print(json.dumps({'status': 'pass', 'scenarios': results, 'tolerance_db': TOLERANCE_DB}))


if __name__ == '__main__':
    main()
