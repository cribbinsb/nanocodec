#!/usr/bin/env python3
"""The analysis mode's PSNR numbers equal an independent offline computation.

Headless Chromium plays a pre-encoded stream (the BWC SOC H.264 1000k
FoodMarket stream) in view A and, after "Match this in the other version", a
live nanocodec H.264 encode at the same settings in view B, with analysis on.
For every frame either view shows, the page's PSNR of each plane (the PSNR
map's frame values, error-map.js), its PSNR-YUV (6:1:1) and the page's own
PSNR-YUV number (nc-web `psnr`) must equal, within TOLERANCE_DB, the PSNR
computed here: the same access units decoded natively (ffmpeg; A from the
stream file, B from the packets the page's encoder made), against the demo's
own source frame of that tick (tools/web/make-reference-streams.py
`write_source`, as web/tests/reference-psnr.py). On a sample of frames every
16x16 luma block's PSNR of the map is checked too. Once on a normal page and
once on a slowed one (the page's main thread stalled 30 ms every 170 ms), as a
slow page once scored frames against the wrong source frame (2026-11-14).

  python3 web/tests/psnr-map.py
"""
import base64
import functools
import http.server
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
DATA = Path.home() / 'nanocodec-data'
STREAM = 'netflix-food-market-bwc-soc-h264-1280x720-30fps-1000k'
CLIP = 'netflix-food-market'
WIDTH, HEIGHT, FPS = 1280, 720, 30
SOURCE_FPS = 30
TOLERANCE_DB = 0.01
MEASURE_S = 10
MIN_FRAMES = 20
# Every BLOCK_SAMPLE_EVERY-th shown frame keeps its whole luma map for the block check.
BLOCK_SAMPLE_EVERY = 8
MAP_BLOCK = 16
MAX_PSNR_DB = 100.0
PLANE_WEIGHTS = (6, 1, 1)
STALL = 'setInterval(() => { const t = performance.now(); while (performance.now() - t < 30); }, 170);'

# Decoders are made in view order (A then B) at every start and analysis switch
# (pipeline.js), so decode worker n shows view n % 2. Each shown frame with a PSNR
# map is recorded; view B's packets (every coded frame, in order) are kept from
# the start of its comparison (an encode worker's creation) for the native decode.
HOOK = '''window.__m = { shown: [], packets: [], lengths: [] };
const Original = Worker; let decoders = 0;
window.Worker = class extends Original {
  constructor(url, ...rest) {
    super(url, ...rest);
    const script = String(url).split('/').pop();
    if (script === 'encode-worker.js') { window.__m.packets = []; window.__m.lengths = []; }
    if (script !== 'decode-worker.js') return;
    const view = decoders++ % 2;
    this.addEventListener('message', ({ data: d }) => {
      if (d.type !== 'frame' || d.held) return;
      if (view === 1) for (const c of [...(d.earlier || []), d]) if (!c.repeat) { window.__m.packets.push(c.packet.slice()); window.__m.lengths.push(c.packet.length); }
      if (!d.errorMap) return;
      const keep = window.__m.shown.length % %(every)d === 0;
      window.__m.shown.push({ view, tick: Math.round(d.timestamp * 30 / 1000), psnr: d.psnr, frame: d.errorMap.frame,
        decoded: view === 1 ? window.__m.lengths.length - 1 : null, repeat: !!d.repeat,
        luma: keep ? Array.from(d.errorMap.coarse.planes[0]) : null, columns: d.errorMap.coarse.columns });
    });
  }
};
window.__packets = () => { const all = new Uint8Array(window.__m.lengths.reduce((a, b) => a + b, 0)); let at = 0;
  for (const p of window.__m.packets) { all.set(p, at); at += p.length; }
  let text = ''; for (let i = 0; i < all.length; i += 32768) text += String.fromCharCode(...all.subarray(i, i + 32768));
  return btoa(text); };'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def page_run(port: int, slowed: bool) -> dict:
    """The shown frames' page PSNR numbers and view B's packets, from one run."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1400, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(HOOK.replace('%(every)d', str(BLOCK_SAMPLE_EVERY)) + (STALL if slowed else ''))
        page.goto(f'http://127.0.0.1:{port}/')
        page.locator('.source-picker').nth(0).select_option('reference')
        page.wait_for_function("document.querySelectorAll('.stream-picker')[0].options.length > 10", timeout=30000)
        page.locator('.stream-picker').nth(0).select_option(STREAM)
        page.wait_for_function(f"document.querySelector('#clip').value === '{CLIP}'", timeout=30000)
        page.locator('.match').nth(0).click()
        compared = "+(document.querySelector('#session-stats').textContent.match(/(\\d+) frames compared/) || [0, 0])[1]"
        page.wait_for_function(f'{compared} >= 10', timeout=120000)
        page.locator('.analysis-toggle').click()
        page.wait_for_function("[...document.querySelectorAll('.analysis-stats')].every(s => s.querySelector('table'))", timeout=60000)
        time.sleep(MEASURE_S)
        page.locator('#pause').click()
        page.wait_for_function("document.querySelector('#runtime-label').textContent === 'Paused'", timeout=30000)
        time.sleep(1)
        data = page.evaluate('window.__m.shown')
        packets = base64.b64decode(page.evaluate('window.__packets()'))
        lengths = page.evaluate('window.__m.lengths')
        browser.close()
    assert not errors, errors
    return {'shown': data, 'packets': packets, 'lengths': lengths}


def generator():
    spec = importlib.util.spec_from_file_location('generator', REPO / 'tools/web/make-reference-streams.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['generator'] = module
    spec.loader.exec_module(module)
    return module


def decode(stream: Path) -> np.ndarray:
    """Every frame of `stream`, decoded natively, as packed I420 rows."""
    raw = subprocess.run([str(DATA / 'anchors/x86_64/bin/ffmpeg-decode'), '-v', 'error', '-i', str(stream),
                          '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, WIDTH * HEIGHT * 3 // 2)


def plane_psnr(decoded: np.ndarray, source: np.ndarray) -> list[float]:
    """PSNR of each plane, capped as nc-web does."""
    luma = WIDTH * HEIGHT
    bounds = ((0, luma), (luma, luma + luma // 4), (luma + luma // 4, luma * 3 // 2))
    result = []
    for start, end in bounds:
        mse = ((decoded[start:end].astype(np.int64) - source[start:end].astype(np.int64)) ** 2).mean()
        result.append(MAX_PSNR_DB if mse == 0 else min(MAX_PSNR_DB, 10 * np.log10(255 ** 2 / mse)))
    return result


def block_psnr(decoded: np.ndarray, source: np.ndarray) -> np.ndarray:
    """PSNR of every 16x16 luma block, row by row."""
    shape = (HEIGHT // MAP_BLOCK, MAP_BLOCK, WIDTH // MAP_BLOCK, MAP_BLOCK)
    error = (decoded[:WIDTH * HEIGHT].astype(np.int64) - source[:WIDTH * HEIGHT].astype(np.int64)) ** 2
    sse = error.reshape(shape).sum(axis=(1, 3)).ravel()
    with np.errstate(divide='ignore'):
        values = np.where(sse == 0, MAX_PSNR_DB, 10 * np.log10(255 ** 2 * MAP_BLOCK * MAP_BLOCK / np.maximum(sse, 1)))
    return np.minimum(values, MAX_PSNR_DB)


def check(run: dict, label: str, source: np.ndarray, stream_a: np.ndarray, scratch: Path, loop: int) -> dict:
    """Compare one run's page numbers with the offline values; returns the frames checked and the largest difference."""
    stream_b = scratch / 'b.h264'
    stream_b.write_bytes(run['packets'])
    decoded_b = decode(stream_b)
    stream_b.unlink()
    assert len(decoded_b) == len(run['lengths']), f'{label}: {len(decoded_b)} frames decoded of {len(run["lengths"])} packets'
    wrong, checked, largest = [], 0, 0.0
    for frame in run['shown']:
        local = frame['tick'] % loop
        decoded = stream_a[local] if frame['view'] == 0 else decoded_b[frame['decoded']]
        planes = plane_psnr(decoded, source[local])
        yuv = sum(w * v for w, v in zip(PLANE_WEIGHTS, planes)) / sum(PLANE_WEIGHTS)
        page_yuv = sum(w * v for w, v in zip(PLANE_WEIGHTS, frame['frame'])) / sum(PLANE_WEIGHTS)
        values = list(zip(['Y', 'Cb', 'Cr', 'YUV map', 'YUV page'], [*frame['frame'], page_yuv, frame['psnr']], [*planes, yuv, yuv]))
        if frame['luma'] is not None:
            blocks = block_psnr(decoded, source[local])
            worst = int(np.argmax(np.abs(np.array(frame['luma']) - blocks)))
            values.append((f'block {worst}', frame['luma'][worst], float(blocks[worst])))
        checked += 1
        largest = max(largest, *(abs(page - offline) for _, page, offline in values))
        wrong += [(label, 'AB'[frame['view']], frame['tick'], name, round(page, 4), round(offline, 4))
                  for name, page, offline in values if abs(page - offline) > TOLERANCE_DB]
    assert not wrong, f'{len(wrong)} values differ from the offline PSNR: {wrong[:8]}'
    for view in (0, 1):
        shown = sum(f['view'] == view for f in run['shown'])
        assert shown >= MIN_FRAMES, f'{label}: view {"AB"[view]} showed only {shown} frames'
    return {'frames': checked, 'largest_difference_db': round(largest, 6)}


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        runs = {'normal': page_run(server.server_port, False), 'slowed': page_run(server.server_port, True)}
    finally:
        server.shutdown()
    loop = next(e for e in json.loads((WEB / 'streams-ref/manifest.json').read_text())['streams'] if e['id'] == STREAM)['loopTicks']
    frame_bytes = WIDTH * HEIGHT * 3 // 2
    with tempfile.TemporaryDirectory(dir=DATA / 'work') as directory:
        scratch = Path(directory)
        source_path = scratch / 'source.yuv'
        make = generator()
        make.write_source(make.Entry(CLIP, '', WIDTH, HEIGHT, FPS, 0), source_path)
        source = np.fromfile(source_path, np.uint8).reshape(-1, frame_bytes)
        source_path.unlink()
        stream_a = decode(WEB / 'streams-ref' / f'{STREAM}.ivf')
        checked = {label: check(run, label, source, stream_a, scratch, loop) for label, run in runs.items()}
    print(json.dumps({'status': 'pass', 'runs': checked, 'tolerance_db': TOLERANCE_DB}))


if __name__ == '__main__':
    main()
