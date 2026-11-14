#!/usr/bin/env python3
"""A pre-encoded view's PSNR on the page equals the offline value, frame by frame.

Headless Chromium plays a pre-encoded stream (the BWC SOC H.264 1000k FoodMarket
stream) in view A against a live view B, with the page's main thread stalled for
30 ms every 170 ms (a busy page: analysis on, a slow machine). For every frame A
shows, the page's PSNR-YUV must equal, within TOLERANCE_DB, the PSNR of the same
stream frame against the demo's source frame of that tick, computed here from
make-reference-streams.py's own source frames and ffmpeg's decode. The capture
labelled frames with the frame callback's time while the element had already
moved on, scoring some frames against the next source frame (2026-11-14).

  python3 web/tests/reference-psnr.py
"""
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
STREAM = 'netflix-food-market-bwc-soc-h264-1280x720-30fps-1000k'
WIDTH, HEIGHT, FPS = 1280, 720, 30
TOLERANCE_DB = 0.05
MEASURE_S = 12
MAX_PSNR_DB = 100.0
# nc-web pixels::psnr: per-plane PSNR weighted 6:1:1.
PLANE_WEIGHTS = (6, 1, 1)
DATA = Path.home() / 'nanocodec-data'

HOOK = '''window.__p = { ticks: {}, frames: [] };
setInterval(() => { const t = performance.now(); while (performance.now() - t < 30); }, 170);
const Original = Worker; let decoders = 0;
window.Worker = class extends Original {
  constructor(url, ...rest) { super(url, ...rest); const script = String(url).split('/').pop(); this.script = script;
    if (script === 'decode-worker.js') this.view = decoders++;
    this.addEventListener('message', e => { const d = e.data;
      if (d && d.type === 'frame' && script === 'decode-worker.js' && !d.held && d.psnr !== null) window.__p.frames.push({ view: this.view, id: d.id, psnr: d.psnr }); }); }
  postMessage(d, ...rest) {
    if (d && d.type === 'frame' && this.script === 'capture-worker.js') window.__p.ticks[d.id] = Math.round(d.timestamp * 30 / 1000);
    return super.postMessage(d, ...rest); }
};'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def page_psnr() -> list[tuple[int, float]]:
    """(stream frame, page PSNR) for every frame view A showed."""
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.add_init_script(HOOK)
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            page.locator('.view-source [data-view-source="reference"]').nth(0).click()
            page.wait_for_function("document.querySelectorAll('.stream-picker')[0].options.length > 10", timeout=30000)
            page.locator('.stream-picker').nth(0).select_option(STREAM)
            page.wait_for_function("document.querySelector('#clip').value === 'netflix-food-market'", timeout=30000)
            page.locator('.match').nth(0).click()
            # Match restarts the comparison; measure once the restarted one delivers.
            page.wait_for_function("+(document.querySelector('#session-stats').textContent.match(/(\\d+) frames compared/) || [0, 0])[1] >= 10", timeout=120000)
            page.evaluate('window.__p.frames.length = 0')
            time.sleep(MEASURE_S)
            data = page.evaluate('window.__p')
            browser.close()
    finally:
        server.shutdown()
    first_view = min(f['view'] for f in data['frames'])
    loop = next(e for e in json.loads((WEB / 'streams-ref/manifest.json').read_text())['streams'] if e['id'] == STREAM)['loopTicks']
    return [(data['ticks'][str(f['id'])] % loop, f['psnr']) for f in data['frames'] if f['view'] == first_view]


def offline_psnr() -> list[float]:
    """Per-frame PSNR-YUV of the stream against the demo's source frames."""
    spec = importlib.util.spec_from_file_location('generator', REPO / 'tools/web/make-reference-streams.py')
    generator = importlib.util.module_from_spec(spec)
    sys.modules['generator'] = generator
    spec.loader.exec_module(generator)
    luma = WIDTH * HEIGHT
    frame_bytes = luma * 3 // 2
    decoded = subprocess.run([str(DATA / 'anchors/x86_64/bin/ffmpeg-decode'), '-v', 'error', '-i', str(WEB / 'streams-ref' / f'{STREAM}.ivf'),
                              '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory(dir=DATA / 'work') as scratch:
        source_path = Path(scratch) / 'source.yuv'
        generator.write_source(generator.Entry('netflix-food-market', '', WIDTH, HEIGHT, FPS, 0), source_path)
        source = np.fromfile(source_path, np.uint8).reshape(-1, frame_bytes)
    frames = np.frombuffer(decoded, np.uint8).reshape(-1, frame_bytes)
    planes = ((0, luma), (luma, luma + luma // 4), (luma + luma // 4, frame_bytes))

    def psnr(error: float) -> float:
        return MAX_PSNR_DB if error == 0 else min(MAX_PSNR_DB, 10 * np.log10(255 ** 2 / error))

    scores = []
    for a, b in zip(frames, source):
        errors = [((a[s:e].astype(np.int32) - b[s:e].astype(np.int32)) ** 2).mean() for s, e in planes]
        scores.append(sum(w * psnr(err) for w, err in zip(PLANE_WEIGHTS, errors)) / sum(PLANE_WEIGHTS))
    return scores


def main():
    shown = page_psnr()
    expected = offline_psnr()
    wrong = [(frame, round(page, 2), round(expected[frame], 2)) for frame, page in shown if abs(page - expected[frame]) > TOLERANCE_DB]
    assert len(shown) >= 20, f'only {len(shown)} frames shown'
    assert not wrong, f'{len(wrong)} of {len(shown)} frames scored against the wrong source: {wrong[:5]}'
    print(json.dumps({'status': 'pass', 'frames': len(shown), 'tolerance_db': TOLERANCE_DB}))


if __name__ == '__main__':
    main()
