#!/usr/bin/env python3
"""A sample-clip comparison does not depend on the machine's speed.

Sample clips are exact sources (web/exact-source.js): the page codes every tick a
view codes, in order, and lets the clip's time run slower than real time when the
pipeline cannot keep up. Headless Chromium runs the default comparison twice:
once as it is, once slowed (pictures reach the capture worker at most once every
SLOW_MS, a third of real time, and the main thread stalls STALL_MS every
STALL_EVERY_MS). In both runs:

- each view codes every tick of its frame rate from the first on, none skipped;
- each view's coded frames are byte-identical across the runs (a hash of every
  packet), and so is every PSNR the page shows for them;
- the bitrate the panel shows equals the shown frames' bytes x 8 / their clip time;
- the slowed run says it ran below real time.

  python3 web/tests/exact-source.py
"""
import functools
import http.server
import json
import re
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
# Coded frames compared per view (4 s of clip at the default 15 fps).
FRAMES = 60
# The slowed run lets a picture through to the capture worker once every SLOW_MS
# (a third of the default 15 fps) and stalls the main thread STALL_MS every STALL_EVERY_MS.
SLOW_MS, STALL_MS, STALL_EVERY_MS = 200, 40, 150
# The slowed run must measure at most this clip-time / wall-time ratio (a third is set).
MAX_SLOWED_SPEED = 0.5
TIMEOUT_MS = 180000
# panel.js: the bitrate covers the last HISTORY_FRAMES coded frames.
HISTORY_FRAMES = 90
# The panel rounds to whole kbps.
BITRATE_TOLERANCE_KBPS = 1
# Longer than the panels' metrics interval (panel.js METRICS_INTERVAL_MS) and the last frames in flight.
SETTLE_MS = 1500

HOOK = '''window.__x = { coded: [[], []] };
const SLOW = %(slow)s, release = [];
if (SLOW) setInterval(() => { const post = release.shift(); if (post) post(); }, %(delay)d);
if (SLOW) setInterval(() => { const t = performance.now(); while (performance.now() - t < %(stall)d); }, %(every)d);
const hash = bytes => { let h = 2166136261; for (const b of bytes) h = Math.imul(h ^ b, 16777619) >>> 0; return h.toString(16); };
const Original = Worker; let decoders = 0;
window.Worker = class extends Original {
  constructor(url, ...rest) { super(url, ...rest); const script = String(url).split('/').pop(); this.script = script;
    if (script === 'decode-worker.js') this.view = decoders++ %% 2;
    this.addEventListener('message', e => { const d = e.data;
      if (!d || d.type !== 'frame' || script !== 'decode-worker.js' || d.held) return;
      for (const coded of [...(d.earlier || []), d]) window.__x.coded[this.view].push({ wall: performance.now(), tick: Math.round(coded.timestamp * 30 / 1000), packet: hash(new Uint8Array(coded.packet)),
        bytes: coded.packet.byteLength, psnr: coded.psnr }); }); }
  postMessage(d, ...rest) {
    if (d && d.type === 'frame' && this.script === 'capture-worker.js') {
      if (SLOW) { release.push(() => super.postMessage(d, ...rest)); return; }
    }
    return super.postMessage(d, ...rest); }
};'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def run(port, slow):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(HOOK % {'slow': 'true' if slow else 'false', 'stall': STALL_MS, 'every': STALL_EVERY_MS, 'delay': SLOW_MS})
        page.goto(f'http://127.0.0.1:{port}/')
        page.wait_for_function(f'window.__x.coded.every(v => v.length >= {FRAMES})', timeout=TIMEOUT_MS)
        # Paused, nothing more arrives and the panels' figures settle on the frames recorded.
        page.locator('#pause').click()
        page.wait_for_function("document.querySelector('#pause').textContent === 'Resume'", timeout=TIMEOUT_MS)
        page.wait_for_timeout(SETTLE_MS)
        data = page.evaluate('window.__x')
        session = page.locator('#session-stats').inner_text()
        shown = page.evaluate("[...document.querySelectorAll('[data-metric=bitrate]')].map(e => +e.textContent)")
        fps = page.evaluate("[...document.querySelectorAll('.version')].map(v => +v.querySelector('.fps').value)")
        browser.close()
    assert not errors, errors
    views = []
    for view, coded in enumerate(data['coded']):
        frames = coded
        step = 30 // fps[view]
        ticks = [f['tick'] for f in frames[:FRAMES]]
        assert ticks == list(range(0, FRAMES * step, step)), f'view {view} skipped ticks: {ticks[:20]}'
        recent = frames[-HISTORY_FRAMES:]
        expected = sum(f['bytes'] for f in recent) * 8 * fps[view] / len(recent) / 1000
        assert abs(shown[view] - expected) <= BITRATE_TOLERANCE_KBPS, (view, shown[view], expected)
        views.append(frames[:FRAMES])
    # How fast the clip ran: clip time over wall time from the second to the last compared frame.
    first, last = views[0][1], views[0][FRAMES - 1]
    speed = (last['tick'] - first['tick']) / 30 / ((last['wall'] - first['wall']) / 1000)
    return views, session, speed


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        normal, _, _ = run(server.server_port, slow=False)
        slowed, session, speed = run(server.server_port, slow=True)
    finally:
        server.shutdown()
    for view, (a, b) in enumerate(zip(normal, slowed)):
        for x, y in zip(a, b):
            assert (x['tick'], x['packet'], x['bytes']) == (y['tick'], y['packet'], y['bytes']), f'view {view} tick {x["tick"]}: {x} vs {y}'
            assert x['psnr'] is None and y['psnr'] is None or abs(x['psnr'] - y['psnr']) < 0.01, (view, x, y)
    # The slowing must really have happened, or the identity above proves nothing.
    assert speed <= MAX_SLOWED_SPEED, f'the slowed run ran at {speed:.2f}x real time'
    assert re.search(r'running at [0-9.]+× real time', session), session
    print(json.dumps({'status': 'pass', 'frames_per_view': FRAMES, 'slowed_speed': round(speed, 2), 'slowed_session': session}))


if __name__ == '__main__':
    main()
