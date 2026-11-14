#!/usr/bin/env python3
"""A live source on a machine that cannot keep up: frames are skipped and
rate control budgets by time, so the real bitrate holds.

The skip accounting runs with the still-camera noise handling off (the page's
toggle, on by default): it tests frame-skip rate control alone. One more run
at share 1/2 with it on checks that the rate still stays under the ceiling.

Headless Chromium's synthetic webcam feeds both views (H.264 and AV1, 640x360,
30 fps, 100 kbps CBR). The pipeline is starved by letting pictures through to
the capture worker at most once every 1000 / (30 * share) ms (a slow encoder:
the page then skips the ticks it cannot take). For share 1/2 and 1/4, over
MEASURE_S seconds of capture time per view:

- the real bitrate (coded bytes x 8 / capture time covered) over the last
  MEASURE_S seconds is within RATE_TOLERANCE of the target, and over the
  whole run never more than RATE_CEILING above it;
- the skipped-frame count the view shows equals the view's ticks it never got.

  python3 web/tests/live-skip.py
"""
import functools
import http.server
import json
import re
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
# A target the synthetic webcam's simple picture needs at every rate tried, so the
# quality ceiling (rate.max_quality_qp: the target is a ceiling) never caps it.
TARGET_KBPS, FPS = 100, 30
MEASURE_S = 20
WARM_S = 6
# Longer than the panels' metrics interval and the frames in flight.
SETTLE_MS = 1500
# Short-run CBR settles within this of the target; the long-run rate must not exceed the ceiling.
RATE_TOLERANCE, RATE_CEILING = 0.10, 0.03

HOOK = '''window.__x = { coded: [[], []] };
const Original = Worker; let decoders = 0;
const release = [], period = %(period)f;
setInterval(() => { const post = release.shift(); if (post) post(); }, period);
window.Worker = class extends Original {
  constructor(url, ...rest) { super(url, ...rest); const script = String(url).split('/').pop(); this.script = script;
    if (script === 'decode-worker.js') this.view = decoders++ %% 2;
    this.addEventListener('message', e => { const d = e.data;
      if (!d || d.type !== 'frame' || script !== 'decode-worker.js' || d.held) return;
      for (const coded of [...(d.earlier || []), d]) window.__x.coded[this.view].push({ t: coded.timestamp, bytes: coded.packet.byteLength }); }); }
  postMessage(d, ...rest) {
    if (d && d.type === 'frame' && this.script === 'capture-worker.js') { release.push(() => super.postMessage(d, ...rest)); return; }
    return super.postMessage(d, ...rest); }
};'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def run(port, share, still_camera=False):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(HOOK % {'period': 1000 / (FPS * share)})
        page.goto(f'http://127.0.0.1:{port}/')
        page.locator('#still-camera').set_checked(still_camera)
        page.locator('[data-source="camera"]').click()
        for picker in page.locator('.resolution').all():
            picker.select_option('640x360')
        for picker in page.locator('.fps').all():
            picker.select_option(str(FPS))
        for selector, kbps in (('.peak-rate', 2 * TARGET_KBPS), ('.target-rate', TARGET_KBPS)):
            for box in page.locator(selector).all():
                box.fill(str(kbps))
                box.dispatch_event('change')
        page.locator('#start').click()
        page.wait_for_function('window.__x.coded.every(v => v.length > 5)', timeout=120000)
        time.sleep(WARM_S + MEASURE_S)
        # Paused, the frames in flight complete and nothing more arrives: the counts settle.
        page.locator('#pause').click()
        page.wait_for_function("document.querySelector('#pause').textContent === 'Resume'", timeout=60000)
        page.wait_for_timeout(SETTLE_MS)
        data = page.evaluate('window.__x')
        counts = page.locator('.frame-count').all_inner_texts()
        browser.close()
    assert not errors, errors
    out = []
    for view, every in enumerate(data['coded']):
        # Skipped: the view's ticks between the frames it coded (30 fps: every tick).
        ticks = [round(c['t'] * FPS / 1000) for c in every]
        expected_skips = sum(b - a - 1 for a, b in zip(ticks, ticks[1:]))
        coded = [c for c in every if c['t'] >= every[-1]['t'] - MEASURE_S * 1000]
        seconds = (coded[-1]['t'] - coded[0]['t']) / 1000 + 1 / FPS
        kbps = sum(c['bytes'] for c in coded) * 8 / seconds / 1000
        delivered = len(coded) / seconds
        # The still-camera handling may spend less on still noisy content (the target is a ceiling).
        assert kbps <= (1 + RATE_TOLERANCE) * TARGET_KBPS, (share, view, kbps)
        if not still_camera:
            assert abs(kbps - TARGET_KBPS) <= RATE_TOLERANCE * TARGET_KBPS, (share, view, kbps)
        whole = sum(c['bytes'] for c in every) * 8 / ((every[-1]['t'] - every[0]['t']) / 1000 + 1 / FPS) / 1000
        assert whole <= (1 + RATE_CEILING) * TARGET_KBPS, (share, view, 'whole run', whole)
        assert delivered < FPS * (share + 0.25), (share, view, delivered)
        shown = re.search(r'(\d+) skipped', counts[view])
        assert shown and int(shown[1]) == expected_skips, (counts[view], expected_skips)
        out.append({'view': view, 'kbps': round(kbps, 1), 'whole_kbps': round(whole, 1), 'fps': round(delivered, 1), 'skipped': expected_skips})
    return out


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        results = {f'1/{int(1 / share)}': run(server.server_port, share) for share in (1 / 2, 1 / 4)}
        results['1/2 still-camera on'] = run(server.server_port, 1 / 2, still_camera=True)
    finally:
        server.shutdown()
    print(json.dumps({'status': 'pass', 'results': results}))


if __name__ == '__main__':
    main()
