#!/usr/bin/env python3
"""A machine too slow for the views' settings is offered a lighter one, and
taking it brings the comparison closer to real time (speed-hint.js).

Chromium runs on one CPU (this process's affinity, inherited by the browser),
so 1280x720 at 30 fps, H.264 vs AV1 Default, cannot keep up on any current
machine. The sample clip is an exact source, so the page slows the clip down
rather than skipping frames. The test checks that:

- the speed note appears, says the clip runs slower than real time with the
  encode time the panel measured, shows no estimated rate, and offers a smaller
  size;
- clicking it moves both views to that size and restarts the comparison;
- the clip's speed against real time (clip time per wall second, over
  MEASURE_S) rises by at least MIN_GAIN.

  python3 web/tests/speed-hint.py
"""
import functools
import http.server
import json
import os
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS = 1280, 720, 30
MEASURE_S = 8
SETTLE_S = 4
# The smaller size cuts the pixels by at least 2.25x; the speed must rise by this much.
MIN_GAIN = 1.3
STATS = '#session-stats'
RUNNING = f"document.querySelector('{STATS}').textContent.includes('frames compared')"
HOOK = '''window.__t = [];
const OriginalWorker = Worker; let decoders = 0;
window.Worker = class extends OriginalWorker {
  constructor(url, ...rest) { super(url, ...rest);
    if (String(url).endsWith('decode-worker.js') && decoders++ % 2 === 0)
      this.addEventListener('message', e => { const d = e.data;
        if (d && d.type === 'frame' && !d.held) window.__t.push([performance.now(), d.timestamp]); }); }
};'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def speed(page):
    """Clip seconds coded per wall second by the first view over MEASURE_S."""
    first = page.evaluate('window.__t.length')
    time.sleep(MEASURE_S)
    times = page.evaluate(f'window.__t.slice({first})')
    assert len(times) > 2, times
    return (times[-1][1] - times[0][1]) / (times[-1][0] - times[0][0])


def run(port):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--autoplay-policy=no-user-gesture-required'])
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(HOOK)
        page.goto(f'http://127.0.0.1:{port}/')
        page.wait_for_function(RUNNING, timeout=120000)
        for picker, codec in zip(page.locator('.codec-picker').all(), ('h264', 'av1')):
            picker.select_option(codec)
            picker.dispatch_event('change')
        for selector, value in (('.resolution', f'{WIDTH}x{HEIGHT}'), ('.fps', str(FPS))):
            for picker in page.locator(selector).all():
                picker.select_option(value)
                picker.dispatch_event('change')
        page.wait_for_timeout(500)
        page.wait_for_function(RUNNING, timeout=120000)
        hint = page.locator('#speed-hint')
        hint.locator('.speed-option').first.wait_for(timeout=120000)
        text = hint.inner_text()
        assert 'slower than real time' in text and ' ms to encode a frame' in text, text
        assert 'fps (' not in text and 'about' not in text, text
        before = speed(page)
        size = hint.locator('.speed-option', has_text='Smaller: ')
        label = size.inner_text()
        width, height = (int(v) for v in label.removeprefix('Smaller: ').split('×'))
        size.click()
        page.wait_for_timeout(500)
        page.wait_for_function(RUNNING, timeout=120000)
        time.sleep(SETTLE_S)
        after = speed(page)
        sizes = [picker.input_value() for picker in page.locator('.resolution').all()]
        browser.close()
    assert not errors, errors
    assert sizes == [f'{width}x{height}'] * 2, sizes
    assert after >= MIN_GAIN * before, (before, after)
    return {'hint': text, 'chosen': label, 'speed_before': round(before, 2), 'speed_after': round(after, 2)}


def main():
    # One CPU of those this process may use: the browser inherits it.
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = run(server.server_port)
    finally:
        server.shutdown()
    print(json.dumps({'status': 'pass', **result}))


if __name__ == '__main__':
    main()
