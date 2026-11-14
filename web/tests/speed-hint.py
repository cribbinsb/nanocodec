#!/usr/bin/env python3
"""A machine too slow for the views' settings is offered a lighter one, and the
setting it offers is lighter: the slow view's encoder needs less CPU per frame
(speed-hint.js).

Chromium runs on one CPU (this process's affinity, inherited by the browser),
so 1280x720 at 30 fps, H.264 vs AV1 Default, cannot keep up on any current
machine. The sample clip is an exact source, so the page slows the clip down
rather than skipping frames. The test checks that:

- the speed note appears, says the clip runs slower than real time with the
  encode time the panel measured, shows no estimated rate, and offers a smaller
  size;
- clicking it moves both views to that size and restarts the comparison;
- the slowest encode worker's CPU time per coded frame (thread time from a
  Chrome trace over TRACE_S, after WARM_FRAMES frames of the run, so start-up,
  module compilation and warm-up are past) falls by at least MIN_GAIN: the
  offered setting is lighter.

Delivered real-time speed is not the measure: it depends on everything else on
the core and on machine load (the post-landing verifier, load 12-14, once saw
0.62x before and 0.46x after a window that began 4 s into the restarted run).

  python3 web/tests/speed-hint.py
"""
import collections
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
# Frames each view codes after a (re)start before the trace window opens.
WARM_FRAMES = 40
TRACE_S = 6
# The smaller size has at most 1/2.25 of the pixels, but per-frame work does not all
# scale with them: 1280x720 -> 854x480 measured 1.27-1.48x less encode CPU for AV1
# (speed-hint.js SIZE_COST_EXPONENT). The suggestion must be lighter by at least this.
MIN_GAIN = 1.15
TRACE_CATEGORIES = ['devtools.timeline', 'disabled-by-default-devtools.timeline', 'toplevel']
TASK_EVENT = 'ThreadControllerImpl::RunTask'
ENCODE_WORKER = 'encode-worker.js'
STATS = '#session-stats'
RUNNING = f"document.querySelector('{STATS}').textContent.includes('frames compared')"
# The decode worker (numbered in creation order, two per run) of each frame shown.
HOOK = '''window.__frames = [];
const OriginalWorker = Worker; let decoders = 0;
window.Worker = class extends OriginalWorker {
  constructor(url, ...rest) { super(url, ...rest);
    if (String(url).endsWith('decode-worker.js')) { const id = decoders++;
      this.addEventListener('message', e => { const d = e.data;
        if (d && d.type === 'frame' && !d.held) window.__frames.push(id); }); } }
};'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def encode_cpu_per_frame(browser, page, first_decoder):
    """The busiest encode worker's CPU ms per coded frame over TRACE_S, once each view
    of the run whose decode workers start at `first_decoder` has coded WARM_FRAMES."""
    counted = f'window.__frames.filter(id => id >= {first_decoder})'
    page.wait_for_function(f'(() => {{ const c = {{}}; for (const id of {counted}) c[id] = (c[id] || 0) + 1;'
                           f' const n = Object.values(c); return n.length === 2 && Math.min(...n) >= {WARM_FRAMES}; }})()',
                           timeout=600000)
    before = page.evaluate(f'{counted}.length')
    browser.start_tracing(page=page, categories=TRACE_CATEGORIES)
    time.sleep(TRACE_S)
    trace = json.loads(browser.stop_tracing())['traceEvents']
    frames = (page.evaluate(f'{counted}.length') - before) / 2
    assert frames > 2, frames
    cpu, scripts = collections.Counter(), collections.defaultdict(collections.Counter)
    for e in trace:
        if e.get('ph') != 'X':
            continue
        key = (e.get('pid'), e.get('tid'))
        if e['name'] == TASK_EVENT:
            cpu[key] += e.get('tdur', e.get('dur', 0)) / 1000
        elif e['name'] == 'FunctionCall':
            scripts[key][e.get('args', {}).get('data', {}).get('url', '').split('/').pop()] += 1
    encoders = [ms for key, ms in cpu.items() if scripts[key] and scripts[key].most_common(1)[0][0] == ENCODE_WORKER]
    assert encoders, 'no encode worker in the trace'
    return max(encoders) / frames


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
        # Decode workers so far: the page's first run and the 720p30 run (two each).
        decoders = page.evaluate('Math.max(...window.__frames) + 1')
        before = encode_cpu_per_frame(browser, page, decoders - 2)
        size = hint.locator('.speed-option', has_text='Smaller: ')
        label = size.inner_text()
        width, height = (int(v) for v in label.removeprefix('Smaller: ').split('×'))
        size.click()
        after = encode_cpu_per_frame(browser, page, decoders)
        sizes = [picker.input_value() for picker in page.locator('.resolution').all()]
        browser.close()
    assert not errors, errors
    assert sizes == [f'{width}x{height}'] * 2, sizes
    assert after * MIN_GAIN <= before, (label, before, after)
    return {'hint': text, 'chosen': label, 'encode_cpu_ms_before': round(before, 1), 'encode_cpu_ms_after': round(after, 1)}


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
