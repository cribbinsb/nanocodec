#!/usr/bin/env python3
"""The comparison restarts by itself on every option change, once per burst.

There is no Start/Stop button: the page restarts the comparison RESTART_DELAY_MS
(300 ms) after the last of a burst of changes (app.js `scheduleRestart`).
Headless Chromium with a synthetic webcam, two live views (`?views=live`), the
page's CPU throttled CPU_SLOWDOWN x (DevTools emulation), so starts are slow
and overlap the next change. Every start makes one capture worker, so their
count is the number of starts. Checks:

- each option of the top box restarts on its own: the Test scene, Webcam and
  Sample clip buttons, the clip picker, a video file (Video file + its chooser);
  so does a view's resolution; each time the new comparison shows frames;
- a rate change applies live: no restart;
- a burst of BURST changes under RESTART_DELAY_MS apart (clip, sizes, frame
  rates, codec, preset) ends in exactly one start, with the burst's final
  settings in both encoders, and frames;
- no page error and no failure notice throughout.

  python3 web/tests/auto-restart.py
"""
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
CPU_SLOWDOWN = 4
FRAME_TIMEOUT_MS = 120000
# Changes in the burst, and the pause between them (under the page's 300 ms restart delay).
BURST, BURST_GAP_MS = 8, 120
# Longer than the restart delay plus a slow start's set-up, to see any second start.
SETTLE_MS = 3000
HOOK = '''window.__starts = 0; window.__inits = [];
const Original = Worker;
window.Worker = class extends Original {
  constructor(url, ...rest) { super(url, ...rest); if (String(url).endsWith('/capture-worker.js')) window.__starts++; }
  postMessage(message, ...rest) {
    if (message && message.type === 'init' && message.config) window.__inits.push(message.config);
    return super.postMessage(message, ...rest);
  }
};'''
# Both views have shown a frame of the current comparison.
SHOWN = "[...document.querySelectorAll('.frame-badge')].every(b => b.dataset.tick !== undefined)"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, restarts = [], {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
            page = browser.new_page(viewport={'width': 1400, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.add_init_script(HOOK)
            page.context.new_cdp_session(page).send('Emulation.setCPUThrottlingRate', {'rate': CPU_SLOWDOWN})
            page.goto(f'http://127.0.0.1:{server.server_port}/?views=live')
            page.wait_for_function(SHOWN, timeout=FRAME_TIMEOUT_MS)
            assert page.locator('#start').count() == 0
            assert page.evaluate('window.__starts') == 1, 'the page opens with one start'

            def restarts_on(name, change, check='true'):
                before = page.evaluate('window.__starts')
                change()
                page.wait_for_function(f'window.__starts > {before} && ({check})', timeout=FRAME_TIMEOUT_MS)
                page.wait_for_function(SHOWN, timeout=FRAME_TIMEOUT_MS)
                notice = page.locator('#notice')
                assert not notice.is_visible(), (name, notice.inner_text())
                restarts[name] = page.evaluate('window.__starts') - before

            restarts_on('test scene', lambda: page.locator('[data-source="scene"]').click(), "document.querySelector('#source-name').textContent.startsWith('Moving shapes')")
            restarts_on('webcam', lambda: page.locator('[data-source="camera"]').click(), "document.querySelector('#source-name').textContent === 'Webcam'")
            restarts_on('sample clip', lambda: page.locator('[data-source="clip"]').click(), "document.querySelector('#summary').textContent.includes('on the clip')")
            restarts_on('clip picker', lambda: page.locator('#clip').select_option('svt-waterfall'), "document.querySelector('#clip').value === 'svt-waterfall'")

            def choose_file():
                with page.expect_file_chooser() as chooser:
                    page.locator('[data-source="file"]').click()
                chooser.value.set_files(str(WEB / 'clips/xiph-johnny-360p.mp4'))
            restarts_on('video file', choose_file, "document.querySelector('#source-name').textContent === 'xiph-johnny-360p.mp4'")
            restarts_on('resolution', lambda: page.locator('.resolution').nth(0).select_option('640x360'),
                        'window.__inits.slice(-2).some(c => c.width === 640)')
            # A rate change applies to the running encoders: no restart.
            before = page.evaluate('window.__starts')
            page.locator('.target-rate').nth(1).fill('500')
            page.locator('.target-rate').nth(1).dispatch_event('change')
            page.wait_for_function("document.querySelectorAll('.rate-status')[1].textContent.startsWith('Target 500 kbps')", timeout=FRAME_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            assert page.evaluate('window.__starts') == before, 'a rate change restarted the comparison'
            # A burst of changes, under load: one start, with the final settings.
            page.locator('[data-source="clip"]').click()
            page.wait_for_function(SHOWN, timeout=FRAME_TIMEOUT_MS)
            before = page.evaluate('window.__starts')
            # Set in the page, BURST_GAP_MS apart by its own clock (Playwright's actions on a
            # throttled page take longer than the restart delay themselves).
            steps = [['#clip', 0, 'xiph-johnny'], ['.resolution', 0, '384x216'], ['.fps', 1, '10'], ['.codec-picker', 0, 'hevc'],
                     ['.preset', 1, 'fast'], ['.resolution', 1, '384x216'], ['.fps', 0, '10'], ['#clip', 0, 'svt-waterfall']]
            assert len(steps) == BURST
            page.evaluate("""async ([steps, gap]) => { for (const [selector, index, value] of steps) {
              const control = document.querySelectorAll(selector)[index]; control.value = value; control.dispatchEvent(new Event('change'));
              await new Promise(resolve => setTimeout(resolve, gap)); } }""", [steps, BURST_GAP_MS])
            page.wait_for_function(f'window.__starts > {before}', timeout=FRAME_TIMEOUT_MS)
            page.wait_for_function(SHOWN, timeout=FRAME_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            burst_starts = page.evaluate('window.__starts') - before
            final = page.evaluate('window.__inits.slice(-2)')
            assert burst_starts == 1, f'{burst_starts} starts for one burst'
            assert [(c['codec'], c['width'], c['fps'], c['preset']) for c in final] == [('hevc', 384, 10, 'default'), ('av1', 384, 10, 'fast')], final
            assert page.locator('#clip').input_value() == 'svt-waterfall'
            assert all(c['rendition'] == 'svt-waterfall-360p.mp4' for c in final), final
            assert not page.locator('#notice').is_visible(), page.locator('#notice').inner_text()
            browser.close()
    finally:
        server.shutdown()
    assert not errors, errors
    print(json.dumps({'status': 'pass', 'restarts': restarts, 'burst_changes': BURST, 'burst_starts': burst_starts, 'cpu_slowdown': CPU_SLOWDOWN}))


if __name__ == '__main__':
    main()
