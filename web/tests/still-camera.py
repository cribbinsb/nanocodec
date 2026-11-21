#!/usr/bin/env python3
"""Each live view's "Still-camera noise handling" toggle reaches its encoder.

The toggle is in each live view's Advanced section, on by default, with its
explanation as the tooltip. For the toggles on (the default) and off, two live
views (`?views=live`) run until FRAMES_BEFORE_PAUSE frames, pause and save the
debug report. Checks:

- every live view's configuration in the report carries its toggle's state, and
  the view's CLI command has `--still-camera` exactly when on;
- each encode worker was created with that state (its init message);
- the report replays natively with zero mismatches (tools/web/replay-debug-report.py
  passes `--still-camera` when the view had it), so the WASM streams are the
  native encoder's with and without the setting.

  NC_NANOCODEC_BIN=path/to/nanocodec python3 web/tests/still-camera.py
"""
import functools
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
FRAMES_BEFORE_PAUSE = 20
COMPARED = "parseInt(document.querySelector('#session-stats').textContent)"
HOOK = '''window.__encoderInits = [];
const OriginalWorker = Worker;
window.Worker = class extends OriginalWorker {
  postMessage(message, ...rest) {
    if (message && message.type === 'init' && message.config) window.__encoderInits.push(Boolean(message.config.stillCamera));
    return super.postMessage(message, ...rest);
  }
};'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def report(page, path: Path) -> dict:
    """Pause after some frames, save the debug report to `path`, and return its JSON block."""
    page.wait_for_function(f'{COMPARED} >= {FRAMES_BEFORE_PAUSE}', timeout=120000)
    page.locator('#pause').click()
    page.wait_for_function("document.querySelector('#runtime-label').textContent === 'Paused'", timeout=30000)
    page.wait_for_timeout(1000)
    page.locator('details.more > summary').click()
    page.locator('#debug-report').click()
    with page.expect_download() as download:
        page.locator('#debug-download').click()
    download.value.save_as(path)
    text = path.read_text()
    start = text.rfind('```json') + len('```json')
    return json.loads(text[start:text.index('\n```', start)])


def run(port, on: bool, directory: Path) -> dict:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--autoplay-policy=no-user-gesture-required'])
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(HOOK)
        page.goto(f'http://127.0.0.1:{port}/?views=live')
        for summary in page.locator('details.advanced > summary').all():
            summary.click()
        toggles = page.locator('.still-camera').all()
        assert len(toggles) == 2 and all(t.is_checked() for t in toggles), 'the toggles are on by default'
        assert all('sensor noise' in t.locator('xpath=..').get_attribute('title') for t in toggles)
        if not on:
            page.wait_for_function(f'{COMPARED} >= 1', timeout=120000)
            for toggle in toggles:
                toggle.set_checked(False)
            page.wait_for_timeout(500)
        state = report(page, directory / f'still-camera-{"on" if on else "off"}.md')
        page.wait_for_function("[...document.querySelectorAll('.reproduce .cli')].every(c => c.textContent)", timeout=10000)
        commands = page.locator('.reproduce .cli').all_text_contents()
        inits = page.evaluate('window.__encoderInits')
        browser.close()
    assert not errors, errors
    live = [view for view in state['views'] if view['config']['source'] == 'live']
    assert live and all(view['config']['stillCamera'] is on for view in live), [v['config'] for v in state['views']]
    assert all(('--still-camera' in command) is on for command in commands), commands
    # The last encoder of each view was created with the toggle's state.
    assert inits[-len(live):] == [on] * len(live), inits
    return {'views': len(live), 'encoder_inits': inits}


def main():
    assert os.environ.get('NC_NANOCODEC_BIN'), 'set NC_NANOCODEC_BIN to a native nanocodec build of this tree'
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = {}
    try:
        with tempfile.TemporaryDirectory(prefix="nanocodec-web-test-") as scratch:
            for on in (True, False):
                result = run(server.server_port, on, Path(scratch))
                replay = subprocess.run([sys.executable, str(REPO / 'tools/web/replay-debug-report.py'),
                                         str(Path(scratch) / f'still-camera-{"on" if on else "off"}.md')],
                                        capture_output=True, text=True)
                assert replay.returncode == 0, replay.stdout[-2000:] + replay.stderr[-2000:]
                results['on' if on else 'off'] = {**result, 'replay': replay.stdout.splitlines()[-1]}
    finally:
        server.shutdown()
    print(json.dumps({'status': 'pass', **results}))


if __name__ == '__main__':
    main()
