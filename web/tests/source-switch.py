#!/usr/bin/env python3
"""Source switches and webcam starts under CPU throttling never break capture.

Just after a source switch, a webcam start or a loop wrap the page's video
element can briefly hold no frame; taking a VideoFrame of it then throws
(capture.js `videoSnapshot` skips that tick instead). Headless Chromium with a
synthetic webcam, the page's CPU throttled 6x (DevTools emulation), switches
between webcam, sample clip and test scene several times, sometimes before the
previous source has produced a frame, and requires frames from each started
source, no page error and no failure notice.

  python3 web/tests/source-switch.py
"""
import functools
import http.server
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
CPU_SLOWDOWN = 6
ROUNDS = 3
FRAME_TIMEOUT_MS = 90000
# Both views have shown a frame since their last restart.
BOTH_SHOWN = "[...document.querySelectorAll('.frame-badge')].every(b => b.dataset.tick !== undefined)"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.context.new_cdp_session(page).send('Emulation.setCPUThrottlingRate', {'rate': CPU_SLOWDOWN})
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            def frames_shown():
                try:
                    page.wait_for_function(BOTH_SHOWN, timeout=FRAME_TIMEOUT_MS)
                except Exception as error:
                    raise AssertionError(f"no frames; notice: {page.locator('#notice').inner_text()!r}; errors: {errors}") from error

            frames_shown()

            def start(source, wait=True):
                page.locator(f'[data-source="{source}"]').click()  # stops the running comparison
                page.locator('#start').click()
                if wait:
                    frames_shown()

            for _ in range(ROUNDS):
                start('camera')
                # Switch away before the clip has a frame, then back to the webcam at once.
                start('scene', wait=False)
                start('camera')
                start('clip', wait=False)
                page.locator('#clip').select_option('svt-waterfall')  # a clip change restarts it
                frames_shown()
            notice = page.locator('#notice')
            assert not errors, errors
            assert not notice.is_visible(), notice.inner_text()
            browser.close()
        print(f'PASS source switches: {ROUNDS} rounds of webcam / scene / clip starts at {CPU_SLOWDOWN}x CPU slowdown')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
