#!/usr/bin/env python3
"""The debug report reproduces: headless Chromium runs a comparison, pauses,
downloads the debug report (web/debug-report.js), and
tools/web/replay-debug-report.py must find zero mismatches in it.

Two runs: the default comparison (live H.264 and AV1 on the default clip) with
the streams embedded, where the native re-encode must give every access unit
the page coded and the shown bitrate, PSNR and times must follow from the
report's window; and a pre-encoded view beside a live one with analysis on
(PSNR-Y/Cb/Cr checked too), without streams, whose report must stay under
MAX_DEFAULT_BYTES.

  NC_NANOCODEC_BIN=path/to/nanocodec python3 web/tests/debug-report.py
"""
import functools
import http.server
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
MAX_DEFAULT_BYTES = 1024 * 1024
FRAMES_BEFORE_PAUSE = 60
STREAM = 'xiph-johnny-openh264-640x360-15fps-150k'
COMPARED = "+(document.querySelector('#session-stats').textContent.match(/(\\d+) frames compared/) || [0, 0])[1]"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def report_of(page, path: Path, streams: bool) -> Path:
    """Wait for frames, pause, and save the debug report to `path`."""
    page.wait_for_function(f'{COMPARED} >= {FRAMES_BEFORE_PAUSE}', timeout=120000)
    page.locator('#pause').click()
    page.wait_for_function("document.querySelector('#runtime-label').textContent === 'Paused'", timeout=30000)
    page.wait_for_timeout(1000)
    page.locator('#debug-streams').set_checked(streams)
    with page.expect_download() as download:
        page.locator('#debug-report').click()
    download.value.save_as(path)
    return path


def replay(path: Path):
    result = subprocess.run([sys.executable, str(REPO / 'tools/web/replay-debug-report.py'), str(path)], capture_output=True, text=True)
    bad = [line for line in result.stdout.splitlines() if line.startswith('BAD')]
    assert result.returncode == 0, '\n'.join(bad[:40]) + result.stdout[-300:] + result.stderr[-2000:]
    print(result.stdout.splitlines()[-1])


def main():
    assert os.environ.get('NC_NANOCODEC_BIN'), 'set NC_NANOCODEC_BIN to a native nanocodec build of this tree'
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    with tempfile.TemporaryDirectory() as scratch, sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(accept_downloads=True, viewport={'width': 1400, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.server_port}/')
        replay(report_of(page, Path(scratch) / 'default.md', streams=True))
        # A pre-encoded view beside a live one, analysis on.
        page.locator('#pause').click()
        page.locator('.view-source [data-view-source="reference"]').nth(0).click()
        page.wait_for_function("document.querySelectorAll('.stream-picker')[0].options.length > 10", timeout=30000)
        page.locator('.stream-picker').nth(0).select_option(STREAM)
        page.wait_for_function("document.querySelectorAll('.frame-badge')[0].textContent.includes('PRE-ENCODED')", timeout=60000)
        page.locator('.analysis-toggle').click()
        page.wait_for_function("[...document.querySelectorAll('.analysis-stats')].every(s => s.querySelector('table'))", timeout=60000)
        report = report_of(page, Path(scratch) / 'reference.md', streams=False)
        assert report.stat().st_size < MAX_DEFAULT_BYTES, report.stat().st_size
        replay(report)
        browser.close()
    server.shutdown()
    assert not errors, errors
    print('debug report: pass')


if __name__ == '__main__':
    main()
