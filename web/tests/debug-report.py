#!/usr/bin/env python3
"""The debug report reproduces: headless Chromium runs a comparison, pauses,
downloads the debug report (web/debug-report.js), and
tools/web/replay-debug-report.py must find zero mismatches in it.

The report is asked for through its dialog (More, Debug report…): Cancel saves
nothing, Download saves the report with or without the coded streams as ticked.

Two runs: two live views (live H.264 and AV1 on the default clip, `?views=live`),
with "Request keyframes" pressed and A's target rate changed midway and the
streams embedded, where the native re-encode must give every access unit
the page coded and the shown bitrate, PSNR and times must follow from the
report's window; and a pre-encoded view beside a live one with analysis on
(PSNR-Y/Cb/Cr checked too), without streams, whose report must stay under
MAX_DEFAULT_BYTES.

  NC_NANOCODEC_BIN=path/to/nanocodec python3 web/tests/debug-report.py
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
    open_dialog(page)
    page.locator('#debug-streams').set_checked(streams)
    with page.expect_download() as download:
        page.locator('#debug-download').click()
    download.value.save_as(path)
    assert not page.locator('#debug-dialog').is_visible()
    return path


def open_dialog(page):
    """More, then Debug report…: the dialog asks about the streams."""
    page.locator('details.more > summary').click()
    page.locator('#debug-report').click()
    dialog = page.locator('#debug-dialog')
    assert dialog.is_visible() and 'Include the coded streams' in dialog.inner_text()
    assert 'base64' in dialog.inner_text() and 'webcam' in dialog.inner_text()


def cancel_saves_nothing(page):
    """Cancel (and Escape) closes the dialog without a download."""
    downloads = []

    def saved(download):
        downloads.append(download)
    page.on('download', saved)
    for close in (lambda: page.locator('#debug-dialog button[value=cancel]').click(), lambda: page.keyboard.press('Escape')):
        open_dialog(page)
        close()
        page.wait_for_function("!document.querySelector('#debug-dialog').open", timeout=5000)
    page.wait_for_timeout(1000)
    assert not downloads, 'Cancel saved a report'
    page.remove_listener('download', saved)


def parse_state(path: Path) -> dict:
    """The report's JSON block."""
    text = path.read_text()
    start = text.rfind('```json') + len('```json')
    return json.loads(text[start:text.index('\n```', start)])


def replay(path: Path):
    result = subprocess.run([sys.executable, str(REPO / 'tools/web/replay-debug-report.py'), str(path)], capture_output=True, text=True)
    bad = [line for line in result.stdout.splitlines() if line.startswith('BAD')]
    assert result.returncode == 0, '\n'.join(bad[:40]) + result.stdout[-300:] + result.stderr[-2000:]
    print(result.stdout.splitlines()[-1])
    return result.stdout


def main():
    assert os.environ.get('NC_NANOCODEC_BIN'), 'set NC_NANOCODEC_BIN to a native nanocodec build of this tree'
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    with tempfile.TemporaryDirectory() as scratch, sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(accept_downloads=True, viewport={'width': 1400, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.server_port}/?views=live')
        # A key frame request midway: the replay codes it at the frame the page did.
        page.wait_for_function(f'{COMPARED} >= {FRAMES_BEFORE_PAUSE // 2}', timeout=120000)
        page.locator('details.more > summary').click()
        page.locator('#keyframe').click()
        page.locator('.target-rate').nth(0).fill('400')
        page.locator('.target-rate').nth(0).press('Enter')
        cancel_saves_nothing(page)
        default = report_of(page, Path(scratch) / 'default.md', streams=True)
        state = parse_state(default)
        assert all(view['stream']['complete'] for view in state['views']), [view.get('stream') for view in state['views']]
        assert all(view['requestedKeys'] for view in state['views']), 'the key frame request was not recorded'
        assert [c['targetKbps'] for c in state['views'][0]['rateChanges']] == [400], state['views'][0]['rateChanges']
        replay(default)
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
        assert all('stream' not in view for view in parse_state(report)['views'])
        replay(report)
        browser.close()
    server.shutdown()
    assert not errors, errors
    print('debug report: pass')


if __name__ == '__main__':
    main()
