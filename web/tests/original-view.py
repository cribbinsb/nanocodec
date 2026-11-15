#!/usr/bin/env python3
"""An "Original (uncompressed)" view shows exactly the source frame of its tick.

Headless Chromium on the CPU display path (`?display=cpu`, so the canvas holds
the converted pixels), live H.264 in A, and B switched to the original at
640x360 (another rendition than A's). Checks:

- B shows the original: its badge, every metric n/a, no CLI line, no Advanced;
- paused, B's canvas equals colour.js `i420ToRgba` of the source frame of its
  tick regenerated offline (tools/web/make-reference-streams.py `write_source`,
  ffmpeg's decode of the clip placed at the view's size, the frames the
  pre-encoded streams and the replay use) pixel for pixel;
- zoomed in, B's canvas is drawn unsmoothed (nearest neighbour) like A's;
- the debug report (with the coded streams) replays with zero mismatches:
  every picture B showed is byte for byte the regenerated source frame.

  NC_NANOCODEC_BIN=path/to/nanocodec python3 web/tests/original-view.py
"""
import base64
import functools
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
DATA = Path(os.environ.get('NC_DATA_ROOT', Path.home() / 'nanocodec-data'))
WIDTH, HEIGHT = 640, 360
SOURCE_FPS = 30
COMPARED = "parseInt(document.querySelector('#session-stats').textContent)"
FRAMES_BEFORE_PAUSE = 30


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def generator():
    spec = importlib.util.spec_from_file_location('make_reference_streams', REPO / 'tools/web/make-reference-streams.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['make_reference_streams'] = module
    spec.loader.exec_module(module)
    return module


def json_block(path: Path) -> dict:
    text = path.read_text()
    start = text.rfind('```json') + len('```json')
    return json.loads(text[start:text.index('\n```', start)])


def main():
    assert os.environ.get('NC_NANOCODEC_BIN'), 'set NC_NANOCODEC_BIN to a native nanocodec build of this tree'
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    try:
        with tempfile.TemporaryDirectory(dir=DATA / 'work' if (DATA / 'work').is_dir() else None) as scratch, sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(accept_downloads=True, viewport={'width': 1400, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/?views=live&display=cpu')
            page.wait_for_function(f'{COMPARED} >= 1', timeout=120000)
            page.locator('.view-source [data-view-source="original"]').nth(1).click()
            page.locator('.resolution').nth(1).select_option(f'{WIDTH}x{HEIGHT}')
            page.wait_for_function("document.querySelectorAll('.frame-badge')[1].textContent.startsWith('ORIGINAL · UNCOMPRESSED')", timeout=120000)
            page.wait_for_function(f'{COMPARED} >= {FRAMES_BEFORE_PAUSE}', timeout=120000)
            b = page.locator('.version').nth(1)
            assert b.locator('.reproduce').is_hidden() and b.locator('details.advanced').is_hidden()
            assert b.locator('.codec-picker').is_hidden() and b.locator('.resolution').is_visible()
            page.locator('#pause').click()
            page.wait_for_function("document.querySelector('#runtime-label').textContent === 'Paused'", timeout=60000)
            page.wait_for_timeout(1000)
            assert page.locator('[data-metric]').evaluate_all('els => els.slice(4).map(e => e.textContent)') == ['n/a'] * 4
            tick = int(page.evaluate("document.querySelectorAll('.frame-badge')[1].dataset.tick"))
            canvas = page.evaluate("""(() => { const c = document.querySelectorAll('.viewport canvas')[1];
              const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let t = '';
              for (let i = 0; i < d.length; i += 32768) t += String.fromCharCode(...d.subarray(i, i + 32768)); return [c.width, c.height, btoa(t)]; })()""")
            assert canvas[:2] == [WIDTH, HEIGHT], canvas[:2]
            # Zoomed in, the pictures are drawn unsmoothed.
            page.locator('.viewport').nth(1).focus()
            for _ in range(4):
                page.keyboard.press('+')
            assert page.evaluate("[...document.querySelectorAll('.viewport canvas')].map(c => c.style.imageRendering)") == ['pixelated', 'pixelated']
            page.keyboard.press('Escape')
            # The debug report, with the coded streams, through its dialog.
            page.locator('details.more > summary').click()
            page.locator('#debug-report').click()
            page.locator('#debug-streams').set_checked(True)
            with page.expect_download() as download:
                page.locator('#debug-download').click()
            report = Path(scratch) / 'original.md'
            download.value.save_as(report)
            state = json_block(report)
            original = state['views'][1]
            assert original['config']['source'] == 'original' and original['stream']['complete'] is False, original['stream']
            assert state['views'][0]['stream']['complete'], 'the live view embeds its stream'
            # The source frame of B's tick, regenerated offline, converted as the page converts it.
            make = generator()
            rendition, fps = original['config']['rendition'], original['config']['fps']
            source_path = Path(scratch) / 'source.yuv'
            make.write_source(make.Entry(state['source']['clip'], '', WIDTH, HEIGHT, fps, 0), source_path, WEB / 'clips' / rendition)
            frames = np.fromfile(source_path, np.uint8).reshape(-1, WIDTH * HEIGHT * 3 // 2)
            source_path.unlink()
            frame = frames[(tick % state['source']['loopLength']) // (SOURCE_FPS // fps)]
            colour = original['frames'][-1]['colour']
            expected = page.evaluate("""([data, w, h, colour]) => import('./colour.js').then(({ i420ToRgba }) => {
              const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0)); const rgba = i420ToRgba(bytes, w, h, colour); let t = '';
              for (let i = 0; i < rgba.length; i += 32768) t += String.fromCharCode(...rgba.subarray(i, i + 32768)); return btoa(t); })""",
                                     [base64.b64encode(frame.tobytes()).decode(), WIDTH, HEIGHT, colour])
            shown, wanted = base64.b64decode(canvas[2]), base64.b64decode(expected)
            differing = sum(a != b for a, b in zip(shown, wanted))
            assert len(shown) == len(wanted) and differing == 0, f'tick {tick}: {differing} canvas bytes differ from the source frame'
            browser.close()
            replay = subprocess.run([sys.executable, str(REPO / 'tools/web/replay-debug-report.py'), str(report)], capture_output=True, text=True)
            bad = [line for line in replay.stdout.splitlines() if line.startswith('BAD')]
            assert replay.returncode == 0, '\n'.join(bad[:40]) + replay.stdout[-300:] + replay.stderr[-2000:]
            pictures = sum(' source picture ' in line and line.startswith('ok') for line in replay.stdout.splitlines())
            assert pictures >= 5, replay.stdout[-2000:]
    finally:
        server.shutdown()
    assert not errors, errors
    print(json.dumps({'status': 'pass', 'tick': tick, 'pictures_replayed': pictures, 'replay': replay.stdout.splitlines()[-1]}))


if __name__ == '__main__':
    main()
