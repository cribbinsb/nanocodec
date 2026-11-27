#!/usr/bin/env python3
"""Live views key "On clip start" by default on a sample clip, where a pre-encoded
stream keys (view-settings.js KEY_AT_CLIP_START, encode-worker.js).

Two live views (`?views=live`) on the default clip run past the clip's first loop,
pause and save the debug report with the coded streams. Checks, per view:

- the key-interval picker says "On clip start" and the rate line "key at each clip start";
- the coded key frames are exactly the first frame of each loop the run reached;
- the CLI command's `--intra-period` is one loop of frames (a loop of whole frames);
- the report replays natively with zero mismatches.

A second run, on a clip whose loop is not a whole number of B's frames at 6 fps: its
key is requested at the first frame coded in the new loop (`requestedKeys`, `--keyframe`
in the CLI); the native replay cannot take such a loop yet (it writes one loop of
source frames). Switching to the webcam then offers no
"On clip start" and the picker shows 30 seconds. A target over half the clip file's
bitrate is flagged in that view's rate line.

  NC_NANOCODEC_BIN=path/to/nanocodec python3 web/tests/clip-start-keys.py
"""
import functools
import http.server
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
SOURCE_FPS = 30
# A clip whose loop (336 ticks) is not a whole number of frames at every PARTIAL_STEP-th tick (6 fps).
PARTIAL_CLIP = 'bbb-tree'
PARTIAL_STEP = 5
# A target over half of any hosted clip file's bitrate (the heaviest is ~8.4 Mbps).
OVER_SOURCE_KBPS = 20000
# Frames past the loop's end before pausing: the key and a few frames after it.
PAST_LOOP_TICKS = 30
TICK = "(+(document.querySelector('.frame-badge').dataset.tick || 0))"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def report(page, path: Path, loop_ticks: int) -> dict:
    """Run past the first loop, pause, save the report (with streams) and return its JSON block."""
    page.wait_for_function(f"[...document.querySelectorAll('.frame-badge')].every(b => +(b.dataset.tick || 0) >= {loop_ticks + PAST_LOOP_TICKS})",
                           timeout=180000)
    page.locator('#pause').click()
    page.wait_for_function("document.querySelector('#runtime-label').textContent === 'Paused'", timeout=30000)
    page.wait_for_timeout(1000)
    page.locator('details.more > summary').click()
    page.locator('#debug-report').click()
    page.locator('#debug-streams').set_checked(True)
    with page.expect_download() as download:
        page.locator('#debug-download').click()
    download.value.save_as(path)
    text = path.read_text()
    start = text.rfind('```json') + len('```json')
    return json.loads(text[start:text.index('\n```', start)])


def replay(path: Path) -> str:
    result = subprocess.run([sys.executable, str(REPO / 'tools/web/replay-debug-report.py'), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return result.stdout.splitlines()[-1]


def check_keys(state: dict, loop_ticks: int):
    """Each live view's coded keys are the first frame it coded in each loop."""
    for view in state['views']:
        step = SOURCE_FPS // view['config']['fps']
        # The report keeps at least the last debug-report.js MIN_HISTORY_FRAMES (300) frames.
        coded = [f for f in view['frames'] if not f['repeat']]
        keys = [f['tick'] for f in coded if f['key']]
        first, last = coded[0]['tick'], coded[-1]['tick']
        starts = [math.ceil(loop * loop_ticks / step) * step for loop in range(last // loop_ticks + 1)]
        expected = [tick for tick in starts if first <= tick]
        assert len(expected) >= 1 and last > loop_ticks, (first, last, loop_ticks)
        assert keys == expected, (view['config']['fps'], keys, expected)


def main():
    assert os.environ.get('NC_NANOCODEC_BIN'), 'set NC_NANOCODEC_BIN to a native nanocodec build of this tree'
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, results = [], {}
    with tempfile.TemporaryDirectory(prefix="nanocodec-web-test-") as scratch, sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
        page = browser.new_page(accept_downloads=True, viewport={'width': 1400, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.server_port}/?views=live')
        assert page.locator('.gop').evaluate_all("s => s.map(g => g.value)") == ['clip-start', 'clip-start']
        page.wait_for_function(f'{TICK} > 0', timeout=120000)
        page.wait_for_function("[...document.querySelectorAll('.rate-status')].every(r => /key at each clip start, every/.test(r.textContent))",
                               timeout=60000)
        # About the loop length, from the rate line ("every 10 s": 300 ticks); the report has it exactly.
        seconds = float(re.search(r'key at each clip start, every ([\d.]+) s', page.locator('.rate-status').nth(0).inner_text())[1])
        state = report(page, Path(scratch) / 'whole.md', math.ceil(seconds * SOURCE_FPS))
        loop_ticks = state['source']['loopLength']
        check_keys(state, loop_ticks)
        commands = page.locator('.reproduce .cli').all_text_contents()
        for view, command in zip(state['views'], commands):
            step = SOURCE_FPS // view['config']['fps']
            if loop_ticks % step == 0:
                assert f'--intra-period {loop_ticks // step} ' in command, command
                assert '--keyframe' not in command, command
        results['whole'] = {'loop_ticks': loop_ticks, 'replay': replay(Path(scratch) / 'whole.md')}
        # A frame rate whose frames do not divide the loop (PARTIAL_CLIP's 336 ticks at every
        # 5th): B's key is requested at the first frame it codes in the new loop.
        # A setting change restarts the paused comparison, running.
        page.locator('#clip').select_option(PARTIAL_CLIP)
        page.locator('.fps').nth(1).select_option(str(SOURCE_FPS // PARTIAL_STEP))
        page.wait_for_function("/not a whole number of frames/.test(document.querySelectorAll('.rate-status')[1].textContent)", timeout=120000)
        seconds = float(re.search(r'key at each clip start, every ([\d.]+) s', page.locator('.rate-status').nth(1).inner_text())[1])
        state = report(page, Path(scratch) / 'partial.md', math.ceil(seconds * SOURCE_FPS))
        loop_ticks = state['source']['loopLength']
        assert loop_ticks % PARTIAL_STEP, loop_ticks
        check_keys(state, loop_ticks)
        assert state['views'][1]['requestedKeys'], 'the loop key was not requested'
        assert '--keyframe' in page.locator('.reproduce .cli').nth(1).inner_text()
        # Not replayed: tools/web/replay-debug-report.py writes one loop of source frames,
        # which needs the loop to be a whole number of frames (make-reference-streams.py write_source).
        results['partial'] = {'loop_ticks': loop_ticks, 'fps': SOURCE_FPS // PARTIAL_STEP}
        # The rate line flags a target over half the clip file's bitrate (view-settings.js overSourceShare).
        page.locator('#pause').click()  # resume: the line is redrawn with frames
        page.locator('.target-rate').nth(0).fill(str(OVER_SOURCE_KBPS))
        page.locator('.target-rate').nth(0).press('Enter')
        page.wait_for_function("/over half the clip file's \\d+ kbps/.test(document.querySelectorAll('.rate-status')[0].textContent)", timeout=60000)
        assert 'over half' not in page.locator('.rate-status').nth(1).inner_text()
        # The webcam does not loop: no "On clip start", 30 s.
        page.locator('#pause').click()
        page.locator('[data-source="camera"]').click()
        assert page.locator('.gop').evaluate_all("s => s.map(g => g.value)") == ['30', '30']
        assert page.locator('.gop option[value="clip-start"]').evaluate_all("o => o.every(x => x.disabled && x.hidden)")
        page.locator('[data-source="clip"]').click()
        assert page.locator('.gop').evaluate_all("s => s.map(g => g.value)") == ['clip-start', 'clip-start']
        browser.close()
    server.shutdown()
    assert not errors, errors
    print(json.dumps({'status': 'pass', **results}))


if __name__ == '__main__':
    main()
