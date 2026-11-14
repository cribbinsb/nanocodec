#!/usr/bin/env python3
"""Headless Chromium smoke of the analysis mode (analysis-view.js), after
tools/web/build.sh: the default page has no analysis elements; switching
analysis on loads the analysis module (pkg/analysis/) without restarting the
comparison, draws the overlay and fills the statistics of the H.264 and AV1
views and of an AV2 view and the line comparing the two views, zooms both
views together and shows a block's record on hover; switching it off restores
the page.

Switching keeps the session (app.js, pipeline.js `switchDecoders`), in every
case: paused or playing, a sample clip (decoded frame by frame, exact-source.js,
and, without WebCodecs, sampled while it plays) or the webcam, a live encode or
a pre-encoded stream. Paused stays paused on the same instant (both views' ticks
and the frame count unchanged), with the overlay drawn; playing goes on from
where it was (the count keeps rising, the ticks do not go back); live encoders
are never re-created. A paused switch shows bit-identical pictures (the new
decoders decode the same stream again), and switched on, the paused frame
gets its MS-SSIM.

The PSNR map (error-map.js): drawn with several colours, a finite per-frame
PSNR of each plane in both views' statistics, the luma value matching an
independent computation (this test's own, from the decoded and source planes
the decode worker posted, on the WebGL display's packed I420) to 0.01 dB, the
block's PSNR in the hover record; off again restores the page."""
import functools
import http.server
import threading
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

web = Path(__file__).resolve().parents[1]
# Frames a playing comparison must deliver after a switch to count as going on.
FRAMES_AFTER_SWITCH = 5
# Time for the frames in flight at a pause to land before its instant is read (ms).
SETTLE_MS = 1000

# Records which workers the page starts (an encoder re-created would be a restart).
WORKER_HOOK = '''window.workerScripts = [];
  const Original = Worker;
  window.Worker = class extends Original {
    constructor(url, ...args) { super(url, ...args); window.workerScripts.push(String(url).split('/').pop()); }
  };'''
STATE = r'''() => ({
  ticks: [...document.querySelectorAll('.frame-badge')].map(b => +b.dataset.tick),
  compared: +(document.querySelector('#session-stats').textContent.match(/(\d+) frames compared/)?.[1] ?? -1),
  label: document.querySelector('#runtime-label').textContent,
  videoPaused: document.querySelector('#input-video').paused,
  encoders: window.workerScripts.filter(s => s === 'encode-worker.js').length,
  decoders: window.workerScripts.filter(s => s === 'decode-worker.js').length,
  // A checksum of each view's picture (the CPU display draws it 2D at its own size; null on a GL display).
  pictures: [...document.querySelectorAll('.viewport canvas:not(.analysis-layer)')].map(c =>
    c.getContext('2d')?.getImageData(0, 0, c.width, c.height).data.reduce((h, v) => (h * 31 + v) >>> 0, 7) ?? null),
})'''
DRAWN = """[...document.querySelectorAll('.analysis-layer')].every(c => !c.hidden &&
  c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some((v, i) => i % 4 === 3 && v))"""
LIVE = "document.querySelector('#runtime-label').textContent === 'Encoding + decoding live'"
# Keeps a copy of the last frame each decode worker posted with a PSNR map:
# its decoded picture (packed I420 on the WebGL display) and its source.
FRAME_HOOK = '''window.mapFrames = [];
  const Original = Worker;
  window.Worker = class extends Original {
    constructor(url, ...args) {
      super(url, ...args);
      if (!String(url).endsWith('decode-worker.js')) return;
      const index = window.mapFrames.push(null) - 1;
      this.addEventListener('message', ({ data }) => {
        if (data.errorMap && data.format === 'i420' && data.source) window.mapFrames[index] = {
          width: data.width, height: data.height, pixels: data.pixels.slice(), source: new Uint8Array(data.source.slice(0)) };
      });
    }
  };'''
# This test's own PSNR-Y of the last mapped frame of each of the two newest decoders (views A and B).
INDEPENDENT_PSNR_Y = """window.mapFrames.slice(-2).map(({ width, height, pixels, source }) => {
  let sse = 0;
  for (let i = 0; i < width * height; i++) sse += (pixels[i] - source[i]) ** 2;
  return sse ? 10 * Math.log10(255 * 255 * width * height / sse) : 100;
})"""
SHOWN_PSNR = r"""[...document.querySelectorAll('.analysis-stats')].map(s => {
  const row = [...s.querySelectorAll('tr')].find(r => r.textContent.startsWith('PSNR vs source'));
  return row ? [...row.querySelector('td').textContent.matchAll(/(Y|Cb|Cr) ([\d.]+)/g)].map(m => +m[2]) : null;
})"""
# Distinct colours on the first view's overlay layer.
LAYER_COLOURS = """(() => { const c = document.querySelector('.analysis-layer'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data, seen = new Set();
  for (let i = 0; i < d.length; i += 4) if (d[i + 3]) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
  return seen.size; })()"""
# The PSNR map shows several colours (a flat map would mean nothing was measured).
MIN_MAP_COLOURS = 4
PSNR_TOLERANCE_DB = 0.01
# A wait for something the page does once (a module load, a decoder switch, a restart) fails after this long (ms).
EVENT_TIMEOUT_MS = 60000
# A wait for frames fails only when no frame has been compared for this long (s): a
# slow machine delivers slowly, a stalled page not at all.
STALL_S = 30
FRAME_POLL_S = 0.5
# Each view's MS-SSIM row names the paused frame's tick (the metrics worker scored it).
PAUSED_FRAME_SCORED = """[...document.querySelectorAll('.version')].every(v => {
  const row = [...v.querySelectorAll('.analysis-stats tr')].find(r => r.textContent.startsWith('MS-SSIM'));
  return row && row.querySelector('td').textContent.endsWith(`tick ${v.querySelector('.frame-badge').dataset.tick}`); })"""
# A paused frame is scored within the next once-a-second slot, after the decoder switch (ms).
MS_SSIM_TIMEOUT_MS = 15000
# Page events kept per page for a failure's report (trace.js lines on the console).
EVENTS_SHOWN = 40
EVENTS = {}


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def state(page):
    return page.evaluate(STATE)


def watch(page):
    """Keep the page's lifecycle events (trace.js), for a failure's report."""
    EVENTS[page] = []
    page.on('console', lambda m: EVENTS[page].append(m.text) if m.text.startswith('page: ') else None)


def stalled(page, what):
    """An assertion naming what the test waited for, the page's state and its last events."""
    session = page.evaluate("document.querySelector('#session-stats').textContent")
    return AssertionError(f'waited for {what}; state {state(page)}; session "{session}"; last events:\n' + '\n'.join(EVENTS.get(page, [])[-EVENTS_SHOWN:]))


def wait(page, condition, what, timeout=EVENT_TIMEOUT_MS):
    """Wait for a JS `condition`; on timeout fail with what the page was doing."""
    try:
        page.wait_for_function(condition, timeout=timeout)
    except PlaywrightTimeout:
        raise stalled(page, what) from None


def wait_frames(page, frames):
    """Wait until `frames` more frames are compared, failing only if none comes for STALL_S."""
    target = state(page)['compared'] + frames
    last, progress_at = None, time.monotonic()
    while (compared := state(page)['compared']) < target:
        if compared != last:
            last, progress_at = compared, time.monotonic()
        elif time.monotonic() - progress_at > STALL_S:
            raise stalled(page, f'{frames} more frames (no frame for {STALL_S} s)')
        time.sleep(FRAME_POLL_S)


def wait_running(page, frames=FRAMES_AFTER_SWITCH):
    """Wait until the comparison is live and has compared `frames` more frames."""
    wait(page, LIVE, 'the comparison to run')
    wait(page, "document.querySelector('#session-stats').textContent.includes('fps')", 'the first delivery rate')
    wait_frames(page, frames)


def toggle_analysis(page, on):
    """Click the analysis switch and wait until the decoders have switched."""
    page.locator('.analysis-toggle').click()
    wait(page, "document.querySelector('#runtime-label').textContent !== 'Switching decoders…'", 'the decoder switch')
    if on:
        wait(page, "[...document.querySelectorAll('.analysis-stats')].every(s => s.querySelector('table'))", 'both statistics tables')
        page.wait_for_function(DRAWN, timeout=10000)
    else:
        page.wait_for_function("[...document.querySelectorAll('.analysis-layer')].every(c => c.hidden)", timeout=10000)


def pause(page):
    page.locator('#pause').click()
    page.wait_for_function("document.querySelector('#runtime-label').textContent === 'Paused'", timeout=30000)
    page.wait_for_timeout(SETTLE_MS)


def resume(page):
    page.locator('#pause').click()
    wait_running(page)


def check_paused_switch(page, case, on):
    """Paused: the switch keeps the same instant, count and pause state."""
    before = state(page)
    toggle_analysis(page, on)
    after = state(page)
    assert after['label'] == 'Paused' and after['videoPaused'], (case, after)
    assert after['ticks'] == before['ticks'], (case, before, after)
    assert after['compared'] == before['compared'], (case, before, after)
    assert after['pictures'] == before['pictures'], (case, before, after)
    assert after['encoders'] == before['encoders'], (case, before, after)
    # Every view's decoder was replaced.
    assert after['decoders'] == before['decoders'] + 2, (case, before, after)
    assert page.locator('#pause').text_content() == 'Resume', case
    if on:
        # The paused frame is always scored (MS-SSIM, at most once a second while playing).
        wait(page, PAUSED_FRAME_SCORED, f'{case}: MS-SSIM of the paused frame', MS_SSIM_TIMEOUT_MS)


def check_playing_switch(page, case, on):
    """Playing: the switch goes on from the instant on screen, encoders kept."""
    before = state(page)
    toggle_analysis(page, on)
    wait_running(page)
    after = state(page)
    assert after['compared'] >= before['compared'] + FRAMES_AFTER_SWITCH, (case, before, after)
    assert all(a >= b for a, b in zip(after['ticks'], before['ticks'])), (case, before, after)
    assert after['encoders'] == before['encoders'], (case, before, after)


def check_switch_cases(page, case):
    """Paused on, paused off, then playing on, playing off, ending with analysis off and playing."""
    pause(page)
    check_paused_switch(page, f'{case} paused', True)
    check_paused_switch(page, f'{case} paused', False)
    resume(page)
    check_playing_switch(page, f'{case} playing', True)
    check_playing_switch(page, f'{case} playing', False)


def check_sampled_clip(browser, port):
    """Without WebCodecs the clip is sampled while it plays (capture.js): switching keeps the session there too."""
    page = browser.new_page(viewport={'width': 1400, 'height': 1000})
    page.on('pageerror', lambda error: errors.append(str(error)))
    watch(page)
    page.add_init_script(WORKER_HOOK + 'delete window.VideoDecoder;')
    page.goto(f'http://127.0.0.1:{port}/?display=cpu')
    wait_running(page)
    check_switch_cases(page, 'sampled clip live')
    page.close()


def check_psnr_map(browser, port):
    """The PSNR map, statistics and hover record, on a paused comparison."""
    page = browser.new_page(viewport={'width': 1400, 'height': 1000})
    page.on('pageerror', lambda error: errors.append(str(error)))
    watch(page)
    page.add_init_script(WORKER_HOOK)
    page.add_init_script(FRAME_HOOK)
    # Headless Chromium's software GL is refused by the automatic choice: ask for WebGL2.
    page.goto(f'http://127.0.0.1:{port}/?display=webgl2')
    wait_running(page)
    pause(page)
    toggle_analysis(page, True)
    # The PSNR map alone (blocks and vectors off), Y plane.
    for layer in ['blocks', 'vectors', 'psnr']:
        page.locator(f'[data-layer="{layer}"]').click()
    page.wait_for_timeout(300)
    assert page.locator('.analysis-psnr').is_visible()
    assert 'PSNR vs source (dB): 20 or less' in page.locator('.analysis-psnr-scale').text_content()
    colours = page.evaluate(LAYER_COLOURS)
    assert colours >= MIN_MAP_COLOURS, colours
    shown = page.evaluate(SHOWN_PSNR)
    assert all(values and len(values) == 3 and all(0 < v <= 100 for v in values) for values in shown), shown
    independent = page.evaluate(INDEPENDENT_PSNR_Y)
    assert all(abs(values[0] - own) <= PSNR_TOLERANCE_DB for values, own in zip(shown, independent)), (shown, independent)
    # The Cb plane draws another map.
    page.locator('[data-plane="1"]').click()
    page.wait_for_timeout(300)
    assert page.evaluate(LAYER_COLOURS) >= 1
    # Hovering a block of A shows its PSNR.
    box = page.locator('.viewport').first.bounding_box()
    page.mouse.move(box['x'] + box['width'] / 3, box['y'] + box['height'] / 3)
    page.wait_for_function("!document.querySelector('.analysis-tip').hidden", timeout=5000)
    assert 'PSNR vs source, 16×16 block' in page.locator('.analysis-tip').first.text_content()
    # Off again: the page is restored, still paused.
    toggle_analysis(page, False)
    assert not page.locator('.analysis-psnr').is_visible()
    assert page.evaluate("[...document.querySelectorAll('.analysis-stats')].every(s => s.hidden)")
    assert page.locator('#runtime-label').text_content() == 'Paused'
    page.close()


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(web)))
threading.Thread(target=server.serve_forever, daemon=True).start()
errors = []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
        page = browser.new_page(viewport={'width': 1400, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        watch(page)
        page.add_init_script(WORKER_HOOK)
        page.goto(f'http://127.0.0.1:{server.server_port}/?display=cpu')
        # The default comparison (A: H.264, B: AV1) runs; no analysis element exists yet.
        wait_running(page)
        assert page.evaluate("document.querySelectorAll('.analysis-layer, .analysis-stats').length") == 0
        assert page.evaluate("[...document.querySelectorAll('.viewport')].map(v => v.querySelectorAll('canvas').length)") == [1, 1]
        # Sample clip, two live encodes: every switch keeps the session.
        check_switch_cases(page, 'clip live')
        assert any('/pkg/analysis/' in n for n in page.evaluate("performance.getEntriesByType('resource').map(e => e.name)"))
        # On again for the overlay checks: statistics, difference line, zoom, hover.
        toggle_analysis(page, True)
        page.wait_for_function("document.querySelector('.analysis-difference').textContent.includes('mean by area')", timeout=10000)
        page.locator('.analysis-zoom [data-zoom="in"]').click()
        page.wait_for_timeout(100)
        transforms = page.evaluate("[...document.querySelectorAll('.viewport')].map(v => v.querySelector('canvas').style.transform)")
        assert all('scale(2)' in t for t in transforms), transforms
        box = page.locator('.viewport').first.bounding_box()
        page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
        page.wait_for_function("!document.querySelector('.analysis-tip').hidden", timeout=5000)
        assert 'QP' in page.locator('.analysis-tip').first.text_content()
        # Every codec's views are analysed: an AV2 view fills its statistics too.
        page.locator('.codec-picker').nth(1).select_option('av2')
        wait(page, "document.querySelectorAll('.analysis-stats')[1].textContent.includes('Prediction')", 'the AV2 view\'s statistics')
        # Off: the overlay hides and the pictures are unzoomed.
        toggle_analysis(page, False)
        assert page.evaluate("[...document.querySelectorAll('.viewport')].every(v => !v.querySelector('canvas').style.transform)")
        # Sample clip, B a pre-encoded stream (its decoder re-decodes from the stream's last key).
        page.locator('.view-source [data-view-source="reference"]').nth(1).click()
        wait(page, "document.querySelectorAll('.frame-badge')[1].textContent.includes('PRE-ENCODED')", 'the pre-encoded view')
        wait_running(page)
        check_switch_cases(page, 'clip reference')
        page.locator('.view-source [data-view-source="live"]').nth(1).click()
        # The webcam (Chromium's fake camera), two live encodes.
        page.locator('[data-source="camera"]').click()
        page.locator('#start').click()
        wait_running(page)
        check_switch_cases(page, 'webcam live')
        check_sampled_clip(browser, server.server_port)
        check_psnr_map(browser, server.server_port)
        browser.close()
finally:
    server.shutdown()
assert not errors, errors
print('analysis smoke: pass')
