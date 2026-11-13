#!/usr/bin/env python3
"""Headless Chromium smoke of the analysis mode (analysis-view.js), after
tools/web/build.sh: the default page has no analysis elements; switching
analysis on restarts the comparison with the analysis module (pkg/analysis/),
draws the overlay and fills the statistics of the H.264 and AV1 views, says
an AV2 view is not supported yet, zooms both views together and shows a block's
record on hover; switching it off restores the page."""
import functools
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

web = Path(__file__).resolve().parents[1]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(web)))
threading.Thread(target=server.serve_forever, daemon=True).start()
errors = []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1400, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.server_port}/?display=cpu')
        # The default comparison (A: H.264, B: AV1) runs; no analysis element exists yet.
        page.wait_for_function("document.querySelector('#session-stats').textContent.includes('fps')", timeout=60000)
        assert page.evaluate("document.querySelectorAll('.analysis-layer, .analysis-stats').length") == 0
        assert page.evaluate("[...document.querySelectorAll('.viewport')].map(v => v.querySelectorAll('canvas').length)") == [1, 1]
        page.locator('.analysis-toggle').click()
        # The restart loads the analysis module; both views' statistics fill (H.264, AV1).
        page.wait_for_function("[...document.querySelectorAll('.analysis-stats')].every(s => s.querySelector('table'))", timeout=60000)
        assert any('/pkg/analysis/' in n for n in page.evaluate("performance.getEntriesByType('resource').map(e => e.name)"))
        page.wait_for_timeout(500)
        drawn = """(() => { const c = document.querySelector('.analysis-layer');
          return c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some((v, i) => i % 4 === 3 && v); })()"""
        assert page.evaluate(drawn)
        # Zoom applies to both pictures.
        page.locator('.analysis-zoom [data-zoom="in"]').click()
        page.wait_for_timeout(100)
        transforms = page.evaluate("[...document.querySelectorAll('.viewport')].map(v => v.querySelector('canvas').style.transform)")
        assert all('scale(2)' in t for t in transforms), transforms
        # Hovering a block of A shows its record.
        box = page.locator('.viewport').first.bounding_box()
        page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
        page.wait_for_function("!document.querySelector('.analysis-tip').hidden", timeout=5000)
        assert 'QP' in page.locator('.analysis-tip').first.text_content()
        # A codec without records yet (AV2) says so.
        page.locator('.codec-picker').nth(1).select_option('av2')
        page.wait_for_function("document.querySelectorAll('.analysis-stats')[1].textContent.includes('not available')", timeout=60000)
        # Off: the overlay hides and the pictures are unzoomed.
        page.locator('.analysis-toggle').click()
        page.wait_for_timeout(200)
        assert page.evaluate("[...document.querySelectorAll('.analysis-layer')].every(c => c.hidden)")
        assert page.evaluate("[...document.querySelectorAll('.viewport')].every(v => !v.querySelector('canvas').style.transform)")
        browser.close()
finally:
    server.shutdown()
assert not errors, errors
print('analysis smoke: pass')
