#!/usr/bin/env python3
"""Comparing against another encoder's stream takes two clicks (panel.js):
"Reference encoder" at the top of a version, then a stream from the picker
(the playing clip's streams grouped by encoder, each named by encoder, codec, size, rate and bitrate; a clip without streams says
so and lists other clips' streams, which switch the clip), then "Match
nanocodec to this stream" sets the other version live at the stream's codec,
size, frame rate and bitrate. A reference view hides the live controls. With
the test scene (or a webcam or file) the reference choice is disabled and the
reason shown.

  python3 web/tests/reference-flow.py
"""
import functools
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
REFERENCE = '.view-source [data-view-source="reference"]'
COMPARED = "+(document.querySelector('#session-stats').textContent.match(/(\\d+) frames compared/) || [0, 0])[1]"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1400, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.server_port}/')
        page.wait_for_function(f'{COMPARED} >= 1', timeout=120000)
        # The default clip has no reference streams: the hint says so, and the first
        # stream offered (another clip's) switches the page to that clip.
        page.locator(REFERENCE).nth(1).click()
        page.wait_for_function("document.querySelectorAll('.frame-badge')[1].textContent.includes('PRE-ENCODED')", timeout=60000)
        clip = page.evaluate("document.querySelector('#clip').value")
        assert clip != 'netflix-bar-scene', clip
        # Now on a clip with streams: they come first, grouped by encoder, in plain words.
        groups = page.evaluate("[...document.querySelectorAll('.stream-picker')[1].querySelectorAll('optgroup')].map(g => g.label)")
        assert not groups[0].startswith('Other clip'), groups
        first = page.evaluate("document.querySelectorAll('.stream-picker')[1].querySelector('optgroup option').textContent")
        assert ' fps · ' in first and 'kbps' in first, first
        assert page.locator('.reference-hint').nth(1).is_hidden()
        # The reference view hides the codec picker and the live settings.
        assert page.locator('.codec-picker').nth(1).is_hidden()
        assert page.locator('.options').nth(1).is_hidden()
        # Match: version A becomes live nanocodec at the stream's settings.
        codec = page.evaluate("document.querySelectorAll('.codec-picker')[1].value")
        assert page.locator('.match').nth(1).text_content() == 'Match nanocodec to this stream'
        page.locator('.match').nth(1).click()
        page.wait_for_timeout(500)
        settings = page.evaluate("""[document.querySelectorAll('.resolution')[0].value, document.querySelectorAll('.fps')[0].value,
          document.querySelectorAll('.codec-picker')[0].value, document.querySelectorAll('.target-rate')[0].value]""")
        facts = page.evaluate("Object.fromEntries([...document.querySelectorAll('.reference-facts')[1].querySelectorAll('dt')].map(dt => [dt.textContent, dt.nextElementSibling.textContent]))")
        assert settings[0] == facts['Size'].replace(' × ', 'x'), (settings, facts)
        assert settings[1] == facts['Frame rate'].split()[0], (settings, facts)
        assert settings[2] == codec and settings[3] == facts['Bitrate'].split()[1], (settings, facts, codec)
        # A source with no reference streams: the choice is disabled and the reason shown.
        page.locator('[data-source="scene"]').click()
        assert page.locator(REFERENCE).nth(0).is_disabled()
        assert page.locator('.source-unavailable').nth(0).is_visible()
        assert 'sample clips only' in page.locator('.source-unavailable').nth(0).text_content()
        browser.close()
    server.shutdown()
    assert not errors, errors
    print('reference flow: pass')


if __name__ == '__main__':
    main()
