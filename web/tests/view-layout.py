#!/usr/bin/env python3
"""The views' layout control and their shared zoom and pan, with and without analysis.

Headless Chromium, two live views (`?views=live`), CPU display path. For each
layout (side by side, stacked, A large, B large), analysis off and on:

- the views' container has the layout's class, and the hidden picture of
  "A large" / "B large" is the other view's;
- scrolling on the shown view zooms both views together (one CSS transform,
  unsmoothed pixels), each shown view gets the "N× · Reset" chip, a drag pans
  both, and each of the chip, a double-click and Escape returns both to the
  whole picture (no transform, chips hidden);
- the debug report (no streams) records the layout and the zoom.

Both live views (H.264 in A, AV1 in B) show the same controls in the same places: a
control that does not apply to a view (spatial layers on H.264, the shown layer with
one layer, the quantizer slider under a bitrate target) stays in place, disabled, with
a tooltip, and switching the rate control or the codec moves nothing.

Also: the layout is remembered across a reload (localStorage), a narrow
window with nothing remembered opens stacked, and a page whose storage throws
still works side by side.

  python3 web/tests/view-layout.py
"""
import functools
import http.server
import json
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
LAYOUTS = ['side', 'stacked', 'expand-A', 'expand-B']
COMPARED = "parseInt(document.querySelector('#session-stats').textContent) || 0"
TRANSFORMS = "[...document.querySelectorAll('.viewport canvas:not(.analysis-layer)')].map(c => c.style.transform)"
AT_FIT = ("[...document.querySelectorAll('.zoom-chip')].every(c => c.hidden)"
          " && [...document.querySelectorAll('.viewport canvas:not(.analysis-layer)')].every(c => !c.style.transform)")

# Each view's controls in its picture options and Advanced general settings: class, offset
# from the view's top-left, and whether it is enabled.
CONTROLS = """[...document.querySelectorAll('.version')].map(view => {
  const origin = view.getBoundingClientRect();
  return [...view.querySelectorAll('.picture-options > *, .advanced-general > *, .advanced-general .peak-options > *')].map(c => {
    const box = c.getBoundingClientRect(), input = c.querySelector('select, input');
    return [c.className.replace(' inapplicable', ''), Math.round(box.left - origin.left), Math.round(box.top - origin.top), Math.round(box.width), input ? !input.disabled : null, c.title];
  });
})"""


def same_controls(page) -> dict:
    """The two live views' controls match in kind and place, whatever applies to each."""
    for summary in page.locator('details.advanced > summary').all():
        summary.click()
    a, b = page.evaluate(CONTROLS)
    assert [c[:4] for c in a] == [c[:4] for c in b], (a, b)
    layers = {c[0]: c for c in a}['compact-control live-only spatial-layers-control']
    assert layers[4] is False and 'no spatial layers' in layers[5], layers  # A is H.264
    assert {c[0]: c for c in b}['compact-control live-only spatial-layers-control'][4] is True  # B is AV1
    # A rate-control change moves nothing: the slider and the bitrate swap enabled, in place.
    before = page.evaluate(CONTROLS)
    page.locator('.rate-mode').nth(0).select_option('qp')
    after = page.evaluate(CONTROLS)
    assert [c[:4] for c in after[0]] == [c[:4] for c in before[0]], (before[0], after[0])
    enabled = {c[0]: c[4] for c in after[0]}
    assert enabled['quality-option live-only'] is True and enabled['compact-control live-only target-option'] is False, enabled
    page.locator('.rate-mode').nth(0).select_option('cbr')
    # A codec change moves nothing either (H.264 -> VP9 enables its spatial layers in place).
    page.locator('.codec-picker').nth(0).select_option('vp9')
    again = page.evaluate(CONTROLS)
    assert [c[:4] for c in again[0]] == [c[:4] for c in before[0]], (before[0], again[0])
    page.locator('.codec-picker').nth(0).select_option('h264')
    for summary in page.locator('details.advanced > summary').all():
        summary.click()
    return {'controls': len(a)}


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def json_block(path: Path) -> dict:
    text = path.read_text()
    start = text.rfind('```json') + len('```json')
    return json.loads(text[start:text.index('\n```', start)])


def zoom_in(page, view: int):
    page.locator('.viewport').nth(view).scroll_into_view_if_needed()
    box = page.locator('.viewport').nth(view).bounding_box()
    page.mouse.move(box['x'] + box['width'] * 0.4, box['y'] + box['height'] * 0.4)
    for _ in range(4):
        page.mouse.wheel(0, -100)
    page.wait_for_function(f"document.querySelectorAll('.viewport canvas:not(.analysis-layer)')[{view}].style.transform.includes('scale(2.44')", timeout=5000)
    return box


def check_layout(page, layout: str, analysis: bool, scratch: Path):
    page.locator('#layout').select_option(layout)
    assert page.evaluate(f"document.querySelector('#versions').classList.contains('layout-{layout}')")
    shown = [page.locator('.viewport').nth(i).is_visible() for i in (0, 1)]
    assert shown == {'expand-A': [True, False], 'expand-B': [False, True]}.get(layout, [True, True]), (layout, shown)
    view = shown.index(True)
    box = zoom_in(page, view)
    # Shown views of one size share one transform (a hidden picture is placed once shown).
    transforms = [t for t, visible in zip(page.evaluate(TRANSFORMS), shown) if visible]
    assert all(t == transforms[0] for t in transforms), (layout, analysis, transforms)
    rendering = page.evaluate("[...document.querySelectorAll('.viewport canvas:not(.analysis-layer)')].map(c => c.style.imageRendering)")
    assert all(r == 'pixelated' for r, visible in zip(rendering, shown) if visible), rendering
    chips = page.evaluate("[...document.querySelectorAll('.zoom-chip')].map(c => !c.hidden && c.textContent)")
    assert all(chip == '2.4× · Reset' for chip in chips), chips
    # A drag pans both views alike.
    page.mouse.move(box['x'] + box['width'] * 0.5, box['y'] + box['height'] * 0.5)
    page.mouse.down()
    page.mouse.move(box['x'] + box['width'] * 0.3, box['y'] + box['height'] * 0.6, steps=5)
    page.mouse.up()
    panned = [t for t, visible in zip(page.evaluate(TRANSFORMS), shown) if visible]
    assert all(t == panned[0] for t in panned) and panned[0] != transforms[0], (layout, analysis, transforms, panned)
    # The report records the layout and the zoom.
    page.locator('details.more > summary').click()
    page.locator('#debug-report').click()
    page.locator('#debug-streams').set_checked(False)
    with page.expect_download() as download:
        page.locator('#debug-download').click()
    path = scratch / f'{layout}-{analysis}.md'
    download.value.save_as(path)
    state = json_block(path)['state']
    assert state['layout'] == layout and abs(state['zoom']['scale'] - 1.25 ** 4) < 1e-9, state
    assert state['analysis']['enabled'] is analysis
    assert f'- Layout {layout} · zoom 2.44140625×' in path.read_text()
    # Every way back to the whole picture.
    for reset in ('chip', 'double-click', 'Escape'):
        if reset == 'chip':
            page.locator('.zoom-chip').nth(view).click()
        elif reset == 'double-click':
            page.locator('.viewport').nth(view).dblclick()
        else:
            page.locator('.viewport').nth(view).focus()
            page.keyboard.press('Escape')
        page.wait_for_function(AT_FIT, timeout=5000)
        if reset != 'Escape':
            zoom_in(page, view)


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/?views=live&display=cpu'
    errors = []
    try:
        with tempfile.TemporaryDirectory(prefix="nanocodec-web-test-") as scratch, sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True, viewport={'width': 1400, 'height': 1800})
            page = context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            assert page.locator('#layout').input_value() == 'side'
            page.wait_for_function(f'{COMPARED} >= 5', timeout=120000)
            controls = same_controls(page)
            page.wait_for_function(f'{COMPARED} >= 5', timeout=120000)
            for analysis in (False, True):
                if analysis:
                    page.locator('#analysis-toggle').click()
                    page.wait_for_function("[...document.querySelectorAll('.analysis-stats')].every(s => s.querySelector('table'))", timeout=60000)
                for layout in LAYOUTS:
                    check_layout(page, layout, analysis, Path(scratch))
            # Remembered per viewer: a reload opens on the last choice.
            page.locator('#layout').select_option('stacked')
            page.reload()
            assert page.locator('#layout').input_value() == 'stacked'
            context.close()
            # A narrow window with nothing remembered opens stacked.
            narrow = browser.new_page(viewport={'width': 400, 'height': 860})
            narrow.goto(url)
            assert narrow.locator('#layout').input_value() == 'stacked'
            narrow.close()
            # Storage that throws: the page works, side by side, and does not remember.
            blocked = browser.new_page(viewport={'width': 1400, 'height': 1000})
            blocked.on('pageerror', lambda error: errors.append(str(error)))
            blocked.add_init_script("Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('denied', 'SecurityError'); } });")
            blocked.goto(url)
            blocked.locator('#layout').select_option('expand-B')
            blocked.wait_for_function(f'{COMPARED} >= 3', timeout=120000)
            assert blocked.evaluate("document.querySelector('#versions').classList.contains('layout-expand-B')")
            browser.close()
    finally:
        server.shutdown()
    assert not errors, errors
    print(json.dumps({'status': 'pass', **controls, 'layouts': LAYOUTS, 'analysis': [False, True]}))


if __name__ == '__main__':
    main()
