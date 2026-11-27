#!/usr/bin/env python3
"""The synthetic test scenes (web/scene.js) are exact, distinct and selectable.

In headless Chromium, each scene in SCENES is drawn at 1280x720 for TICKS source
ticks, twice: the two drawings must be byte-identical (an exact source), each
scene's frames must differ from every other scene's as a set, and a moving scene's from
tick to tick. A drawing must take under MAX_DRAW_MS once the scene is set up (the page draws one per tick).
The waving hand's room is the CC0 photograph, decoded to the committed pixel hash
and placed unchanged at its own size; the hand holds still frame-identical, leaves the bare room behind, and covers
about 15-30 % of the picture mid-wave. Then the page opened with `?views=live&scene=ID` runs on that scene (the source
line names it), the scene picker switches to another, and the debug report names
the scene and its parameters. Last, at a desktop and a phone width, each source shows only its
own picker (none for the webcam or a file), keeps the hidden picker's choice, and the views
do not move.

  python3 web/tests/test-scenes.py
"""
import functools
import http.server
import json
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
TICKS = (0, 1, 45, 121, 240)
# A 30 Hz page draws one scene per tick (33 ms) beside the codecs. A scene draws in about
# 5 ms at 720p on a quiet build box (printed); the bound is the whole tick, as draws are
# timed under other jobs' load (37 ms once on a busy shared box).
MAX_DRAW_MS = 33
LINKED, PICKED = 'zone-plate', 'cuts-fades'
DRAW = '''async ticks => {
  const { SCENES, drawScene, prepareScene } = await import('./scene.js');
  for (const { id } of SCENES) await prepareScene(id);
  const canvas = document.createElement('canvas'); canvas.width = 1280; canvas.height = 720;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const digest = async () => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', ctx.getImageData(0, 0, 1280, 720).data)))
    .map(b => b.toString(16).padStart(2, '0')).join('');
  const out = {};
  for (const { id } of SCENES) {
    out[id] = { first: [], second: [], ms: 0 };
    for (const pass of ['first', 'second']) for (const tick of ticks) {
      const start = performance.now();
      drawScene(ctx, 1280, 720, tick / 30, id);
      if (pass === 'second') out[id].ms = Math.max(out[id].ms, performance.now() - start);
      out[id][pass].push(await digest());
    }
  }
  return out;
}'''

# The waving hand (scene.js HAND, 10 s cycle): still from 5 to 7.5 s, out of frame from 8.5 to 9.5 s.
HAND_HOLD_TICKS, HAND_OUT_TICKS, HAND_WAVE_TICK = (155, 190, 220), (258, 270, 282), 37
# The decoded RGBA pixels of web/scenes/room-photo-1280x720.png (tools/web/make-scene-photo.py).
PHOTO_PIXELS_SHA256 = 'ba53996367c221e999c96a4bd12e08469a1121e54517aac78c91147f0bf54ff4'
# The share of the picture the hand covers mid-wave (the owner asked for about 15-30 %).
HAND_COVER = (.12, .35)
PHASES = '''async ([hold, out, wave]) => {
  const { SCENES, drawScene, prepareScene } = await import('./scene.js');
  await prepareScene('waving-hand');
  const hand = SCENES.find(scene => scene.id === 'waving-hand');
  const canvas = document.createElement('canvas'); canvas.width = 1280; canvas.height = 720;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const hex = async data => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), b => b.toString(16).padStart(2, '0')).join('');
  const pixels = tick => { drawScene(ctx, 1280, 720, tick / 30, 'waving-hand'); return ctx.getImageData(0, 0, 1280, 720).data; };
  const key = data => { let h = 0; for (let i = 0; i < data.length; i += 7) h = (Math.imul(h, 31) + data[i]) | 0; return h; };
  const room = pixels(out[0]), moving = pixels(wave);
  let covered = 0;
  for (let i = 0; i < room.length; i += 4) if (room[i] !== moving[i] || room[i + 1] !== moving[i + 1] || room[i + 2] !== moving[i + 2]) covered++;
  return { hold: hold.map(t => key(pixels(t))), out: out.map(t => key(pixels(t))), room: key(room), cover: covered / (1280 * 720),
    background: hand.background(), roomSha256: await hex(room), photoSha256: hand.params.photoPixelsSha256, version: hand.params.version };
}'''

# Desktop and phone widths for the source bar's layout check.
VIEWPORT_WIDTHS = (1400, 400)
VIEWS_TOP = "Math.round(document.querySelector('#versions').getBoundingClientRect().top)"
# The pickers' box against the source bar it sits in.
PICKER_BOX = ("(r => [Math.round(r.top - document.querySelector('.source-bar').getBoundingClientRect().top), Math.round(r.width),"
              " Math.round(r.height)])(document.querySelector('.clip-choice').getBoundingClientRect())")


def source_pickers(page, width: int) -> int:
    """Each source shows only its own picker (the clip's or the scene's, none for the webcam or a
    file), the hidden choice is kept, and the pickers' box keeps its place and size. At a
    desktop width the views do not move either (at a phone width the summary sentence above
    them wraps by its own length, which differs between sources); returns their tops."""
    page.set_viewport_size({'width': width, 'height': 1000})
    clip = page.locator('#clip').input_value()
    tops, boxes = {}, {}

    def choose_file():
        with page.expect_file_chooser() as chooser:
            page.locator('[data-source="file"]').click()
        chooser.value.set_files(str(WEB / 'clips/xiph-johnny-360p.mp4'))
    for source, choose in (('scene', lambda: page.locator('[data-source="scene"]').click()),
                           ('camera', lambda: page.locator('[data-source="camera"]').click()),
                           ('file', choose_file), ('clip', lambda: page.locator('[data-source="clip"]').click())):
        choose()
        assert page.locator('#clip').is_visible() is (source == 'clip'), (source, 'clip picker')
        assert page.locator('#scene').is_visible() is (source == 'scene'), (source, 'scene picker')
        tops[source], boxes[source] = page.evaluate(VIEWS_TOP), tuple(page.evaluate(PICKER_BOX))
    assert page.locator('#scene').input_value() == PICKED and page.locator('#clip').input_value() == clip, 'a hidden picker lost its choice'
    assert len(set(boxes.values())) == 1, (width, boxes)
    if width == VIEWPORT_WIDTHS[0]:
        assert len(set(tops.values())) == 1, (width, tops)
    return tops


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    try:
        with tempfile.TemporaryDirectory(prefix="nanocodec-web-test-") as scratch, sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
            page = browser.new_page(accept_downloads=True)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/?views=live&scene={LINKED}')
            drawn = page.evaluate(DRAW, list(TICKS))
            for scene, d in drawn.items():
                assert d['first'] == d['second'], f'{scene}: two drawings of the same tick differ'
                assert d['ms'] < MAX_DRAW_MS, (scene, d['ms'])
                assert len(set(d['first'])) > 1, f'{scene}: nothing changes between ticks'
            # Scenes may share a picture (the cuts scene opens on the study): none shares them all.
            pictures = {scene: frozenset(d['first']) for scene, d in drawn.items()}
            assert len(set(pictures.values())) == len(pictures), 'two scenes drew the same pictures'
            # The waving hand holds still, then leaves the room alone, each phase frame-identical.
            phases = page.evaluate(PHASES, [list(HAND_HOLD_TICKS), list(HAND_OUT_TICKS), HAND_WAVE_TICK])
            assert len(set(phases['hold'])) == 1, 'the held hand moved'
            assert len(set(phases['out'])) == 1, 'the empty room changed'
            assert phases['hold'][0] != phases['out'][0], 'the hand never left'
            assert phases['out'][0] == phases['room'], 'the hand left something behind'
            assert HAND_COVER[0] < phases['cover'] < HAND_COVER[1], phases['cover']
            # The room is the photograph, decoded and placed bit-exactly: at its own size the
            # empty frame is the committed picture.
            assert phases['background'] == 'photo', phases['background']
            assert phases['roomSha256'] == phases['photoSha256'] == PHOTO_PIXELS_SHA256 and phases['version'] == 2, phases
            # The linked scene runs, and the picker switches scenes.
            page.wait_for_function("[...document.querySelectorAll('.frame-badge')].every(b => b.dataset.tick !== undefined)", timeout=120000)
            assert page.locator('[data-source="scene"]').get_attribute('aria-pressed') == 'true'
            assert page.locator('#source-name').inner_text().startswith('Moving zone plate')
            assert page.locator('#scene').is_visible()
            page.locator('#scene').select_option(PICKED)
            page.wait_for_function("document.querySelector('#source-name').textContent.startsWith('Cuts, flashes and fades')", timeout=30000)
            page.wait_for_function("[...document.querySelectorAll('.frame-badge')].every(b => +(b.dataset.tick ?? -1) > 10)", timeout=120000)
            page.locator('#pause').click()
            page.wait_for_function("document.querySelector('#runtime-label').textContent === 'Paused'", timeout=30000)
            page.locator('details.more > summary').click()
            page.locator('#debug-report').click()
            with page.expect_download() as download:
                page.locator('#debug-download').click()
            path = Path(scratch) / 'scene.md'
            download.value.save_as(path)
            text = path.read_text()
            start = text.rfind('```json') + len('```json')
            source = json.loads(text[start:text.index('\n```', start)])['source']
            assert source['kind'] == 'scene' and source['scene'] == PICKED and source['sceneParams']['shotSeconds'] == 2, source
            layout = {width: source_pickers(page, width) for width in VIEWPORT_WIDTHS}
            browser.close()
    finally:
        server.shutdown()
    assert not errors, errors
    print(json.dumps({'status': 'pass', 'scenes': len(drawn), 'hand_cover': round(phases['cover'], 3), 'views_top': layout, 'max_draw_ms': {k: round(v['ms'], 1) for k, v in drawn.items()}}))


if __name__ == '__main__':
    main()
