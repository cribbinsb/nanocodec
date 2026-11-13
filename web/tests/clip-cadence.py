#!/usr/bin/env python3
"""Every sample clip plays one distinct frame per 30 Hz tick: no repeats.

The page samples sources on a 30 Hz timeline (web/timeline.js). Clips shot at
24 or 25 fps are retimed to 30 fps (tools/web/retime-clips.py, make-mot-clips.py)
so that no frame is captured twice: a repeat judders and is nearly free to code.
Headless Chromium plays the 720p file of each 30 fps clip for a moment; for every
frame it presents (requestVideoFrameCallback) the media time must fall on the
30 fps grid, frame numbers must strictly increase, and consecutive pictures must
differ (a small thumbnail of each, as the page would capture it).

  python3 web/tests/clip-cadence.py
"""
import functools
import http.server
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
TIMELINE_FPS = 30
PLAY_MS = 1500
# Media times within this of a 1/30 s grid point (MP4 timestamps are rounded).
GRID_TOLERANCE_S = 0.002
# Thumbnail size: large enough that two different frames of a static shot differ
# in some sample, small enough to copy quickly. A repeated frame is identical.
THUMB_WIDTH, THUMB_HEIGHT = 160, 90

PLAY = '''async ({ url, playMs, THUMB_WIDTH, THUMB_HEIGHT }) => {
  const video = document.createElement('video'); video.muted = true; video.src = url;
  document.documentElement.append(video);
  await new Promise((resolve, reject) => { video.onloadeddata = resolve; video.onerror = () => reject(new Error(url)); });
  const canvas = new OffscreenCanvas(THUMB_WIDTH, THUMB_HEIGHT), ctx = canvas.getContext('2d', { willReadFrequently: true });
  const frames = [];
  const onFrame = (_, meta) => {
    ctx.drawImage(video, 0, 0, THUMB_WIDTH, THUMB_HEIGHT);
    frames.push({ t: meta.mediaTime, pixels: Array.from(ctx.getImageData(0, 0, THUMB_WIDTH, THUMB_HEIGHT).data) });
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
  await video.play();
  await new Promise(resolve => setTimeout(resolve, playMs));
  video.pause(); video.remove();
  return frames;
}'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def check(name, frames):
    assert len(frames) >= 10, f'{name}: only {len(frames)} frames presented'
    numbers = []
    for frame in frames:
        number = round(frame['t'] * TIMELINE_FPS)
        assert abs(frame['t'] - number / TIMELINE_FPS) < GRID_TOLERANCE_S, f"{name}: media time {frame['t']} off the 30 fps grid"
        numbers.append(number)
    assert all(b > a for a, b in zip(numbers, numbers[1:])), f'{name}: frame numbers {numbers}'
    for a, b in zip(frames, frames[1:]):
        assert a['pixels'] != b['pixels'], f"{name}: frames at {a['t']} and {b['t']} are identical"


def main():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f'http://127.0.0.1:{server.server_port}/README.md')
            clips = page.evaluate("import('/clips.js').then(({ CLIPS }) => CLIPS.filter(c => c.fps === 30).map(c => c.id))")
            for clip in clips:
                check(clip, page.evaluate(PLAY, {'url': f'/clips/{clip}-720p.mp4', 'playMs': PLAY_MS, 'THUMB_WIDTH': THUMB_WIDTH, 'THUMB_HEIGHT': THUMB_HEIGHT}))
            browser.close()
        print(f'PASS clip cadence: {len(clips)} clips at 30 fps, one distinct frame per tick')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
