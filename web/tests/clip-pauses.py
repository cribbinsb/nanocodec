#!/usr/bin/env python3
"""A sample-clip comparison plays evenly: no long pauses between shown frames.

Sample clips are exact sources (web/exact-source.js): a slow machine shows a
slower clip, never skipped frames, and should do so evenly. A heavy clip once
played in stop-and-go bursts: the clip decoder (web/clip-source.js) was fed on a
timer whatever its backlog, queued the rest of the clip, held hundreds of
unclosed frames and stalled for seconds. Headless Chromium runs the default
comparison on CLIP (8 Mbps, 60 fps 720p: the heaviest clip file) for a whole
loop of the clip and more, and after a warm-up requires
no gap between shown frames (both views' results of a tick are in) over both
MAX_GAP_RATIO x the median gap and MIN_PAUSE_MS. The gaps between pictures sent
to the views (the source starving the pipeline shows there first) are printed
for diagnosis; the frames in flight (pipeline.js) absorb short ones.

Run it on few cores, as a typical viewer's machine has, e.g.

  taskset -c 12,13 python3 web/tests/clip-pauses.py
  python3 web/tests/clip-pauses.py --clip mot17-mall-walk --verbose

`--verbose` prints every frame's timings (when it was sent and the gap, encode
and decode per view, key frames, the shown gap) for finding what a pause lines up with.
"""
import argparse
import functools
import http.server
import statistics
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
CLIP = 'netflix-toddler-fountain'
# Shown frames skipped before measuring (start-up: module compile, warm-up, first keys).
WARM_UP_FRAMES = 30
# Shown frames measured after the warm-up: at the default 15 fps, 660 ticks, two
# loops of the 10 s clip (300 ticks), so the loop back is measured too.
MEASURE_FRAMES = 330
# Wall-time limit for the measured part (a loaded 2-core machine shows ~3 frames a second).
MEASURE_TIMEOUT_S = 150
# A pause: a shown-frame gap over MAX_GAP_RATIO median gaps and MIN_PAUSE_MS.
# Measured 2026-11-15 on this PC at load ~60, shown gaps: before the fix up to
# 16.2 s (toddler, 203 ms median, 2 cores) and 5.4 s (mall walk, 111 ms); after
# it at most 883 ms (2.7x, 2 cores) and 464 ms (3.4x, all cores), scheduling noise.
MAX_GAP_RATIO = 4
MIN_PAUSE_MS = 1000
STATS = '#session-stats'

# Per tick: when the page sent its pictures to the capture worker, and when each
# view's decoded result came back (with its stage times and key flag).
HOOK = '''window.__g = { sent: {}, shown: {} };
const Original = Worker; let decoders = 0;
window.Worker = class extends Original {
  constructor(url, ...rest) { super(url, ...rest); this.script = String(url).split('/').pop();
    if (this.script !== 'decode-worker.js') return;
    const view = decoders++ % 2;
    this.addEventListener('message', ({ data: d }) => {
      if (!d || d.type !== 'frame' || d.held) return;
      const tick = Math.round(d.timestamp * 30 / 1000);
      (window.__g.shown[tick] ??= [])[view] = { at: performance.now(), encode: d.encodeMs, decode: d.decodeMs, key: d.key };
    }); }
  postMessage(d, ...rest) {
    if (d && d.type === 'frame' && this.script === 'capture-worker.js') window.__g.sent[Math.round(d.timestamp * 30 / 1000)] = performance.now();
    return super.postMessage(d, ...rest);
  }
};'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def record(port, clip):
    """Each tick's send time and both views' results, from the first shown frame on."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--autoplay-policy=no-user-gesture-required'])
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(HOOK)
        page.goto(f'http://127.0.0.1:{port}/')
        page.wait_for_function(f"document.querySelector('{STATS}').textContent.includes('frames compared')", timeout=120000)
        page.select_option('#clip', clip)
        page.evaluate('window.__g = { sent: {}, shown: {} }')
        goal = WARM_UP_FRAMES + MEASURE_FRAMES
        deadline = time.time() + MEASURE_TIMEOUT_S + 60
        while time.time() < deadline:
            if page.evaluate('Object.values(window.__g.shown).filter(v => v[0] && v[1]).length') >= goal:
                break
            time.sleep(1)
        data = page.evaluate('window.__g')
        session = page.locator(STATS).inner_text()
        browser.close()
    assert not errors, errors
    return data, session


def gaps(times):
    return [b - a for a, b in zip(times, times[1:])]


def check(data, session, verbose):
    shown = {int(t): v for t, v in data['shown'].items() if len(v) == 2 and all(v)}
    ticks = sorted(shown)
    assert len(ticks) >= WARM_UP_FRAMES + MEASURE_FRAMES // 2, f'only {len(ticks)} frames shown: {session}'
    # Results come back in capture order; a tick is shown once both views are in.
    measured = ticks[WARM_UP_FRAMES:]
    shown_at = [max(r['at'] for r in shown[t]) for t in measured]
    sent_at = [data['sent'][str(t)] for t in measured]
    if verbose:
        print('tick  sent_ms  send_gap  enc_A  dec_A  enc_B  dec_B  key  shown_gap')
        for i, t in enumerate(measured):
            a, b = shown[t]
            print(f"{t:4d} {sent_at[i]:9.0f} {sent_at[i] - sent_at[i - 1] if i else 0:8.0f} {a['encode'] or 0:6.1f} {a['decode']:6.1f} "
                  f"{b['encode'] or 0:6.1f} {b['decode']:6.1f}  {'K' if a['key'] or b['key'] else ' '}  {shown_at[i] - shown_at[i - 1] if i else 0:8.0f}")
    summary = {}
    for name, times in (('shown', shown_at), ('sent', sent_at)):
        g = gaps(times)
        median, worst = statistics.median(g), max(g)
        summary[name] = median, worst, measured[g.index(worst) + 1]
        print(f'{name}: {len(g)} gaps, median {median:.0f} ms, p95 {sorted(g)[int(0.95 * len(g))]:.0f} ms, max {worst:.0f} ms at tick {summary[name][2]}')
    print(session)
    median, worst, at = summary['shown']
    assert worst <= max(MAX_GAP_RATIO * median, MIN_PAUSE_MS), \
        f'a {worst:.0f} ms pause before tick {at}: over {MAX_GAP_RATIO}x the median gap ({median:.0f} ms) and {MIN_PAUSE_MS} ms'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--clip', default=CLIP)
    parser.add_argument('--verbose', action='store_true', help="print every frame's timings")
    parser.add_argument('--web', type=Path, default=WEB, help='the web directory to serve (e.g. a copy of another build)')
    args = parser.parse_args()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(args.web)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        check(*record(server.server_port, args.clip), args.verbose)
        print(f'PASS clip pauses: {args.clip} plays without long gaps')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
