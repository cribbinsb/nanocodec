#!/usr/bin/env python3
"""Where the demo page's time goes, per compared frame.

Runs a comparison in headless Chromium and, after a warm-up, measures two
windows:

1. untraced (tracing slows the page): the delivered rate (frames the faster
   view showed per second) and the clip's speed against real time (exact
   sources slow time instead of skipping frames);
2. traced: each thread's CPU time per compared frame (its top-level tasks),
   worker threads named by the script they run, the total CPU per frame, the
   part of it outside the encode and decode workers (capture, clip decoding,
   the page, drawing and compositing), and the main thread's time by event.

The workers' own per-frame stage times (encode, decode, the decode stage with
scoring and display conversion) come from their messages, per view; they are
wall times, so a loaded machine inflates them where thread CPU time is not.

  python3 web/tests/pipeline-profile.py --preset default
  python3 web/tests/pipeline-profile.py --preset 720p30 --runs 3
  python3 web/tests/pipeline-profile.py --gpu     # hardware WebGL display
  taskset -c 4-7 python3 web/tests/pipeline-profile.py   # a 4-core machine

Thread CPU time leaves out waiting for a core, but a loaded machine still
inflates it (shared SMT cores, lower clocks): measured on 2026-11-14 at load
12-16, the same build gave 60 and 116 ms per 720p frame. Compare builds only
on a quiet machine, interleaving runs (--web serves another copy of the web
directory).
"""
import argparse
import collections
import functools
import http.server
import json
import statistics
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
WARM_S, MEASURE_S, TRACE_S = 6, 10, 6
# Both views' (width, height, fps) and codecs A, B; None: the page as it opens.
PRESETS = {
    'default': None,
    '720p30': ((1280, 720, 30), 'h264', 'av1'),
}
# Hardware WebGL through ANGLE on Vulkan (headless Chromium otherwise has only
# SwiftShader, flags it a major performance caveat, and the page draws on the CPU).
GPU_ARGS = ['--use-gl=angle', '--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist']
TRACE_CATEGORIES = ['devtools.timeline', 'disabled-by-default-devtools.timeline', 'toplevel']
# The trace event of one top-level task on a thread.
TASK_EVENT = 'ThreadControllerImpl::RunTask'
STATS = '#session-stats'
# The workers that run the codecs (with their own conversions, copies and scoring).
CODEC_WORKERS = {'encode-worker.js', 'decode-worker.js'}
# Main-thread events listed, and the smallest thread row shown (ms per frame).
MAIN_EVENTS_SHOWN, MIN_THREAD_MS = 30, 0.1

HOOK = '''window.__p = [];
const OriginalWorker = Worker; let decoders = 0;
window.Worker = class extends OriginalWorker {
  constructor(url, ...rest) { super(url, ...rest);
    if (String(url).endsWith('decode-worker.js')) { const view = decoders++ % 2;
      this.addEventListener('message', e => { const d = e.data;
        if (d && d.type === 'frame' && !d.held) window.__p.push({ view, t: d.timestamp, format: d.format,
          encode: d.encodeMs, decode: d.decodeMs, stage: d.decodeStageMs }); }); } }
};'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def configure(page, preset):
    """Apply the preset's settings; False for the page's own defaults."""
    settings = PRESETS[preset]
    if settings is None:
        return False
    (width, height, fps), *codecs = settings
    for picker, codec in zip(page.locator('.codec-picker').all(), codecs):
        picker.select_option(codec)
        picker.dispatch_event('change')
    for selector, value in (('.resolution', f'{width}x{height}'), ('.fps', str(fps))):
        for picker in page.locator(selector).all():
            picker.select_option(value)
            picker.dispatch_event('change')
    return True


def window(page, seconds):
    """The results (both views) that arrived over `seconds` of wall time, and that time."""
    first = page.evaluate('window.__p.length')
    start = time.time()
    time.sleep(seconds)
    results = page.evaluate(f'window.__p.slice({first})')
    return results, time.time() - start


def compared(results):
    """Frames compared: results of the view with the most (the faster view)."""
    return max(collections.Counter(r['view'] for r in results).values(), default=0)


def threads(trace, frames):
    """CPU ms per compared frame by thread (thread time, so waiting for a core
    does not count), and the main thread's ms by event."""
    events = trace['traceEvents']
    names = {(e['pid'], e['tid']): e['args']['name'] for e in events if e.get('ph') == 'M' and e.get('name') == 'thread_name'}
    main = {key for key, name in names.items() if name == 'CrRendererMain'}
    busy, by_event = collections.Counter(), collections.Counter()
    scripts = collections.defaultdict(collections.Counter)
    for e in events:
        if e.get('ph') != 'X':
            continue
        key, ms = (e.get('pid'), e.get('tid')), e.get('tdur', e.get('dur', 0)) / 1000
        if e['name'] == TASK_EVENT:
            busy[key] += ms
            continue
        data = e.get('args', {}).get('data', {})
        if e['name'] == 'FunctionCall':
            scripts[key][data.get('url', '').split('/').pop()] += ms
        if key in main:
            label = f"{data.get('functionName') or '(anonymous)'} {data.get('url', '').split('/').pop()}" if e['name'] == 'FunctionCall' else e['name']
            by_event[label] += ms
    rows = []
    for key, ms in busy.most_common():
        name = names.get(key, '?')
        if name == 'DedicatedWorker thread' and scripts[key]:
            name = scripts[key].most_common(1)[0][0]
        rows.append((name, ms / frames))
    return rows, [(label, ms / frames) for label, ms in by_event.most_common(MAIN_EVENTS_SHOWN)]


def median(results, view, key):
    values = [r[key] for r in results if r['view'] == view and r.get(key) is not None]
    return round(statistics.median(values), 1) if values else None


def run(port, preset, gpu=False, trace_out=None):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--autoplay-policy=no-user-gesture-required', *(GPU_ARGS if gpu else [])])
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(HOOK)
        page.goto(f'http://127.0.0.1:{port}/')
        running = f"document.querySelector('{STATS}').textContent.includes('frames compared')"
        page.wait_for_function(running, timeout=120000)
        if configure(page, preset):
            page.wait_for_timeout(500)
            page.wait_for_function(running, timeout=120000)
        time.sleep(WARM_S)
        plain, plain_s = window(page, MEASURE_S)
        session = page.locator(STATS).inner_text()
        browser.start_tracing(page=page, categories=TRACE_CATEGORIES)
        traced, _ = window(page, TRACE_S)
        raw = browser.stop_tracing()
        if trace_out:
            trace_out.write_bytes(raw)
        trace = json.loads(raw)
        browser.close()
    assert not errors, errors
    frames = compared(plain)
    clip_s = (max(r['t'] for r in plain) - min(r['t'] for r in plain)) / 1000 + 1 / max(frames / plain_s, 1)
    rows, main = threads(trace, max(compared(traced), 1))
    total = sum(ms for _, ms in rows)
    codec_workers = sum(ms for name, ms in rows if name in CODEC_WORKERS)
    return {
        'preset': preset, 'display': sorted({r['format'] for r in plain}), 'delivered_fps': round(frames / plain_s, 1),
        'clip_speed': round(clip_s / plain_s, 2), 'session': session,
        'stages_ms': {f'view {v}': {k: median(plain, v, k) for k in ('encode', 'decode', 'stage')} for v in (0, 1)},
        'cpu_ms_per_frame': round(total, 1), 'outside_codec_workers_cpu_ms_per_frame': round(total - codec_workers, 1),
        'threads_ms_per_frame': [(n, round(ms, 1)) for n, ms in rows if ms >= MIN_THREAD_MS],
        'main_ms_per_frame': [(n, round(ms, 2)) for n, ms in main],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--preset', choices=sorted(PRESETS), default='default')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', help='draw with the hardware GPU (WebGL), as most visitors do')
    parser.add_argument('--trace', type=Path, help='also save the Chrome trace (JSON) here')
    parser.add_argument('--web', type=Path, default=WEB, help='the web directory to serve (e.g. a copy of another build)')
    args = parser.parse_args()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(args.web)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for _ in range(args.runs):
            print(json.dumps(run(server.server_port, args.preset, args.gpu, args.trace)), flush=True)
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
