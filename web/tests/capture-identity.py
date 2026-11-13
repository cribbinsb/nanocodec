#!/usr/bin/env python3
"""The capture worker (web/capture-worker.js) gives exactly the pixels of the
page's former main-thread capture (letterboxed drawImage of the video element
on a software canvas, then getImageData), so moving capture off the page
changes no stream.

Headless Chromium seeks every sample clip variant (web/clips/*-360p.mp4 and
*-720p.mp4) to a few times and compares both captures at the variant's own
size (the worker's VideoFrame.copyTo path) and at the demo's scaled sizes that
use the variant (the worker's canvas path), with software GL and, if
`--gpu vulkan`, ANGLE on the host GPU. Every byte must match.

  python3 web/tests/capture-identity.py [--gpu swiftshader,vulkan] [--clips N]
"""
import argparse
import functools
import http.server
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
GPU_FLAGS = {"swiftshader": [], "vulkan": ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist"]}
# Each variant at its own size, and the demo's other sizes it is scaled to (web/clips.js clipUrl).
SIZES = {"360p": [[640, 360], [384, 216]], "720p": [[1280, 720], [854, 480]]}
TIMES_S = [0.5, 3.3]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", default="swiftshader")
    parser.add_argument("--clips", type=int, help="only the first N clips")
    args = parser.parse_args()
    clips = sorted(WEB.glob("clips/*-360p.mp4")) + sorted(WEB.glob("clips/*-720p.mp4"))
    if args.clips:
        clips = [c for c in clips if c.name.rsplit("-", 1)[0] in sorted({c.name.rsplit("-", 1)[0] for c in clips})[:args.clips]]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    failures = []
    try:
        with sync_playwright() as p:
            for gpu in args.gpu.split(","):
                browser = p.chromium.launch(headless=True, args=GPU_FLAGS[gpu])
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/tests/capture-identity.html")
                page.wait_for_function("window.ready")
                for clip in clips:
                    sizes = SIZES[clip.stem.rsplit("-", 1)[1]]
                    results = page.evaluate("o => window.compare(o)", {"url": f"../clips/{clip.name}", "sizes": sizes, "times": TIMES_S})
                    bad = [r for r in results if r["count"] or not r["length"]]
                    print(f"{gpu:11} {clip.name:40} {'differs ' + str(bad) if bad else 'identical'}")
                    failures += [(gpu, clip.name, r) for r in bad]
                assert not errors, errors
                browser.close()
    finally:
        server.shutdown()
    assert not failures, failures
    print(f"capture identical: {len(clips)} clip variants, {args.gpu}")


if __name__ == "__main__":
    main()
