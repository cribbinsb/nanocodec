#!/usr/bin/env python3
"""Our browser module against the browser's own decoders (WebCodecs), in the same browser.

    taskset -c 1 python3 tools/web/webcodecs_yardstick.py 5 out.json --pkg PKG/relaxed \\
        --streams DIR [DIR ...] [--channel chrome|chromium] [--hardware] [--chrome-arg ARG ...]

Chrome (Playwright, headless) loads tools/web/webcodecs-bench.html from a local server, one
fresh page per run; the page decodes the stream twice (a fresh decoder each pass; the first
also warms the JavaScript engine's optimising compiler) and reports the faster pass, for:

- ours-chrome: our module's BrowserDecoder.decode over every packet (module load and fetch
  untimed);
- webcodecs-sw: VideoDecoder with hardwareAcceleration 'prefer-software' (Chrome's own native
  software decoders: FFmpeg for H.264, libvpx for VP9, dav1d for AV1, with their assembly);
- webcodecs-hw (--hardware): 'prefer-hardware', which fails rather than falls back to software,
  so a run that decodes was decoded by the platform decoder.

The WebCodecs span is from the first decode() to flush() resolving, every chunk queued at once
and each output frame closed at once (never read back). The driver's CPU affinity (taskset)
binds the browser and every thread it starts, so all three share the one core. Rounds are
interleaved (the order reversed every other round). A codec a mode cannot decode is probed
once (isConfigSupported, then a decode) and recorded under "unsupported" with the reason.
Also written beside the record: chrome://gpu's text (out.gpu.txt), the browser version and
flags. Record format as tools/dev/decoder_speed.py: {"res": {"stream|codec|decoder":
{"seconds": [...], "frames": n}}, ...}.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError, sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dev"))
from decoder_speed import codec_of  # noqa: E402

OURS, SOFTWARE, HARDWARE = "ours-chrome", "webcodecs-sw", "webcodecs-hw"
#: Formats the WebCodecs codec registry has no string for: never offered to VideoDecoder.
NO_WEBCODECS = {"vvc", "av2"}
PAGE_TIMEOUT_MS = 600_000
#: chrome://gpu's text, one line per leaf element: the page renders into nested shadow roots.
GPU_TEXT = """() => {
  const out = [];
  const walk = (node) => {
    if (node.shadowRoot) walk(node.shadowRoot);
    for (const child of node.children || []) walk(child);
    if (node.nodeType === 1 && !node.children.length && node.textContent.trim()) out.push(node.textContent.trim());
  };
  walk(document.documentElement);
  return out.join('\\n');
}"""


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files, cross-origin isolated (performance.now() at its finest resolution)."""

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *_):
        pass


def serve(root: Path) -> str:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def site(pkg: Path, streams: list[Path]) -> Path:
    """The served directory: the page, packets.mjs, pkg/ and streams/ (symlinks)."""
    root = Path(tempfile.mkdtemp(prefix="nanocodec-webcodecs-"))
    (root / "index.html").symlink_to(HERE / "webcodecs-bench.html")
    (root / "packets.mjs").symlink_to(HERE / "packets.mjs")
    (root / "pkg").symlink_to(pkg)
    (root / "streams").mkdir()
    for stream in streams:
        (root / "streams" / stream.name).symlink_to(stream)
    return root


def run_page(browser, base: str, decoder: str, codec: str, name: str) -> dict:
    page = browser.new_page()
    try:
        page.goto(f"{base}/index.html")
        page.wait_for_function("window.benchReady === true")
        url = f"streams/{name}"
        if decoder == OURS:
            return page.evaluate("([c, u]) => oursDecode(c, u)", [codec, url])
        return page.evaluate("([c, u, h]) => webcodecsDecode(c, u, h)", [codec, url, decoder == HARDWARE])
    finally:
        page.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rounds", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pkg", type=Path, required=True, help="our browser module (tools/web/build.sh)")
    parser.add_argument("--streams", type=Path, nargs="+", required=True)
    parser.add_argument("--channel", default="chrome", help="Playwright channel: chrome (Google Chrome) or chromium")
    parser.add_argument("--hardware", action="store_true", help="also time webcodecs-hw")
    parser.add_argument("--chrome-arg", action="append", default=[], help="an extra browser flag")
    parser.add_argument("--headed", action="store_true",
                        help="a browser window, for VA-API decode: headless Chrome cannot create the "
                             "image processor its VA-API output needs. Give it a private headless "
                             "compositor (WAYLAND_DISPLAY of `gnome-shell --headless --wayland`), never "
                             "the desktop's display")
    args = parser.parse_args()
    streams = sorted((p.absolute() for d in args.streams for p in d.iterdir()
                      if p.is_file() and not p.name.startswith("pan_")),
                     key=lambda p: p.name)
    base = serve(site(args.pkg.resolve(), streams))
    decoders = [OURS, SOFTWARE] + ([HARDWARE] if args.hardware else [])
    record: dict = {"channel": args.channel, "flags": args.chrome_arg, "headed": args.headed, "res": {}, "unsupported": {},
                    "codec_strings": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, args=args.chrome_arg, headless=not args.headed,
                                    timeout=PAGE_TIMEOUT_MS)
        browser.contexts  # noqa: B018 (the default context is made per page)
        record["browser"] = browser.version
        gpu = browser.new_page()
        gpu.goto("chrome://gpu")
        gpu.wait_for_timeout(2000)
        args.out.with_suffix(".gpu.txt").write_text(gpu.evaluate(GPU_TEXT))
        gpu.close()
        jobs = []
        for stream in streams:
            codec = codec_of(stream.name)
            for decoder in decoders:
                if decoder != OURS and codec in NO_WEBCODECS:
                    record["unsupported"][f"{codec}|{decoder}"] = "no WebCodecs codec string for this format"
                    continue
                jobs.append((stream.name, codec, decoder))
        for index in range(args.rounds):
            for name, codec, decoder in jobs if index % 2 == 0 else jobs[::-1]:
                if f"{codec}|{decoder}" in record["unsupported"] or f"{name}|{decoder}" in record["unsupported"]:
                    continue
                try:
                    result = run_page(browser, base, decoder, codec, name)
                except PlaywrightError as error:  # a decoder that throws on this stream
                    result = {"supported": False, "reason": f"error: {error.message.splitlines()[0]}"}
                if "codec" in result:
                    record["codec_strings"][f"{name}|{decoder}"] = result["codec"]
                if not result["supported"]:
                    reason = result.get("reason", "")
                    # a format the mode lacks is unsupported for every stream; a decode error, for this one
                    scope = f"{codec}|{decoder}" if reason.startswith(("isConfigSupported", "no ")) else f"{name}|{decoder}"
                    record["unsupported"][scope] = reason
                    print(f"{scope}: unsupported ({reason})", file=sys.stderr, flush=True)
                    continue
                entry = record["res"].setdefault(f"{name}|{codec}|{decoder}", {"seconds": [], "frames": result["frames"]})
                entry["seconds"].append(result["ms"] / 1000)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(record, indent=1) + "\n")
            print(f"round {index} done", file=sys.stderr, flush=True)
        browser.close()


if __name__ == "__main__":
    main()
