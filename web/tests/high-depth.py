#!/usr/bin/env python3
"""Play deep fixtures through the demo's decode worker and CPU/WebGL renderers."""
import argparse
import functools
import http.server
import json
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", type=Path, required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument(
        "--expected", type=Path, required=True, help="high-depth.mjs output"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    web = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    expected = json.loads(args.expected.read_text())["streams"]

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def translate_path(self, path):
            path = path.split("?", 1)[0]
            if path.startswith("/pkg/"):
                return str(args.pkg.resolve() / path.removeprefix("/pkg/"))
            if path.startswith("/test-streams/"):
                return str(args.streams.resolve() / Path(path).name)
            if path == "/test-packets.mjs":
                return str(web.parent / "tools/web/packets.mjs")
            return super().translate_path(path)

        def do_GET(self):
            if self.path == "/high-depth-test":
                body = b'<!doctype html><meta charset="utf-8"><title>10-bit playback</title>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(web))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    reports = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, args=["--no-sandbox", "--enable-unsafe-swiftshader"]
            )
            page = browser.new_page(viewport={"width": 1320, "height": 760})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/high-depth-test")
            for fixture in expected:
                if fixture["codec"] not in ("av1", "hevc"):
                    continue
                for preferred in ("cpu", "webgl2"):
                    result = page.evaluate(
                        "async params => (await import('/tests/high-depth-browser.js')).play(params)",
                        {"fixture": fixture, "preferred": preferred},
                    )
                    page.screenshot(
                        path=str(args.output / f'{fixture["codec"]}-{preferred}.png')
                    )
                    reports.append(result)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
    (args.output / "report.json").write_text(
        json.dumps({"status": "pass", "runs": reports}, indent=2) + "\n"
    )
    print(json.dumps({"status": "pass", "runs": reports}))


if __name__ == "__main__":
    main()
