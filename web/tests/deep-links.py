#!/usr/bin/env python3
"""The links between the results pages and the demo round-trip (deep-link.js).

- The results overview opened with `?codec=av1&preset=fast` is filtered to that
  preset and selects its point; the point's "Try in the demo" link opens the demo
  with view B at that codec, preset, size, frame rate and bitrate, and view A on
  the original source.
- View B's "How does this preset compare?" link opens the overview on the same point.
- A reference encoder's point (x264 medium) links to the demo with that
  encoder's pre-encoded stream in view A and view B live on the same codec.
- Every page shares the site header (Demo · Results · Method · GitHub).

  python3 web/tests/deep-links.py
"""
import functools
import http.server
import sys
import threading
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
LINKED = {"codec": "av1", "preset": "fast"}
SITE_SECTIONS = ["Demo", "Results", "Method", "Docs", "GitHub"]
TIMEOUT_MS = 15000


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def check(condition: bool, problem: str, problems: list[str]) -> None:
    if not condition:
        problems.append(problem)


def sections(page) -> list[str]:
    return [t.strip().rstrip("↗").strip() for t in page.locator(".site-sections a").all_inner_texts()]


def version(page, letter: str):
    return page.locator("article.version").nth(ord(letter) - ord("A"))


def main() -> int:
    handler = functools.partial(QuietHandler, directory=str(WEB))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_default_timeout(TIMEOUT_MS)
        page_errors: list[str] = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        # Results -> demo, for one of our presets.
        page.goto(urljoin(base, "results/?codec=av1&preset=fast"))
        link = page.locator("a.demo-link")
        link.wait_for()
        check(page.locator(".detail h3").inner_text() == "nanocodec AV1 Fast", "overview did not select AV1 Fast", problems)
        check(sections(page) == SITE_SECTIONS, f"results header: {sections(page)}", problems)
        href = urljoin(page.url, link.get_attribute("href"))
        query = {k: v[0] for k, v in parse_qs(urlparse(href).query).items()}
        check(query.get("codec") == "av1" and query.get("preset") == "fast" and "size" in query and "kbps" in query,
              f"demo link {href}", problems)

        page.goto(href)
        b = version(page, "B")
        b.locator(".codec-picker").wait_for()
        page.wait_for_function("document.querySelectorAll('article.version').length === 2")
        settings = {
            "codec": b.locator(".codec-picker").input_value(),
            "preset": b.locator(".preset").input_value(),
            "size": b.locator(".resolution").input_value(),
            "fps": b.locator(".fps").input_value(),
            "kbps": b.locator(".target-rate").input_value(),
        }
        expected = {k: query[k] for k in ("codec", "preset", "size", "fps", "kbps")}
        check(settings == expected, f"demo view B {settings} != {expected}", problems)
        shown = version(page, "A").locator(".view-source button[aria-pressed=true]").get_attribute("data-view-source")
        check(shown == "original", f"demo view A shows {shown}, not the original", problems)
        check(sections(page) == SITE_SECTIONS, f"demo header: {sections(page)}", problems)

        # Demo -> results, from view B.
        compare = urljoin(page.url, b.locator(".compare-link a").get_attribute("href"))
        page.goto(compare)
        page.locator(".detail h3").wait_for()
        check(page.locator(".detail h3").inner_text() == "nanocodec AV1 Fast", f"round trip landed on {page.url}", problems)
        check(page.locator(".controls select[aria-label=Codec]").input_value() == LINKED["codec"], "codec filter not set", problems)

        # A reference point links to its encoder's pre-encoded stream.
        page.goto(urljoin(base, "results/"))
        page.locator("main .hit[aria-label^='x264 medium:']").first.click()
        link = page.locator("a.demo-link")
        link.wait_for()
        href = urljoin(page.url, link.get_attribute("href"))
        stream = parse_qs(urlparse(href).query).get("stream", [""])[0]
        check("x264-medium" in stream, f"x264 medium links to stream {stream!r}", problems)
        page.goto(href)
        a = version(page, "A")
        a.locator(".stream-picker").wait_for(state="attached")
        page.wait_for_function(f"document.querySelector('article.version .stream-picker').value === {stream!r}")
        check(version(page, "B").locator(".codec-picker").input_value() == "h264", "view B is not live H.264", problems)
        check(not page_errors or all("WebAssembly" in e or "pkg/" in e for e in page_errors),
              f"page errors: {page_errors}", problems)
        browser.close()
    server.shutdown()
    for problem in problems:
        print(problem)
    print(f"deep links: {'FAIL' if problems else 'ok'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
