#!/usr/bin/env python3
"""The usage pages (web/docs/) render what the code generated, and the browser example runs.

Each page is loaded in Chromium at desktop and phone width: it must finish
filling its generated parts (body[data-ready="true"]) without console errors
or horizontal page scroll; the command-line page must show one row per option
in data/cli.json and the presets page one row per codec; each example block
must show its file's text exactly. Then the standalone browser example
(docs/examples/browser-roundtrip.html) runs against web/pkg (tools/web/build.sh)
and must decode every frame it codes.

  python3 web/tests/docs-pages.py
"""
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
DOCS = WEB / "docs"
PAGES = ("index.html", "cli.html", "c-api.html", "rust-api.html", "browser.html", "presets.html")
VIEWPORTS = {"desktop": (1280, 900), "phone": (400, 860)}
EXAMPLE = "docs/examples/browser-roundtrip.html"
TIMEOUT_MS = 120000


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def check_page(page, base: str, name: str, viewport: str, reference: dict, problems: list[str]) -> None:
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{base}docs/{name}")
    page.wait_for_function("document.body.dataset.ready", timeout=TIMEOUT_MS)
    where = f"{name} ({viewport})"
    if page.evaluate("document.body.dataset.ready") != "true":
        problems.append(f"{where}: could not load its data")
    if errors:
        problems.append(f"{where}: console errors {errors}")
    if page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
        problems.append(f"{where}: scrolls horizontally")
    if name == "cli.html":
        expected = sum(len(g["options"]) for c in reference["commands"] for g in c["groups"])
        shown = page.locator("#options table.options tr td:first-child").count()
        if shown != expected:
            problems.append(f"{where}: {shown} option rows, cli.json has {expected}")
    if name == "presets.html":
        shown = page.locator(".presets tr td:first-child").count()
        if shown != len(reference["presets"]):
            problems.append(f"{where}: {shown} preset rows, cli.json has {len(reference['presets'])}")
    for block in page.locator("[data-source]").all():
        source = block.get_attribute("data-source")
        if block.text_content() != (DOCS / source).read_text():
            problems.append(f"{where}: {source} is not shown verbatim")


def check_example(page, base: str, problems: list[str]) -> None:
    if not (WEB / "pkg" / "nc_web.js").is_file():
        problems.append("web/pkg is not built (tools/web/build.sh): cannot run the browser example")
        return
    page.goto(base + EXAMPLE)
    page.wait_for_function("document.getElementById('status').dataset.decoded"
                           " || document.getElementById('status').dataset.error", timeout=TIMEOUT_MS)
    status = page.locator("#status")
    if status.get_attribute("data-error") or not status.get_attribute("data-decoded"):
        problems.append(f"browser example: {status.inner_text()}")


def main() -> int:
    handler = functools.partial(QuietHandler, directory=str(WEB))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    reference = json.loads((DOCS / "data" / "cli.json").read_text())
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for viewport, (width, height) in VIEWPORTS.items():
            for name in PAGES:
                page = browser.new_page(viewport={"width": width, "height": height})
                check_page(page, base, name, viewport, reference, problems)
                page.close()
        page = browser.new_page()
        check_example(page, base, problems)
        browser.close()
    server.shutdown()
    for problem in problems:
        print(f"FAIL {problem}")
    print("docs pages: " + ("FAILED" if problems else "ok"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
