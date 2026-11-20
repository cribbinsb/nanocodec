#!/usr/bin/env python3
"""Every figure the results pages (web/results/) show comes from their data.

Each page is loaded in Chromium at desktop and phone width, with every option
of every segmented control clicked in turn; after each render, every number in
the page's text must be one of the values in web/results/data/*.json (at any
display rounding) or one of the few setup constants the pages' own prose names
(NAMED_CONSTANTS). A figure typed into a page, or a value computed wrongly from
the data, fails the test. It also fails on any console error, on a page that
reports it could not load its data, and on horizontal page scroll at phone width.

  python3 web/tests/results-pages.py
"""
import functools
import http.server
import json
from decimal import ROUND_HALF_UP, Decimal
import re
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
# Each page and the data files it renders: its figures may come only from those.
PAGES = {"index.html": ("chart",), "codecs.html": ("chart",),
         "speed.html": ("chart", "wasm", "decoders", "wasm-yardstick", "hardware"), "rate-control.html": ("rc-cost",),
         "fixed-qp.html": ("fixed-qp-lens", "chart"), "clips.html": ("clips",),
         "svc.html": ("svc",), "method.html": ("chart", "window"), "yardsticks.html": (),
         "history.html": ("history",)}
VIEWPORTS = {"desktop": (1280, 900), "phone": (400, 860)}
# Numbers that the pages' prose uses as names or setup facts, not results:
# resolutions and formats (720p, 4:2:0, 8-bit, 10-bit), product names
# (Ryzen 9 5950X, x264, x265, Cortex-A78AE, RTX 5090, H.264, AV1, ...), the
# 150-frame and 600-frame windows of the window study (5 s / 20 s, a key every
# 10 s) and the 95% bootstrap intervals.
NAMED_CONSTANTS = {"0", "0.04", "0.92", "1", "2", "3", "4", "5", "8", "9", "10", "20", "30", "95", "150", "180",
                   "264", "265", "360", "420", "600", "720", "5950", "5090", "78"}
NUMBER = re.compile(r"(?<![\w.])[−+-]?\d[\d,]*(?:\.\d+)?")
# Product and format names whose digits are not figures.
NAMES = re.compile(r"Ryzen 9 5950X|Cortex-A78AE|RTX 5090|H\.26[45]|x26[45]|AV[12]|VP9|"
                   r"4:2:0|\d+-bit|L\dT\d(?:_KEY)?|\d{3,4}p|ffmpeg\.wasm|dav[12]d|M\d+|rt \d+|"
                   r"p[47]|libvpx v[\d.]+|v\d+\.\d+\.\d+|\b[0-9a-f]{8}\b|SIMD128|NEON|AVX(?:2|-512)|"
                   r"\d{4}-\d{2}-\d{2}|[\w-]+\.(?:csv|json|py|md)|svc\d?")


def allowed_values(names: tuple[str, ...]) -> set[str]:
    """Every number in these data files at the roundings the pages use, without sign."""
    values: set[str] = set(NAMED_CONSTANTS)

    def add(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            for digits in (0, 1, 2, 3):
                values.add(f"{abs(x):.{digits}f}")
                # JavaScript's toFixed rounds an exact half up (1.25 -> 1.3), Python to even
                values.add(str(Decimal(repr(abs(x))).quantize(Decimal(1).scaleb(-digits),
                                                              ROUND_HALF_UP)))
            if x:
                values.add(f"{abs(x):g}")
        elif isinstance(x, str):
            for token in NUMBER.findall(x):
                values.add(token.lstrip("−+-").replace(",", ""))
        elif isinstance(x, dict):
            for k, v in x.items():
                add(k)
                add(v)
        elif isinstance(x, list):
            for v in x:
                add(v)

    data = {name: json.loads((WEB / "results" / "data" / f"{name}.json").read_text()) for name in names}
    add(list(data.values()))
    # Ratios the speed page derives from two data values (native / WASM time).
    for row in data.get("wasm", {}).get("rows", []):
        values.add(f"{row['native_fps'] / row['wasm_fps']:.2f}")
    return values


def figures(text: str) -> list[str]:
    return [t.lstrip("−+-").replace(",", "") for t in NUMBER.findall(NAMES.sub(" ", text))]


def check_page(page, url: str, allowed: set[str], problems: list[str], label: str) -> None:
    page.goto(url)
    page.wait_for_load_state("networkidle")

    def scan(state: str) -> None:
        # Chart axes carry tick values, not results: everything else is checked.
        text = page.evaluate("""() => {
            const main = document.querySelector('main').cloneNode(true);
            main.querySelectorAll('svg').forEach((s) => s.remove());
            document.body.append(main);
            const text = main.innerText;
            main.remove();
            return text;
        }""")
        if "Could not load the results data" in text:
            problems.append(f"{label}: data did not load ({state})")
        for token in figures(text):
            if token not in allowed:
                problems.append(f"{label} [{state}]: figure {token!r} is not in the data")

    scan("default")
    # The fixed-QP page's rate-mode switch links back to the overview: each page is
    # checked on its own (the overview's own Fixed QP mode is an in-page chart).
    groups = [g for g in page.query_selector_all("main .segmented")
              if not g.evaluate("g => Boolean(g.closest('.mode-switch'))")]
    for g, group in enumerate(groups):
        for b, button in enumerate(group.query_selector_all("button")):
            if button.is_disabled():
                continue
            button.click()
            scan(f"control {g} option {b}")
    if page.query_selector("main .hit"):
        # by keyboard: two points may sit close enough that one's target covers the other
        page.focus("main .hit")
        page.keyboard.press("Enter")
        scan("point selected")
    if page.query_selector("details.table-view"):
        page.click("details.table-view > summary")
        scan("table view")
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    if overflow > 1:
        problems.append(f"{label}: page scrolls sideways by {overflow}px")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def main() -> int:
    handler = functools.partial(QuietHandler, directory=str(WEB))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/results"
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, (width, height) in VIEWPORTS.items():
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.on("console", lambda m, n=name: problems.append(f"{n}: console {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e, n=name: problems.append(f"{n}: {e}"))
            for page_name, data_files in PAGES.items():
                check_page(page, f"{base}/{page_name}", allowed_values(data_files), problems,
                           f"{name} {page_name}")
            context.close()
        browser.close()
    server.shutdown()
    for problem in dict.fromkeys(problems):
        print(problem)
    print(f"results pages: {len(PAGES)} pages, {len(VIEWPORTS)} widths, "
          f"{'FAIL' if problems else 'ok'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
