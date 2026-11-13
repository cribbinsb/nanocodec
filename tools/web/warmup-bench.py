#!/usr/bin/env python3
"""Decoder start-up latency with and without the decode worker's warm-up.

Headless Chromium (Playwright) runs the demo's own decode worker
(web/decode-worker.js, relaxed module of web/pkg) in a fresh browser per pass,
so no compiled code survives between passes. Each pass feeds one 720p stream
packet by packet, as the encode worker does, after one of three starts:

- cold: warm-up off, packets from the moment the worker is ready;
- warm: warm-up on, packets once it has finished;
- join: warm-up on, packets 150 ms after ready (the first packet usually
  interrupts it, as when a viewer joins and data is already flowing).

Reported per codec: frame 0, the first 30 frames and the rest, as the worker's
decode time (`decodeMs`) and the page's round trip per frame, best of the
interleaved rounds.

  tools/web/build.sh
  python3 tools/web/warmup-bench.py --streams DIR [--codecs h264,hevc,av1,vp9,av2,vvc] [--rounds 3]

Streams are bench.py's cache names (`<clip>.<kbps>k.<codec>.<ext>`, 720p).
"""
import argparse
import functools
import http.server
import json
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[2]
EXTENSIONS = {"h264": "264", "hevc": "hevc", "av1": "ivf", "vp9": "ivf", "av2": "ivf", "vvc": "vvc"}
MODES = {"cold": {"warmUp": False, "waitWarm": False, "delayMs": 0},
         "warm": {"warmUp": True, "waitWarm": True, "delayMs": 0},
         "join": {"warmUp": True, "waitWarm": False, "delayMs": 150}}
FIRST = 30


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def summary(times):
    return {"frame0": times[0], "first30": sum(times[:FIRST]), "rest": sum(times[FIRST:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--streams", type=Path, nargs="+", required=True, help="stream files or directories")
    parser.add_argument("--clip", default="crowd_mot17_09.600k")
    parser.add_argument("--codecs", default="h264,hevc,av1,vp9,av2,vvc")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    codecs = args.codecs.split(",")
    paths = {}
    for codec in codecs:
        name = f"{args.clip}.{codec}.{EXTENSIONS[codec]}"
        found = [d / name for d in args.streams if (d / name).exists()]
        if not found:
            raise SystemExit(f"no {name} in {args.streams}")
        paths[codec] = found[0]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(REPO)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = f"http://127.0.0.1:{server.server_port}"
    results = {(codec, mode): [] for codec in codecs for mode in MODES}
    try:
        with sync_playwright() as p:
            for round_index in range(args.rounds):
                for codec in codecs:
                    modes = list(MODES) if round_index % 2 == 0 else list(MODES)[::-1]
                    for mode in modes:
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page()
                        errors = []
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        page.route("**/streams/*", lambda route: route.fulfill(body=paths[route.request.url.rsplit("/", 1)[1]].read_bytes()))
                        page.goto(f"{root}/tools/web/warmup-bench.html")
                        page.wait_for_function("window.ready")
                        out = page.evaluate("o => window.runPass(o)", {"url": f"/streams/{codec}", "codec": codec,
                                                                        "width": 1280, "height": 720, **MODES[mode]})
                        browser.close()
                        assert not errors, errors
                        results[(codec, mode)].append(out)
                print(f"round {round_index + 1}/{args.rounds} done", flush=True)
    finally:
        server.shutdown()
    rows = []
    print(f"\nms, best of {args.rounds}: decode (worker) | round trip (page)")
    for codec in codecs:
        for mode in MODES:
            passes = results[(codec, mode)]
            best = min(passes, key=lambda o: sum(o["latencyMs"][:FIRST]))
            d, l = summary(best["decodeMs"]), summary(best["latencyMs"])
            warmed = [o["warmedMs"] for o in passes if o["warmedMs"] is not None]
            rows.append({"codec": codec, "mode": mode, "decode": d, "latency": l, "warmed_ms": warmed})
            print(f"{codec:5} {mode:5} frame0 {d['frame0']:6.1f} | {l['frame0']:6.1f}   first30 {d['first30']:7.1f} | "
                  f"{l['first30']:7.1f}   rest {d['rest']:7.1f} | {l['rest']:7.1f}   warm-up ms {[round(w) for w in warmed]}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
