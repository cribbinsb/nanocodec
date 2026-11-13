#!/usr/bin/env python3
"""Browser decode + display time per frame: the CPU path against WebGL.

Headless Chromium (Playwright, as web/tests/browser-product-smoke.py) decodes
each stream in a module worker with the relaxed-SIMD module of web/pkg and draws
every frame on a canvas with web/renderer.js, frame after frame
(tools/web/display-bench.html):

- cpu: the worker converts to RGBA in WASM (`rgba_frame`), copies it out and
  the page draws it with putImageData;
- webgl2 / webgl: the worker packs I420 (`i420_frame`), copies it out and the
  page uploads three one-channel textures and converts in a shader.

End-to-end time per frame covers decode, hand-off, transfer to the page, the
draw call and a one-pixel read-back that waits until the frame is drawn
(`--async` drops the read-back: then the GPU's work overlaps the next frame).
Passes of the renderers alternate per stream and round (interleaved); each
stream's best pass per renderer is reported. A first untimed pass per stream
compares every GL frame with the CPU conversion of the same frame (largest
per-channel difference).

GPU: `--gpu swiftshader` (default headless: software GL on the CPU) or
`--gpu vulkan` (ANGLE on the host GPU).

  tools/web/build.sh
  python3 tools/web/display-bench.py --streams DIR [--codecs h264,hevc,av1,vp9]
      [--res 720p,360p] [--kbps 720p=1200,360p=300] [--rounds 3] [--gpu vulkan] [--json out.json]

Streams are bench.py's cache (`<clip>.<kbps>k.<codec>.<ext>`): by default 720p
uses the 1200 kbps encodes, 360p (`<clip>_360p`) the 300 kbps ones (`--kbps`).
"""
import argparse
import functools
import http.server
import json
from pathlib import Path
import statistics
import threading
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[2]
EXTENSIONS = {"h264": "264", "hevc": "hevc", "av1": "ivf", "vp9": "ivf", "av2": "ivf", "vvc": "vvc"}
# Clip-name suffix per resolution, and the default stream bitrate.
RESOLUTIONS = {"720p": "", "360p": "_360p"}
DEFAULT_KBPS = "720p=1200,360p=300"
GPU_FLAGS = {"swiftshader": [], "vulkan": ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist"]}
PASS_TIMEOUT_MS = 120_000


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def streams(directory: Path, codecs, resolutions, kbps_by_res):
    """(codec, resolution, path) for every clip in the cache."""
    found = []
    for res in resolutions:
        suffix, kbps = RESOLUTIONS[res], kbps_by_res[res]
        for codec in codecs:
            for path in sorted(directory.glob(f"*{suffix}.{kbps}k.{codec}.{EXTENSIONS[codec]}")):
                if suffix or "_360p" not in path.name:
                    found.append((codec, res, path))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--codecs", default="h264,hevc,av1,vp9")
    parser.add_argument("--res", default="720p,360p")
    parser.add_argument("--kbps", default=DEFAULT_KBPS, help="stream bitrate per resolution")
    parser.add_argument("--renderers", default="cpu,webgl2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gpu", choices=GPU_FLAGS, default="swiftshader")
    parser.add_argument("--async", dest="synced", action="store_false")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    kinds = args.renderers.split(",")
    kbps_by_res = {res: int(kbps) for res, kbps in (item.split("=") for item in args.kbps.split(","))}
    jobs = streams(args.streams, args.codecs.split(","), args.res.split(","), kbps_by_res)
    if not jobs:
        raise SystemExit(f"no streams in {args.streams}")
    files = {f"/streams/{path.name}": path for _, _, path in jobs}

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(REPO)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rows = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=GPU_FLAGS[args.gpu])
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/streams/*", lambda route: route.fulfill(body=files[route.request.url.split(server_root, 1)[1]].read_bytes()))
            server_root = f"http://127.0.0.1:{server.server_port}"
            page.goto(f"{server_root}/tools/web/display-bench.html")
            page.wait_for_function("window.ready")
            run = lambda **kw: page.evaluate("options => window.runPass(options)", kw)
            results = {}
            for codec, res, path in jobs:
                url = f"/streams/{path.name}"
                checks = {kind: run(url=url, codec=codec, kind=kind, synced=True, check=True)["diff"] for kind in kinds if kind != "cpu"}
                results[path.name] = {"codec": codec, "res": res, "checks": checks, "passes": {kind: [] for kind in kinds}}
            for round_index in range(args.rounds):
                for codec, res, path in jobs:
                    order = kinds if round_index % 2 == 0 else kinds[::-1]
                    for kind in order:
                        out = run(url=f"/streams/{path.name}", codec=codec, kind=kind, synced=args.synced, check=False)
                        results[path.name]["passes"][kind].append(out)
                        results[path.name].setdefault("renderers", {})[kind] = out["renderer"]
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
    for name, result in results.items():
        best = {kind: min(passes, key=lambda o: o["perFrame"]["endToEnd"])["perFrame"] for kind, passes in result["passes"].items()}
        rows.append({"stream": name, "codec": result["codec"], "res": result["res"], "best_ms": best, "checks": result["checks"],
                     "renderers": result["renderers"]})
        cells = "  ".join(f"{kind} {b['endToEnd']:.2f} (dec {b['decode']:.2f} hand {b['handoff']:.2f} draw {b['draw']:.2f})" for kind, b in best.items())
        checks = " ".join(f"{kind} max|d| {c['max']}" for kind, c in result["checks"].items())
        print(f"{result['codec']:5} {result['res']:5} {name:42} {cells}  {checks}")
    print("\nmedian per codec/res, end-to-end ms per frame:")
    for codec in args.codecs.split(","):
        for res in args.res.split(","):
            group = [r for r in rows if r["codec"] == codec and r["res"] == res]
            if not group:
                continue
            medians = {kind: statistics.median(r["best_ms"][kind]["endToEnd"] for r in group) for kind in kinds}
            print(f"{codec:5} {res:5} " + "  ".join(f"{kind} {m:.2f}" for kind, m in medians.items()))
    if args.json:
        args.json.write_text(json.dumps({"gpu": args.gpu, "synced": args.synced, "rounds": args.rounds, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
