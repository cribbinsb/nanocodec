#!/usr/bin/env python3
"""The browser module and the demo page in each browser engine: Chromium,
Firefox and WebKit (Safari's engine), as Playwright ships them headless.

The node checks (tools/ci/wasm_conformance.sh) run the module under V8 only and
the page tests (tools/ci/web_tests.py) in Chromium only; this runs the same
module in every engine, so "plays in any current browser" is a checked claim.

    browser_engines.py conformance [--engine E ...] [--build B ...] [--codec C ...]
                                   [--only NAMES] [--jobs N] [--json OUT]
    browser_engines.py smoke [--engine E ...] [--no-simd] [--json OUT]
    browser_engines.py all --out DIR      # both, every engine: the record DIR/engines.json

conformance: the official corpora (as wasm_conformance.sh: H.264 the streams the
native decoder must pass, the others all 4:2:0 progressive streams) decoded in
the engine's page by module workers (engine-conformance.html), whole-output MD5
against the manifest. Builds: `auto` (default) is the one web/wasm.js picks in
that engine, `simd128`, `relaxed`, `scalar` force one; `scalar-forced` runs the
scalar module in an engine without WebAssembly SIMD: Chromium with V8 told the
CPU lacks SSE4.1 (`--js-flags=--no-enable-sse4-1`, as V8 on such a CPU offers
no SIMD; Firefox 150 has no switch for it), which exercises wasm.js's fallback.

smoke: web/tests/browser-product-smoke.py with --engine (the demo end to end);
--no-simd (Chromium) runs it without WebAssembly SIMD, on the scalar module.

Exit status 0 when everything checked passes. Needs Playwright's browsers
(`python3 -m playwright install chromium firefox webkit`; on Ubuntu 26.04 with
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64, and WebKit's missing
libraries from tools/web/webkit_host_libs.sh, run here when needed) and a built
web/pkg (tools/web/build.sh).
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
CORPORA = DATA / "conformance"
ENGINES = ("chromium", "firefox", "webkit")
#: Module directories under web/pkg (tools/web/build.sh).
BUILDS = {"simd128": "web/pkg", "relaxed": "web/pkg/relaxed", "scalar": "web/pkg/scalar"}
#: codec -> (manifest under tools/ci, must-pass list or None), as tools/ci/wasm_conformance.sh.
SUITES = {
    "h264": ("h264_conformance.json", "h264_conformance_passing.txt"),
    "hevc": ("hevc_conformance_v1.json", None),
    "vvc": ("vvc_conformance_v1.json", None),
    "av1": ("av1_conformance_libaom.json", None),
    "vp9": ("vp9_conformance_libvpx.json", None),
}
#: One stream's decode in a page, at most (the largest H.264 vectors take ~1 min scalar).
STREAM_TIMEOUT_S = 600
WEBKIT_LIBS = DATA / "tools/webkit-noble-libs/root/usr/lib/x86_64-linux-gnu"


def playwright_environment() -> None:
    """No display for any browser, and Playwright's Ubuntu 24.04 builds on newer
    Ubuntu. Every launch is headless; without DISPLAY and WAYLAND_DISPLAY no
    engine can open a window on the owner's session even if one ignored that
    (owner, 2026-11-27: test windows stole his focus). Playwright 1.60 knows
    no Ubuntu 26.04: it runs its Ubuntu 24.04 builds (as
    tools/dev/nc_remote_job.py sets on the build box)."""
    for name in ("DISPLAY", "WAYLAND_DISPLAY"):
        os.environ.pop(name, None)
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        return
    if release.get("ID") == "ubuntu" and release.get("VERSION_ID", "") > "24.04":
        os.environ.setdefault("PLAYWRIGHT_HOST_PLATFORM_OVERRIDE", "ubuntu24.04-x64")


def launch_options(browser_type, engine: str, *, camera: bool = False, simd: bool = True) -> dict:
    """Launch options for ENGINE: a synthetic webcam when CAMERA (Chromium's fake
    device, Firefox's fake streams; Playwright's WebKit has mock capture devices),
    WebAssembly SIMD switched off when not SIMD (Chromium only: V8 offers SIMD
    only on CPUs with SSE4.1, and can be told the CPU lacks it)."""
    options: dict = {"headless": True, "args": []}
    if not simd and engine != "chromium":
        raise ValueError("only Chromium can switch WebAssembly SIMD off")
    if engine == "chromium":
        if camera:
            options["args"] += ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
        if not simd:
            options["args"].append("--js-flags=--no-enable-sse4-1")
    elif engine == "firefox":
        if camera:
            options["firefox_user_prefs"] = {"media.navigator.streams.fake": True,
                                             "media.navigator.permission.disabled": True}
    elif engine == "webkit":
        if not Path("/usr/lib/x86_64-linux-gnu/libicuuc.so.74").exists():
            subprocess.run([REPO / "tools/web/webkit_host_libs.sh"], check=True)
            os.environ["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "1"
            options["executable_path"] = str(REPO / "tools/web/webkit-minibrowser.sh")
            options["env"] = {**os.environ, "NC_WEBKIT_DIR": str(Path(browser_type.executable_path).parent),
                              "NC_WEBKIT_LIBS": str(WEBKIT_LIBS)}
    else:
        raise ValueError(f"unknown engine {engine}")
    return options


class Files(http.server.SimpleHTTPRequestHandler):
    """The repository, with each corpus's streams under /streams/<codec>/."""

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm"}

    def translate_path(self, path):
        parts = path.split("?")[0].split("/")
        if len(parts) > 3 and parts[1] == "streams":
            return str(CORPORA / parts[2] / "streams" / "/".join(parts[3:]))
        return super().translate_path(path)

    def log_message(self, *_):
        pass


def serve() -> http.server.ThreadingHTTPServer:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Files, directory=str(REPO)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def comparable(codec: str, only: set[str]) -> tuple[list[dict], set[str] | None]:
    """The corpus entries the check compares (4:2:0 progressive, as
    conformance-runner.mjs) and the must-pass names (None: all)."""
    manifest, must = SUITES[codec]
    data = json.loads((REPO / "tools/ci" / manifest).read_text())
    entries = [e for e in (data if isinstance(data, list) else data["streams"])
               if (not only or e["name"] in only)
               and e["chroma_format"] == "4:2:0" and e.get("scan", "progressive") == "progressive"]
    if must is None:
        return entries, None
    names = {line.strip() for line in (REPO / "tools/ci" / must).read_text().splitlines()}
    return entries, {n for n in names if n and not n.startswith("#")}


async def engine_conformance(p, engine: str, simd: bool, builds: list[str], codecs: list[str], only: set[str],
                             jobs: int, port: int) -> dict:
    """Every build in BUILDS over every corpus in CODECS, in one engine (with
    WebAssembly SIMD switched off when not SIMD)."""
    browser_type = getattr(p, engine)
    browser = await browser_type.launch(**launch_options(browser_type, engine, simd=simd))
    page = await browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    await page.goto(f"http://127.0.0.1:{port}/tools/web/engine-conformance.html")
    await page.wait_for_function("window.conformance !== undefined", timeout=30000)
    detected = await page.evaluate("window.conformance.detected")
    await page.evaluate(f"window.conformance.start({jobs})")
    record = {"engine": engine, "version": browser.version, "user_agent": await page.evaluate("navigator.userAgent"),
              "simd_switched_off": not simd, "detected": detected, "builds": {}}
    auto = Path(detected["module"]).parent.relative_to("/").as_posix()
    for build in builds:
        pkg = auto if build in ("auto", "scalar-forced") else BUILDS[build]
        if build == "scalar-forced" and pkg != BUILDS["scalar"]:
            raise SystemExit(f"{engine} with SIMD switched off still picked {pkg}")
        name = next(k for k, v in BUILDS.items() if v == pkg)
        suites = {}
        for codec in codecs:
            if not (CORPORA / codec / "streams").is_dir():
                suites[codec] = {"skipped": f"no corpus in {CORPORA / codec / 'streams'}"}
                print(f"{engine} {codec}: SKIPPED, {suites[codec]['skipped']}", flush=True)
                continue
            suites[codec] = await corpus(page, pkg, codec, only, port)
            s = suites[codec]
            print(f"{engine} {browser.version} {name}{' (picked)' if build == 'auto' else ''} {codec}: "
                  f"{s['passed']}/{s['compared']} match, {s['required_failing']} required failing "
                  f"({s['seconds']:.0f} s)", flush=True)
        record["builds"][f"{build}:{name}" if build in ("auto", "scalar-forced") else name] = suites
    record["page_errors"] = errors
    await browser.close()
    return record


async def corpus(page, pkg: str, codec: str, only: set[str], port: int) -> dict:
    entries, must = comparable(codec, only)
    began = time.monotonic()

    async def one(entry):
        job = {"pkg": f"http://127.0.0.1:{port}/{pkg}", "codec": codec,
               "url": f"http://127.0.0.1:{port}/streams/{codec}/{entry['bitstream']}",
               "deep": any(d > 8 for d in entry.get("bit_depth") or [])}
        try:
            result = await asyncio.wait_for(page.evaluate("job => window.conformance.decode(job)", job), STREAM_TIMEOUT_S)
        except asyncio.TimeoutError:
            return f"timed out after {STREAM_TIMEOUT_S} s"
        if "error" in result:
            return f"{result['error']} after {result['pictures']} pictures"
        return None if result["md5"] == entry["md5"] else f"the {result['pictures']} pictures differ from the reference's output"

    problems = await asyncio.gather(*(one(e) for e in entries))
    streams = {e["name"]: {"required": must is None or e["name"] in must, "problem": problem}
               for e, problem in zip(entries, problems)}
    for name, s in streams.items():
        if s["problem"]:
            print(f"  {'FAIL' if s['required'] else 'fail (not required)'} {name}: {s['problem']}", flush=True)
    return {"compared": len(entries), "passed": sum(not s["problem"] for s in streams.values()),
            "required_failing": sum(bool(s["problem"]) and s["required"] for s in streams.values()),
            "seconds": round(time.monotonic() - began, 1), "streams": streams}


async def conformance(engines, builds, codecs, only, jobs) -> list[dict]:
    from playwright.async_api import async_playwright
    server = serve()
    try:
        async with async_playwright() as p:
            records = []
            for engine in engines:
                # `scalar-forced` needs a launch without SIMD, which only Chromium has.
                runs = [(True, [b for b in builds if b != "scalar-forced"])]
                if "scalar-forced" in builds and engine == "chromium":
                    runs.append((False, ["scalar-forced"]))
                for simd, chosen in runs:
                    if chosen:
                        records.append(await engine_conformance(p, engine, simd, chosen, codecs, only, jobs,
                                                                server.server_port))
            return records
    finally:
        server.shutdown()


def conformance_passed(records: list[dict]) -> bool:
    """No page error, no required stream failing and no corpus missing (a check
    that does not run must not pass, as tools/ci/wasm_conformance.sh)."""
    return all(not r["page_errors"] and all("skipped" not in s and s["required_failing"] == 0
                                            for b in r["builds"].values() for s in b.values())
               for r in records)


def smoke(engine: str, no_simd: bool = False) -> dict:
    """The product smoke in ENGINE; its summary line, or the failure's tail."""
    began = time.monotonic()
    run = subprocess.run([sys.executable, REPO / "web/tests/browser-product-smoke.py", "--engine", engine,
                          *(["--no-simd"] if no_simd else [])], capture_output=True, text=True)
    seconds = round(time.monotonic() - began, 1)
    if run.returncode == 0:
        return {"status": "pass", "seconds": seconds, "summary": json.loads(run.stdout.strip().splitlines()[-1])}
    return {"status": "fail", "seconds": seconds, "error": (run.stdout + run.stderr)[-4000:]}


def main() -> int:
    playwright_environment()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    conf = sub.add_parser("conformance")
    conf.add_argument("--engine", action="append", choices=ENGINES)
    conf.add_argument("--build", action="append", choices=["auto", "scalar-forced", *BUILDS])
    conf.add_argument("--codec", action="append", choices=list(SUITES))
    conf.add_argument("--only", default="", help="comma-separated stream names")
    conf.add_argument("--jobs", type=int, default=8)
    conf.add_argument("--json", type=Path)
    sm = sub.add_parser("smoke")
    sm.add_argument("--engine", action="append", choices=ENGINES)
    sm.add_argument("--no-simd", action="store_true")
    sm.add_argument("--json", type=Path)
    every = sub.add_parser("all")
    every.add_argument("--out", type=Path, required=True)
    every.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    if args.command == "conformance":
        records = asyncio.run(conformance(args.engine or list(ENGINES), args.build or ["auto"],
                                          args.codec or list(SUITES), set(filter(None, args.only.split(","))), args.jobs))
        if args.json:
            args.json.write_text(json.dumps(records, indent=1) + "\n")
        return 0 if conformance_passed(records) else 1
    if args.command == "smoke":
        results = {engine: smoke(engine, args.no_simd) for engine in args.engine or ENGINES}
        for engine, result in results.items():
            print(f"{engine}: {result['status']} ({result['seconds']} s)" + (f"\n{result['error']}" if result["status"] == "fail" else ""))
        if args.json:
            args.json.write_text(json.dumps(results, indent=1) + "\n")
        return 0 if all(r["status"] == "pass" for r in results.values()) else 1
    args.out.mkdir(parents=True, exist_ok=True)
    records = asyncio.run(conformance(list(ENGINES), ["auto", "scalar-forced"], list(SUITES), set(), args.jobs))
    smokes = {engine: smoke(engine) for engine in ENGINES}
    smokes["chromium-no-simd"] = smoke("chromium", no_simd=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    record = {"commit": commit, "date": time.strftime("%Y-%m-%d"), "host": platform.platform(),
              "conformance": records, "product_smoke": smokes,
              "passed": conformance_passed(records) and all(s["status"] == "pass" for s in smokes.values())}
    (args.out / "engines.json").write_text(json.dumps(record, indent=1) + "\n")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
