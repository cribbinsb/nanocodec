#!/usr/bin/env python3
"""Spatial SVC in headless Chromium: the demo's `svc` and `svc3` presets, their operating points.

The browser module (web/pkg, tools/web/build.sh) encodes a short synthetic clip
with the `svc` preset of each SVC codec (presets/av1-svc.toml, presets/vp9-svc.toml:
a half-size base layer and the full-size layer) and AV1's `svc3` preset
(presets/av1-svc3.toml: quarter-, half- and full-size layers). Three
BrowserDecoders decode it in the page, as the decode worker does
(decode-worker.js): one at operating point 0 (full size), one at the lowest
operating point (the base: AV1 drops the upper layers' OBUs, VP9 the
superframe's second frame), and one that switches on the fly the way the demo
does: down to the base at once, back up at the key frame the page then asks the
encoder for. The three-layer stream is also decoded at operating point 1 (half
size). Every decoded picture must equal the native
decoder's (`nanocodec decode --operating-point N --frame-md5`) on the same
stream, by MD5, so the browser path is the product's.
"""
import functools
import hashlib
import http.server
import os
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
REPO = WEB.parent
WIDTH, HEIGHT, FRAMES, FPS = 160, 96, 12, 30
# The switching decoder: down to the base before this frame, a key requested
# and the switch back up taken at the key frame.
DOWN_AT, KEY_AT = 4, 8

#: The codecs with spatial SVC in the demo (web/codecs.js `svc`) and their IVF FourCC.
CODECS = {"av1": b"AV01", "vp9": b"VP90"}
#: The cases: (codec, demo preset, spatial layers).
CASES = (("av1", "svc", 2), ("vp9", "svc", 2), ("av1", "svc3", 3))

PAGE_SCRIPT = '''async ([codec, preset, lowest, width, height, frames, downAt, keyAt]) => {
  const module = await import('/pkg/nc_web.js');
  await module.default();
  const encoder = new module.BrowserEncoder(codec, width, height, 180, 30, 900, preset);
  const full = new module.BrowserDecoder(codec), base = new module.BrowserDecoder(codec);
  const middle = new module.BrowserDecoder(codec);
  const switching = new module.BrowserDecoder(codec);
  base.set_operating_point(lowest);
  middle.set_operating_point(1);
  const out = { packets: [], full: [], base: [], middle: [], switching: [], keys: [] };
  const picture = d => ({ width: d.width, height: d.height, i420: Array.from(d.decoded_i420()) });
  let pending = null;
  for (let frame = 0; frame < frames; frame++) {
    const rgba = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      rgba.set([(x * 3 + y + frame * 5) & 255, (x + y * 2 + frame * 3) & 255,
        ((x ^ y) + frame * 7) & 255, 255], i);
    }
    if (frame === keyAt) encoder.request_keyframe();
    const packet = encoder.encode(rgba);
    out.packets.push(Array.from(packet)); out.keys.push(encoder.key);
    full.decode(packet); out.full.push(picture(full));
    base.decode(packet); out.base.push(picture(base));
    if (lowest > 1) { middle.decode(packet); out.middle.push(picture(middle)); }
    // As decode-worker.js: down at once, up at the next key frame.
    if (frame === downAt) switching.set_operating_point(lowest);
    if (frame === downAt + 1) pending = 0;
    if (encoder.key && pending !== null) { switching.set_operating_point(pending); pending = null; }
    switching.decode(packet); out.switching.push(picture(switching));
  }
  return out;
}'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def native_binary() -> Path:
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.config import find_tool  # noqa: E402
    return Path(os.environ.get("NC_NANOCODEC_BIN") or find_tool("nanocodec"))


def write_ivf(path: Path, codec: str, packets: list[bytes]) -> None:
    header = struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32, CODECS[codec], WIDTH, HEIGHT, FPS, 1,
                         len(packets), 0)
    with path.open("wb") as out:
        out.write(header)
        for pts, packet in enumerate(packets):
            out.write(struct.pack("<IQ", len(packet), pts) + packet)


def native_md5(binary: Path, codec: str, stream: Path, operating_point: int) -> list[str]:
    grain = ["--film-grain", "0"] if codec == "av1" else []
    result = subprocess.run([str(binary), "decode", "--codec", codec, *grain,
                             "--operating-point", str(operating_point), "--input", str(stream),
                             "--frame-md5", "-"], capture_output=True, text=True, check=True)
    return [line.strip().lower() for line in result.stdout.splitlines() if line.strip()]


def md5s(pictures: list[dict]) -> list[str]:
    return [hashlib.md5(bytes(p["i420"])).hexdigest() for p in pictures]


def main() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            outs = {(codec, preset, layers): page.evaluate(
                PAGE_SCRIPT, [codec, preset, layers - 1, WIDTH, HEIGHT, FRAMES, DOWN_AT, KEY_AT])
                for codec, preset, layers in CASES}
            browser.close()
    finally:
        server.shutdown()
    binary = native_binary()
    for (codec, preset, layers), out in outs.items():
        check(binary, codec, preset, layers, out)


def check(binary: Path, codec: str, preset: str, layers: int, out: dict) -> None:
    """One case's pictures against the native decoder's."""
    name, lowest = f"{codec.upper()} {preset}", layers - 1
    low_size = (WIDTH >> lowest, HEIGHT >> lowest)
    assert out["keys"][0] and out["keys"][KEY_AT], \
        f"{name}: the first frame and the requested one are keys"
    assert all((p["width"], p["height"]) == (WIDTH, HEIGHT) for p in out["full"]), name
    assert all((p["width"], p["height"]) == low_size for p in out["base"]), name
    with tempfile.TemporaryDirectory(prefix="svc-web-") as scratch:
        stream = Path(scratch) / "svc.ivf"
        write_ivf(stream, codec, [bytes(p) for p in out["packets"]])
        full = native_md5(binary, codec, stream, 0)
        base = native_md5(binary, codec, stream, lowest)
        middle = native_md5(binary, codec, stream, 1) if lowest > 1 else []
    assert md5s(out["full"]) == full, f"{name}: operating point 0 differs from the native decoder"
    assert md5s(out["base"]) == base, \
        f"{name}: operating point {lowest} differs from the native decoder"
    assert md5s(out["middle"]) == middle, f"{name}: operating point 1 differs from the native decoder"
    expected = full[:DOWN_AT] + base[DOWN_AT:KEY_AT] + full[KEY_AT:]
    assert md5s(out["switching"]) == expected, \
        f"{name}: switching decoder differs from the native decodes"
    print(f"PASS {name} SVC in Chromium: op 0 ({WIDTH}x{HEIGHT}) to op {lowest} "
          f"({low_size[0]}x{low_size[1]}) equal the native decoder on {FRAMES} frames; "
          f"switched down at frame {DOWN_AT}, back up at the key at frame {KEY_AT}")


if __name__ == "__main__":
    main()
