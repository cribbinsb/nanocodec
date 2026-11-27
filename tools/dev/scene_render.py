"""Render the demo's synthetic scenes (web/scene.js) to raw I420, as the page codes them.

Each frame is drawn in headless Chromium on a 2D canvas at the coded size (the page's
FrameSource draws the scene at the view size, capture.js) and converted RGBA -> I420
with the integer BT.601 limited-range forward matrix of nc-dsp `rgba_to_i420`
(convert/scalar.rs: luma per pixel, chroma from the rounded 2x2 RGB mean), which is
what nc-web `SourceFrame.from_rgba` applies on the page's RGBA path.

  python3 tools/dev/scene_render.py OUT_DIR WIDTHxHEIGHT FRAMES SCENE [SCENE ...]
  (SCENE_RGBA=1 also writes each frame's canvas RGBA to a .rgba file beside the .yuv.)
"""
import base64, functools, http.server, os, sys, threading, time
from pathlib import Path
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[2] / "web"
SETUP = '''async ([w, h]) => {
  const { drawScene, prepareScene, sceneById } = await import('./scene.js');
  window.prepare = async id => { await prepareScene(id); return sceneById(id).background?.() ?? 'drawn'; };
  const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const Y = [66, 129, 25], CB = [-38, -74, 112], CR = [112, -94, -18];
  const fwd = (k, off, r, g, b) => Math.min(255, Math.max(0, off + ((k[0] * r + k[1] * g + k[2] * b + 128) >> 8)));
  window.renderFrame = (id, frame) => {
    drawScene(ctx, w, h, frame / 30, id);
    const d = ctx.getImageData(0, 0, w, h).data, cw = w >> 1, ch = h >> 1;
    const out = new Uint8Array(w * h + 2 * cw * ch);
    for (let i = 0, p = 0; i < w * h; i++, p += 4) out[i] = fwd(Y, 16, d[p], d[p + 1], d[p + 2]);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      let r = 0, g = 0, b = 0;
      for (const yy of [2 * y, 2 * y + 1]) for (const xx of [2 * x, 2 * x + 1]) {
        const p = 4 * (yy * w + xx); r += d[p]; g += d[p + 1]; b += d[p + 2];
      }
      r = (r + 2) >> 2; g = (g + 2) >> 2; b = (b + 2) >> 2;
      out[w * h + y * cw + x] = fwd(CB, 128, r, g, b);
      out[w * h + cw * ch + y * cw + x] = fwd(CR, 128, r, g, b);
    }
    if (window.keepRgba) window.lastRgba = d;
    let s = ''; for (let i = 0; i < out.length; i += 0x8000) s += String.fromCharCode.apply(null, out.subarray(i, i + 0x8000));
    return btoa(s);
  };
}'''


RGBA_B64 = '''() => { const d = window.lastRgba; let s = '';
  for (let i = 0; i < d.length; i += 0x8000) s += String.fromCharCode.apply(null, d.subarray(i, i + 0x8000));
  return btoa(s); }'''


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    out, size, frames, scenes = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    w, h = map(int, size.split("x"))
    out.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/index.html?views=none")
        page.evaluate(SETUP, [w, h])
        for scene in scenes:
            start = time.time()
            # Scenes drawn from a photograph load and hash-check it first (scene.js prepareScene).
            background = page.evaluate("id => window.prepare(id)", scene)
            if background not in ("drawn", "photo"):
                raise SystemExit(f"{scene}: background is {background!r}")
            path = out / f"{scene}_{w}x{h}_{frames}f.yuv"
            rgba = open(path.with_suffix(".rgba"), "wb") if os.environ.get("SCENE_RGBA") else None
            page.evaluate(f"window.keepRgba = {'true' if rgba else 'false'}")
            with open(path, "wb") as f:
                for n in range(frames):
                    f.write(base64.b64decode(page.evaluate("([id, n]) => renderFrame(id, n)", [scene, n])))
                    if rgba:
                        rgba.write(base64.b64decode(page.evaluate(RGBA_B64)))
            if rgba:
                rgba.close()
            print(f"{path} {time.time() - start:.1f}s", flush=True)
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main()
