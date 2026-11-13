#!/usr/bin/env python3
"""A live view codes exactly the clip file's decoded planes: the capture worker
(web/capture-worker.js) hands the encoder the video's own YUV planes and nc-web
`SourceFrame` places them in the view, giving the same bytes as ffmpeg's decode
of the mp4 placed by tools/web/source_frames.py (which the pre-encoded
reference streams code, tools/web/make-reference-streams.py).

Headless Chromium plays every sample clip variant (web/clips/*-360p.mp4 and
*-720p.mp4) and hashes the converted frame at the variant's own
size and at the demo's scaled sizes that use the variant, with software
decoding (I420 frames) and, if `--gpu vulkan`, ANGLE on the host GPU (NV12
frames read back from the GPU). Every frame must take the planes path and
every hash must match. The clips play, and each frame is staged in its frame
callback as the page does (web/capture.js). Needs web/pkg (tools/web/build.sh)
and ffmpeg.

  python3 web/tests/capture-identity.py [--gpu swiftshader,vulkan] [--clips N]
"""
import argparse
import functools
import hashlib
import http.server
import json
from pathlib import Path
import subprocess
import sys
import threading
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB.parent / "tools/web"))
from source_frames import split_i420, view_frame  # noqa: E402
GPU_FLAGS = {"swiftshader": [], "vulkan": ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist"]}
# Each variant at its own size, and the demo's other sizes it is scaled to (web/clips.js clipUrl).
SIZES = {"360p": [[640, 360], [384, 216]], "720p": [[1280, 720], [854, 480]]}
# Clips are limited range. Chromium reads a file whose H.264 VUI names the BT.709
# matrix but not its primaries or transfer as SMPTE 170M at SD sizes and as BT.709
# at HD sizes (most 360p variants), and shows the <video> element that way too.
LIMITED_COLOURS = ("bt601", "bt709")
# Media times (s) from which a frame is compared in each clip (the first presented at or after each).
TIMES_S = [0.5, 2.0]


def frame_index(result: dict, clip: Path) -> int:
    """The decoded frame a presented media time names (frames start at index / fps)."""
    return round(result["mediaTime"] * frame_rate(clip))


@functools.cache
def frame_rate(clip: Path) -> float:
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate",
                           "-of", "csv=p=0", str(clip)], capture_output=True, text=True, check=True).stdout.strip()
    num, den = map(int, rate.split("/"))
    return num / den


def decoded_frames(clip: Path, indices: set[int]) -> tuple[dict[int, bytes], int, int, float]:
    """ffmpeg's yuv420p planes of the wanted frame indices, with the size and frame rate."""
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                       "stream=width,height,r_frame_rate", "-of", "json", str(clip)],
                                      capture_output=True, text=True, check=True).stdout)["streams"][0]
    width, height = probe["width"], probe["height"]
    num, den = map(int, probe["r_frame_rate"].split("/"))
    size = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", str(max(indices) + 1),
                                "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    frames = {}
    for index in range(max(indices) + 1):
        data = decoder.stdout.read(size)
        if index in indices:
            frames[index] = data
    decoder.stdout.close()
    decoder.wait()
    return frames, width, height, num / den


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", default="swiftshader")
    parser.add_argument("--clips", type=int, help="only the first N clips")
    args = parser.parse_args()
    clips = sorted(WEB.glob("clips/*-360p.mp4")) + sorted(WEB.glob("clips/*-720p.mp4"))
    if args.clips:
        clips = [c for c in clips if c.name.rsplit("-", 1)[0] in sorted({c.name.rsplit("-", 1)[0] for c in clips})[:args.clips]]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    failures = []
    try:
        with sync_playwright() as p:
            for gpu in args.gpu.split(","):
                browser = p.chromium.launch(headless=True, args=GPU_FLAGS[gpu])
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/tests/capture-identity.html")
                page.wait_for_function("window.ready", timeout=60000)
                for clip in clips:
                    sizes = SIZES[clip.stem.rsplit("-", 1)[1]]
                    results = page.evaluate("o => window.compare(o)", {"url": f"../clips/{clip.name}", "sizes": sizes, "times": TIMES_S})
                    frames, width, height, fps = decoded_frames(clip, {frame_index(r, clip) for r in results})
                    bad = []
                    for r in results:
                        planes = split_i420(frames[frame_index(r, clip)], width, height)
                        expected = [hashlib.sha256(view_frame(planes, tuple(size))).hexdigest() for size in sizes]
                        # The colour is the browser's reading of the file's tags (colour.js
                        # `colourOf`): limited range either way; the samples must match.
                        if r["kind"] != "planes" or r["colour"] not in LIMITED_COLOURS or r["hashes"] != expected:
                            bad.append({k: r.get(k) for k in ("mediaTime", "kind", "format", "colour")}
                                       | {"sizes": [s for s, a, b in zip(sizes, r["hashes"], expected) if a != b]})
                    paths = sorted({f"{r['format']} {r['colour']}" for r in results})
                    print(f"{gpu:11} {clip.name:40} {'differs ' + str(bad) if bad else 'identical'} ({', '.join(paths)})")
                    failures += [(gpu, clip.name, r) for r in bad]
                assert not errors, errors
                browser.close()
    finally:
        server.shutdown()
    assert not failures, failures
    print(f"capture identical to the decoded planes: {len(clips)} clip variants, {args.gpu}")


if __name__ == "__main__":
    main()
