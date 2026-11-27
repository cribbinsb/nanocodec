#!/usr/bin/env python3
"""Make the waving-hand test scene's background photograph (web/scenes/room-photo-1280x720.png).

Source: "Abbotsford House Study Room" by Michael D Beckwith, CC0 1.0, Wikimedia Commons
(web/clips/ATTRIBUTION.md). The 2048x1365 file is cropped to 16:9 (2048x1152 from row 106),
scaled to 1280x720 (ffmpeg Lanczos), softened (Gaussian radius 0.45) and held to 7 bits a
channel so the lossless PNG stays under ~1.5 MB, and saved with no colour chunks (no ICC,
gAMA, cHRM, sRGB or pHYs: a browser must not colour-manage it; web/scene.js checks the
decoded pixels against PHOTO_PIXELS_SHA256). Prints the file's and the RGBA pixels' SHA-256.

  tools/web/make-scene-photo.py
"""
import hashlib
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

URL = "https://upload.wikimedia.org/wikipedia/commons/6/6a/Abbotsford_House_Study_Room_%28248456373%29.jpeg"
OUT = Path(__file__).resolve().parents[2] / "web/scenes/room-photo-1280x720.png"
CROP = "crop=2048:1152:0:106"
SIZE = (1280, 720)
SOFTEN_RADIUS = 0.45
DROPPED_BITS = 1


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nanocodec-scene-photo-") as scratch:
        source, scaled = Path(scratch) / "source.jpeg", Path(scratch) / "scaled.png"
        subprocess.run(["curl", "-sSfL", "-A", "nanocodec-asset-fetch/1.0", "-o", str(source), URL], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf",
                        f"{CROP},scale={SIZE[0]}:{SIZE[1]}:flags=lanczos,format=rgb24", "-map_metadata", "-1", str(scaled)],
                       check=True)
        pixels = np.asarray(Image.open(scaled).convert("RGB").filter(ImageFilter.GaussianBlur(SOFTEN_RADIUS)))
    kept = (pixels >> DROPPED_BITS) << DROPPED_BITS
    kept = kept | (kept >> (8 - DROPPED_BITS))  # 7-bit levels spread over 0..255
    image = Image.fromarray(kept.astype(np.uint8))
    encoded = io.BytesIO()
    image.save(encoded, "PNG", optimize=True)
    OUT.write_bytes(encoded.getvalue())
    print(f"{OUT.name}: {len(encoded.getvalue())} bytes")
    print("file sha256  ", hashlib.sha256(encoded.getvalue()).hexdigest())
    print("pixels sha256", hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest())


if __name__ == "__main__":
    main()
