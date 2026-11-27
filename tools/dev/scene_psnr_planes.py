"""Per-plane PSNR of a reconstruction against its source (raw I420, same size), mean over frames.

  python3 tools/dev/scene_psnr_planes.py SOURCE RECON WIDTH HEIGHT FRAMES [FIRST]  (mean from frame FIRST on)
"""
import sys
import numpy as np

src, rec, w, h, n = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6])
first = int(sys.argv[6]) if len(sys.argv) > 6 else 0
size = w * h * 3 // 2
a = np.fromfile(src, np.uint8, n * size).reshape(n, size).astype(np.int32)
b = np.fromfile(rec, np.uint8, n * size).reshape(n, size).astype(np.int32)
planes = {"Y": slice(0, w * h), "U": slice(w * h, w * h * 5 // 4), "V": slice(w * h * 5 // 4, size)}
out = []
for name, s in planes.items():
    mse = ((a[first:, s] - b[first:, s]) ** 2).mean(axis=1)
    out.append(f"{name} {np.mean(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-10))):.2f}")
print(" ".join(out))
