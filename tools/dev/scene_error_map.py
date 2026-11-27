"""Where a plane's error sits in one frame: the 16x16-luma-area blocks (8x8 chroma) with the largest
squared error, and the share of the plane's error in the top blocks.

  python3 tools/dev/scene_error_map.py SOURCE RECON WIDTH HEIGHT FRAME PLANE(Y|U|V)
"""
import sys
import numpy as np

src, rec, w, h, f, plane = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6]), sys.argv[6]
size = w * h * 3 // 2
def frame(path):
    d = np.fromfile(path, np.uint8, size, offset=f * size).astype(np.int32)
    cw, ch = w // 2, h // 2
    return {"Y": d[:w * h].reshape(h, w), "U": d[w * h:w * h + cw * ch].reshape(ch, cw),
            "V": d[w * h + cw * ch:].reshape(ch, cw)}[plane]
e = (frame(src) - frame(rec)) ** 2
b = 16 if plane == "Y" else 8
H, W = e.shape[0] // b, e.shape[1] // b
blocks = e[:H * b, :W * b].reshape(H, b, W, b).sum(axis=(1, 3))
order = np.argsort(blocks, axis=None)[::-1]
total = blocks.sum()
print(f"frame {f} {plane}: MSE {e.mean():.3f}; top 1% blocks hold {blocks.flat[order[:max(1, blocks.size // 100)]].sum() / total:.0%},"
      f" top 10% {blocks.flat[order[:blocks.size // 10]].sum() / total:.0%}")
scale = 16 if plane == "Y" else 16  # report positions in luma samples
for i in order[:12]:
    y, x = divmod(int(i), W)
    print(f"  luma ({x * scale},{y * scale}) block MSE {blocks.flat[i] / b / b:.1f}")
