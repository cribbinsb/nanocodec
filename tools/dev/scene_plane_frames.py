"""Per-frame chroma error of a reconstruction against its source (raw I420): the frames
whose U or V differ, with their MSE and mean signed error.

  python3 tools/dev/scene_plane_frames.py SOURCE RECON WIDTH HEIGHT FRAMES
"""
import sys
import numpy as np

src, rec, w, h, n = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6])
size = w * h * 3 // 2
a = np.fromfile(src, np.uint8, n * size).reshape(n, size).astype(np.int32)
b = np.fromfile(rec, np.uint8, n * size).reshape(n, size).astype(np.int32)
u = slice(w * h, w * h * 5 // 4)
d = b[:, u] - a[:, u]
bad = np.nonzero((d != 0).any(axis=1))[0]
print(f"frames with U error: {len(bad)} of {n}: {bad.tolist()[:40]}")
for i in bad[:12]:
    nz = d[i] != 0
    print(f"  frame {i}: mse {np.mean(d[i] ** 2):.2f} mean {d[i].mean():.2f} nonzero {nz.mean():.3f} "
          f"range {d[i].min()}..{d[i].max()}")
