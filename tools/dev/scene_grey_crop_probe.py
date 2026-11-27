"""A4 probe: which crop of the rendered zone plate frame keeps AV2's corner chroma bug.

  python3 tools/dev/scene_grey_crop_probe.py BIN SOURCE_YUV (1280x720, frame 0 used)
"""
import subprocess, sys, tempfile
import numpy as np
from pathlib import Path

binary, source = sys.argv[1], sys.argv[2]
W, H = 1280, 720
y = np.fromfile(source, np.uint8, W * H).reshape(H, W)
with tempfile.TemporaryDirectory(prefix="nanocodec-a4crop-") as tmp:
    for w, h in [(1280, 720), (640, 360), (512, 288), (256, 144), (128, 128), (64, 64)]:
        frame = np.concatenate([y[:h, :w].ravel(), np.full(w * h // 2, 128, np.uint8)])
        src, rec = Path(tmp, "s.yuv"), Path(tmp, "r.yuv")
        frame.tofile(src)
        for qp in (230, 240):
            subprocess.run([binary, "encode", "--codec", "av2", "--preset", "medium", "--input", str(src),
                            "--width", str(w), "--height", str(h), "--fps", "30", "--frames", "1", "--qp", str(qp),
                            "--output", str(Path(tmp, "o")), "--display-recon", str(rec)],
                           check=True, capture_output=True)
            u = np.fromfile(rec, np.uint8)[w * h:]
            print(f"{w}x{h} qp {qp}: chroma values {np.unique(u).tolist()[:6]}", flush=True)
