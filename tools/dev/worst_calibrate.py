"""Worst-1% calibration (nceval.jreal WORST_WINDOW): windowed against per-frame worst-1% on the real-use anchors.

usage: python3 tools/dev/worst_calibrate.py [CLIPS] [FRAMES]   (default: natural VAL, 150)
Arms: SVT-AV1 M12 rtc, openh264, x264 veryfast and x265 superfast (zerolatency). Prints per arm the mean
per-point lift of the window form over the frame form (dB), and per arm pair the class-balanced worst-1% BD
of both forms beside the PSNR-YUV BD. Uses the shared real-use cache (measures what is missing).
"""
import itertools, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import jreal, realuse
from nceval.config import load_paths
from nceval.gatepolicy import EVALUATION_GATE

clip_spec = sys.argv[1] if len(sys.argv) > 1 else "val"
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 150
paths = load_paths()
families = realuse.anchor_families(paths, realuse.OPENH264_HARNESS,
                                   extra=["x264-veryfast", "x265-superfast"])
clips = realuse.screen_clips(clip_spec, paths)
runner = realuse.Runner(paths, EVALUATION_GATE, 4, False)
curves = runner.measure("worst-1% calibration", families, clips, frames)
for f in families:
    lift = [p["worst_psnr_y"] - p["worst_frame_psnr_y"] for c in clips for p in curves[(f.name, c.name)]]
    print(f"{f.name:18s} {len(lift):3d} points  window minus frame worst-1% PSNR-Y "
          f"{statistics.fmean(lift):+.2f} dB")
for anchor, test in itertools.permutations(families, 2):
    summary = jreal.summarise(realuse.compare(curves, anchor.name, test.name, clips))
    cell = lambda term: realuse.pct(summary[term]["balanced"])
    print(f"{test.name:18s} vs {anchor.name:18s}  w1%Y window {cell('bd_worst_psnr_y'):>7s} "
          f"frame {cell('bd_worst_frame_psnr_y'):>7s}  w1%VMAF window {cell('bd_worst_vmaf_v1'):>7s} "
          f"frame {cell('bd_worst_frame_vmaf_v1'):>7s}  PSNR {cell('bd_psnr_yuv'):>7s}")
