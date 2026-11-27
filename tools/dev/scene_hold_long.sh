#!/usr/bin/env bash
# Spare-budget refinement on long single-key runs: per-third luma PSNR and rate at CBR
# targets, refinement off and on (hold.refine_fullness), H.264 and AV1 medium, on the long
# still clip and the waving-hand scene's 30 s (three cycles).
set -euo pipefail
B=target/gate/nanocodec
cargo build -q --profile gate -p nanocodec-cli
S=$HOME/nanocodec-data/scratch/scene-hold; mkdir -p "$S"
python3 tools/dev/scene_render.py "$S" 854x480 900 waving-hand >/dev/null
T=$(mktemp -d /tmp/nanocodec-ss-XXXX)
run() {  # name path w h frames fps kbps codec
  for refine in 0 0.5; do
    "$B" encode --codec "$8" --input "$2" --width "$3" --height "$4" --fps "$6" --frames "$5" --intra-period 100000 \
      --bitrate "$7" --peak-bitrate $((2 * $7)) --buffer-ms 1000 --param "hold.refine_fullness=$refine" \
      --output "$T/s" --display-recon "$T/r.yuv" >/dev/null
    python3 - "$2" "$T/r.yuv" "$3" "$4" "$5" "$6" "$T/s" "$1 $8 ${7}k refine=$refine" <<'PY'
import os, sys
import numpy as np
src, rec, w, h, n, fps, stream, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7], sys.argv[8]
size = w * h * 3 // 2
a = np.memmap(src, np.uint8, "r", shape=(n, size))[:, :w * h]
b = np.memmap(rec, np.uint8, "r", shape=(n, size))[:, :w * h]
psnr = [10 * np.log10(255 ** 2 / max(((a[i].astype(np.int32) - b[i]) ** 2).mean(), 1e-10)) for i in range(n)]
thirds = [np.mean(psnr[k * n // 3:(k + 1) * n // 3]) for k in range(3)]
kbps = os.path.getsize(stream) * 8 * fps / n / 1000
print(f"{label}: {kbps:.0f} kbps, luma PSNR thirds " + " / ".join(f"{t:.2f}" for t in thirds))
PY
  done
}
L=$HOME/nanocodec-data/clips/long/long_static_chirla_s024c2.yuv
for codec in h264 av1; do
  for kbps in 150 600; do
    run chirla "$L" 1080 720 1800 30 "$kbps" "$codec"
    run hand "$S/waving-hand_854x480_900f.yuv" 854 480 900 30 "$kbps" "$codec"
  done
done
rm -r -- "$T" "$S"
