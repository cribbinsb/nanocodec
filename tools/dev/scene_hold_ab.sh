#!/usr/bin/env bash
# Scene-sanity: waving-hand at the demo's CBR settings with parameter variants: the bits and
# luma PSNR of each phase of the 10 s cycle (wave, hold, leave, empty, return), from the rate
# log and the reconstruction.
set -euo pipefail
size=${SIZE:-854x480}; kbps=${KBPS:-400}; codecs=${CODECS:-h264 av1}
S=$HOME/nanocodec-data/scratch/scene-sanity/src-hand
python3 tools/dev/scene_render.py "$S" "$size" 300 waving-hand >/dev/null
src="$S/waving-hand_${size}_300f.yuv"; W=${size%x*}; H=${size#*x}
B=${NC_BIN:-target/gate/nanocodec}
T=$(mktemp -d /tmp/nanocodec-ss-XXXX)
for codec in $codecs; do
  for variant in "" ${VARIANTS:-hold.refine_octaves=0}; do
    params=(); [ -n "$variant" ] && params=(--param "$variant")
    "$B" encode --codec "$codec" --input "$src" --width "$W" --height "$H" --fps 30 --frames 300 --intra-period 900 \
      --bitrate "$kbps" --peak-bitrate $((2 * kbps)) --buffer-ms 1000 "${params[@]}" \
      --output "$T/s" --display-recon "$T/r.yuv" --rate-log "$T/rate.csv" >/dev/null
    python3 - "$src" "$T/r.yuv" "$T/rate.csv" "$W" "$H" "$codec [$variant]" <<'PY'
import csv, sys
import numpy as np
src, rec, log, w, h, label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
size = w * h * 3 // 2
a = np.fromfile(src, np.uint8).reshape(-1, size)[:, :w * h].astype(np.int32)
b = np.fromfile(rec, np.uint8).reshape(-1, size)[:, :w * h].astype(np.int32)
psnr = 10 * np.log10(255 ** 2 / np.maximum(((a - b) ** 2).mean(axis=1), 1e-10))
rows = list(csv.DictReader(open(log)))
bits = np.zeros(len(psnr)); qp = np.full(len(psnr), np.nan)
for r in rows:
    bits[int(r["frame"])] = 8 * int(r["bytes"]); qp[int(r["frame"])] = float(r["qp"])
phases = {"wave": (30, 150), "hold": (150, 225), "leave": (225, 255), "empty": (255, 285), "return": (285, 300)}
total = bits.sum() * 30 / len(bits) / 1000
cells = [f"{k} {bits[s:e].mean() / 1000:5.1f}kb {psnr[s:e].mean():5.2f}dB" for k, (s, e) in phases.items()]
print(f"{label}: {total:.0f} kbps | " + " | ".join(cells) + f" | hold end {psnr[220:225].mean():.2f} dB, empty end {psnr[280:285].mean():.2f} dB")
PY
  done
done
rm -r -- "$T"
