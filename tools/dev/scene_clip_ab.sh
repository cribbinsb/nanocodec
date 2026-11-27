#!/usr/bin/env bash
# scene-diag-2: one pool clip, base vs candidate --param at several CBR rates: total kbps,
# per-plane PSNR, and per-second bytes / mean qp / luma PSNR from the rate log and recon.
# CLIP=path.mp4 W= H= FPS= CODEC= RATES="a b" PARAMS="k=v,k=v" FRAMES=150 START=0 (first source frame)
set -euo pipefail
B=${NC_BIN:-target/gate/nanocodec}; n=${FRAMES:-150}
T=$(mktemp -d "${TMPDIR:-/tmp}/nanocodec-sd2clip-XXXX")
ffmpeg -v quiet -i "$CLIP" -vf "select=gte(n\\,${START:-0})" -fps_mode passthrough -frames:v "$n" -f rawvideo -pix_fmt yuv420p "$T/src.yuv"
for kbps in $RATES; do
  for v in "" "$PARAMS"; do
    args=(); IFS=, read -ra ps <<< "$v"; for p in "${ps[@]}"; do [ -n "$p" ] && args+=(--param "$p"); done
    "$B" encode --codec "$CODEC" --input "$T/src.yuv" --width "$W" --height "$H" --fps "$FPS" --frames "$n" \
      --bitrate "$kbps" --peak-bitrate $((2 * kbps)) --buffer-ms 1000 "${args[@]}" \
      --output "$T/s.bin" --display-recon "$T/r.yuv" --rate-log "$T/rate.csv" >/dev/null
    python3 - "$T/src.yuv" "$T/r.yuv" "$T/rate.csv" "$W" "$H" "$FPS" "$kbps [$v]" <<'PY'
import csv, sys
import numpy as np
src, rec, log, w, h, fps, label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
size = w * h * 3 // 2; cw = w // 2 * (h // 2)
a = np.fromfile(src, np.uint8).reshape(-1, size).astype(np.int32)
b = np.fromfile(rec, np.uint8).reshape(-1, size).astype(np.int32)
n = min(len(a), len(b)); a, b = a[:n], b[:n]
def ps(s): return 10 * np.log10(255 ** 2 / np.maximum(((a[:, s] - b[:, s]) ** 2).mean(axis=1), 1e-10))
py, pu, pv = ps(slice(0, w * h)), ps(slice(w * h, w * h + cw)), ps(slice(w * h + cw, size))
rows = list(csv.DictReader(open(log)))
by = np.zeros(n); qp = np.full(n, np.nan)
for r in rows:
    f = int(r["frame"])
    if f < n: by[f] = int(r["bytes"]); qp[f] = float(r["qp"])
sec = max(1, int(round(fps)))
print(f"{label}: {by.sum() * 8 * fps / n / 1000:.0f} kbps Y {py.mean():.2f} U {pu.mean():.2f} V {pv.mean():.2f}; key {by[0]:.0f} B")
print("  KB/s " + " ".join(f"{by[i:i+sec].sum()/1000:.1f}" for i in range(0, n, sec)))
print("  qp   " + " ".join(f"{np.nanmean(qp[i:i+sec]):.0f}" for i in range(0, n, sec)))
print("  Y    " + " ".join(f"{py[i:i+sec].mean():.1f}" for i in range(0, n, sec)))
print("  V    " + " ".join(f"{pv[i:i+sec].mean():.1f}" for i in range(0, n, sec)))
PY
  done
done
rm -r -- "$T"
