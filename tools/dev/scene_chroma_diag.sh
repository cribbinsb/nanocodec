#!/usr/bin/env bash
# scene-diag-2: AV1 chroma decay on still backgrounds: per-plane PSNR by second and where the
# last frame's V error sits (8x8 chroma blocks: mean |error|, even vs odd rows), per variant.
set -euo pipefail
S=$HOME/nanocodec-data/scratch/scene-diag-2/src
B=${NC_BIN:-target/gate/nanocodec}
python3 tools/dev/scene_render.py "$S" 854x480 300 ${SCENES:-one-mover gradient-pan} >/dev/null
T=$(mktemp -d /tmp/nanocodec-sd2-XXXX)
for sc in ${SCENES:-one-mover gradient-pan}; do
  src="$S/${sc}_854x480_300f.yuv"
  IFS=';' read -ra variants <<< "${VARIANTS:-}"
  for v in "" "${variants[@]}"; do
    args=(); IFS=, read -ra ps <<< "$v"; for p in "${ps[@]}"; do [ -n "$p" ] && args+=(--param "$p"); done
    "$B" encode --codec ${CODEC:-av1} --input "$src" --width 854 --height 480 --fps 30 --frames 300 --intra-period 900 \
      --bitrate ${KBPS:-400} --peak-bitrate $((2 * ${KBPS:-400})) --buffer-ms 1000 "${args[@]}" \
      --output "$T/s.bin" --display-recon "$T/r.yuv" >/dev/null
    python3 - "$src" "$T/r.yuv" "$sc [$v] $(stat -c %s "$T/s.bin")B" <<'PY'
import sys
import numpy as np
src, rec, label = sys.argv[1:4]
w, h = 854, 480; size = w * h * 3 // 2; cw, ch = w // 2, h // 2
a = np.fromfile(src, np.uint8).reshape(-1, size).astype(np.int32)
b = np.fromfile(rec, np.uint8).reshape(-1, size).astype(np.int32)
def p(s): return 10 * np.log10(255 ** 2 / np.maximum(((a[:, s] - b[:, s]) ** 2).mean(axis=1), 1e-10))
U, V = slice(w * h, w * h + cw * ch), slice(w * h + cw * ch, size)
pu, pv, py = p(U), p(V), p(slice(0, w * h))
print(label)
print("  Y by s: " + " ".join(f"{py[i:i+30].mean():.1f}" for i in range(0, 300, 30)))
print("  U by s: " + " ".join(f"{pu[i:i+30].mean():.1f}" for i in range(0, 300, 30)))
print("  V by s: " + " ".join(f"{pv[i:i+30].mean():.1f}" for i in range(0, 300, 30)))
e = (a[-1, V] - b[-1, V]).reshape(ch, cw)
print(f"  last V: mean|e| {np.abs(e).mean():.3f}, even rows MSE {(e[0::2]**2).mean():.3f} odd rows {(e[1::2]**2).mean():.3f}")
H, W = ch // 8, cw // 8
blk = np.abs(e[:H*8, :W*8]).reshape(H, 8, W, 8)
mse = (blk ** 2).mean(axis=(1, 3)); order = np.argsort(mse, axis=None)[::-1]
print(f"  top 5% blocks hold {mse.flat[order[:mse.size//20]].sum()/mse.sum():.0%} of V error")
for i in order[:6]:
    y, x = divmod(int(i), W)
    bb = blk[y, :, x, :]
    sgn = (a[-1, V] - b[-1, V]).reshape(ch, cw)[y*8:y*8+8, x*8:x*8+8]
    print(f"   chroma ({x*8},{y*8}) mse {mse.flat[i]:.1f} mean|e| {bb.mean():.2f} mean e {sgn.mean():+.2f} max {bb.max()} even-row mean|e| {bb[0::2].mean():.2f}")
PY
  done
done
rm -r -- "$T"
