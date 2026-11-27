#!/usr/bin/env bash
# Scene-sanity: fixed-QP probe of parameter variants on one rendered scene: P-frame bytes
# (the stream minus its key) and per-plane PSNR over the P frames.
# VARIANTS: ';'-separated lists of codec:qp:param=value,...  (empty params = the default).
set -euo pipefail
scene=${SCENE:-scrolling-text}; size=${SIZE:-854x480}; frames=${FRAMES:-60}
S=$HOME/nanocodec-data/scratch/scene-sanity/src-probe
python3 tools/dev/scene_render.py "$S" "$size" 300 "$scene" >/dev/null
src="$S/${scene}_${size}_300f.yuv"; W=${size%x*}; H=${size#*x}
B=${NC_BIN:-target/gate/nanocodec}
T=$(mktemp -d /tmp/nanocodec-ss-XXXX)
IFS=';' read -ra variants <<< "$VARIANTS"
for v in "${variants[@]}"; do
  IFS=: read -r codec qp params <<< "$v"
  args=(); IFS=, read -ra ps <<< "$params"; for p in "${ps[@]}"; do [ -n "$p" ] && args+=(--param "$p"); done
  for n in 1 "$frames"; do
    "$B" encode --codec "$codec" --input "$src" --width "$W" --height "$H" --fps 30 --frames "$n" --intra-period 900 \
      --qp "$qp" "${args[@]}" --output "$T/s$n" --display-recon "$T/r.yuv" >/dev/null
  done
  echo "$v: P bytes $(( $(stat -c %s "$T/s$frames") - $(stat -c %s "$T/s1") )) $(python3 tools/dev/scene_psnr_planes.py "$src" "$T/r.yuv" "$W" "$H" "$frames" 1)"
done
rm -r -- "$T"
