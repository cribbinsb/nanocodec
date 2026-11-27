#!/usr/bin/env bash
# A4 probe: AV2 on the zone plate, CBR as the demo runs it (scene_sanity.py: preset, peak 2x,
# 1 s buffer, 30 s key) and fixed QP; per-plane PSNR of the reconstruction.
# VARIANTS: ';'-separated mode:value:param=v,...  mode = cbr (kbps) or qp.
set -euo pipefail
scene=${SCENE:-zone-plate}; size=${SIZE:-1280x720}; frames=${FRAMES:-300}; codec=${CODEC:-av2}
S=$HOME/nanocodec-data/scratch/scene-diag-2-a4/src
[ -f "$S/${scene}_${size}_${frames}f.yuv" ] || python3 tools/dev/scene_render.py "$S" "$size" "$frames" "$scene" >/dev/null
src="$S/${scene}_${size}_${frames}f.yuv"; W=${size%x*}; H=${size#*x}
B=${NC_BIN:-target/gate/nanocodec}
T=$(mktemp -d "${TMPDIR:-/tmp}/nanocodec-a4-XXXXXX")
IFS=';' read -ra variants <<< "$VARIANTS"
for v in "${variants[@]}"; do
  IFS=: read -r mode val params <<< "$v"
  args=(); IFS=, read -ra ps <<< "$params"; for p in "${ps[@]}"; do [ -n "$p" ] && args+=(--param "$p"); done
  if [ "$mode" = cbr ]; then rc=(--bitrate "$val" --buffer-ms 1000 --peak-bitrate $((2 * val))); else rc=(--qp "$val"); fi
  "$B" encode --codec "$codec" --input "$src" --width "$W" --height "$H" --fps 30/1 --frames "$frames" --intra-period 900 \
    --temporal-layers 1 --preset "${PRESET:-medium}" "${rc[@]}" "${args[@]}" --output "$T/s" --display-recon "$T/r.yuv" ${EXTRA:-} >"$T/log" 2>&1
  [ -n "${FRAMEDIAG:-}" ] && python3 tools/dev/scene_plane_frames.py "$src" "$T/r.yuv" "$W" "$H" "$frames"
  [ -n "${SHOWLOG:-}" ] && grep -iE "$SHOWLOG" "$T/log" | head -40
  echo "$v: bytes $(stat -c %s "$T/s") $(python3 tools/dev/scene_psnr_planes.py "$src" "$T/r.yuv" "$W" "$H" "$frames")"
done
rm -r -- "$T"
