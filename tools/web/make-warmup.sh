#!/usr/bin/env bash
# Rebuild web/warmup/<codec>.ivf: the tiny built-in streams the demo's decode
# worker decodes once at start-up (web/warmup.js), so V8 has compiled the hot
# decode paths before the first real frame. Eight 256x144 frames (one key, seven
# inter) of the Big Buck Bunny demo clip at 250 kbps, enough residual for every
# coding tool on the default presets' paths.
#
#   tools/web/make-warmup.sh NANOCODEC_BINARY
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
nanocodec=$1
# Piped end to end: no intermediate files.
for codec in h264 hevc av1 vp9 av2 vvc; do
  ffmpeg -v error -ss 3 -i "$repo_root/web/clips/bbb-tree-360p.mp4" -frames:v 8 \
    -vf scale=256:144:flags=area -pix_fmt yuv420p -f rawvideo - |
    "$nanocodec" encode --codec "$codec" --input - --width 256 --height 144 \
      --fps 15 --frames 8 --bitrate 250 --buffer-ms 1000 --output - |
    node "$repo_root/tools/web/make-warmup.mjs" "$codec" /dev/stdin \
      "$repo_root/web/warmup/$codec.ivf" 256 144
done
