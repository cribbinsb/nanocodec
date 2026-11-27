#!/usr/bin/env bash
# Scene-sanity: the motion search's superblock paths (static, one position, window at zero,
# hierarchical, sparse) and motion time per superblock, for zero-vector gate settings, on
# still, noisy, crowd and bodycam clips and synthetic scenes (nc-analysis analyze example).
# CONFIGS: ';'-separated "cell scale sparse" triples.
set -euo pipefail
cargo build -q --release -p nanocodec-analysis --example analyze
A=target/release/examples/analyze
S=$HOME/nanocodec-data/scratch/scene-gates; mkdir -p "$S"
python3 tools/dev/scene_render.py "$S" 1280x720 150 crowd one-mover sensor-noise
ls -la "$S"
C=$HOME/nanocodec-data/clips/quick
IFS=';' read -ra configs <<< "${CONFIGS:-64 1 0;8 1 4}"
for clip in "$C/static_chirla_s000c3.yuv 1080 720" "$C/static_chirla_s000c3_noise_high.yuv 1080 720" \
    "$C/crowd_mot17_09.yuv 1280 720" "$C/bodycam_airport.yuv 1280 720" "$S/crowd_1280x720_150f.yuv 1280 720" \
    "$S/one-mover_1280x720_150f.yuv 1280 720" "$S/sensor-noise_1280x720_150f.yuv 1280 720"; do
  read -r path w h <<< "$clip"
  for cfg in "${configs[@]}"; do
    read -r cell scale sparse <<< "$cfg"
    out=$("$A" "$path" "$w" "$h" --frames 150 --gate-cell "$cell" --gate-scale "$scale" --sparse "$sparse" \
      | grep -E "superblocks/frame|motion us per SB|final 8x8" | tr '\n' ' ')
    echo "$(basename "$path") cell $cell scale $scale sparse $sparse: $out"
  done
done
rm -rf -- "$S"
