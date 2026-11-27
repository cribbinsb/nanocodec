#!/usr/bin/env bash
# Scene-sanity: stream statistics (nanocodec stat, analysis build) of one rendered scene coded
# by H.264 and HEVC at one fixed QP, to compare block, motion-vector and skip choices.
set -euo pipefail
scene=${SCENE:-scrolling-text}; size=${SIZE:-854x480}; qp=${QP:-34}; frames=${FRAMES:-60}
S=$HOME/nanocodec-data/scratch/scene-sanity/src
python3 tools/dev/scene_render.py "$S" "$size" 300 "$scene" >/dev/null
CARGO_TARGET_DIR=$HOME/nanocodec-data/targets/scene-sanity-analysis cargo build -q --release -p nanocodec-cli --features analysis
A=$HOME/nanocodec-data/targets/scene-sanity-analysis/release/nanocodec
T=$(mktemp -d /tmp/nanocodec-ss-XXXX)
for codec in h264 hevc; do
  "$A" encode --codec "$codec" --input "$S/${scene}_${size}_300f.yuv" --width "${size%x*}" --height "${size#*x}" \
    --fps 30 --frames "$frames" --intra-period 900 --qp "$qp" --output "$T/s.$codec" >/dev/null
  echo "== $codec $scene $size qp $qp: $(stat -c %s "$T/s.$codec") bytes"
  (cd eval && NC_ANALYSIS_BIN=$(dirname "$A") python3 -m nceval.streamstats --codec "$codec" "$T/s.$codec")
done
rm -r -- "$T"
