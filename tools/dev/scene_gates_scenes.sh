#!/usr/bin/env bash
# Scene-sanity: the crowd and one-mover scenes (854x480, 200/400/800 kbps, six codecs, medium)
# under one zero-vector gate setting: scene_gates_scenes.sh LABEL PARAMS (name=value,...).
set -euo pipefail
label=$1; params=$2
cargo build -q --profile gate -p nanocodec-cli
export NC_MIN_FREE_GIB=30 NC_METRICS_BACKEND=libvmaf-cpu NC_CPU_SLOTS=24 NC_SLOT_DIR=$HOME/nanocodec-data/scratch/scene-gates-slots-$label
S=$HOME/nanocodec-data/scratch/scene-gates-src-$label
python3 tools/dev/scene_sanity.py --sources "$S" --work "$HOME/nanocodec-data/scratch/scene-gates-work-$label" \
  --sizes 854x480 --scenes crowd,one-mover --codecs h264,hevc,vvc,av1,vp9,av2 --rates 200,400,800 --workers 12 \
  --bin target/gate/nanocodec --out "remote-out/scenes-$label.json" --params "$params" --label="-$label"
rm -rf -- "$S" "$HOME/nanocodec-data/scratch/scene-gates-work-$label"
