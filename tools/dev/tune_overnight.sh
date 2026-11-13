#!/usr/bin/env bash
# Unattended preset tuning (eval/nceval/tune_pool.py; docs/handoff/tuning.md):
# a queue of presets, each a sensitivity sweep of its codec's candidate
# parameters (eval/tune_spaces/<codec>-sensitivity.txt), then an equal-CPU
# study on the live ones (maximise J at x86 CPU <= 1.01 x the preset) whose
# finalists get a Jetson cost and single-point L3. LANES presets run at once,
# each lane taking the next unclaimed preset; every step in a MemoryMax scope.
#
#   tools/dev/tune_overnight.sh COMMIT [ROTATION [TRIALS]]
#
# COMMIT is built once (nceval.levels.build), so the binary, its presets and
# the Jetson build all match. Resumable: rerun the same command (same
# ROTATION); finished steps are cached or complete, a claimed preset whose
# lane died is released by deleting its claim directory (claims/ below).
# Log: $NC_DATA_ROOT/tuning/overnight-ROTATION.log.
set -euo pipefail
commit=$(git rev-parse "${1:?usage: tune_overnight.sh COMMIT [ROTATION [TRIALS]]}^{commit}")
rotation=${2:-$(date -u +%F)}
trials=${3:-120}
lanes=2
workers=6
data=${NC_DATA_ROOT:-$HOME/nanocodec-data}
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
log=$data/tuning/overnight-$rotation.log
claims=$data/tuning/overnight-$rotation-claims
mkdir -p "$claims"
cd "$root/eval"
# codec preset k (CPU weight of L3's J: 0.6 Fast/Faster, 0.1 Default)
queue=("av1 av1-fast 0.6" "hevc hevc-faster 0.6" "h264 default 0.1" "av1 default 0.1"
       "hevc default 0.1" "av1 av1-faster 0.6" "h264 h264-faster 0.6" "av2 default 0.1"
       "vp9 default 0.1")

scoped() { systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 "$@"; }

tune_preset() { # codec preset k
  local codec=$1 preset=$2 k=$3 lane_log=$data/tuning/overnight-$rotation-$1-$2.log
  echo "== $codec $preset: sensitivity ($(date -u +%T))" >> "$log"
  scoped python3 -m nceval.tune sensitivity --codec "$codec" --preset "$preset" \
    --binary "$commit" --space "tune_spaces/$codec-sensitivity.txt" --rotation "$rotation" \
    --workers "$workers" >> "$lane_log" 2>&1 || { echo "!! $codec $preset: sensitivity failed" >> "$log"; return; }
  echo "== $codec $preset: study at CPU <= 1.01 ($(date -u +%T))" >> "$log"
  scoped python3 -m nceval.tune search --study "$codec-$preset-eq2-$rotation" --codec "$codec" \
    --preset "$preset" --binary "$commit" --space "$data/tuning/sens-$codec-$preset-$rotation/space.txt" \
    --rotation "$rotation" --cpu-max 1.01 --k "$k" --trials "$trials" --workers "$workers" \
    >> "$lane_log" 2>&1 || echo "!! $codec $preset: study failed" >> "$log"
  echo "== $codec $preset: done ($(date -u +%T))" >> "$log"
}

lane() {
  for job in "${queue[@]}"; do
    read -r codec preset k <<< "$job"
    mkdir "$claims/$codec-$preset" 2>/dev/null || continue  # another lane has it
    tune_preset "$codec" "$preset" "$k"
  done
}

for _ in $(seq "$lanes"); do
  lane &
  sleep 60  # stagger the lanes' first builds and anchor curves
done
wait
echo "== queue done ($(date -u +%T))" >> "$log"
