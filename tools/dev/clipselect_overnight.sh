#!/usr/bin/env bash
# Overnight response matrix for clip-subset selection (eval/nceval/clipselect.py):
# H.264 Fast, then AV1 Fast, each COUNT random trust-region perturbations on one
# clip of every pool family at one rung, then the SVD / subset report.
# Resumable: rerun the same command and each matrix continues where it stopped.
#
#   tools/dev/clipselect_overnight.sh BINARY [COUNT]
#
# Output: $NC_DATA_ROOT/tuning/clipselect/{h264,av1}-fast.{json,subsets.json,log}.
# About 16 perturbations x ~150 points per codec: 1.5-2 h each on a shared machine.
set -euo pipefail
binary=${1:?usage: clipselect_overnight.sh BINARY [COUNT]}
count=${2:-16}
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
out=${NC_DATA_ROOT:-$HOME/nanocodec-data}/tuning/clipselect
mkdir -p "$out"
cd "$root/eval"
for codec in h264 av1; do
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
    python3 -m nceval.clipselect measure --codec "$codec" --preset "$codec-fast" \
      --binary "$binary" --space "tune_spaces/$codec-fast.txt" --count "$count" \
      --out "$out/$codec-fast.json" >> "$out/$codec-fast.log" 2>&1
  python3 -m nceval.clipselect select --matrix "$out/$codec-fast.json" \
    --out "$out/$codec-fast.subsets.json" >> "$out/$codec-fast.log" 2>&1
done
