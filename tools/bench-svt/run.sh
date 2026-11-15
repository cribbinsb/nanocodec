#!/usr/bin/env bash
# Build the SVT-AV1 kernel bench against the anchors' static library, run it
# and nc-checkasm's bench on one pinned core, and print both side by side.
#
#   tools/bench-svt/run.sh [core]        (default core 3)
#
# Run it on an otherwise idle machine: both benches report best-of-5 timings,
# but other load still skews them.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
arch=$(uname -m)
anchors=${NC_ANCHORS:-$HOME/nanocodec-data/anchors/$arch}
svt=$anchors/src/svtav1-v4.3.0
core=${1:-3}
out=${TMPDIR:-/tmp}/nc-bench-svt
mkdir -p "$out"

case $arch in
    x86_64) define=-DARCH_X86_64=1 ;;
    aarch64) define=-DARCH_AARCH64=1 ;;
    *) echo "unsupported architecture $arch" >&2; exit 1 ;;
esac

gcc -O2 -I"$svt/Source/API" -I"$svt/Source/Lib/Codec" $define \
    "$repo/tools/bench-svt/svt_bench.c" "$anchors/lib/libSvtAv1Enc.a" -lm -lpthread \
    -o "$out/svt_bench"
PATH=$HOME/.cargo/bin:$PATH cargo build --release --quiet -p nc-checkasm \
    --manifest-path "$repo/Cargo.toml"

# Alternate the two benches over several rounds; compare.py keeps each
# kernel's fastest time, which filters out interference from other load.
rounds=${ROUNDS:-3}
: > "$out/svt.txt"
: > "$out/nanocodec.txt"
for _ in $(seq "$rounds"); do
    taskset -c "$core" "$out/svt_bench" >> "$out/svt.txt"
    taskset -c "$core" "$repo/target/release/nc-checkasm" --bench >> "$out/nanocodec.txt"
done
python3 "$repo/tools/bench-svt/compare.py" "$out/svt.txt" "$out/nanocodec.txt"
