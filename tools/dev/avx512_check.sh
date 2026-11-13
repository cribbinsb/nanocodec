#!/usr/bin/env bash
# AVX-512 correctness only: SDE timings are not hardware performance evidence.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/nanocodec-data/targets/avx512}
sde=${NC_SDE:-$HOME/nanocodec-data/ext/sde/sde64}
work=${NC_AVX512_WORK:-$HOME/nanocodec-data/work/avx512-check}
jobs=${CARGO_BUILD_JOBS:-8}
[[ "$jobs" =~ ^[1-8]$ ]] || { echo "CARGO_BUILD_JOBS must be between 1 and 8" >&2; exit 1; }
baseline=()
[[ -z "${NC_AVX512_BASELINE:-}" ]] || baseline=(--baseline "$NC_AVX512_BASELINE")
[[ -x "$sde" ]] || { echo "Missing Intel SDE: set NC_SDE (see docs/handoff/avx512.md)" >&2; exit 1; }
# Each invocation is bounded separately; no emulated run exceeds 30 minutes.
timeout 1200 cargo build -j "$jobs" --profile gate -p nc-cli -p nc-checkasm
binary=$CARGO_TARGET_DIR/gate/nanocodec
checker=$CARGO_TARGET_DIR/gate/nc-checkasm
timeout 600 "$checker" --isa avx2
for model in spr icx; do
    seed=1
    [[ "$model" != icx ]] || seed=2
    timeout 1500 "$sde" "-$model" -- "$checker" --isa avx512 --seed "$seed"
done
for codec in h264 hevc av1 vp9 av2 vvc; do
    timeout 1500 python3 tools/dev/avx512_identity.py --binary "$binary" --sde "$sde" \
        --codec "$codec" --work "$work" "${baseline[@]}"
done
