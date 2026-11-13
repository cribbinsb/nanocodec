#!/usr/bin/env bash
# Repository checks: formatting, lints, tests, kernel bit-exactness, the WASM
# build, the crate layering rule and (full run only) the syntax-coverage
# stress test and the H.264 decoder conformance streams. Run before every commit:
#
#   tools/ci/check.sh            all checks
#   tools/ci/check.sh --quick    skip the WASM checkasm run, the syntax stress test and
#                                the H.264 conformance streams
#                                (the VP9/AV2/VVC browser-module smoke tests always run)
#   tools/ci/check.sh --fast [--crates ...]
#                                formatting and clippy only (seconds to a
#                                minute): what land.sh runs before landing; the
#                                rest runs after, in `land.sh --verify`
#   tools/ci/check.sh --quick --crates "nc-vvc nc-cli"
#                                clippy and test only these crates (land.sh passes
#                                the changed crates plus their dependents); host
#                                checkasm runs only if nc-checkasm is listed and
#                                the browser module only if nc-web is
#
# Exits non-zero on the first failure.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PATH="$HOME/.cargo/bin:$PATH"

quick=false
fast=false
crates=""
while (($#)); do
    case $1 in
        --quick) quick=true ;;
        --fast) fast=true quick=true ;;
        --crates) crates=${2:?--crates needs a list}; shift ;;
        *) printf 'check.sh: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
if [[ -n "$crates" ]]; then
    $quick || { echo 'check.sh: --crates needs --quick' >&2; exit 2; }
    selection=()
    for name in $crates; do selection+=(-p "$name"); done
else
    selection=(--workspace)
fi
# Whether the selection includes crate $1 (the whole workspace always does).
selected() { [[ -z "$crates" || " $crates " == *" $1 "* ]]; }

# Each step prints the wall time of the one before it (land.sh's stage profile).
step_name="" step_began=$SECONDS
step_done() { [[ -z "$step_name" ]] || printf -- '-- %s: %ds\n' "$step_name" $((SECONDS - step_began)); }
step() { step_done; step_name=$*; step_began=$SECONDS; printf '\n== %s\n' "$*"; }

# Agents build into their own target directories (docs/PROCEDURES.md).
target_dir=${CARGO_TARGET_DIR:-target}

step "format"
cargo fmt --all --check

if $fast; then
    # clippy, not `cargo check`: the same compile, and a lint failure otherwise
    # lands and blocks the verified head for everyone (3537ac94, 2026-11-13).
    step "clippy (host): ${crates:-workspace}"
    cargo clippy "${selection[@]}" --all-targets --quiet -- -D warnings
    step_done
    exit 0
fi

step "clippy (host): ${crates:-workspace}"
cargo clippy "${selection[@]}" --all-targets -- -D warnings

step "tests: ${crates:-workspace}"
cargo test "${selection[@]}" --quiet

if selected nc-checkasm; then
    step "checkasm (host SIMD vs scalar)"
    cargo run --profile gate --quiet -p nc-checkasm
fi

step "crate layering, integration-test modules"
python3 tools/ci/check_layering.py
python3 tools/ci/check_test_modules.py

# Runs in --quick too (land.sh), so browser API changes cannot land unchecked.
if selected nc-web; then
    step "browser module (VP9/AV2/VVC WASM smoke tests under node, product smoke in Chromium)"
    tools/ci/browser_smoke.sh
fi

if ! $quick; then
    step "syntax stress (random legal decisions vs JM/HM/aomdec/dav1d/ffmpeg)"
    cargo build --release --quiet -p nc-cli
    NC_NANOCODEC_BIN="$target_dir/release" python3 tools/ci/syntax_stress.py --quick

    step "clippy + build (wasm32, SIMD128)"
    export RUSTFLAGS="-C target-feature=+simd128"
    cargo clippy --workspace --all-targets --target wasm32-wasip1 -- -D warnings
    cargo build --release --quiet -p nc-dsp --target wasm32-unknown-unknown
    cargo build --release --quiet -p nc-checkasm --target wasm32-wasip1
    unset RUSTFLAGS

    step "checkasm (SIMD128 under node)"
    node --no-warnings tools/ci/wasi-run.mjs "$target_dir"/wasm32-wasip1/release/nc-checkasm.wasm

    # The browser's second module: relaxed SIMD must match scalar exactly too.
    step "checkasm (SIMD128 + relaxed SIMD under node)"
    RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" CARGO_TARGET_DIR="$target_dir/relaxed-simd" \
        cargo build --release --quiet -p nc-checkasm --target wasm32-wasip1
    node --no-warnings tools/ci/wasi-run.mjs "$target_dir"/relaxed-simd/wasm32-wasip1/release/nc-checkasm.wasm

    # The decoder analysis builds (`nanocodec stat`, the demo's analysis
    # module): off in every normal build, so only the full run keeps them
    # compiling and tested (docs/handoff/stream-analysis.md).
    step "analysis feature (clippy + tests)"
    cargo clippy -p nc-cli -p nc-web --features analysis --all-targets -- -D warnings
    cargo test -p nc-common -p nc-hevc -p nc-h264 -p nc-av1 -p nc-vp9 -p nc-cli --features analysis --quiet

    step "clippy (aarch64 cross-check)"
    cargo clippy --workspace --all-targets --target aarch64-unknown-linux-gnu -- -D warnings

    step "H.264 decoder conformance (skipped without the downloaded streams)"
    tools/ci/h264_conformance.sh
fi

step_done
printf '\nall checks passed\n'
