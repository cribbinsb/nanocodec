#!/usr/bin/env bash
# Repository checks: formatting, lints, tests, kernel bit-exactness, the WASM
# build, the crate layering rule and (full run only) the syntax-coverage
# stress test and the H.264 decoder conformance streams. Run before every commit:
#
#   tools/ci/check.sh            all checks
#   tools/ci/check.sh --quick    skip the WASM checkasm run, the syntax stress test and
#                                the H.264 conformance streams
#                                (the VP9/AV2/VVC browser-module smoke tests always run)
#   tools/ci/check.sh --full-only
#                                only the steps --quick leaves out (the nightly verifier,
#                                tools/dev/verify_schedule.py: the hourly verify runs --quick)
#   tools/ci/check.sh [--quick] [--crates ...] --preflight
#                                only list the tools those checks would need and lack
#   tools/ci/check.sh --quick --crates "nc-vvc nc-cli"
#                                clippy and test only these crates (land.sh --verify
#                                passes the changed crates plus their dependents); host
#                                checkasm runs only if nc-checkasm is listed and
#                                the browser module only if nc-web is
#
# NC_CHECKASM_REQUIRE=avx2,avx512: host checkasm fails when a listed tier cannot run
# here instead of skipping it (the verifiers set it for their machines).
#
# Exits non-zero on the first failure.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PATH="$HOME/.cargo/bin:$PATH"

quick=false full_only=false preflight_only=false
crates=""
while (($#)); do
    case $1 in
        --quick) quick=true ;;
        --full-only) full_only=true ;;
        --preflight) preflight_only=true ;;
        --crates) crates=${2:?--crates needs a list}; shift ;;
        *) printf 'check.sh: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
if $quick && $full_only; then
    echo 'check.sh: --quick and --full-only exclude each other' >&2; exit 2
fi
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

# Preflight: every tool the selected steps need, listed at once before minutes of
# building. Missing ones fail the run with NC_CHECK_STRICT=1 (land.sh's verify on the
# build box: a check that silently skips there is not a verify), else only warn
# (the page tests skip themselves without Playwright).
missing=()
need_target() {
    rustup target list --installed 2>/dev/null | grep -qx "$1" || missing+=("rust target $1")
}
command -v cargo >/dev/null || missing+=(cargo)
if selected nc-web && [[ "${NC_BROWSER_TESTS:-1}" == 1 ]]; then
    command -v node >/dev/null || missing+=(node)
    need_target wasm32-unknown-unknown
    bindgen=${WASM_BINDGEN:-$(command -v wasm-bindgen || echo "${NC_DATA_ROOT:-$HOME/nanocodec-data}/tools/browser-demo/wasm-bindgen-0.2.105-x86_64-unknown-linux-musl/wasm-bindgen")}
    [[ -x "$bindgen" && "$("$bindgen" --version 2>/dev/null)" == 'wasm-bindgen 0.2.105' ]] ||
        missing+=("wasm-bindgen-cli 0.2.105")
    if [[ "${NC_PAGE_TESTS:-1}" != 0 ]]; then
        python3 -c 'import playwright' 2>/dev/null || missing+=("python playwright (page tests)")
    fi
fi
if ! $quick; then
    command -v node >/dev/null || missing+=(node)
    need_target wasm32-wasip1
    need_target wasm32-unknown-unknown
    need_target aarch64-unknown-linux-gnu
fi
if ((${#missing[@]})); then
    printf 'check.sh preflight: missing: %s\n' "${missing[*]}" >&2
    [[ "${NC_CHECK_STRICT:-0}" != 1 ]] || exit 3
fi
if $preflight_only; then
    ((${#missing[@]})) && exit 3
    echo 'check.sh preflight: every tool is there'
    exit 0
fi

# The steps --quick runs too (all but the extras below).
quick_steps() {
    step "format"
    cargo fmt --all --check

    step "clippy (host): ${crates:-workspace}"
    cargo clippy "${selection[@]}" --all-targets -- -D warnings

    step "tests: ${crates:-workspace}"
    # gate-test (Cargo.toml): optimised like gate, with debug assertions and overflow
    # checks on (they catch a wrapped i16/u16 value). A one-crate change re-tests in
    # 43 s against 68 s on the opt-level-1 test profile (2026-11-15, -j8; 30 s with
    # the checks off).
    cargo test "${selection[@]}" --profile gate-test --quiet

    # land.sh sets NC_CHECKASM=0 when no kernel crate changed (tools/dev/land_scope.py).
    if selected nc-checkasm && [[ "${NC_CHECKASM:-1}" == 1 ]]; then
        step "checkasm (host SIMD vs scalar)"
        cargo run --profile gate --quiet -p nc-checkasm -- ${NC_CHECKASM_REQUIRE:+--require "$NC_CHECKASM_REQUIRE"}
        step "no SIMD code outside its #[target_feature] context (host checkasm)"
        python3 tools/ci/check_outlined_simd.py "$target_dir"/gate/nc-checkasm
    fi

    step "crate layering, integration-test modules, encode paths, AWS launchers, decoder register"
    python3 tools/ci/check_layering.py
    python3 tools/ci/check_test_modules.py
    python3 tools/ci/check_encode_paths.py
    python3 tools/ci/check_names.py
    python3 tools/ci/check_sealed_sets.py
    python3 tools/ci/check_aws_launchers.py
    python3 tools/ci/decoder_gaps.py

    # Runs in --quick too (land.sh), so browser API changes cannot land unchecked.
    # land.sh sets NC_BROWSER_TESTS=0 unless web/, nc-web or the kernels changed.
    if selected nc-web && [[ "${NC_BROWSER_TESTS:-1}" == 1 ]]; then
        step "browser module (VP9/AV2/VVC WASM smoke tests under node, product smoke in Chromium)"
        tools/ci/browser_smoke.sh
    fi
}

if ! $full_only; then
    quick_steps
fi

if ! $quick; then
    step "syntax stress (random legal decisions vs JM/HM/aomdec/dav1d/ffmpeg)"
    cargo build --release --quiet -p nanocodec-cli
    NC_NANOCODEC_BIN="$target_dir/release" python3 tools/ci/syntax_stress.py --quick

    step "clippy + build (wasm32, SIMD128)"
    export RUSTFLAGS="-C target-feature=+simd128"
    cargo clippy --workspace --all-targets --target wasm32-wasip1 -- -D warnings
    cargo build --release --quiet -p nanocodec-dsp --target wasm32-unknown-unknown
    cargo build --release --quiet -p nc-checkasm --target wasm32-wasip1
    unset RUSTFLAGS

    step "checkasm (SIMD128 under node)"
    node --no-warnings tools/ci/wasi-run.mjs "$target_dir"/wasm32-wasip1/release/nc-checkasm.wasm

    # The browser's second module: relaxed SIMD must match scalar exactly too.
    step "checkasm (SIMD128 + relaxed SIMD under node)"
    RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" CARGO_TARGET_DIR="$target_dir/relaxed-simd" \
        cargo build --release --quiet -p nc-checkasm --target wasm32-wasip1
    node --no-warnings tools/ci/wasi-run.mjs "$target_dir"/relaxed-simd/wasm32-wasip1/release/nc-checkasm.wasm
    python3 tools/ci/check_outlined_simd.py "$target_dir"/wasm32-wasip1/release/nc-checkasm.wasm \
        "$target_dir"/relaxed-simd/wasm32-wasip1/release/nc-checkasm.wasm

    # The decoder analysis builds (`nanocodec stat`, the demo's analysis
    # module): off in every normal build, so only the full run keeps them
    # compiling and tested (docs/handoff/stream-analysis.md).
    step "analysis feature (clippy + tests)"
    cargo clippy -p nanocodec-cli -p nanocodec-web --features analysis --all-targets -- -D warnings
    cargo test -p nanocodec-common -p nanocodec-hevc -p nanocodec-h264 -p nanocodec-av1 -p nanocodec-vp9 -p nanocodec-vvc -p nanocodec-av2 -p nanocodec-cli --features analysis --quiet

    step "clippy (aarch64 cross-check)"
    cargo clippy --workspace --all-targets --target aarch64-unknown-linux-gnu -- -D warnings

    step "H.264 decoder conformance (fails without the downloaded streams)"
    tools/ci/h264_conformance.sh
fi

step_done
printf '\nall checks passed\n'
