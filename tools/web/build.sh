#!/usr/bin/env bash
# Build the real browser codec module. Generated output is not source-controlled.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export PATH="$HOME/.cargo/bin:$PATH"
# The pinned binary under the data root is used when none is on PATH.
pinned_bindgen=${NC_DATA_ROOT:-$HOME/nanocodec-data}/tools/browser-demo/wasm-bindgen-0.2.105-x86_64-unknown-linux-musl/wasm-bindgen
if [[ -z "${WASM_BINDGEN:-}" ]] && ! command -v wasm-bindgen >/dev/null 2>&1 && [[ -x "$pinned_bindgen" ]]; then
  WASM_BINDGEN=$pinned_bindgen
fi
bindgen=${WASM_BINDGEN:-wasm-bindgen}
if ! command -v "$bindgen" >/dev/null 2>&1; then
  echo 'Install wasm-bindgen-cli 0.2.105, or set WASM_BINDGEN to its executable.' >&2
  echo 'cargo install wasm-bindgen-cli --version 0.2.105 --locked' >&2
  exit 1
fi
if [[ "$($bindgen --version)" != 'wasm-bindgen 0.2.105' ]]; then
  echo 'The browser crate requires wasm-bindgen-cli 0.2.105 (matching Cargo.lock).' >&2
  exit 1
fi
# Two modules: plain SIMD128, and SIMD128 plus relaxed SIMD in pkg/relaxed/
# (the page loads it where the engine validates relaxed SIMD; nc-dsp uses
# only relaxed operations whose results are fully specified, so both give
# identical output). Each has its own target directory, as the flags differ.
base_flags="${RUSTFLAGS:-} -C link-arg=-zstack-size=8388608"
target_root=${CARGO_TARGET_DIR:-$repo_root/target}
# PKG_DIR builds elsewhere (e.g. a benchmark A/B) without replacing web/pkg.
pkg_dir=${PKG_DIR:-web/pkg}
# WEB_PROFILE=gate (tools/ci/browser_smoke.sh) builds without whole-program
# LTO: the same behaviour, a much faster rebuild, a larger module.
profile=${WEB_PROFILE:-release}
build() { # build TARGET_DIR FEATURES OUT_DIR [CARGO_FEATURES]
  RUSTFLAGS="$base_flags -C target-feature=$2" CARGO_TARGET_DIR=$1 \
    cargo build --locked --profile "$profile" -p nc-web --target wasm32-unknown-unknown -j "${BUILD_JOBS:-2}" \
    ${4:+--features "$4"}
  mkdir -p "$3"
  "$bindgen" "$1/wasm32-unknown-unknown/$profile/nc_web.wasm" --target web --out-dir "$3" --out-name nc_web
}
build "$target_root" +simd128 "$pkg_dir"
build "$target_root/relaxed-simd" +simd128,+relaxed-simd "$pkg_dir/relaxed"
# The analysis module (decoders that record every block, web/analysis-view.js):
# loaded only when a viewer switches analysis on, so one SIMD128 build.
build "$target_root/analysis" +simd128 "$pkg_dir/analysis" analysis
printf 'Built %s, %s/relaxed and %s/analysis.' "$pkg_dir" "$pkg_dir" "$pkg_dir"
printf ' Serve with: python3 -m http.server 8765 --bind 127.0.0.1 --directory web\n'
