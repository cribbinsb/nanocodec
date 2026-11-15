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
# Both carry the decoders' analysis records (nc-web `analysis`, switched on
# at run time by `set_analysis`; web/analysis-view.js): off, they cost one
# branch per block (decode speed within noise, docs/handoff/browser-demo.md
# "Build"), where a third, analysis-only module cost a third build per
# publish and a second 8 MB download for its viewers.
base_flags="${RUSTFLAGS:-} -C link-arg=-zstack-size=8388608"
target_root=${CARGO_TARGET_DIR:-$repo_root/target}
# PKG_DIR builds elsewhere (e.g. a benchmark A/B) without replacing web/pkg.
pkg_dir=${PKG_DIR:-web/pkg}
# `web` (Cargo.toml) is the published build; WEB_PROFILE=gate
# (tools/ci/browser_smoke.sh) builds without LTO: the same behaviour, a faster
# rebuild, a larger and slower module.
profile=${WEB_PROFILE:-web}
# The two builds run side by side, each with half the jobs: most of a build's
# wall time is its single-threaded LTO and code generation tail.
jobs=$(( (${BUILD_JOBS:-2} + 1) / 2 ))
build() { # build TARGET_DIR FEATURES OUT_DIR
  RUSTFLAGS="$base_flags -C target-feature=$2" CARGO_TARGET_DIR=$1 \
    cargo build --locked --profile "$profile" -p nc-web --target wasm32-unknown-unknown -j "$jobs" \
    --features analysis
  mkdir -p "$3"
  "$bindgen" "$1/wasm32-unknown-unknown/$profile/nc_web.wasm" --target web --out-dir "$3" --out-name nc_web
}
mkdir -p "$target_root"
build "$target_root" +simd128 "$pkg_dir" > "$target_root/web-simd128.log" 2>&1 & simd128=$!
build "$target_root/relaxed-simd" +simd128,+relaxed-simd "$pkg_dir/relaxed" > "$target_root/web-relaxed.log" 2>&1 & relaxed=$!
failed=0
for name in simd128 relaxed; do
  if ! wait "${!name}"; then
    echo "The $name module failed to build ($target_root/web-$name.log):" >&2
    tail -n 40 "$target_root/web-$name.log" >&2
    failed=1
  fi
done
[[ $failed == 0 ]] || exit 1
# The commit the modules were built from, which the page's debug report names
# (web/debug-report.js); "dirty": the crates or the page had local changes.
dirty=false
git diff --quiet HEAD -- crates web Cargo.toml Cargo.lock || dirty=true
printf '{"commit": "%s", "dirty": %s, "profile": "%s"}\n' "$(git rev-parse HEAD)" "$dirty" "$profile" > "$pkg_dir/build.json"
printf 'Built %s and %s/relaxed.' "$pkg_dir" "$pkg_dir"
printf ' Serve with: python3 -m http.server 8765 --bind 127.0.0.1 --directory web\n'
