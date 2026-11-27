#!/usr/bin/env bash
# Build the browser module (tools/web/build.sh) and run its VP9/AV2/VVC WASM
# smoke tests, the page's module unit tests and the pre-encoded reference
# streams' WASM decode check under node, then the product smoke in headless Chromium
# (web/tests/browser-product-smoke.py: the demo page end to end, workers,
# pipelining, display), the decode worker's warm-up interrupt test
# (web/tests/warmup-interrupt.py), source switches and webcam starts under CPU
# throttling (web/tests/source-switch.py), a pre-encoded view's PSNR equal to the
# offline value frame by frame on a busy page (web/tests/reference-psnr.py), a sample-clip
# comparison byte-identical on a slowed page (web/tests/exact-source.py), the heaviest
# clip playing without long pauses (web/tests/clip-pauses.py), a starved
# webcam holding its bitrate with counted skips (web/tests/live-skip.py), a one-CPU
# machine offered a lighter size that speeds it up (web/tests/speed-hint.py), the analysis mode
# (web/tests/analysis-smoke.py), its PSNR map and per-plane PSNR equal to an offline
# native decode against the demo's source frames, for a pre-encoded and a live view,
# on a normal and a slowed page (web/tests/psnr-map.py), what a live view codes against the
# clip file's decoded planes on three clips (web/tests/capture-identity.py; run
# it without --clips for every clip) and spatial SVC's operating points against
# the native decoder (web/tests/svc-operating-points.py), and the debug report reproduced
# natively with zero mismatches (web/tests/debug-report.py, through its dialog with and without the
# coded streams), the still-camera toggle reaching the live encoders on and off, each report
# replaying natively (web/tests/still-camera.py), an Original view's pixels equal to the source
# frame and its report replaying (web/tests/original-view.py), the comparison restarting by itself
# on every option and once per burst of changes under load (web/tests/auto-restart.py), and the
# layout control with the shared zoom and pan in each layout, with and without analysis
# (web/tests/view-layout.py), and the usage pages with their browser example
# (web/tests/docs-pages.py). nc-web embeds the encoder and every codec crate, so
# a change there can break the page (frame hold once stopped it: the
# workers expected a packet per input). Called by check.sh (every run, --quick included, so
# browser API changes cannot land unchecked) and by land.sh for a candidate
# that changes only web/. It builds with the gate profile (Cargo.toml: no
# fat LTO, so rebuilds are fast) unless WEB_PROFILE says otherwise, and
# web/pkg then holds that build: run tools/web/build.sh before serving or
# timing the browser module.
# Not run here: web/tests/clip-cadence.py (one distinct frame per 30 Hz tick in every
# sample clip) takes 6-10 minutes and only checks the clip files, so run it by hand
# after adding or retiming a clip (tools/web/make-mot-clips.py and the like).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.cargo/bin:$PATH"
WEB_PROFILE=${WEB_PROFILE:-gate} BUILD_JOBS=${BUILD_JOBS:-8} tools/web/build.sh
node --no-warnings web/tests/vp9-wasm-smoke.mjs
node --no-warnings web/tests/av2-wasm-smoke.mjs
node --no-warnings web/tests/vvc-wasm-smoke.mjs
# Display colours: the JS conversion equals nc-dsp's for every matrix.
node --no-warnings web/tests/colour-wasm.mjs
# The page's own modules (timeline, pre-encoded stream manifest and loader, containers).
node --test web/tests/*.test.js
# Every pre-encoded reference stream decodes bit-exactly in all three WASM builds.
node --no-warnings web/tests/reference-streams-wasm.mjs
# The official conformance corpora (where fetched) through all three WASM builds.
tools/ci/wasm_conformance.sh
# The page tests need Playwright's Chromium (this PC; not the Jetson). land.sh sets
# NC_PAGE_TESTS=0 for a change that does not touch the page (web/, crates/nc-web):
# the node tests above still cover the module, and tools/web/publish.sh runs the
# page tests before a build goes live.
if [[ "${NC_PAGE_TESTS:-1}" == 0 ]]; then
    echo 'page tests in Chromium skipped: the page did not change (NC_PAGE_TESTS=0)' >&2
elif python3 -c 'import playwright' 2>/dev/null; then
    # Side by side, against the module built above, only the tests the changes
    # since NC_WEB_TESTS_SINCE (land.sh sets it) can affect: tools/ci/web_tests.py.
    # The native-replay tests compare against our native decoder, built here with
    # the same profile as the browser module (NC_NANOCODEC_BIN overrides it).
    if [[ -z "${NC_NANOCODEC_BIN:-}" ]] && python3 tools/ci/web_tests.py --needs-native; then
        cargo build --frozen --profile "${WEB_PROFILE:-gate}" -p nanocodec-cli -j "${BUILD_JOBS:-8}"
        export NC_NANOCODEC_BIN=${CARGO_TARGET_DIR:-target}/${WEB_PROFILE:-gate}/nanocodec
    fi
    python3 tools/ci/web_tests.py
else
    echo 'browser product smoke skipped: Playwright is not installed' >&2
fi
