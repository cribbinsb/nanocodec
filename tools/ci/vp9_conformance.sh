#!/usr/bin/env bash
# VP9 decoder conformance: decode every libvpx v1.16.0 profile 0 and
# profile 2 test vector (the 303 vp90-2-* and the 2 vp92-2-* 10/12-bit
# 4:2:0 vectors, tools/conformance/vp9_manifest.txt) with nanocodec's own
# decoder and require its output (8-bit samples as bytes, deeper ones as
# 16-bit little-endian) to be MD5-exact to vpxdec's (built with high bit
# depth: tools/anchors/build_anchors.sh vpxhbd), whole and, where the
# picture size is constant, per picture, through the generic runner
# tools/ci/conformance.py (manifest tools/ci/vp9_conformance_libvpx.json).
# Profiles 1 and 3 (4:4:4, 4:2:2, 4:4:0) are not supported.
#
# The vectors live outside the repository as untrusted data in
# ~/nanocodec-data/conformance/vp9/streams (override: NC_VP9_CONFORMANCE);
# fetch them with tools/dev/conformance_fetch.py --suite vp9-libvpx and
# rebuild the manifest with tools/dev/conformance_manifest.py --suite
# vp9-libvpx --profiles "Profile 0,Profile 2" --max-bit-depth 12. Uses
# NANOCODEC if set, else builds nc-cli in release mode. Nothing decoded is
# written to disk. JOBS (default 4, at most 8) sets the parallel decodes;
# extra arguments go to the runner (e.g. --max-bit-depth 8,
# --only vp92-2-20-10bit-yuv420).

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
STREAMS=${NC_VP9_CONFORMANCE:-$HOME/nanocodec-data/conformance/vp9/streams}
if [[ ! -d $STREAMS ]]; then
    echo "missing $STREAMS: run tools/dev/conformance_fetch.py --suite vp9-libvpx" >&2
    exit 2
fi
if [[ -z ${NANOCODEC:-} ]]; then
    (cd "$REPO" && cargo build --release --quiet -p nanocodec-cli)
    NANOCODEC=${CARGO_TARGET_DIR:-$REPO/target}/release/nanocodec
fi
exec python3 -I "$REPO/tools/ci/conformance.py" --suite vp9-libvpx --bin "$NANOCODEC" \
    --streams "$STREAMS" --jobs "${JOBS:-4}" "$@"
