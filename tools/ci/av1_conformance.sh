#!/usr/bin/env bash
# AV1 decoder conformance: decode every libaom Main-profile test vector (the
# 176 av1-1-b8-* and 66 av1-1-b10-* vectors of libaom v3.15.3,
# tools/conformance/av1_manifest.txt) with nanocodec's own decoder and
# require its output (8-bit samples as bytes, deeper ones as 16-bit
# little-endian, monochrome with grey chroma, film grain applied) to be
# MD5-exact to aomdec's, whole and per picture, through the generic runner
# tools/ci/conformance.py (manifest tools/ci/av1_conformance_libaom.json).
#
# The vectors live outside the repository as untrusted data in
# ~/nanocodec-data/conformance/av1/streams (override: NC_AV1_CONFORMANCE);
# fetch them with tools/dev/conformance_fetch.py --suite av1-libaom and
# rebuild the manifest with tools/dev/conformance_manifest.py --suite
# av1-libaom. Uses NANOCODEC if set, else builds nc-cli in release mode.
# Nothing decoded is written to disk. JOBS (default 4, at most 8) sets the
# parallel decodes; extra arguments go to the runner (e.g. --max-bit-depth 8,
# --only av1-1-b10-24-monochrome).

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
STREAMS=${NC_AV1_CONFORMANCE:-$HOME/nanocodec-data/conformance/av1/streams}
if [[ ! -d $STREAMS ]]; then
    echo "missing $STREAMS: run tools/dev/conformance_fetch.py --suite av1-libaom" >&2
    exit 2
fi
if [[ -z ${NANOCODEC:-} ]]; then
    (cd "$REPO" && cargo build --release --quiet -p nanocodec-cli)
    NANOCODEC=${CARGO_TARGET_DIR:-$REPO/target}/release/nanocodec
fi
exec python3 -I "$REPO/tools/ci/conformance.py" --suite av1-libaom --bin "$NANOCODEC" \
    --streams "$STREAMS" --jobs "${JOBS:-4}" "$@"
