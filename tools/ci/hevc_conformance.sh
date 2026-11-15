#!/usr/bin/env bash
# HEVC decoder conformance: decode every Main, Main Still Picture and Main 10
# stream of the JCT-VC HEVC_v1 conformance set (ITU-T H.265.1) with
# nanocodec's own decoder and require its output (output order, conformance
# window applied; 16-bit little-endian samples beyond 8 bits) to be
# MD5-exact to HM's, whole and per picture: all 147 streams of
# tools/ci/hevc_conformance_v1.json, through the generic runner
# tools/ci/conformance.py.
#
# The streams live outside the repository as untrusted data in
# ~/nanocodec-data/conformance/hevc/streams (override: NC_HEVC_CONFORMANCE);
# fetch them with tools/dev/conformance_fetch.py --suite hevc-v1 and rebuild
# the manifest with tools/dev/conformance_manifest.py --suite hevc-v1.
# Uses NANOCODEC if set, else builds nc-cli in release mode. Nothing decoded
# is written to disk. JOBS (default 4, at most 8) sets the parallel decodes;
# extra arguments go to the runner (e.g. --profiles "Main 10").

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
STREAMS=${NC_HEVC_CONFORMANCE:-$HOME/nanocodec-data/conformance/hevc/streams}
if [[ ! -d $STREAMS ]]; then
    echo "missing $STREAMS: run tools/dev/conformance_fetch.py --suite hevc-v1" >&2
    exit 2
fi
if [[ -z ${NANOCODEC:-} ]]; then
    (cd "$REPO" && cargo build --release --quiet -p nanocodec-cli)
    NANOCODEC=${CARGO_TARGET_DIR:-$REPO/target}/release/nanocodec
fi
exec python3 -I "$REPO/tools/ci/conformance.py" --suite hevc-v1 --bin "$NANOCODEC" \
    --streams "$STREAMS" --jobs "${JOBS:-4}" "$@"
