#!/usr/bin/env bash
# HEVC decoder conformance: decode every Main / Main Still Picture stream of
# the JCT-VC HEVC_v1 conformance set (ITU-T H.265.1) with nanocodec's own
# decoder and require the MD5 of its output (output order, conformance window
# applied) to equal HM's, for all 136 streams of tools/ci/hevc_conformance_main.json.
#
# The streams live outside the repository as untrusted data in
# ~/nanocodec-data/conformance/hevc/streams (override: NC_HEVC_CONFORMANCE);
# fetch them and rebuild the manifest with tools/dev/hevc_conformance_fetch.py.
# Uses NANOCODEC if set, else builds nc-cli in release mode. Nothing decoded
# is written to disk. JOBS (default 4, at most 8) sets the parallel decodes.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
STREAMS=${NC_HEVC_CONFORMANCE:-$HOME/nanocodec-data/conformance/hevc/streams}
if [[ ! -d $STREAMS ]]; then
    echo "missing $STREAMS: run tools/dev/hevc_conformance_fetch.py" >&2
    exit 2
fi
if [[ -z ${NANOCODEC:-} ]]; then
    (cd "$REPO" && cargo build --release --quiet -p nc-cli)
    NANOCODEC=${CARGO_TARGET_DIR:-$REPO/target}/release/nanocodec
fi
exec python3 -I "$REPO/tools/ci/hevc_conformance.py" --bin "$NANOCODEC" \
    --streams "$STREAMS" --jobs "${JOBS:-4}" "$@"
