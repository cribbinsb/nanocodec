#!/usr/bin/env bash
# Official H.266.1 8-bit Main decoder gate; decoded pixels remain in pipes.
# Fetch with tools/dev/vvc_conformance_fetch.py. Missing data is an error.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PATH="$HOME/.cargo/bin:$PATH"
if [[ -z ${NANOCODEC:-} ]]; then
    (cd "$REPO" && cargo build --release --quiet -j4 -p nc-cli)
    NANOCODEC=${CARGO_TARGET_DIR:-$REPO/target}/release/nanocodec
fi
exec python3 -I "$REPO/tools/ci/vvc_conformance.py" --bin "$NANOCODEC" \
    --streams "${NC_VVC_CONFORMANCE:-$HOME/nanocodec-data/conformance/vvc/streams}" "$@"
