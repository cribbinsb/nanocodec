#!/usr/bin/env bash
# H.264 decoder conformance: decode every Constrained High stream of the
# manifest (tools/ci/h264_conformance.json: the ITU-T H.264.1 streams of that
# profile plus x264 streams for the High-only tools) and compare each frame
# with the reference YUV (MD5). Optional: skipped when the streams are not
# downloaded (see docs/handoff/h264.md, "Decoder conformance").
#
#   tools/ci/h264_conformance.sh [--only REGEX]
#
# Exits non-zero when any stream fails.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PATH="$HOME/.cargo/bin:$PATH"

root=${NC_H264_CONFORMANCE:-$HOME/nanocodec-data/conformance/h264}
if [[ ! -d $root/streams || ! -d $root/supplementary ]]; then
    echo "h264 conformance: no streams under $root; skipped"
    exit 0
fi
target_dir=${CARGO_TARGET_DIR:-target}
cargo build --release --quiet -j "${BUILD_JOBS:-8}" -p nc-cli
python3 -I tools/ci/h264_conformance.py --bin "$target_dir/release" --root "$root" \
    --jobs "${JOBS:-4}" "$@"
