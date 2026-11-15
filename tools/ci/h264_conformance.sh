#!/usr/bin/env bash
# H.264 decoder conformance on the generic runner (tools/ci/conformance.py):
# the ITU-T H.264.1 suite (`h264`: AVCv1, FRExt, High 10; every stream of
# tools/ci/h264_conformance_passing.txt must match JM's output MD5) and the
# x264 streams for I/P-only High tools (`h264-x264`, all must match).
# Optional: skipped when the streams are not downloaded
# (tools/dev/conformance_fetch.py --suite h264; docs/handoff/h264.md).
#
#   tools/ci/h264_conformance.sh [conformance.py options, e.g. --only NAME]
#
# Exits non-zero when a required stream fails.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PATH="$HOME/.cargo/bin:$PATH"

data=${NC_DATA_ROOT:-$HOME/nanocodec-data}
if [[ ! -d $data/conformance/h264/streams ]]; then
    echo "h264 conformance: no streams under $data/conformance/h264; skipped"
    exit 0
fi
target_dir=${CARGO_TARGET_DIR:-target}
cargo build --release --quiet -j "${BUILD_JOBS:-8}" -p nanocodec-cli
binary=$target_dir/release/nanocodec
python3 -I tools/ci/conformance.py --suite h264 --bin "$binary" --jobs "${JOBS:-4}" \
    --must-pass tools/ci/h264_conformance_passing.txt "$@"
if [[ -d $data/conformance/h264/supplementary ]]; then
    python3 -I tools/ci/conformance.py --suite h264-x264 --bin "$binary" --jobs "${JOBS:-4}" "$@"
fi
