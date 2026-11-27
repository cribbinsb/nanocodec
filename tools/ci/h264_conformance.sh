#!/usr/bin/env bash
# H.264 decoder conformance on the generic runner (tools/ci/conformance.py):
# the ITU-T H.264.1 suite (`h264`: AVCv1, FRExt, High 10; every stream of
# tools/ci/h264_conformance_passing.txt must match JM's output MD5), its
# 4:2:2 / 4:4:4 streams (`h264-chroma`, fetched with --suite h264-chroma:
# those of tools/ci/h264_conformance_chroma_passing.txt must match), the
# x264 streams for I/P-only High tools (`h264-x264`, all must match) and
# for 4:2:2, 4:4:4 and lossless coding (`h264-x264-chroma`, all must
# match), and the JM streams for Baseline / Extended tools the suite lacks
# (`h264-jm`, tools/dev/h264_conformance_jm.sh; those of
# tools/ci/h264_conformance_jm_passing.txt must match).
# Missing streams are a failure, not a skip (a check that does not run must not pass):
# fetch them with tools/dev/conformance_fetch.py --suite h264 (docs/handoff/h264.md).
#
#   tools/ci/h264_conformance.sh [conformance.py options, e.g. --only NAME]
#
# Exits non-zero when a required stream fails.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PATH="$HOME/.cargo/bin:$PATH"

data=${NC_DATA_ROOT:-$HOME/nanocodec-data}
if [[ ! -d $data/conformance/h264/streams ]]; then
    echo "h264 conformance: no streams under $data/conformance/h264: run tools/dev/conformance_fetch.py --suite h264" >&2
    exit 2
fi
target_dir=${CARGO_TARGET_DIR:-target}
cargo build --release --quiet -j "${BUILD_JOBS:-8}" -p nanocodec-cli
binary=$target_dir/release/nanocodec
python3 -I tools/ci/conformance.py --suite h264 --bin "$binary" --jobs "${JOBS:-4}" \
    --must-pass tools/ci/h264_conformance_passing.txt "$@"
if [[ -d $data/conformance/h264/supplementary ]]; then
    python3 -I tools/ci/conformance.py --suite h264-x264 --bin "$binary" --jobs "${JOBS:-4}" "$@"
fi
if [[ -d $data/conformance/h264/jm ]]; then
    python3 -I tools/ci/conformance.py --suite h264-jm --bin "$binary" --jobs "${JOBS:-4}" \
        --must-pass tools/ci/h264_conformance_jm_passing.txt "$@"
fi
if [[ -d $data/conformance/h264/streams/Hi422FR1_SONY_A ]]; then
    python3 -I tools/ci/conformance.py --suite h264-chroma --bin "$binary" --jobs "${JOBS:-4}" \
        --must-pass tools/ci/h264_conformance_chroma_passing.txt "$@"
fi
if [[ -d $data/conformance/h264/supplementary-chroma ]]; then
    python3 -I tools/ci/conformance.py --suite h264-x264-chroma --bin "$binary" \
        --jobs "${JOBS:-4}" "$@"
fi
