#!/usr/bin/env bash
# The official H.264, HEVC, VVC, AV1 and VP9 conformance corpora through the browser
# module, both builds (web/pkg: SIMD128; web/pkg/relaxed: plus relaxed SIMD),
# under node, split into per-picture packets (IVF frames, or Annex-B access
# units) as the page's tools split them
# (tools/web/conformance-runner.mjs has the comparison). The native corpus runs
# (tools/ci/*_conformance.sh) cover the decoders themselves; this
# covers the browser entry point and the Annex-B splitter, where a one-packet-
# per-slice split once failed DELTAQP_B_SONY_3 (four slices per picture).
#
# Run after tools/web/build.sh. H.264 must pass the streams the native decoder
# must (tools/ci/h264_conformance_passing.txt), the others all. A corpus not fetched
# (tools/dev/conformance_fetch.py) fails the run (a suite that does not run must not pass). JOBS (default 8)
# sets the parallel decodes; extra arguments go to every runner (e.g. --only).
# JSON_DIR, when set, receives one result file per build and corpus
# (<build>-<codec>.json, the runner's --json).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${NC_DATA_ROOT:-$HOME/nanocodec-data}/conformance
RUNNER=$REPO/tools/web/conformance-runner.mjs
status=0
for pkg in "$REPO/web/pkg" "$REPO/web/pkg/relaxed"; do
    for suite in "h264 h264_conformance.json --must-pass $REPO/tools/ci/h264_conformance_passing.txt" \
                 "hevc hevc_conformance_v1.json" \
                 "vvc vvc_conformance_v1.json" \
                 "av1 av1_conformance_libaom.json" \
                 "vp9 vp9_conformance_libvpx.json"; do
        read -r codec manifest extra <<< "$suite"
        if [[ ! -d $DATA/$codec/streams ]]; then
            echo "FAILED $codec: no corpus in $DATA/$codec/streams (tools/dev/conformance_fetch.py)" >&2
            status=1
            continue
        fi
        json=()
        if [[ -n ${JSON_DIR:-} ]]; then
            json=(--json "$JSON_DIR/$(basename "$pkg")-$codec.json")
        fi
        # shellcheck disable=SC2086  # $extra is a flag and its value
        node --no-warnings "$RUNNER" --pkg "$pkg" --codec "$codec" \
            --manifest "$REPO/tools/ci/$manifest" --streams "$DATA/$codec/streams" \
            --jobs "${JOBS:-8}" $extra "${json[@]}" "$@" || status=1
    done
done
exit $status
