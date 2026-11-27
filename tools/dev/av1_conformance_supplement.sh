#!/usr/bin/env bash
# Supplementary AV1 Main-profile conformance streams from third-party
# encoders.
#
# The libaom test vectors (tools/ci/av1_conformance.sh) leave several Main
# profile tools without a stream: quantizer matrices, superblock loop
# filter deltas, segment features other than the quantizer (loop filter
# levels, reference, skip), lossless segments in lossy frames, odd frame
# sizes and switch frames. This script codes short streams that exercise
# them with libaom (aomenc, and libaom's svc_encoder_rtc example rebuilt
# with its ROI-map test switched by NC_ROI_FEATURE) and SVT-AV1, and keeps a
# stream only when aomdec and dav1d decode it to the same output (MD5 of
# every frame). Their aomdec output is the reference
# (tools/dev/conformance_manifest.py --suite av1-supplementary).
#
#   tools/dev/av1_conformance_supplement.sh SOURCE_YUV [out-dir]
#
# SOURCE_YUV: FourPeople_480x270_60.yuv (8-bit 4:2:0, public JCT-VC test
# sequence; ~/nanocodec-data/ctc-sources). Output:
# <out-dir>/<name>/<name>.ivf, default
# ~/nanocodec-data/conformance/av1/supplementary. Needs aomenc, aomdec,
# dav1d and SvtAv1EncApp in ~/nanocodec-data/anchors/x86_64/bin and the
# libaom v3.15.3 source (for the ROI encoder), cmake and a C++ compiler.
set -euo pipefail

source_yuv=${1:?usage: av1_conformance_supplement.sh SOURCE_YUV [out-dir]}
out=${2:-$HOME/nanocodec-data/conformance/av1/supplementary}
anchors=$HOME/nanocodec-data/anchors/x86_64
bin=$anchors/bin
libaom_src=$anchors/src/libaom-v3.15.3
source_size=(480 270)
frames=12
scratch=$(mktemp -d "${TMPDIR:-/tmp}/nanocodec-av1-conformance-supplement-XXXXXX")
trap 'rm -rf "$scratch"' EXIT

# The source cropped to WxH from its top-left corner (odd sizes included),
# as raw 8-bit 4:2:0 with ceil(W/2) x ceil(H/2) chroma.
crop() {
    local w=$1 h=$2 dst=$3
    python3 -I - "$source_yuv" "${source_size[@]}" "$w" "$h" "$frames" "$dst" <<'EOF'
import sys
src, sw, sh, w, h, frames, dst = sys.argv[1], *map(int, sys.argv[2:7]), sys.argv[7]
planes = [(sw, sh, w, h), ((sw + 1) // 2, (sh + 1) // 2, (w + 1) // 2, (h + 1) // 2),
          ((sw + 1) // 2, (sh + 1) // 2, (w + 1) // 2, (h + 1) // 2)]
frame_bytes = sum(pw * ph for pw, ph, _, _ in planes)
with open(src, "rb") as f, open(dst, "wb") as o:
    for _ in range(frames):
        data = f.read(frame_bytes)
        at = 0
        for pw, ph, cw, ch in planes:
            for y in range(ch):
                o.write(data[at + y * pw:at + y * pw + cw])
            at += pw * ph
EOF
}

# libaom's svc_encoder_rtc with its ROI-map test (segment 1 in the frame's
# middle) taking the feature from NC_ROI_FEATURE: 0 skip, 1 quantizer
# delta, 2 loop filter delta, 3 reference (GOLDEN).
build_roi_encoder() {
    local build=$scratch/aom-build
    cp -r "$libaom_src" "$scratch/aom-src"
    sed -i \
        -e 's/const int test_roi_map = 0;/const int test_roi_map = getenv("NC_ROI_FEATURE") != NULL;/' \
        -e 's/set_roi_map(&cfg, &codec, kDeltaQ);/set_roi_map(\&cfg, \&codec, atoi(getenv("NC_ROI_FEATURE")));/' \
        "$scratch/aom-src/examples/svc_encoder_rtc.cc"
    grep -q 'getenv("NC_ROI_FEATURE")' "$scratch/aom-src/examples/svc_encoder_rtc.cc"
    cmake -S "$scratch/aom-src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=0 \
        -DENABLE_DOCS=0 -DENABLE_TOOLS=0 -DCONFIG_AV1_DECODER=1 >/dev/null
    make -C "$build" -j"$(nproc)" svc_encoder_rtc >/dev/null
    roi_encoder=$(find "$build" -name svc_encoder_rtc -type f -perm -u+x | head -n 1)
    [[ -n $roi_encoder ]] || { echo "svc_encoder_rtc was not built" >&2; exit 1; }
}

# Keep $1 (an IVF) when aomdec and dav1d agree on every frame. 4:0:0 is
# compared as luma only (aomdec --rawvideo and dav1d's yuv output agree on
# that form).
check() {
    local name=$1 ivf=$2
    rm -f -- "$scratch/aom.yuv" "$scratch/dav1d.yuv"
    "$bin/aomdec" --rawvideo -o "$scratch/aom.yuv" "$ivf" || return 1
    "$bin/dav1d" -q -i "$ivf" -o "$scratch/dav1d.yuv" || return 1
    [[ -s $scratch/aom.yuv ]] || return 1
    if ! cmp -s "$scratch/aom.yuv" "$scratch/dav1d.yuv"; then
        echo "$name: aomdec and dav1d disagree: rejected" >&2
        return 1
    fi
    mkdir -p "$out/$name"
    cp "$ivf" "$out/$name/$name.ivf"
    echo "$name: $(stat -c %s "$ivf") bytes, aomdec == dav1d"
}

aom() {
    local name=$1 size=$2; shift 2
    local w=${size%x*} h=${size#*x}
    crop "$w" "$h" "$scratch/src.yuv"
    "$bin/aomenc" --quiet --limit="$frames" --width="$w" --height="$h" --i420 --fps=30/1 \
        --threads=1 --passes=1 --cpu-used=5 --end-usage=q --cq-level=32 "$@" \
        -o "$scratch/$name.ivf" "$scratch/src.yuv"
    check "$name" "$scratch/$name.ivf"
}

svt() {
    local name=$1 size=$2; shift 2
    local w=${size%x*} h=${size#*x}
    crop "$w" "$h" "$scratch/src.yuv"
    "$bin/SvtAv1EncApp" --width "$w" --height "$h" -i "$scratch/src.yuv" --frames "$frames" \
        --fps-num 30 --fps-denom 1 --preset 8 --crf 35 --lp 1 "$@" -b "$scratch/$name.ivf" >/dev/null 2>&1
    check "$name" "$scratch/$name.ivf"
}

# `roi NAME FEATURE KBPS ...`: FEATURE - for no ROI map.
roi() {
    local name=$1 feature=$2 rate=$3; shift 3
    crop "${source_size[@]}" "$scratch/src.yuv"
    local env=(env -u NC_ROI_FEATURE)
    [[ $feature != - ]] && env=(env NC_ROI_FEATURE="$feature")
    "${env[@]}" "$roi_encoder" --width="${source_size[0]}" --height="${source_size[1]}" \
        --spatial-layers=1 --temporal-layers=1 --layering-mode=0 --speed=7 \
        --target-bitrate="$rate" --bitrates="$rate" --test-decode=0 "$@" \
        -o "$scratch/$name.ivf" "$scratch/src.yuv" >/dev/null
    check "$name" "$scratch/$name.ivf"
}

# An encoder that refuses its options is reported and skipped; a stream the
# reference decoders disagree on is rejected (check).
stream() {
    "$@" || echo "$2: not made (exit $?)" >&2
}

mkdir -p "$out"
build_roi_encoder

full=${source_size[0]}x${source_size[1]}
# Quantizer matrices (§7.12.3): 8- and 10-bit, monochrome, every level.
stream aom aom_qm "$full" --enable-qm=1 --qm-min=0 --qm-max=15
stream aom aom_qm_10bit "$full" --enable-qm=1 --qm-min=0 --qm-max=8 --bit-depth=10
stream aom aom_qm_mono "$full" --enable-qm=1 --qm-min=2 --qm-max=12 --monochrome
stream aom aom_qm_chroma_deltaq "$full" --enable-qm=1 --enable-chroma-deltaq=1 --cq-level=24
stream svt svt_qm "$full" --enable-qm 1 --qm-min 0 --qm-max 15 --chroma-qm-min 0 --chroma-qm-max 12
# Superblock quantizer and loop filter deltas (§5.11.12–13, §7.14.4).
stream aom aom_deltaq "$full" --deltaq-mode=1 --enable-tpl-model=1 --lag-in-frames=12
stream aom aom_deltalf "$full" --deltaq-mode=1 --enable-tpl-model=1 --lag-in-frames=12 --delta-lf-mode=1
stream aom aom_deltalf_10bit "$full" --deltaq-mode=1 --enable-tpl-model=1 --lag-in-frames=12 \
    --delta-lf-mode=1 --bit-depth=10
stream aom aom_deltalf_mono "$full" --deltaq-mode=1 --enable-tpl-model=1 --lag-in-frames=12 \
    --delta-lf-mode=1 --monochrome
stream aom aom_deltaq_variance_boost "$full" --deltaq-mode=6 --deltaq-strength=200
stream svt svt_variance_boost "$full" --enable-variance-boost 1 --variance-boost-strength 3
# Segmentation: quantizer AQ modes (§5.9.14), and the ROI map's loop filter
# level, reference and skip features (§7.14.4, §5.11.10–11, §5.11.20,
# §5.11.25) through segment ids read before skip (SegIdPreSkip).
stream aom aom_aq_variance "$full" --aq-mode=1
stream aom aom_aq_complexity "$full" --aq-mode=2
stream aom aom_aq_cyclic "$full" --aq-mode=3 --end-usage=cbr --target-bitrate=300 --lag-in-frames=0
stream svt svt_aq_variance "$full" --aq-mode 1
stream roi rtc_roi_skip 0 400
stream roi rtc_roi_deltaq 1 400
stream roi rtc_roi_deltalf 2 400
stream roi rtc_roi_reference 3 400
# Lossless frames (§7.13.3). No encoder here makes lossless segments in a
# lossy frame (libaom turns AQ off at qindex 0, and its AQ and ROI deltas
# stop at qindex 1): the Argon streams cover them (tools/dev/av1_argon.py).
stream aom aom_lossless "$full" --lossless=1
# Odd frame sizes (§5.9.5, §7.11.3.3): luma and chroma edges, with and
# without superres, restoration and monochrome.
stream aom aom_odd_479x269 479x269
stream aom aom_odd_99x57 99x57 --cq-level=20
stream aom aom_odd_351x201_superres 351x201 --superres-mode=1 --superres-denominator=13
stream aom aom_odd_mono_10bit 99x57 --monochrome --bit-depth=10
stream svt svt_odd_479x269 479x269
# Switch frames and error resilience (§5.9.2, §7.20).
stream aom aom_sframe "$full" --sframe-dist=4 --sframe-mode=1 --lag-in-frames=0
stream aom aom_sframe_mode2 "$full" --sframe-dist=3 --sframe-mode=2 --lag-in-frames=0 --error-resilient=1
stream svt svt_sframe "$full" --sframe-dist 4 --sframe-mode 2
echo "done: $(find "$out" -name '*.ivf' | wc -l) streams in $out"
