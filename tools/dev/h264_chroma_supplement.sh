#!/usr/bin/env bash
# Supplementary 4:2:2 and 4:4:4 H.264 streams (suite `h264-x264-chroma`).
#
# Every 4:4:4 stream of the ITU-T H.264.1 suite codes its colour planes
# separately, and its 4:2:2 ones are 8-bit and 10-bit High 4:2:2 only, so
# 4:4:4 with Cb and Cr coded like luma (CAVLC and CABAC, 8x8 transforms,
# B slices, weighted prediction), lossless transform bypass in every chroma
# format, and odd 4:4:4 sizes have no official stream. This script makes
# small x264 streams of them from a project clip. Only the streams are kept:
# tools/dev/conformance_manifest.py --suite h264-x264-chroma records JM's
# output MD5s (and ffmpeg's as a cross-check), no decoded YUV is written.
#
#   tools/dev/h264_chroma_supplement.sh [out-dir]
#
# Output: <out-dir>/<name>/<name>.264, default
# ~/nanocodec-data/conformance/h264/supplementary-chroma.
set -euo pipefail

out=${1:-$HOME/nanocodec-data/conformance/h264/supplementary-chroma}
bin=$HOME/nanocodec-data/anchors/x86_64/bin
clip=$HOME/nanocodec-data/clips/quick/crowd_mot17_09.yuv
clip_size=1280x720
frames=12
scratch=$(mktemp -d "${TMPDIR:-/tmp}/nanocodec-h264-chroma-supplement-XXXXXX")
trap 'rm -rf "$scratch"' EXIT

# Scaling lists with distinct Cb and Cr 8x8 lists (4:4:4 codes all six
# 8x8 lists, H.264 §7.3.2.1.1).
cat >"$scratch/chroma.cqm" <<'EOF'
INTRA4X4_LUMA =
6,13,20,28,13,20,28,32,20,28,32,37,28,32,37,42
INTRA4X4_CHROMAU =
8,14,21,29,14,21,29,33,21,29,33,38,29,33,38,43
INTRA4X4_CHROMAV =
9,15,22,30,15,22,30,34,22,30,34,39,30,34,39,44
INTER4X4_LUMA =
10,14,20,24,14,20,24,27,20,24,27,30,24,27,30,34
INTER4X4_CHROMAU =
11,15,21,25,15,21,25,28,21,25,28,31,25,28,31,35
INTER4X4_CHROMAV =
12,16,22,26,16,22,26,29,22,26,29,32,26,29,32,36
INTRA8X8_LUMA =
6,10,13,16,18,23,25,27,10,11,16,18,23,25,27,29,13,16,18,23,25,27,29,31,16,18,23,25,27,29,31,33,18,23,25,27,29,31,33,36,23,25,27,29,31,33,36,38,25,27,29,31,33,36,38,40,27,29,31,33,36,38,40,42
INTER8X8_LUMA =
9,13,15,17,19,21,22,24,13,13,17,19,21,22,24,25,15,17,19,21,22,24,25,27,17,19,21,22,24,25,27,28,19,21,22,24,25,27,28,30,21,22,24,25,27,28,30,32,22,24,25,27,28,30,32,33,24,25,27,28,30,32,33,35
INTRA8X8_CHROMAU =
7,11,14,17,19,24,26,28,11,12,17,19,24,26,28,30,14,17,19,24,26,28,30,32,17,19,24,26,28,30,32,34,19,24,26,28,30,32,34,37,24,26,28,30,32,34,37,39,26,28,30,32,34,37,39,41,28,30,32,34,37,39,41,43
INTER8X8_CHROMAU =
10,14,16,18,20,22,23,25,14,14,18,20,22,23,25,26,16,18,20,22,23,25,26,28,18,20,22,23,25,26,28,29,20,22,23,25,26,28,29,31,22,23,25,26,28,29,31,33,23,25,26,28,29,31,33,34,25,26,28,29,31,33,34,36
INTRA8X8_CHROMAV =
8,12,15,18,20,25,27,29,12,13,18,20,25,27,29,31,15,18,20,25,27,29,31,33,18,20,25,27,29,31,33,35,20,25,27,29,31,33,35,38,25,27,29,31,33,35,38,40,27,29,31,33,35,38,40,42,29,31,33,35,38,40,42,44
INTER8X8_CHROMAV =
11,15,17,19,21,23,24,26,15,15,19,21,23,24,26,27,17,19,21,23,24,26,27,29,19,21,23,24,26,27,29,30,21,23,24,26,27,29,30,32,23,24,26,27,29,30,32,34,24,26,27,29,30,32,34,35,26,27,29,30,32,34,35,37
EOF

common=(--frames "$frames" --fps 30 --threads 1 --no-progress --quiet --keyint 6)

# name | size | chroma format | bit depth | x264 options
streams=(
  "x264_422_cabac_8x8_bframes|352x288|i422|8|--crf 26 --8x8dct --bframes 2 --b-pyramid none --ref 3 --weightp 2 --slices 2"
  "x264_422_cavlc_10bit|352x288|i422|10|--crf 24 --no-cabac --8x8dct --bframes 1 --ref 2 --chroma-qp-offset 2"
  "x264_444_cabac_8x8_bframes|352x288|i444|8|--crf 26 --8x8dct --bframes 2 --b-pyramid none --ref 3 --weightp 2 --direct temporal"
  "x264_444_cavlc_slices|352x288|i444|8|--crf 28 --no-cabac --8x8dct --ref 2 --slices 3 --chroma-qp-offset -3"
  "x264_444_cabac_10bit_cqm|352x288|i444|10|--crf 24 --8x8dct --bframes 1 --cqmfile $scratch/chroma.cqm"
  "x264_444_cabac_odd_size|351x199|i444|8|--crf 26 --8x8dct --bframes 1 --constrained-intra"
  "x264_444_lossless_cabac|352x288|i444|8|--qp 0 --8x8dct --bframes 1 --ref 2"
  "x264_444_lossless_cavlc_10bit|352x288|i444|10|--qp 0 --no-cabac --8x8dct"
  "x264_422_lossless_cabac|352x288|i422|8|--qp 0 --8x8dct --bframes 1"
  "x264_420_lossless_cavlc|352x288|i420|8|--qp 0 --no-cabac --8x8dct --bframes 1"
)

mkdir -p "$out"
for entry in "${streams[@]}"; do
    IFS='|' read -r name size csp depth options <<<"$entry"
    dir=$out/$name
    mkdir -p "$dir"
    # x264 is built without swscale: the source is made in the output format.
    pix_fmt=yuv${csp#i}p
    [[ $depth == 10 ]] && pix_fmt=${pix_fmt}10le
    ffmpeg -y -hide_banner -loglevel error -nostdin -f rawvideo -pix_fmt yuv420p -s "$clip_size" \
        -i "$clip" -frames:v "$frames" -vf "scale=${size/x/:}:flags=bicubic" -f rawvideo \
        -pix_fmt "$pix_fmt" "$scratch/src.yuv"
    # shellcheck disable=SC2086 # options are a word list by design
    "$bin/x264" "${common[@]}" --input-res "$size" --input-csp "$csp" --output-csp "$csp" \
        --input-depth "$depth" --output-depth "$depth" $options -o "$dir/$name.264" "$scratch/src.yuv"
    echo "$name: $(stat -c %s "$dir/$name.264") bytes"
done
