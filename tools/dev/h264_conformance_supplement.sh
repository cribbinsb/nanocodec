#!/usr/bin/env bash
# Supplementary Constrained High conformance streams.
#
# The ITU-T H.264.1 suite has no progressive High-profile stream without B
# slices, so the High-only tools of Constrained High (8x8 transform and
# intra 8x8, scaling matrices, 4:0:0, CAVLC with 8x8 transforms, separate Cb/Cr
# QP offsets) have no official I/P-only coverage. This script makes small
# x264 streams that exercise them and the official decoders' output as their
# reference: JM (ldecod) and ffmpeg must agree MD5-exactly on every frame,
# otherwise the stream is rejected.
#
#   tools/dev/h264_conformance_supplement.sh [out-dir]
#
# Output: <out-dir>/<name>/{<name>.264,<name>.yuv}, default
# ~/nanocodec-data/conformance/h264/supplementary. 4:0:0 references are
# 4:2:0 with chroma at 128, as both JM and ffmpeg write them.
set -euo pipefail

out=${1:-$HOME/nanocodec-data/conformance/h264/supplementary}
bin=$HOME/nanocodec-data/anchors/x86_64/bin
clip=$HOME/nanocodec-data/clips/quick/crowd_mot17_09.yuv
clip_size=1280x720
frames=30
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# Non-flat scaling lists in x264's cqmfile syntax (H.264 §7.4.2.1.1.1
# order: zig-zag). Different intra/inter and luma/chroma lists so a decoder
# applying the wrong list (or the fall-back rule wrongly) cannot match.
cat >"$scratch/custom.cqm" <<'EOF'
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
EOF

common=(--profile high --bframes 0 --frames "$frames"
        --fps 30 --threads 1 --no-progress --quiet)

# name | size | x264 options
streams=(
  "x264_cabac_8x8_jvtcqm_wp|640x360|--crf 30 --8x8dct --cqm jvt --ref 4 --weightp 2 --slices 3"
  "x264_cavlc_8x8_customcqm|640x360|--crf 30 --no-cabac --8x8dct --cqmfile $scratch/custom.cqm --ref 3 --weightp 2 --slices 4"
  "x264_cabac_mono|640x360|--crf 28 --output-csp i400 --8x8dct --ref 2"
  "x264_cavlc_mono_customcqm|640x360|--crf 28 --output-csp i400 --no-cabac --8x8dct --cqmfile $scratch/custom.cqm --slices 2"
  "x264_cabac_constrained_intra|640x360|--crf 30 --constrained-intra --8x8dct --keyint 10 --slices 2 --ref 2"
  "x264_cavlc_constrained_intra|640x360|--crf 30 --constrained-intra --no-cabac --8x8dct --keyint 10 --slices 2"
  "x264_cabac_crop_l41_vui|1916x1076|--crf 32 --level 4.1 --8x8dct --ref 4 --deblock -2:1 --chroma-qp-offset 3 --sar 4:3 --overscan show --colorprim bt709 --transfer bt709 --colormatrix bt709 --chromaloc 1 --vbv-maxrate 4000 --vbv-bufsize 4000 --nal-hrd vbr"
  "x264_cavlc_crop_l41|1916x1076|--crf 32 --level 4.1 --no-cabac --8x8dct --ref 4 --slice-max-size 1500"
)

mkdir -p "$out"
for entry in "${streams[@]}"; do
    IFS='|' read -r name size options <<<"$entry"
    dir=$out/$name
    mkdir -p "$dir"
    # x264 is built without swscale: 4:0:0 streams get a luma-only source.
    pix_fmt=yuv420p
    [[ $options == *i400* ]] && pix_fmt=gray
    ffmpeg -y -hide_banner -loglevel error -nostdin -f rawvideo -pix_fmt yuv420p -s "$clip_size" \
        -i "$clip" -frames:v "$frames" -vf "scale=${size/x/:}:flags=bicubic" -f rawvideo \
        -pix_fmt "$pix_fmt" "$scratch/src.yuv"
    input_csp=i420
    [[ $pix_fmt == gray ]] && input_csp=i400
    # shellcheck disable=SC2086 # options are a word list by design
    "$bin/x264" "${common[@]}" --input-res "$size" --input-csp "$input_csp" $options \
        -o "$dir/$name.264" "$scratch/src.yuv"
    (cd "$scratch" && "$bin/ldecod" -i "$dir/$name.264" -o "$scratch/jm.yuv" >/dev/null 2>&1)
    # ffmpeg, like JM, outputs 4:0:0 as 4:2:0 with mid-grey chroma.
    ffmpeg -y -hide_banner -loglevel error -nostdin -i "$dir/$name.264" -f rawvideo \
        -pix_fmt yuv420p "$scratch/ff.yuv"
    cmp -s "$scratch/jm.yuv" "$scratch/ff.yuv" || { echo "$name: JM and ffmpeg disagree" >&2; exit 1; }
    mv "$scratch/jm.yuv" "$dir/$name.yuv"
    echo "$name: $(stat -c %s "$dir/$name.264") bytes, JM == ffmpeg"
done
