#!/usr/bin/env bash
# Conformance smoke test for nanocodec's own streams (CLAUDE.md "Testing").
#
# Encodes synthetic noisy test patterns over a matrix of sizes (including
# sizes that are not superblock multiples), block sizes and quantisers, then
# requires the encoder's reconstruction to be bit-identical to the output of
# every standard decoder: aomdec, dav1d and ffmpeg (libdav1d) for AV1; the
# HM reference decoder and ffmpeg for HEVC; the JM reference decoder and
# ffmpeg for H.264; and nanocodec's own decoder.
#
# Needs the pinned decoders from tools/anchors/build_anchors.sh and a release
# build of nanocodec (cargo build --release -p nc-cli).

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ANCHORS=${NC_ANCHORS:-$HOME/nanocodec-data/anchors/$(uname -m)}/bin
NANOCODEC=${NANOCODEC:-$REPO/target/release/nanocodec}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# width height base_q_idx block frames
MATRIX=(
  "352 288 100 16 3"
  "352 288 100 8 3"
  "640 360 1 16 2"     # lowest quantiser: exercises Golomb-coded levels
  "640 360 255 16 2"   # highest quantiser
  "200 120 30 8 2"     # neither dimension a multiple of 16
  "64 64 5 16 2"
  "1280 720 150 16 2"
  "1920 1080 120 16 1"
)

# AV1 with inter frames: width height base_q_idx block frames intra_period
# [intra_refresh [loop_filter [inter_tx_split]]]. Intra-refresh columns split superblocks into
# 8x8 intra and inter blocks next to 16x16/32x32 ones, which exercises the
# reference-MV scan step rule (AV1 §7.10.2.2, the 2026-11-08 desync).
# loop_filter 1 enables deblocking (AV1 §7.14) at the pick-from-Q level;
# inter_tx_split 1 codes TX_MODE_SELECT inter frames with transform trees.
# The test pattern moves, so inter blocks use non-zero vectors, NEAREST/NEAR
# from neighbours, and references that reach outside the picture.
AV1_INTER_MATRIX=(
  "352 288 100 16 12 5"
  "352 288 60 8 8 8"
  "200 120 30 8 8 4"   # neither dimension a multiple of 16
  "64 64 50 8 6 3"
  "640 360 200 16 10 10"
  "1280 720 120 16 6 6"
  "642 362 100 16 4 3"  # odd-multiple sizes: 8x8-aligned coding, true frame size
  "1884 720 120 16 3 30"
  "1916 1076 120 16 2 30"
  "256 128 120 16 6 1000 7"
  "264 136 120 16 6 1000 7"  # not a multiple of 16, with intra refresh
  "352 288 120 16 6 3 0 1"     # deblocking: every filter length, key + inter
  "1280 720 200 16 4 300 0 1"  # deblocking at a coarse quantiser, 64x64 blocks
  "642 362 140 16 4 3 0 1"     # deblocking, odd-multiple frame size
  "256 128 120 16 6 1000 7 1"  # deblocking across intra-refresh splits
  "352 288 160 16 8 4 0 0 1"    # inter transform trees, key + inter
  "1280 720 200 16 4 300 0 1 1" # transform trees with deblocking (tx edges)
  "264 136 120 16 6 1000 7 0 1" # transform trees across intra-refresh splits
)

# AV1 full-swing inter residuals: base_q_idx values for flat frames that
# alternate black and white, so every inter residual is +-255. At fine steps
# the levels reach 2^(15 + dqDenom) / step (Golomb-coded, up to 32768); a
# fixed level cap once halved them (2026-11-08). The reconstruction must
# equal the source as well as match every decoder.
AV1_FULL_SWING_QUANTISERS=(1 5 10)

# HEVC: width height QP block frames (sizes must be multiples of 8).
HEVC_MATRIX=(
  "352 288 32 16 3"
  "352 288 32 8 3"
  "352 288 22 32 2"
  "640 360 0 16 2"     # lowest QP: exercises Exp-Golomb-coded levels
  "640 360 51 16 2"    # highest QP
  "200 120 30 8 2"     # neither dimension a multiple of 16
  "64 64 12 32 2"
  "1280 720 37 16 2"
  "1920 1080 30 32 1"
)

# HEVC with P frames: width height QP block frames intra_period [deblock_inter [tools]]. The moving
# pattern exercises skip, merge, AMVP with sub-sample vectors, references
# outside the picture, 64x64 CUs (split transform tree) and deblocking.
HEVC_INTER_MATRIX=(
  "352 288 32 16 8 30"
  "352 288 27 32 8 4"
  "352 288 37 64 8 30"
  "200 120 30 8 8 5"   # neither dimension a multiple of 16
  "640 360 0 16 4 30"  # lowest QP
  "640 360 51 16 6 30" # highest QP
  "1280 720 34 16 6 30"
  "1920 1080 30 64 3 30"
  "352 288 32 16 8 30 0" # hevc.deblock_inter 0: P slices override the PPS, deblocking off
  "642 362 30 16 4 3"   # not multiples of 8: conformance window
  "1884 720 34 16 3 30"
  "1916 1076 30 32 2 30"
  # Optional coding tools (HEVC_TOOL_ARGS): planned key CUs with PART_NxN and
  # intra transform splits down to 4x4 (DST), and inter transform trees.
  "352 288 32 16 8 30 1 intra"
  "200 120 22 8 6 3 1 intra"
  "640 360 40 16 6 4 1 inter"
  "642 362 30 16 4 3 1 all"
)
declare -A HEVC_TOOL_ARGS=(
  [intra]="--param hevc.intra_leaf_cost=1 --param hevc.intra_nxn=1 --param hevc.intra_tu_depth=2"
  [inter]="--param hevc.inter_tu_split=1"
  [all]="--param hevc.intra_leaf_cost=1 --param hevc.intra_nxn=1 --param hevc.intra_tu_depth=2 --param hevc.inter_tu_split=1 --param hevc.coarse_inter_intra=0"
)

# H.264: width height QP frames intra_period (16x16 macroblocks only; other
# sizes are padded and cropped). The pattern scrolls by a non-integer number
# of samples per frame, so P pictures use quarter-sample vectors, P_Skip,
# I_16x16 macroblocks and references reaching outside the picture; the
# deblocking filter is on.
H264_MATRIX=(
  "352 288 30 12 5"
  "352 288 0 6 3"      # lowest QP: exercises Exp-Golomb-coded levels
  "640 360 51 10 4"    # highest QP
  "200 120 20 8 8"     # neither dimension a multiple of 16: cropping
  "202 98 40 10 30"
  "16 16 26 6 3"       # a single macroblock
  "1280 720 34 6 30"
  "1920 1080 28 4 2"   # 1080 is not a multiple of 16
  "642 362 30 4 3"
  "1884 720 34 3 30"
  "1916 1076 30 2 30"
  # Optional coding tools (H264_TOOL_ARGS): Intra_8x8 next to Intra_4x4 and
  # Intra_16x16, in key and P pictures (every mode, every edge case of the
  # neighbours, the 8x8 transform's CABAC contexts).
  "352 288 26 8 4 i8x8"
  "200 120 34 6 3 i8x8"
  "1280 720 20 3 30 i8x8"
)
declare -A H264_TOOL_ARGS=(
  [i8x8]="--param h264.intra8x8=2 --param h264.intra8x8_all_modes=1 --param h264.intra4x4_gate_satd=0"
)

md5() { md5sum <"$1" | cut -d' ' -f1; }

failures=0
for entry in "${MATRIX[@]}"; do
  read -r w h q block frames <<<"$entry"
  src=$WORK/src.yuv stream=$WORK/out.ivf recon=$WORK/recon.yuv
  ffmpeg -v error -y -f lavfi -i "testsrc2=s=${w}x${h}:r=30,noise=alls=20:allf=t" \
    -frames:v "$frames" -pix_fmt yuv420p -f rawvideo "$src"
  "$NANOCODEC" encode --codec av1 --input "$src" --width "$w" --height "$h" --fps 30 \
    --qp "$q" --frames "$frames" --block "$block" --output "$stream" --recon "$recon"
  "$ANCHORS/aomdec" --rawvideo --i420 -o "$WORK/aomdec.yuv" "$stream"
  "$ANCHORS/dav1d" -q -i "$stream" -o "$WORK/dav1d.yuv"
  ffmpeg -v error -y -xerror -c:v libdav1d -i "$stream" -f rawvideo -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
  want=$(md5 "$recon")
  status=PASS
  "$NANOCODEC" decode --codec av1 --input "$stream" --output "$WORK/own.yuv"
  for dec in own aomdec dav1d ffmpeg; do
    [[ $(md5 "$WORK/$dec.yuv") == "$want" ]] || status="FAIL($dec)"
  done
  [[ $status == PASS ]] || failures=$((failures + 1))
  printf '%-10s q%-3s block %-2s %5d bytes  %s\n' "${w}x${h}" "$q" "$block" "$(stat -c%s "$stream")" "$status"
done

for entry in "${AV1_INTER_MATRIX[@]}"; do
  read -r w h q block frames period refresh loop_filter tx_split <<<"$entry"
  src=$WORK/src.yuv stream=$WORK/out.ivf recon=$WORK/recon.yuv
  ffmpeg -v error -y -f lavfi -i "testsrc2=s=${w}x${h}:r=30,noise=alls=8:allf=t" \
    -frames:v "$frames" -pix_fmt yuv420p -f rawvideo "$src"
  refresh_args=()
  [[ ${refresh:-0} != 0 ]] && refresh_args+=(--intra-refresh "$refresh")
  # Explicit both ways: the AV1 Default filters (av1.loop_filter = 1).
  refresh_args+=(--param "av1.loop_filter=${loop_filter:-0}")
  [[ ${tx_split:-0} != 0 ]] && refresh_args+=(--param av1.inter_tx_split=1)
  "$NANOCODEC" encode --codec av1 --input "$src" --width "$w" --height "$h" --fps 30 \
    --qp "$q" --frames "$frames" --block "$block" --intra-period "$period" \
    "${refresh_args[@]}" --output "$stream" --recon "$recon"
  "$ANCHORS/aomdec" --rawvideo --i420 -o "$WORK/aomdec.yuv" "$stream"
  "$ANCHORS/dav1d" -q -i "$stream" -o "$WORK/dav1d.yuv"
  ffmpeg -v error -y -xerror -c:v libdav1d -i "$stream" -f rawvideo -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
  want=$(md5 "$recon")
  status=PASS
  "$NANOCODEC" decode --codec av1 --input "$stream" --output "$WORK/own.yuv"
  for dec in own aomdec dav1d ffmpeg; do
    [[ $(md5 "$WORK/$dec.yuv") == "$want" ]] || status="FAIL($dec)"
  done
  [[ $status == PASS ]] || failures=$((failures + 1))
  printf 'av1-inter %-10s q%-3s block %-2s ip%-3s %7d bytes  %s\n' "${w}x${h}" "$q" "$block" "$period" "$(stat -c%s "$stream")" "$status"
done

for q in "${AV1_FULL_SWING_QUANTISERS[@]}"; do
  w=320 h=240 frames=6
  src=$WORK/src.yuv stream=$WORK/out.ivf recon=$WORK/recon.yuv
  ffmpeg -v error -y -f lavfi -i "color=c=black:s=${w}x${h}:r=30,geq=lum='255*mod(N,2)':cb=128:cr=128" \
    -frames:v "$frames" -pix_fmt yuv420p -f rawvideo "$src"
  "$NANOCODEC" encode --codec av1 --input "$src" --width "$w" --height "$h" --fps 30 \
    --qp "$q" --frames "$frames" --intra-period 300 --output "$stream" --recon "$recon"
  "$ANCHORS/aomdec" --rawvideo --i420 -o "$WORK/aomdec.yuv" "$stream"
  "$ANCHORS/dav1d" -q -i "$stream" -o "$WORK/dav1d.yuv"
  ffmpeg -v error -y -xerror -c:v libdav1d -i "$stream" -f rawvideo -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
  want=$(md5 "$recon")
  status=PASS
  [[ $(md5 "$src") == "$want" ]] || status="FAIL(recon differs from source)"
  "$NANOCODEC" decode --codec av1 --input "$stream" --output "$WORK/own.yuv"
  for dec in own aomdec dav1d ffmpeg; do
    [[ $(md5 "$WORK/$dec.yuv") == "$want" ]] || status="FAIL($dec)"
  done
  [[ $status == PASS ]] || failures=$((failures + 1))
  printf 'av1-full-swing %-10s q%-3s %7d bytes  %s\n' "${w}x${h}" "$q" "$(stat -c%s "$stream")" "$status"
done

for entry in "${HEVC_MATRIX[@]}"; do
  read -r w h q block frames <<<"$entry"
  src=$WORK/src.yuv stream=$WORK/out.hevc recon=$WORK/recon.yuv
  ffmpeg -v error -y -f lavfi -i "testsrc2=s=${w}x${h}:r=30,noise=alls=20:allf=t" \
    -frames:v "$frames" -pix_fmt yuv420p -f rawvideo "$src"
  "$NANOCODEC" encode --codec hevc --input "$src" --width "$w" --height "$h" --fps 30 \
    --qp "$q" --frames "$frames" --block "$block" --output "$stream" --recon "$recon"
  "$ANCHORS/TAppDecoder" -b "$stream" -o "$WORK/hm.yuv" -d 8 >/dev/null
  ffmpeg -v error -y -xerror -err_detect explode -f hevc -i "$stream" -f rawvideo \
    -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
  want=$(md5 "$recon")
  status=PASS
  "$NANOCODEC" decode --codec hevc --input "$stream" --output "$WORK/own.yuv"
  for dec in own hm ffmpeg; do
    [[ $(md5 "$WORK/$dec.yuv") == "$want" ]] || status="FAIL($dec)"
  done
  [[ $status == PASS ]] || failures=$((failures + 1))
  printf 'hevc %-10s qp%-3s block %-2s %7d bytes  %s\n' "${w}x${h}" "$q" "$block" "$(stat -c%s "$stream")" "$status"
done

for entry in "${HEVC_INTER_MATRIX[@]}"; do
  read -r w h q block frames period deblock_inter tools <<<"$entry"
  src=$WORK/src.yuv stream=$WORK/out.hevc recon=$WORK/recon.yuv
  ffmpeg -v error -y -f lavfi -i "testsrc2=s=${w}x${h}:r=30,noise=alls=12:allf=t" \
    -frames:v "$frames" -pix_fmt yuv420p -f rawvideo "$src"
  deblock_args=()
  [[ ${deblock_inter:-1} == 0 ]] && deblock_args+=(--param hevc.deblock_inter=0)
  # shellcheck disable=SC2206 # the tool arguments are space-separated words
  [[ -n ${tools:-} ]] && deblock_args+=(${HEVC_TOOL_ARGS[$tools]})
  "$NANOCODEC" encode --codec hevc --input "$src" --width "$w" --height "$h" --fps 30 \
    --qp "$q" --frames "$frames" --block "$block" --intra-period "$period" \
    "${deblock_args[@]}" --output "$stream" --recon "$recon"
  "$ANCHORS/TAppDecoder" -b "$stream" -o "$WORK/hm.yuv" -d 8 >/dev/null
  ffmpeg -v error -y -xerror -err_detect explode -f hevc -i "$stream" -f rawvideo \
    -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
  want=$(md5 "$recon")
  status=PASS
  "$NANOCODEC" decode --codec hevc --input "$stream" --output "$WORK/own.yuv"
  for dec in own hm ffmpeg; do
    [[ $(md5 "$WORK/$dec.yuv") == "$want" ]] || status="FAIL($dec)"
  done
  [[ $status == PASS ]] || failures=$((failures + 1))
  printf 'hevc-inter %-10s qp%-3s block %-2s ip%-3s %-5s %7d bytes  %s\n' "${w}x${h}" "$q" "$block" "$period" "${tools:-}" "$(stat -c%s "$stream")" "$status"
done

for entry in "${H264_MATRIX[@]}"; do
  read -r w h q frames period tools <<<"$entry"
  src=$WORK/src.yuv stream=$WORK/out.264 recon=$WORK/recon.yuv
  tool_args=()
  # shellcheck disable=SC2206 # the tool arguments are space-separated words
  [[ -n ${tools:-} ]] && tool_args+=(${H264_TOOL_ARGS[$tools]})
  ffmpeg -v error -y -f lavfi \
    -i "testsrc2=s=${w}x${h}:r=30,scroll=horizontal=0.013:vertical=0.007,noise=alls=12:allf=t" \
    -frames:v "$frames" -pix_fmt yuv420p -f rawvideo "$src"
  "$NANOCODEC" encode --codec h264 --input "$src" --width "$w" --height "$h" --fps 30 \
    --qp "$q" --frames "$frames" --intra-period "$period" "${tool_args[@]}" \
    --output "$stream" --recon "$recon"
  # JM writes its log files into the working directory.
  (cd "$WORK" && "$ANCHORS/ldecod" -i "$stream" -o "$WORK/jm.yuv" >/dev/null)
  ffmpeg -v error -y -xerror -err_detect explode -f h264 -i "$stream" -f rawvideo \
    -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
  want=$(md5 "$recon")
  status=PASS
  "$NANOCODEC" decode --codec h264 --input "$stream" --output "$WORK/own.yuv"
  for dec in own jm ffmpeg; do
    [[ $(md5 "$WORK/$dec.yuv") == "$want" ]] || status="FAIL($dec)"
  done
  [[ $status == PASS ]] || failures=$((failures + 1))
  printf 'h264 %-10s qp%-3s ip%-3s %-5s %7d bytes  %s\n' "${w}x${h}" "$q" "$period" "${tools:-}" "$(stat -c%s "$stream")" "$status"
done

if ((failures)); then
  echo "nanocodec conformance: $failures failure(s)"
  exit 1
fi
echo "nanocodec conformance: all streams bit-exact with the standard decoders"
