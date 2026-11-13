#!/usr/bin/env bash
# Conformance of hierarchical-P (temporal layer) streams and of their lower
# operating points (docs/ARCHITECTURE.md §6.1, docs/handoff/hierarchical-p.md).
#
# For each codec and structure (2 and 3 temporal layers) a moving, noisy
# test pattern is encoded with keys that interrupt mini-GOPs (intra period
# not a multiple of the mini-GOP) and one picture size that needs cropping.
# The full stream, and every sub-stream produced by `nanocodec extract
# --max-layer M`, must decode bit-exactly to the encoder's reconstruction of
# the kept frames in nanocodec's own decoder and in every standard decoder:
# JM and ffmpeg (H.264), HM and ffmpeg (HEVC), aomdec, dav1d and ffmpeg
# (AV1), avmdec and dav2d (AV2). Decoders that can select the operating
# point themselves (HM -t, dav1d and aomdec --oppoint) must also produce the
# same frames from the full stream; AV2 streams carry no operating-point
# set, so their lower layers are checked through extracted sub-streams.
#
# Usage: tools/ci/temporal_layers_conformance.sh [codec ...]   (default: all
# codecs with layered support). Needs the pinned decoders from
# tools/anchors/build_anchors.sh and a release build of nanocodec.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ANCHORS=${NC_ANCHORS:-$HOME/nanocodec-data/anchors/$(uname -m)}/bin
NANOCODEC=${NANOCODEC:-$REPO/target/release/nanocodec}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

CODECS=("$@")
((${#CODECS[@]})) || CODECS=(h264 hevc av1 av2 vvc)

# width height frames intra_period
SHAPES=(
  "352 288 23 11"
  "200 120 13 6"
  "128 96 45 300"     # long GOP: H.264 frame_num wraps
  "64 64 262 300"     # HEVC POC LSB wrap (256)
)

# Adaptive structure (layers.adaptive=1) over a still-then-pan source.
ADAPTIVE_SHAPES=(
  "352 288 40 300"
  "200 120 30 13"
)

md5() { md5sum <"$1" | cut -d' ' -f1; }

# Write the frames of recon $1 (w h) whose layer (one per line in $4, from
# `nanocodec extract --list`) is at most $5 to $6.
expected() {
  local recon=$1 w=$2 h=$3 layers_file=$4 max=$5 out=$6
  local size=$((w * h * 3 / 2)) i=0 layer
  : >"$out"
  while read -r layer; do
    if ((layer <= max)); then
      dd if="$recon" bs="$size" skip="$i" count=1 status=none >>"$out"
    fi
    i=$((i + 1))
  done <"$layers_file"
}

# A source for $1 x $2 with $3 frames: a still-camera half, then a fast pan
# (the adaptive structure must layer the first and switch to flat).
still_then_pan() {
  local w=$1 h=$2 frames=$3 out=$4 half=$(($3 / 2))
  ffmpeg -v error -y -f lavfi \
    -i "testsrc2=s=${w}x${h}:r=30,trim=end_frame=1,loop=loop=-1:size=1,noise=alls=12:allf=t" \
    -frames:v "$half" -pix_fmt yuv420p -f rawvideo "$out"
  ffmpeg -v error -y -f lavfi \
    -i "testsrc2=s=${w}x${h}:r=30,scroll=horizontal=0.03:vertical=0.01,noise=alls=12:allf=t" \
    -frames:v "$((frames - half))" -pix_fmt yuv420p -f rawvideo - >>"$out"
}

# Decode $2 (codec $1) with every standard decoder and nanocodec into
# $WORK/<decoder>.yuv; prints the decoder names.
decode_all() {
  local codec=$1 stream=$2
  "$NANOCODEC" decode --codec "$codec" --input "$stream" --output "$WORK/own.yuv"
  case $codec in
    h264)
      (cd "$WORK" && "$ANCHORS/ldecod" -i "$stream" -o "$WORK/jm.yuv" >/dev/null)
      ffmpeg -v error -y -xerror -err_detect explode -f h264 -i "$stream" -f rawvideo \
        -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
      echo own jm ffmpeg
      ;;
    hevc)
      "$ANCHORS/TAppDecoder" -b "$stream" -o "$WORK/hm.yuv" -d 8 >/dev/null
      ffmpeg -v error -y -xerror -err_detect explode -f hevc -i "$stream" -f rawvideo \
        -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
      echo own hm ffmpeg
      ;;
    av1)
      "$ANCHORS/aomdec" --rawvideo --i420 -o "$WORK/aomdec.yuv" "$stream"
      "$ANCHORS/dav1d" -q -i "$stream" -o "$WORK/dav1d.yuv"
      # Extracted streams keep their timestamps: no frame duplication for gaps.
      ffmpeg -v error -y -xerror -c:v libdav1d -i "$stream" -fps_mode passthrough \
        -f rawvideo -pix_fmt yuv420p "$WORK/ffmpeg.yuv"
      echo own aomdec dav1d ffmpeg
      ;;
    av2)
      # Every shape is whole 8x8 blocks: no conformance-window cropping
      # (which dav2d ignores).
      "$ANCHORS/avmdec" --rawvideo -o "$WORK/avmdec.yuv" "$stream" >/dev/null 2>&1
      "$ANCHORS/dav2d" -q -i "$stream" -o "$WORK/dav2d.yuv" --muxer yuv
      echo own avmdec dav2d
      ;;
  esac
}

# Decoders that select operating point $3 (of $4 layers) from the FULL
# stream $2 themselves; prints the names of the outputs written.
decode_operating_point() {
  local codec=$1 stream=$2 max=$3 layers=$4
  case $codec in
    hevc)
      "$ANCHORS/TAppDecoder" -b "$stream" -o "$WORK/hm_t.yuv" -d 8 -t "$max" >/dev/null
      echo hm_t
      ;;
    av1)
      local op=$((layers - 1 - max))
      "$ANCHORS/dav1d" -q --oppoint "$op" -i "$stream" -o "$WORK/dav1d_op.yuv"
      "$ANCHORS/aomdec" --rawvideo --i420 --oppoint="$op" -o "$WORK/aomdec_op.yuv" "$stream"
      echo dav1d_op aomdec_op
      ;;
  esac
}

failures=0
for codec in "${CODECS[@]}"; do
  if [[ $codec == vvc ]]; then
    NANOCODEC="$NANOCODEC" python3 "$REPO/tools/ci/vvc_temporal_layers.py"
    continue
  fi
  case $codec in
    h264) q=30 ext=264 ;; hevc) q=30 ext=hevc ;; av1) q=120 ext=ivf ;; av2) q=140 ext=ivf ;;
  esac
  for layers in 2 3; do
    for shape in "${SHAPES[@]}" "${ADAPTIVE_SHAPES[@]/#/adaptive }"; do
      mode=fixed
      [[ $shape == adaptive* ]] && mode=adaptive shape=${shape#adaptive }
      read -r w h frames period <<<"$shape"
      src=$WORK/src.yuv stream=$WORK/full.$ext recon=$WORK/recon.yuv
      extra=(--param layers.adaptive=0)
      if [[ $mode == adaptive ]]; then
        still_then_pan "$w" "$h" "$frames" "$src"
        extra=(--param layers.adaptive=1)
      else
        ffmpeg -v error -y -f lavfi \
          -i "testsrc2=s=${w}x${h}:r=30,scroll=horizontal=0.013:vertical=0.007,noise=alls=12:allf=t" \
          -frames:v "$frames" -pix_fmt yuv420p -f rawvideo "$src"
      fi
      "$NANOCODEC" encode --codec "$codec" --input "$src" --width "$w" --height "$h" --fps 30 \
        --qp "$q" --frames "$frames" --intra-period "$period" --temporal-layers "$layers" \
        "${extra[@]}" --output "$stream" --recon "$recon"
      "$NANOCODEC" extract --codec "$codec" --temporal-layers "$layers" --list \
        --input "$stream" >"$WORK/layers.txt"
      for ((max = layers - 1; max >= 0; max--)); do
        sub=$WORK/sub.$ext
        "$NANOCODEC" extract --codec "$codec" --temporal-layers "$layers" --max-layer "$max" \
          --input "$stream" --output "$sub"
        expected "$recon" "$w" "$h" "$WORK/layers.txt" "$max" "$WORK/want.yuv"
        want=$(md5 "$WORK/want.yuv")
        status=PASS
        for dec in $(decode_all "$codec" "$sub") \
          $(decode_operating_point "$codec" "$stream" "$max" "$layers"); do
          [[ $(md5 "$WORK/$dec.yuv") == "$want" ]] || status="FAIL($dec)"
        done
        [[ $status == PASS ]] || failures=$((failures + 1))
        printf '%-4s L1T%s %-8s %-8s ip%-3s max T%s %7d bytes  %s\n' "$codec" "$layers" "$mode" \
          "${w}x${h}" "$period" "$max" "$(stat -c%s "$sub")" "$status"
      done
    done
  done
done

if ((failures)); then
  echo "temporal layer conformance: $failures failure(s)"
  exit 1
fi
echo "temporal layer conformance: all operating points bit-exact"
