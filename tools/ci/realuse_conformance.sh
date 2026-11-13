#!/usr/bin/env bash
# Conformance of the opt-in deployment features (docs/handoff/realuse.md):
# colour signalling (H.264/HEVC VUI, AV1 color_config), frame timing
# (VUI / timing_info, camera timestamps with variable frame rate),
# encoder-side frame dropping at an unreachable target, scene-cut key
# frames, intra refresh, loss recovery (key frame, or long-term reference
# where the codec supports it) and mid-stream rate and resolution changes.
#
# For every codec and setting: the stream must decode bit-exactly against
# the encoder's reconstruction with every standard decoder (JM + ffmpeg,
# HM + ffmpeg, aomdec + dav1d + ffmpeg, avmdec + dav2d) and with our own
# decoder; ffprobe must report the signalled colour description, and AV1
# IVF packets must carry the input timestamps. AV2 runs the settings it
# supports (no colour signalling or intra refresh yet).
#
# Usage: tools/ci/realuse_conformance.sh [codec ...]   (default: all).
# Needs the pinned decoders from tools/anchors/build_anchors.sh and a release
# build of nanocodec (cargo build --release -p nc-cli).

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ANCHORS=${NC_ANCHORS:-$HOME/nanocodec-data/anchors/$(uname -m)}/bin
NANOCODEC=${NANOCODEC:-$REPO/target/release/nanocodec}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# 202x98 is not a block multiple in any codec, so cropping is exercised too.
W=202 H=98 FRAMES=24 PERIOD=12

# Colour settings: nanocodec flags | expected ffprobe "range,primaries,transfer,space".
COLOUR_CASES=(
  "--colour bt709 --range full|pc,bt709,bt709,bt709"
  "--colour bt601|tv,smpte170m,smpte170m,smpte170m"
  "--colour bt601-625 --range limited|tv,bt470bg,smpte170m,bt470bg"
  "--range full|pc,unknown,unknown,unknown"
)

# Timing settings (flags; files are relative to $WORK). The timestamps
# simulate a 30 fps camera that drops two frames in every ten (ts.txt) and
# one switching from 30 to 15 fps (ts-rate.txt).
TIMING_CASES=(
  "--timing constant"
  "--timing variable"
  "--timestamps $WORK/ts.txt --bitrate 300"
  "--timestamps $WORK/ts-rate.txt --bitrate 300"
)

# Frame dropping: a target far below the maximum-quantizer rate, so frames
# must be dropped (checked in the rate log), flat and layered.
DROP_CASES=(
  "--bitrate 8 --frame-drop"
  "--bitrate 8 --peak-bitrate 8 --frame-drop"
  "--bitrate 8 --peak-bitrate 8 --frame-drop --temporal-layers 2"
  "--bitrate 8 --peak-bitrate 8 --frame-drop --temporal-layers 3"
)

# Scene cuts: the source cuts to different content at CUT_AT; the key
# frame must land there (flat and layered).
CUT_AT=14
SCENE_CASES=(
  "--param scene.detect=1 --bitrate 400"
  "--param scene.detect=1 --bitrate 400 --temporal-layers 3"
)

# Intra refresh instead of periodic keys, at fixed QP and under rate
# control (recovery itself is tested in each decoder's unit tests); period 7
# puts range edges inside superblocks (HEVC splits nodes there).
REFRESH_CASES=(
  "--intra-refresh 4"
  "--intra-refresh 3 --bitrate 300"
  "--intra-refresh 7"
)

# Mid-stream reconfiguration: rate changes (target, peak, buffer) at given
# frames; the rate log must show each new target from its frame on.
RECONFIG_CASES=(
  "--bitrate 300 --rate-change 8:150 --rate-change 16:600:1200:250|8:150000 16:600000"
  "--bitrate 300 --temporal-layers 3 --rate-change 5:900|5:900000"
)

# Resolution changes (--resize FRAME:WxH): a key frame with new parameter
# sets or sequence header at each switch; the source changes size there
# (resize.yuv: 10 frames at W x H, 8 at the RESIZE_TO size, 6 at W x H).
RESIZE_TO=128x72
RESIZE_FLAGS="--resize 10:$RESIZE_TO --resize 18:${W}x$H"
RESIZE_CASES=(
  "--bitrate 300 $RESIZE_FLAGS"
  "$RESIZE_FLAGS"
  "--bitrate 300 --temporal-layers 3 $RESIZE_FLAGS"
  "--bitrate 300 --ltr-interval 4 $RESIZE_FLAGS"
)

# Loss recovery: the frame given to --simulate-loss is reported lost and
# left out of the stream and the reconstruction; every decoder must decode
# the rest exactly, and the frame after the loss must be of the given type.
# With a long-term reference the recovery frame is an inter frame. With two
# fixed temporal layers frame 10 is a T0 frame (recovered from the LTR
# marked at frame 8) and frame 9 a T1 frame, whose loss needs no recovery
# at all; the adaptive structure (the default) may code either flat. With
# three fixed layers frame 10 is a T1 frame, also recovered from the LTR.
LOST=9
L1T2="--temporal-layers 2 --param layers.adaptive=0"
L1T3="--temporal-layers 3 --param layers.adaptive=0"
LOSS_CASES=(
  "all|--simulate-loss $LOST --bitrate 300|Key"
  "h264 hevc av1 av2|--ltr-interval 4 --simulate-loss $LOST --bitrate 300|Inter"
  "h264 hevc av1 av2|$L1T2 --ltr-interval 4 --simulate-loss 10 --bitrate 300|Inter"
  "h264 hevc av1 av2|--temporal-layers 2 --ltr-interval 4 --simulate-loss 10 --bitrate 300|Inter"
  "h264 hevc av1 av2|$L1T3 --ltr-interval 4 --simulate-loss 10 --bitrate 300|Inter"
  "all|$L1T2 --simulate-loss $LOST --bitrate 300|Inter"
)

# Second reference (refs.second, engine second_reference.rs): the last
# high-quality frame as a second reference, chosen per block; H.264 only so
# far. Fixed QP and rate control (cadence base frames refresh it), on the
# scene-cut source, with and without the cut detected (a key mid-stream).
SECOND_REF_CODECS=" h264 "
SECOND_REF_CASES=(
  "--param refs.second=1"
  "--param refs.second=1 --bitrate 300"
  "--param refs.second=1 --param refs.second_full=1 --bitrate 300"
  "--param refs.second=1 --param scene.detect=1 --bitrate 400"
)

md5() { md5sum <"$1" | cut -d' ' -f1; }

{
  echo "# timestamp format v2"
  for ((i = 0, t = 0; i < FRAMES; i++)); do
    echo "$((t / 1000)).$(printf '%03d' $((t % 1000)))"
    t=$((t + (i % 10 == 9 ? 100000 : 33333)))
  done
} >"$WORK/ts.txt"
# A camera switching from 30 to 15 fps at frame 12 (a frame-rate change).
{
  echo "# timestamp format v2"
  for ((i = 0, t = 0; i < FRAMES; i++)); do
    echo "$((t / 1000)).$(printf '%03d' $((t % 1000)))"
    t=$((t + (i < 12 ? 33333 : 66667)))
  done
} >"$WORK/ts-rate.txt"

ffmpeg -v error -y -f lavfi \
  -i "testsrc2=s=${W}x${H}:r=30,scroll=horizontal=0.013:vertical=0.007,noise=alls=12:allf=t" \
  -frames:v "$FRAMES" -pix_fmt yuv420p -f rawvideo "$WORK/src.yuv"
# Scene-cut source: a slowly scrolling pattern, then colour bars.
ffmpeg -v error -y -f lavfi \
  -i "testsrc2=s=${W}x${H}:r=30,scroll=horizontal=0.004,noise=alls=2:allf=t" \
  -f lavfi -i "smptebars=s=${W}x${H}:r=30,noise=alls=2:allf=t" \
  -filter_complex "[0]trim=end_frame=$CUT_AT[a];[1]trim=end_frame=$((FRAMES - CUT_AT))[b];[a][b]concat" \
  -pix_fmt yuv420p -f rawvideo "$WORK/cut.yuv"

# Resolution-change source (RESIZE_CASES).
for part in "${W}x$H:10" "$RESIZE_TO:8" "${W}x$H:6"; do
  ffmpeg -v error -y -f lavfi -i "testsrc2=s=${part%%:*}:r=30,noise=alls=8:allf=t" \
    -frames:v "${part##*:}" -pix_fmt yuv420p -f rawvideo -
done >"$WORK/resize.yuv"

# JM decode of H.264 stream $1 into $2. The pinned JM cannot change picture
# size mid-stream (it misdecodes x264 streams with a size change too), so a
# stream whose SPS changes (JM_SPLIT=1, the --resize cases) is decoded per
# coded video sequence: split before every SPS (each starts an IDR access
# unit that depends on nothing before it) and concatenated.
jm_decode() {
  local stream=$1 out=$2
  if [[ ${JM_SPLIT:-0} != 1 ]]; then
    (cd "$WORK" && "$ANCHORS/ldecod" -i "$stream" -o "$out" >/dev/null)
    return
  fi
  rm -f "$WORK"/cvs-*.264 "$out"
  python3 - "$stream" "$WORK/cvs" <<'PY'
import re, sys
data = open(sys.argv[1], "rb").read()
starts = [m.start() for m in re.finditer(b"\x00\x00\x00\x01", data) if data[m.start() + 4] & 0x1F == 7]
for n, (a, b) in enumerate(zip(starts, starts[1:] + [len(data)])):
    open(f"{sys.argv[2]}-{n:03d}.264", "wb").write(data[a:b])
PY
  for part in "$WORK"/cvs-*.264; do
    (cd "$WORK" && "$ANCHORS/ldecod" -i "$part" -o "$WORK/jm-part.yuv" >/dev/null)
    cat "$WORK/jm-part.yuv" >>"$out"
  done
}

# Decode $2 (codec $1) with every standard decoder into $WORK/<decoder>.yuv
# and print the decoder names. ffmpeg writes its native pixel format, so a
# full-range stream is not range-converted, and passes frames through
# without duplicating any to fill timestamp gaps.
standard_decode() {
  local codec=$1 stream=$2
  case $codec in
    h264)
      jm_decode "$stream" "$WORK/jm.yuv"
      ffmpeg -v error -y -xerror -err_detect explode -f h264 -i "$stream" -noautoscale -fps_mode passthrough -f rawvideo "$WORK/ffmpeg.yuv"
      echo jm ffmpeg ;;
    hevc)
      "$ANCHORS/TAppDecoder" -b "$stream" -o "$WORK/hm.yuv" -d 8 >/dev/null
      ffmpeg -v error -y -xerror -err_detect explode -f hevc -i "$stream" -noautoscale -fps_mode passthrough -f rawvideo "$WORK/ffmpeg.yuv"
      echo hm ffmpeg ;;
    av1)
      "$ANCHORS/aomdec" --rawvideo --i420 -o "$WORK/aomdec.yuv" "$stream"
      "$ANCHORS/dav1d" -q -i "$stream" -o "$WORK/dav1d.yuv"
      ffmpeg -v error -y -xerror -c:v libdav1d -i "$stream" -noautoscale -fps_mode passthrough -f rawvideo "$WORK/ffmpeg.yuv"
      echo aomdec dav1d ffmpeg ;;
    av2)
      "$ANCHORS/avmdec" --rawvideo -o "$WORK/avmdec.yuv" "$stream" >/dev/null 2>&1
      # dav2d ignores the conformance window: crop its coded-size frames.
      "$ANCHORS/dav2d" -q -i "$stream" -o "$WORK/dav2d-coded.yuv" --muxer yuv
      crop_coded "$WORK/dav2d-coded.yuv" "$WORK/dav2d.yuv"
      echo avmdec dav2d ;;
  esac
}

# Crop the coded-size (whole 8x8 blocks) I420 frames of $1 to their visible
# size in $2: W x H, or per segment from FRAME_SIZES ("count:WxH ...", the
# resize cases) when set.
crop_coded() {
  python3 - "$1" "$2" "${FRAME_SIZES:-0:${W}x$H}" <<'PY'
import sys
src, dst, plan = sys.argv[1], sys.argv[2], sys.argv[3].split()
data = open(src, "rb").read()
sizes = []
for segment in plan:
    count, size = segment.split(":")
    w, h = map(int, size.split("x"))
    sizes += [(w, h)] * (int(count) or 1 << 30)
out, base = bytearray(), 0
for w, h in sizes:
    if base >= len(data):
        break
    cw, ch = -(-w // 8) * 8, -(-h // 8) * 8
    for pw, ph, pcw, off in ((w, h, cw, 0), (w // 2, h // 2, cw // 2, cw * ch),
                             (w // 2, h // 2, cw // 2, cw * ch * 5 // 4)):
        for y in range(ph):
            row = base + off + y * pcw
            out += data[row:row + pw]
    base += cw * ch * 3 // 2
open(dst, "wb").write(out)
PY
}

# Where the stream of codec $1 is written.
stream_path() {
  case $1 in h264) echo "$WORK/out.264" ;; hevc) echo "$WORK/out.hevc" ;; av1 | av2) echo "$WORK/out.ivf" ;; esac
}

# Encode with "$@" as extra flags, then check every decoder against the
# reconstruction; prints the status (PASS or FAIL(decoder)).
encode_and_check() {
  local codec=$1
  shift
  local q stream
  case $codec in h264) q=30 ;; hevc) q=32 ;; av1) q=120 ;; av2) q=140 ;; esac
  stream=$(stream_path "$codec")
  "$NANOCODEC" encode --codec "$codec" --input "${SRC:-$WORK/src.yuv}" --width "$W" --height "$H" \
    --fps 30 --qp "$q" --frames "$FRAMES" --intra-period "$PERIOD" "$@" \
    --output "$stream" --recon "$WORK/recon.yuv"
  local want status=PASS
  want=$(md5 "$WORK/recon.yuv")
  for dec in $(standard_decode "$codec" "$stream"); do
    [[ $(md5 "$WORK/$dec.yuv") == "$want" ]] || status="FAIL($dec)"
  done
  "$NANOCODEC" decode --codec "$codec" --input "$stream" --output "$WORK/own.yuv"
  [[ $(md5 "$WORK/own.yuv") == "$want" ]] || status="FAIL(own)"
  echo "$status"
}

CODECS=("$@")
((${#CODECS[@]})) || CODECS=(h264 hevc av1 av2)

failures=0
for codec in "${CODECS[@]}"; do
  stream=$(stream_path "$codec")
  # AV2 has no colour signalling or intra refresh yet.
  colour_cases=("${COLOUR_CASES[@]}") refresh_cases=("${REFRESH_CASES[@]}")
  [[ $codec == av2 ]] && colour_cases=() refresh_cases=()
  for entry in "${colour_cases[@]}"; do
    flags=${entry%%|*} expected=${entry##*|}
    # shellcheck disable=SC2086 # flags are intentionally word-split
    status=$(encode_and_check "$codec" $flags)
    got=$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=color_range,color_primaries,color_transfer,color_space \
      -of csv=p=0 "$stream" | head -1)
    [[ $status != PASS || $got == "$expected" ]] || status="FAIL(ffprobe $got)"
    [[ $status == PASS ]] || failures=$((failures + 1))
    printf '%-5s %-36s %-40s %s\n' "$codec" "$flags" "$got" "$status"
  done
  for flags in "${TIMING_CASES[@]}"; do
    # shellcheck disable=SC2086 # flags are intentionally word-split
    status=$(encode_and_check "$codec" $flags)
    if [[ $status == PASS && $codec == av1 && $flags == *--timestamps* ]]; then
      # IVF packet times (microsecond time base) must equal the input file's.
      got=$(ffprobe -v error -select_streams v:0 -show_entries packet=pts_time -of csv=p=0 \
        "$stream" | awk '{printf "%.3f\n", $1 * 1000}')
      ts_file=$(sed -n 's/.*--timestamps \([^ ]*\).*/\1/p' <<<"$flags")
      [[ $got == "$(tail -n +2 "$ts_file")" ]] || status="FAIL(ivf timestamps)"
    fi
    [[ $status == PASS ]] || failures=$((failures + 1))
    printf '%-5s %-77s %s\n' "$codec" "${flags//$WORK\//}" "$status"
  done
  for entry in "${RECONFIG_CASES[@]}"; do
    flags=${entry%%|*} expected=${entry##*|}
    # shellcheck disable=SC2086 # flags are intentionally word-split
    status=$(encode_and_check "$codec" $flags --rate-log "$WORK/rate.csv")
    for check in $expected; do
      got=$(awk -F, -v f="${check%%:*}" '$1 == f { print $10 }' "$WORK/rate.csv")
      [[ $status != PASS || $got == "${check##*:}" ]] || status="FAIL(target at ${check%%:*}: $got)"
    done
    [[ $status == PASS ]] || failures=$((failures + 1))
    printf '%-5s %-77s %s
' "$codec" "$flags" "$status"
  done
  for flags in "${RESIZE_CASES[@]}"; do
    # shellcheck disable=SC2086 # flags are intentionally word-split
    status=$(SRC=$WORK/resize.yuv JM_SPLIT=1 FRAME_SIZES="10:${W}x$H 8:$RESIZE_TO 6:${W}x$H" \
      encode_and_check "$codec" $flags)
    if [[ $status == PASS && $flags == *--bitrate* ]]; then
      # shellcheck disable=SC2086 # flags are intentionally word-split
      SRC=$WORK/resize.yuv encode_and_check "$codec" $flags --rate-log "$WORK/rate.csv" >/dev/null
      keys=$(awk -F, '$2 == "Key" { printf "%s ", $1 }' "$WORK/rate.csv")
      [[ $keys == "0 10 12 18 " ]] || status="FAIL(keys at $keys)"
    fi
    [[ $status == PASS ]] || failures=$((failures + 1))
    printf '%-5s %-77s %s\n' "$codec" "$flags" "$status"
  done
  for flags in "${DROP_CASES[@]}"; do
    # shellcheck disable=SC2086 # flags are intentionally word-split
    status=$(encode_and_check "$codec" $flags --rate-log "$WORK/rate.csv")
    dropped=$(grep -c ',Dropped,' "$WORK/rate.csv" || true)
    [[ $status != PASS || $dropped -gt 0 ]] || status="FAIL(nothing dropped)"
    [[ $status == PASS ]] || failures=$((failures + 1))
    printf '%-5s %-65s %2d dropped %s\n' "$codec" "$flags" "$dropped" "$status"
  done
  for flags in "${refresh_cases[@]}"; do
    # shellcheck disable=SC2086 # flags are intentionally word-split
    status=$(encode_and_check "$codec" $flags)
    [[ $status == PASS ]] || failures=$((failures + 1))
    printf '%-5s %-77s %s\n' "$codec" "$flags" "$status"
  done
  for entry in "${LOSS_CASES[@]}"; do
    IFS='|' read -r codecs flags recovery <<<"$entry"
    [[ $codecs == all || " $codecs " == *" $codec "* ]] || continue
    # shellcheck disable=SC2086 # flags are intentionally word-split
    status=$(encode_and_check "$codec" $flags --rate-log "$WORK/rate.csv")
    lost=$(sed -n 's/.*--simulate-loss \([0-9]*\).*/\1/p' <<<"$flags")
    got=$(awk -F, -v f=$((lost + 1)) '$1 == f { print $2 }' "$WORK/rate.csv")
    [[ $status != PASS || $got == "$recovery" ]] || status="FAIL(recovery frame $got)"
    [[ $status == PASS ]] || failures=$((failures + 1))
    printf '%-5s %-65s %-5s %s\n' "$codec" "$flags" "$got" "$status"
  done
  if [[ $SECOND_REF_CODECS == *" $codec "* ]]; then
    for flags in "${SECOND_REF_CASES[@]}"; do
      # shellcheck disable=SC2086 # flags are intentionally word-split
      status=$(PERIOD=1000 SRC=$WORK/cut.yuv encode_and_check "$codec" $flags)
      [[ $status == PASS ]] || failures=$((failures + 1))
      printf '%-5s %-77s %s\n' "$codec" "$flags" "$status"
    done
  fi
  for flags in "${SCENE_CASES[@]}"; do
    # shellcheck disable=SC2086 # flags are intentionally word-split
    status=$(PERIOD=1000 SRC=$WORK/cut.yuv encode_and_check "$codec" $flags \
      --rate-log "$WORK/rate.csv")
    keys=$(awk -F, '$2 == "Key" { printf "%s ", $1 }' "$WORK/rate.csv")
    [[ $status != PASS || $keys == "0 $CUT_AT " ]] || status="FAIL(keys at $keys)"
    [[ $status == PASS ]] || failures=$((failures + 1))
    printf '%-5s %-65s keys %-8s %s\n' "$codec" "$flags" "$keys" "$status"
  done
done

if ((failures)); then
  echo "realuse conformance: $failures failure(s)"
  exit 1
fi
echo "realuse conformance: all streams bit-exact with the standard decoders and our own"
