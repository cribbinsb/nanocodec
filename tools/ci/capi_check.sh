#!/usr/bin/env bash
# C API check: build libnanocodec, compile crates/nc-capi/examples/encode.c
# against include/nanocodec.h, and require its output to be byte-identical
# with the CLI's for the same input and settings (NV12 and I420 input, fixed
# QP and CBR, every codec); then decode it with ffmpeg as a smoke test.
#
#   tools/ci/capi_check.sh        (uses $CARGO_TARGET_DIR or target/)

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PATH="$HOME/.cargo/bin:$PATH"
TARGET=${CARGO_TARGET_DIR:-$REPO/target}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cargo build --release --quiet -p nc-capi -p nc-cli
cc -std=c99 -Wall -Wextra -Werror -I crates/nc-capi/include crates/nc-capi/examples/encode.c \
  "$TARGET/release/libnanocodec.a" -lpthread -ldl -lm -o "$WORK/encode"

W=202 H=98 FRAMES=10
ffmpeg -v error -y -f lavfi -i "testsrc2=s=${W}x${H}:r=30,scroll=horizontal=0.01" \
  -frames:v "$FRAMES" -pix_fmt yuv420p -f rawvideo "$WORK/in.i420"
ffmpeg -v error -y -f rawvideo -pix_fmt yuv420p -s "${W}x${H}" -i "$WORK/in.i420" \
  -pix_fmt nv12 -f rawvideo "$WORK/in.nv12"
{
  echo "# timestamp format v2"
  for ((i = 0; i < FRAMES; i++)); do
    us=$((i * 1000000 / 30))
    printf '%d.%03d\n' $((us / 1000)) $((us % 1000))
  done
} >"$WORK/ts.txt"

# The payload of an IVF file: frame payloads back to back.
ivf_payload() {
  python3 -I -c '
import sys
data = open(sys.argv[1], "rb").read()
pos, out = 32, bytearray()
while pos < len(data):
    size = int.from_bytes(data[pos:pos + 4], "little")
    out += data[pos + 12:pos + 12 + size]
    pos += 12 + size
sys.stdout.buffer.write(out)' "$1"
}

failures=0
for codec in h264 hevc av1; do
  for format in nv12 i420; do
    for kbps in 0 200; do
      rate=(--qp 32)
      [[ $codec == av1 ]] && rate=(--qp 180)
      [[ $kbps != 0 ]] && rate=(--bitrate "$kbps")
      "$WORK/encode" "$codec" "$format" "$W" "$H" "$FRAMES" "$kbps" "$WORK/in.$format" "$WORK/capi.out"
      "$TARGET/release/nanocodec" encode --codec "$codec" --input "$WORK/in.$format" \
        --input-format "$format" --width "$W" --height "$H" --fps 30 --frames "$FRAMES" \
        --intra-period 300 "${rate[@]}" --timestamps "$WORK/ts.txt" --timing nominal \
        --output "$WORK/cli.out" >/dev/null
      if [[ $codec == av1 ]]; then
        ivf_payload "$WORK/cli.out" >"$WORK/cli.payload"
        demux=(-f obu)
      else
        cp "$WORK/cli.out" "$WORK/cli.payload"
        demux=(-f "$codec")
      fi
      status=PASS
      cmp -s "$WORK/capi.out" "$WORK/cli.payload" || status="FAIL(differs from CLI)"
      ffmpeg -v error -xerror "${demux[@]}" -i "$WORK/capi.out" -f null - || status="FAIL(ffmpeg)"
      [[ $status == PASS ]] || failures=$((failures + 1))
      printf '%-5s %-5s %4s kbps  %s\n' "$codec" "$format" "$kbps" "$status"
    done
  done
done
if ((failures)); then
  echo "capi check: $failures failure(s)"
  exit 1
fi
echo "capi check: the C API reproduces the CLI's streams"
