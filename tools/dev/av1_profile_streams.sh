#!/usr/bin/env bash
# AV1 High and Professional profile streams (seq_profile 1 and 2).
#
# libaom's test vectors cover only the Main profile, so this script makes
# small aomenc streams of the other profiles: 4:4:4 (8/10/12-bit), 4:2:2
# (8/10/12-bit), 12-bit 4:2:0 and monochrome, RGB with the identity matrix,
# and the coding tools whose chroma geometry depends on the format (screen
# content palette and intra block copy, film grain, superres, lossless,
# quantizer matrices with chroma delta q, 128x128 superblocks, odd sizes,
# realtime and good-quality GOPs with tiles). aomenc refuses 12-bit 4:2:2
# and 4:4:4 input ("Failed to set chroma subsampling x", libaom v3.15), so
# those streams come from libaom through ffmpeg. Their reference is aomdec;
# tools/dev/conformance_manifest.py --suite av1-profiles records it and
# dav1d's output as a cross-check.
#
#   tools/dev/av1_profile_streams.sh [out-dir]
#
# Output: <out-dir>/<name>/<name>.ivf, default
# ~/nanocodec-data/conformance/av1/profiles.
set -euo pipefail

out=${1:-$HOME/nanocodec-data/conformance/av1/profiles}
bin=$HOME/nanocodec-data/anchors/x86_64/bin
clip=/mldata/tracking_original/mot/video/MOT20-01.mp4
frames=20
scratch=$(mktemp -d "${TMPDIR:-/tmp}/nanocodec-av1-profile-streams-XXXXXX")
trap 'rm -rf "$scratch"' EXIT

# The source of a stream: the clip (or ffmpeg's synthetic screen content)
# in an ffmpeg pixel format at a size, as y4m, or for RGB (gbrp: planes G,
# B, R, the AV1 order of the identity matrix) as raw 4:4:4 planes; prints
# aomenc's input options.
source() { # name pix_fmt size [screen]
  local input=(-i "$clip")
  [[ ${4:-} == screen ]] && input=(-f lavfi -i "testsrc2=size=$3:rate=30")
  local filter=(-frames:v "$frames" -vf "scale=$3:flags=bicubic,format=$2")
  if [[ $2 == gbrp* ]]; then
    ffmpeg -v error -nostdin -y "${input[@]}" "${filter[@]}" -f rawvideo "$scratch/$1.src"
    echo "--i444 --width=${3%x*} --height=${3#*x} --fps=30/1"
  else
    ffmpeg -v error -nostdin -y "${input[@]}" "${filter[@]}" -strict -1 -f yuv4mpegpipe \
      "$scratch/$1.src"
  fi
}

common=(--limit="$frames" --threads=1 --ivf --quiet --kf-max-dist=30 --tile-columns=1)
good=(--good --cpu-used=4 --end-usage=vbr --target-bitrate=500 --lag-in-frames=10 --auto-alt-ref=1)
rt=(--rt --cpu-used=8 --end-usage=cbr --target-bitrate=400 --lag-in-frames=0)

# name | pix_fmt | size | profile | bit depth | aomenc options
streams=(
  "p1_444_8_good|yuv444p|352x200|1|8|${good[*]}"
  "p1_444_10_good|yuv444p10le|352x200|1|10|${good[*]}"
  "p1_444_8_rt|yuv444p|352x200|1|8|${rt[*]}"
  "p1_444_8_odd|yuv444p|351x199|1|8|${good[*]}"
  "p1_444_8_sb128_superres|yuv444p|352x200|1|8|${good[*]} --sb-size=128 --superres-mode=1 --superres-denominator=13"
  "p1_444_8_lossless|yuv444p|352x200|1|8|${good[*]} --lossless=1"
  "p1_444_8_qm_deltaq|yuv444p|352x200|1|8|${good[*]} --enable-qm=1 --qm-min=4 --enable-chroma-deltaq=1 --deltaq-mode=1"
  "p1_444_10_grain|yuv444p10le|352x200|1|10|${good[*]} --film-grain-test=1"
  "p1_444_8_screen|yuv444p|352x200|1|8|--good --cpu-used=4 --end-usage=q --cq-level=30 --lag-in-frames=0 --tune-content=screen --enable-palette=1 --enable-intrabc=1 --kf-max-dist=5|screen"
  "p1_444_8_srgb|gbrp|352x200|1|8|${good[*]} --color-primaries=bt709 --transfer-characteristics=srgb --matrix-coefficients=identity"
  "p1_444_10_identity|gbrp10le|352x200|1|10|${good[*]} --color-primaries=bt2020 --transfer-characteristics=smpte2084 --matrix-coefficients=identity"
  "p2_422_8_good|yuv422p|352x200|2|8|${good[*]}"
  "p2_422_10_good|yuv422p10le|352x200|2|10|${good[*]}"
  "p2_422_8_rt|yuv422p|352x200|2|8|${rt[*]}"
  "p2_422_8_odd|yuv422p|351x199|2|8|${good[*]}"
  "p2_422_8_sb128_superres|yuv422p|352x200|2|8|${good[*]} --sb-size=128 --superres-mode=1 --superres-denominator=11"
  "p2_422_10_lossless|yuv422p10le|352x200|2|10|${good[*]} --lossless=1"
  "p2_422_8_grain|yuv422p|352x200|2|8|${good[*]} --film-grain-test=2"
  "p2_422_8_screen|yuv422p|352x200|2|8|--good --cpu-used=4 --end-usage=q --cq-level=30 --lag-in-frames=0 --tune-content=screen --enable-palette=1 --enable-intrabc=1 --kf-max-dist=5|screen"
  "p2_420_12_good|yuv420p12le|352x200|2|12|${good[*]}"
  "p2_420_12_rt_odd|yuv420p12le|351x199|2|12|${rt[*]}"
  "p2_420_12_lossless|yuv420p12le|352x200|2|12|${good[*]} --lossless=1"
  "p2_400_12_good|yuv420p12le|352x200|2|12|${good[*]} --monochrome"
)

for entry in "${streams[@]}"; do
  IFS='|' read -r name pix_fmt size profile depth options kind <<<"$entry"
  input=$(source "$name" "$pix_fmt" "$size" "${kind:-}")
  mkdir -p "$out/$name"
  # shellcheck disable=SC2086 # options are word lists
  "$bin/aomenc" "${common[@]}" $options $input --profile="$profile" --bit-depth="$depth" \
    --input-bit-depth="$depth" -o "$out/$name/$name.ivf" "$scratch/$name.src"
  rm -- "$scratch/$name.src"
  echo "$name: $(stat -c %s "$out/$name/$name.ivf") bytes"
done

# 12-bit 4:2:2 and 4:4:4 through ffmpeg's libaom-av1 (profile 2 follows
# from the depth and format; superres is not among ffmpeg's -aom-params).
# name | pix_fmt | size | -aom-params
ffmpeg_streams=(
  "p2_422_12_good|yuv422p12le|352x200|"
  "p2_444_12_good|yuv444p12le|352x200|"
  "p2_444_12_grain|yuv444p12le|352x200|film-grain-test=3"
  "p2_422_12_grain|yuv422p12le|352x200|film-grain-test=4"
)
for entry in "${ffmpeg_streams[@]}"; do
  IFS='|' read -r name pix_fmt size params <<<"$entry"
  mkdir -p "$out/$name"
  ffmpeg -v error -nostdin -y -i "$clip" -frames:v "$frames" \
    -vf "scale=$size:flags=bicubic,format=$pix_fmt" -c:v libaom-av1 -usage good -cpu-used 4 \
    -b:v 500k -lag-in-frames 10 -auto-alt-ref 1 -g 30 -tile-columns 1 -threads 1 \
    ${params:+-aom-params "$params"} -f ivf "$out/$name/$name.ivf"
  echo "$name: $(stat -c %s "$out/$name/$name.ivf") bytes"
done
