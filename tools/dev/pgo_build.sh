#!/usr/bin/env bash
# pgo_build.sh TREE TARGET_DIR OUT_BINARY: profile-guided release build of
# nanocodec (docs/PROCEDURES.md "Build profile").
#
# 1. An instrumented release build of TREE (`-Cprofile-generate`) in
#    TARGET_DIR/pgo-generate.
# 2. Training: every shipped preset of every codec encodes the first
#    PGO_FRAMES frames of each PGO_CLIPS clip in CBR (600 kbps, 1 s buffer),
#    and our decoder decodes each stream. The clips are not the chart's
#    timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1).
# 3. llvm-profdata from the toolchain (`rustup component add llvm-tools`)
#    merges the profiles, and the release build (`-Cprofile-use`) goes to
#    TARGET_DIR, its binary copied to OUT_BINARY.
#
# PGO changes code layout, inlining and branch order only: streams are
# byte-identical to the plain release build (chart_refresh checks every
# timed stream against the scan's bytes). Extra RUSTFLAGS and NC_BUILD_COMMIT
# pass through. Env: PGO_CLIPS (paths of 1280x720 I420 clips; default four
# clips of ~/nanocodec-data/clips/main), PGO_FRAMES (default 20), JOBS
# (cargo -j, default 8).
set -euo pipefail
if [ $# -ne 3 ]; then
  sed -n "2,22p" "$0"; exit 2
fi
tree=$(realpath "$1"); target=$(realpath -m "$2"); out=$(realpath -m "$3")
clips_dir=$HOME/nanocodec-data/clips/main
read -r -a clips <<< "${PGO_CLIPS:-$clips_dir/crowd_mot20_02.yuv $clips_dir/lowlight_bdd_00c497ae.yuv $clips_dir/pan_jaad_0117.yuv $clips_dir/bodycam_airport.yuv}"
frames=${PGO_FRAMES:-20}
jobs=${JOBS:-8}
cargo=$HOME/.cargo/bin/cargo
sysroot=$("$HOME/.cargo/bin/rustc" --print sysroot)
host=$("$HOME/.cargo/bin/rustc" -vV | sed -n 's/^host: //p')
profdata=$sysroot/lib/rustlib/$host/bin/llvm-profdata
[ -x "$profdata" ] || { echo "pgo_build: no $profdata (rustup component add llvm-tools)" >&2; exit 1; }

# Profiles and training streams live in a fresh directory under the target.
work=$target/pgo-work
mkdir -p "$target"
[ ! -e "$work" ] || rm -r -- "$target/pgo-work"
mkdir -p "$work/profiles" "$work/streams"

base_flags=${RUSTFLAGS:-}
cd "$tree"
RUSTFLAGS="$base_flags -Cprofile-generate=$work/profiles" CARGO_TARGET_DIR=$target/pgo-generate \
  "$cargo" build --release -j"$jobs" -p nc-cli
instrumented=$target/pgo-generate/release/nanocodec

# Every codec with its shipped presets (presets/<codec>-*.toml) and its default.
for codec in h264 hevc av1 vp9 av2 vvc; do
  presets=(default)
  for file in presets/"$codec"-*.toml; do
    [ -e "$file" ] && presets+=("$(basename "$file" .toml)")
  done
  case $codec in h264) ext=264 ;; hevc) ext=hevc ;; vvc) ext=vvc ;; *) ext=ivf ;; esac
  for preset in "${presets[@]}"; do
    for clip in "${clips[@]}"; do
      stream=$work/streams/$codec-$preset-$(basename "$clip" .yuv).$ext
      preset_args=()
      [ "$preset" = default ] || preset_args=(--preset "$preset")
      "$instrumented" encode --codec "$codec" --input "$clip" --width 1280 --height 720 \
        --fps 30 --frames "$frames" --intra-period 300 --bitrate 600 --buffer-ms 1000 \
        "${preset_args[@]}" --output "$stream" > /dev/null
      "$instrumented" decode --codec "$codec" --input "$stream" --frame-md5 /dev/null > /dev/null
    done
  done
done

"$profdata" merge -o "$work/merged.profdata" "$work/profiles"
RUSTFLAGS="$base_flags -Cprofile-use=$work/merged.profdata" CARGO_TARGET_DIR=$target \
  "$cargo" build --release -j"$jobs" -p nc-cli
if [ "$out" != "$target/release/nanocodec" ]; then
  mkdir -p "$(dirname "$out")"
  cp "$target/release/nanocodec" "$out"
fi
rm -r -- "$target/pgo-work/streams"
echo "pgo_build: $out"
