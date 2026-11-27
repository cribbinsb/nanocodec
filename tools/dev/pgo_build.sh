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
#    TARGET_DIR, its binary copied to OUT_BINARY; with PGO_CAPI=1 the C API's
#    static and shared libraries (nanocodec-capi) are built with the same
#    profile and copied beside it (libnanocodec.a, libnanocodec.so).
#
# PGO changes code layout, inlining and branch order only: streams are
# byte-identical to the plain release build (chart_refresh checks every
# timed stream against the scan's bytes; a release checks every shipped
# preset with preset_identity.py). Extra RUSTFLAGS and NC_BUILD_COMMIT
# pass through (RUSTFLAGS joined to the workspace's .cargo/config.toml flags). Env: PGO_CLIPS (paths of 1280x720 I420 clips; default four
# clips of ~/nanocodec-data/clips/main; a release trains on public clips,
# tools/release/release.py), PGO_FRAMES (default 20), PGO_CODECS (default
# every codec; fewer only for a pipe-clean), PGO_CAPI, JOBS (cargo -j,
# default 8). PGO_CODEC_LAYOUT=1 selects aligned codec text groups for the
# Linux CLI final link.
set -euo pipefail
if [ $# -ne 3 ]; then
  sed -n "2,27p" "$0"; exit 2
fi
tree=$(realpath "$1"); target=$(realpath -m "$2"); out=$(realpath -m "$3")
root=$(realpath "$(dirname "$0")/../..")  # this script's checkout: its eval/ lists the presets
clips_dir=$HOME/nanocodec-data/clips/main
read -r -a clips <<< "${PGO_CLIPS:-$clips_dir/crowd_mot20_02.yuv $clips_dir/lowlight_bdd_00c497ae.yuv $clips_dir/pan_jaad_0117.yuv $clips_dir/bodycam_airport.yuv}"
frames=${PGO_FRAMES:-20}
read -r -a codecs <<< "${PGO_CODECS:-h264 hevc av1 vp9 av2 vvc}"
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

# The profile flags go in through `cargo --config`, which joins them to the
# workspace's own (.cargo/config.toml: target-cpu on x86_64); RUSTFLAGS would
# replace those, so a caller's RUSTFLAGS move into the same list.
read -r -a extra_flags <<< "${RUSTFLAGS:-}"
unset RUSTFLAGS
rustflags_config() {  # the --config value: target.<host>.rustflags = [extra flags, stage flags]
  local items="" flag
  for flag in "${extra_flags[@]}" "$@"; do items+="\"$flag\", "; done
  echo "target.$host.rustflags=[${items%, }]"
}
generate_flags=$(rustflags_config "-Cprofile-generate=$work/profiles")
cd "$tree"
CARGO_TARGET_DIR=$target/pgo-generate \
  "$cargo" build --config "$generate_flags" --release -j"$jobs" --bin nanocodec
instrumented=$target/pgo-generate/release/nanocodec

# Every codec's shipped presets (medium and presets/<codec>/*.toml), one line
# each: codec, stem, then its CLI options (nceval.encoders.nanocodec). A tree
# from before the 2026-11-15 preset rename keeps them flat as
# presets/<codec>-*.toml, each its own --preset name.
shipped_presets() {
  presets=crates/nc-encoder/presets
  [ -d "$presets" ] || presets=presets  # a tree from before they moved into the crate
  if [ -d "$presets/h264" ]; then
    python3 -c 'import sys; sys.path.insert(0, sys.argv[1])
from pathlib import Path
from nceval.encoders.nanocodec import preset_cli_args, shipped_presets
for codec in sys.argv[3:]:
    for stem in shipped_presets(codec, Path(sys.argv[2])):
        print(codec, stem, *preset_cli_args(codec, stem, Path(sys.argv[2])))' "$root/eval" "$presets" "${codecs[@]}"
    return
  fi
  for codec in "${codecs[@]}"; do
    echo "$codec default"
    for file in presets/"$codec"-*.toml; do
      # (a codec without flat presets must not end the function on a failed test: set -e)
      if [ -e "$file" ]; then echo "$codec $(basename "$file" .toml) --preset $(basename "$file" .toml)"; fi
    done
  done
}
shipped_presets > "$work/presets.txt"
# a build from before the decoder had a CLI (the history page's early frames) trains on
# its encoder alone
can_decode=1
"$instrumented" decode --help > /dev/null 2>&1 || can_decode=0
# a build from before CBR (2026-11-06, the history page's first day) trains at a fixed QP
rate_args=(--bitrate 600 --buffer-ms 1000)
"$instrumented" encode --help 2>&1 | grep -q -- --bitrate || rate_args=(--qp 30)
while read -r codec preset preset_args; do
  read -r -a preset_args <<< "$preset_args"
  case $codec in h264) ext=264 ;; hevc) ext=hevc ;; vvc) ext=vvc ;; *) ext=ivf ;; esac
  for clip in "${clips[@]}"; do
    stream=$work/streams/$codec-$preset-$(basename "$clip" .yuv).$ext
    "$instrumented" encode --codec "$codec" --input "$clip" --width 1280 --height 720 \
      --fps 30 --frames "$frames" --intra-period 300 "${rate_args[@]}" \
      "${preset_args[@]}" --output "$stream" > /dev/null
    [ "$can_decode" = 0 ] ||
      "$instrumented" decode --codec "$codec" --input "$stream" --frame-md5 /dev/null > /dev/null
  done
done < "$work/presets.txt"

"$profdata" merge -o "$work/merged.profdata" "$work/profiles"
profile_use="-Cprofile-use=$work/merged.profdata"
use_flags=$(rustflags_config "$profile_use")
if [[ "${PGO_CODEC_LAYOUT:-0}" == 1 ]]; then
  case "$host" in
    x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu) ;;
    *) echo "pgo_build: codec text layout needs a native Linux build" >&2; exit 2 ;;
  esac
  codec_flags=$(rustflags_config "$profile_use" "-Clinker=$root/tools/dev/pgo_codec_layout.py")
  NC_PGO_GCC_LD="$sysroot/lib/rustlib/$host/bin/gcc-ld" CARGO_TARGET_DIR=$target \
    "$cargo" build --config "$codec_flags" --release -j"$jobs" --bin nanocodec
  if ! readelf -WS "$target/release/nanocodec" | awk '$2 == ".text.nc_h264" { found=1 } END { exit !found }'; then
    echo "pgo_build: expected codec text groups in the CLI" >&2
    exit 1
  fi
else
  CARGO_TARGET_DIR=$target "$cargo" build --config "$use_flags" --release -j"$jobs" --bin nanocodec
fi
if [ "$out" != "$target/release/nanocodec" ]; then
  mkdir -p "$(dirname "$out")"
  cp "$target/release/nanocodec" "$out"
fi
if [ "${PGO_CAPI:-0}" = 1 ]; then
  CARGO_TARGET_DIR=$target \
    "$cargo" build --config "$use_flags" --release -j"$jobs" -p nanocodec-capi
  cp "$target/release/libnanocodec.a" "$target/release/libnanocodec.so" "$(dirname "$out")/"
fi
rm -r -- "$target/pgo-work/streams"
echo "pgo_build: $out"
