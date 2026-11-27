#!/usr/bin/env bash
# build_wasm_yardsticks.sh — the WebAssembly reference decoders our browser module is timed
# against (tools/web/wasm_yardsticks.py): the pinned native references (versions.env) compiled
# by us with emscripten, single-threaded, -O3 -msimd128.
#
#   tools/web/build_wasm_yardsticks.sh PREFIX [JOBS]
#
# Installs into PREFIX/bin, each a node program (emscripten .js + .wasm, NODERAWFS, so it reads
# the stream from the host file system):
#   libavcodec-bench.js  tools/anchors/libavcodec_bench.c against FFmpeg n8.0.1 (decoders
#                        h264, hevc, vvc, vp9 and libdav1d) and dav1d 1.5.4: the same in-process
#                        harness the native comparison times, so the method is identical.
#   vpx-bench.js         tools/anchors/vpx_bench.c against libvpx v1.16.0 (VP9 decoder).
# and PREFIX/build-flags.json, every configure line and compiler flag.
#
# SIMD: FFmpeg is configured for --arch=wasm with its simd128 extension on, so its hand-written
# WebAssembly SIMD kernels are used: in n8.0.1 these are HEVC's inverse transforms and SAO
# (libavcodec/wasm/hevc); everything else (H.264, VP9, VVC, the rest of HEVC) is C, vectorised
# only by the compiler. dav1d and libvpx have no WebAssembly kernels upstream: their C paths
# (dav1d -Denable_asm=false, libvpx --target=generic-gnu), compiler-vectorised under -msimd128.
# No threads (emscripten without -pthread; every decoder opened one-threaded by its harness).
set -Eeuo pipefail

readonly SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly ANCHORS=$SCRIPT_DIR/../anchors
# shellcheck source=../anchors/versions.env
source "$ANCHORS/versions.env"
#: The emscripten release (emsdk), pinned.
readonly EMSDK_VERSION=4.0.10
readonly EMSDK_URL=https://github.com/emscripten-core/emsdk.git
#: Every C file of every library and harness.
readonly CFLAGS_WASM="-O3 -msimd128"
#: The harnesses' link: node, the host file system, a growable heap and a stack large enough
#: for FFmpeg's and dav1d's decoders (emscripten's default is 64 KiB).
readonly LDFLAGS_WASM="-O3 -msimd128 -sENVIRONMENT=node -sNODERAWFS=1 -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 -sSTACK_SIZE=8388608 -sEXIT_RUNTIME=1"

PREFIX=$(mkdir -p "$1" && cd "$1" && pwd)
JOBS=${2:-$(nproc)}
SRC=$PREFIX/src
BLD=$PREFIX/build
mkdir -p "$SRC" "$BLD" "$PREFIX/bin"

log() { printf '[wasm-yardsticks] %s\n' "$*" >&2; }

fetch() {  # name url ref commit: a pristine checkout at the pinned commit in $SRC/name
    local dir=$SRC/$1
    [[ -d $dir/.git && $(git -C "$dir" rev-parse HEAD) == "$4" ]] && return
    rm -rf "$dir"
    git init -q "$dir"
    git -C "$dir" fetch -q --depth 1 "$2" "+refs/tags/$3:refs/tags/$3" || git -C "$dir" fetch -q --depth 1 "$2" "$4"
    git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
    [[ $(git -C "$dir" rev-parse HEAD) == "$4" ]] || { log "$1: not at the pinned $4"; exit 1; }
}

setup_emsdk() {
    if [[ ! -x $PREFIX/emsdk/emsdk ]]; then
        git clone -q "$EMSDK_URL" "$PREFIX/emsdk"
    fi
    "$PREFIX/emsdk/emsdk" install "$EMSDK_VERSION" >/dev/null
    "$PREFIX/emsdk/emsdk" activate "$EMSDK_VERSION" >/dev/null
    # shellcheck disable=SC1091
    EMSDK_QUIET=1 source "$PREFIX/emsdk/emsdk_env.sh"
    emcc --version | head -1 >&2
}

build_dav1d() {
    fetch dav1d "$DAV1D_URL" "$DAV1D_REF" "$DAV1D_COMMIT"
    cat > "$BLD/emscripten.cross" <<EOF
[binaries]
c = 'emcc'
ar = 'emar'
strip = 'true'
[built-in options]
c_args = ['-O3', '-msimd128']
[host_machine]
system = 'emscripten'
cpu_family = 'wasm32'
cpu = 'wasm32'
endian = 'little'
EOF
    DAV1D_MESON="--cross-file $BLD/emscripten.cross --prefix=$BLD/dav1d --libdir=lib --buildtype=release --default-library=static -Denable_asm=false -Denable_tools=false -Denable_tests=false -Denable_examples=false"
    rm -rf "$BLD/dav1d-build"
    # shellcheck disable=SC2086
    meson setup "$BLD/dav1d-build" "$SRC/dav1d" $DAV1D_MESON
    # meson gives emscripten's threads dependency -pthread (shared memory); the harness is
    # single-threaded and links without it, and dav1d opened with one thread starts none.
    sed -i 's/ -pthread//g' "$BLD/dav1d-build/build.ninja"
    ninja -C "$BLD/dav1d-build" -j "$JOBS" install
    sed -i 's/ -pthread//g; s/ -sPTHREAD_POOL_SIZE=[0-9]*//g' "$BLD/dav1d/lib/pkgconfig/dav1d.pc"
}

build_ffmpeg() {
    fetch ffmpeg "$FFMPEG_URL" "$FFMPEG_REF" "$FFMPEG_COMMIT"
    FFMPEG_CONFIGURE="--target-os=none --arch=wasm32 --enable-cross-compile --cc=emcc --cxx=em++ --ar=emar --ranlib=emranlib --nm=emnm --disable-stripping --disable-runtime-cpudetect --enable-simd128 --disable-pthreads --disable-w32threads --disable-os2threads --disable-autodetect --disable-everything --disable-programs --disable-doc --disable-network --disable-debug --pkg-config-flags=--static --enable-libdav1d --enable-decoder=h264,hevc,vvc,vp9,libdav1d --enable-parser=h264,hevc,vvc,vp9,av1 --enable-demuxer=h264,hevc,vvc,ivf --enable-bsf=vp9_superframe_split --enable-protocol=file --optflags=-O3 --extra-cflags=-msimd128"
    rm -rf "$BLD/ffmpeg"
    mkdir -p "$BLD/ffmpeg"
    # shellcheck disable=SC2086
    (cd "$BLD/ffmpeg" && PKG_CONFIG_PATH="$BLD/dav1d/lib/pkgconfig" "$SRC/ffmpeg/configure" $FFMPEG_CONFIGURE >configure.log)
    grep -q '^#define HAVE_SIMD128 1' "$BLD/ffmpeg/config.h" || { log "ffmpeg: simd128 not enabled"; exit 1; }
    make -C "$BLD/ffmpeg" -j "$JOBS" libavformat/libavformat.a libavcodec/libavcodec.a libavutil/libavutil.a
    # shellcheck disable=SC2086
    emcc $LDFLAGS_WASM -std=c17 -D_POSIX_C_SOURCE=200809L -I"$BLD/ffmpeg" -I"$SRC/ffmpeg" "$ANCHORS/libavcodec_bench.c" \
        -o "$PREFIX/bin/libavcodec-bench.js" "$BLD/ffmpeg/libavformat/libavformat.a" \
        "$BLD/ffmpeg/libavcodec/libavcodec.a" "$BLD/ffmpeg/libavutil/libavutil.a" -L"$BLD/dav1d/lib" -ldav1d
}

build_libvpx() {
    fetch libvpx "$LIBVPX_URL" "$LIBVPX_REF" "$LIBVPX_COMMIT"
    LIBVPX_CONFIGURE="--target=generic-gnu --disable-multithread --disable-runtime-cpu-detect --disable-vp8 --disable-vp9-encoder --enable-vp9-decoder --disable-examples --disable-tools --disable-docs --disable-unit-tests --disable-install-bins --extra-cflags=-msimd128"
    rm -rf "$BLD/libvpx"
    mkdir -p "$BLD/libvpx"
    # shellcheck disable=SC2086
    (cd "$BLD/libvpx" && CFLAGS="$CFLAGS_WASM" emconfigure "$SRC/libvpx/configure" $LIBVPX_CONFIGURE >configure.log)
    emmake make -C "$BLD/libvpx" -j "$JOBS" libvpx.a
    # shellcheck disable=SC2086
    emcc $LDFLAGS_WASM -std=c17 -D_POSIX_C_SOURCE=200809L -I"$ANCHORS" -I"$SRC/libvpx" "$ANCHORS/vpx_bench.c" \
        -o "$PREFIX/bin/vpx-bench.js" "$BLD/libvpx/libvpx.a"
}

record() {
    python3 - "$PREFIX/build-flags.json" <<EOF
import json, sys
json.dump({
    "emscripten": "$(emcc --version | head -1)",
    "emsdk": "$EMSDK_VERSION",
    "cflags": "$CFLAGS_WASM",
    "link": "$LDFLAGS_WASM",
    "ffmpeg": {"ref": "$FFMPEG_REF", "commit": "$FFMPEG_COMMIT", "configure": "$FFMPEG_CONFIGURE",
               "simd128_kernels": "libavcodec/wasm/hevc: HEVC inverse transforms 4x4-32x32 and SAO band/edge filters (8-bit; 10-bit transforms); no other hand-written WebAssembly SIMD in n8.0.1"},
    "dav1d": {"ref": "$DAV1D_REF", "commit": "$DAV1D_COMMIT", "meson": "$DAV1D_MESON",
              "simd128_kernels": "none upstream: C paths, compiler-vectorised"},
    "libvpx": {"ref": "$LIBVPX_REF", "commit": "$LIBVPX_COMMIT", "configure": "$LIBVPX_CONFIGURE",
               "cflags_env": "$CFLAGS_WASM",
               "simd128_kernels": "none upstream: C paths (generic-gnu), compiler-vectorised"},
}, open(sys.argv[1], "w"), indent=1)
EOF
}

setup_emsdk
log "dav1d"; build_dav1d
log "ffmpeg"; build_ffmpeg
log "libvpx"; build_libvpx
record
log "done: $(ls "$PREFIX/bin")"
