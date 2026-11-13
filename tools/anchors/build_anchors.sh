#!/usr/bin/env bash
# build_anchors.sh — fetch and build the pinned anchor encoders, reference
# decoders and metric tools used by the evaluation suite (eval/README.md
# "Methodology").
#
# Every source is pinned in versions.env (tag + commit hash). The script is
# idempotent: a tool is rebuilt only when its pinned commit or its build
# recipe (the build_<tool> function below) changes, or with --force.
#
# Layout under the prefix:
#   src/<tool>-<ref>/     pristine git checkouts
#   build/<tool>-<ref>/   out-of-tree build dirs (+ .stamp)
#   build/logs/<tool>.log full build log per tool
#   bin/ lib/ include/ share/   installed results
#   venv/                 Python tools: meson, ninja, clang-format
#
# Usage: build_anchors.sh [--prefix DIR] [--jobs N] [--force] [--fetch-only]
#                         [tool ...]
#   tools: python openh264 harness x264 x265 kvazaar svtav1 libaom dav1d jm
#          hm vmaf aominspect (default: all, in that order)
#   harness = our openh264 driver, built from ../openh264-harness (next to
#   this script's directory) against the pinned openh264.
#   aominspect = the pinned libaom built a second time with CONFIG_INSPECTION
#   and CONFIG_ACCOUNTING, installing only its `inspect` example as
#   bin/aom-inspect: per-block modes/MVs/tx/qindex/skip and per-symbol bit
#   accounting for any AV1 stream (eval/nceval/streamstats.py). Kept separate
#   so the normal libaom library and tools stay a plain release build.
#   aomsvc / vpxsvc = the pinned libaom / libvpx built again for their
#   real-time spatial SVC example encoders only (bin/svc_encoder_rtc, what
#   WebRTC's AV1 SVC is modelled on, and bin/vp9_spatial_svc_encoder): the
#   references of the SVC chart (eval/nceval/svc_chart.py). Separate builds,
#   so aomenc / vpxenc and the curves fingerprinted by them never change.
#   Opt-in tools: libvpx avm dav2d vvdec ffmpeg aomsvc vpxsvc (not built by the default
#   all-tools invocation). vvdec installs bin/vvdecapp, the VVC conformance
#   gate's second decoder (eval/nceval/conformance.py). ffmpeg installs
#   bin/ffmpeg-decode, a static decode-only FFmpeg (with the pinned dav1d)
#   that is the gates' libavcodec decoder for H.264/HEVC/VVC/AV1/VP9.
# Default prefix: $HOME/nanocodec-data/anchors/$(uname -m)

set -Eeuo pipefail

readonly SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly ALL_TOOLS=(python openh264 harness x264 x265 kvazaar svtav1 libaom dav1d jm hm vmaf aominspect)
readonly OPTIONAL_TOOLS=(libvpx avm dav2d vvdec ffmpeg aomsvc vpxsvc)
readonly HARNESS_DIR=$(cd "$SCRIPT_DIR/../openh264-harness" 2>/dev/null && pwd || true)
# Minimum CMake that every CMake-based tool here accepts.
readonly CMAKE_MIN_VERSION=3.16

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

PLATFORM=$(uname -m)
PREFIX="$HOME/nanocodec-data/anchors/$PLATFORM"
JOBS=$(nproc)
FORCE=0
FETCH_ONLY=0
CURRENT_TOOL=""
CURRENT_LOG=""

# --- logging ---------------------------------------------------------------

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

on_error() {
    local status=$?
    if [[ -n $CURRENT_LOG && -f $CURRENT_LOG ]]; then
        log "ERROR: '$CURRENT_TOOL' failed (exit $status). Last lines of $CURRENT_LOG:"
        tail -n 30 "$CURRENT_LOG" >&2
    else
        log "ERROR: failed (exit $status) at line ${BASH_LINENO[0]}"
    fi
    exit "$status"
}
trap on_error ERR

# Run a command with its output appended to the current tool's log.
run() {
    printf '+ %s\n' "$*" >>"$CURRENT_LOG"
    "$@" >>"$CURRENT_LOG" 2>&1
}

# --- argument parsing ------------------------------------------------------

usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; }

parse_args() {
    local tools=()
    while (($#)); do
        case $1 in
            --prefix) PREFIX=$2; shift ;;
            --jobs|-j) JOBS=$2; shift ;;
            --force) FORCE=1 ;;
            --fetch-only) FETCH_ONLY=1 ;;
            -h|--help) usage; exit 0 ;;
            -*) die "unknown option: $1" ;;
            *) tools+=("$1") ;;
        esac
        shift
    done
    ((${#tools[@]})) || tools=("${ALL_TOOLS[@]}")
    local t
    for t in "${tools[@]}"; do
        [[ " ${ALL_TOOLS[*]} ${OPTIONAL_TOOLS[*]} " == *" $t "* ]] || die "unknown tool: $t"
    done
    SELECTED_TOOLS=("${tools[@]}")
}

# --- toolchain checks ------------------------------------------------------

version_ge() { [[ $(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1) == "$2" ]]; }

check_toolchain() {
    local tool
    for tool in git gcc g++ make python3; do
        command -v "$tool" >/dev/null || die "required tool not found: $tool"
    done
    command -v cmake >/dev/null || die "cmake not found on PATH"
    local cmake_version
    cmake_version=$(cmake --version | awk 'NR==1 {print $3}')
    version_ge "$cmake_version" "$CMAKE_MIN_VERSION" \
        || die "cmake $cmake_version is too old (need >= $CMAKE_MIN_VERSION); put a newer one first on PATH"
    if [[ $PLATFORM == x86_64 ]]; then
        command -v nasm >/dev/null || die "nasm is required on x86_64 (x264, openh264, libaom, dav1d, vmaf asm)"
    fi
    log "platform=$PLATFORM prefix=$PREFIX jobs=$JOBS"
    log "gcc $(gcc -dumpfullversion), cmake $cmake_version$(command -v nasm >/dev/null && echo ", nasm $(nasm -v | awk '{print $3}')")"
}

# --- source fetching -------------------------------------------------------

# fetch_source NAME URL REF COMMIT -> sets SRC to a checkout of exactly COMMIT.
# Shallow-fetches the tag together with its tag ref, so `git describe` (used by
# x265, dav1d and vmaf to embed their version string) works. Refs that are not
# tags (x264's branch) are fetched by commit hash, then by name for servers
# that refuse fetch-by-hash. Fails if the result is not the pinned commit.
fetch_source() {
    local name=$1 url=$2 ref=$3 commit=$4
    SRC="$PREFIX/src/$name-$ref"
    # Reuse an existing checkout only if it is at the pinned commit and has
    # no stray files (a build that wrote into it would make it non-pristine).
    if [[ -d $SRC/.git && $(git -C "$SRC" rev-parse HEAD) == "$commit" &&
          -z $(git -C "$SRC" status --porcelain --ignored) ]]; then
        return
    fi
    log "$name: fetching $ref ($commit) from $url"
    rm -rf "$SRC"
    mkdir -p "$SRC"
    run git -C "$SRC" init -q
    run git -C "$SRC" remote add origin "$url"
    local tag="refs/tags/$ref"
    run git -C "$SRC" fetch -q --depth 1 origin "+$tag:$tag" \
        || run git -C "$SRC" fetch -q --depth 1 origin "$commit" \
        || run git -C "$SRC" fetch -q --depth 1 origin "$ref"
    run git -C "$SRC" -c advice.detachedHead=false checkout -q FETCH_HEAD
    local got
    got=$(git -C "$SRC" rev-parse HEAD)
    [[ $got == "$commit" ]] \
        || die "$name: $ref is now $got, pinned $commit — update versions.env deliberately"
}

# --- per-tool build recipes --------------------------------------------------
# Each build_<tool> runs with SRC (pristine checkout) and BLD (empty build dir)
# set, and installs into PREFIX. Static linking is preferred so binaries keep
# working if the prefix moves.

cmake_configure() {
    # CMAKE_POLICY_VERSION_MINIMUM lets CMake 4 accept older projects (x265,
    # HM) that still declare cmake_minimum_required < 3.5.
    run cmake -S "$1" -B "$BLD" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        "${@:2}"
}

cmake_build_install() {
    run cmake --build "$BLD" -j "$JOBS"
    run cmake --install "$BLD"
}

meson_build_install() {
    run "$PREFIX/venv/bin/meson" setup "$BLD" "$1" --prefix="$PREFIX" \
        --libdir=lib --buildtype=release --default-library=static "${@:2}"
    run "$PREFIX/venv/bin/meson" compile -C "$BLD" -j "$JOBS"
    run "$PREFIX/venv/bin/meson" install -C "$BLD"
}

# JM, HM and kvazaar write objects/binaries into their source tree whatever
# the build dir, so they are built from a copy in BLD to keep src/ pristine.
copy_source_to_build() {
    run cp -a "$SRC/." "$BLD/tree"
    rm -rf "$BLD/tree/.git"
}

build_python() {
    # Some systems (the Jetson) lack ensurepip and so cannot create a venv
    # with its own pip; there the venv borrows the system pip and
    # --ignore-installed still puts the pinned packages inside the venv.
    if python3 -c 'import ensurepip' 2>/dev/null; then
        run python3 -m venv "$PREFIX/venv"
    else
        run python3 -m venv --without-pip --system-site-packages "$PREFIX/venv"
    fi
    run "$PREFIX/venv/bin/python" -m pip install -q --ignore-installed \
        "meson==$MESON_PIP_VERSION" "ninja==$NINJA_PIP_VERSION" \
        "clang-format==$CLANG_FORMAT_PIP_VERSION"
    mkdir -p "$PREFIX/bin"
    ln -sfn ../venv/bin/clang-format "$PREFIX/bin/clang-format"
}

build_openh264() {
    # The Makefile supports out-of-tree builds when invoked with -f.
    run make -C "$BLD" -f "$SRC/Makefile" -j "$JOBS" BUILDTYPE=Release \
        PREFIX="$PREFIX" all
    run make -C "$BLD" -f "$SRC/Makefile" PREFIX="$PREFIX" install-static
    install -m 755 "$BLD/h264enc" "$BLD/h264dec" "$PREFIX/bin/"
    mkdir -p "$PREFIX/share/openh264"
    install -m 644 "$SRC"/testbin/*.cfg "$PREFIX/share/openh264/"
}

# Needs openh264 installed in PREFIX (headers in include/wels, static lib).
build_harness() {
    [[ -f $PREFIX/lib/libopenh264.a ]] || die "build openh264 before the harness"
    cmake_configure "$HARNESS_DIR" -DOPENH264_PREFIX="$PREFIX"
    cmake_build_install
}

build_x264() {
    (cd "$BLD" && run "$SRC/configure" --prefix="$PREFIX" --enable-static \
        --disable-opencl --disable-lavf --disable-swscale --disable-ffms \
        --disable-gpac --disable-lsmash)
    run make -C "$BLD" -j "$JOBS"
    run make -C "$BLD" install
}

build_x265() {
    local arch_opts=()
    # x265 compiles its SVE/SVE2 kernels with -march=armv9-a, which gcc < 12
    # rejects; the Orin's Cortex-A78AE has no SVE anyway (Neon is kept).
    [[ $PLATFORM == aarch64 ]] \
        && arch_opts=(-DENABLE_SVE=OFF -DENABLE_SVE2=OFF -DENABLE_SVE2_BITPERM=OFF)
    cmake_configure "$SRC/source" -DENABLE_SHARED=OFF -DENABLE_CLI=ON "${arch_opts[@]}"
    cmake_build_install
}

build_kvazaar() {
    copy_source_to_build
    cmake_configure "$BLD/tree" -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF
    cmake_build_install
}

build_svtav1() {
    # SVT-AV1 writes its binaries to <source>/Bin unless told otherwise.
    cmake_configure "$SRC" -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF \
        -DCMAKE_OUTPUT_DIRECTORY="$BLD/Bin"
    cmake_build_install
}

build_libaom() {
    cmake_configure "$SRC" -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF \
        -DENABLE_DOCS=OFF -DENABLE_EXAMPLES=ON -DENABLE_TOOLS=OFF
    cmake_build_install
}

build_libvpx() {
    (cd "$BLD" && run "$SRC/configure" --prefix="$PREFIX" \
        --disable-shared --enable-static --disable-vp8 --enable-vp9 \
        --disable-unit-tests --disable-docs --disable-webm-io)
    run make -C "$BLD" -j "$JOBS"
    run make -C "$BLD" install
}

build_avm() {
    cmake_configure "$SRC" -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF \
        -DENABLE_DOCS=OFF -DENABLE_EXAMPLES=ON -DENABLE_TOOLS=OFF
    # Install tools only: the reference library is not a nanocodec dependency.
    run cmake --build "$BLD" -j "$JOBS" --target avmenc avmdec
    mkdir -p "$PREFIX/bin"
    install -m 755 "$BLD/avmenc" "$BLD/avmdec" "$PREFIX/bin/"
}

build_aominspect() {
    # Inspection and accounting slow the decoder down and change its build,
    # so only the example binary is installed, never the library.
    cmake_configure "$SRC" -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF \
        -DENABLE_DOCS=OFF -DENABLE_EXAMPLES=ON -DENABLE_TOOLS=OFF \
        -DCONFIG_INSPECTION=1 -DCONFIG_ACCOUNTING=1 -DCONFIG_AV1_ENCODER=0 \
        -DCMAKE_C_FLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
    # (inspect.c passes snprintf the whole buffer size at advancing offsets,
    # which the distribution's default _FORTIFY_SOURCE aborts on.)
    run cmake --build "$BLD" -j "$JOBS" --target inspect
    install -m 755 "$BLD/examples/inspect" "$PREFIX/bin/aom-inspect"
}

build_aomsvc() {
    # Only the example encoder is installed, never a second libaom library.
    cmake_configure "$SRC" -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF \
        -DENABLE_DOCS=OFF -DENABLE_EXAMPLES=ON -DENABLE_TOOLS=OFF
    run cmake --build "$BLD" -j "$JOBS" --target svc_encoder_rtc
    install -m 755 "$BLD/examples/svc_encoder_rtc" "$PREFIX/bin/svc_encoder_rtc"
}

build_vpxsvc() {
    # The same configuration as build_libvpx (VP9 only, static); the default
    # target builds the examples too, and only the SVC one is installed.
    (cd "$BLD" && run "$SRC/configure" --prefix="$PREFIX" \
        --disable-shared --enable-static --disable-vp8 --enable-vp9 \
        --disable-unit-tests --disable-docs --disable-webm-io)
    run make -C "$BLD" -j "$JOBS"
    install -m 755 "$BLD/examples/vp9_spatial_svc_encoder" "$PREFIX/bin/vp9_spatial_svc_encoder"
}

build_dav1d() {
    meson_build_install "$SRC" -Denable_tests=false -Denable_tools=true
}

build_dav2d() {
    # Initial independent final-AV2 conformance oracle: scalar 8-bit, not a
    # production decoder-speed anchor. Keep flags identical to validated build.
    meson_build_install "$SRC" -Denable_tests=false -Denable_tools=true \
        -Denable_asm=false -Dbitdepths=8
}

build_vvdec() {
    # VVdeC writes its binaries into <source>/bin/<type>-<link>, so it is
    # built from a copy to keep src/ pristine. Static, app only.
    copy_source_to_build
    cmake_configure "$BLD/tree" -DBUILD_SHARED_LIBS=OFF
    run cmake --build "$BLD" -j "$JOBS" --target vvdecapp
    install -m 755 "$BLD/tree/bin/release-static/vvdecapp" "$PREFIX/bin/vvdecapp"
}

build_ffmpeg() {
    # The gates must not depend on whichever ffmpeg is first on PATH (a conda
    # FFmpeg 7.1.1 mis-decodes VVC ALF streams, versions.env). A static,
    # decode-only build: exactly the demuxers, parsers and decoders the gates
    # use, raw output, and no autodetected system libraries, so the binary's
    # hash is its whole identity (nceval.conformance's verified-stream cache).
    # Its libdav1d is the pinned dav1d, built here as a private static library.
    local ffmpeg_src=$SRC dav1d_prefix="$BLD/dav1d"
    fetch_source dav1d "$DAV1D_URL" "$DAV1D_REF" "$DAV1D_COMMIT"  # sets SRC
    local dav1d_src=$SRC
    SRC=$ffmpeg_src
    # meson finds the venv's pinned ninja on PATH (the Jetson has no system one).
    PATH="$PREFIX/venv/bin:$PATH" run meson setup "$BLD/dav1d-build" "$dav1d_src" \
        --prefix="$dav1d_prefix" --libdir=lib --buildtype=release \
        --default-library=static -Denable_tests=false -Denable_tools=false
    PATH="$PREFIX/venv/bin:$PATH" run meson install -C "$BLD/dav1d-build"
    (cd "$BLD" && PKG_CONFIG_PATH="$dav1d_prefix/lib/pkgconfig" run "$SRC/configure" \
        --prefix="$BLD/install" --pkg-config-flags=--static \
        --disable-autodetect --disable-everything --disable-doc --disable-network \
        --disable-ffplay --disable-ffprobe --disable-shared --enable-static \
        --enable-libdav1d \
        --enable-decoder=h264,hevc,vvc,vp9,libdav1d,rawvideo \
        --enable-parser=h264,hevc,vvc,vp9,av1 \
        --enable-demuxer=h264,hevc,vvc,ivf,rawvideo \
        --enable-bsf=vp9_superframe_split \
        --enable-encoder=rawvideo --enable-muxer=rawvideo \
        --enable-filter=scale,format,null,copy \
        --enable-protocol=file,pipe)
    run make -C "$BLD" -j "$JOBS" ffmpeg
    install -m 755 "$BLD/ffmpeg" "$PREFIX/bin/ffmpeg-decode"
}

build_jm() {
    copy_source_to_build
    # CFLAGS goes through the environment so the Makefile's own `CFLAGS +=`
    # (-std=gnu99 etc.) still applies. -fcommon: JM defines globals in
    # headers, which gcc >= 10 rejects by default.
    CFLAGS="-fcommon -Wno-error" run make -C "$BLD/tree/ldecod" -j "$JOBS"
    install -m 755 "$BLD/tree/bin/ldecod.exe" "$PREFIX/bin/ldecod"
}

build_hm() {
    copy_source_to_build
    # OpenSSL is only for SEI digital signatures, which we never use; leaving
    # the search on can pick up an incompatible libssl (e.g. from conda).
    cmake_configure "$BLD/tree" -DBUILD_STATIC=ON -DENABLE_SEARCH_OPENSSL=OFF
    run cmake --build "$BLD" -j "$JOBS" --target TAppDecoder
    install -m 755 "$BLD/tree/bin/umake/gcc-"*/*/release/TAppDecoder "$PREFIX/bin/TAppDecoder"
}

build_vmaf() {
    meson_build_install "$SRC/libvmaf" -Denable_tests=false -Denable_docs=false \
        -Dbuilt_in_models=true -Denable_float=true
    mkdir -p "$PREFIX/share/vmaf"
    rm -rf "$PREFIX/share/vmaf/model"
    cp -r "$SRC/model" "$PREFIX/share/vmaf/model"
}

# --- driver ----------------------------------------------------------------

# Upper-case variable prefix in versions.env for a tool name.
var_prefix() { echo "${1^^}"; }

build_tool() {
    local tool=$1
    CURRENT_TOOL=$tool
    CURRENT_LOG="$PREFIX/build/logs/$tool.log"
    mkdir -p "$PREFIX/build/logs" "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/share"

    # "commit" identifies the exact input being built: the pinned commit for
    # fetched tools; for the harness, its sources plus the openh264 it links.
    local ref=local commit
    case $tool in
        python) commit=none ;;
        aominspect|aomsvc)
            # Same pinned source as libaom, built with different options.
            ref=$LIBAOM_REF
            commit=$LIBAOM_COMMIT
            : >"$CURRENT_LOG"
            fetch_source libaom "$LIBAOM_URL" "$ref" "$commit" ;;
        vpxsvc)
            # Same pinned source as libvpx, only its SVC example built.
            ref=$LIBVPX_REF
            commit=$LIBVPX_COMMIT
            : >"$CURRENT_LOG"
            fetch_source libvpx "$LIBVPX_URL" "$ref" "$commit" ;;
        ffmpeg)
            # Its private libdav1d is part of the binary: a dav1d bump rebuilds it.
            ref=$FFMPEG_REF
            commit=$FFMPEG_COMMIT-$DAV1D_COMMIT
            : >"$CURRENT_LOG"
            fetch_source ffmpeg "$FFMPEG_URL" "$ref" "$FFMPEG_COMMIT" ;;
        harness)
            [[ -f $HARNESS_DIR/CMakeLists.txt ]] \
                || die "harness sources not found at $SCRIPT_DIR/../openh264-harness"
            commit=$(cat "$HARNESS_DIR"/* | sha1sum | cut -d' ' -f1)-$OPENH264_COMMIT ;;
        *)
            local p
            p=$(var_prefix "$tool")
            local url_var="${p}_URL" ref_var="${p}_REF" commit_var="${p}_COMMIT"
            ref=${!ref_var}
            commit=${!commit_var}
            : >"$CURRENT_LOG"
            fetch_source "$tool" "${!url_var}" "$ref" "$commit" ;;
    esac
    ((FETCH_ONLY)) && return

    BLD="$PREFIX/build/$tool-$ref"
    # The stamp covers the pinned commit, the recipe text and the pip pins, so
    # editing a recipe or bumping a version triggers a rebuild.
    local stamp
    stamp=$({ echo "$commit"; declare -f "build_$tool"; grep PIP_VERSION "$SCRIPT_DIR/versions.env"; } | sha1sum | cut -d' ' -f1)
    if ((!FORCE)) && [[ -f $BLD/.stamp && $(<"$BLD/.stamp") == "$stamp" ]]; then
        log "$tool: up to date ($ref)"
        return
    fi
    log "$tool: building $ref (log: $CURRENT_LOG)"
    : >"$CURRENT_LOG"
    rm -rf "$BLD"
    mkdir -p "$BLD"
    local start=$SECONDS
    "build_$tool"
    echo "$stamp" >"$BLD/.stamp"
    log "$tool: done in $((SECONDS - start)) s"
}

main() {
    parse_args "$@"
    check_toolchain
    mkdir -p "$PREFIX"
    PREFIX=$(cd "$PREFIX" && pwd)
    local tool
    for tool in "${SELECTED_TOOLS[@]}"; do
        build_tool "$tool"
    done
    CURRENT_LOG=""
    if ((FETCH_ONLY)); then
        log "all requested sources are fetched in $PREFIX/src"
    else
        log "all requested tools are installed in $PREFIX"
    fi
}

# Exit explicitly: bash reads scripts lazily, so this stops an edit made to
# the file during a long build from being executed.
main "$@"
exit
