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
#                         [--opt [--pgo-clips DIR]] [tool ...]
#   --opt = the reference build the charts time (docs/PROCEDURES.md "Reference
#   builds"): each encoder with the link-time (LTO) and profile-guided (PGO)
#   optimisation its own build system offers, runtime ISA dispatch as shipped,
#   no -march. Default prefix $HOME/nanocodec-data/anchors/$(uname -m)-opt.
#   PGO trains on --pgo-clips (default: the release's PGO training set,
#   $HOME/nanocodec-data/release/pgo-clips-2026-11-23, 1280x720 I420 .yuv,
#   eval/clipset/pgo_training_2026-11-23.json: disjoint from every dataset)
#   with the benchmark record's own commands (pgo_train.py: every charted
#   family of the tool, CBR and Fixed QP). A project's own PGO route is used
#   where it has one (SVT-AV1, libvpx); the others get gcc's generic
#   -fprofile-generate / -fprofile-use (gcc_pgo_flags). openh264 is trained
#   through the harness; aomenc and svc_encoder_rtc share one libaom profile.
#   Each tool's exact configure line, compiler flags and PGO training go
#   to PREFIX/build-flags.json (tools/anchors/build_record.py), which
#   tools/anchors/build-flags.json copies for the pages.
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
#   vpxhbd = the pinned libvpx built again with --enable-vp9-highbitdepth,
#   installing only bin/vpxdec-hbd: the VP9 profile 2 (10/12-bit) reference
#   of the vp9-libvpx conformance suite. Separate, so the 8-bit vpxdec of
#   the landing gates is unchanged.
#   dav2dhbd = the pinned dav2d built again with its 16-bit template
#   (-Dbitdepths=8,16), installing only bin/dav2d-hbd: the AV2 10- and
#   12-bit reference beside avmdec. Separate, so the 8-bit oracle and its
#   speed anchor are unchanged.
#   Opt-in tools: libvpx avm dav2d dav2dasm dav2dhbd vvdec vvenc ffmpeg avbench decbench aomsvc vpxsvc vpxhbd (not built by
#   the default all-tools invocation). vvdec installs bin/vvdecapp, the VVC conformance
#   gate's second decoder (eval/nceval/conformance.py). vvenc installs
#   bin/vvencFFapp and its configuration files under share/vvenc/cfg: the
#   fixed-QP VVC reference (nceval.cbr vvenc-ld-<preset>). libvpx installs
#   vpxenc / vpxdec, the VP9 anchor and gate decoder. ffmpeg installs
#   bin/ffmpeg-decode, a static decode-only FFmpeg (with the pinned dav1d)
#   that is the gates' libavcodec decoder for H.264/HEVC/VVC/AV1/VP9.
#   avbench (after ffmpeg) installs bin/libavcodec-bench, the same libraries
#   timed in process (libavcodec_bench.c, tools/dev/decoder_speed.py).
#   decbench (after libvpx, vvdec, avm and/or dav2d in the same prefix) installs
#   bin/vpx-bench, bin/vvdec-bench, bin/avm-bench and bin/dav2d-bench, those
#   libraries timed in process the same way (vpx_bench.c, vvdec_bench.c,
#   avm_bench.c, dav2d_bench.c), for each library the prefix holds.
# Default prefix: $HOME/nanocodec-data/anchors/$(uname -m)

set -Eeuo pipefail

readonly SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly ALL_TOOLS=(python openh264 harness x264 x265 kvazaar svtav1 libaom dav1d jm hm vmaf aominspect)
readonly OPTIONAL_TOOLS=(libvpx avm dav2d dav2dasm dav2dhbd vvdec vvenc ffmpeg avbench decbench aomsvc vpxsvc vpxhbd)
readonly HARNESS_DIR=$(cd "$SCRIPT_DIR/../openh264-harness" 2>/dev/null && pwd || true)
# Minimum CMake that every CMake-based tool here accepts.
readonly CMAKE_MIN_VERSION=3.16

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

PLATFORM=$(uname -m)
# macOS (the Mac series, 2026-11-26): the platform compiler is Apple clang, whose PGO writes raw
# profiles that llvm-profdata merges before the -fprofile-use build (pgo_train), and whose
# archiver for LTO objects is the system ar.
DARWIN=0
[[ $(uname -s) == Darwin ]] && DARWIN=1
LTO_AR=gcc-ar
((DARWIN)) && LTO_AR=ar
PREFIX=""
JOBS=$(nproc)
FORCE=0
FETCH_ONLY=0
# 1 for the timed reference build (--opt): LTO and PGO where the project offers them.
OPT=0
PGO_CLIPS="$HOME/nanocodec-data/release/pgo-clips-2026-11-23"
# The PGO clips' size (tools/release/release.py TRAIN_WIDTH/HEIGHT).
readonly PGO_WIDTH=1280 PGO_HEIGHT=720
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
            --opt) OPT=1 ;;
            --pgo-clips) PGO_CLIPS=$2; shift ;;
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
    [[ -n $PREFIX ]] || PREFIX="$HOME/nanocodec-data/anchors/$PLATFORM$( ((OPT)) && echo -opt || true)"
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
    # HM) that still declare cmake_minimum_required < 3.5. The compile
    # database is what build_record.py reads the exact flags from.
    run cmake -S "$1" -B "$BLD" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        "${@:2}"
}

# --- the timed reference build (--opt) ---------------------------------------
# Every encoder gets the optimisation its own build system offers, nothing
# more: no -march (each keeps its shipped runtime ISA dispatch), no flags it
# does not document. REC_LTO / REC_PGO say what a recipe used, for the record.

# CMake's link-time optimisation, for projects whose CMake honours
# CMAKE_INTERPROCEDURAL_OPTIMIZATION (policy CMP0069; set NEW for projects
# that declare an older minimum, such as x265). Empty without --opt.
cmake_lto_opts() {
    LTO_OPTS=()
    REC_LTO="no (plain build)"
    if ((OPT)); then
        LTO_OPTS=(-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POLICY_DEFAULT_CMP0069=NEW)
        REC_LTO="CMake CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
    fi
}

# The PGO training clips (*.yuv, PGO_WIDTH x PGO_HEIGHT I420) into PGO_CLIP_LIST.
pgo_clip_list() {
    PGO_CLIP_LIST=("$PGO_CLIPS"/*.yuv)
    [[ -f ${PGO_CLIP_LIST[0]} ]] || die "no PGO training clips (*.yuv) in $PGO_CLIPS"
}

# Frames in one PGO training clip.
pgo_frames() { echo $(($(stat -c %s "$1") / (PGO_WIDTH * PGO_HEIGHT * 3 / 2))); }

# The PGO training description for the record: the route, the commands, the clips.
pgo_record() {
    local names
    names=$(printf '%s ' "${PGO_CLIP_LIST[@]##*/}")
    REC_PGO="$1; training: $2; clips: ${names% } ($(pgo_frames "${PGO_CLIP_LIST[0]}") frames each)"
}

# Train instrumented binaries on the record's commands for TOOL (pgo_train.py);
# $1 tool, $2 directory holding the binaries under the record's names, $3 the
# VVenC cfg directory (optional). Sets PGO_TRAINING to its description.
pgo_train() {
    pgo_clip_list
    local cfg=()
    [[ -n ${3:-} ]] && cfg=(--cfg "$3")
    printf '+ pgo_train %s\n' "$1" >>"$CURRENT_LOG"
    PGO_TRAINING=$(python3 -I "$SCRIPT_DIR/pgo_train.py" --tool "$1" --bin "$2" "${cfg[@]}" \
        --work "$BLD/pgo-train" --jobs "$JOBS" "${PGO_CLIP_LIST[@]}" 2>>"$CURRENT_LOG" | tail -n1)
    [[ -n $PGO_TRAINING ]] || die "$1: PGO training failed (log $CURRENT_LOG)"
    rm -rf "$BLD/pgo-train"
    # clang's raw profiles (every route here writes them under BLD/pgo) are merged into the
    # default.profdata that -fprofile-use=BLD/pgo reads.
    if ((DARWIN)) && compgen -G "$BLD/pgo/*.profraw" >/dev/null; then
        run xcrun llvm-profdata merge -o "$BLD/pgo/default.profdata" "$BLD/pgo"
    fi
}

# gcc's generic PGO for projects with no route of their own: every compile and
# link of the instrumented build gets PGO_GEN, the training runs, and the same
# build dir is rebuilt with PGO_USE (gcc finds each object's profile under
# BLD/pgo by its mangled object path, so the rebuild must use the same paths).
# -fprofile-partial-training keeps code the training never reached optimised
# as in a plain build; -Wno-missing-profile quiets objects it never reached.
gcc_pgo_flags() {
    PGO_DIR="$BLD/pgo"
    PGO_GEN="-fprofile-generate=$PGO_DIR"
    PGO_USE="-fprofile-use=$PGO_DIR -fprofile-partial-training -Wno-missing-profile"
    REC_GCC_PGO="gcc -fprofile-generate / -fprofile-use -fprofile-partial-training"
    if ((DARWIN)); then
        # clang reads PGO_DIR/default.profdata, which pgo_train merges from the raw profiles.
        PGO_USE="-fprofile-use=$PGO_DIR -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date"
        REC_GCC_PGO="clang -fprofile-generate / llvm-profdata merge / -fprofile-use"
    fi
}

# A CMake project under gcc PGO (--opt): configure with $1 as the extra C, C++
# and linker flags; $2 the source dir; the rest configure arguments.
cmake_configure_flags() {
    local flags=$1 src=$2
    shift 2
    cmake_configure "$src" -DCMAKE_C_FLAGS="$flags" -DCMAKE_CXX_FLAGS="$flags" \
        -DCMAKE_EXE_LINKER_FLAGS="$flags" "$@"
}

# A directory of links to the instrumented binaries under the record's names.
pgo_bin_dir() {
    local name
    PGO_BIN="$BLD/pgo-bin"
    mkdir -p "$PGO_BIN"
    while (($#)); do
        name=$1
        ln -sfn "$2" "$PGO_BIN/$name"
        shift 2
    done
}

# Write this tool's exact build (configure line, compile and link flags, LTO,
# PGO) into PREFIX/build-flags.json. $1: a source file whose compile line
# stands for the encoder's generic C/C++ code.
record_build() {
    python3 -I "$SCRIPT_DIR/build_record.py" --prefix "$PREFIX" --tool "$CURRENT_TOOL" \
        --build "$BLD" --log "$CURRENT_LOG" --source "$1" \
        --ref "$ref" --commit "$commit" --lto "${REC_LTO:-no}" --pgo "${REC_PGO:-no}" \
        --profile "$( ((OPT)) && echo opt || echo plain)"
}

cmake_build_install() {
    run cmake --build "$BLD" -j "$JOBS"
    run cmake --install "$BLD"
}

meson_build_install() {
    # meson finds the venv's pinned ninja on PATH (the Jetson has no system one).
    local PATH="$PREFIX/venv/bin:$PATH"
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

openh264_make() {
    run make -C "$BLD" -f "$SRC/Makefile" -j "$JOBS" BUILDTYPE=Release V=Yes "$@"
}

build_openh264() {
    # The Makefile supports out-of-tree builds when invoked with -f.
    # V=Yes prints each compile line for the record. Its Makefile offers
    # neither LTO nor PGO (CFLAGS_OPT=-O3 is its whole optimisation setting),
    # so --opt adds gcc's to the static library (all the record runs, through
    # the harness): -flto (archived with gcc-ar, so the harness's link
    # optimises across the library) and generic PGO, trained through a harness
    # built against the instrumented library (BLD/stage).
    REC_LTO="no (its Makefile offers none)" REC_PGO="no (none offered)"
    if ((OPT)); then
        gcc_pgo_flags
        local lto="-flto=auto" stage="$BLD/stage" lib=libopenh264.a
        openh264_make CFLAGS_OPT="-O3 $lto $PGO_GEN" AR="$LTO_AR" "$lib"
        openh264_make PREFIX="$stage" install-static
        local harness="$BLD/harness-gen"
        run cmake -S "$HARNESS_DIR" -B "$harness" -DCMAKE_BUILD_TYPE=Release \
            -DOPENH264_PREFIX="$stage" -DCMAKE_EXE_LINKER_FLAGS="$lto $PGO_GEN"
        run cmake --build "$harness"
        pgo_bin_dir openh264-harness "$harness/openh264-harness"
        pgo_train openh264-harness "$PGO_BIN"
        openh264_make clean
        openh264_make CFLAGS_OPT="-O3 $lto $PGO_USE" AR="$LTO_AR" "$lib"
        openh264_make PREFIX="$PREFIX" install-static
        REC_LTO="-flto=auto (CFLAGS_OPT, archived with $LTO_AR; the harness links with -flto)"
        pgo_record "$REC_GCC_PGO, trained through openh264-harness" "$PGO_TRAINING"
    else
        openh264_make PREFIX="$PREFIX" all
        openh264_make PREFIX="$PREFIX" install-static
        install -m 755 "$BLD/h264enc" "$BLD/h264dec" "$PREFIX/bin/"
    fi
    mkdir -p "$PREFIX/share/openh264"
    install -m 644 "$SRC"/testbin/*.cfg "$PREFIX/share/openh264/"
    record_build codec/encoder/core/src/encoder.cpp
}

# Needs openh264 installed in PREFIX (headers in include/wels, static lib).
build_harness() {
    [[ -f $PREFIX/lib/libopenh264.a ]] || die "build openh264 before the harness"
    local lto=()
    REC_LTO="no (a thin driver; the encoder is libopenh264.a)"
    REC_PGO="no (a thin driver; libopenh264.a carries the profile)"
    if ((OPT)); then
        lto=(-DCMAKE_EXE_LINKER_FLAGS=-flto=auto)
        REC_LTO="links with -flto=auto: link-time optimisation of the LTO-built libopenh264.a"
    fi
    cmake_configure "$HARNESS_DIR" -DOPENH264_PREFIX="$PREFIX" "${lto[@]}"
    cmake_build_install
    record_build openh264_harness.c
}

build_x264() {
    local lto=()
    REC_LTO="no (plain build)" REC_PGO="no (plain build)"
    if ((OPT)) && ((DARWIN)); then
        # Apple clang's LTO build of x264 crashes at its first frame (exit 139; the plain and
        # the -fprofile-generate builds run: measured 2026-11-26), so macOS builds it without.
        REC_LTO="no (x264 --enable-lto crashes under Apple clang)"
    elif ((OPT)); then
        lto=(--enable-lto)
        REC_LTO="x264 configure --enable-lto"
    fi
    (cd "$BLD" && run "$SRC/configure" --prefix="$PREFIX" --enable-static \
        --disable-opencl --disable-lavf --disable-swscale --disable-ffms \
        --disable-gpac --disable-lsmash "${lto[@]}")
    if ((OPT)); then
        # x264's own PGO hooks (CFLAGSPROF / LDFLAGSPROF, what its make
        # fprofiled uses) with the benchmark's commands in place of
        # fprofiled's eight fixed option sets.
        gcc_pgo_flags
        run make -C "$BLD" -j "$JOBS" x264 CFLAGSPROF="$PGO_GEN" LDFLAGSPROF="$PGO_GEN"
        pgo_bin_dir x264 "$BLD/x264"
        pgo_train x264 "$PGO_BIN"
        run make -C "$BLD" clean
        run make -C "$BLD" -j "$JOBS" CFLAGSPROF="$PGO_USE" LDFLAGSPROF="$PGO_USE"
        pgo_record "x264 CFLAGSPROF/LDFLAGSPROF (its fprofiled hooks), $REC_GCC_PGO" "$PGO_TRAINING"
    else
        run make -C "$BLD" -j "$JOBS"
    fi
    run make -C "$BLD" install
    record_build encoder/encoder.c
}

build_x265() {
    local arch_opts=()
    # x265 compiles its SVE/SVE2 kernels with -march=armv9-a, which gcc < 12
    # rejects; the Orin's Cortex-A78AE has no SVE anyway (Neon is kept). With
    # gcc >= 12 (the Graviton build) they are kept: x265 picks them at run
    # time (getauxval, source/common/aarch64/cpu.h).
    # Apple silicon has no SVE (and x265 detects it through Linux's getauxval only).
    if ((DARWIN)) || { [[ $PLATFORM == aarch64 ]] && ! version_ge "$(gcc -dumpversion)" 12; }; then
        arch_opts=(-DENABLE_SVE=OFF -DENABLE_SVE2=OFF -DENABLE_SVE2_BITPERM=OFF)
    fi
    cmake_lto_opts
    local args=("$SRC/source" -DENABLE_SHARED=OFF -DENABLE_CLI=ON "${arch_opts[@]}" "${LTO_OPTS[@]}")
    REC_PGO="no (plain build)"
    if ((OPT)); then
        # x265 documents no PGO route: gcc's generic one.
        gcc_pgo_flags
        cmake_configure_flags "$PGO_GEN" "${args[@]}"
        run cmake --build "$BLD" -j "$JOBS" --target cli
        pgo_bin_dir x265 "$BLD/x265"
        pgo_train x265 "$PGO_BIN"
        cmake_configure_flags "$PGO_USE" "${args[@]}"
        pgo_record "$REC_GCC_PGO (x265 documents no PGO route)" "$PGO_TRAINING"
    else
        cmake_configure "${args[@]}"
    fi
    cmake_build_install
    record_build encoder/encoder.cpp
}

build_kvazaar() {
    copy_source_to_build
    cmake_lto_opts
    local args=("$BLD/tree" -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF "${LTO_OPTS[@]}")
    REC_PGO="no (plain build)"
    if ((OPT)); then
        # Kvazaar documents no PGO route: gcc's generic one.
        gcc_pgo_flags
        cmake_configure_flags "$PGO_GEN" "${args[@]}"
        run cmake --build "$BLD" -j "$JOBS"
        pgo_bin_dir kvazaar "$(find "$BLD" -name kvazaar -type f -perm -u+x | head -n1)"
        pgo_train kvazaar "$PGO_BIN"
        cmake_configure_flags "$PGO_USE" "${args[@]}"
        pgo_record "$REC_GCC_PGO (Kvazaar documents no PGO route)" "$PGO_TRAINING"
    else
        cmake_configure "${args[@]}"
    fi
    cmake_build_install
    record_build src/encoderstate.c
}

build_svtav1() {
    # SVT-AV1 writes its binaries to <source>/Bin unless told otherwise. Its
    # own SVT_AV1_LTO is on by default with gcc >= 9 (plain build included).
    local pgo=()
    REC_LTO="SVT-AV1 SVT_AV1_LTO=ON (its default with gcc >= 9)" REC_PGO="no (plain build)"
    ((OPT)) && pgo=(-DSVT_AV1_PGO=ON -DSVT_AV1_PGO_DIR="$BLD/pgo")
    # Under clang, SVT-AV1's PGOCompileUse merges *.profraw in the build directory, so the
    # profiles must be written there (its default SVT_AV1_PGO_DIR).
    ((OPT)) && ((DARWIN)) && pgo=(-DSVT_AV1_PGO=ON -DSVT_AV1_PGO_DIR="$BLD")
    cmake_configure "$SRC" -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF \
        -DCMAKE_OUTPUT_DIRECTORY="$BLD/Bin" -DSVT_AV1_LTO=ON "${pgo[@]}"
    if ((OPT)); then
        # SVT-AV1's PGO targets (Docs/CommonQuestions.md), with our training
        # in place of its PGOGenerateProfile (preset 2 with film grain, an
        # offline workload): the real-time presets the charts run.
        run cmake --build "$BLD" -j "$JOBS" --target PGOCompileGen
        mkdir -p "$BLD/pgo"
        pgo_bin_dir SvtAv1EncApp "$(find "$BLD/Bin" -name SvtAv1EncApp -type f | head -n1)"
        pgo_train SvtAv1EncApp "$PGO_BIN"
        run cmake --build "$BLD" -j "$JOBS" --target PGOCompileUse
        pgo_record "SVT-AV1 SVT_AV1_PGO (PGOCompileGen/PGOCompileUse)" "$PGO_TRAINING"
    else
        run cmake --build "$BLD" -j "$JOBS"
    fi
    run cmake --install "$BLD"
    record_build Source/Lib/Globals/enc_handle.c
}

build_libaom() {
    cmake_lto_opts
    local args=("$SRC" -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF -DENABLE_DOCS=OFF
                -DENABLE_EXAMPLES=ON -DENABLE_TOOLS=OFF "${LTO_OPTS[@]}")
    REC_PGO="no (plain build)"
    if ((OPT)); then
        # libaom documents no PGO route: gcc's generic one, one profile for
        # both programs the record runs (aomenc in the main rows, its real-time
        # example svc_encoder_rtc in the +drop rows), installed side by side. It is
        # trained through aomenc: pgo_train leaves the +drop rows out.
        gcc_pgo_flags
        cmake_configure_flags "$PGO_GEN" "${args[@]}"
        run cmake --build "$BLD" -j "$JOBS" --target aomenc svc_encoder_rtc
        pgo_bin_dir aomenc "$BLD/aomenc" svc_encoder_rtc "$BLD/examples/svc_encoder_rtc"
        pgo_train aomenc "$PGO_BIN"
        cmake_configure_flags "$PGO_USE" "${args[@]}"
        pgo_record "$REC_GCC_PGO (libaom documents no PGO route), one profile for aomenc and svc_encoder_rtc" "$PGO_TRAINING"
    else
        cmake_configure "${args[@]}"
    fi
    cmake_build_install
    if ((OPT)); then
        install -m 755 "$BLD/examples/svc_encoder_rtc" "$PREFIX/bin/svc_encoder_rtc"
    fi
    record_build av1/encoder/encoder.c
}

libvpx_configure() {
    (cd "$BLD" && run "$SRC/configure" --prefix="$PREFIX" \
        --disable-shared --enable-static --disable-vp8 --enable-vp9 \
        --disable-unit-tests --disable-docs --disable-webm-io "$@")
}

build_libvpx() {
    # libvpx's configure offers no LTO switch; its PGO switch is
    # --use-profile (build/make/configure.sh), fed by an instrumented build
    # of the same tree (same object paths, so gcc finds each profile).
    REC_LTO="no (its configure offers none)" REC_PGO="no (plain build)"
    if ((OPT)); then
        local profile="$BLD/pgo"
        mkdir -p "$profile"
        LDFLAGS="-fprofile-generate=$profile" \
            libvpx_configure --extra-cflags="-fprofile-generate=$profile"
        run make -C "$BLD" -j "$JOBS" V=1
        pgo_bin_dir vpxenc "$BLD/vpxenc"
        pgo_train vpxenc "$PGO_BIN"
        run make -C "$BLD" clean
        libvpx_configure --use-profile="$profile"
        pgo_record "libvpx configure --use-profile" "$PGO_TRAINING"
    else
        libvpx_configure
    fi
    run make -C "$BLD" -j "$JOBS" V=1
    run make -C "$BLD" install
    record_build vp9/encoder/vp9_encoder.c
}

build_avm() {
    cmake_lto_opts
    cmake_configure "$SRC" -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF \
        -DENABLE_DOCS=OFF -DENABLE_EXAMPLES=ON -DENABLE_TOOLS=OFF "${LTO_OPTS[@]}"
    # Install tools only: the reference library is not a nanocodec dependency.
    run cmake --build "$BLD" -j "$JOBS" --target avmenc avmdec
    mkdir -p "$PREFIX/bin"
    install -m 755 "$BLD/avmenc" "$BLD/avmdec" "$PREFIX/bin/"
    REC_PGO="no (AVM documents no PGO route)" record_build av2/encoder/encoder.c
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

build_vpxhbd() {
    # build_libvpx's configuration plus high bit depth; only vpxdec is
    # installed, renamed.
    (cd "$BLD" && run "$SRC/configure" --prefix="$PREFIX" \
        --disable-shared --enable-static --disable-vp8 --enable-vp9 \
        --enable-vp9-highbitdepth --disable-unit-tests --disable-docs --disable-webm-io)
    run make -C "$BLD" -j "$JOBS"
    install -m 755 "$BLD/vpxdec" "$PREFIX/bin/vpxdec-hbd"
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

build_dav2dhbd() {
    # build_dav2d's oracle with both templates; only the CLI is installed,
    # renamed.
    local PATH="$PREFIX/venv/bin:$PATH"
    run "$PREFIX/venv/bin/meson" setup "$BLD" "$SRC" --buildtype=release \
        --default-library=static -Denable_tests=false -Denable_tools=true \
        -Denable_asm=false -Dbitdepths=8,16
    run "$PREFIX/venv/bin/meson" compile -C "$BLD" -j "$JOBS"
    install -m 755 "$BLD/tools/dav2d" "$PREFIX/bin/dav2d-hbd"
}

build_dav2dasm() {
    # The same pinned dav2d with its assembly on: the decode-speed reference
    # (decbench's dav2d-bench links it in preference to the scalar oracle above).
    # Only its library and CLI are built; nothing is installed over the oracle.
    local PATH="$PREFIX/venv/bin:$PATH"
    run "$PREFIX/venv/bin/meson" setup "$BLD" "$SRC" --buildtype=release \
        --default-library=static -Denable_tests=false -Denable_tools=true \
        -Denable_asm=true -Dbitdepths=8
    run "$PREFIX/venv/bin/meson" compile -C "$BLD" -j "$JOBS"
}

build_vvdec() {
    # VVdeC writes its binaries into <source>/bin/<type>-<link>, so it is
    # built from a copy to keep src/ pristine. Static, app only.
    copy_source_to_build
    cmake_configure "$BLD/tree" -DBUILD_SHARED_LIBS=OFF
    run cmake --build "$BLD" -j "$JOBS" --target vvdecapp
    install -m 755 "$BLD/tree/bin/release-static/vvdecapp" "$PREFIX/bin/vvdecapp"
}

build_vvenc() {
    # Like VVdeC, VVenC writes its binaries into <source>/bin/<type>-<link>:
    # built from a copy. Static, the full-featured app only (it reads the
    # .cfg preset files, installed beside it).
    # VVENC_ENABLE_LINK_TIME_OPT is on by default (its CMakeLists.txt).
    copy_source_to_build
    local args=("$BLD/tree" -DBUILD_SHARED_LIBS=OFF -DVVENC_ENABLE_LINK_TIME_OPT=ON)
    local app="$BLD/tree/bin/release-static/vvencFFapp"
    REC_PGO="no (plain build)"
    if ((OPT)); then
        # VVenC documents no PGO route: gcc's generic one. Its default -Werror stops
        # the instrumented build on aarch64 (gcc 15: -Wmaybe-uninitialized in simde's
        # NEON loads under -fprofile-generate, 2026-11-23); warnings change no code.
        args+=(-DVVENC_ENABLE_WERROR=OFF)
        gcc_pgo_flags
        cmake_configure_flags "$PGO_GEN" "${args[@]}"
        run cmake --build "$BLD" -j "$JOBS" --target vvencFFapp
        pgo_bin_dir vvencFFapp "$app"
        pgo_train vvencFFapp "$PGO_BIN" "$SRC/cfg"
        cmake_configure_flags "$PGO_USE" "${args[@]}"
        pgo_record "$REC_GCC_PGO (VVenC documents no PGO route)" "$PGO_TRAINING"
    else
        cmake_configure "${args[@]}"
    fi
    run cmake --build "$BLD" -j "$JOBS" --target vvencFFapp
    install -m 755 "$app" "$PREFIX/bin/vvencFFapp"
    mkdir -p "$PREFIX/share/vvenc"
    rm -rf "$PREFIX/share/vvenc/cfg"
    cp -r "$SRC/cfg" "$PREFIX/share/vvenc/cfg"
    # our picture-structure overrides, read after a preset file (nceval.cbr ~flat)
    mkdir -p "$PREFIX/share/vvenc/cfg/nanocodec"
    install -m 644 "$SCRIPT_DIR"/vvenc/*.cfg "$PREFIX/share/vvenc/cfg/nanocodec/"
    REC_LTO="VVenC VVENC_ENABLE_LINK_TIME_OPT=ON (its default)" \
        record_build source/Lib/EncoderLib/EncLib.cpp
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

build_avbench() {
    # Links the ffmpeg tool's build (its static libraries and private dav1d).
    local ffmpeg_bld="$PREFIX/build/ffmpeg-$FFMPEG_REF"
    [[ -f $ffmpeg_bld/libavcodec/libavcodec.a ]] || die "build ffmpeg before avbench"
    run "${CC:-cc}" -O2 -std=c17 -I"$ffmpeg_bld" -I"$PREFIX/src/ffmpeg-$FFMPEG_REF" \
        "$SCRIPT_DIR/libavcodec_bench.c" -o "$PREFIX/bin/libavcodec-bench" \
        "$ffmpeg_bld/libavformat/libavformat.a" "$ffmpeg_bld/libavcodec/libavcodec.a" \
        "$ffmpeg_bld/libavutil/libavutil.a" -L"$ffmpeg_bld/dav1d/lib" -ldav1d \
        -pthread -lm -latomic -ldl
}

build_decbench() {
    # Each harness links its library's build in this prefix; a library the
    # prefix lacks is skipped (x86 keeps libvpx and AVM in its vp9-av2 prefix).
    local cc=${CC:-cc} cxx=${CXX:-c++} built=0 dir
    local cflags=(-O2 -std=c17 -D_POSIX_C_SOURCE=200809L -I"$SCRIPT_DIR")
    if [[ -f $PREFIX/lib/libvpx.a ]]; then
        run "$cc" "${cflags[@]}" -I"$PREFIX/include" "$SCRIPT_DIR/vpx_bench.c" \
            -o "$PREFIX/bin/vpx-bench" "$PREFIX/lib/libvpx.a" -pthread -lm
        built=1
    fi
    dir="$PREFIX/build/vvdec-$VVDEC_REF"
    if [[ -f $dir/tree/lib/release-static/libvvdec.a ]]; then
        run "$cc" "${cflags[@]}" -I"$dir/tree/include" -I"$dir" -c "$SCRIPT_DIR/vvdec_bench.c" \
            -o "$BLD/vvdec_bench.o"
        run "$cxx" -o "$PREFIX/bin/vvdec-bench" "$BLD/vvdec_bench.o" \
            "$dir/tree/lib/release-static/libvvdec.a" -pthread -lm
        built=1
    fi
    dir="$PREFIX/build/avm-$AVM_REF"
    if [[ -f $dir/libavm.a ]]; then
        run "$cc" "${cflags[@]}" -I"$PREFIX/src/avm-$AVM_REF" -I"$dir" -c "$SCRIPT_DIR/avm_bench.c" \
            -o "$BLD/avm_bench.o"
        # avmdec's own link line (libavm.a and its TensorFlow Lite and abseil
        # dependencies) with the harness in place of avmdec's objects.
        local link=() token
        read -ra link <"$dir/CMakeFiles/avmdec.dir/link.txt"
        local args=()
        for token in "${link[@]}"; do
            [[ $token == *.o || $token == -o || $token == avmdec || $token == -Wl,--dependency-file=* ]] \
                || args+=("$token")
        done
        # the object before the libraries, so the linker resolves it against them
        (cd "$dir" && run "${args[0]}" "$BLD/avm_bench.o" "${args[@]:1}" -o "$PREFIX/bin/avm-bench")
        built=1
    fi
    dir="$PREFIX/build/dav2dasm-$DAV2D_REF"
    [[ -f $dir/src/libdav2d.a ]] || dir="$PREFIX/build/dav2d-$DAV2D_REF"
    if [[ -f $dir/src/libdav2d.a ]]; then
        run "$cc" "${cflags[@]}" -I"$PREFIX/src/dav2d-$DAV2D_REF/include" -I"$dir/include" \
            "$SCRIPT_DIR/dav2d_bench.c" -o "$PREFIX/bin/dav2d-bench" "$dir/src/libdav2d.a" \
            -pthread -lm -ldl
        built=1
    fi
    ((built)) || die "decbench: no libvpx, vvdec, avm or dav2d build in $PREFIX"
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
        vpxsvc|vpxhbd)
            # Same pinned source as libvpx, built with different options.
            ref=$LIBVPX_REF
            commit=$LIBVPX_COMMIT
            : >"$CURRENT_LOG"
            fetch_source libvpx "$LIBVPX_URL" "$ref" "$commit" ;;
        dav2dasm|dav2dhbd)
            # Same pinned source as dav2d, built with its assembly or the
            # 16-bit template.
            ref=$DAV2D_REF
            commit=$DAV2D_COMMIT
            : >"$CURRENT_LOG"
            fetch_source dav2d "$DAV2D_URL" "$ref" "$commit" ;;
        ffmpeg)
            # Its private libdav1d is part of the binary: a dav1d bump rebuilds it.
            ref=$FFMPEG_REF
            commit=$FFMPEG_COMMIT-$DAV1D_COMMIT
            : >"$CURRENT_LOG"
            fetch_source ffmpeg "$FFMPEG_URL" "$ref" "$FFMPEG_COMMIT" ;;
        decbench)
            # The harness sources plus the libraries they link.
            commit=$(cat "$SCRIPT_DIR"/decode_bench.h "$SCRIPT_DIR"/{vpx,vvdec,avm,dav2d}_bench.c | sha1sum | cut -d' ' -f1)-$LIBVPX_COMMIT-$VVDEC_COMMIT-$AVM_COMMIT-$DAV2D_COMMIT ;;
        avbench)
            # The harness source plus the FFmpeg (and dav1d) it links.
            commit=$(sha1sum <"$SCRIPT_DIR/libavcodec_bench.c" | cut -d' ' -f1)-$FFMPEG_COMMIT-$DAV1D_COMMIT ;;
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
    stamp=$({ echo "$commit opt=$OPT"; declare -f "build_$tool"; grep PIP_VERSION "$SCRIPT_DIR/versions.env"; } | sha1sum | cut -d' ' -f1)
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
