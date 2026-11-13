#!/usr/bin/env bash
# smoke_test.sh — check that the tools installed by build_anchors.sh run and
# agree with each other.
#
# 1. Prints every tool's version.
# 2. Encodes a short synthetic clip (ffmpeg testsrc2) with each anchor encoder,
#    using its single-thread settings.
# 3. Decodes every stream with the matching reference decoder(s) AND ffmpeg,
#    and requires all decoded outputs of a stream to have the same MD5.
# 4. Scores one decode with the vmaf CLI using the vmaf_v0.6.1neg model.
#
# Usage: smoke_test.sh [--prefix DIR] [--work DIR]
#   --prefix  install prefix (default $HOME/nanocodec-data/anchors/$(uname -m))
#   --work    scratch dir for clips and streams (default: a fresh mktemp dir,
#             deleted on success)
# Env: FFMPEG (default: ffmpeg on PATH) — used to make the clip and as the
#      independent decoder.
# Exits non-zero if any tool fails or any decoder disagrees.

set -euo pipefail

readonly WIDTH=352
readonly HEIGHT=288
readonly FPS=30
readonly FRAMES=10
readonly QP=30

PREFIX="$HOME/nanocodec-data/anchors/$(uname -m)"
WORK=""
FFMPEG=${FFMPEG:-ffmpeg}
FAILURES=0

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

parse_args() {
    while (($#)); do
        case $1 in
            --prefix) PREFIX=$2; shift ;;
            --work) WORK=$2; shift ;;
            -h|--help) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;;
            *) die "unknown argument: $1" ;;
        esac
        shift
    done
}

# Run a tool quietly; on failure show the tail of its output. Callers count
# failures (a failed encoder shows up as a MISSING stream).
quiet() {
    local out
    if ! out=$("$@" 2>&1); then
        log "FAIL: $*"
        printf '%s\n' "$out" | tail -n 15 >&2
        return 1
    fi
}

print_versions() {
    log "tool versions:"
    {
        # h264enc has no version flag; report the installed library header.
        echo "openh264    $(grep -m1 -o 'OpenH264 version:[0-9.]*' "$PREFIX/include/wels/codec_ver.h")"
        echo "x264        $(x264 --version | head -n1)"
        echo "x265        $(x265 --version 2>&1 | grep -m1 -o 'version.*')"
        echo "kvazaar     $(kvazaar --version | head -n1)"
        echo "SvtAv1Enc   $(SvtAv1EncApp --version 2>&1 | head -n1)"
        echo "aomenc      $(aomenc --help 2>&1 | grep -m1 -o 'AOMedia.*')"
        echo "aomdec      $(aomdec --help 2>&1 | grep -m1 -o 'AOMedia.*')"
        echo "dav1d       $(dav1d --version 2>&1)"
        echo "ldecod      $(ldecod -v 2>&1 | grep -m1 -io 'JM.*')"
        echo "TAppDecoder $(TAppDecoder 2>&1 | grep -m1 -o 'HM software.*Version \[[^]]*\]')"
        echo "vmaf        $(vmaf --version 2>&1 | head -n1)"
        echo "clang-format $(clang-format --version)"
        echo "ffmpeg      $("$FFMPEG" -version | head -n1)"
    } | sed 's/^/    /' >&2
}

make_clip() {
    "$FFMPEG" -v error -y -f lavfi -i "testsrc2=size=${WIDTH}x${HEIGHT}:rate=$FPS" \
        -frames:v "$FRAMES" -pix_fmt yuv420p -f rawvideo "$WORK/in.yuv"
}

# --- encoders (fixed QP, low delay, one thread) ------------------------------

encode_all() {
    log "encoding $FRAMES frames ${WIDTH}x$HEIGHT at QP $QP"
    rm -f "$WORK"/*.264 "$WORK"/*.hevc "$WORK"/*.ivf
    # openh264: CAMERA_VIDEO_REAL_TIME (utype 0), MEDIUM_COMPLEXITY (1),
    # RC off (-1) with layer QP, frame skip off, one thread, constant SPS id.
    quiet h264enc -org "$WORK/in.yuv" -sw "$WIDTH" -sh "$HEIGHT" -frin "$FPS" \
        -frms "$FRAMES" -bf "$WORK/openh264.264" -numl 1 -dw 0 "$WIDTH" \
        -dh 0 "$HEIGHT" -frout 0 "$FPS" -utype 0 -complexity 1 -rc -1 \
        -lqp 0 "$QP" -fs 0 -threadIdc 1 -numtl 1 -ltr 0 -spsid 0 || true
    # Our openh264 driver (tools/openh264-harness): the "mid" config with two
    # temporal layers, once with CAVLC and once with CABAC.
    local cabac
    for cabac in 0 1; do
        quiet openh264-harness --input "$WORK/in.yuv" \
            --output "$WORK/harness-cabac$cabac.264" --width "$WIDTH" \
            --height "$HEIGHT" --fps "$FPS" --frames "$FRAMES" --qp "$QP" \
            --intra-period 60 --complexity medium --cabac "$cabac" \
            --temporal-layers 2 || true
    done
    quiet x264 --preset ultrafast --tune zerolatency --threads 1 --qp "$QP" \
        --input-res "${WIDTH}x$HEIGHT" --fps "$FPS" --frames "$FRAMES" \
        -o "$WORK/x264.264" "$WORK/in.yuv" || true
    quiet x265 --preset ultrafast --tune zerolatency --pools none \
        --frame-threads 1 --no-wpp --qp "$QP" --input-res "${WIDTH}x$HEIGHT" \
        --fps "$FPS" --frames "$FRAMES" --input "$WORK/in.yuv" \
        -o "$WORK/x265.hevc" || true
    quiet kvazaar --preset ultrafast --threads 0 --owf 0 --no-wpp --qp "$QP" \
        --input-res "${WIDTH}x$HEIGHT" --input-fps "$FPS" -n "$FRAMES" \
        -i "$WORK/in.yuv" -o "$WORK/kvazaar.hevc" || true
    quiet SvtAv1EncApp --preset 12 --pred-struct 1 --rc 0 --aq-mode 0 --qp "$QP" \
        --lp 1 -w "$WIDTH" -h "$HEIGHT" --fps-num "$FPS" --fps-denom 1 -n "$FRAMES" \
        -i "$WORK/in.yuv" -b "$WORK/svtav1.ivf" || true
    quiet aomenc --rt --cpu-used=10 --threads=1 --lag-in-frames=0 \
        --end-usage=q --cq-level="$QP" --i420 -w "$WIDTH" -h "$HEIGHT" \
        --fps="$FPS/1" --limit="$FRAMES" --ivf -o "$WORK/aom.ivf" "$WORK/in.yuv" || true
}

# --- decoders ----------------------------------------------------------------

# decode DECODER STREAM OUT
decode() {
    local decoder=$1 stream=$2 out=$3
    case $decoder in
        ffmpeg) quiet "$FFMPEG" -v error -y -i "$stream" -f rawvideo \
                    -pix_fmt yuv420p "$out" ;;
        # JM writes trace/log files into the cwd, so run it inside WORK.
        jm) (cd "$WORK" && quiet ldecod -d null -i "$stream" -o "$out") ;;
        openh264) quiet h264dec "$stream" "$out" ;;
        hm) quiet TAppDecoder -b "$stream" -o "$out" -d 8 ;;
        aomdec) quiet aomdec --rawvideo -o "$out" "$stream" ;;
        dav1d) quiet dav1d -q -i "$stream" -o "$out" ;;
        *) die "unknown decoder $decoder" ;;
    esac
}

# check_stream STREAM DECODER... — decode with each, require identical MD5s.
check_stream() {
    local stream=$1; shift
    local name dec out sums=() ref=""
    name=$(basename "$stream")
    if [[ ! -s $stream ]]; then
        printf '    %-19s MISSING (encoder failed)\n' "$name" >&2
        FAILURES=$((FAILURES + 1))
        return
    fi
    local verdict=MATCH md5
    for dec in "$@"; do
        out="$WORK/$name.$dec.yuv"
        rm -f "$out"
        if decode "$dec" "$stream" "$out" && [[ -s $out ]]; then
            md5=$(md5sum <"$out" | cut -c1-12)
        else
            md5=FAILED
            verdict=FAILED
        fi
        sums+=("$dec=$md5")
        ref=${ref:-$md5}
        [[ $md5 == "$ref" || $verdict == FAILED ]] || verdict=MISMATCH
    done
    [[ $verdict == MATCH ]] || FAILURES=$((FAILURES + 1))
    printf '    %-19s %-8s %s (%d bytes)\n' "$name" "$verdict" "${sums[*]}" \
        "$(stat -c %s "$stream")" >&2
}

decode_all() {
    log "decoding (all decoders of a stream must give the same MD5):"
    check_stream "$WORK/openh264.264" jm ffmpeg openh264
    check_stream "$WORK/harness-cabac0.264" jm ffmpeg openh264
    check_stream "$WORK/harness-cabac1.264" jm ffmpeg
    check_stream "$WORK/x264.264" jm ffmpeg openh264
    check_stream "$WORK/x265.hevc" hm ffmpeg
    check_stream "$WORK/kvazaar.hevc" hm ffmpeg
    check_stream "$WORK/svtav1.ivf" aomdec dav1d ffmpeg
    check_stream "$WORK/aom.ivf" aomdec dav1d ffmpeg
}

score_vmaf() {
    local model="$PREFIX/share/vmaf/model/vmaf_v0.6.1neg.json"
    [[ -f $model ]] || { log "FAIL: missing $model"; FAILURES=$((FAILURES + 1)); return; }
    if quiet vmaf -r "$WORK/in.yuv" -d "$WORK/x264.264.jm.yuv" -w "$WIDTH" \
        -h "$HEIGHT" -p 420 -b 8 --threads 1 -m "path=$model" \
        --json -o "$WORK/vmaf.json"; then
        log "vmaf_v0.6.1neg mean score (x264 stream): $(python3 -I -c \
            'import json, sys; print(json.load(open(sys.argv[1]))["pooled_metrics"]["vmaf"]["mean"])' \
            "$WORK/vmaf.json")"
    else
        FAILURES=$((FAILURES + 1))
    fi
}

main() {
    parse_args "$@"
    [[ -x $PREFIX/bin/x264 ]] || die "no tools in $PREFIX/bin — run build_anchors.sh first"
    export PATH="$PREFIX/bin:$PATH"
    command -v "$FFMPEG" >/dev/null || die "ffmpeg not found (set FFMPEG)"
    local own_work=0
    if [[ -z $WORK ]]; then
        WORK=$(mktemp -d)
        own_work=1
    fi
    mkdir -p "$WORK"
    WORK=$(cd "$WORK" && pwd)

    print_versions
    make_clip
    encode_all
    decode_all
    score_vmaf

    if ((FAILURES)); then
        die "$FAILURES failure(s); files kept in $WORK"
    fi
    ((own_work)) && rm -rf "$WORK"
    log "smoke test passed"
}

main "$@"
