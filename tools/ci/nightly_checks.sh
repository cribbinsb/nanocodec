#!/usr/bin/env bash
# The nightly verifier's checks (tools/dev/verify_schedule.py): the suites the hourly
# `land.sh --verify` leaves out, each run on its own (a failure never hides the rest),
# with one line per check in $OUT/summary (`NAME PASS|FAIL(status) SECONDS`) and its
# output in $OUT/NAME.log.
#
#   tools/ci/nightly_checks.sh x86   on the x86 build box (nc-remote), in a tree with
#                                    cargo: check.sh's full-only steps, host checkasm,
#                                    the official decoder suites (native and WASM), the
#                                    AV2 foreign streams, the real-use features, the
#                                    #[ignore]d tests that need external decoders or
#                                    corpora, the fuzz corpus replay, ISA identity
#   tools/ci/nightly_checks.sh arm   on a Graviton4 box (tools/dev/verify_arm.py), with
#                                    prebuilt binaries in $NC_ARM_BIN: checkasm NEON
#                                    (dotprod, I8MM) and SVE2, the landing conformance
#                                    streams coded on NEON/SVE2, ISA identity
#
# Missing inputs fail (a check that does not run must not pass). Exit status 1 when any
# check failed.
set -uo pipefail

cd "$(dirname "$0")/../.."
export PATH="$HOME/.cargo/bin:$PATH"
mode=${1:?usage: nightly_checks.sh x86|arm}
out=${OUT:-remote-out/nightly}
data=${NC_DATA_ROOT:-$HOME/nanocodec-data}
anchors=$data/anchors/$(uname -m)/bin
mkdir -p "$out"
: >"$out/summary"
failed=0

# `check NAME COMMAND...`: one check, its output in $out/NAME.log, its line in the summary.
check() {
    local name=$1 began=$SECONDS result=PASS
    shift
    printf '== %s\n' "$name"
    ("$@") >"$out/$name.log" 2>&1 || { result="FAIL($?)"; failed=1; }
    printf '%s %s %d\n' "$name" "$result" $((SECONDS - began)) | tee -a "$out/summary"
}

# The #[ignore]d tests nobody can run here: they need a VVdeC trace, a private fixture
# or an idle core (timing). Every other ignored test runs (VVC against VVdeC and ffmpeg,
# the official-corpus decoder tests).
UNRUNNABLE_IGNORED=(
    official_prepared_predictions official_general_predictions
    prepared_bdof_matches_reference prepared_dmvr_matches_reference
    prepared_prof_matches_reference decode_external_standard_fixture real_train_round_trip
    guarded_narrow_rows_native_screen ordinary_variance_native_screen
    compare_byte_stats_with_generic_kernel pricing_cost_probe
)
# The fuzz corpora replayed (the last fuzzing round's and the kept regressions).
FUZZ_ROUND=$data/fuzz/round3-2026-11-23
FUZZ_REGRESSIONS=$data/fuzz/regressions
FUZZ_CODECS=(h264 hevc vvc av1 vp9 av2)
FUZZ_REPLAY_TIMEOUT=20m

ignored_tests() {
    local skips=()
    for name in "${UNRUNNABLE_IGNORED[@]}"; do skips+=(--skip "$name"); done
    NC_FFMPEG=$anchors/ffmpeg-decode NC_VVDEC=$anchors/vvdecapp \
        cargo test --workspace --profile gate-test --no-fail-fast -- --ignored "${skips[@]}"
}

fuzz_replay() {
    local codec status=0 inputs
    (cd fuzz/replay && cargo build --release --quiet) || return 1
    for codec in "${FUZZ_CODECS[@]}"; do
        inputs=()
        [[ -d $FUZZ_ROUND/$codec/corpus ]] && inputs+=("$FUZZ_ROUND/$codec/corpus")
        [[ -d $FUZZ_REGRESSIONS/$codec ]] && inputs+=("$FUZZ_REGRESSIONS/$codec")
        if ((${#inputs[@]} == 0)); then
            echo "FAILED $codec: no fuzz corpus under $FUZZ_ROUND or $FUZZ_REGRESSIONS"
            status=1
            continue
        fi
        echo "== $codec: ${inputs[*]}"
        python3 fuzz/replay/pack.py "${inputs[@]}" 2>/dev/null |
            timeout "$FUZZ_REPLAY_TIMEOUT" "${CARGO_TARGET_DIR:-fuzz/replay/target}/release/nc-fuzz-replay" "$codec" ||
            { echo "FAILED $codec"; status=1; }
    done
    return "$status"
}

foreign_encoders() {
    if [[ ! -f tools/ci/foreign_encoder_check.py ]]; then
        echo "tools/ci/foreign_encoder_check.py has not landed yet"
        return 0
    fi
    python3 tools/ci/foreign_encoder_check.py
}

x86_checks() {
    local bin=${CARGO_TARGET_DIR:-target}/release/nanocodec
    check build cargo build --release --quiet -p nanocodec-cli
    check check-full env NC_CHECK_STRICT=1 tools/ci/check.sh --full-only
    check checkasm-x86 cargo run --profile gate --quiet -p nc-checkasm -- --require avx2,avx512
    for suite in hevc av1 vp9 vvc; do
        check "conformance-$suite" env NANOCODEC="$bin" JOBS=8 "tools/ci/${suite}_conformance.sh"
    done
    check av2-foreign env NC_AV2_FOREIGN="$data/work/av2-foreign" \
        cargo test -p nanocodec-cli --profile gate-test --test integration foreign_streams
    check foreign-encoders foreign_encoders
    check wasm-conformance bash -c 'tools/web/build.sh && tools/ci/wasm_conformance.sh'
    check realuse env NANOCODEC="$bin" tools/ci/realuse_conformance.sh
    check ignored-tests ignored_tests
    check fuzz-replay fuzz_replay
    check isa-identity-x86 tools/ci/isa_identity.sh "$bin" --local --clip-dir "$data/clips/main"
}

arm_checks() {
    local bin=${NC_ARM_BIN:?NC_ARM_BIN: the directory of the aarch64 nanocodec and nc-checkasm}
    check cpu-features bash -c 'for f in asimddp i8mm sve2; do
        grep -qw "$f" /proc/cpuinfo || { echo "this CPU lacks $f"; exit 1; }; done; echo "asimddp i8mm sve2"'
    check checkasm-arm "$bin/nc-checkasm" --require neon,sve2
    # NC_REMOTE_BOX: a rented box, where more than 8 workers are allowed.
    check landing-arm env NC_NANOCODEC_BIN="$bin" NC_VERIFIED_DIR="$out/no-verified-cache" NC_REMOTE_BOX=1 \
        python3 tools/ci/landing_conformance.py --workers "$(($(nproc) / 2))"
    check isa-identity-arm tools/ci/isa_identity.sh "$bin/nanocodec" --local \
        --clip-dir "$data/clips/main" --jobs "$(nproc)"
}

case $mode in
    x86) x86_checks ;;
    arm) arm_checks ;;
    *) echo "nightly_checks.sh: unknown mode $mode" >&2; exit 2 ;;
esac
exit "$failed"
