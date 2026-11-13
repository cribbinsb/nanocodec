#!/usr/bin/env bash
# AV1 Main-profile 8-bit decoder conformance: decode every libaom test vector
# in the manifest with nanocodec and require each output frame's MD5 to equal
# the vector's .md5 file (from `aomdec --rawvideo --md5`, one line per frame).
#
# Vectors: ~/nanocodec-data/conformance/av1 (untrusted downloaded data, kept
# out of the repo). Fetch them with --fetch: it downloads the av1-1-b8-*
# vectors and .md5 files named in tools/conformance/av1_manifest.txt from the
# libaom test-data bucket, checks both against the manifest's SHA-1s, and
# rewraps the two Matroska vectors as IVF (tools/conformance/webm_to_ivf.py).
#
# A failing vector reports how many leading frames matched, so partial
# progress (e.g. key frames right, inter frames not yet) stays visible.
#
# Usage: tools/ci/av1_conformance.sh [--fetch] [--jobs N] [NAME_FILTER]
# Env: NANOCODEC (binary, default target/release/nanocodec),
#      NC_AV1_VECTORS (vector directory).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
MANIFEST=$REPO/tools/conformance/av1_manifest.txt
NANOCODEC=${NANOCODEC:-$REPO/target/release/nanocodec}
VECTORS=${NC_AV1_VECTORS:-$HOME/nanocodec-data/conformance/av1}
MIRROR=https://storage.googleapis.com/aom-test-data
JOBS=4
FETCH=0
FILTER=
while [ $# -gt 0 ]; do
  case $1 in
    --fetch) FETCH=1 ;;
    --jobs) JOBS=$2; shift ;;
    *) FILTER=$1 ;;
  esac
  shift
done

if [ "$FETCH" = 1 ]; then
  mkdir -p "$VECTORS/dl" "$VECTORS/ivf"
  while read -r sha md5sha name; do
    for file in "$name" "$name.md5"; do
      [ -s "$VECTORS/dl/$file" ] || curl -sfo "$VECTORS/dl/$file" "$MIRROR/$file"
    done
    printf '%s  %s\n%s  %s\n' "$sha" "$VECTORS/dl/$name" "$md5sha" "$VECTORS/dl/$name.md5" \
      | sha1sum -c --quiet
    ivf=$VECTORS/ivf/${name%.*}.ivf
    case $name in
      *.mkv) python3 -I "$REPO/tools/conformance/webm_to_ivf.py" "$VECTORS/dl/$name" "$ivf" AV01 ;;
      *) cp "$VECTORS/dl/$name" "$ivf" ;;
    esac
  done < <(grep -v '^#' "$MANIFEST")
fi

[ -x "$NANOCODEC" ] || { echo "missing $NANOCODEC (cargo build --release -p nc-cli)"; exit 1; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

check() {
  local name=$1 base=${1%.*}
  local expected=$VECTORS/dl/$name.md5 got=$WORK/$base.md5
  local status=0
  "$NANOCODEC" decode --codec av1 --input "$VECTORS/ivf/$base.ivf" --frame-md5 "$got" \
    2> "$WORK/$base.err" || status=$?
  local frames matched
  frames=$(wc -l < "$expected")
  matched=$(paste -d' ' <(awk '{print $1}' "$expected") "$got" 2>/dev/null \
    | awk '$1 != $2 {exit} {n++} END {print n + 0}')
  if [ "$status" = 0 ] && [ "$matched" = "$frames" ] && [ "$(wc -l < "$got")" = "$frames" ]; then
    echo "PASS $name ($frames frames)"
  elif [ "$status" != 0 ]; then
    echo "FAIL $name: $matched/$frames frames before error: $(tail -1 "$WORK/$base.err")"
  else
    echo "FAIL $name: $matched/$frames frames match"
  fi
}
export -f check
export NANOCODEC VECTORS WORK

grep -v '^#' "$MANIFEST" | awk '{print $3}' | grep -e "${FILTER}" \
  | xargs -P "$JOBS" -I{} bash -c 'check {}' | sort > "$WORK/results"
cat "$WORK/results"
pass=$(grep -c '^PASS' "$WORK/results" || true)
total=$(wc -l < "$WORK/results")
echo "AV1 conformance: $pass / $total vectors bit-exact"
[ "$pass" = "$total" ]
