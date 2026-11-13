#!/usr/bin/env bash
# VP9 Profile 0 decoder conformance: decode every libvpx test vector in the
# manifest with nanocodec and require each shown frame's MD5 to equal the
# vector's .md5 file (from `vpxdec --md5 --i420`).
#
# Vectors: ~/nanocodec-data/conformance/vp9 (untrusted downloaded data, kept
# out of the repo). Fetch them with --fetch: it downloads the vp90-2-*
# vectors and .md5 files named in tools/conformance/vp9_manifest.txt from
# the libvpx test-data mirror, checks them against the SHA-1 list in the
# manifest, and rewraps WebM as IVF (tools/conformance/webm_to_ivf.py).
#
# Usage: tools/ci/vp9_conformance.sh [--fetch] [--jobs N] [NAME_FILTER]
# Env: NANOCODEC (binary, default target/release/nanocodec),
#      NC_VP9_VECTORS (vector directory).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
MANIFEST=$REPO/tools/conformance/vp9_manifest.txt
NANOCODEC=${NANOCODEC:-$REPO/target/release/nanocodec}
VECTORS=${NC_VP9_VECTORS:-$HOME/nanocodec-data/conformance/vp9}
MIRROR=https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx
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
  while read -r sha name; do
    for file in "$name" "$name.md5"; do
      [ -s "$VECTORS/dl/$file" ] || curl -sfo "$VECTORS/dl/$file" "$MIRROR/$file"
    done
    echo "$sha  $VECTORS/dl/$name" | sha1sum -c --quiet
    ivf=$VECTORS/ivf/${name%.*}.ivf
    case $name in
      *.webm) python3 -I "$REPO/tools/conformance/webm_to_ivf.py" "$VECTORS/dl/$name" "$ivf" ;;
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
  if ! "$NANOCODEC" decode --codec vp9 --input "$VECTORS/ivf/$base.ivf" --frame-md5 "$got" \
      2> "$WORK/$base.err"; then
    echo "FAIL $name: $(tail -1 "$WORK/$base.err")"
    return
  fi
  local frames
  frames=$(wc -l < "$expected")
  if cmp -s <(awk '{print $1}' "$expected") "$got"; then
    echo "PASS $name ($frames frames)"
  else
    local first
    first=$(diff <(awk '{print $1}' "$expected") "$got" | head -1)
    echo "FAIL $name: MD5 mismatch at $first of $frames frames ($(wc -l < "$got") decoded)"
  fi
}
export -f check
export NANOCODEC VECTORS WORK

grep -v '^#' "$MANIFEST" | awk '{print $2}' | grep -e "${FILTER}" \
  | xargs -P "$JOBS" -I{} bash -c 'check {}' | sort > "$WORK/results"
cat "$WORK/results"
pass=$(grep -c '^PASS' "$WORK/results" || true)
total=$(wc -l < "$WORK/results")
echo "VP9 conformance: $pass / $total vectors bit-exact"
[ "$pass" = "$total" ]
