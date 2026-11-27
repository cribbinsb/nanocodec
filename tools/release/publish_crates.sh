#!/usr/bin/env bash
# Publish the workspace's crates on crates.io, in dependency order.
#
#   tools/release/publish_crates.sh                 dry run (the default): package and
#                                                   build every crate, upload nothing
#   tools/release/publish_crates.sh --yes-publish   publish for real
#
# A published version can never be removed (only yanked), so the real run
# needs the explicit flag and a clean tree at a commit. It is resumable: a
# crate whose version is already in the index is skipped, and each crate is
# waited for in the index before the next one (which depends on it) is sent.
# It uses the login of whoever runs it (`cargo login`); it never logs in,
# changes owners or yanks.
set -euo pipefail

# Dependency order: each crate after everything it depends on.
CRATES=(
  nanocodec-common nanocodec-rtp nanocodec-dsp nanocodec-analysis nanocodec-encoder
  nanocodec-h264 nanocodec-hevc nanocodec-vvc nanocodec-av1 nanocodec-vp9 nanocodec-av2
  nanocodec nanocodec-capi nanocodec-cli nanocodec-web
)
INDEX=https://index.crates.io
# Seconds to wait for a published crate to appear in the index.
INDEX_TIMEOUT=600
INDEX_POLL=10

mode=dry-run
case "${1:-}" in
  "" | --dry-run) ;;
  --yes-publish) mode=publish ;;
  *) echo "usage: $0 [--dry-run | --yes-publish]" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/../.."
jobs=${BUILD_JOBS:-8}

# The index path of a crate name (names here are all 4+ characters).
index_url() { echo "$INDEX/${1:0:2}/${1:2:2}/$1"; }

# Whether version $2 of crate $1 is in the index.
published() {
  curl -fsS -A "nanocodec-release-script" "$(index_url "$1")" 2>/dev/null |
    grep -q "\"vers\":\"$2\""
}

version_of() {
  cargo metadata --no-deps --format-version 1 --offline |
    python3 -c 'import json, sys
name = sys.argv[1]
print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == name))' "$1"
}

if [ "$mode" = dry-run ]; then
  # One workspace dry run: cargo packages every crate and builds each from its
  # package against the others' packages, as crates.io users will get them.
  packages=()
  for crate in "${CRATES[@]}"; do packages+=(-p "$crate"); done
  cargo publish --dry-run -j "$jobs" "${packages[@]}"
  echo "dry run passed for ${#CRATES[@]} crates; nothing was uploaded."
  echo "to publish: $0 --yes-publish"
  exit 0
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "the tree has uncommitted changes: publish from a clean commit" >&2
  exit 1
fi

for crate in "${CRATES[@]}"; do
  version=$(version_of "$crate")
  if published "$crate" "$version"; then
    echo "== $crate $version is already published: skipped"
    continue
  fi
  echo "== publishing $crate $version"
  cargo publish -j "$jobs" -p "$crate"
  waited=0
  until published "$crate" "$version"; do
    if [ "$waited" -ge "$INDEX_TIMEOUT" ]; then
      echo "$crate $version did not appear in the index after ${INDEX_TIMEOUT}s;" \
        "run this script again to resume" >&2
      exit 1
    fi
    sleep "$INDEX_POLL"
    waited=$((waited + INDEX_POLL))
  done
done
echo "published ${#CRATES[@]} crates."
