#!/usr/bin/env bash
# Publish the browser demo to gh-pages quickly: an incremental WASM build in a
# persistent worktree and target directory, the product smoke test only when
# web/ or the embedded crates changed since the last published build, then a
# wait for GitHub Pages to
# build exactly the pushed commit (retried, as Pages builds are flaky).
#
# Usage: tools/web/publish.sh [--build-only] [--smoke] [message]
#   --build-only  build (warm the cache) but do not publish
#   --smoke       run the smoke test even if web/ is unchanged
# Paths: NC_DATA_ROOT (default ~/nanocodec-data) holds the persistent
# worktrees/demo-publish, targets/demo-publish and the gh-pages worktree.
# Never delete targets/demo-publish in disk cleanups: it is what keeps a
# republish to a few minutes.
set -euo pipefail

data=${NC_DATA_ROOT:-$HOME/nanocodec-data}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
branch=${PUBLISH_BRANCH:-variance-integration}
tree=$data/worktrees/demo-publish
target=$data/targets/demo-publish
pages=$data/worktrees/gh-pages
build_only=0 force_smoke=0 message=""
for arg in "$@"; do
  case $arg in
    --build-only) build_only=1 ;;
    --smoke) force_smoke=1 ;;
    *) message=$arg ;;
  esac
done

# One publish at a time: two share the build worktree, and a second one
# checking out its branch mid-build made the first publish the wrong page.
exec 9>"$data/run/publish.lock"
flock 9

if [[ ! -e $tree/.git ]]; then
  git -C "$repo" worktree add --detach "$tree" "$branch"
fi
git -C "$tree" checkout -q --detach "$branch"
head=$(git -C "$tree" rev-parse --short=8 HEAD)

cd "$tree"
CARGO_TARGET_DIR=$target BUILD_JOBS=${BUILD_JOBS:-16} tools/web/build.sh
[[ $build_only == 1 ]] && { echo "built $head (not published)"; exit 0; }

# The product smoke runs when anything that reaches the page differs from the
# last published build: web/ (already tested by land.sh) or the crates the
# module embeds (land.sh runs only the module's node tests for those, so this
# is where a codec or engine change meets the page in a browser).
last=$(git -C "$pages" log -1 --format=%s | sed -n 's/^Browser demo build from \([0-9a-f]*\).*/\1/p')
if [[ $force_smoke == 1 || -z $last ]] || ! git -C "$tree" diff --quiet "$last" HEAD -- web/ crates/ presets/ Cargo.lock 2>/dev/null; then
  python3 web/tests/browser-product-smoke.py
fi

rsync -a --delete --exclude .git --exclude .nojekyll --exclude tests "$tree/web/" "$pages/"
cd "$pages"
git add -A
git commit -q -m "Browser demo build from $head${message:+: $message}"
git push -q origin HEAD:gh-pages
pushed=$(git rev-parse HEAD)

# Wait for Pages to build this commit; re-request on failure.
status=waiting
for attempt in 1 2 3 4; do
  for _ in $(seq 1 40); do
    status=$(gh api repos/cribbinsb/nanocodec/pages/builds/latest \
      --jq "if .commit == \"$pushed\" then .status else \"waiting\" end")
    [[ $status == built || $status == errored ]] && break
    sleep 10
  done
  [[ $status == built ]] && break
  gh api -X POST repos/cribbinsb/nanocodec/pages/builds >/dev/null
done
echo "published $head: pages $status"
[[ $status == built ]]
