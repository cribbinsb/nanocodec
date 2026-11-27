#!/usr/bin/env bash
# Publish the browser demo to gh-pages quickly: an incremental WASM build in a
# persistent worktree and target directory, then a wait for GitHub Pages to
# build exactly the pushed commit (retried, as Pages builds are flaky).
#
# Run it from worktrees/demo-publish (the persistent publish tree, below), never
# from an agent's worktree: the script that runs is the one in the tree it is
# started from, and an agent's tree holds work in progress.
#
# Usage: tools/web/publish.sh [--build-only] [--smoke | --no-smoke] [message]
#   --build-only  build (warm the cache) but do not publish
#   --smoke       run the browser product smoke test before pushing
#   --no-smoke    the default: build, push, wait for Pages, nothing else
# No test runs by default: every web/ and crate change already passed the
# browser tests in landing (tools/dev/land.sh) and the verifier, and the smoke
# test made a publish several minutes longer on a loaded machine.
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
build_only=0 smoke=0 message=""
for arg in "$@"; do
  case $arg in
    --build-only) build_only=1 ;;
    --smoke) smoke=1 ;;
    --no-smoke) smoke=0 ;;
    *) message=$arg ;;
  esac
done

# One publish at a time: two share the build worktree, and a second one
# checking out its branch mid-build made the first publish the wrong page.
# Each stage prints as it starts, with the seconds since the start: a publish
# that once sat silent for 15 min (cargo waiting on the package cache lock, then
# a cold build) now shows where it is.
began=$SECONDS
stage() { printf '== publish +%ds: %s\n' $((SECONDS - began)) "$*"; }
exec 9>"$data/run/publish.lock"
flock -n 9 || { stage "waiting for another publish (run/publish.lock)"; flock 9; }

if [[ ! -e $tree/.git ]]; then
  git -C "$repo" worktree add --detach "$tree" "$branch"
fi
git -C "$tree" checkout -q --detach "$branch"
head=$(git -C "$tree" rev-parse --short=8 HEAD)

cd "$tree"
# The WASM modules depend only on crates/, the lockfile and the build script: a
# data- or page-only publish (new chart numbers, a page fix) reuses the last
# build instead of re-running the bindgen/opt steps (minutes on a loaded box).
inputs=$(git rev-parse HEAD:crates HEAD:Cargo.lock HEAD:tools/web/build.sh | sha256sum | cut -c1-16)
stamp=$target/publish-inputs
if [[ -f $stamp && $(cat "$stamp") == "$inputs" && -f web/pkg/nc_web_bg.wasm ]]; then
  stage "WASM inputs unchanged since the last build ($inputs): build skipped"
else
  stage "building $head in $tree (target $target)"
  CARGO_TARGET_DIR=$target BUILD_JOBS=${BUILD_JOBS:-16} tools/web/build.sh
  echo "$inputs" > "$stamp"
fi
[[ $build_only == 1 ]] && { stage "built $head (not published)"; exit 0; }

if [[ $smoke == 1 ]]; then
  stage "browser product smoke"
  python3 web/tests/browser-product-smoke.py
fi

stage "pushing to gh-pages"
rsync -a --delete --exclude .git --exclude .nojekyll --exclude tests "$tree/web/" "$pages/"
cd "$pages"
git add -A
git commit -q -m "Browser demo build from $head${message:+: $message}"
git push -q origin HEAD:gh-pages
pushed=$(git rev-parse HEAD)

# Wait for Pages to build this commit; re-request on failure.
stage "waiting for GitHub Pages to build ${pushed:0:8}"
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
stage "published $head: pages $status"
[[ $status == built ]]
