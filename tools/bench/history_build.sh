#!/usr/bin/env bash
# history_build.sh DAYS_JSON PARALLEL: the PGO builds of the BP-3 history series, one per day
# (docs/benchmark-procedure.md section 14), run on the build box through nc-remote.
#
# Each day's commit is built from its source archive (bench/history-days/src/<commit>.tar.gz)
# with today's tools/dev/pgo_build.sh and toolchain, trained like the benchmark's own build
# (anchors/x86_64-opt-bp3/BUILD.json): PGO_FRAMES=20 on the four cuts of
# eval/clipset/pgo_training_2026-11-23.json, every preset of the codecs that day's tree had
# (the day's "codecs" field). A day whose build fails is reported and left out, never patched
# (coordinator, 2026-11-24). Output per commit, under the job's own tree
# (remote-out/history-builds/<commit>/, fetched with `--fetch remote-out/history-builds`; a
# build box is replaced when idle, so nothing is kept in its home): nanocodec, its SHA-256,
# `--version`, and the build log.
set -euo pipefail
days_json=$1 parallel=${2:-4}
data=$HOME/nanocodec-data/bench/history-days
clips_zst=$HOME/nanocodec-data/release/pgo-clips-2026-11-23-zst
root=$PWD  # the job's checkout: today's pgo_build.sh

clips_dir=$HOME/hist-pgo-clips
mkdir -p "$clips_dir"
for z in "$clips_zst"/*.yuv.zst; do
  out=$clips_dir/$(basename "$z" .zst)
  [ -s "$out" ] || zstd -dqf "$z" -o "$out"
done
export PGO_CLIPS="$(ls "$clips_dir"/*.yuv | tr '\n' ' ')" PGO_FRAMES=20 JOBS=$((96 / parallel))

build_one() {
  local commit=$1 full=$2 codecs=$3
  # scratch beside the job's tree, so the box's disk eviction can reclaim it (aws-cost agent)
  local work=$root/../scratch/$commit out=$root/remote-out/history-builds/$commit
  mkdir -p "$work/src" "$out"
  tar -xzf "$data/src/$commit.tar.gz" -C "$work/src"
  if NC_BUILD_COMMIT=$full PGO_CODECS=$codecs \
      "$root/tools/dev/pgo_build.sh" "$work/src" "$work/target" "$out/nanocodec" \
      > "$out/build.log" 2>&1; then
    sha256sum "$out/nanocodec" | cut -d' ' -f1 > "$out/sha256"
    "$out/nanocodec" --version > "$out/version" 2>&1 || true
    echo "$commit built: $(cat "$out/version")"
  else
    echo "$commit FAILED (tail of $out/build.log):"; tail -5 "$out/build.log"
  fi
}
export -f build_one
export data root
python3 -c 'import json, sys
for d in json.load(open(sys.argv[1])):
    print(d["commit"], d["full"], d["codecs"].replace(" ", ","))' "$days_json" |
  xargs -P "$parallel" -L 1 bash -c 'build_one "$0" "$1" "${2//,/ }"'
