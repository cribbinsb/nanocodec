#!/usr/bin/env bash
# Off-means-free check (docs/PROCEDURES.md "Off means free"): does a commit
# that adds a lever switched off by default cost anything with the lever off?
#
#   tools/dev/offpath_ab.sh [--jetson [--no-x86]] [--rounds N] COMMIT
#   tools/dev/offpath_ab.sh [--jetson [--no-x86]] [--rounds N] --head COMMIT [PATH...]
#
# Builds two nanocodec binaries: "off" (without the commit) and "on" (with
# it). By default they are COMMIT's parent and COMMIT itself. With --head
# they are the current HEAD with COMMIT's diff reverse-applied (only PATH...
# when given, e.g. the crates whose hot code it touched, when later edits to
# its other files block the revert) and HEAD itself: use it for a landed
# commit whose cost may have changed since. Then, at every codec's Default
# and Fast preset (31 f crowd 720p CBR 600):
#   - stream identity (byte compare); a DIFF means the commit changed
#     default streams, not only CPU;
#   - x86 callgrind instructions (exact, safe on a busy PC);
#   - with --jetson: jperf instructions and cycles (min of N rounds, default
#     5) of the Default encodes under the shared timing lock.
# A lever that is off must show +0.00% instructions (layout noise of a few
# 0.01% either way is fine) and cycles within their spread.
# Run it from your own worktree, in a MemoryMax scope. Scratch (bitstreams
# only) lives under ~/nanocodec-data/work/offpath-ab and is removed at exit.
set -euo pipefail

CLIP=$HOME/nanocodec-data/clips/main/crowd_mot17_09.yuv
JETSON_CLIP='~/nanocodec-data/work/adopted-fast-reference-native/crowd_mot17_09-150f.yuv'
FRAMES=31
CODECS=(h264 hevc av1 vp9 av2 vvc)

jetson=0
x86=1
rounds=5
at_head=0
while [[ ${1:-} == --* ]]; do
  case $1 in
    --jetson) jetson=1; shift ;;
    --rounds) rounds=$2; shift 2 ;;
    --head) at_head=1; shift ;;
    --no-x86) x86=0; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
if ((jetson)); then
  # The Jetson's SSH target lives in the machine's local settings, not the repo.
  local_env=${NC_DATA_ROOT:-$HOME/nanocodec-data}/local.env
  [[ -f $local_env ]] && source "$local_env"
  JETSON=${NC_JETSON:?"set NC_JETSON=user@host in $local_env or the environment"}
fi
commit=${1:?usage: offpath_ab.sh [--jetson] [--rounds N] [--head] COMMIT [PATH...]}
shift
paths=("$@")

repo=$(git rev-parse --show-toplevel)
short=$(git -C "$repo" rev-parse --short "$commit")
work=$HOME/nanocodec-data/work/offpath-ab/$short
targets=$HOME/nanocodec-data/targets/offpath-ab
mkdir -p "$work"
cleanup() {
  for side in off on; do
    git -C "$repo" worktree remove --force "$work/$side-tree" 2>/dev/null || true
  done
  rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 1' INT TERM

if ((at_head)); then
  git -C "$repo" worktree add --detach "$work/off-tree" HEAD >/dev/null
  git -C "$repo" show "$commit" -- "${paths[@]}" | git -C "$work/off-tree" apply -R
  git -C "$repo" worktree add --detach "$work/on-tree" HEAD >/dev/null
  echo "off = HEAD with $short reverse-applied, on = HEAD"
else
  git -C "$repo" worktree add --detach "$work/off-tree" "$commit^" >/dev/null
  git -C "$repo" worktree add --detach "$work/on-tree" "$commit" >/dev/null
  echo "off = $short^, on = $short"
fi

export PATH=$HOME/.cargo/bin:$PATH
for side in off on; do
  ((x86)) || break
  (cd "$work/$side-tree" &&
    nice -n 19 cargo build --release -j4 -p nc-cli --target-dir "$targets/$side" 2>&1 | tail -1)
done

encode_args() { # CODEC
  echo --codec "$1" --width 1280 --height 720 --fps 30 --frames $FRAMES \
    --intra-period 300 --bitrate 600 --buffer-ms 1000
}

((x86)) && echo "== x86 callgrind instructions, $FRAMES f crowd 720p CBR 600"
for codec in "${CODECS[@]}"; do
  ((x86)) || break
  for preset in default fast; do
    preset_args=()
    [[ $preset == fast ]] && preset_args=(--preset "$codec-fast")
    for side in off on; do
      # shellcheck disable=SC2046
      nice -n 19 valgrind --tool=callgrind --callgrind-out-file="$work/$preset-$side.cg" \
        "$targets/$side/release/nanocodec" encode --input "$CLIP" $(encode_args "$codec") \
        "${preset_args[@]}" --output "$work/$preset-$side.bin" >/dev/null 2>&1 &
    done
  done
  wait
  for preset in default fast; do
    i_off=$(awk '/^summary:/ {print $2}' "$work/$preset-off.cg")
    i_on=$(awk '/^summary:/ {print $2}' "$work/$preset-on.cg")
    same=DIFF
    cmp -s "$work/$preset-off.bin" "$work/$preset-on.bin" && same=SAME
    awk -v c="$codec $preset" -v a="$i_off" -v b="$i_on" -v s="$same" 'BEGIN {
      printf "  %-12s off %8.4fG  on %8.4fG  %+.3f%%  streams %s\n", c, a / 1e9, b / 1e9, 100 * (b / a - 1), s }'
  done
  rm -f "$work/"*.cg "$work/"*.bin
done

if ((jetson)); then
  remote=nanocodec-scratch/offpath-ab/$short
  ssh "$JETSON" "rm -rf ~/$remote"
  # One target dir per side: tar keeps the files' mtimes, so a shared one
  # would take the second tree as already built.
  for side in off on; do
    (cd "$work/$side-tree" && { git ls-files; ls presets/*.toml; } | sort -u | tar --verbatim-files-from -T - -c) |
      ssh "$JETSON" "rm -rf ~/$remote/src && mkdir -p ~/$remote/src && tar -x -C ~/$remote/src"
    ssh "$JETSON" "cd ~/$remote/src && ~/nanocodec-data/jetson-build.sh nice -n 19 \
      ~/.cargo/bin/cargo build --release -j3 -p nc-cli --target-dir ~/$remote/tgt-$side 2>&1 | tail -1 \
      && cp ~/$remote/tgt-$side/release/nanocodec ~/$remote/$side && rm -rf ~/$remote/src"
  done
  scp -q "$repo/tools/dev/jperf.py" "$JETSON:$remote/jperf.py"
  echo "== Jetson jperf (min of $rounds), $FRAMES f crowd 720p CBR 600, Default"
  # All codecs under one hold of the timing lock, output through a remote
  # file so that nothing left behind on the Jetson can hold the ssh channel
  # open (jetson-timed.sh's watchdog sleep once did, for 10 minutes).
  runs=""
  for codec in "${CODECS[@]}"; do
    runs+="echo '  $codec'; python3 ~/$remote/jperf.py --rounds $rounds ~/$remote/off ~/$remote/on \
      -- encode --input $JETSON_CLIP $(encode_args "$codec") --output /dev/null; "
  done
  ssh "$JETSON" "~/nanocodec-data/jetson-timed.sh sh -c \"$runs\" > ~/$remote/jperf.txt 2>&1 < /dev/null;
    grep -v '^==' ~/$remote/jperf.txt"
  ssh "$JETSON" "rm -rf ~/$remote"
fi
