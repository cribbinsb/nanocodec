#!/usr/bin/env bash
# Run cargo in a memory-capped scope, at most NC_BUILD_SLOTS builds machine-wide
# (owner, 2026-11-12: "don't let the machine run out of memory"; with ~16 agents
# swap reached 5.7 of 8 GiB). docs/PROCEDURES.md "Memory".
#
#   tools/dev/cargo-capped.sh <cargo args...>     (~/nanocodec-data/bin/cargo-capped)
#
# Every agent build, test and clippy run goes through this. It takes one of
# NC_BUILD_SLOTS shared slots (a flock(1) on a slot file, released when cargo
# exits, however it exits) and runs cargo in a user systemd scope capped at
# NC_BUILD_MEM without swap, so one build cannot take the machine down. A
# build waits, polling every SLOT_POLL_SECONDS, while every slot is taken.
#
# NC_BUILD_RESERVED=1 skips the shared slots (the scope still applies): the
# landing script's builds are serialised by the landing lock, so they keep
# their own reserved slot instead of queueing behind agents' builds
# (tools/dev/land.sh runs every cargo call this way, through a `cargo` shim
# on its PATH). Invoked as `cargo` (that shim, a symlink), it runs the real
# cargo from CARGO or ~/.cargo/bin.
set -euo pipefail

DEFAULT_SLOTS=4
DEFAULT_MEM=6G
SLOT_POLL_SECONDS=5

slots=${NC_BUILD_SLOTS:-$DEFAULT_SLOTS}
mem=${NC_BUILD_MEM:-$DEFAULT_MEM}
slot_dir=${NC_DATA_ROOT:-$HOME/nanocodec-data}/locks/build-slots
real_cargo=${CARGO:-$HOME/.cargo/bin/cargo}

capped() {
    exec systemd-run --user --scope --quiet -p MemoryMax="$mem" -p MemorySwapMax=0 -- \
        "$real_cargo" "$@"
}

if [[ "${NC_BUILD_RESERVED:-0}" == 1 ]]; then
    capped "$@"
fi
mkdir -p "$slot_dir"
while :; do
    for i in $(seq 1 "$slots"); do
        exec {fd}>"$slot_dir/slot$i"
        if flock -n "$fd"; then
            capped "$@"  # exec: the scope inherits the locked descriptor until cargo exits
        fi
        exec {fd}>&-
    done
    sleep "$SLOT_POLL_SECONDS"
done
