#!/usr/bin/env bash
# Machine-wide memory guard for builds (owner, 2026-11-12: "don't let the
# machine run out of memory"). docs/PROCEDURES.md "Memory".
#
#   systemd-run --user --unit nc-mem-guard tools/dev/mem-guard.sh
#   (~/nanocodec-data/bin/mem-guard.sh; runs as the nc-mem-guard user unit)
#
# Every POLL_SECONDS: while MemAvailable is below LOW_KIB, or swap use is above
# SWAP_HIGH_KIB while MemAvailable is below 3 x LOW_KIB, pause (SIGSTOP) the
# newest rustc or clippy-driver process, one per poll, and log it. Swap alone
# does not count: stale swapped-out pages can keep it high with tens of GiB
# available, which once held a build paused for 20 minutes. Once
# MemAvailable is above HIGH_KIB (checked first, whatever the swap use), resume (SIGCONT)
# the paused ones, oldest first, one per poll. Pausing is reversible and
# nothing is killed. The landing script's builds (their command line names
# the lander target dir, targets/lander*) are never paused: they hold the
# landing lock every other lander waits for. Log: LOG.
set -uo pipefail

LOW_KIB=$((6 * 1024 * 1024))
HIGH_KIB=$((14 * 1024 * 1024))
SWAP_HIGH_KIB=$((7 * 1024 * 1024))
POLL_SECONDS=5
LOG=${NC_DATA_ROOT:-$HOME/nanocodec-data}/work/mem-guard.log
LANDER_TARGET=targets/lander

paused=()
while :; do
    avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    swap_used=$(awk '/^SwapTotal:/ {t = $2} /^SwapFree:/ {f = $2} END {print t - f}' /proc/meminfo)
    # Resuming depends on MemAvailable alone and comes first, so swap use
    # can never keep a build paused while memory is plentiful.
    if ((avail > HIGH_KIB && ${#paused[@]} > 0)); then
        pid=${paused[0]}
        paused=("${paused[@]:1}")
        kill -CONT "$pid" 2>/dev/null &&
            printf '%s RESUME %s avail=%dM\n' "$(date +%T)" "$pid" $((avail / 1024)) >>"$LOG"
    elif ((avail < LOW_KIB || (swap_used > SWAP_HIGH_KIB && avail < 3 * LOW_KIB))); then
        pid=$(ps -eo pid=,comm= --sort=-start_time | awk '$2 ~ /^(rustc|clippy-driver)$/ {print $1}' |
            while read -r candidate; do
                grep -q "$LANDER_TARGET" "/proc/$candidate/cmdline" 2>/dev/null || { echo "$candidate"; break; }
            done)
        if [[ -n "$pid" ]] && kill -STOP "$pid" 2>/dev/null; then
            paused+=("$pid")
            printf '%s PAUSE %s avail=%dM swap=%dM %s\n' "$(date +%T)" "$pid" $((avail / 1024)) \
                $((swap_used / 1024)) "$(tr '\0' ' ' <"/proc/$pid/cmdline" | cut -c1-120)" >>"$LOG"
        fi
    fi
    sleep "$POLL_SECONDS"
done
