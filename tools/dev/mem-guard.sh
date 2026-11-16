#!/usr/bin/env bash
# Machine-wide memory guard for builds (owner, 2026-11-12: "don't let the
# machine run out of memory"). docs/PROCEDURES.md "Memory".
#
#   systemd-run --user --unit nc-mem-guard tools/dev/mem-guard.sh
#   (~/nanocodec-data/bin/mem-guard.sh; runs as the nc-mem-guard user unit)
#
# Every POLL_SECONDS: while MemAvailable is below LOW_KIB, pause (SIGSTOP) the
# newest running rustc or clippy-driver process, one per poll, and log it.
# Swap use is not a signal: after a build spike the 8 GiB swap stays full of
# stale desktop pages, and a swap rule kept up to 7 builds paused with 17 GiB
# available (2026-11-16). Each build also has its own MemoryMax scope. Once
# MemAvailable is above HIGH_KIB (checked first), resume (SIGCONT)
# the paused ones, oldest first, one per poll. Pausing is reversible and
# nothing is killed. The landing script's builds (their command line names
# the lander target dir, targets/lander*) are never paused: they hold the
# landing lock every other lander waits for. Log: LOG.
set -uo pipefail

LOW_KIB=$((6 * 1024 * 1024))
HIGH_KIB=$((14 * 1024 * 1024))
POLL_SECONDS=5
LOG=${NC_DATA_ROOT:-$HOME/nanocodec-data}/work/mem-guard.log
LANDER_TARGET=targets/lander

paused=()
while :; do
    avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    # Resuming comes first, so a build is never held paused while memory is plentiful.
    if ((avail > HIGH_KIB && ${#paused[@]} > 0)); then
        pid=${paused[0]}
        paused=("${paused[@]:1}")
        kill -CONT "$pid" 2>/dev/null &&
            printf '%s RESUME %s avail=%dM\n' "$(date +%T)" "$pid" $((avail / 1024)) >>"$LOG"
    elif ((avail < LOW_KIB)); then
        pid=$(ps -eo pid=,stat=,comm= --sort=-start_time | awk '$2 !~ /^T/ && $3 ~ /^(rustc|clippy-driver)$/ {print $1}' |
            while read -r candidate; do
                grep -q "$LANDER_TARGET" "/proc/$candidate/cmdline" 2>/dev/null || { echo "$candidate"; break; }
            done)
        if [[ -n "$pid" ]] && kill -STOP "$pid" 2>/dev/null; then
            paused+=("$pid")
            printf '%s PAUSE %s avail=%dM %s\n' "$(date +%T)" "$pid" $((avail / 1024)) \
                "$(tr '\0' ' ' <"/proc/$pid/cmdline" | cut -c1-120)" >>"$LOG"
        fi
    fi
    sleep "$POLL_SECONDS"
done
