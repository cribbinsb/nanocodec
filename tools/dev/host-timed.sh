#!/usr/bin/env bash
# host-timed.sh CMD...: run an x86 timing command on this PC's reserved timing
# cores, one timing job at a time (the PC's counterpart of the Jetson's
# jetson-timed.sh).
#
# The PC reserves physical cores 13 and 14 (logical CPUs 14,15 and their SMT
# siblings 30,31) for timing: the user's systemd slices (app, session,
# background) and system.slice are confined to the other CPUs, and
# timing.slice (~/.config/systemd/user/timing.slice) owns 14,15,30,31. CPUs 1
# and 17 are faulty and stay offline. This script takes the exclusive host
# timing lock (~/nanocodec-data/host-timing.lock), then runs CMD in a scope in
# timing.slice under the usual 8 GiB memory cap. Inside, pin each timed run to
# 14 or 15 (taskset); their siblings 30,31 stay idle for the orchestrator.
# Reports how long the lock was held.
set -euo pipefail
LOCK=$HOME/nanocodec-data/host-timing.lock
if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
  sed -n "2,14p" "$0"; exit 0
fi
exec 9>>"$LOCK"
flock -x 9
start=$(date +%s)
status=0
systemd-run --user --scope -q --slice=timing.slice -p MemoryMax=8G -p MemorySwapMax=0 \
  -- "$@" 9>&- || status=$?
echo "host-timed: lock held $(( $(date +%s) - start ))s" >&2
exit $status
