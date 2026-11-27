#!/bin/bash
# nc-remote build box idle check (/usr/local/sbin/ncbox-idle): shut the box down after
# @IDLE_MINUTES@ min without a job or a sync, unless a finished job's outputs are not yet
# on the PC. Installed by the box's user-data (nc_remote_box.sh), run once a minute by
# its timer, and refreshed and run by `nc-remote reap` over SSH (nc_remote.py), so a box
# launched by an older nc-remote follows this version too.
BOX=/home/ubuntu/ncbox
limit=$((@IDLE_MINUTES@ * 60))
for slot in "$BOX"/slots/*; do
  flock -n "$slot" true || exit 0   # a job holds this slot
done
# Unfetched outputs of this boot (nc_remote_job.py UNFETCHED): never discarded by an idle
# stop (2026-11-24: 19 finished PGO builds were lost that way). `nc-remote reap` fetches
# them; the 12 h hard lifetime remains the backstop.
booted=$(awk '/^btime/ {print $2}' /proc/stat)
pending=$(find "$BOX"/jobs -mindepth 3 -maxdepth 3 -path '*/unfetched/*.json' \
  -newermt "@$booted" -print -quit 2>/dev/null)
if [ -n "$pending" ]; then
  logger -t ncbox-idle "unfetched outputs ($pending): not shutting down"
  exit 0
fi
last=$(stat -c %Y "$BOX/activity" 2>/dev/null || echo 0)
if [ $(( $(date +%s) - last )) -ge "$limit" ]; then
  logger -t ncbox-idle "idle for @IDLE_MINUTES@ min: shutting down"
  rm -rf -- /home/ubuntu/nanocodec-data/clips   # clips never outlive their use
  shutdown -h now
fi
