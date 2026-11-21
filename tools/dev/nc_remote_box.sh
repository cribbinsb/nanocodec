#!/bin/bash
# nc-remote build box: user-data, run as root on every launch (also from the
# baked AMI). It sets the two limits that hold even if the PC goes away:
#   1. the hard lifetime (`shutdown -h +LIFETIME_MINUTES`; the instance's
#      shutdown behaviour is terminate);
#   2. the idle watchdog: a systemd timer that, once a minute, shuts the box
#      down when no job slot is held and nothing has touched the activity
#      file for IDLE_MINUTES (tools/dev/nc_remote.py, "Lifecycle").
# nc_remote.py substitutes the @...@ placeholders before launching.
set -eu

shutdown -h +@LIFETIME_MINUTES@

BOX=/home/ubuntu/ncbox
install -d -o ubuntu -g ubuntu "$BOX" "$BOX/slots" "$BOX/jobs"
# One slot per @JOBS_PER_SLOT@ vCPU of whichever type the fleet chose.
for slot in $(seq 0 $(( $(nproc) / @JOBS_PER_SLOT@ - 1 ))); do
  install -o ubuntu -g ubuntu -m 644 /dev/null "$BOX/slots/$slot"
done
# The idle clock starts at boot.
install -o ubuntu -g ubuntu -m 644 /dev/null "$BOX/activity"

cat > /usr/local/sbin/ncbox-idle <<'EOF'
#!/bin/bash
# Shut the build box down after @IDLE_MINUTES@ min without a job or a sync.
BOX=/home/ubuntu/ncbox
limit=$((@IDLE_MINUTES@ * 60))
for slot in "$BOX"/slots/*; do
  flock -n "$slot" true || exit 0   # a job holds this slot
done
last=$(stat -c %Y "$BOX/activity" 2>/dev/null || echo 0)
if [ $(( $(date +%s) - last )) -ge "$limit" ]; then
  logger -t ncbox-idle "idle for @IDLE_MINUTES@ min: shutting down"
  rm -rf -- /home/ubuntu/nanocodec-data/clips   # clips never outlive their use
  shutdown -h now
fi
EOF
chmod 755 /usr/local/sbin/ncbox-idle

cat > /etc/systemd/system/ncbox-idle.service <<'EOF'
[Unit]
Description=nc-remote build box idle check
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/ncbox-idle
EOF
cat > /etc/systemd/system/ncbox-idle.timer <<'EOF'
[Unit]
Description=nc-remote build box idle check, every minute
[Timer]
OnBootSec=2min
OnUnitActiveSec=1min
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now ncbox-idle.timer
