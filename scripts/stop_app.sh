#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Stop whatever app is currently running on the robot via the daemon API.
#
# This is equivalent to clicking "Stop" on the dashboard.
#
# USAGE:
#   ./stop_app.sh                      # default: reachy-mini.local
#   ./stop_app.sh 192.168.1.42        # custom IP
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

HOST="${1:-reachy-mini.local}"
REMOTE="pollen@${HOST}"
DAEMON_API="http://127.0.0.1:8000/api/apps"

echo "==> Stopping current app on ${HOST}..."
RESP=$(ssh -o ConnectTimeout=5 "${REMOTE}" \
    "curl -sf -X POST ${DAEMON_API}/stop-current-app 2>/dev/null" 2>/dev/null) || true

# Check what's running now
STATUS=$(ssh "${REMOTE}" \
    "curl -sf ${DAEMON_API}/current-app-status 2>/dev/null" 2>/dev/null) || true

if [ "$STATUS" = "null" ] || [ -z "$STATUS" ]; then
    echo "    No app running."
else
    echo "    Status: ${STATUS}"
fi
