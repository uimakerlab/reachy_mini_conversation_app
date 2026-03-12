#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Profile the conversation app on Wireless using Scalene.
#
# Installs scalene if needed, syncs source, stops any running app,
# then runs the conversation app directly under scalene (not via daemon).
# The profile is saved as an HTML report and copied back locally.
#
# USAGE:
#   ./scripts/profile_wireless.sh                  # default: reachy-mini.local
#   ./scripts/profile_wireless.sh 192.168.1.42    # custom IP
#
# Press Ctrl+C on the robot to stop profiling. The report is then
# fetched to ./scalene_profiles/ locally.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

HOST="${1:-reachy-mini.local}"
REMOTE="pollen@${HOST}"
SITE_PACKAGES="/venvs/apps_venv/lib/python3.12/site-packages"
DAEMON_API="http://127.0.0.1:8000/api/apps"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${SCRIPT_DIR}/src/reachy_mini_conversation_app/"
TARGET="${REMOTE}:${SITE_PACKAGES}/reachy_mini_conversation_app/"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_PROFILE="/tmp/scalene_profile_${TIMESTAMP}.json"

# ── 1. Sync source ───────────────────────────────────────────────────
echo "==> Syncing source to ${HOST}..."
rsync -az --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'profiles/' \
    "$SOURCE" "$TARGET"

# ── 2. Install scalene if needed ─────────────────────────────────────
echo "==> Ensuring scalene is installed..."
ssh "${REMOTE}" "/venvs/apps_venv/bin/pip show scalene >/dev/null 2>&1 || /venvs/apps_venv/bin/pip install scalene -q"

# ── 3. Stop any running app ──────────────────────────────────────────
echo "==> Stopping current app..."
ssh "${REMOTE}" "curl -sf -X POST ${DAEMON_API}/stop-current-app >/dev/null 2>&1 || true"
sleep 1

# ── 4. Run under scalene ─────────────────────────────────────────────
DURATION="${2:-60}"
echo "==> Running conversation app under scalene for ${DURATION}s..."
echo "    Profile will be saved to ${REMOTE_PROFILE}"
echo ""
# Create a wrapper that runs the app for a fixed duration then exits cleanly
ssh "${REMOTE}" "cat > /tmp/scalene_runner.py << PYEOF
import signal, threading
duration = ${DURATION}
timer = threading.Timer(duration, lambda: signal.raise_signal(signal.SIGINT))
timer.daemon = True
timer.start()
from reachy_mini_conversation_app.main import main
main()
PYEOF
/venvs/apps_venv/bin/python -m scalene run \
    --html --outfile ${REMOTE_PROFILE} \
    --cpu-only --reduced-profile --profile-all \
    --- /tmp/scalene_runner.py" || true

# ── 5. Fetch profile ─────────────────────────────────────────────────
mkdir -p "${SCRIPT_DIR}/scalene_profiles"
LOCAL_PROFILE="${SCRIPT_DIR}/scalene_profiles/scalene_profile_${TIMESTAMP}.json"
echo ""
echo "==> Fetching profile..."
scp "${REMOTE}:${REMOTE_PROFILE}" "${LOCAL_PROFILE}" 2>/dev/null && \
    echo "    Saved to ${LOCAL_PROFILE}" && \
    echo "    View with: scalene view ${LOCAL_PROFILE}" || \
    echo "    WARNING: Could not fetch profile (app may not have produced one)"
