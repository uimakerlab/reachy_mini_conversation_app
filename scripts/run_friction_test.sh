#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Deploy and run friction characterization tests on the robot, then
# download results for local analysis.
#
# USAGE:
#   ./run_friction_test.sh                    # run all tests
#   ./run_friction_test.sh --test antenna     # antenna tests only
#   ./run_friction_test.sh --test head_z      # head Z tests only
#   ./run_friction_test.sh --test sinus       # sinusoidal baseline only
#   ./run_friction_test.sh --list             # list all tests
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

HOST="${REACHY_HOST:-reachy-mini.local}"
REMOTE="pollen@${HOST}"
REMOTE_DIR="/tmp/friction_test"
REMOTE_SCRIPT="/tmp/friction_test.py"
LOCAL_RESULTS="$(cd "$(dirname "$0")/.." && pwd)/friction_results"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_SCRIPT="${SCRIPT_DIR}/friction_test.py"

# Pass all args through
TEST_ARGS="$*"

# ── 0. Handle --list locally ───────────────────────────────────────
if [[ " ${TEST_ARGS} " == *" --list "* ]]; then
    ssh "${REMOTE}" "cd /tmp && /venvs/apps_venv/bin/python3 ${REMOTE_SCRIPT} --list 2>/dev/null" 2>/dev/null || \
        python3 "${LOCAL_SCRIPT}" --list
    exit 0
fi

# ── 1. Stop any running app ───────────────────────────────────────
echo "==> Stopping any running app on ${HOST}..."
ssh -o ConnectTimeout=5 "${REMOTE}" \
    "curl -sf -X POST http://127.0.0.1:8000/api/apps/stop-current-app >/dev/null 2>&1 || true"
sleep 2

# ── 2. Upload friction test script ────────────────────────────────
echo "==> Uploading friction_test.py to ${HOST}..."
scp "${LOCAL_SCRIPT}" "${REMOTE}:${REMOTE_SCRIPT}"

# ── 3. Run friction tests ─────────────────────────────────────────
echo "==> Running friction tests on ${HOST}..."
echo "    Args: ${TEST_ARGS:-all (default)}"
echo ""
ssh "${REMOTE}" "/venvs/apps_venv/bin/python3 ${REMOTE_SCRIPT} --output ${REMOTE_DIR} ${TEST_ARGS}"

# ── 4. Download results ───────────────────────────────────────────
echo ""
echo "==> Downloading results to ${LOCAL_RESULTS}/"
mkdir -p "${LOCAL_RESULTS}"
rsync -az "${REMOTE}:${REMOTE_DIR}/" "${LOCAL_RESULTS}/"

echo ""
echo "Results:"
ls -la "${LOCAL_RESULTS}"/*.csv 2>/dev/null || echo "  (no CSV files found)"
echo ""
echo "Done. Analyze with:"
echo "  python3 scripts/analyze_friction.py --plot"
