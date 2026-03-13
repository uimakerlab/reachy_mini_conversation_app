#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Deploy and run the breathing review on the robot.
# Cycles through configs with big banners. Ctrl+C to skip, 2x to exit.
#
# USAGE:
#   ./run_review.sh                           # top picks, 10s each
#   ./run_review.sh --duration 15             # 15s each
#   ./run_review.sh --config curious sleepy   # specific configs
#   ./run_review.sh --all                     # all configs
#   ./run_review.sh --list                    # list configs
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

HOST="${REACHY_HOST:-reachy-mini.local}"
REMOTE="pollen@${HOST}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. Stop any running app ──────────────────────────────────────────
echo "==> Stopping any running app on ${HOST}..."
ssh -o ConnectTimeout=5 "${REMOTE}" \
    "curl -sf -X POST http://127.0.0.1:8000/api/apps/stop-current-app >/dev/null 2>&1 || true"
sleep 2

# ── 2. Upload scripts ────────────────────────────────────────────────
echo "==> Uploading scripts to ${HOST}..."
scp "${SCRIPT_DIR}/breathing_bench.py" "${REMOTE}:/tmp/breathing_bench.py"
scp "${SCRIPT_DIR}/review_breathing.py" "${REMOTE}:/tmp/review_breathing.py"

# ── 3. Run review (interactive — logs stream to terminal) ────────────
echo "==> Starting review on ${HOST}..."
echo ""
ssh -t "${REMOTE}" "cd /tmp && /venvs/apps_venv/bin/python3 /tmp/review_breathing.py $*"
