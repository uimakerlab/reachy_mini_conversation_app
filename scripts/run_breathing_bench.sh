#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Deploy and run the breathing benchmark on the robot, then download results.
#
# USAGE:
#   ./run_breathing_bench.sh                         # run baseline config
#   ./run_breathing_bench.sh --config all            # run all configs
#   ./run_breathing_bench.sh --config gentle         # specific config
#   ./run_breathing_bench.sh --config all --duration 30
#   ./run_breathing_bench.sh --list                  # list configs
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

HOST="${REACHY_HOST:-reachy-mini.local}"
REMOTE="pollen@${HOST}"
REMOTE_DIR="/tmp/breathing_bench"
REMOTE_SCRIPT="/tmp/breathing_bench.py"
LOCAL_RESULTS="$(cd "$(dirname "$0")/.." && pwd)/breathing_results"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_SCRIPT="${SCRIPT_DIR}/breathing_bench.py"

# Pass all args through
BENCH_ARGS="$*"

# ── 0. Handle --list locally (no need to deploy) ─────────────────────
if [[ " ${BENCH_ARGS} " == *" --list "* ]]; then
    echo "Available configs:"
    ssh "${REMOTE}" "cd /tmp && /venvs/apps_venv/bin/python3 ${REMOTE_SCRIPT} --list 2>/dev/null" 2>/dev/null || \
        python3 "${LOCAL_SCRIPT}" --list
    exit 0
fi

# ── 1. Stop any running app (avoid conflicts) ────────────────────────
echo "==> Stopping any running app on ${HOST}..."
ssh -o ConnectTimeout=5 "${REMOTE}" \
    "curl -sf -X POST http://127.0.0.1:8000/api/apps/stop-current-app >/dev/null 2>&1 || true"
sleep 2

# ── 2. Upload benchmark script ───────────────────────────────────────
echo "==> Uploading breathing_bench.py to ${HOST}..."
scp "${LOCAL_SCRIPT}" "${REMOTE}:${REMOTE_SCRIPT}"

# ── 3. Run benchmark ─────────────────────────────────────────────────
echo "==> Running benchmark on ${HOST}..."
echo "    Args: ${BENCH_ARGS:-baseline (default)}"
echo ""
ssh "${REMOTE}" "/venvs/apps_venv/bin/python3 ${REMOTE_SCRIPT} --output ${REMOTE_DIR} ${BENCH_ARGS}"

# ── 4. Download results ──────────────────────────────────────────────
echo ""
echo "==> Downloading results to ${LOCAL_RESULTS}/"
mkdir -p "${LOCAL_RESULTS}"
rsync -az "${REMOTE}:${REMOTE_DIR}/" "${LOCAL_RESULTS}/"

echo ""
echo "Results:"
ls -la "${LOCAL_RESULTS}"/*.csv 2>/dev/null || echo "  (no CSV files found)"
echo ""
echo "Done. Analyze with: python3 scripts/analyze_breathing.py"
