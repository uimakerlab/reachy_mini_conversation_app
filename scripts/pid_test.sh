#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Test antenna PID parameters: modify PID on robot, restart daemon,
# wake up, and run a trajectory test to measure the effect.
#
# USAGE:
#   ./scripts/pid_test.sh 200 0 0        # P=200 I=0 D=0 (default)
#   ./scripts/pid_test.sh 150 0 50       # P=150 I=0 D=50
#   ./scripts/pid_test.sh 200 10 0       # P=200 I=10 D=0
#   ./scripts/pid_test.sh --restore      # restore default PID
#   ./scripts/pid_test.sh --current      # show current PID on robot
#
# This modifies ONLY left_antenna (ID 18) and right_antenna (ID 17).
# Stewart platform motors are NOT touched.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

HOST="${REACHY_HOST:-reachy-mini.local}"
REMOTE="pollen@${HOST}"
ROBOT_CONFIG="/venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/assets/config/hardware_config.yaml"

# Default PID values
DEFAULT_P=200
DEFAULT_I=0
DEFAULT_D=0

show_current() {
    echo "==> Current antenna PID on robot:"
    ssh "${REMOTE}" "grep -A 10 'antenna' ${ROBOT_CONFIG}" 2>/dev/null
}

if [[ "${1:-}" == "--current" ]]; then
    show_current
    exit 0
fi

if [[ "${1:-}" == "--restore" ]]; then
    P=$DEFAULT_P; I=$DEFAULT_I; D=$DEFAULT_D
    echo "==> Restoring default PID: P=${P} I=${I} D=${D}"
else
    if [[ $# -lt 3 ]]; then
        echo "Usage: $0 P I D"
        echo "       $0 --restore"
        echo "       $0 --current"
        exit 1
    fi
    P=$1; I=$2; D=$3
fi

echo "==> Setting antenna PID to: P=${P} I=${I} D=${D}"

# Step 1: Stop any running app
echo "==> Stopping any running app..."
ssh "${REMOTE}" "curl -sf -X POST http://127.0.0.1:8000/api/apps/stop-current-app >/dev/null 2>&1 || true"
sleep 1

# Step 2: Modify PID in config (only antennas)
echo "==> Modifying hardware_config.yaml on robot..."
ssh "${REMOTE}" "python3 -c \"
import yaml

with open('${ROBOT_CONFIG}') as f:
    config = yaml.safe_load(f)

for motor_entry in config['motors']:
    for name, motor in motor_entry.items():
        if 'antenna' in name:
            motor['pid'] = [${P}, ${I}, ${D}]
            print(f'  {name}: pid=[${P}, ${I}, ${D}]')

with open('${ROBOT_CONFIG}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
\""

# Step 3: Restart daemon
echo "==> Restarting reachy-mini-daemon..."
ssh "${REMOTE}" "sudo systemctl restart reachy-mini-daemon"

# Step 4: Wait for daemon to be ready
echo "==> Waiting for daemon to start..."
for i in $(seq 1 30); do
    if ssh "${REMOTE}" "curl -sf -X POST http://127.0.0.1:8000/health-check >/dev/null 2>&1"; then
        echo "    Daemon ready after ${i}s"
        break
    fi
    sleep 1
done

# Step 5: Enable motors and wake up via daemon API
echo "==> Enabling motors..."
ssh "${REMOTE}" "curl -sf -X POST http://127.0.0.1:8000/api/motors/set_mode/enabled"
echo ""

echo "==> Triggering wake-up animation..."
ssh "${REMOTE}" "curl -sf -X POST http://127.0.0.1:8000/api/move/play/wake_up"
echo ""

echo "==> Waiting for wake-up to complete..."
sleep 5

# Step 6: Upload and run a trajectory test
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "==> Uploading trajectory_test.py..."
scp "${SCRIPT_DIR}/trajectory_test.py" "${REMOTE}:/tmp/trajectory_test.py"

echo "==> Running trajectory test (s_curve, best from previous round)..."
ssh "${REMOTE}" "/venvs/apps_venv/bin/python3 /tmp/trajectory_test.py --name s_curve --output /tmp/pid_test_P${P}_I${I}_D${D}"

# Step 7: Download results
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/pid_results"
mkdir -p "${LOCAL_DIR}"
RESULT_NAME="pid_P${P}_I${I}_D${D}"
rsync -az "${REMOTE}:/tmp/pid_test_P${P}_I${I}_D${D}/" "${LOCAL_DIR}/${RESULT_NAME}/"

echo ""
echo "==> Results saved to ${LOCAL_DIR}/${RESULT_NAME}/"
echo "    PID: P=${P} I=${I} D=${D}"
echo ""
echo "Analyze with:"
echo "  python3 scripts/analyze_trajectories.py --results-dir ${LOCAL_DIR}/${RESULT_NAME} --plot"
