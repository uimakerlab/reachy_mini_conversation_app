#!/usr/bin/env bash
# Restore default PID on robot after a bad PID config caused a crash.
# Waits for robot to come online, then fixes config and restarts daemon.

set -euo pipefail

HOST="${REACHY_HOST:-reachy-mini.local}"
REMOTE="pollen@${HOST}"
CONFIG="/venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/assets/config/hardware_config.yaml"

echo "Waiting for robot to come online..."
while true; do
    if ping -c 1 -W 2 "${HOST}" &>/dev/null; then
        echo "Robot is reachable! Waiting 10s for SSH..."
        sleep 10
        break
    fi
    sleep 3
done

echo "Restoring default PID (P=200 I=0 D=0 for antennas, P=300 I=0 D=0 for stewart)..."
ssh -o ConnectTimeout=10 "${REMOTE}" "python3 -c \"
import yaml

with open('${CONFIG}') as f:
    config = yaml.safe_load(f)

for motor_entry in config['motors']:
    for name, motor in motor_entry.items():
        if 'antenna' in name or 'body' in name:
            motor['pid'] = [200, 0, 0]
            print(f'  {name}: pid=[200, 0, 0]')
        elif 'stewart' in name:
            motor['pid'] = [300, 0, 0]
            print(f'  {name}: pid=[300, 0, 0]')

with open('${CONFIG}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
\""

echo "Restarting daemon..."
ssh "${REMOTE}" "sudo systemctl restart reachy-mini-daemon"

echo "Waiting for daemon..."
for i in $(seq 1 30); do
    if ssh "${REMOTE}" "curl -sf -X POST http://127.0.0.1:8000/health-check >/dev/null 2>&1"; then
        echo "Daemon ready!"
        break
    fi
    sleep 1
done

echo "Enabling motors and waking up..."
ssh "${REMOTE}" "curl -sf -X POST http://127.0.0.1:8000/api/motors/set_mode/enabled"
ssh "${REMOTE}" "curl -sf -X POST http://127.0.0.1:8000/api/move/play/wake_up"
sleep 5

echo "Done. Robot should be back to normal."
