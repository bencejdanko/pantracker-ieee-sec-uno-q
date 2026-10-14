#!/usr/bin/env bash
# Scorecard on the board: stops tracker.service, records conditions, runs matrix.py.
#   ./run_matrix.sh [ROWS]        e.g. ./run_matrix.sh L1,L2,L5,Y1
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
systemctl --user stop tracker 2>/dev/null || true
{ date -Is; uname -r; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /sys/class/devfreq/5900000.gpu/governor;
  for h in /sys/class/hwmon/hwmon*; do echo "$(cat $h/name) $(cat $h/temp1_input 2>/dev/null)"; done; } > results/conditions.txt
../.venv/bin/python matrix.py ${1:+--rows "$1"} 2>&1 | grep -v "^I 00:" | tee results/matrix.log
