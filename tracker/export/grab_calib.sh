#!/usr/bin/env bash
# Grab N raw frames from the running tracker service (demo camera) for detector int8 PTQ calibration.
set -euo pipefail
N=${1:-200}; H=${HOST:-gatos.local}
out=$(dirname "$0")/calib; mkdir -p "$out"
for i in $(seq -w 1 "$N"); do curl -s -o "$out/cam_$i.jpg" "http://$H:8000/frame.jpg"; sleep 0.25; done
ls "$out" | wc -l
