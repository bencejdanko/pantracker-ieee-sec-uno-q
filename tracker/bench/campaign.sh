#!/usr/bin/env bash
# Full measurement campaign on the board (tracker.service stopped for clean numbers). Logs to results/campaign.log
#   nohup ./campaign.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"; mkdir -p results; exec > >(tee -a results/campaign.log) 2>&1
PY=../.venv/bin/python
systemctl --user stop tracker
echo "== $(date -Is) parity"; $PY parity_run.py 2>&1 | grep -v "^I 00:"
echo "== $(date -Is) scorecard"; ./run_matrix.sh "${ROWS:-L1,L2,L3,L4,L5,P3,P4,V1,V2,V3,V4,L0}"
echo "== $(date -Is) otb full"
for t in lightfc:xnn32 kcf nano; do $PY otb_eval.py --trackers "$t" 2>&1 | grep -v "^I 00:"; done
echo "== $(date -Is) otb 300-frame drift"
for t in lightfc:xnn32 lightfc:xnn8 lightfc:vk16 lightfc:vk32 "lightfc:x=vk16,fusion=xnn32,head=xnn32"; do
  $PY otb_eval.py --trackers "$t" --max-frames 300 --out results/otb300 2>&1 | grep -v "^I 00:"; done
echo "== $(date -Is) done"; systemctl --user start tracker
