#!/usr/bin/env bash
# Build deck.pdf from slides.md (Marp). Charts first: values come from tracker/bench/SCORECARD.md.
set -euo pipefail
cd "$(dirname "$0")"
../tracker/export/.venv/bin/python charts.py
CHROME_PATH=${CHROME_PATH:-/usr/bin/google-chrome} npx -y @marp-team/marp-cli@4.5.1 slides.md --theme econ.css --html --allow-local-files --pdf --browser-timeout 60 -o deck.pdf </dev/null
