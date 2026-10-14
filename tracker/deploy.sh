#!/usr/bin/env bash
# Ship python/, models/ (.pte + LightFC blocks.json + NanoTrack) and libinfer to gatos, then restart tracker.service.
#   ./deploy.sh            sync + restart
#   ./deploy.sh --no-restart
#   ./deploy.sh --code-only  python/ + bench/ + restart (no models/lib; use for code changes)
#   ./deploy.sh logs
set -euo pipefail
cd "$(dirname "$0")"
HOST=${HOST:-gatos.local}
PASS=$(cd .. && pwd)/.ssh_pass
r() { sshpass -f "$PASS" ssh -o StrictHostKeyChecking=no arduino@$HOST "$@"; }
if [[ "${1:-}" == "logs" ]]; then r "journalctl --user -u tracker -n 100 -f"; exit; fi
if [[ "${1:-}" == "--code-only" ]]; then  # python/ + bench/ only (seconds); skips models, libinfer, unit file
  tar -C python --exclude=__pycache__ --exclude=config.json -cf - . | r "tar -C ~/tracker/python -xf -"
  tar -C bench --exclude=results --exclude=__pycache__ -cf - . | r "tar -C ~/tracker/bench -xf -"
  r "systemctl --user restart tracker && systemctl --user is-active tracker"
  exit
fi

r "mkdir -p ~/tracker/{python,models/lightfc,lib,bench} ~/.config/systemd/user"
# no rsync on the board: tar over ssh
tar -C python --exclude=__pycache__ --exclude=config.json -cf - . | r "tar -C ~/tracker/python -xf -"
r "test -f ~/tracker/python/config.json" || tar -C python -cf - config.json | r "tar -C ~/tracker/python -xf -"
tar -C native/out -cf - libinfer.so infer_bench | r "tar -C ~/tracker/lib -xf -"
tar -C bench --exclude=results --exclude=__pycache__ -cf - . | r "tar -C ~/tracker/bench -xf -"
( cd export/models && tar -cf - $(ls *.pte) lightfc/blocks.json nanotrack $(cd lightfc && ls *.pte | sed 's|^|lightfc/|') ) \
  | r "tar -C ~/tracker/models -xf -"
tar -cf - tracker.service | r "tar -C ~/.config/systemd/user -xf - && systemctl --user daemon-reload && systemctl --user enable tracker >/dev/null 2>&1; loginctl show-user \$USER -p Linger | grep -q yes || echo 'NOTE: run  sudo loginctl enable-linger arduino  so tracker.service starts at boot'"
[[ "${1:-}" == "--no-restart" ]] || r "systemctl --user restart tracker && sleep 1 && systemctl --user --no-pager status tracker | head -5"
