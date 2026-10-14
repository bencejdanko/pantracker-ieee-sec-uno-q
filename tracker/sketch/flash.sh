#!/usr/bin/env bash
# Flash the MCU sketch through App Lab (arduino-app-cli builds + uploads on the board).
set -euo pipefail
cd "$(dirname "$0")"
HOST=${HOST:-gatos.local}
PASS=$(cd ../.. && pwd)/.ssh_pass
DEST=/home/arduino/ArduinoApps/tracker-mcu
r() { sshpass -f "$PASS" ssh -o StrictHostKeyChecking=no arduino@$HOST "$1"; }
r "mkdir -p $DEST && rm -rf $DEST/sketch $DEST/python"
tar -c --exclude=flash.sh . | sshpass -f "$PASS" ssh arduino@$HOST "tar -x -C $DEST"
r "for id in \$(arduino-app-cli app list | awk '/ running /{print \$1}'); do arduino-app-cli app stop \"\$id\"; done"
r "arduino-app-cli app start $DEST"
