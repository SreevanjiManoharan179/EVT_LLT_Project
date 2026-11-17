#!/usr/bin/env bash
# Group 14 EV Battery Digital-Twin Virtual Lab: one-command launcher for Linux and macOS.
#
#   bash run.sh                     start the lab (the first run sets everything up and needs internet once)
#   bash run.sh --port 8600         use a specific port
#   bash run.sh --no-browser        do not open a browser tab
#   bash run.sh --rebuild           rebuild data and the calibrated twin from data/raw, then start (about 7 min)
#   bash run.sh --install-shortcut  add "Battery Digital Twin Lab" to the app menu (Linux)
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
# Settings from other Python setups (ROS, conda, ...) must not leak into the lab's own environment.
unset PYTHONPATH PYTHONHOME
export PIP_DISABLE_PIP_VERSION_CHECK=1
# Started from the app-menu shortcut: keep the window open after an error so the message can be read.
pause_on_error() {
  local status=$?
  if [ "${G14_PAUSE:-0}" = 1 ] && [ "$status" -ne 0 ]; then read -r -p "Press Enter to close this window." _ || true; fi
}
trap pause_on_error EXIT

# Optional: an app-menu entry that runs this script in a terminal window.
if [ "${1:-}" = "--install-shortcut" ]; then
  apps="$HOME/.local/share/applications"
  mkdir -p "$apps"
  entry="$apps/group14-battery-lab.desktop"
  cat > "$entry" <<EOF
[Desktop Entry]
Type=Application
Name=Battery Digital Twin Lab
GenericName=EV battery virtual lab
Comment=Start the Group 14 EV battery digital-twin virtual lab (SOC/SOH)
Exec=env G14_PAUSE=1 bash "$ROOT/run.sh"
Path=$ROOT
Icon=$ROOT/static/icons/battery-lab.svg
Terminal=true
Categories=Education;
Keywords=battery;digital twin;SOC;SOH;EV;
Actions=tidy;

[Desktop Action tidy]
Name=Tidy the project folder
Exec=env G14_PAUSE=1 bash "$ROOT/scripts/tidy.sh"
EOF
  chmod +x "$entry"
  if [ -d "$HOME/Desktop" ]; then  # a desktop icon too, already marked as allowed to launch
    icon="$HOME/Desktop/Battery Digital Twin Lab.desktop"
    cp "$entry" "$icon" && chmod +x "$icon"
    gio set "$icon" metadata::trusted true 2>/dev/null || true
  fi
  echo "Launcher added: search for \"Battery Digital Twin Lab\" in the app menu, or use the desktop icon."
  exit 0
fi

# 1. A Python 3.10+ interpreter.
PY=""
for c in "${PYTHON:-}" python3 python; do
  [ -n "$c" ] || continue
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "Python 3.10 or newer is required (https://www.python.org/downloads/)."; exit 1; }

# 2. A private environment in .venv (re-created if it is missing or broken).
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
  echo ">> Creating the virtual environment (.venv) ..."
  if ! "$PY" -m venv .venv; then
    rm -rf .venv
    echo "Could not create .venv. On Ubuntu or Debian run: sudo apt install python3-venv"
    exit 1
  fi
fi

# 3. Dependencies: installed on the first run, and again only when requirements.txt changes.
REQ_SUM="$(cksum < requirements.txt)"
if [ "$(cat .venv/.requirements.sum 2>/dev/null)" != "$REQ_SUM" ]; then
  echo ">> Installing dependencies (first run only: needs internet, takes a few minutes) ..."
  if ! .venv/bin/python -m pip install -r requirements.txt -q; then
    echo "Installing the dependencies failed. Check the internet connection and run again."
    exit 1
  fi
  echo "$REQ_SUM" > .venv/.requirements.sum
fi

# 4. Check the data and models, start the lab and open the browser.
.venv/bin/python scripts/launch.py "$@"
