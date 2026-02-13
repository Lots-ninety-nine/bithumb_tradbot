#!/usr/bin/env bash
set -euo pipefail

BOT_DIR="${1:-$HOME/bithumb_tradbot}"
BOT_USER="$(whoami)"
SERVICE_NAME="bithumb-tradbot"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON_BIN="${BOT_DIR}/.venv/bin/python"
PIP_BIN="${BOT_DIR}/.venv/bin/pip"

if [[ ! -d "${BOT_DIR}" ]]; then
  echo "[ERROR] BOT_DIR not found: ${BOT_DIR}" >&2
  echo "Usage: $0 /absolute/path/to/bithumb_tradbot" >&2
  exit 1
fi

if [[ ! -f "${BOT_DIR}/requirements.txt" || ! -f "${BOT_DIR}/main.py" ]]; then
  echo "[ERROR] main.py or requirements.txt not found in ${BOT_DIR}" >&2
  exit 1
fi

echo "[1/7] Install OS packages"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git ca-certificates curl

echo "[2/7] Create virtualenv"
if [[ ! -d "${BOT_DIR}/.venv" ]]; then
  python3 -m venv "${BOT_DIR}/.venv"
fi

echo "[3/7] Install Python deps"
"${PIP_BIN}" install --upgrade pip
"${PIP_BIN}" install -r "${BOT_DIR}/requirements.txt"

echo "[4/7] Ensure runtime directories"
mkdir -p "${BOT_DIR}/logs"

echo "[5/7] Validate config and env presence"
if [[ ! -f "${BOT_DIR}/config.yaml" ]]; then
  echo "[ERROR] config.yaml not found: ${BOT_DIR}/config.yaml" >&2
  exit 1
fi
if [[ ! -f "${BOT_DIR}/.env" ]]; then
  echo "[WARN] .env not found: ${BOT_DIR}/.env"
  echo "       Create it before starting the service."
fi

echo "[6/7] Install systemd service"
sudo tee "${SERVICE_PATH}" > /dev/null <<SERVICE
[Unit]
Description=Bithumb Tradbot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${BOT_USER}
WorkingDirectory=${BOT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PYTHON_BIN} ${BOT_DIR}/main.py --config ${BOT_DIR}/config.yaml
Restart=always
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
SERVICE

echo "[7/7] Enable and start service"
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager

echo
echo "Done. Useful commands:"
echo "  journalctl -u ${SERVICE_NAME} -f"
echo "  sudo systemctl restart ${SERVICE_NAME}"
echo "  sudo systemctl stop ${SERVICE_NAME}"
