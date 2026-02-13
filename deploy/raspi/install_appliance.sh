#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi (Debian/Ubuntu) appliance installer
# - installs dependencies
# - builds virtualenv
# - installs systemd services for bot + web control panel

if [[ "${EUID}" -ne 0 ]]; then
  echo "[ERR] Run as root: sudo bash deploy/raspi/install_appliance.sh [PROJECT_DIR]"
  exit 1
fi

PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -f "${PROJECT_DIR}/main.py" ]]; then
  echo "[ERR] main.py not found in PROJECT_DIR=${PROJECT_DIR}"
  exit 1
fi

BOT_USER="${BOT_USER:-${SUDO_USER:-}}"
if [[ -z "${BOT_USER}" ]]; then
  BOT_USER="$(stat -c '%U' "${PROJECT_DIR}")"
fi

if [[ -z "${BOT_USER}" || "${BOT_USER}" == "root" ]]; then
  echo "[ERR] BOT_USER is empty/root. Set BOT_USER explicitly."
  echo "      Example: BOT_USER=pi sudo bash deploy/raspi/install_appliance.sh ${PROJECT_DIR}"
  exit 1
fi

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] BOT_USER=${BOT_USER}"

apt-get update -y
apt-get install -y python3 python3-venv python3-pip curl openssl ca-certificates

mkdir -p "${PROJECT_DIR}/logs"
chown -R "${BOT_USER}:${BOT_USER}" "${PROJECT_DIR}/logs"

if [[ ! -d "${PROJECT_DIR}/.venv" ]]; then
  sudo -u "${BOT_USER}" python3 -m venv "${PROJECT_DIR}/.venv"
fi
sudo -u "${BOT_USER}" "${PROJECT_DIR}/.venv/bin/pip" install --upgrade pip
sudo -u "${BOT_USER}" "${PROJECT_DIR}/.venv/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"

mkdir -p /etc/bithumb-tradbot
if [[ ! -f /etc/bithumb-tradbot/appliance.env ]]; then
  umask 077
  TOKEN="$(openssl rand -hex 16)"
  cat > /etc/bithumb-tradbot/appliance.env <<EOF
UI_TOKEN=${TOKEN}
WEB_HOST=0.0.0.0
WEB_PORT=8080
TRADBOT_SERVICE=bithumb-tradbot
TRADBOT_LOG_FILE=${PROJECT_DIR}/logs/runtime.log
EOF
fi

cat > /etc/systemd/system/bithumb-tradbot.service <<EOF
[Unit]
Description=Bybit Tradbot Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${BOT_USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PROJECT_DIR}/.venv/bin/python ${PROJECT_DIR}/main.py
Restart=always
RestartSec=5
StandardOutput=append:${PROJECT_DIR}/logs/runtime.log
StandardError=append:${PROJECT_DIR}/logs/runtime.log

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/bithumb-tradbot-web.service <<EOF
[Unit]
Description=Bybit Tradbot Web Control Panel
After=network-online.target bithumb-tradbot.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=/etc/bithumb-tradbot/appliance.env
ExecStart=${PROJECT_DIR}/.venv/bin/python ${PROJECT_DIR}/deploy/raspi/appliance_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now bithumb-tradbot.service
systemctl enable --now bithumb-tradbot-web.service

TOKEN="$(grep '^UI_TOKEN=' /etc/bithumb-tradbot/appliance.env | cut -d'=' -f2- || true)"
IP_ADDR="$(hostname -I | awk '{print $1}')"

echo
echo "[DONE] Services enabled."
echo "  - Bot: systemctl status bithumb-tradbot"
echo "  - Web: systemctl status bithumb-tradbot-web"
echo
echo "Open from another device:"
echo "  http://${IP_ADDR}:8080"
echo "UI token:"
echo "  ${TOKEN}"
echo
echo "If needed, rotate token:"
echo "  sudo nano /etc/bithumb-tradbot/appliance.env"
echo "  sudo systemctl restart bithumb-tradbot-web"
