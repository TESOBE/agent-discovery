#!/bin/bash
# install-service.sh — Install agent-discovery as a systemd service on a Raspberry Pi.
#
# What this does:
#   1. Writes a systemd unit file that runs start-agent.sh at boot
#   2. Enables the service so it starts automatically
#   3. Starts it immediately
#
# Usage:
#   sudo bash install-service.sh
#
# After install:
#   sudo systemctl status agent-discovery   # check status
#   sudo journalctl -u agent-discovery -f   # follow logs
#   sudo systemctl restart agent-discovery  # restart
#   sudo systemctl stop agent-discovery     # stop
#   sudo systemctl disable agent-discovery  # disable auto-start
#
# To uninstall:
#   sudo systemctl stop agent-discovery
#   sudo systemctl disable agent-discovery
#   sudo rm /etc/systemd/system/agent-discovery.service
#   sudo systemctl daemon-reload

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo bash install-service.sh)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="$SCRIPT_DIR/start-agent.sh"

if [ ! -x "$START_SCRIPT" ]; then
    echo "ERROR: $START_SCRIPT not found or not executable." >&2
    exit 1
fi

# Detect the user who owns the project directory (don't run the agent as root)
OWNER="$(stat -c '%U' "$SCRIPT_DIR")"
OWNER_GROUP="$(stat -c '%G' "$SCRIPT_DIR")"

echo "Project dir : $SCRIPT_DIR"
echo "Run as user : $OWNER"

# ── Write the systemd unit ──────────────────────────────────────────
cat > /etc/systemd/system/agent-discovery.service << EOF
[Unit]
Description=Agent Discovery Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=$OWNER
Group=$OWNER_GROUP
WorkingDirectory=$SCRIPT_DIR
ExecStart=$START_SCRIPT
Restart=on-failure
RestartSec=5

# Give the .env file's variables to the process
EnvironmentFile=-$SCRIPT_DIR/.env

# Audio access
SupplementaryGroups=audio

# Logging goes to journald (viewable with journalctl -u agent-discovery)
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ── Enable and start ────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable agent-discovery.service
systemctl start agent-discovery.service

echo ""
echo "Done. agent-discovery service is installed and running."
echo ""
echo "Useful commands:"
echo "  sudo systemctl status agent-discovery     # check status"
echo "  sudo journalctl -u agent-discovery -f     # follow logs"
echo "  sudo systemctl restart agent-discovery    # restart"
echo "  sudo systemctl stop agent-discovery       # stop"
echo "  sudo systemctl disable agent-discovery    # disable auto-start"
