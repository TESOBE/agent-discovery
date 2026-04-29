#!/bin/bash
# install-stream-service.sh — Install the ffmpeg streaming systemd service
# on a Raspberry Pi.
#
# The service itself is NOT enabled at boot. The agent's `system-commands`
# handler starts and stops it on demand (stream-start / stream-stop), and
# you can also drive it manually with `systemctl` for layer-2 testing.
#
# Prerequisites:
#   1. ffmpeg installed: sudo apt install ffmpeg
#   2. Project dir contains a filled-in .stream-env file:
#        cp .stream-env.example .stream-env && $EDITOR .stream-env
#
# Usage:
#   sudo bash install-stream-service.sh
#
# After install:
#   sudo systemctl start stream.service       # start streaming manually
#   sudo systemctl stop stream.service        # stop
#   sudo systemctl status stream.service      # status
#   sudo journalctl -u stream.service -f      # logs
#
# To uninstall:
#   sudo systemctl stop stream.service
#   sudo rm /etc/systemd/system/stream.service
#   sudo systemctl daemon-reload

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo bash install-stream-service.sh)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.stream-env"
EXAMPLE="$SCRIPT_DIR/.stream-env.example"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$EXAMPLE" ]; then
        echo "ERROR: $ENV_FILE not found. Create it from the template and fill in:" >&2
        echo "  cp $EXAMPLE $ENV_FILE && \$EDITOR $ENV_FILE" >&2
    else
        echo "ERROR: $ENV_FILE not found, and no template at $EXAMPLE" >&2
    fi
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not installed. Run: sudo apt install ffmpeg" >&2
    exit 1
fi

# Detect the user who owns the project directory (don't run ffmpeg as root)
OWNER="$(stat -c '%U' "$SCRIPT_DIR")"
OWNER_GROUP="$(stat -c '%G' "$SCRIPT_DIR")"

echo "Project dir : $SCRIPT_DIR"
echo "Run as user : $OWNER"
echo "Env file    : $ENV_FILE"

# ── Write the systemd unit ──────────────────────────────────────────
# `${VAR}` references are expanded by systemd at unit-execution time
# from the EnvironmentFile, NOT by this shell — hence the `\${...}`.
cat > /etc/systemd/system/stream.service << EOF
[Unit]
Description=Pi field streamer (ffmpeg → rtmp)
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=$OWNER
Group=$OWNER_GROUP
SupplementaryGroups=audio video
EnvironmentFile=$ENV_FILE
ExecStart=/usr/bin/ffmpeg \\
    -f v4l2 -i \${V4L2_DEVICE} \\
    -f alsa -i \${ALSA_DEVICE} \\
    -vcodec libx264 -preset veryfast \\
    -b:v \${BITRATE_KBPS}k -minrate \${BITRATE_KBPS}k -maxrate \${BITRATE_KBPS}k \\
    -bufsize \${BUFSIZE_KBPS}k -g 60 -keyint_min 60 \\
    -acodec aac -b:a 160k \\
    -f flv \${RTMP_URL}/\${RTMP_STREAM_KEY}
Restart=on-failure
RestartSec=5

# Logs to journald (viewable with journalctl -u stream.service)
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo ""
echo "Done. stream.service is installed."
echo ""
echo "Note: the service is NOT enabled at boot. The agent controls it via"
echo "      the OBP system-commands channel (stream-start / stream-stop)."
echo "      You can also drive it manually for testing:"
echo ""
echo "  sudo systemctl start stream.service"
echo "  sudo journalctl -u stream.service -f"
echo "  sudo systemctl stop stream.service"
