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

# Read TEST_PATTERN from the env file to decide which ExecStart variant
# to bake into the unit. Defaults to test pattern if absent or empty —
# safer because it has no capture-device dependency.
TP_VAL="$(grep -E '^[[:space:]]*TEST_PATTERN[[:space:]]*=' "$ENV_FILE" \
    | tail -1 | cut -d= -f2- | tr -d '"' | xargs || true)"
case "$TP_VAL" in
    0|false|no|off) USE_TEST_PATTERN=0 ;;
    *)              USE_TEST_PATTERN=1 ;;
esac

if [ "$USE_TEST_PATTERN" = "1" ]; then
    MODE_DESC="TEST PATTERN (lavfi testsrc2 + 1 kHz sine)"
    EXEC_START="/usr/bin/ffmpeg -re \\
    -f lavfi -i testsrc2=size=1280x720:rate=30 \\
    -f lavfi -i sine=frequency=1000:sample_rate=48000 \\
    -vcodec libx264 -preset veryfast \\
    -b:v \${BITRATE_KBPS}k -minrate \${BITRATE_KBPS}k -maxrate \${BITRATE_KBPS}k \\
    -bufsize \${BUFSIZE_KBPS}k -g 60 -keyint_min 60 \\
    -acodec aac -b:a 128k -ar 48000 \\
    -f flv \${RTMP_URL}/\${RTMP_STREAM_KEY}"
else
    MODE_DESC="CAPTURE (\${V4L2_DEVICE} + \${ALSA_DEVICE})"
    EXEC_START="/usr/bin/ffmpeg \\
    -f v4l2 -i \${V4L2_DEVICE} \\
    -f alsa -i \${ALSA_DEVICE} \\
    -vcodec libx264 -preset veryfast \\
    -b:v \${BITRATE_KBPS}k -minrate \${BITRATE_KBPS}k -maxrate \${BITRATE_KBPS}k \\
    -bufsize \${BUFSIZE_KBPS}k -g 60 -keyint_min 60 \\
    -acodec aac -b:a 160k \\
    -f flv \${RTMP_URL}/\${RTMP_STREAM_KEY}"
fi

echo "Project dir : $SCRIPT_DIR"
echo "Run as user : $OWNER"
echo "Env file    : $ENV_FILE"
echo "Mode        : $MODE_DESC"

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
ExecStart=$EXEC_START
Restart=on-failure
RestartSec=5

# Logs to journald (viewable with journalctl -u stream.service)
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable stream.service
systemctl restart stream.service

echo ""
echo "Done. stream.service is installed, enabled at boot, and (re)started."
echo ""
echo "Useful commands:"
echo "  sudo systemctl status stream.service       # check if running"
echo "  sudo journalctl -u stream.service -f       # follow logs"
echo "  sudo systemctl restart stream.service      # restart"
echo "  sudo systemctl stop stream.service         # stop (until next boot)"
echo "  sudo systemctl disable stream.service      # disable auto-start at boot"
echo ""
echo "To switch between test pattern and capture mode: edit TEST_PATTERN in"
echo "$ENV_FILE and re-run this script."
