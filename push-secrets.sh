#!/bin/bash
# push-secrets.sh — Copy local secret env files (.env, .stream-env) to the Pi
# and apply them: chmod 600, restart only the services that read what changed.
#
# - .env pushed         → restart agent-discovery (it re-reads .env on start)
# - .stream-env pushed  → re-run install-stream-service.sh on the Pi (which
#                         regenerates the unit in case TEST_PATTERN changed
#                         and restarts the stream so runtime vars take effect)
#
# Code-only updates: run `bash update-pi.sh` on the Pi instead.
#
# Configure target via env vars (or edit the defaults below):
#   PI_HOST  — ~/.ssh/config alias or user@ip   (default: pi)
#   PI_DIR   — path to agent-discovery on the Pi
#
# Usage (from the desktop project root):
#   bash push-secrets.sh

set -euo pipefail

PI_HOST="${PI_HOST:-pi}"
PI_DIR="${PI_DIR:-/home/colourpi/Documents/workspace/agent-discovery}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Collect the secret files that exist locally — push only those that do.
FILES=()
[ -f .env ]        && FILES+=(".env")
[ -f .stream-env ] && FILES+=(".stream-env")

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "ERROR: neither .env nor .stream-env found in $SCRIPT_DIR" >&2
    exit 1
fi

ENV_PUSHED=0
STREAM_PUSHED=0
for f in "${FILES[@]}"; do
    [ "$f" = ".env" ]        && ENV_PUSHED=1
    [ "$f" = ".stream-env" ] && STREAM_PUSHED=1
done

echo "==> Pushing to ${PI_HOST}:${PI_DIR}/"
for f in "${FILES[@]}"; do
    echo "    $f"
done

scp "${FILES[@]}" "${PI_HOST}:${PI_DIR}/"

# Build the remote command. Each step is && so a failure aborts the rest.
REMOTE="cd '$PI_DIR' && chmod 600 ${FILES[*]}"
if [ "$ENV_PUSHED" = "1" ]; then
    REMOTE+=" && sudo systemctl restart agent-discovery"
fi
if [ "$STREAM_PUSHED" = "1" ]; then
    REMOTE+=" && sudo bash install-stream-service.sh"
fi

echo "==> Applying on $PI_HOST"
ssh -t "$PI_HOST" "$REMOTE"

echo ""
echo "Done."
