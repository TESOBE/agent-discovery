#!/bin/bash
# update-pi.sh — Sync the Pi to the latest committed code and restart services.
#
#   1. git pull (--ff-only, so divergence fails loudly)
#   2. cargo build --release (incremental; near-no-op if nothing changed)
#   3. restart agent-discovery
#   4. if .stream-env is newer than the installed stream.service unit,
#      regenerate the unit and restart the stream — otherwise leave it alone
#      so a live stream isn't interrupted by a routine agent update.
#
# Run as your normal user. sudo is invoked selectively for the systemctl bits.
#   bash update-pi.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Non-interactive SSH shells skip ~/.bashrc, so cargo isn't on PATH unless we
# source its env explicitly.
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi

echo "==> git pull"
git pull --ff-only

echo "==> cargo build --release"
cargo build --release

echo "==> restarting agent-discovery"
sudo systemctl restart agent-discovery

ENV_FILE="$SCRIPT_DIR/.stream-env"
UNIT_FILE="/etc/systemd/system/stream.service"

if [ -f "$ENV_FILE" ]; then
    if [ ! -f "$UNIT_FILE" ] || [ "$ENV_FILE" -nt "$UNIT_FILE" ]; then
        echo "==> .stream-env newer than $UNIT_FILE — regenerating stream.service"
        sudo bash install-stream-service.sh
    else
        echo "==> stream.service up to date with .stream-env (skipped)"
    fi
fi

echo ""
echo "Done."
echo ""
echo "  sudo journalctl -u agent-discovery -f      # agent log"
if [ -f "$UNIT_FILE" ]; then
    echo "  sudo journalctl -u stream.service -f       # stream log"
fi
