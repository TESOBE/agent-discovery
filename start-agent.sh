#!/bin/bash
# start-agent.sh — Boot the agent-discovery daemon on a Raspberry Pi.
# The agent name is derived from the first 12 hex characters of the
# SHA-256 hash of the primary network interface's MAC address, giving
# each Pi a stable, unique identity.
#
# Usage:
#   bash start-agent.sh            # auto-detect interface
#   bash start-agent.sh eth0       # use a specific interface
#   bash start-agent.sh wlan0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Resolve MAC address ─────────────────────────────────────────────
IFACE="${1:-}"

if [ -z "$IFACE" ]; then
    # Pick the first non-loopback interface that has a MAC address
    for candidate in /sys/class/net/*/address; do
        iface_name="$(basename "$(dirname "$candidate")")"
        [ "$iface_name" = "lo" ] && continue
        mac="$(cat "$candidate")"
        [ "$mac" = "00:00:00:00:00:00" ] && continue
        IFACE="$iface_name"
        break
    done
fi

if [ -z "$IFACE" ]; then
    echo "ERROR: No suitable network interface found." >&2
    exit 1
fi

MAC_FILE="/sys/class/net/${IFACE}/address"
if [ ! -f "$MAC_FILE" ]; then
    echo "ERROR: Interface '$IFACE' not found ($MAC_FILE does not exist)." >&2
    exit 1
fi

MAC="$(cat "$MAC_FILE")"
if [ -z "$MAC" ] || [ "$MAC" = "00:00:00:00:00:00" ]; then
    echo "ERROR: Interface '$IFACE' has no valid MAC address." >&2
    exit 1
fi

# ── Derive agent name ───────────────────────────────────────────────
HASH="$(echo -n "$MAC" | sha256sum | awk '{print $1}')"
AGENT_NAME="${HASH:0:12}"

echo "Interface : $IFACE"
echo "MAC       : $MAC"
echo "Agent name: $AGENT_NAME"

# ── Load .env if present (but AGENT_NAME always comes from MAC) ─────
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading .env (AGENT_NAME will be overridden by MAC-derived value)"
fi

# ── Build (release) if binary is missing or stale ──────────────────
BINARY="$SCRIPT_DIR/target/release/agent-discovery"
if [ ! -x "$BINARY" ]; then
    echo "Building release binary…"
    cargo build --release
fi

# ── Start the agent ─────────────────────────────────────────────────
export AGENT_NAME
echo "Starting agent '$AGENT_NAME'…"
exec "$BINARY" run
