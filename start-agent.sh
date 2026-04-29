#!/bin/bash
# start-agent.sh — Boot the agent-discovery daemon on a Raspberry Pi.
# The agent name is derived deterministically from the primary network
# interface's MAC address, in the form
#   <adjective>-<verb>-<arch>-<hash6>
# e.g. "swift-running-pi5-a1b2c3" or "calm-listening-x86-deadbe".
# Same MAC always produces the same name across reboots.
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

# ── Detect architecture (Pi-aware, fallback to uname -m) ───────────
ARCH=""
if [ -f /sys/firmware/devicetree/base/model ]; then
    MODEL="$(tr -d '\0' < /sys/firmware/devicetree/base/model)"
    case "$MODEL" in
        *"Raspberry Pi 5"*) ARCH="pi5" ;;
        *"Raspberry Pi 4"*) ARCH="pi4" ;;
        *"Raspberry Pi 3"*) ARCH="pi3" ;;
        *"Raspberry Pi 2"*) ARCH="pi2" ;;
        *"Raspberry Pi"*)   ARCH="pi"  ;;
    esac
fi
if [ -z "$ARCH" ]; then
    case "$(uname -m)" in
        x86_64)        ARCH="x86" ;;
        aarch64)       ARCH="arm64" ;;
        armv7l|armv6l) ARCH="arm32" ;;
        *)             ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')" ;;
    esac
fi

# ── Derive agent name: <adjective>-<verb>-<arch>-<hash6> ───────────
# Word indices and the disambiguating hash tail come from the MAC's
# SHA-256, so the same Pi always yields the same name.
HASH="$(echo -n "$MAC" | sha256sum | awk '{print $1}')"

ADJECTIVES=(swift calm bright steady quiet eager brave gentle sharp clever keen lively merry modest noble patient quick ready smart strong sturdy tidy vivid witty bold busy candid clear crisp curious dapper deft earnest fair fleet frank fresh glad hale hardy jolly kind light loyal lucid mild neat nimble plain polite prime prompt proper quaint sage snug sound spry suave sunny terse tough true warm)
VERBS=(running watching listening signaling scouting probing scanning tracking mapping sensing polling pinging syncing relaying beaming calling charting coding dialing drifting echoing flagging flashing gliding guarding guiding hailing hopping humming hunting indexing joining jumping knocking lacing leading linking looping marking meshing mining moving naming noting pacing parsing paving peering pinning pulsing quizzing racing ringing roaming routing scaling scouring seeking serving signing sorting sounding steering surveying)

ADJ_IDX=$(( 16#${HASH:0:2} % ${#ADJECTIVES[@]} ))
VERB_IDX=$(( 16#${HASH:2:2} % ${#VERBS[@]} ))
HASH_TAIL="${HASH:4:6}"

AGENT_NAME="${ADJECTIVES[$ADJ_IDX]}-${VERBS[$VERB_IDX]}-${ARCH}-${HASH_TAIL}"

echo "Interface : $IFACE"
echo "MAC       : $MAC"
echo "Arch      : $ARCH"
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
