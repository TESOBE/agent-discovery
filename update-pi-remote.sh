#!/bin/bash
# update-pi-remote.sh — From the desktop, trigger the Pi to git pull and rebuild.
#
# Runs `bash update-pi.sh` on the Pi over SSH (which does git pull --ff-only,
# cargo build --release, and restarts agent-discovery; stream.service is only
# touched if .stream-env is newer than the installed unit).
#
# For pushing secret env files instead, use push-secrets.sh.
#
# Configure target via env vars (or edit the defaults below):
#   PI_HOST  — ~/.ssh/config alias or user@ip   (default: pi)
#   PI_DIR   — path to agent-discovery on the Pi
#
# Usage (from the desktop project root):
#   bash update-pi-remote.sh

set -euo pipefail

PI_HOST="${PI_HOST:-pi}"
PI_DIR="${PI_DIR:-/home/colourpi/Documents/workspace/agent-discovery}"

echo "==> Triggering update on ${PI_HOST}:${PI_DIR}"
# Non-interactive SSH skips ~/.bashrc, so source cargo's env here too — belt
# and braces alongside the matching block inside update-pi.sh.
ssh -t "$PI_HOST" "[ -f \$HOME/.cargo/env ] && . \$HOME/.cargo/env; cd '$PI_DIR' && bash update-pi.sh"

echo ""
echo "Done."
