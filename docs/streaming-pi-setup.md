# Streaming Pi Setup — Architecture Summary

## Problem

A Sony Cinema Line camera → ATEM Mini Pro → OBS on Linux laptop → Restream → Twitch streaming chain has been consistently unreliable over years. The core issues:

- The laptop is a developer machine — a moving target with frequent updates that break the setup
- Streaming happens only every 1-2 months, so the setup is always out of date by the time it's needed
- Blackmagic (ATEM) Linux drivers break on kernel updates
- OBS on Linux adds further fragility
- Restream adds an extra hop between OBS and Twitch

---

## Solution: Dedicated Frozen Pi 5

Replace the laptop entirely as the streaming device with a **Raspberry Pi 5** running a minimal, never-updated OS. The Pi handles both streaming and network connectivity.

### Key Principles
- **Pi OS Lite** (headless, no desktop) — leaner and more stable
- **Never run `apt upgrade`** — treat it as an appliance, not a computer
- **Snapshot the SD card** once working — instant restore if anything breaks

---

## Final Hardware Chain

```
Sony HDMI ──┐
             ├→ ATEM Mini Pro → USB → Pi 5 → ffmpeg → Restream → Twitch
Mixer 3.5mm ┘
```

- **Sony camera** outputs clean 1080p / Rec.709 / SDR via HDMI
- **ATEM Mini Pro** consolidates camera HDMI + mixer 3.5mm audio into a single USB feed
- **Pi 5** ingests USB capture, encodes, and streams
- **No OBS** — replaced by a single ffmpeg command
- **No laptop** in the signal chain

### Why Keep the ATEM?
Although explored removing it, the ATEM earns its place by combining two sources (camera HDMI + mixer audio) into one USB feed to the Pi. Already owned, so no cost argument against it.

---

## Streaming: ffmpeg Instead of OBS

OBS is overkill — no scenes, no overlays, no switching needed. A single ffmpeg command replaces it:

```bash
ffmpeg \
  -f v4l2 -i /dev/video0 \
  -f alsa -i hw:0 \
  -vcodec libx264 -preset veryfast -b:v 5500k -minrate 5500k -maxrate 5500k \
  -bufsize 11000k -g 60 -keyint_min 60 \
  -acodec aac -b:a 160k \
  -f flv rtmp://live.restream.io/live/YOUR_KEY
```

Runs as a **systemd service** — starts automatically on boot when ATEM is connected.

### OBS Settings (for reference, if OBS ever used)
- Encoder: x264 or NVENC
- Rate Control: CBR
- Bitrate: 5,500 kbps
- Keyframe Interval: 2 seconds
- Audio Bitrate: 160 kbps

---

## Network: Phone Hotspot as Primary

- Pi connects automatically to a **fixed phone hotspot** (known SSID + password)
- Venue WiFi is unreliable for streaming (unknown upload speed, captive portals, shared bandwidth)
- Phone hotspot data for a 1-2 hour stream at 5.5 Mbps ≈ 3-4 GB — acceptable

---

## Agent: OBP Signal-Based Remote Control

Rather than a dumb appliance, the Pi runs an agent that can be commanded remotely via **OBP (Open Bank Project) Signal messages**, sent from a phone using the OBP API Manager.

OBP signal channels are generic: named channels accepting arbitrary JSON payloads over the existing `/obp/.../signal/channels/<name>/messages` endpoints. **No new OBP endpoints are required** — only a new channel name and a payload schema the agent knows how to route.

### Why This Matters
- At a venue, you can control the Pi from your phone without touching it
- Commands can include WiFi credentials, stream start/stop, status checks
- The agent-discovery repo (`github.com/TESOBE/agent-discovery`) is the starting point — a Rust agent that already handles OBP signal communication

### Channel Design: Dedicated `system-commands`

The agent already polls a `task-requests` channel and routes each message through Claude (see `run_task_request_poller` in `src/agent.rs`). That path is right for natural-language tasks but wrong for hard system operations like switching WiFi — we don't want Claude interpreting intent when a bad join could take the Pi offline.

A separate **`system-commands`** channel is polled by its own handler with:
- **Direct dispatch, no Claude in the path** — no hallucination, no API dependency, no extra latency
- **Strict schema** — payloads deserialise into a typed enum or are rejected
- **Instructor-only auth** — reuses the existing `config.instructor_user_ids` check
- **Independent poll cadence** — faster (e.g. 10s) than Claude-mediated tasks

Responses go back on `system-command-responses`.

### Payload Schema

```json
{"type": "wifi-connect", "ssid": "VenueWiFi", "password": "...", "priority": 50}
{"type": "wifi-forget",  "ssid": "VenueWiFi"}
{"type": "stream-start"}
{"type": "stream-stop"}
{"type": "stream-status"}
```

The `password` field maps internally to NetworkManager's `wifi-sec.psk` property (WPA/WPA2/WPA3-Personal pre-shared key).

### Communication Flow
```
Phone → OBP API Manager → system-commands channel → Pi agent → nmcli/systemctl
                                                          ↓
                               system-command-responses channel ← Pi agent
```

### Bootstrapping Problem
The Pi needs internet to receive OBP signals — but needs a signal to get internet credentials. Solution: **phone hotspot as guaranteed first connection**. Once online, the agent can receive further commands including switching to venue WiFi.

---

## Robust WiFi Switching: Confirm-or-Revert

Switching WiFi from a remote command is dangerous: a wrong SSID/PSK leaves the Pi offline and unreachable. The agent follows a confirm-or-revert protocol:

```
1. Receive   {type: "wifi-connect", ssid, psk}
2. Snapshot  current active connection name via `nmcli -t -f NAME c show --active`
3. Add/update profile via nmcli (autoconnect yes, priority 50)
4. Attempt   `nmcli c up <ssid>` with ~20s timeout
5. Probe     HTTP GET OBP base URL, 3 tries, 10s each
6. On success → post `wifi-connected` signal, done
   On failure → `nmcli c down <ssid>`, `nmcli c up <previous>`,
                post `wifi-connect-failed` (arrives once old network restored)
```

### Belt-and-Braces Fallbacks

- **Hotspot profile pinned at `connection.autoconnect-priority 100`** so NetworkManager auto-falls-back even if agent revert logic itself crashes
- **Watchdog task** — a separate loop that, if OBP has been unreachable for N minutes, forcibly re-joins the hotspot profile. Handles the case where the agent process wedges mid-switch

---

## What Needs Building

**On the Pi:**
- New Rust module `src/system_commands/` in the agent:
  - `mod.rs` — `SystemCommand` enum, poller task, response publishing
  - `wifi.rs` — `nmcli` wrappers, snapshot/attempt/probe/revert state machine
  - `stream.rs` — `systemctl start|stop|status` wrappers for the ffmpeg unit
  - `watchdog.rs` — reachability monitor + forced hotspot fallback
- Wire the poller into `agent.rs` alongside the existing task-request poller
- systemd unit for the ffmpeg streaming service
- Pre-configured NetworkManager profile for the phone hotspot at priority 100

**On OBP:**
- Channel `system-commands` created (and `system-command-responses` for replies)
- Pi agent registered with instructor user IDs authorised to post commands
- No new endpoint definitions — existing signal-channel APIs are sufficient
