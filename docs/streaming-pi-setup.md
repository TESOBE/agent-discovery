# Streaming Pi Setup — Architecture Summary

## Problem

A Sony Cinema Line camera → ATEM Mini Pro → OBS on Linux laptop → Restream → Twitch streaming chain has been consistently unreliable over years. The core issues:

- The laptop is a developer machine — a moving target with frequent updates that break the setup
- Streaming happens only every 1-2 months, so the setup is always out of date by the time it's needed
- Blackmagic (ATEM) Linux drivers break on kernel updates
- OBS on Linux adds further fragility
- Restream added an extra hop between OBS and Twitch (since dropped — see below)

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
             ├→ ATEM Mini Pro → USB → Pi 5 → ffmpeg → YouTube Live
Mixer 3.5mm ┘
```

- **Sony camera** outputs clean 1080p / Rec.709 / SDR via HDMI
- **ATEM Mini Pro** consolidates camera HDMI + mixer 3.5mm audio into a single USB feed
- **Pi 5** ingests USB capture, encodes, and streams
- **No OBS** — replaced by a single ffmpeg command
- **No laptop** in the signal chain
- **No Restream** — ffmpeg pushes directly to YouTube's RTMP ingest, removing one external service from the critical path

### Why Keep the ATEM?
Although explored removing it, the ATEM earns its place by combining two sources (camera HDMI + mixer audio) into one USB feed to the Pi. Already owned, so no cost argument against it.

### Why YouTube Direct (No Restream)?
Restream was previously used as a fan-out hub. Streaming directly to YouTube's RTMP ingest removes a third-party hop, simplifies failure modes (one service to be up, not two), and avoids Restream's pricing tiers. The trade-off is no automatic fan-out to multiple platforms — fine for a single-destination workflow.

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
  -f flv rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY
```

YouTube's primary RTMP ingest is `rtmp://a.rtmp.youtube.com/live2`; the backup is `rtmp://b.rtmp.youtube.com/live2?backup=1`. The stream key lives in YouTube Studio under Live → Stream → Stream key (use a "default stream" / persistent key for repeatable testing).

Runs as a **systemd service** (`stream.service`, installed via `install-stream-service.sh`). The unit reads `RTMP_URL` and `RTMP_STREAM_KEY` from `.stream-env` so the key isn't baked into the unit file.

The installer enables the service at boot — the Pi starts streaming as soon as it's online. A `TEST_PATTERN=1` flag in `.stream-env` selects a synthetic lavfi source (colour bars + 1 kHz sine) so this works even without capture hardware connected; flip to `TEST_PATTERN=0` and re-run the installer to switch to v4l2/alsa capture. The agent's `system-commands` handler can also start/stop the service on demand for the duration of the current boot.

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

## Implementation Status

**Done in this repo:**
- `src/system_commands/` module — `mod.rs` (poller + dispatch), `wifi.rs` (nmcli confirm-or-revert), `stream.rs` (`systemctl` wrappers), `watchdog.rs` (reachability + forced hotspot fallback)
- Wired into `agent.rs` alongside the existing task-request poller
- `install-stream-service.sh` + `.stream-env.example` — generates `/etc/systemd/system/stream.service` from per-Pi env (RTMP_URL, RTMP_STREAM_KEY, TEST_PATTERN, V4L2_DEVICE, ALSA_DEVICE, BITRATE_KBPS, BUFSIZE_KBPS). The installer enables the unit at boot. `TEST_PATTERN=1` (default) picks a synthetic lavfi source so it works without capture hardware; `TEST_PATTERN=0` switches to v4l2/alsa capture
- `cargo run -- stream <start|stop|status>` — CLI smoke-test for the systemctl wrappers without going through OBP

**Per-Pi setup (manual):**
- Install ffmpeg: `sudo apt install ffmpeg`
- Fill in `.stream-env` from the example, with the YouTube RTMP URL and stream key
- `sudo bash install-stream-service.sh` to write the unit
- NetworkManager profile for the phone hotspot at `connection.autoconnect-priority 100` (so it auto-falls-back even if the agent's revert logic crashes mid-switch)

**On OBP:**
- Channel `system-commands` is auto-created on first publish (no setup needed)
- The Pi agent's `INSTRUCTOR_USER_IDS` env var must include the OBP user IDs allowed to post commands
- No new endpoint definitions — existing signal-channel APIs are sufficient
