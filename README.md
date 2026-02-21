# Agent Discovery via Audio Chirps

Autonomous agents discover each other by playing audio chirps through speakers and listening via microphones. A chirp is a tone that sweeps from one frequency to another — agents use these sweeping tones to announce their presence and transmit connection details. Once discovered, agents use Claude to negotiate a communication channel and then explore OBP (Open Bank Project) dynamic entities together.

## Design Approach

Most agent and service discovery systems use network-level mechanisms like mDNS (multicast DNS, also known as Bonjour/Zeroconf), multicast UDP, or a central service registry. This project deliberately uses audible sound waves instead — it works across devices that share physical space but might not share a network.

Once agents find each other via audio, they communicate over raw TCP/IP sockets using **length-prefixed message framing**: each message is preceded by a 4-byte big-endian length, followed by a JSON payload. This is a well-established systems programming pattern (used by databases like PostgreSQL and Redis, message brokers like Kafka, and game servers) that is simpler and more efficient than HTTP when you already have a direct connection and don't need the overhead of HTTP headers and routing.

## How It Works

### 1. Audio Discovery

Each agent periodically transmits a two-part audio signal:

- **CALL chirp** — a short tone sweeping from ~800 Hz to ~3200 Hz (lasting 120–180 ms). Each agent derives a unique frequency signature from its name (via FNV-1a hash), so no two agents sound exactly alike.
- **Binary chirp message** — immediately after the CALL chirp (80 ms gap), the agent transmits its TCP/IP port number and capability flags as binary data encoded in chirps. An up-chirp (1500 Hz rising to 2500 Hz) represents a `1` bit; a down-chirp (2500 Hz falling to 1500 Hz) represents a `0` bit. Each bit takes 100 ms, giving a data rate of 10 bits per second. The full message (preamble + sync + 16-bit port + 8-bit capabilities + 8-bit checksum) is 44 bits and takes ~4.4 seconds to transmit.

All agents listen continuously. When an agent hears a CALL chirp followed by a valid binary message, it knows another agent exists and how to reach it.

### 2. Self-Echo Rejection

Since an agent's own speaker output reaches its own microphone, each agent must distinguish its own transmissions from those of others. Two mechanisms handle this:

- **TX muting** — while transmitting (and for 300 ms after), incoming audio is discarded.
- **Agent-specific signatures** — if a chirp is detected, the agent checks whether it matches its own unique frequency signature. If so, it is discarded as a self-echo.
- **Port check** — if the decoded port matches the agent's own port, the message is discarded.

### 3. Transmission Scheduling

Each agent is assigned a fixed second-within-the-minute (derived from its name) to avoid collisions. Announce intervals back off as discovery progresses:

- Before first contact: every 1 minute
- After hearing a peer via audio: every 10 minutes
- After TCP/IP handshake: every 40 minutes

Amplitude ramps from 0.50 to 0.80 over the first 3 announces to avoid startling nearby microphones.

### 4. Channel Negotiation

Once a peer is discovered via audio, the agent asks Claude (via the Anthropic API) to choose the best communication channel. For agents on the same machine, Claude typically selects TCP/IP on localhost.

### 5. TCP/IP Handshake

The discovering agent connects to the peer's TCP/IP port (learned from the binary chirp message) and exchanges JSON messages:

```
→ {"type": "hello", "agent_id": "...", "agent_name": "...", "message": "..."}
← {"type": "hello_ack", "agent_id": "...", "agent_name": "...", "message": "..."}
```

Messages are framed with a 4-byte big-endian length prefix. A handshake chime plays on success.

### 6. OBP Dynamic Entity Exploration

After the TCP/IP handshake, both agents collaboratively explore OBP's dynamic entity system over the same TCP/IP connection. The protocol runs in 6 phases:

1. **ExploreStart / ExploreAck** — agree to begin exploration
2. **Discover management endpoint** — find the `createSystemLevelDynamicEntity` endpoint (via MCP or hardcoded fallback)
3. **Get entity schema** — retrieve the full endpoint specification
4. **Create entity** — create an `agent_handshake` dynamic entity in OBP
5. **Discover CRUD endpoints** — parse HATEOAS `_links` from OBP responses to find create/read/update/delete URLs (both agents verify independently)
6. **Create and verify test record** — write a handshake record and confirm the other agent can read it back

### 7. UDP Fallback

If no TCP/IP handshake has completed after 10 minutes, the agent falls back to UDP (User Datagram Protocol) broadcast discovery on port 7399. The `--udp` flag skips the 10-minute wait and starts UDP immediately.

## Cross-Network Communication via OBP Signal Channels

Currently, agents on the same local network discover each other via audio and connect via TCP/IP. For agents on different networks, OBP provides **Signal Channels** — a Redis-backed messaging system designed specifically for AI agent discovery and coordination.

Unlike dynamic entities (which persist to a database), signal channels are ephemeral: channels are auto-created on first publish, messages expire after a configurable TTL (time to live, default 1 hour), and nothing is written to a database. This makes them ideal for lightweight agent-to-agent signaling across network boundaries.

### Signal Channel Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/obp/v6.0.0/signal/channels` | List active signal channels |
| `POST` | `/obp/v6.0.0/signal/channels/CHANNEL_NAME/messages` | Publish a message to a channel |
| `GET` | `/obp/v6.0.0/signal/channels/CHANNEL_NAME/messages` | Fetch messages (oldest-first, with offset/limit pagination) |
| `GET` | `/obp/v6.0.0/signal/channels/CHANNEL_NAME/info` | Get channel metadata (message count, remaining TTL) |
| `DELETE` | `/obp/v6.0.0/signal/channels/CHANNEL_NAME` | Delete a channel and all its messages |

### Key Features

- **No database writes** — all messages are stored in Redis and expire automatically
- **Auto-created channels** — publishing to a channel name creates it; no setup needed
- **Broadcast and private messages** — leave `to_user_id` empty for broadcast, or set it to send a private message visible only to sender and recipient
- **Polling with offset** — track your read position and poll for new messages using the `offset` parameter
- **Capped channels** — maximum 1000 messages per channel (configurable)
- **Privacy filtering** — the server only returns broadcasts and private messages addressed to you

Agents could use signal channels both for discovery (announcing presence on a well-known channel name) and for ongoing coordination, while continuing to use dynamic entities for persistent shared data like handshake records.

## Audio Systems

This project contains two separate audio encoding systems:

### Chirp System (`src/audio/chirp.rs`) — used at runtime

The chirp system is used during `cargo run -- run` for agent discovery. It encodes data as frequency sweeps (chirps) in two bands:

| Band | Frequency Range | Purpose |
|------|----------------|---------|
| CALL | 800–3400 Hz | Agent presence announcement |
| Data | 1500–2500 Hz | Binary message (port + capabilities) |

The data band is deliberately narrow (1000 Hz span, 60 ms per chirp) so it does not trigger the CALL chirp detector (which requires a wider sweep of at least 500 Hz over at least 100 ms in the 600–3600 Hz range).

Detection uses FFT (Fast Fourier Transform) spectrogram analysis with adaptive noise-floor thresholds.

### FSK System (`src/audio/modulator.rs`, `src/audio/demodulator.rs`) — CLI test commands only

FSK (Frequency-Shift Keying) encodes data by switching between two fixed frequencies: one for `1` bits and another for `0` bits (rather than sweeping). This system provides 9 non-overlapping frequency bands (800 Hz to 11400 Hz) at 300 bits per second.

The FSK system is **not** used during agent discovery at runtime. It is available for CLI test commands:

```
cargo run -- test-tone -m "HI"    # Play an FSK-encoded tone
cargo run -- decode -d 10          # Listen for FSK tones for 10 seconds
cargo run -- test-roundtrip        # Play and decode an FSK tone via mic
```

## Setup

### Install Rust

Install Rust via [rustup](https://rustup.rs/) (works on Linux, macOS, Windows, and Raspberry Pi):

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the prompts (the defaults are fine), then either restart your shell or run:

```
source "$HOME/.cargo/env"
```

On a Raspberry Pi you may also need system libraries for audio and TLS:

```
sudo apt update
sudo apt install -y libasound2-dev pkg-config
```

Verify the install:

```
rustc --version
cargo --version
```

### Configure environment

```
cp .env.example .env
# Edit .env - at minimum set CLAUDE_API_KEY
# Set INSTRUCTOR_USER_IDS to the OBP user IDs allowed to send task requests
```

## Run

```
cargo run -- run
```

Multiple agents on the same machine:
```
AGENT_NAME=mumma1 AGENT_LISTEN_PORT=7312 cargo run -- run
AGENT_NAME=mumma2 AGENT_LISTEN_PORT=7313 cargo run -- run
AGENT_NAME=mumma3 AGENT_LISTEN_PORT=7314 cargo run -- run
```

## Other Commands

```
cargo run -- list-devices          # Show audio devices
cargo run -- test-tone -m "HI"    # Play an FSK-encoded tone
cargo run -- decode -d 10          # Listen for FSK tones for 10 seconds
cargo run -- test-roundtrip        # Play and decode via mic
```

## Tests

```
cargo test
```
