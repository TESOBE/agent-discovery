# Agent Discovery via Audio FSK

Autonomous agents discover each other by playing audio tones (FSK) through speakers and listening via microphones. Once discovered, agents use Claude to negotiate how to communicate, optionally using OBP as shared storage.

## Setup

```
cp .env.example .env
# Edit .env - at minimum set CLAUDE_API_KEY
```

## Run

```
cargo run -- run
```

Two agents on the same machine:
```
AGENT_NAME=mumma1 AGENT_LISTEN_PORT=7312 cargo run -- run
AGENT_NAME=mumma2 AGENT_LISTEN_PORT=7313 cargo run -- run
```

## Other commands

```
cargo run -- list-devices       # Show audio devices
cargo run -- test-tone -m "HI"  # Play an FSK tone
cargo run -- decode -d 10       # Listen for 10 seconds
```

## Tests

```
cargo test
```
