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

Multiple agents on the same machine:
```
AGENT_NAME=mumma1 AGENT_LISTEN_PORT=7312 cargo run -- run
AGENT_NAME=mumma2 AGENT_LISTEN_PORT=7313 cargo run -- run
AGENT_NAME=mumma3 AGENT_LISTEN_PORT=7314 cargo run -- run
```

Each agent transmits on one of 9 frequency bands (chosen from the last digit in the agent name). All agents listen on all 9 bands so they hear everyone. TX muting prevents self-echo. Announce intervals back off from 2s to 30s when no peers respond, and reset when a peer is found.

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
