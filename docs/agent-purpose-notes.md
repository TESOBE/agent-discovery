# Notes on the agent's purpose

A reflection on the question: "is streaming a second purpose for the agent,
on top of the original agent-to-agent connection purpose?"

## The agent doesn't stream — it operates the streamer

The actual streaming is `ffmpeg` running in a separate `stream.service`
systemd unit. The agent just calls `systemctl start|stop|is-active` on it.
So strictly speaking, the new purpose isn't "stream" — it's "be a remote
control plane for a box that happens to stream." That distinction matters
because it tells you where the agent's responsibility stops.

## The watchdog and wifi pieces aren't really "for streaming" — they protect purpose #1

If the Pi loses network, you can't reach the agent *at all*, and neither
purpose works. WiFi switching and the hotspot watchdog are infrastructure
for keeping the agent reachable. They'd be just as relevant to a Pi that
only did agent-to-agent work in the field.

## The interesting architectural move is the split, not the addition

There are now two command surfaces over the same OBP signal substrate:

- `task-requests` → Claude-mediated, open-ended, flexible
- `system-commands` → strict typed enum, no interpretation

That's a deliberate "Claude is the wrong tool for this" decision, and the
comment in `system_commands/mod.rs` calls it out. Worth keeping that line
sharp as more commands are added — anything where a misread payload could
brick the Pi belongs on the typed channel; anything that benefits from
natural language belongs on the Claude one.

## Three surfaces, not two

Stepping back further, the agent actually exposes three distinct surfaces,
and they answer different questions:

- **Audio / proximity discovery** (`src/audio/*`, `src/discovery/*`,
  chirp / DTMF / modulator / demodulator) → "who is near me, and how do
  we find each other in the first place?" This isn't a command channel;
  it's the bootstrap that lets two agents become aware of each other
  out-of-band from any network.
- **`task-requests`** (Claude-mediated) → "do this open-ended thing for
  me," with an LLM in the loop to interpret intent.
- **`system-commands`** (typed enum) → "execute this exact operation,"
  with no interpretation step, for things where misreading the payload
  would brick the Pi.

The streamer branch only adds the third. The first — audio discovery —
is untouched and remains a first-class capability. It's worth being
explicit that audio discovery is conceptually upstream of the other two:
without it, in a field setting with no shared network, the agents have
nothing to send commands *over* in the first place.

## Where it could drift

Each new `SystemCommand` variant nudges the agent toward "mini OS over
OBP." `WifiConnect`, `StreamStart` are fine — they're the things you'd
lose remote access trying to do any other way. But it's worth having a
rule for what *doesn't* go in there (e.g. anything you could do via a
one-off Claude task without risk to reachability), or the typed enum
will sprawl.

## Reframe

Less: "second purpose = streaming."

More: the agent is becoming a **remote ops endpoint for a field-deployed
Pi**, with streaming as the first concrete workload riding on that
capability. The next workload (whatever it is) will probably reuse the
same `system-commands` channel rather than add a third one.
