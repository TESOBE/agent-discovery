/// Transport for the post-discovery exploration protocol.
///
/// Two flavours:
/// - `Tcp`: direct point-to-point TCP stream. Fast (millisecond round-trips),
///   only works when both peers are on the same network.
/// - `Signal`: each agent has an inbox channel `agent-inbox-{agent_id}` on
///   OBP. To send a message, post to the peer's inbox. To receive, poll our
///   own inbox and filter for messages from the expected peer. Slower (paced
///   by the poll interval), but works between peers on different networks
///   that share an OBP host.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::comms::tcp::TcpChannel;
use crate::comms::channel::CommChannel;
use crate::obp::client::{ObpClient, API_VERSION};
use crate::obp::exploration::ExplorationMsg;

/// Inbox channel name for a given agent id.
pub fn agent_inbox_channel(agent_id: Uuid) -> String {
    format!("agent-inbox-{}", agent_id)
}

/// Default poll interval for signal-channel inbox reads.
pub const DEFAULT_INBOX_POLL_INTERVAL: Duration = Duration::from_secs(15);

/// Default total timeout for waiting on a single recv() over a signal channel.
pub const DEFAULT_INBOX_RECV_TIMEOUT: Duration = Duration::from_secs(600);

/// Bidirectional transport that ferries `ExplorationMsg` between two agents.
pub enum HandshakeChannel {
    Tcp(Arc<TcpChannel>),
    Signal(SignalHandshakeChannel),
}

impl HandshakeChannel {
    /// One-line transport description, for log spans.
    pub fn description(&self) -> String {
        match self {
            HandshakeChannel::Tcp(c) => c.description(),
            HandshakeChannel::Signal(s) => format!(
                "signal({}@{})",
                s.peer_agent_id,
                s.obp.label()
            ),
        }
    }

    pub async fn send(&self, msg: &ExplorationMsg) -> Result<()> {
        match self {
            HandshakeChannel::Tcp(ch) => {
                let bytes = serde_json::to_vec(msg)?;
                let ch = ch.clone();
                tokio::task::spawn_blocking(move || ch.send_message(&bytes)).await??;
                Ok(())
            }
            HandshakeChannel::Signal(s) => s.send(msg).await,
        }
    }

    pub async fn recv(&self) -> Result<ExplorationMsg> {
        match self {
            HandshakeChannel::Tcp(ch) => {
                let ch = ch.clone();
                let bytes = tokio::task::spawn_blocking(move || ch.recv_message()).await??;
                let msg: ExplorationMsg = serde_json::from_slice(&bytes)?;
                Ok(msg)
            }
            HandshakeChannel::Signal(s) => s.recv().await,
        }
    }
}

/// Signal-channel transport. Each instance is scoped to a single peer
/// conversation. Sends post the envelope to the peer's inbox; receives
/// poll our own inbox and filter for messages from this peer.
pub struct SignalHandshakeChannel {
    pub obp: Arc<ObpClient>,
    pub self_agent_id: Uuid,
    pub peer_agent_id: Uuid,
    pub poll_interval: Duration,
    pub recv_timeout: Duration,
    seen_msg_ids: Mutex<HashSet<String>>,
}

impl SignalHandshakeChannel {
    pub fn new(obp: Arc<ObpClient>, self_id: Uuid, peer_id: Uuid) -> Self {
        Self {
            obp,
            self_agent_id: self_id,
            peer_agent_id: peer_id,
            poll_interval: DEFAULT_INBOX_POLL_INTERVAL,
            recv_timeout: DEFAULT_INBOX_RECV_TIMEOUT,
            seen_msg_ids: Mutex::new(HashSet::new()),
        }
    }

    /// Pre-populate the seen-set so a message we already consumed via the
    /// inbox-listener loop isn't returned again by the first `recv()`.
    pub async fn mark_seen(&self, msg_id: String) {
        self.seen_msg_ids.lock().await.insert(msg_id);
    }

    async fn send(&self, msg: &ExplorationMsg) -> Result<()> {
        use crate::obp::entities::iso_now;
        let msg_id = Uuid::new_v4().to_string();
        let envelope = serde_json::json!({
            "payload": {
                "type": "handshake",
                "from_agent_id": self.self_agent_id.to_string(),
                "to_agent_id": self.peer_agent_id.to_string(),
                "msg_id": msg_id,
                "exploration_msg": serde_json::to_value(msg)?,
                "timestamp": iso_now(),
            }
        });
        let path = format!(
            "/obp/{}/signal/channels/{}/messages",
            API_VERSION,
            agent_inbox_channel(self.peer_agent_id)
        );
        self.obp.post(&path, &envelope).await.with_context(|| {
            format!(
                "Failed to send handshake msg to peer inbox {} on host {}",
                self.peer_agent_id,
                self.obp.label()
            )
        })?;
        Ok(())
    }

    async fn recv(&self) -> Result<ExplorationMsg> {
        let path = format!(
            "/obp/{}/signal/channels/{}/messages",
            API_VERSION,
            agent_inbox_channel(self.self_agent_id)
        );
        let start = std::time::Instant::now();
        loop {
            match self.obp.get(&path).await {
                Ok(resp) => {
                    if let Some(found) = self.scan_for_next(&resp).await? {
                        return Ok(found);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Signal handshake inbox poll failed (host {}): {}",
                        self.obp.label(),
                        e
                    );
                }
            }

            if start.elapsed() > self.recv_timeout {
                anyhow::bail!(
                    "SignalHandshakeChannel recv timeout from peer {} on host {}",
                    self.peer_agent_id,
                    self.obp.label()
                );
            }
            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Walk the messages array, return the first one matching this peer that
    /// we haven't already returned. Marks it as seen.
    async fn scan_for_next(&self, resp: &serde_json::Value) -> Result<Option<ExplorationMsg>> {
        let messages = match resp.get("messages").and_then(|m| m.as_array()) {
            Some(m) => m,
            None => return Ok(None),
        };
        let mut seen = self.seen_msg_ids.lock().await;
        for msg in messages {
            let payload = match msg.get("payload") {
                Some(p) => p,
                None => continue,
            };
            if payload.get("type").and_then(|v| v.as_str()) != Some("handshake") {
                continue;
            }
            if payload.get("from_agent_id").and_then(|v| v.as_str())
                != Some(&self.peer_agent_id.to_string())
            {
                continue;
            }
            let msg_id = match payload.get("msg_id").and_then(|v| v.as_str()) {
                Some(id) if !id.is_empty() => id.to_string(),
                _ => continue,
            };
            if seen.contains(&msg_id) {
                continue;
            }
            let exploration_val = match payload.get("exploration_msg") {
                Some(v) => v.clone(),
                None => continue,
            };
            let exp_msg: ExplorationMsg = serde_json::from_value(exploration_val)
                .context("Failed to deserialize exploration_msg from inbox")?;
            seen.insert(msg_id);
            return Ok(Some(exp_msg));
        }
        Ok(None)
    }
}
