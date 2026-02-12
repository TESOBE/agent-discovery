/// Peer data structure and timeout tracking.

use crate::protocol::message::Capabilities;
use std::time::Instant;
use uuid::Uuid;

/// How long before a peer is considered expired (no announce received).
pub const PEER_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone)]
pub struct Peer {
    pub agent_id: Uuid,
    pub address: String,
    pub capabilities: Capabilities,
    pub first_seen: Instant,
    pub last_seen: Instant,
}

impl Peer {
    pub fn new(agent_id: Uuid, address: String, capabilities: Capabilities) -> Self {
        let now = Instant::now();
        Self {
            agent_id,
            address,
            capabilities,
            first_seen: now,
            last_seen: now,
        }
    }

    /// Update the last_seen timestamp and optionally the address/capabilities.
    pub fn update(&mut self, address: String, capabilities: Capabilities) {
        self.last_seen = Instant::now();
        self.address = address;
        self.capabilities = capabilities;
    }

    /// Check if this peer has timed out.
    pub fn is_expired(&self) -> bool {
        self.last_seen.elapsed().as_secs() >= PEER_TIMEOUT_SECS
    }
}
