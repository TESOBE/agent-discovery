/// Peer data structure and timeout tracking.

use crate::audio::chirp::UrlChunk;
use crate::protocol::message::Capabilities;
use std::time::Instant;
use uuid::Uuid;

/// Fallback timeout if a peer doesn't advertise its next hello interval.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Safety margin added to the peer's advertised interval before expiring it.
/// Gives time for network jitter, clock skew, and transmission delays.
const GRACE_PERIOD_SECS: u64 = 30;

#[derive(Debug, Clone)]
pub struct Peer {
    pub agent_id: Uuid,
    pub address: String,
    pub capabilities: Capabilities,
    pub first_seen: Instant,
    pub last_seen: Instant,
    /// How many minutes until the peer's next hello (as advertised).
    /// 0 means unknown / not advertised — use DEFAULT_TIMEOUT_SECS.
    pub next_hello_mins: u8,
    /// The OBP API base URL this peer uses (empty if unknown).
    pub obp_api_base_url: String,
    /// Accumulator for URL chunks arriving over multiple hello cycles.
    /// `None` = no chunks arriving yet.
    /// `Some(vec)` = vec of length `total_chunks`, each slot `None` or `Some(data)`.
    pub url_chunks: Option<Vec<Option<Vec<u8>>>>,
}

impl Peer {
    pub fn new(agent_id: Uuid, address: String, capabilities: Capabilities, obp_api_base_url: String) -> Self {
        let now = Instant::now();
        Self {
            agent_id,
            address,
            capabilities,
            first_seen: now,
            last_seen: now,
            next_hello_mins: 0,
            obp_api_base_url,
            url_chunks: None,
        }
    }

    /// Insert a received URL chunk into the accumulator.
    ///
    /// Initializes the accumulator on the first chunk. Resets if `total_chunks`
    /// changes (URL changed on the peer side).
    pub fn receive_url_chunk(&mut self, chunk: &UrlChunk) {
        let total = chunk.total_chunks as usize;
        let idx = chunk.chunk_index as usize;

        // Reset if total_chunks changed (peer URL changed)
        if let Some(ref slots) = self.url_chunks {
            if slots.len() != total {
                self.url_chunks = None;
            }
        }

        // Initialize if needed
        if self.url_chunks.is_none() {
            self.url_chunks = Some(vec![None; total]);
        }

        if let Some(ref mut slots) = self.url_chunks {
            if idx < slots.len() {
                slots[idx] = Some(chunk.data.clone());
            }
        }
    }

    /// Try to assemble the full URL from accumulated chunks.
    ///
    /// Returns `Some(url)` if all chunk slots are filled and the result is valid UTF-8.
    pub fn try_assemble_url(&self) -> Option<String> {
        let slots = self.url_chunks.as_ref()?;
        let mut all_bytes = Vec::new();
        for slot in slots {
            all_bytes.extend_from_slice(slot.as_ref()?);
        }
        String::from_utf8(all_bytes).ok()
    }

    /// Update the last_seen timestamp and optionally the address/capabilities.
    pub fn update(&mut self, address: String, capabilities: Capabilities, obp_api_base_url: String) {
        self.last_seen = Instant::now();
        self.address = address;
        self.capabilities = capabilities;
        self.obp_api_base_url = obp_api_base_url;
    }

    /// Update with next_hello_mins included.
    pub fn update_with_interval(&mut self, address: String, capabilities: Capabilities, next_hello_mins: u8) {
        self.update(address, capabilities, self.obp_api_base_url.clone());
        self.next_hello_mins = next_hello_mins;
    }

    /// The timeout for this peer in seconds, based on its advertised interval.
    pub fn timeout_secs(&self) -> u64 {
        if self.next_hello_mins == 0 {
            DEFAULT_TIMEOUT_SECS
        } else {
            (self.next_hello_mins as u64) * 60 + GRACE_PERIOD_SECS
        }
    }

    /// Check if this peer has timed out.
    pub fn is_expired(&self) -> bool {
        self.last_seen.elapsed().as_secs() >= self.timeout_secs()
    }
}
