/// Discovery manager: handles announce scheduling, receiving, peer tracking, and events.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::audio::demodulator;
use crate::audio::device::LoopbackAudioEngine;
use crate::audio::modulator;
use crate::audio::modulator::{BANDS, NUM_BANDS};
use crate::protocol::codec;
use crate::protocol::frame;
use crate::protocol::message::{Capabilities, DiscoveryMessage};

use super::peer::{Peer, PEER_TIMEOUT_SECS};

/// Discovery events emitted by the manager.
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    PeerDiscovered(Peer),
    PeerUpdated(Peer),
    PeerLost(Uuid),
}

/// Configuration for the discovery manager.
pub struct DiscoveryConfig {
    pub agent_id: Uuid,
    pub agent_address: String,
    pub capabilities: Capabilities,
    pub announce_interval: Duration,
    pub jitter_max_ms: u64,
    /// Which of the 9 frequency bands this agent transmits on (0-8).
    pub tx_band: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            agent_id: Uuid::new_v4(),
            agent_address: "127.0.0.1:7312".to_string(),
            capabilities: Capabilities::new().with_tcp().with_claude(),
            announce_interval: Duration::from_secs(5),
            jitter_max_ms: 1000,
            tx_band: 0,
        }
    }
}

/// The discovery manager tracks known peers and coordinates announcement/reception.
pub struct DiscoveryManager {
    config: DiscoveryConfig,
    peers: HashMap<Uuid, Peer>,
    event_tx: broadcast::Sender<DiscoveryEvent>,
}

impl DiscoveryManager {
    pub fn new(config: DiscoveryConfig) -> (Self, broadcast::Receiver<DiscoveryEvent>) {
        let (event_tx, event_rx) = broadcast::channel(64);
        (
            Self {
                config,
                peers: HashMap::new(),
                event_tx,
            },
            event_rx,
        )
    }

    /// Get a new event receiver (can be called multiple times).
    pub fn subscribe(&self) -> broadcast::Receiver<DiscoveryEvent> {
        self.event_tx.subscribe()
    }

    /// Build the announce message for this agent.
    pub fn build_announce(&self) -> DiscoveryMessage {
        DiscoveryMessage::announce(
            self.config.agent_id,
            &self.config.agent_address,
            self.config.capabilities,
        )
    }

    /// Build a goodbye message for this agent.
    pub fn build_goodbye(&self) -> DiscoveryMessage {
        DiscoveryMessage::goodbye(self.config.agent_id)
    }

    /// Encode a discovery message into FSK audio samples on this agent's TX band.
    pub fn encode_to_samples(&self, msg: &DiscoveryMessage) -> Vec<f32> {
        let msg_bytes = codec::encode_message(msg);
        let frame_bytes = frame::encode_frame(&msg_bytes);
        modulator::modulate_bytes_freq(&frame_bytes, &BANDS[self.config.tx_band])
    }

    /// Try to decode audio samples by attempting all 9 frequency bands
    /// at multiple bit-alignment offsets. The live mic signal won't be
    /// aligned to our bit boundaries, so we try 16 evenly-spaced offsets
    /// within one bit period (160 samples) for each band.
    pub fn decode_from_samples(&self, samples: &[f32]) -> Option<DiscoveryMessage> {
        let step = (modulator::SAMPLES_PER_BIT as usize) / 16; // 10 samples
        for band in 0..NUM_BANDS {
            for offset_idx in 0..16 {
                let offset = offset_idx * step;
                let bits = demodulator::demodulate_samples_freq_at(samples, &BANDS[band], offset);
                let bytes = modulator::bits_to_bytes(&bits);
                if let Some(payload) = frame::decode_frame(&bytes) {
                    if let Some(msg) = codec::decode_message(&payload) {
                        tracing::info!(band, offset, "Decoded on band {} at offset {}", band, offset);
                        return Some(msg);
                    }
                }
            }
        }
        None
    }

    /// Process a received discovery message. Returns true if it was handled (not self-echo).
    pub fn handle_message(&mut self, msg: &DiscoveryMessage) -> bool {
        // Self-echo filtering
        if msg.agent_id == self.config.agent_id {
            return false;
        }

        match msg.msg_type {
            crate::protocol::message::MessageType::Announce
            | crate::protocol::message::MessageType::Ack => {
                if let Some(peer) = self.peers.get_mut(&msg.agent_id) {
                    peer.update(msg.address.clone(), msg.capabilities);
                    let _ = self.event_tx.send(DiscoveryEvent::PeerUpdated(peer.clone()));
                } else {
                    let peer = Peer::new(msg.agent_id, msg.address.clone(), msg.capabilities);
                    tracing::info!(
                        peer_id = %msg.agent_id,
                        address = %msg.address,
                        "New peer discovered"
                    );
                    let _ = self
                        .event_tx
                        .send(DiscoveryEvent::PeerDiscovered(peer.clone()));
                    self.peers.insert(msg.agent_id, peer);
                }
                true
            }
            crate::protocol::message::MessageType::Goodbye => {
                if self.peers.remove(&msg.agent_id).is_some() {
                    tracing::info!(peer_id = %msg.agent_id, "Peer said goodbye");
                    let _ = self.event_tx.send(DiscoveryEvent::PeerLost(msg.agent_id));
                }
                true
            }
        }
    }

    /// Expire peers that haven't been seen recently.
    pub fn expire_peers(&mut self) {
        let expired: Vec<Uuid> = self
            .peers
            .iter()
            .filter(|(_, peer)| peer.is_expired())
            .map(|(id, _)| *id)
            .collect();

        for id in expired {
            self.peers.remove(&id);
            tracing::info!(peer_id = %id, "Peer expired (timeout={}s)", PEER_TIMEOUT_SECS);
            let _ = self.event_tx.send(DiscoveryEvent::PeerLost(id));
        }
    }

    /// Get a snapshot of known peers.
    pub fn peers(&self) -> Vec<&Peer> {
        self.peers.values().collect()
    }

    /// Get this agent's ID.
    pub fn agent_id(&self) -> Uuid {
        self.config.agent_id
    }

    /// Get the discovery config.
    pub fn config(&self) -> &DiscoveryConfig {
        &self.config
    }
}

/// Run the discovery announce loop using a loopback engine (for testing).
pub async fn run_announce_loop_loopback(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<LoopbackAudioEngine>,
) -> Result<()> {
    loop {
        let samples = {
            let mgr = manager.lock().await;
            let msg = mgr.build_announce();
            mgr.encode_to_samples(&msg)
        };

        engine.send_samples(samples)?;

        let interval = {
            let mgr = manager.lock().await;
            let jitter = rand_jitter(mgr.config.jitter_max_ms);
            mgr.config.announce_interval + Duration::from_millis(jitter)
        };

        tokio::time::sleep(interval).await;
    }
}

/// Run the discovery receive loop using a loopback engine (for testing).
pub async fn run_receive_loop_loopback(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<LoopbackAudioEngine>,
) -> Result<()> {
    loop {
        match engine.try_recv_samples() {
            Some(samples) => {
                let mut mgr = manager.lock().await;
                if let Some(msg) = mgr.decode_from_samples(&samples) {
                    mgr.handle_message(&msg);
                }
            }
            None => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }

        // Periodically expire peers
        {
            let mut mgr = manager.lock().await;
            mgr.expire_peers();
        }
    }
}

fn rand_jitter(max_ms: u64) -> u64 {
    // Simple jitter using time-based randomness (no extra crate needed)
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    nanos % max_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_echo_filtering() {
        let config = DiscoveryConfig::default();
        let agent_id = config.agent_id;
        let (mut manager, _rx) = DiscoveryManager::new(config);

        let msg = DiscoveryMessage::announce(
            agent_id,
            "127.0.0.1:9000",
            Capabilities::new().with_tcp(),
        );

        // Self-echo should be ignored
        assert!(!manager.handle_message(&msg));
        assert!(manager.peers().is_empty());
    }

    #[test]
    fn test_peer_discovery() {
        let config = DiscoveryConfig::default();
        let (mut manager, mut rx) = DiscoveryManager::new(config);

        let peer_id = Uuid::new_v4();
        let msg = DiscoveryMessage::announce(
            peer_id,
            "192.168.1.42:9000",
            Capabilities::new().with_tcp().with_obp(),
        );

        assert!(manager.handle_message(&msg));
        assert_eq!(manager.peers().len(), 1);
        assert_eq!(manager.peers()[0].agent_id, peer_id);

        // Check event was emitted
        let event = rx.try_recv().expect("Should have event");
        match event {
            DiscoveryEvent::PeerDiscovered(peer) => {
                assert_eq!(peer.agent_id, peer_id);
            }
            _ => panic!("Expected PeerDiscovered event"),
        }
    }

    #[test]
    fn test_peer_update() {
        let config = DiscoveryConfig::default();
        let (mut manager, mut rx) = DiscoveryManager::new(config);

        let peer_id = Uuid::new_v4();
        let msg1 = DiscoveryMessage::announce(
            peer_id,
            "192.168.1.42:9000",
            Capabilities::new().with_tcp(),
        );
        let msg2 = DiscoveryMessage::announce(
            peer_id,
            "192.168.1.42:9001",
            Capabilities::new().with_tcp().with_obp(),
        );

        manager.handle_message(&msg1);
        let _ = rx.try_recv(); // consume PeerDiscovered

        manager.handle_message(&msg2);
        assert_eq!(manager.peers().len(), 1);
        assert_eq!(manager.peers()[0].address, "192.168.1.42:9001");

        let event = rx.try_recv().expect("Should have update event");
        matches!(event, DiscoveryEvent::PeerUpdated(_));
    }

    #[test]
    fn test_goodbye_removes_peer() {
        let config = DiscoveryConfig::default();
        let (mut manager, mut rx) = DiscoveryManager::new(config);

        let peer_id = Uuid::new_v4();
        let announce = DiscoveryMessage::announce(
            peer_id,
            "192.168.1.42:9000",
            Capabilities::new().with_tcp(),
        );
        manager.handle_message(&announce);
        let _ = rx.try_recv();

        let goodbye = DiscoveryMessage::goodbye(peer_id);
        manager.handle_message(&goodbye);
        assert!(manager.peers().is_empty());

        let event = rx.try_recv().expect("Should have lost event");
        matches!(event, DiscoveryEvent::PeerLost(_));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // All agents share the same channel, so encode/decode roundtrips.
        let config = DiscoveryConfig::default();
        let (manager, _rx) = DiscoveryManager::new(config);

        let msg = manager.build_announce();
        let samples = manager.encode_to_samples(&msg);
        let decoded = manager.decode_from_samples(&samples).expect("Should decode");

        assert_eq!(msg.agent_id, decoded.agent_id);
        assert_eq!(msg.address, decoded.address);
        assert_eq!(msg.capabilities, decoded.capabilities);
    }
}
