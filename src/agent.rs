/// Agent orchestrator: ties audio, discovery, negotiation, and OBP together.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use tracing::Instrument;
use uuid::Uuid;

use crate::audio::device::AudioEngine;
use crate::audio::modulator::{band_from_name, BANDS};
use crate::comms::channel::CommChannel;
use crate::comms::tcp::{TcpChannel, TcpCommListener};
use crate::config::Config;
use crate::discovery::manager::{DiscoveryConfig, DiscoveryEvent, DiscoveryManager};
use crate::mcp::client::McpClient;
use crate::negotiation::claude::ClaudeNegotiator;
use crate::obp::client::ObpClient;
use crate::obp::entities;
use crate::protocol::message::Capabilities;

/// Run the main agent loop.
pub async fn run(config: Config) -> Result<()> {
    let agent_id = Uuid::new_v4();
    let agent_name = config.agent_name.clone();
    let agent_address = format!("127.0.0.1:{}", config.agent_listen_port);

    // Span that appears on every log line for this agent
    let agent_span = tracing::info_span!("agent", name = %agent_name);
    let _span_guard = agent_span.enter();

    tracing::info!(
        agent_id = %agent_id,
        address = %agent_address,
        "Agent starting"
    );

    // Determine capabilities
    let mut capabilities = Capabilities::new().with_tcp();
    if !config.claude_api_key.is_empty() {
        capabilities = capabilities.with_claude();
    }
    if !config.obp_username.is_empty() {
        capabilities = capabilities.with_obp();
    }

    // Try to set up MCP client
    let mcp_client = match McpClient::new(&config).await {
        Ok(client) => {
            tracing::info!("MCP client connected, {} tools available", client.tools().len());
            capabilities = capabilities.with_obp();
            Some(Arc::new(tokio::sync::Mutex::new(client)))
        }
        Err(e) => {
            tracing::warn!("MCP client unavailable: {}. Continuing without MCP.", e);
            None
        }
    };

    // Set up OBP dynamic entities if MCP is available
    if let Some(ref mcp) = mcp_client {
        let mut mcp_guard = mcp.lock().await;
        if let Err(e) = entities::setup_entities_via_mcp(&mut mcp_guard, &config.obp_bank_id).await {
            tracing::warn!("Failed to set up OBP entities via MCP: {}", e);

            // Fallback: try direct HTTP
            let mut obp_client = ObpClient::new(&config);
            if obp_client.authenticate().await.is_ok() {
                let _ = entities::setup_entities_via_http(&obp_client).await;
            }
        }
    }

    // Pick TX band from agent name
    let tx_band = band_from_name(&config.agent_name);
    let tx_freqs = &BANDS[tx_band];
    tracing::info!(
        band = tx_band,
        mark = tx_freqs.mark,
        space = tx_freqs.space,
        "TX band chosen from agent name"
    );

    let discovery_config = DiscoveryConfig {
        agent_id,
        agent_address: agent_address.clone(),
        capabilities,
        tx_band,
        ..Default::default()
    };

    let (discovery_manager, mut event_rx) = DiscoveryManager::new(discovery_config);
    let discovery_manager = Arc::new(tokio::sync::Mutex::new(discovery_manager));

    // Set up TCP listener
    let tcp_listener = TcpCommListener::bind(config.agent_listen_port)?;
    tracing::info!("TCP listener ready on port {}", config.agent_listen_port);

    // Set up audio engine
    let audio_engine = match AudioEngine::new() {
        Ok(engine) => {
            tracing::info!("Audio engine initialized");
            Some(Arc::new(engine))
        }
        Err(e) => {
            tracing::warn!("Audio engine unavailable: {}. Running in discovery-only mode.", e);
            None
        }
    };

    // Shared flag: when true, the RX loop discards samples to avoid self-echo
    let is_transmitting = Arc::new(AtomicBool::new(false));
    // Set to true when a peer is discovered, tells TX loop to reset back-off
    let peer_found = Arc::new(AtomicBool::new(false));

    // Start the announce loop (audio TX)
    if let Some(ref engine) = audio_engine {
        let mgr = discovery_manager.clone();
        let eng = engine.clone();
        let tx_flag = is_transmitting.clone();
        let pf_flag = peer_found.clone();
        let span = tracing::info_span!("agent", name = %agent_name);
        tokio::spawn(async move {
            run_announce_loop(mgr, eng, tx_flag, pf_flag).await;
        }.instrument(span));
    }

    // Start the receive loop (audio RX)
    if let Some(ref engine) = audio_engine {
        let mgr = discovery_manager.clone();
        let eng = engine.clone();
        let tx_flag = is_transmitting.clone();
        let span = tracing::info_span!("agent", name = %agent_name);
        tokio::spawn(async move {
            run_receive_loop(mgr, eng, tx_flag).await;
        }.instrument(span));
    }

    // Start TCP accept loop
    let config_clone = config.clone();
    let span = tracing::info_span!("agent", name = %agent_name);
    tokio::spawn(async move {
        run_tcp_accept_loop(tcp_listener, config_clone).await;
    }.instrument(span));

    // Main event loop: handle discovery events
    let negotiator = ClaudeNegotiator::new(&config);

    tracing::info!("Agent is running. Waiting for peer discoveries...");

    loop {
        match event_rx.recv().await {
            Ok(event) => match event {
                DiscoveryEvent::PeerDiscovered(peer) => {
                    // Reset announce back-off so we respond quickly
                    peer_found.store(true, Ordering::Release);

                    tracing::info!(
                        peer_id = %peer.agent_id,
                        address = %peer.address,
                        "New peer discovered! Starting negotiation..."
                    );

                    // Negotiate with the peer
                    if !config.claude_api_key.is_empty() && peer.capabilities.has_claude() {
                        let mut mcp_guard = if let Some(ref mcp) = mcp_client {
                            Some(mcp.lock().await)
                        } else {
                            None
                        };

                        let mcp_ref = mcp_guard.as_deref_mut();

                        match negotiator
                            .negotiate(
                                &agent_id.to_string(),
                                &agent_address,
                                &peer,
                                mcp_ref,
                            )
                            .await
                        {
                            Ok(result) => {
                                tracing::info!(
                                    channel = %result.chosen_channel,
                                    reasoning = %result.reasoning,
                                    "Negotiation complete"
                                );

                                // Establish the chosen channel
                                match result.chosen_channel.to_lowercase().as_str() {
                                    "tcp" => {
                                        if let Ok(_channel) = TcpChannel::connect(&peer.address) {
                                            tracing::info!(
                                                "TCP channel established to {}",
                                                peer.address
                                            );
                                            // Channel is ready for use
                                        }
                                    }
                                    "obp" => {
                                        tracing::info!(
                                            "OBP shared storage channel configured"
                                        );
                                    }
                                    other => {
                                        tracing::warn!(
                                            "Unknown channel type: {}",
                                            other
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("Negotiation failed: {}", e);
                            }
                        }
                    } else {
                        tracing::info!(
                            "Skipping negotiation (no Claude API key or peer doesn't support it)"
                        );
                    }
                }
                DiscoveryEvent::PeerUpdated(peer) => {
                    tracing::debug!(peer_id = %peer.agent_id, "Peer updated");
                }
                DiscoveryEvent::PeerLost(id) => {
                    tracing::info!(peer_id = %id, "Peer lost");
                }
            },
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("Event receiver lagged by {} messages", n);
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                tracing::info!("Event channel closed, shutting down");
                break;
            }
        }
    }

    Ok(())
}

/// Simple xorshift64 PRNG. Not cryptographic, but gives genuinely
/// different values on each call (unlike reading subsec_nanos in a loop).
struct Rng(u64);

impl Rng {
    fn from_entropy() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        // Ensure non-zero seed
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Returns a value in [0, max).
    fn range(&mut self, max: u64) -> u64 {
        self.next() % max
    }
}

/// Back-off parameters for the announce loop.
const INITIAL_INTERVAL_MS: u64 = 2000;
const MAX_INTERVAL_MS: u64 = 30000;
const BACKOFF_MULTIPLIER: f64 = 1.5;

/// Run the FSK announce loop with exponential back-off.
async fn run_announce_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
    is_transmitting: Arc<AtomicBool>,
    peer_found: Arc<AtomicBool>,
) {
    let mut rng = Rng::from_entropy();

    // Initial random delay so agents started at the same time don't collide
    let startup_delay = rng.range(3000) + 500;
    tracing::info!(
        delay_ms = startup_delay,
        "Announce loop starting after initial delay"
    );
    tokio::time::sleep(std::time::Duration::from_millis(startup_delay)).await;

    let mut announce_count: u64 = 0;
    let mut current_interval_ms: u64 = INITIAL_INTERVAL_MS;

    loop {
        announce_count += 1;

        // Check if a peer was found - reset back-off
        if peer_found.swap(false, Ordering::AcqRel) {
            current_interval_ms = INITIAL_INTERVAL_MS;
            tracing::info!("TX: Peer found, resetting announce interval");
        }

        let (samples, duration_secs) = {
            let mgr = manager.lock().await;
            let msg = mgr.build_announce();
            let samples = mgr.encode_to_samples(&msg);
            let duration = samples.len() as f32 / crate::audio::modulator::SAMPLE_RATE as f32;
            (samples, duration)
        };

        tracing::info!(
            announce = announce_count,
            duration_ms = format!("{:.0}", duration_secs * 1000.0),
            next_interval_s = format!("{:.1}", current_interval_ms as f64 / 1000.0),
            "TX: Broadcasting announce tone"
        );

        // Mute RX while transmitting so the mic doesn't pick up our own signal
        is_transmitting.store(true, Ordering::Release);

        if let Err(e) = engine.send_samples(samples) {
            tracing::error!("TX: Failed to send announce: {}", e);
        }

        // Wait for the tone to finish playing, plus a small margin for echo decay
        let tx_duration_ms = (duration_secs * 1000.0) as u64 + 200;
        tokio::time::sleep(std::time::Duration::from_millis(tx_duration_ms)).await;

        is_transmitting.store(false, Ordering::Release);

        // Add jitter: +/- 25% of current interval
        let jitter_range = current_interval_ms / 4;
        let jitter = if jitter_range > 0 { rng.range(jitter_range * 2) } else { 0 };
        let sleep_ms = current_interval_ms - jitter_range + jitter;

        tracing::debug!(
            sleep_ms,
            "TX: Sleeping before next announce"
        );
        tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;

        // Back off: increase interval for next round
        current_interval_ms = ((current_interval_ms as f64 * BACKOFF_MULTIPLIER) as u64)
            .min(MAX_INTERVAL_MS);
    }
}

/// Run the FSK receive loop.
async fn run_receive_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
    is_transmitting: Arc<AtomicBool>,
) {
    let mut sample_buffer = Vec::new();
    let mut peak: f32 = 0.0;
    let mut last_peak_log = std::time::Instant::now();
    let mut decode_attempts: u64 = 0;

    tracing::info!("RX: Listening for FSK tones on all 9 bands...");

    loop {
        // If we're currently transmitting, drain and discard mic samples
        // to avoid decoding our own signal
        if is_transmitting.load(Ordering::Acquire) {
            while engine.try_recv_samples().is_some() {}
            sample_buffer.clear();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            continue;
        }

        // Collect samples and track peak level
        while let Some(chunk) = engine.try_recv_samples() {
            for &s in &chunk {
                let abs = s.abs();
                if abs > peak {
                    peak = abs;
                }
            }
            sample_buffer.extend_from_slice(&chunk);
        }

        // Log loudest sample heard in the last 10 seconds
        if last_peak_log.elapsed() > std::time::Duration::from_secs(10) {
            tracing::info!(
                peak = format!("{:.4}", peak),
                buffer = sample_buffer.len(),
                decodes = decode_attempts,
                "RX: Loudest sample in last 10s (0=silence, >0.01=sound, >0.1=loud)"
            );
            peak = 0.0;
            last_peak_log = std::time::Instant::now();
        }

        // Try to decode when we have enough samples for a frame.
        // A typical frame is ~35 bytes * 8 bits * 160 samples/bit ~ 44800 samples.
        // We use a sliding window: on failed decode, drop the oldest chunk
        // so we don't keep retrying the same corrupted data.
        let frame_samples = 50000usize;
        if sample_buffer.len() > frame_samples {
            decode_attempts += 1;
            let mut mgr = manager.lock().await;
            if let Some(msg) = mgr.decode_from_samples(&sample_buffer) {
                tracing::info!(
                    peer_id = %msg.agent_id,
                    msg_type = ?msg.msg_type,
                    address = %msg.address,
                    "RX: Decoded FSK frame!"
                );
                mgr.handle_message(&msg);
                sample_buffer.clear();
            } else {
                // Failed decode - drop the oldest frame's worth of samples
                // so we advance to fresh data instead of retrying the same noise.
                let drop = frame_samples.min(sample_buffer.len());
                sample_buffer.drain(..drop);
            }
        }

        // Expire old peers periodically
        {
            let mut mgr = manager.lock().await;
            mgr.expire_peers();
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// Run the TCP connection accept loop.
async fn run_tcp_accept_loop(listener: TcpCommListener, _config: Config) {
    let listener = Arc::new(listener);
    loop {
        let listener = listener.clone();
        match tokio::task::spawn_blocking(move || listener.accept()).await {
            Ok(Ok(channel)) => {
                tracing::info!("Accepted TCP connection: {}", channel.description());
                tokio::spawn(async move {
                    handle_tcp_connection(channel).await;
                });
            }
            Ok(Err(e)) => {
                tracing::error!("TCP accept error: {}", e);
            }
            Err(e) => {
                tracing::error!("TCP accept task error: {}", e);
                break;
            }
        }
    }
}

async fn handle_tcp_connection(channel: TcpChannel) {
    tracing::info!("Handling TCP connection: {}", channel.description());

    // Simple echo for now - a real implementation would route messages
    loop {
        match channel.recv_message() {
            Ok(data) => {
                tracing::info!(
                    "Received {} bytes via TCP",
                    data.len()
                );
                if let Err(e) = channel.send_message(&data) {
                    tracing::error!("TCP send error: {}", e);
                    break;
                }
            }
            Err(e) => {
                tracing::debug!("TCP connection closed: {}", e);
                break;
            }
        }
    }
}
