/// Agent orchestrator: ties audio, discovery, negotiation, and OBP together.

use std::sync::Arc;

use anyhow::Result;
use uuid::Uuid;

use crate::audio::device::AudioEngine;
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
    let agent_address = format!("127.0.0.1:{}", config.agent_listen_port);

    tracing::info!(
        agent_id = %agent_id,
        agent_name = %config.agent_name,
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

    // Set up discovery
    let discovery_config = DiscoveryConfig {
        agent_id,
        agent_address: agent_address.clone(),
        capabilities,
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

    // Start the announce loop (audio TX)
    if let Some(ref engine) = audio_engine {
        let mgr = discovery_manager.clone();
        let eng = engine.clone();
        tokio::spawn(async move {
            run_announce_loop(mgr, eng).await;
        });
    }

    // Start the receive loop (audio RX)
    if let Some(ref engine) = audio_engine {
        let mgr = discovery_manager.clone();
        let eng = engine.clone();
        tokio::spawn(async move {
            run_receive_loop(mgr, eng).await;
        });
    }

    // Start TCP accept loop
    let config_clone = config.clone();
    tokio::spawn(async move {
        run_tcp_accept_loop(tcp_listener, config_clone).await;
    });

    // Main event loop: handle discovery events
    let negotiator = ClaudeNegotiator::new(&config);

    tracing::info!("Agent is running. Waiting for peer discoveries...");

    loop {
        match event_rx.recv().await {
            Ok(event) => match event {
                DiscoveryEvent::PeerDiscovered(peer) => {
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
                                match result.chosen_channel.as_str() {
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

/// Simple pseudo-random number using system time entropy.
/// Returns a value in [0, max).
fn pseudo_random(max: u64) -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    // Mix bits for better distribution
    let mixed = nanos.wrapping_mul(6364136223846793005).wrapping_add(1);
    mixed % max
}

/// Run the FSK announce loop.
async fn run_announce_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
) {
    // Initial random delay so agents started at the same time don't collide
    let startup_delay = pseudo_random(3000) + 500;
    tracing::info!(
        delay_ms = startup_delay,
        "Announce loop starting after initial delay"
    );
    tokio::time::sleep(std::time::Duration::from_millis(startup_delay)).await;

    let mut announce_count: u64 = 0;

    loop {
        announce_count += 1;

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
            "TX: Broadcasting announce tone"
        );

        if let Err(e) = engine.send_samples(samples) {
            tracing::error!("TX: Failed to send announce: {}", e);
        }

        // Irregular interval: 3-8 seconds, heavily randomised to avoid collision
        let base_ms = 3000u64;
        let jitter_ms = pseudo_random(5000);
        let sleep_ms = base_ms + jitter_ms;

        tracing::debug!(
            sleep_ms,
            "TX: Sleeping before next announce"
        );
        tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;
    }
}

/// Run the FSK receive loop.
async fn run_receive_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
) {
    let mut sample_buffer = Vec::new();
    let mut listen_chunks: u64 = 0;

    tracing::info!("RX: Listening for FSK tones...");

    loop {
        // Collect samples
        let mut new_samples = 0usize;
        while let Some(chunk) = engine.try_recv_samples() {
            new_samples += chunk.len();
            sample_buffer.extend_from_slice(&chunk);
        }

        if new_samples > 0 {
            listen_chunks += 1;
            if listen_chunks % 50 == 0 {
                tracing::debug!(
                    buffer_samples = sample_buffer.len(),
                    buffer_secs = format!("{:.1}", sample_buffer.len() as f32 / crate::audio::modulator::SAMPLE_RATE as f32),
                    "RX: Listening... (buffered audio)"
                );
            }
        }

        // Try to decode when we have enough samples
        // A typical frame is ~35 bytes * 8 bits * 160 samples/bit ~ 44800 samples
        if sample_buffer.len() > 10000 {
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
            } else if sample_buffer.len() > 200000 {
                // Prevent unbounded growth, keep recent data
                let drain = sample_buffer.len() - 100000;
                tracing::debug!(
                    trimming_samples = drain,
                    "RX: Trimming old samples from buffer"
                );
                sample_buffer.drain(..drain);
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
