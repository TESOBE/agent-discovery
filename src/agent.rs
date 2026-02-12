/// Agent orchestrator: ties audio, discovery, negotiation, and OBP together.

use std::sync::Arc;

use anyhow::Result;
use uuid::Uuid;

use crate::audio::device::AudioEngine;
use crate::comms::channel::CommChannel;
use crate::comms::tcp::{TcpChannel, TcpCommListener};
use crate::config::Config;
use crate::discovery::manager::{DiscoveryConfig, DiscoveryEvent, DiscoveryManager};
use crate::mcp::client::StdioMcpClient;
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
    let mcp_client = match StdioMcpClient::new(&config).await {
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

/// Run the FSK announce loop.
async fn run_announce_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
) {
    loop {
        let samples = {
            let mgr = manager.lock().await;
            let msg = mgr.build_announce();
            mgr.encode_to_samples(&msg)
        };

        if let Err(e) = engine.send_samples(samples) {
            tracing::error!("Failed to send announce: {}", e);
        }

        // 4-6 second interval with jitter
        let jitter_ms = {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64;
            nanos % 2000
        };
        tokio::time::sleep(std::time::Duration::from_millis(4000 + jitter_ms)).await;
    }
}

/// Run the FSK receive loop.
async fn run_receive_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
) {
    let mut sample_buffer = Vec::new();

    loop {
        // Collect samples
        while let Some(chunk) = engine.try_recv_samples() {
            sample_buffer.extend_from_slice(&chunk);
        }

        // Try to decode when we have enough samples
        // A typical frame is ~35 bytes * 8 bits * 160 samples/bit ≈ 44800 samples
        if sample_buffer.len() > 10000 {
            let mut mgr = manager.lock().await;
            if let Some(msg) = mgr.decode_from_samples(&sample_buffer) {
                mgr.handle_message(&msg);
                sample_buffer.clear();
            } else if sample_buffer.len() > 200000 {
                // Prevent unbounded growth, keep recent data
                let drain = sample_buffer.len() - 100000;
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
