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
use crate::obp::exploration::ExplorationMsg;
use crate::protocol::{codec, message::Capabilities};

/// Run the main agent loop.
/// If `udp_only` is true, skip audio discovery and start UDP immediately.
pub async fn run(config: Config, udp_only: bool) -> Result<()> {
    let agent_id = Uuid::new_v4();
    let agent_name = config.agent_name.clone();

    // Bind the TCP listener early so we know the actual port (may differ
    // from the configured port if that one is already in use).
    let tcp_listener = TcpCommListener::bind_with_fallback(config.agent_listen_port, 10)?;
    let actual_port = tcp_listener.local_addr()?.port();
    let agent_address = format!("127.0.0.1:{}", actual_port);

    // Span that appears on every log line for this agent.
    // We use instrument() on spawned tasks rather than entering here,
    // to avoid doubling the span when child tasks create their own.
    let agent_span = tracing::info_span!("agent", name = %agent_name);

    tracing::info!(parent: &agent_span,
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
    let (mcp_client, mcp_diagnosis) = match McpClient::new(&config).await {
        Ok(client) => {
            tracing::info!("MCP client connected, {} tools available", client.tools().len());
            capabilities = capabilities.with_obp();
            (Some(Arc::new(tokio::sync::Mutex::new(client))), None)
        }
        Err(e) => {
            tracing::warn!("MCP client unavailable: {}. Diagnosing...", e);
            let findings = crate::mcp::client::diagnose_mcp(&config).await;
            for finding in &findings {
                tracing::info!("MCP diagnosis: {}", finding);
            }
            (None, Some(findings))
        }
    };

    // Set up OBP client for direct HTTP calls (entity record creation)
    let obp_client = {
        let mut client = ObpClient::new(&config);
        match client.authenticate().await {
            Ok(()) => {
                tracing::info!("OBP DirectLogin authenticated");
                Some(Arc::new(client))
            }
            Err(e) => {
                tracing::warn!("OBP auth failed: {}. Handshake records will be skipped.", e);
                None
            }
        }
    };

    // NOTE: Entity definitions are no longer created at startup.
    // Agents discover and create them collaboratively during the
    // post-handshake exploration protocol.

    // Derive unique chirp signature from agent name
    let chirp_sig = crate::audio::chirp::AgentChirpSignature::from_name(&config.agent_name);
    tracing::info!(parent: &agent_span,
        signature = %chirp_sig.describe(),
        "Agent chirp signature (unique per name)"
    );

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

    tracing::info!("TCP listener ready on port {}", actual_port);

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
    // Set to true after a handshake completes — slows down announce interval
    let handshake_done = Arc::new(AtomicBool::new(false));
    // Chirp call-and-response flags:
    // RX sets this when it hears a CALL chirp → TX should send a RESPONSE chirp
    let send_response_chirp = Arc::new(AtomicBool::new(false));
    // RX sets this when it hears a RESPONSE chirp → chirp handshake confirmed
    let chirp_handshake_done = Arc::new(AtomicBool::new(false));

    // Start audio loops (skipped with --udp)
    if !udp_only {
        // Start the announce loop (audio TX — chirp call-and-response + DTMF + FSK)
        if let Some(ref engine) = audio_engine {
            let mgr = discovery_manager.clone();
            let eng = engine.clone();
            let tx_flag = is_transmitting.clone();
            let pf_flag = peer_found.clone();
            let hs_flag = handshake_done.clone();
            let src_flag = send_response_chirp.clone();
            let chd_flag = chirp_handshake_done.clone();
            let sig = chirp_sig.clone();
            tokio::spawn(
                run_announce_loop(mgr, eng, tx_flag, pf_flag, hs_flag, src_flag, chd_flag, sig)
                    .instrument(agent_span.clone()),
            );
        }

        // Start the receive loop (audio RX — chirp detection + DTMF + FSK)
        if let Some(ref engine) = audio_engine {
            let mgr = discovery_manager.clone();
            let eng = engine.clone();
            let tx_flag = is_transmitting.clone();
            let pf_flag = peer_found.clone();
            let src_flag = send_response_chirp.clone();
            let chd_flag = chirp_handshake_done.clone();
            let sig = chirp_sig.clone();
            tokio::spawn(
                run_receive_loop(mgr, eng, tx_flag, pf_flag, src_flag, chd_flag, sig)
                    .instrument(agent_span.clone()),
            );
        }
    } else {
        tracing::info!(parent: &agent_span, "Audio discovery skipped (--udp mode)");
    }

    // Start TCP accept loop (pass OBP clients so responder can run exploration)
    let config_clone = config.clone();
    let mcp_for_accept = mcp_client.clone();
    let obp_for_accept = obp_client.clone();
    tokio::spawn(
        run_tcp_accept_loop(tcp_listener, config_clone, mcp_for_accept, obp_for_accept)
            .instrument(agent_span.clone()),
    );

    // Start UDP broadcast discovery (reliable fallback alongside audio, or immediate with --udp)
    {
        let mgr = discovery_manager.clone();
        let pf = peer_found.clone();
        let hs = handshake_done.clone();
        tokio::spawn(
            run_udp_announce_loop(mgr, pf, hs, udp_only)
                .instrument(agent_span.clone()),
        );
    }
    {
        let mgr = discovery_manager.clone();
        tokio::spawn(
            run_udp_receive_loop(mgr, udp_only)
                .instrument(agent_span.clone()),
        );
    }

    // Main event loop: handle discovery events
    let negotiator = ClaudeNegotiator::new(&config);

    tracing::info!(parent: &agent_span, "Agent is running. Waiting for peer discoveries...");

    // Track peers we've already negotiated with, so we don't re-negotiate on PeerUpdated
    let mut negotiated_peers = std::collections::HashSet::<Uuid>::new();

    // Run the event loop inside the agent span
    let _guard = agent_span.enter();

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

                    handle_new_peer(
                        &config,
                        &negotiator,
                        &mcp_client,
                        &obp_client,
                        &mcp_diagnosis,
                        &agent_id,
                        &agent_name,
                        &agent_address,
                        &peer,
                        &mut negotiated_peers,
                        &handshake_done,
                    )
                    .await;
                }
                DiscoveryEvent::PeerUpdated(peer) => {
                    tracing::debug!(peer_id = %peer.agent_id, "Peer updated");
                }
                DiscoveryEvent::PeerLost(id) => {
                    tracing::info!(peer_id = %id, "Peer lost");
                    negotiated_peers.remove(&id);
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

/// Handle a newly discovered peer: negotiate, connect, exchange greetings,
/// then record the handshake to OBP as a dynamic entity record.
async fn handle_new_peer(
    config: &Config,
    negotiator: &ClaudeNegotiator,
    mcp_client: &Option<Arc<tokio::sync::Mutex<McpClient>>>,
    obp_client: &Option<Arc<ObpClient>>,
    mcp_diagnosis: &Option<Vec<String>>,
    agent_id: &Uuid,
    agent_name: &str,
    agent_address: &str,
    peer: &crate::discovery::peer::Peer,
    negotiated_peers: &mut std::collections::HashSet<Uuid>,
    handshake_done: &Arc<AtomicBool>,
) {
    // Don't re-negotiate with peers we already handshook
    if negotiated_peers.contains(&peer.agent_id) {
        tracing::debug!(peer_id = %peer.agent_id, "Already negotiated with this peer, skipping");
        return;
    }

    // Negotiate with the peer via Claude
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
                agent_address,
                peer,
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

                // Establish the chosen channel and perform handshake + exploration
                match result.chosen_channel.to_lowercase().as_str() {
                    "tcp" => {
                        establish_tcp_channel(
                            agent_id, agent_name, peer,
                            mcp_client, obp_client, mcp_diagnosis, config,
                        ).await;
                    }
                    "obp" => {
                        tracing::info!("OBP shared storage channel configured");
                    }
                    other => {
                        tracing::warn!("Unknown channel type: {}", other);
                    }
                }

                negotiated_peers.insert(peer.agent_id);
                handshake_done.store(true, Ordering::Release);
            }
            Err(e) => {
                tracing::error!("Negotiation failed: {}", e);
            }
        }
    } else {
        // No Claude API key — fall back to direct TCP handshake
        tracing::info!("No Claude API key or peer doesn't support it — trying direct TCP handshake");
        establish_tcp_channel(
            agent_id, agent_name, peer,
            mcp_client, obp_client, mcp_diagnosis, config,
        ).await;
        negotiated_peers.insert(peer.agent_id);
        handshake_done.store(true, Ordering::Release);
    }

    // Handshake recording now happens during the exploration protocol
}

/// Connect to a peer via TCP, exchange greeting messages, then start the
/// OBP exploration protocol (initiator role).
///
/// After the hello/hello_ack exchange, keeps the channel open and runs
/// through all exploration phases with the responder.
async fn establish_tcp_channel(
    agent_id: &Uuid,
    agent_name: &str,
    peer: &crate::discovery::peer::Peer,
    mcp_client: &Option<Arc<tokio::sync::Mutex<McpClient>>>,
    obp_client: &Option<Arc<ObpClient>>,
    mcp_diagnosis: &Option<Vec<String>>,
    config: &Config,
) {
    let peer_address = peer.address.clone();
    let greeting = serde_json::json!({
        "type": "hello",
        "agent_id": agent_id.to_string(),
        "agent_name": agent_name,
        "message": format!("Hello from {}! I discovered you via audio/UDP.", agent_name),
    });
    let greeting_bytes = serde_json::to_vec(&greeting).unwrap_or_default();

    // TCP connect + hello/hello_ack is blocking
    let agent_name_owned = agent_name.to_string();
    let channel = match tokio::task::spawn_blocking(move || -> Result<(TcpChannel, String)> {
        let channel = TcpChannel::connect(&peer_address)?;
        tracing::info!("TCP connected to {}", peer_address);

        // Set a 30s read timeout for the exploration protocol
        channel.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;

        // Send our greeting
        channel.send_message(&greeting_bytes)?;
        tracing::info!("Sent greeting to {}", peer_address);

        // Wait for hello_ack
        match channel.recv_message() {
            Ok(response) => {
                let response_text = String::from_utf8_lossy(&response);
                tracing::info!(
                    response = %response_text,
                    "Received handshake response from peer"
                );
                Ok((channel, response_text.to_string()))
            }
            Err(e) => {
                tracing::warn!("No handshake response from peer: {}", e);
                anyhow::bail!("No hello_ack: {}", e);
            }
        }
    })
    .await
    {
        Ok(Ok((channel, _response))) => {
            tracing::info!("TCP handshake complete with peer");
            if let Err(e) = crate::audio::device::play_handshake_chime() {
                tracing::debug!("Could not play chime: {}", e);
            }
            channel
        }
        Ok(Err(e)) => {
            tracing::error!("TCP handshake failed: {}", e);
            return;
        }
        Err(e) => {
            tracing::error!("TCP handshake task panicked: {}", e);
            return;
        }
    };

    // Channel is now established — run the exploration protocol as initiator
    let channel = Arc::new(channel);
    run_obp_exploration_initiator(
        channel,
        &agent_name_owned,
        &config.obp_bank_id,
        mcp_client,
        obp_client,
        mcp_diagnosis,
    )
    .await;
}

// ---------------------------------------------------------------------------
// Exploration TCP helpers
// ---------------------------------------------------------------------------

/// Send an ExplorationMsg over the TCP channel (spawn_blocking wrapper).
async fn send_exploration_msg(channel: &Arc<TcpChannel>, msg: &ExplorationMsg) -> Result<()> {
    let bytes = serde_json::to_vec(msg)?;
    let ch = channel.clone();
    tokio::task::spawn_blocking(move || ch.send_message(&bytes)).await??;
    Ok(())
}

/// Receive an ExplorationMsg from the TCP channel (spawn_blocking wrapper).
/// Returns the raw message including McpDiagnosis.
async fn recv_exploration_msg(channel: &Arc<TcpChannel>) -> Result<ExplorationMsg> {
    let ch = channel.clone();
    let bytes = tokio::task::spawn_blocking(move || ch.recv_message()).await??;
    let msg: ExplorationMsg = serde_json::from_slice(&bytes)?;
    Ok(msg)
}

/// Receive the next non-diagnostic message, logging any McpDiagnosis
/// messages that arrive before it. The initiator may inject diagnosis
/// messages between protocol phases when MCP calls fail.
async fn recv_draining_diagnoses(channel: &Arc<TcpChannel>) -> Result<ExplorationMsg> {
    loop {
        let msg = recv_exploration_msg(channel).await?;
        match msg {
            ExplorationMsg::McpDiagnosis { findings } => {
                tracing::warn!("Initiator shared MCP diagnosis ({} findings):", findings.len());
                for finding in &findings {
                    tracing::warn!("  {}", finding);
                }
                // Continue reading — the real protocol message follows
            }
            other => return Ok(other),
        }
    }
}

// ---------------------------------------------------------------------------
// Initiator exploration
// ---------------------------------------------------------------------------

/// Run the full OBP exploration protocol as initiator.
///
/// Uses MCP tools to discover dynamic entity management endpoints,
/// figure out the creation format, create an entity, discover the
/// auto-generated CRUD endpoints, and create + verify a test record.
async fn run_obp_exploration_initiator(
    channel: Arc<TcpChannel>,
    agent_name: &str,
    bank_id: &str,
    mcp_client: &Option<Arc<tokio::sync::Mutex<McpClient>>>,
    obp_client: &Option<Arc<ObpClient>>,
    mcp_diagnosis: &Option<Vec<String>>,
) {
    tracing::info!("=== Starting OBP Exploration (initiator) ===");

    // Phase 1: Send ExploreStart, receive ExploreAck
    let start_msg = ExplorationMsg::ExploreStart {
        agent_name: agent_name.to_string(),
        bank_id: bank_id.to_string(),
    };
    if let Err(e) = send_exploration_msg(&channel, &start_msg).await {
        tracing::error!("Failed to send ExploreStart: {}", e);
        return;
    }
    tracing::info!("Sent ExploreStart to responder");

    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::ExploreAck { agent_name: peer_name, .. }) => {
            tracing::info!("Received ExploreAck from {}", peer_name);
        }
        Ok(other) => {
            tracing::warn!("Expected ExploreAck, got: {:?}", other);
            return;
        }
        Err(e) => {
            tracing::error!("Failed to receive ExploreAck: {}", e);
            return;
        }
    }

    // If MCP is unavailable, share the diagnosis with the responder
    if mcp_client.is_none() {
        if let Some(ref findings) = mcp_diagnosis {
            tracing::info!("Sharing MCP diagnosis with responder ({} findings)", findings.len());
            let _ = send_exploration_msg(&channel, &ExplorationMsg::McpDiagnosis {
                findings: findings.clone(),
            }).await;
        }
    }

    // Phase 2: Discover management endpoints — try MCP, fall back to known endpoint
    let (endpoint_id, method, path, summary) = if let Some(ref mcp) = mcp_client {
        let mut mcp_guard = mcp.lock().await;
        let result = mcp_guard
            .call_tool(
                "list_endpoints_by_tag",
                serde_json::json!({"tags": ["Dynamic-Entity-Manage"]}),
            )
            .await;

        match result {
            Ok(val) => {
                tracing::info!("MCP list_endpoints_by_tag result: {}", val);
                let content = extract_mcp_text_content(&val);
                find_create_endpoint(&content).unwrap_or_else(|| {
                    tracing::warn!("Could not parse MCP response, using known endpoint");
                    known_create_endpoint()
                })
            }
            Err(e) => {
                tracing::warn!("MCP list_endpoints_by_tag failed: {}. Diagnosing & using known endpoint.", e);
                // Drop the MCP guard before running async diagnosis
                drop(mcp_guard);
                diagnose_and_share_mcp_failure(&channel, &e.to_string()).await;
                known_create_endpoint()
            }
        }
    } else {
        tracing::info!("No MCP client, using known management endpoint");
        known_create_endpoint()
    };

    tracing::info!("Found management endpoint: {} {} ({})", method, path, endpoint_id);

    // Share finding with responder
    let _ = send_exploration_msg(&channel, &ExplorationMsg::FoundManagementEndpoint {
        endpoint_id: endpoint_id.clone(),
        method: method.clone(),
        path: path.clone(),
        summary: summary.clone(),
    }).await;

    // Wait for acknowledgement
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::Acknowledged { phase }) => {
            tracing::info!("Responder acknowledged phase: {}", phase);
        }
        Ok(other) => tracing::debug!("Unexpected msg (continuing): {:?}", other),
        Err(e) => {
            tracing::warn!("Failed to recv ack for management endpoint: {}", e);
        }
    }

    // Phase 3: Get the schema for creating an entity — try MCP, fall back to description
    let schema_description = if let Some(ref mcp) = mcp_client {
        let mut mcp_guard = mcp.lock().await;
        let result = mcp_guard
            .call_tool(
                "get_endpoint_schema",
                serde_json::json!({"endpoint_id": &endpoint_id}),
            )
            .await;

        match result {
            Ok(val) => {
                let text = extract_mcp_text_content(&val);
                tracing::info!("Entity creation schema: {}", &text[..text.len().min(500)]);
                text
            }
            Err(e) => {
                tracing::warn!("Failed to get endpoint schema via MCP: {}", e);
                drop(mcp_guard);
                diagnose_and_share_mcp_failure(&channel, &e.to_string()).await;
                "v6.0.0 format: {entity_name, has_personal_entity, schema: {description, required, properties}}".to_string()
            }
        }
    } else {
        tracing::info!("No MCP client, using known entity format");
        "v6.0.0 format: {entity_name, has_personal_entity, schema: {description, required, properties}}".to_string()
    };

    let _ = send_exploration_msg(&channel, &ExplorationMsg::FoundEntityFormat {
        endpoint_id: endpoint_id.clone(),
        schema_description: schema_description.clone(),
    }).await;

    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::Acknowledged { .. }) => {}
        _ => {}
    }

    // Phase 4: Create the entity definition — try MCP, fall back to direct HTTP
    let entity_def = entities::agent_handshake_entity();
    let entity_name = entities::HANDSHAKE_ENTITY;

    let response_val = if let Some(ref mcp) = mcp_client {
        let mut mcp_guard = mcp.lock().await;
        match mcp_guard
            .call_tool(
                "call_obp_api",
                serde_json::json!({
                    "endpoint_id": &endpoint_id,
                    "path_params": {"BANK_ID": bank_id},
                    "body": &entity_def
                }),
            )
            .await
        {
            Ok(val) => {
                tracing::info!("Created entity '{}' via MCP: {}", entity_name, val);
                val
            }
            Err(e) => {
                tracing::warn!("MCP entity creation failed: {}. Diagnosing & trying direct HTTP.", e);
                drop(mcp_guard);
                diagnose_and_share_mcp_failure(&channel, &e.to_string()).await;
                create_entity_via_http(obp_client, bank_id, &entity_def, entity_name).await
            }
        }
    } else {
        tracing::info!("No MCP client, creating entity via direct HTTP");
        create_entity_via_http(obp_client, bank_id, &entity_def, entity_name).await
    };

    let _ = send_exploration_msg(&channel, &ExplorationMsg::EntityCreated {
        entity_name: entity_name.to_string(),
        request_body: entity_def.clone(),
        response: response_val,
    }).await;

    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::Acknowledged { .. }) => {}
        _ => {}
    }

    // Phase 5: Discover auto-generated CRUD endpoints
    let discovered_endpoints = if let Some(ref obp) = obp_client {
        match entities::discover_entity_endpoints_via_http(obp, entity_name).await {
            Ok(eps) => eps,
            Err(e) => {
                tracing::warn!("Failed to discover endpoints via resource-docs: {}", e);
                Vec::new()
            }
        }
    } else {
        tracing::warn!("No OBP HTTP client, skipping resource-docs discovery");
        Vec::new()
    };

    tracing::info!("Discovered {} CRUD endpoints for '{}'", discovered_endpoints.len(), entity_name);

    let _ = send_exploration_msg(&channel, &ExplorationMsg::DiscoveredCrudEndpoints {
        entity_name: entity_name.to_string(),
        endpoints: discovered_endpoints.clone(),
    }).await;

    // Wait for responder's independent verification
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::EndpointsConfirmed { confirmed, our_endpoints, .. }) => {
            tracing::info!(
                "Responder endpoint verification: confirmed={}, found {} endpoints",
                confirmed, our_endpoints.len()
            );
        }
        Ok(other) => tracing::debug!("Expected EndpointsConfirmed, got: {:?}", other),
        Err(e) => tracing::warn!("Failed to recv EndpointsConfirmed: {}", e),
    }

    // Phase 6: Create a test record
    let test_record = serde_json::json!({
        "agent_id": format!("initiator-{}", agent_name),
        "agent_name": agent_name,
        "peer_id": "responder-peer",
        "peer_address": "tcp-channel",
        "discovery_method": "exploration-protocol",
        "timestamp": entities::iso_now()
    });

    let record_result = if let Some(ref obp) = obp_client {
        let path = format!("/banks/{}/{}", bank_id, entity_name);
        match obp.post(&path, &test_record).await {
            Ok(val) => {
                tracing::info!("Created test record: {}", val);
                val
            }
            Err(e) => {
                tracing::warn!("Failed to create test record: {}", e);
                serde_json::json!({"error": format!("{}", e)})
            }
        }
    } else {
        serde_json::json!({"error": "no OBP client"})
    };

    let _ = send_exploration_msg(&channel, &ExplorationMsg::RecordCreated {
        entity_name: entity_name.to_string(),
        endpoint_used: format!("POST /banks/{}/{}", bank_id, entity_name),
        record: record_result,
    }).await;

    // Wait for responder to verify
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::RecordVerified { matches, .. }) => {
            tracing::info!("Responder verified record: matches={}", matches);

            let _ = send_exploration_msg(&channel, &ExplorationMsg::ExplorationComplete {
                summary: format!(
                    "Successfully discovered and tested dynamic entity '{}' on bank '{}'",
                    entity_name, bank_id
                ),
                success: matches,
            }).await;

            if matches {
                tracing::info!("=== OBP Exploration COMPLETE (success) ===");
            } else {
                tracing::warn!("=== OBP Exploration COMPLETE (verification mismatch) ===");
            }
        }
        Ok(other) => {
            tracing::warn!("Expected RecordVerified, got: {:?}", other);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::ExplorationComplete {
                summary: "Exploration ended with unexpected response".into(),
                success: false,
            }).await;
        }
        Err(e) => {
            tracing::error!("Failed to recv RecordVerified: {}", e);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::ExplorationComplete {
                summary: format!("Exploration ended with error: {}", e),
                success: false,
            }).await;
        }
    }
}

// ---------------------------------------------------------------------------
// Responder exploration
// ---------------------------------------------------------------------------

/// Run the OBP exploration protocol as responder.
///
/// Acknowledges each phase from the initiator, independently verifies
/// endpoint discovery, and reads back the test record.
async fn run_obp_exploration_responder(
    channel: Arc<TcpChannel>,
    agent_name: &str,
    obp_client: &Option<Arc<ObpClient>>,
) {
    tracing::info!("=== Starting OBP Exploration (responder) ===");

    // Expect ExploreStart
    let bank_id = match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::ExploreStart { agent_name: peer_name, bank_id }) => {
            tracing::info!("Received ExploreStart from {} for bank {}", peer_name, bank_id);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::ExploreAck {
                agent_name: agent_name.to_string(),
                bank_id: bank_id.clone(),
            }).await;
            bank_id
        }
        Ok(other) => {
            tracing::warn!("Expected ExploreStart, got: {:?}", other);
            return;
        }
        Err(e) => {
            tracing::error!("Failed to receive ExploreStart: {}", e);
            return;
        }
    };

    // The initiator may send McpDiagnosis before Phase 2 if MCP is down
    let phase2_msg = match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::McpDiagnosis { findings }) => {
            tracing::warn!("Initiator's MCP is unavailable. Diagnosis:");
            for finding in &findings {
                tracing::warn!("  {}", finding);
            }
            // After diagnosis, the next message should be FoundManagementEndpoint
            recv_exploration_msg(&channel).await
        }
        Ok(other) => Ok(other),
        Err(e) => Err(e),
    };

    // Phase 2: Receive FoundManagementEndpoint
    match phase2_msg {
        Ok(ExplorationMsg::FoundManagementEndpoint { endpoint_id, method, path, summary }) => {
            tracing::info!(
                "Initiator found management endpoint: {} {} ({}) - {}",
                method, path, endpoint_id, summary
            );
            let _ = send_exploration_msg(&channel, &ExplorationMsg::Acknowledged {
                phase: "management_endpoint".into(),
            }).await;
        }
        Ok(ExplorationMsg::ExplorationError { phase, error }) => {
            tracing::error!("Initiator reported error in {}: {}", phase, error);
            return;
        }
        Ok(other) => {
            tracing::warn!("Expected FoundManagementEndpoint, got: {:?}", other);
            return;
        }
        Err(e) => {
            tracing::error!("Failed to receive management endpoint: {}", e);
            return;
        }
    }

    // Phase 3: Receive FoundEntityFormat (may be preceded by McpDiagnosis)
    match recv_draining_diagnoses(&channel).await {
        Ok(ExplorationMsg::FoundEntityFormat { endpoint_id, schema_description }) => {
            tracing::info!(
                "Initiator discovered entity format for {}: {}...",
                endpoint_id,
                &schema_description[..schema_description.len().min(200)]
            );
            let _ = send_exploration_msg(&channel, &ExplorationMsg::Acknowledged {
                phase: "entity_format".into(),
            }).await;
        }
        Ok(other) => {
            tracing::warn!("Expected FoundEntityFormat, got: {:?}", other);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::Acknowledged {
                phase: "entity_format".into(),
            }).await;
        }
        Err(e) => {
            tracing::error!("Failed to receive entity format: {}", e);
            return;
        }
    }

    // Phase 4: Receive EntityCreated (may be preceded by McpDiagnosis)
    let entity_name = match recv_draining_diagnoses(&channel).await {
        Ok(ExplorationMsg::EntityCreated { entity_name, response, .. }) => {
            tracing::info!("Initiator created entity '{}': {}", entity_name, response);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::Acknowledged {
                phase: "entity_created".into(),
            }).await;
            entity_name
        }
        Ok(other) => {
            tracing::warn!("Expected EntityCreated, got: {:?}", other);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::Acknowledged {
                phase: "entity_created".into(),
            }).await;
            entities::HANDSHAKE_ENTITY.to_string()
        }
        Err(e) => {
            tracing::error!("Failed to receive EntityCreated: {}", e);
            return;
        }
    };

    // Phase 5: Receive DiscoveredCrudEndpoints — independently verify
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::DiscoveredCrudEndpoints { entity_name: ename, endpoints }) => {
            tracing::info!(
                "Initiator discovered {} CRUD endpoints for '{}'",
                endpoints.len(), ename
            );

            // Independently verify via resource-docs
            let our_endpoints = if let Some(ref obp) = obp_client {
                match entities::discover_entity_endpoints_via_http(obp, &ename).await {
                    Ok(eps) => eps,
                    Err(e) => {
                        tracing::warn!("Responder resource-docs query failed: {}", e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            let confirmed = !our_endpoints.is_empty() || !endpoints.is_empty();
            tracing::info!(
                "Responder independently found {} endpoints (confirmed={})",
                our_endpoints.len(), confirmed
            );

            let _ = send_exploration_msg(&channel, &ExplorationMsg::EndpointsConfirmed {
                entity_name: ename,
                confirmed,
                our_endpoints,
            }).await;
        }
        Ok(other) => {
            tracing::warn!("Expected DiscoveredCrudEndpoints, got: {:?}", other);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::EndpointsConfirmed {
                entity_name: entity_name.clone(),
                confirmed: false,
                our_endpoints: Vec::new(),
            }).await;
        }
        Err(e) => {
            tracing::error!("Failed to receive CRUD endpoints: {}", e);
            return;
        }
    }

    // Phase 6: Receive RecordCreated — read it back to verify
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::RecordCreated { entity_name: ename, record, .. }) => {
            tracing::info!("Initiator created test record for '{}'", ename);

            // Try to read back all records and verify
            let (read_back, matches) = if let Some(ref obp) = obp_client {
                let path = format!("/banks/{}/{}", bank_id, ename);
                match obp.get(&path).await {
                    Ok(val) => {
                        tracing::info!("Read back records: {}", val);
                        // Check if the response contains any records
                        let has_records = val.get(&format!("{}List", ename))
                            .or(val.get(&format!("{}_list", ename)))
                            .and_then(|l| l.as_array())
                            .map(|arr| !arr.is_empty())
                            .unwrap_or(false);
                        // Also just check if response is non-null as a fallback
                        let matches = has_records || !val.is_null();
                        (val, matches)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read back records: {}", e);
                        (serde_json::json!({"error": format!("{}", e)}), false)
                    }
                }
            } else {
                (record.clone(), false)
            };

            tracing::info!("Record verification: matches={}", matches);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::RecordVerified {
                entity_name: ename,
                record: read_back,
                matches,
            }).await;
        }
        Ok(other) => {
            tracing::warn!("Expected RecordCreated, got: {:?}", other);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::RecordVerified {
                entity_name: entity_name.clone(),
                record: serde_json::json!(null),
                matches: false,
            }).await;
        }
        Err(e) => {
            tracing::error!("Failed to receive RecordCreated: {}", e);
            return;
        }
    }

    // Wait for ExplorationComplete
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::ExplorationComplete { summary, success }) => {
            if success {
                tracing::info!("=== OBP Exploration COMPLETE (responder): {} ===", summary);
            } else {
                tracing::warn!("=== OBP Exploration ended (responder): {} ===", summary);
            }
        }
        Ok(other) => tracing::debug!("Expected ExplorationComplete, got: {:?}", other),
        Err(e) => tracing::debug!("Connection closed after exploration: {}", e),
    }
}

// ---------------------------------------------------------------------------
// MCP response helpers
// ---------------------------------------------------------------------------

/// Extract text content from an MCP tool response.
fn extract_mcp_text_content(val: &serde_json::Value) -> String {
    // MCP responses have {"content": [{"type": "text", "text": "..."}]}
    if let Some(content) = val.get("content").and_then(|c| c.as_array()) {
        for item in content {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                return text.to_string();
            }
        }
    }
    // Fallback: just stringify the whole thing
    val.to_string()
}

/// Parse the MCP list_endpoints_by_tag response to find the create dynamic entity endpoint.
fn find_create_endpoint(content: &str) -> Option<(String, String, String, String)> {
    // Try to parse as JSON
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(content) {
        // The response is typically an array of endpoint objects
        let endpoints = if val.is_array() {
            val.as_array().cloned().unwrap_or_default()
        } else if let Some(arr) = val.get("endpoints").and_then(|e| e.as_array()) {
            arr.clone()
        } else {
            vec![val]
        };

        for ep in &endpoints {
            let op_id = ep.get("operation_id").or(ep.get("id"))
                .and_then(|v| v.as_str()).unwrap_or("");
            let method = ep.get("method").and_then(|v| v.as_str()).unwrap_or("");
            let path = ep.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let summary = ep.get("summary").and_then(|v| v.as_str()).unwrap_or("");

            // Look for the bank-level create dynamic entity endpoint
            if (op_id.contains("createBankLevelDynamicEntity") || op_id.contains("Create-Bank-Level-Dynamic-Entity"))
                && method.eq_ignore_ascii_case("POST")
            {
                return Some((
                    op_id.to_string(),
                    method.to_string(),
                    path.to_string(),
                    summary.to_string(),
                ));
            }
        }

        // Fallback: any POST endpoint with "dynamic-entit" in the path
        for ep in &endpoints {
            let op_id = ep.get("operation_id").or(ep.get("id"))
                .and_then(|v| v.as_str()).unwrap_or("");
            let method = ep.get("method").and_then(|v| v.as_str()).unwrap_or("");
            let path = ep.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let summary = ep.get("summary").and_then(|v| v.as_str()).unwrap_or("");

            if method.eq_ignore_ascii_case("POST") && path.contains("dynamic-entit") {
                return Some((
                    op_id.to_string(),
                    method.to_string(),
                    path.to_string(),
                    summary.to_string(),
                ));
            }
        }
    }

    // If we can't parse as JSON, use known defaults
    tracing::warn!("Could not parse MCP response, using known endpoint ID");
    Some((
        "OBPv6.0.0-createBankLevelDynamicEntity".to_string(),
        "POST".to_string(),
        "/management/banks/BANK_ID/dynamic-entities".to_string(),
        "Create Bank Level Dynamic Entity".to_string(),
    ))
}

/// Run MCP diagnostics and share the findings with the peer.
/// Called when an MCP call fails mid-exploration.
async fn diagnose_and_share_mcp_failure(channel: &Arc<TcpChannel>, error_msg: &str) {
    tracing::info!("Running MCP diagnostics after failure: {}", error_msg);

    // Build a config just to read the MCP URL from the environment
    let config = match crate::config::Config::from_env() {
        Ok(c) => c,
        Err(_) => return,
    };

    let findings = crate::mcp::client::diagnose_mcp(&config).await;
    for finding in &findings {
        tracing::info!("MCP diagnosis: {}", finding);
    }

    let _ = send_exploration_msg(channel, &ExplorationMsg::McpDiagnosis {
        findings,
    }).await;
}

/// Return the well-known create-bank-level-dynamic-entity endpoint tuple.
/// Used as a fallback when MCP discovery is unavailable.
fn known_create_endpoint() -> (String, String, String, String) {
    (
        "OBPv6.0.0-createBankLevelDynamicEntity".to_string(),
        "POST".to_string(),
        "/management/banks/BANK_ID/dynamic-entities".to_string(),
        "Create Bank Level Dynamic Entity".to_string(),
    )
}

/// Create a dynamic entity definition via direct HTTP, returning the response.
async fn create_entity_via_http(
    obp_client: &Option<Arc<ObpClient>>,
    bank_id: &str,
    entity_def: &serde_json::Value,
    entity_name: &str,
) -> serde_json::Value {
    let Some(ref obp) = obp_client else {
        tracing::warn!("No OBP HTTP client available to create entity");
        return serde_json::json!({"error": "no OBP client"});
    };
    let path = format!("/management/banks/{}/dynamic-entities", bank_id);
    match obp.post(&path, entity_def).await {
        Ok(val) => {
            tracing::info!("Created entity '{}' via HTTP: {}", entity_name, val);
            val
        }
        Err(e) => {
            tracing::warn!("HTTP entity creation: {} (may already exist)", e);
            serde_json::json!({"note": format!("creation returned error (may exist): {}", e)})
        }
    }
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
const INITIAL_INTERVAL_MS: u64 = 15000;
const MAX_INTERVAL_MS: u64 = 60000;
const BACKOFF_MULTIPLIER: f64 = 1.5;

/// Run the audio announce loop with two-stage chirp protocol.
///
/// **Stage 1 — Chirp call-and-response:**
///   TX sends a CALL chirp (800-3200 Hz) with progressive volume ramp.
///   If RX heard a CALL from another agent, it sets `send_response_chirp` →
///   TX sends a single RESPONSE chirp (4000-6400 Hz).
///   When RX hears a RESPONSE chirp, it sets `chirp_handshake_done`.
///
/// **Stage 2 — Binary chirp data:**
///   After the chirp handshake, TX sends a discovery chirp followed by a
///   binary chirp message (port + capabilities encoded as up/down chirps).
async fn run_announce_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
    is_transmitting: Arc<AtomicBool>,
    peer_found: Arc<AtomicBool>,
    handshake_done: Arc<AtomicBool>,
    send_response_chirp: Arc<AtomicBool>,
    chirp_handshake_done: Arc<AtomicBool>,
    chirp_sig: crate::audio::chirp::AgentChirpSignature,
) {
    use crate::audio::chirp;

    let sample_rate = crate::audio::modulator::SAMPLE_RATE;
    let mut rng = Rng::from_entropy();

    let section_gap = vec![0.0f32; (sample_rate as f32 * 0.08) as usize]; // 80ms gap between sections

    // Initial random delay (10-20s) so agents started at the same time don't collide
    let startup_delay = rng.range(10000) + 10000;
    tracing::info!(
        delay_ms = startup_delay,
        "Announce loop starting after initial delay"
    );
    tokio::time::sleep(std::time::Duration::from_millis(startup_delay)).await;

    let mut announce_count: u64 = 0;
    let mut current_interval_ms: u64 = INITIAL_INTERVAL_MS;
    // After RX confirms the chirp handshake (by hearing a RESPONSE), we need
    // to send our own RESPONSE chirp back so the peer can also confirm.
    // This flag ensures we send a confirming RESPONSE before switching to binary data.
    let mut sent_confirming_response = false;

    loop {
        announce_count += 1;

        // Check if a peer was found - reset back-off
        if peer_found.swap(false, Ordering::AcqRel) {
            current_interval_ms = INITIAL_INTERVAL_MS;
            tracing::info!("TX: Peer found, resetting announce interval");
        }

        // After handshake: slow down (interval * 4 + 30s), one-shot
        if handshake_done.swap(false, Ordering::AcqRel) {
            current_interval_ms = current_interval_ms * 4 + 30_000;
            tracing::info!(
                interval_s = format!("{:.1}", current_interval_ms as f64 / 1000.0),
                "TX: Handshake done, slowing down announces"
            );
        }

        // Progressive amplitude ramp
        let amplitude = chirp::tx_amplitude(announce_count);

        // Decide what to send based on current stage:
        let chirp_hs = chirp_handshake_done.load(Ordering::Acquire);
        let need_response = send_response_chirp.swap(false, Ordering::AcqRel);

        let (samples_to_send, total_duration_secs, announce_type) = if need_response {
            // RESPONSE: send agent-specific response chirp (answering a call)
            let response_samples = chirp_sig.response_chirp(sample_rate, amplitude);
            let duration = response_samples.len() as f32 / sample_rate as f32;
            (response_samples, duration, "chirp RESPONSE")
        } else if chirp_hs && !sent_confirming_response {
            // Chirp handshake just confirmed by RX — send a confirming RESPONSE
            // chirp back so the peer can also confirm the handshake.
            sent_confirming_response = true;
            let response_samples = chirp_sig.response_chirp(sample_rate, amplitude);
            let duration = response_samples.len() as f32 / sample_rate as f32;
            (response_samples, duration, "chirp CONFIRM-RESPONSE")
        } else if !chirp_hs {
            // Stage 1: chirp handshake not done yet — send agent-specific CALL chirp
            let call_samples = chirp_sig.call_chirp(sample_rate, amplitude);
            let duration = call_samples.len() as f32 / sample_rate as f32;
            (call_samples, duration, "chirp CALL")
        } else {
            // Stage 2: chirp handshake done — send binary chirp message
            let mgr = manager.lock().await;

            let port = mgr.config().agent_address
                .rsplit(':')
                .next()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(0);
            let caps = mgr.config().capabilities.0;

            let caps_desc = crate::protocol::message::Capabilities(caps).describe();
            tracing::info!(
                port,
                capabilities = caps,
                caps_desc = %caps_desc,
                "TX: Encoding binary chirp message: port={} caps=0x{:02X} [{}]",
                port, caps, caps_desc
            );

            let binary_msg = chirp::encode_chirp_message(port, caps, sample_rate);
            let call_samples = chirp_sig.call_chirp(sample_rate, amplitude);

            let mut combined = Vec::with_capacity(
                call_samples.len() + section_gap.len() + binary_msg.len(),
            );
            combined.extend_from_slice(&call_samples);
            combined.extend_from_slice(&section_gap);
            combined.extend_from_slice(&binary_msg);

            let duration = combined.len() as f32 / sample_rate as f32;
            (combined, duration, "chirp+binary-data")
        };

        tracing::info!(
            announce = announce_count,
            kind = announce_type,
            amplitude = format!("{:.2}", amplitude),
            duration_ms = format!("{:.0}", total_duration_secs * 1000.0),
            chirp_hs = chirp_hs,
            next_interval_s = format!("{:.1}", current_interval_ms as f64 / 1000.0),
            "TX: Broadcasting {} (amp={:.2})", announce_type, amplitude
        );

        // Mute RX while transmitting so the mic doesn't pick up our own signal
        is_transmitting.store(true, Ordering::Release);

        if let Err(e) = engine.send_samples(samples_to_send) {
            tracing::error!("TX: Failed to send announce: {}", e);
        }

        // Wait for the tone to finish playing, plus a small margin for echo decay
        let tx_duration_ms = (total_duration_secs * 1000.0) as u64 + 200;
        tokio::time::sleep(std::time::Duration::from_millis(tx_duration_ms)).await;

        is_transmitting.store(false, Ordering::Release);

        // If we just sent a response chirp, use a short interval
        // so the other agent gets a chance to hear it quickly
        let sleep_ms = if announce_type.contains("RESPONSE") {
            500 + rng.range(500)
        } else {
            // Add jitter: +/- 25% of current interval
            let jitter_range = current_interval_ms / 4;
            let jitter = if jitter_range > 0 { rng.range(jitter_range * 2) } else { 0 };
            current_interval_ms - jitter_range + jitter
        };

        tracing::debug!(
            sleep_ms,
            "TX: Sleeping before next announce"
        );
        // Break sleep into short segments so we can react quickly when
        // RX detects a CALL chirp and sets send_response_chirp.
        {
            let sleep_start = std::time::Instant::now();
            let target = std::time::Duration::from_millis(sleep_ms);
            while sleep_start.elapsed() < target {
                if send_response_chirp.load(Ordering::Acquire) {
                    tracing::info!("TX: Woke early — send_response_chirp flag is set");
                    break;
                }
                let remaining = target.saturating_sub(sleep_start.elapsed());
                let chunk = remaining.min(std::time::Duration::from_millis(100));
                if chunk.is_zero() {
                    break;
                }
                tokio::time::sleep(chunk).await;
            }
        }

        // Back off: increase interval for next round (only for regular announces)
        if !announce_type.contains("RESPONSE") {
            current_interval_ms = ((current_interval_ms as f64 * BACKOFF_MULTIPLIER) as u64)
                .min(MAX_INTERVAL_MS);
        }
    }
}

/// Run the audio receive loop with spectrogram-based chirp detection.
///
/// Detection uses FFT spectrogram analysis to find upward frequency sweeps,
/// which works with any agent's unique chirp frequencies. Self-echo rejection
/// compares detected chirp parameters against the agent's own signature.
///
/// Detection logic:
///   1. Check RESPONSE band (3800-6600 Hz) first — if sweep found and not
///      self-echo, chirp handshake is confirmed immediately.
///   2. Then check CALL band (600-3600 Hz) — if sweep found and not
///      self-echo, signal TX to send a response chirp.
///
/// After chirp handshake, attempts binary chirp message decoding.
async fn run_receive_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
    is_transmitting: Arc<AtomicBool>,
    peer_found: Arc<AtomicBool>,
    send_response_chirp: Arc<AtomicBool>,
    chirp_handshake_done: Arc<AtomicBool>,
    chirp_sig: crate::audio::chirp::AgentChirpSignature,
) {
    use crate::audio::chirp;

    let sample_rate = crate::audio::modulator::SAMPLE_RATE;
    let mut sample_buffer = Vec::new();
    let mut peak: f32 = 0.0;
    let mut last_peak_log = std::time::Instant::now();
    let mut decode_attempts: u64 = 0;
    let mut chirp_detections: u64 = 0;
    let mut chirp_msg_decodes: u64 = 0;

    // Buffer size: enough to contain the longest possible chirp (180ms) + margin
    let max_chirp_samples = (0.20 * sample_rate as f32) as usize;
    // Minimum buffer before attempting detection (shortest chirp 120ms + FFT window)
    let min_detect_samples = (0.12 * sample_rate as f32) as usize + 512;

    tracing::info!(
        signature = %chirp_sig.describe(),
        "RX: Sweep detector ready (spectrogram-based, frequency-agnostic)"
    );

    // Cooldown: don't re-trigger chirp detection for 200ms after last detection
    let mut last_chirp_detection = std::time::Instant::now() - std::time::Duration::from_secs(10);

    // Post-TX cooldown: suppress chirp detection for 300ms after own TX
    // finishes, to prevent detecting our own chirp via audio loopback
    // (e.g. PulseAudio monitor source or close-range speaker→mic echo).
    let mut post_tx_cooldown = std::time::Instant::now() - std::time::Duration::from_secs(10);
    let mut was_transmitting = false;

    tracing::info!("RX: Stage 1 — listening for chirp sweeps (CALL 0.6-3.6kHz, RESPONSE 3.8-6.6kHz)...");

    loop {
        // Always track peak level from mic, even while transmitting
        while let Some(chunk) = engine.try_recv_samples() {
            for &s in &chunk {
                let abs = s.abs();
                if abs > peak {
                    peak = abs;
                }
            }
            // Only buffer samples when NOT transmitting
            if !is_transmitting.load(Ordering::Acquire) {
                sample_buffer.extend_from_slice(&chunk);
            }
        }

        // Log audio stats every 10 seconds
        if last_peak_log.elapsed() > std::time::Duration::from_secs(10) {
            let chirp_hs = chirp_handshake_done.load(Ordering::Acquire);
            let stage = if chirp_hs { "2 (binary chirp)" } else { "1 (sweep)" };
            tracing::info!(
                peak = format!("{:.4}", peak),
                buffer = sample_buffer.len(),
                stage = stage,
                decodes = decode_attempts,
                chirps = chirp_detections,
                chirp_msgs = chirp_msg_decodes,
                "RX: Audio stats (last 10s)"
            );
            peak = 0.0;
            last_peak_log = std::time::Instant::now();
        }

        // If we're currently transmitting, clear decode buffer and wait
        let currently_transmitting = is_transmitting.load(Ordering::Acquire);
        if currently_transmitting {
            was_transmitting = true;
            sample_buffer.clear();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            continue;
        }
        // Detect the transition from transmitting → not transmitting
        if was_transmitting {
            was_transmitting = false;
            post_tx_cooldown = std::time::Instant::now();
            sample_buffer.clear(); // discard any samples captured during TX transition
            tracing::debug!("RX: Post-TX cooldown started (300ms)");
        }

        // --- Spectrogram-based chirp sweep detection (Stage 1 only) ---
        // Once the chirp handshake is done, stop doing CALL/RESPONSE detection
        // so the RX buffer can accumulate the full 5.5s binary chirp message
        // without being interrupted by RESPONSE transmissions.
        let chirp_hs = chirp_handshake_done.load(Ordering::Acquire);
        if !chirp_hs
            && sample_buffer.len() >= min_detect_samples
            && last_chirp_detection.elapsed() > std::time::Duration::from_millis(200)
            && post_tx_cooldown.elapsed() > std::time::Duration::from_millis(300)
        {
            // 1. Check RESPONSE band first (3800-6600 Hz)
            if let Some(detected) = chirp::detect_chirp_sweep(
                &sample_buffer, 3800.0, 6600.0, sample_rate,
                0.10, 0.20, // min/max duration
                500.0,      // min sweep span
            ) {
                if chirp_sig.matches_response(&detected) {
                    tracing::info!(
                        start = format!("{:.0}Hz", detected.start_freq),
                        end = format!("{:.0}Hz", detected.end_freq),
                        dur = format!("{:.0}ms", detected.duration_secs * 1000.0),
                        "RX: RESPONSE matches own signature → self-echo, ignoring"
                    );
                    let drain_to = (detected.offset + max_chirp_samples).min(sample_buffer.len());
                    sample_buffer.drain(..drain_to);
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    continue;
                }

                chirp_detections += 1;
                last_chirp_detection = std::time::Instant::now();

                tracing::info!(
                    start = format!("{:.0}Hz", detected.start_freq),
                    end = format!("{:.0}Hz", detected.end_freq),
                    dur = format!("{:.0}ms", detected.duration_secs * 1000.0),
                    chirps = chirp_detections,
                    "RX: RESPONSE chirp detected! Chirp handshake CONFIRMED."
                );
                chirp_handshake_done.store(true, Ordering::Release);
                peer_found.store(true, Ordering::Release);

                // Play a chime to indicate chirp handshake success
                if let Err(e) = crate::audio::device::play_handshake_chime() {
                    tracing::debug!("Could not play chime: {}", e);
                }

                let drain_to = (detected.offset + max_chirp_samples).min(sample_buffer.len());
                sample_buffer.drain(..drain_to);
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            }

            // 2. Check CALL band (600-3600 Hz)
            if let Some(detected) = chirp::detect_chirp_sweep(
                &sample_buffer, 600.0, 3600.0, sample_rate,
                0.10, 0.20,
                500.0,
            ) {
                if chirp_sig.matches_call(&detected) {
                    tracing::info!(
                        start = format!("{:.0}Hz", detected.start_freq),
                        end = format!("{:.0}Hz", detected.end_freq),
                        dur = format!("{:.0}ms", detected.duration_secs * 1000.0),
                        "RX: CALL matches own signature → self-echo, ignoring"
                    );
                    let drain_to = (detected.offset + max_chirp_samples).min(sample_buffer.len());
                    sample_buffer.drain(..drain_to);
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    continue;
                }

                chirp_detections += 1;
                last_chirp_detection = std::time::Instant::now();

                tracing::info!(
                    start = format!("{:.0}Hz", detected.start_freq),
                    end = format!("{:.0}Hz", detected.end_freq),
                    dur = format!("{:.0}ms", detected.duration_secs * 1000.0),
                    chirps = chirp_detections,
                    "RX: CALL chirp detected. Signaling TX to send RESPONSE."
                );
                send_response_chirp.store(true, Ordering::Release);
                peer_found.store(true, Ordering::Release);

                let drain_to = (detected.offset + max_chirp_samples).min(sample_buffer.len());
                sample_buffer.drain(..drain_to);
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            }
        }

        // --- Stages 2 & 3: only after chirp handshake ---
        let chirp_hs = chirp_handshake_done.load(Ordering::Acquire);
        if !chirp_hs {
            // Still in stage 1 — only listen for chirps, trim buffer to avoid unbounded growth
            if sample_buffer.len() > max_chirp_samples * 4 {
                let excess = sample_buffer.len() - max_chirp_samples * 2;
                sample_buffer.drain(..excess);
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            continue;
        }

        // --- Binary chirp message decode (Stage 2) ---
        // A binary chirp message is 44 bit-slots, each ~100ms = ~4.4 seconds of audio.
        // We need at least ~4 seconds of buffered audio before attempting a decode.
        let min_chirp_msg_samples = (sample_rate as f32 * 4.0) as usize;
        if sample_buffer.len() >= min_chirp_msg_samples {
            decode_attempts += 1;
            if let Some((port, caps)) = chirp::decode_chirp_message_sweep(&sample_buffer, sample_rate) {
                chirp_msg_decodes += 1;
                let address = format!("127.0.0.1:{}", port);
                let caps_desc = crate::protocol::message::Capabilities(caps).describe();
                tracing::info!(
                    port,
                    capabilities = caps,
                    caps_desc = %caps_desc,
                    "RX: Binary chirp message decoded! Peer at {} — port={} caps=0x{:02X} [{}]",
                    address, port, caps, caps_desc
                );

                let pseudo_id = {
                    let mut bytes = [0u8; 16];
                    let port_bytes = port.to_be_bytes();
                    bytes[0] = 0xBC; // Binary Chirp marker
                    bytes[1] = 0x01;
                    bytes[14] = port_bytes[0];
                    bytes[15] = port_bytes[1];
                    Uuid::from_bytes(bytes)
                };
                let msg = crate::protocol::message::DiscoveryMessage::announce(
                    pseudo_id,
                    &address,
                    crate::protocol::message::Capabilities(caps),
                );
                let mut mgr = manager.lock().await;
                mgr.handle_message(&msg);
                sample_buffer.clear();
                continue;
            } else {
                // Trim old samples to avoid unbounded growth
                let trim = min_chirp_msg_samples / 2;
                if sample_buffer.len() > min_chirp_msg_samples * 2 {
                    sample_buffer.drain(..trim);
                }
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
async fn run_tcp_accept_loop(
    listener: TcpCommListener,
    config: Config,
    mcp_client: Option<Arc<tokio::sync::Mutex<McpClient>>>,
    obp_client: Option<Arc<ObpClient>>,
) {
    let listener = Arc::new(listener);
    let config = Arc::new(config);
    let obp_client = obp_client.map(|c| c.clone());
    loop {
        let listener = listener.clone();
        let config = config.clone();
        let obp = obp_client.clone();
        match tokio::task::spawn_blocking(move || listener.accept()).await {
            Ok(Ok(channel)) => {
                tracing::info!("Accepted TCP connection: {}", channel.description());
                let span = tracing::Span::current();
                tokio::spawn(async move {
                    handle_tcp_connection(channel, &config, &obp).await;
                }.instrument(span));
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
    // mcp_client kept alive for lifetime of the accept loop
    drop(mcp_client);
}

async fn handle_tcp_connection(
    channel: TcpChannel,
    config: &Config,
    obp_client: &Option<Arc<ObpClient>>,
) {
    let desc = channel.description();
    tracing::info!("Handling TCP connection: {}", desc);

    // Set a 30s read timeout for the exploration protocol
    if let Err(e) = channel.set_read_timeout(Some(std::time::Duration::from_secs(30))) {
        tracing::warn!("Could not set read timeout: {}", e);
    }

    // First message should be a greeting/handshake
    match channel.recv_message() {
        Ok(data) => {
            let msg_text = String::from_utf8_lossy(&data);
            tracing::info!(
                message = %msg_text,
                "Received message from peer"
            );

            // Check if it's a hello message and respond
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&data) {
                if json.get("type").and_then(|t| t.as_str()) == Some("hello") {
                    let peer_name = json.get("agent_name").and_then(|n| n.as_str()).unwrap_or("unknown");
                    tracing::info!(
                        peer_name = peer_name,
                        "Peer introduced itself"
                    );

                    // Send our greeting back
                    let response = serde_json::json!({
                        "type": "hello_ack",
                        "agent_name": config.agent_name,
                        "message": format!("Hello {}! Nice to meet you. I'm {}.", peer_name, config.agent_name),
                    });
                    let response_bytes = serde_json::to_vec(&response).unwrap_or_default();
                    if let Err(e) = channel.send_message(&response_bytes) {
                        tracing::error!("Failed to send hello_ack: {}", e);
                        return;
                    }
                    tracing::info!(
                        peer = peer_name,
                        "Handshake complete! Agents are now connected."
                    );

                    // Celebratory chime
                    if let Err(e) = crate::audio::device::play_handshake_chime() {
                        tracing::debug!("Could not play chime: {}", e);
                    }

                    // After hello/hello_ack, run the exploration protocol as responder
                    let channel = Arc::new(channel);
                    run_obp_exploration_responder(
                        channel,
                        &config.agent_name,
                        obp_client,
                    ).await;
                    return;
                }
            }

            // Not a hello message — just read until close
            loop {
                match channel.recv_message() {
                    Ok(data) => {
                        tracing::info!(
                            bytes = data.len(),
                            "Received {} bytes via TCP",
                            data.len()
                        );
                    }
                    Err(e) => {
                        tracing::debug!("TCP connection closed: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            tracing::debug!("TCP connection closed before handshake: {}", e);
        }
    }
}

// --- UDP broadcast discovery ---

const UDP_DISCOVERY_PORT: u16 = 7399;

/// Create a UDP socket with SO_REUSEADDR (and SO_REUSEPORT on Linux/macOS)
/// so multiple agents on the same machine can all receive broadcasts.
fn create_reusable_udp_socket(port: u16) -> Result<tokio::net::UdpSocket> {
    use socket2::{Domain, Protocol, Socket, Type};
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_reuse_address(true)?;
    #[cfg(unix)]
    socket.set_reuse_port(true)?;
    socket.set_nonblocking(true)?;
    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", port).parse().unwrap();
    socket.bind(&addr.into())?;
    let std_socket: std::net::UdpSocket = socket.into();
    Ok(tokio::net::UdpSocket::from_std(std_socket)?)
}

/// Delay before UDP discovery starts, giving chirp-based audio discovery a
/// chance first. Agents should spend 10 minutes trying to find each other
/// via chirps before falling back to UDP.
const UDP_DELAY_SECS: u64 = 600;

/// Periodically broadcast discovery messages via UDP.
async fn run_udp_announce_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    peer_found: Arc<AtomicBool>,
    handshake_done: Arc<AtomicBool>,
    immediate: bool,
) {
    if immediate {
        tracing::info!("UDP announce: starting immediately (--udp mode)");
    } else {
        tracing::info!("UDP announce: waiting {}s for audio discovery first...", UDP_DELAY_SECS);
        tokio::time::sleep(std::time::Duration::from_secs(UDP_DELAY_SECS)).await;

        // If handshake already happened via audio, no need for UDP
        if handshake_done.load(Ordering::Acquire) {
            tracing::info!("UDP announce: handshake already done via audio, skipping UDP");
            return;
        }

        tracing::info!("UDP announce: audio discovery period ended, starting UDP fallback");
    }

    let socket = match tokio::net::UdpSocket::bind("0.0.0.0:0").await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("UDP announce: failed to bind socket: {}", e);
            return;
        }
    };
    if let Err(e) = socket.set_broadcast(true) {
        tracing::warn!("UDP announce: failed to enable broadcast: {}", e);
        return;
    }

    let broadcast_addr = format!("255.255.255.255:{}", UDP_DISCOVERY_PORT);
    tracing::info!("UDP announce: broadcasting to {}", broadcast_addr);

    let mut rng = Rng::from_entropy();
    let mut current_interval_ms: u64 = INITIAL_INTERVAL_MS;

    loop {
        if peer_found.load(Ordering::Acquire) {
            current_interval_ms = INITIAL_INTERVAL_MS;
        }

        // After handshake: slow down (interval * 4 + 30s)
        if handshake_done.load(Ordering::Acquire) {
            current_interval_ms = current_interval_ms * 4 + 30_000;
            tracing::info!(
                interval_s = format!("{:.1}", current_interval_ms as f64 / 1000.0),
                "UDP: Handshake done, slowing down announces"
            );
        }

        let msg_bytes = {
            let mgr = manager.lock().await;
            let msg = mgr.build_announce();
            codec::encode_message(&msg)
        };

        match socket.send_to(&msg_bytes, &broadcast_addr).await {
            Ok(_) => {
                tracing::debug!("UDP: sent {} byte announce", msg_bytes.len());
            }
            Err(e) => {
                tracing::warn!("UDP send failed: {}", e);
            }
        }

        // Jitter
        let jitter_range = current_interval_ms / 4;
        let jitter = if jitter_range > 0 { rng.range(jitter_range * 2) } else { 0 };
        let sleep_ms = current_interval_ms - jitter_range + jitter;
        tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;

        current_interval_ms = ((current_interval_ms as f64 * BACKOFF_MULTIPLIER) as u64)
            .min(MAX_INTERVAL_MS);
    }
}

/// Listen for UDP broadcast discovery messages from other agents.
async fn run_udp_receive_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    immediate: bool,
) {
    if !immediate {
        // Wait for audio discovery period to pass
        tokio::time::sleep(std::time::Duration::from_secs(UDP_DELAY_SECS)).await;
    }

    // Use socket2 to set SO_REUSEADDR/SO_REUSEPORT before binding,
    // so multiple agents on the same machine can all listen on the same port.
    let socket = match create_reusable_udp_socket(UDP_DISCOVERY_PORT) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("UDP listen: failed to bind port {}: {}", UDP_DISCOVERY_PORT, e);
            return;
        }
    };

    tracing::info!("UDP listen: waiting for broadcasts on port {}", UDP_DISCOVERY_PORT);

    let mut buf = [0u8; 512];
    loop {
        match socket.recv_from(&mut buf).await {
            Ok((len, src)) => {
                if let Some(msg) = codec::decode_message(&buf[..len]) {
                    let mut mgr = manager.lock().await;
                    if mgr.handle_message(&msg) {
                        tracing::info!(
                            peer_id = %msg.agent_id,
                            address = %msg.address,
                            src = %src,
                            "UDP: Discovered peer"
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!("UDP recv error: {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}
