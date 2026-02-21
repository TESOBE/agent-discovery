/// Agent orchestrator: ties audio, discovery, negotiation, and OBP together.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
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

// ---------------------------------------------------------------------------
// Agent mood — a simple status reflecting connectivity to other peers.
// ---------------------------------------------------------------------------

/// Agent mood based on peer connectivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AgentMood {
    /// Audio doesn't work, can't reach signal channels or network.
    Struggling = 0,
    /// Can broadcast (audio or signal channels) but no peer is responding.
    Lonely = 1,
    /// Connected to exactly one other agent.
    Happy = 2,
    /// Connected to exactly two other agents.
    VeryHappy = 3,
    /// Connected to three or more other agents.
    Party = 4,
}

impl AgentMood {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Struggling,
            1 => Self::Lonely,
            2 => Self::Happy,
            3 => Self::VeryHappy,
            4 => Self::Party,
            _ => Self::Party,
        }
    }

    /// Human-friendly label for logs.
    pub fn label(self) -> &'static str {
        match self {
            Self::Struggling => "Struggling",
            Self::Lonely => "Lonely",
            Self::Happy => "Happy",
            Self::VeryHappy => "Very Happy",
            Self::Party => "Party!",
        }
    }

    /// Compute mood from the number of connected peers.
    pub fn from_peer_count(count: usize) -> Self {
        match count {
            0 => Self::Lonely,
            1 => Self::Happy,
            2 => Self::VeryHappy,
            _ => Self::Party,
        }
    }
}

impl std::fmt::Display for AgentMood {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Thread-safe mood tracker backed by an AtomicU8.
#[derive(Clone)]
pub struct MoodTracker {
    value: Arc<AtomicU8>,
}

impl MoodTracker {
    pub fn new(initial: AgentMood) -> Self {
        Self { value: Arc::new(AtomicU8::new(initial as u8)) }
    }

    pub fn get(&self) -> AgentMood {
        AgentMood::from_u8(self.value.load(Ordering::Relaxed))
    }

    /// Update mood and log if it changed. Returns true if mood changed.
    pub fn set(&self, mood: AgentMood) -> bool {
        let old = self.value.swap(mood as u8, Ordering::Relaxed);
        if old != mood as u8 {
            let old_mood = AgentMood::from_u8(old);
            tracing::info!(
                old = %old_mood, new = %mood,
                "Agent mood changed: {} -> {}", old_mood, mood
            );
            true
        } else {
            false
        }
    }

    /// Update mood from the current peer count.
    pub fn update_from_peers(&self, peer_count: usize) {
        self.set(AgentMood::from_peer_count(peer_count));
    }
}

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
            let error_msg = format!("{}", e);
            tracing::warn!("Audio engine unavailable: {}. Running in discovery-only mode.", error_msg);

            // Post a help request to the signal channel so other agents
            // (or humans monitoring) can see this agent needs audio help.
            if let Some(ref obp) = obp_client {
                let help_payload = serde_json::json!({
                    "payload": {
                        "agent_name": config.agent_name,
                        "type": "audio-help-needed",
                        "error": error_msg,
                        "message": format!(
                            "Agent '{}' cannot initialise audio: {}. \
                             Running in discovery-only mode (UDP). \
                             Needs help configuring audio devices.",
                            config.agent_name, error_msg
                        ),
                        "mac_address": get_mac_address(),
                        "timestamp": crate::obp::entities::iso_now(),
                    }
                });
                let path = format!(
                    "/obp/{}/signal/channels/agent-help/messages",
                    crate::obp::client::API_VERSION
                );
                tracing::info!(
                    "Signal POST {}{} — requesting audio help",
                    obp.base_url(), path
                );
                match obp.post(&path, &help_payload).await {
                    Ok(val) => tracing::info!("Posted audio help request to signal channel: {}", val),
                    Err(post_err) => tracing::warn!("Failed to post help request: {}", post_err),
                }
            }

            None
        }
    };

    // Agent mood — reflects connectivity status
    let has_audio = audio_engine.is_some();
    let has_obp = obp_client.is_some();
    let initial_mood = if !has_audio && !has_obp {
        AgentMood::Struggling
    } else {
        AgentMood::Lonely
    };
    let mood = MoodTracker::new(initial_mood);
    tracing::info!(mood = %mood.get(), "Initial agent mood: {}", mood.get());

    // Shared flag: when true, the RX loop discards samples to avoid self-echo
    let is_transmitting = Arc::new(AtomicBool::new(false));
    // Set to true when a peer is discovered, tells TX loop to reset back-off
    let peer_found = Arc::new(AtomicBool::new(false));
    // Set to true after a handshake completes — slows down announce interval
    let handshake_done = Arc::new(AtomicBool::new(false));

    // Start audio loops (skipped with --udp)
    if !udp_only {
        // Start the announce loop (audio TX — says "hello" with binary chirp data)
        if let Some(ref engine) = audio_engine {
            let mgr = discovery_manager.clone();
            let eng = engine.clone();
            let tx_flag = is_transmitting.clone();
            let pf_flag = peer_found.clone();
            let hs_flag = handshake_done.clone();
            let sig = chirp_sig.clone();
            let name = agent_name.clone();
            tokio::spawn(
                run_announce_loop(mgr, eng, tx_flag, pf_flag, hs_flag, sig, name)
                    .instrument(agent_span.clone()),
            );
        }

        // Start the receive loop (audio RX — listens for "hello" binary chirp data)
        if let Some(ref engine) = audio_engine {
            let mgr = discovery_manager.clone();
            let eng = engine.clone();
            let tx_flag = is_transmitting.clone();
            let pf_flag = peer_found.clone();
            tokio::spawn(
                run_receive_loop(mgr, eng, tx_flag, pf_flag, actual_port)
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

    // Periodic mood/status logger (every 30 seconds)
    {
        let mgr = discovery_manager.clone();
        let m = mood.clone();
        let name = agent_name.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                let peer_count = mgr.lock().await.peers().len();
                m.update_from_peers(peer_count);
                tracing::info!(
                    mood = %m.get(),
                    peers = peer_count,
                    "Agent '{}' status: {} ({} peer{})",
                    name, m.get(), peer_count, if peer_count == 1 { "" } else { "s" }
                );
            }
        }.instrument(agent_span.clone()));
    }

    // Run the event loop inside the agent span
    let _guard = agent_span.enter();

    loop {
        match event_rx.recv().await {
            Ok(event) => match event {
                DiscoveryEvent::PeerDiscovered(peer) => {
                    // Reset announce back-off so we respond quickly
                    peer_found.store(true, Ordering::Release);

                    let peer_count = {
                        let mgr = discovery_manager.lock().await;
                        mgr.peers().len()
                    };
                    mood.update_from_peers(peer_count);

                    tracing::info!(
                        peer_id = %peer.agent_id,
                        address = %peer.address,
                        mood = %mood.get(),
                        "New peer discovered! Mood: {}. Starting negotiation...",
                        mood.get()
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
                    negotiated_peers.remove(&id);
                    let peer_count = {
                        let mgr = discovery_manager.lock().await;
                        mgr.peers().len()
                    };
                    mood.update_from_peers(peer_count);
                    tracing::info!(
                        peer_id = %id,
                        mood = %mood.get(),
                        "Peer lost. Mood: {} ({} peer{} remaining)",
                        mood.get(), peer_count, if peer_count == 1 { "" } else { "s" }
                    );
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
                            mcp_client, obp_client, mcp_diagnosis,
                        ).await;
                    }
                    "obp" => {
                        tracing::info!("OBP shared storage channel configured");
                    }
                    "signal" => {
                        tracing::info!("Signal channel chosen — establishing TCP for exploration, signal channels for ongoing coordination");
                        establish_tcp_channel(
                            agent_id, agent_name, peer,
                            mcp_client, obp_client, mcp_diagnosis,
                        ).await;
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
            mcp_client, obp_client, mcp_diagnosis,
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
    mcp_client: &Option<Arc<tokio::sync::Mutex<McpClient>>>,
    obp_client: &Option<Arc<ObpClient>>,
    mcp_diagnosis: &Option<Vec<String>>,
) {
    tracing::info!("=== Starting OBP Exploration (initiator) ===");

    // Phase 1: Send ExploreStart, receive ExploreAck
    let start_msg = ExplorationMsg::ExploreStart {
        agent_name: agent_name.to_string(),
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

    let obp_base = obp_client.as_ref().map(|o| o.base_url()).unwrap_or("(no OBP client)");
    tracing::info!("Found management endpoint: {} {}{} ({})", method, obp_base, path, endpoint_id);

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
                create_entity_via_http(obp_client, &entity_def, entity_name).await
            }
        }
    } else {
        tracing::info!("No MCP client, creating entity via direct HTTP");
        create_entity_via_http(obp_client, &entity_def, entity_name).await
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

    // Phase 5: Discover CRUD endpoints via HATEOAS (_links)
    let discovered_endpoints = if let Some(ref obp) = obp_client {
        match entities::discover_entity_endpoints_via_http(obp, entity_name).await {
            Ok(eps) => eps,
            Err(e) => {
                tracing::warn!("Failed to discover CRUD endpoints via _links: {}", e);
                Vec::new()
            }
        }
    } else {
        tracing::warn!("No OBP HTTP client, skipping CRUD endpoint discovery");
        Vec::new()
    };

    tracing::info!("Discovered {} CRUD endpoints for '{}':", discovered_endpoints.len(), entity_name);
    for ep in &discovered_endpoints {
        tracing::info!("  {} {}{} - {}", ep.method, obp_base, ep.path, ep.summary);
    }

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

    // Phase 6: Create a test record using discovered POST endpoint
    let test_record = serde_json::json!({
        "agent_id": format!("initiator-{}", agent_name),
        "agent_name": agent_name,
        "peer_id": "responder-peer",
        "peer_address": "tcp-channel",
        "discovery_method": "exploration-protocol",
        "timestamp": entities::iso_now()
    });

    let create_path = entities::find_create_record_path(&discovered_endpoints);
    let record_result = if let Some(ref obp) = obp_client {
        match create_path {
            Some(path) => {
                tracing::info!("Using discovered POST endpoint: {}{}", obp.base_url(), path);
                match obp.post(path, &test_record).await {
                    Ok(val) => {
                        tracing::info!("Created test record: {}", val);
                        val
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create test record: {}", e);
                        serde_json::json!({"error": format!("{}", e)})
                    }
                }
            }
            None => {
                tracing::warn!("No POST endpoint discovered for '{}', cannot create record", entity_name);
                serde_json::json!({"error": "no POST endpoint discovered via _links"})
            }
        }
    } else {
        serde_json::json!({"error": "no OBP client"})
    };

    let _ = send_exploration_msg(&channel, &ExplorationMsg::RecordCreated {
        entity_name: entity_name.to_string(),
        endpoint_used: create_path.unwrap_or("none").to_string(),
        record: record_result,
    }).await;

    // Wait for responder to verify
    let phase6_success = match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::RecordVerified { matches, .. }) => {
            tracing::info!("Responder verified record: matches={}", matches);
            matches
        }
        Ok(other) => {
            tracing::warn!("Expected RecordVerified, got: {:?}", other);
            false
        }
        Err(e) => {
            tracing::error!("Failed to recv RecordVerified: {}", e);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::ExplorationComplete {
                summary: format!("Exploration ended with error: {}", e),
                success: false,
            }).await;
            return;
        }
    };

    // Phase 7: Signal channel discovery (non-fatal)
    tracing::info!("=== Phase 7: Signal Channel Discovery ===");

    let signal_ok = 'signal: {
        // Step 1: Discover signal endpoints via MCP or fallback
        let signal_endpoints = if let Some(ref mcp) = mcp_client {
            let mut mcp_guard = mcp.lock().await;
            match mcp_guard
                .call_tool(
                    "list_endpoints_by_tag",
                    serde_json::json!({"tags": ["Signaling", "Signal", "Channel"]}),
                )
                .await
            {
                Ok(val) => {
                    let content = extract_mcp_text_content(&val);
                    tracing::info!("MCP signal endpoint discovery: {}", &content[..content.len().min(300)]);
                    // Try to parse endpoints from MCP response
                    parse_signal_endpoints(&content).unwrap_or_else(|| {
                        tracing::info!("Could not parse MCP signal response, using known endpoints");
                        known_signal_endpoints()
                    })
                }
                Err(e) => {
                    tracing::warn!("MCP signal endpoint discovery failed: {}. Using known endpoints.", e);
                    known_signal_endpoints()
                }
            }
        } else {
            tracing::info!("No MCP client, using known signal endpoints");
            known_signal_endpoints()
        };

        tracing::info!("Found {} signal endpoints:", signal_endpoints.len());
        for ep in &signal_endpoints {
            tracing::info!("  {} {}{} - {}", ep.method, obp_base, ep.path, ep.summary);
        }

        // Step 2: Share signal endpoints with responder
        if let Err(e) = send_exploration_msg(&channel, &ExplorationMsg::FoundSignalEndpoints {
            endpoints: signal_endpoints.clone(),
        }).await {
            tracing::warn!("Failed to send FoundSignalEndpoints: {}", e);
            break 'signal false;
        }

        // Wait for acknowledgement
        match recv_exploration_msg(&channel).await {
            Ok(ExplorationMsg::Acknowledged { phase }) => {
                tracing::info!("Responder acknowledged signal phase: {}", phase);
            }
            Ok(other) => tracing::debug!("Expected Acknowledged for signal, got: {:?}", other),
            Err(e) => {
                tracing::warn!("Failed to recv signal ack: {}", e);
                break 'signal false;
            }
        }

        // Step 3: Publish a test message to the agent-discovery channel
        let test_channel = "agent-discovery";
        let test_payload = serde_json::json!({
            "payload": {
                "from": agent_name,
                "type": "exploration-test",
                "timestamp": entities::iso_now(),
                "message": format!("Signal channel test from {}", agent_name)
            }
        });

        let publish_result = if let Some(ref mcp) = mcp_client {
            let mut mcp_guard = mcp.lock().await;
            match mcp_guard
                .call_tool(
                    "call_obp_api",
                    serde_json::json!({
                        "endpoint_id": "OBPv6.0.0-publishSignalMessage",
                        "path_params": {"CHANNEL_NAME": test_channel},
                        "body": &test_payload,
                    }),
                )
                .await
            {
                Ok(val) => {
                    let text = extract_mcp_text_content(&val);
                    tracing::info!("Published signal message via MCP: {}", text);
                    serde_json::from_str::<serde_json::Value>(&text).ok()
                }
                Err(e) => {
                    tracing::warn!("MCP signal publish failed: {}. Trying direct HTTP.", e);
                    None
                }
            }
        } else {
            None
        };

        // Fallback to direct HTTP if MCP failed
        let publish_result = match publish_result {
            Some(val) => val,
            None => {
                if let Some(ref obp) = obp_client {
                    match publish_signal_message_via_http(obp, test_channel, &test_payload).await {
                        Ok(val) => {
                            tracing::info!("Published signal message via HTTP: {}", val);
                            val
                        }
                        Err(e) => {
                            tracing::warn!("Signal channel publish via HTTP failed: {}", e);
                            break 'signal false;
                        }
                    }
                } else {
                    tracing::warn!("No OBP client available for signal channel test");
                    break 'signal false;
                }
            }
        };

        let message_id = publish_result
            .get("message_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        tracing::info!("Signal test message published: id={}", message_id);

        // Step 4: Tell responder about the test
        if let Err(e) = send_exploration_msg(&channel, &ExplorationMsg::SignalChannelTested {
            channel_name: test_channel.to_string(),
            message_id: message_id.clone(),
            payload: test_payload.clone(),
        }).await {
            tracing::warn!("Failed to send SignalChannelTested: {}", e);
            break 'signal false;
        }

        // Step 5: Wait for responder's verification
        match recv_exploration_msg(&channel).await {
            Ok(ExplorationMsg::SignalChannelVerified { channel_name, verified }) => {
                tracing::info!(
                    "Signal channel '{}' verification: {}",
                    channel_name,
                    if verified { "SUCCESS" } else { "FAILED" }
                );
                verified
            }
            Ok(other) => {
                tracing::warn!("Expected SignalChannelVerified, got: {:?}", other);
                false
            }
            Err(e) => {
                tracing::warn!("Failed to recv SignalChannelVerified: {}", e);
                false
            }
        }
    };

    if signal_ok {
        tracing::info!("=== Phase 7: Signal channels verified ===");
    } else {
        tracing::warn!("=== Phase 7: Signal channels not available (non-fatal) ===");
    }

    // Send ExplorationComplete (success based on Phase 6 result)
    let _ = send_exploration_msg(&channel, &ExplorationMsg::ExplorationComplete {
        summary: format!(
            "Explored dynamic entity '{}' (phase6={}) and signal channels (phase7={})",
            entity_name, phase6_success, signal_ok
        ),
        success: phase6_success,
    }).await;

    if phase6_success {
        tracing::info!("=== OBP Exploration COMPLETE (success) ===");
    } else {
        tracing::warn!("=== OBP Exploration COMPLETE (verification mismatch) ===");
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
    let obp_base = obp_client.as_ref().map(|o| o.base_url()).unwrap_or("(no OBP client)");
    tracing::info!("=== Starting OBP Exploration (responder) ===");

    // Expect ExploreStart
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::ExploreStart { agent_name: peer_name }) => {
            tracing::info!("Received ExploreStart from {}", peer_name);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::ExploreAck {
                agent_name: agent_name.to_string(),
            }).await;
        }
        Ok(other) => {
            tracing::warn!("Expected ExploreStart, got: {:?}", other);
            return;
        }
        Err(e) => {
            tracing::error!("Failed to receive ExploreStart: {}", e);
            return;
        }
    }

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
                "Initiator found management endpoint: {} {}{} ({}) - {}",
                method, obp_base, path, endpoint_id, summary
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

    // Phase 5: Receive DiscoveredCrudEndpoints — independently verify via _links
    let responder_endpoints = match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::DiscoveredCrudEndpoints { entity_name: ename, endpoints }) => {
            tracing::info!(
                "Initiator discovered {} CRUD endpoints for '{}':",
                endpoints.len(), ename
            );
            for ep in &endpoints {
                tracing::info!("  {} {}{} - {}", ep.method, obp_base, ep.path, ep.summary);
            }

            // Independently verify via _links
            let our_endpoints = if let Some(ref obp) = obp_client {
                match entities::discover_entity_endpoints_via_http(obp, &ename).await {
                    Ok(eps) => eps,
                    Err(e) => {
                        tracing::warn!("Responder _links query failed: {}", e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            let confirmed = !our_endpoints.is_empty() || !endpoints.is_empty();
            tracing::info!(
                "Responder independently found {} endpoints (confirmed={}):",
                our_endpoints.len(), confirmed
            );
            for ep in &our_endpoints {
                tracing::info!("  {} {}{} - {}", ep.method, obp_base, ep.path, ep.summary);
            }
            tracing::info!(
                "Responder endpoint verification complete (confirmed={})",
                confirmed
            );

            let _ = send_exploration_msg(&channel, &ExplorationMsg::EndpointsConfirmed {
                entity_name: ename,
                confirmed,
                our_endpoints: our_endpoints.clone(),
            }).await;
            our_endpoints
        }
        Ok(other) => {
            tracing::warn!("Expected DiscoveredCrudEndpoints, got: {:?}", other);
            let _ = send_exploration_msg(&channel, &ExplorationMsg::EndpointsConfirmed {
                entity_name: entity_name.clone(),
                confirmed: false,
                our_endpoints: Vec::new(),
            }).await;
            Vec::new()
        }
        Err(e) => {
            tracing::error!("Failed to receive CRUD endpoints: {}", e);
            return;
        }
    };

    // Phase 6: Receive RecordCreated — read it back using discovered GET endpoint
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::RecordCreated { entity_name: ename, record, .. }) => {
            tracing::info!("Initiator created test record for '{}'", ename);

            // Try to read back records using the discovered GET endpoint
            let list_path = entities::find_list_records_path(&responder_endpoints);
            let (read_back, matches) = if let Some(ref obp) = obp_client {
                match list_path {
                    Some(path) => {
                        tracing::info!("Reading back records via discovered GET: {}{}", obp.base_url(), path);
                        match obp.get(path).await {
                            Ok(val) => {
                                tracing::info!("Read back records: {}", val);
                                let has_records = val.get(&format!("{}List", ename))
                                    .or(val.get(&format!("{}_list", ename)))
                                    .and_then(|l| l.as_array())
                                    .map(|arr| !arr.is_empty())
                                    .unwrap_or(false);
                                let matches = has_records || !val.is_null();
                                (val, matches)
                            }
                            Err(e) => {
                                tracing::warn!("Failed to read back records: {}", e);
                                (serde_json::json!({"error": format!("{}", e)}), false)
                            }
                        }
                    }
                    None => {
                        tracing::warn!("No GET endpoint discovered, cannot verify record");
                        (serde_json::json!({"error": "no GET endpoint discovered"}), false)
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

    // Phase 7: Signal channel discovery and verification (responder side)
    // The next message may be FoundSignalEndpoints (Phase 7) or ExplorationComplete (no Phase 7)
    match recv_exploration_msg(&channel).await {
        Ok(ExplorationMsg::FoundSignalEndpoints { endpoints }) => {
            tracing::info!("=== Phase 7 (responder): Signal channel discovery ({} endpoints) ===", endpoints.len());
            for ep in &endpoints {
                tracing::info!("  Signal endpoint: {} {}{} - {}", ep.method, obp_base, ep.path, ep.summary);
            }
            let _ = send_exploration_msg(&channel, &ExplorationMsg::Acknowledged {
                phase: "signal_endpoints".into(),
            }).await;

            // Receive SignalChannelTested — then verify by reading the channel
            match recv_exploration_msg(&channel).await {
                Ok(ExplorationMsg::SignalChannelTested { channel_name, message_id, payload: _ }) => {
                    tracing::info!(
                        "Initiator tested signal channel '{}' with message_id={}",
                        channel_name, message_id
                    );

                    // Try to read back the message from the signal channel
                    let verified = if let Some(ref obp) = obp_client {
                        match read_signal_messages_via_http(obp, &channel_name).await {
                            Ok(val) => {
                                tracing::info!("Signal channel '{}' messages: {}", channel_name, val);
                                // Check if our test message is in the response
                                let found = if let Some(messages) = val.get("messages").and_then(|m| m.as_array()) {
                                    messages.iter().any(|m| {
                                        m.get("message_id")
                                            .and_then(|id| id.as_str())
                                            .map(|id| id == message_id)
                                            .unwrap_or(false)
                                    })
                                } else {
                                    // If we got any response at all, the channel exists
                                    !val.is_null()
                                };
                                found
                            }
                            Err(e) => {
                                tracing::warn!("Failed to read signal channel '{}': {}", channel_name, e);
                                false
                            }
                        }
                    } else {
                        tracing::warn!("No OBP client to verify signal channel");
                        false
                    };

                    tracing::info!("Signal channel '{}' verified: {}", channel_name, verified);
                    let _ = send_exploration_msg(&channel, &ExplorationMsg::SignalChannelVerified {
                        channel_name,
                        verified,
                    }).await;
                }
                Ok(other) => {
                    tracing::warn!("Expected SignalChannelTested, got: {:?}", other);
                    let _ = send_exploration_msg(&channel, &ExplorationMsg::SignalChannelVerified {
                        channel_name: "unknown".into(),
                        verified: false,
                    }).await;
                }
                Err(e) => {
                    tracing::warn!("Failed to receive SignalChannelTested: {}", e);
                }
            }

            // Now wait for ExplorationComplete
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
        Ok(ExplorationMsg::ExplorationComplete { summary, success }) => {
            // No Phase 7 — initiator went straight to complete
            if success {
                tracing::info!("=== OBP Exploration COMPLETE (responder, no signal phase): {} ===", summary);
            } else {
                tracing::warn!("=== OBP Exploration ended (responder, no signal phase): {} ===", summary);
            }
        }
        Ok(other) => tracing::debug!("Expected FoundSignalEndpoints or ExplorationComplete, got: {:?}", other),
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

/// Parse the MCP list_endpoints_by_tag response to extract signal channel endpoints.
fn parse_signal_endpoints(content: &str) -> Option<Vec<crate::obp::exploration::DiscoveredEndpoint>> {
    use crate::obp::exploration::DiscoveredEndpoint;
    let val = serde_json::from_str::<serde_json::Value>(content).ok()?;

    let raw_endpoints = if val.is_array() {
        val.as_array().cloned().unwrap_or_default()
    } else if let Some(arr) = val.get("endpoints").and_then(|e| e.as_array()) {
        arr.clone()
    } else {
        vec![val]
    };

    let mut result = Vec::new();
    for ep in &raw_endpoints {
        let op_id = ep.get("operation_id").or(ep.get("id"))
            .and_then(|v| v.as_str()).unwrap_or("");
        let method = ep.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let path = ep.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let summary = ep.get("summary").and_then(|v| v.as_str()).unwrap_or("");

        // Filter for signal-related endpoints
        let is_signal = op_id.to_lowercase().contains("signal")
            || path.to_lowercase().contains("signal")
            || summary.to_lowercase().contains("signal");

        if is_signal && !op_id.is_empty() {
            result.push(DiscoveredEndpoint {
                method: method.to_string(),
                path: path.to_string(),
                operation_id: op_id.to_string(),
                summary: summary.to_string(),
            });
        }
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
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

            // Look for the system-level create dynamic entity endpoint
            if (op_id.contains("createSystemLevelDynamicEntity") || op_id.contains("Create-System-Level-Dynamic-Entity"))
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
        "OBPv6.0.0-createSystemLevelDynamicEntity".to_string(),
        "POST".to_string(),
        "/obp/v6.0.0/management/system-dynamic-entities".to_string(),
        "Create System Level Dynamic Entity".to_string(),
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

/// Return the well-known create-system-level-dynamic-entity endpoint tuple.
/// Used as a fallback when MCP discovery is unavailable.
fn known_create_endpoint() -> (String, String, String, String) {
    (
        "OBPv6.0.0-createSystemLevelDynamicEntity".to_string(),
        "POST".to_string(),
        "/obp/v6.0.0/management/system-dynamic-entities".to_string(),
        "Create System Level Dynamic Entity".to_string(),
    )
}

/// Create a system-level dynamic entity definition via direct HTTP, returning the response.
async fn create_entity_via_http(
    obp_client: &Option<Arc<ObpClient>>,
    entity_def: &serde_json::Value,
    entity_name: &str,
) -> serde_json::Value {
    let Some(ref obp) = obp_client else {
        tracing::warn!("No OBP HTTP client available to create entity");
        return serde_json::json!({"error": "no OBP client"});
    };
    let path = format!("/obp/{}/management/system-dynamic-entities", crate::obp::client::API_VERSION);
    match obp.post(&path, entity_def).await {
        Ok(val) => {
            tracing::info!("Created system entity '{}' via HTTP: {}", entity_name, val);
            val
        }
        Err(e) => {
            tracing::warn!("HTTP system entity creation: {} (may already exist)", e);
            serde_json::json!({"note": format!("creation returned error (may exist): {}", e)})
        }
    }
}

/// Return hardcoded signal channel endpoint descriptions as a fallback
/// when MCP discovery is unavailable.
fn known_signal_endpoints() -> Vec<crate::obp::exploration::DiscoveredEndpoint> {
    use crate::obp::exploration::DiscoveredEndpoint;
    vec![
        DiscoveredEndpoint {
            method: "GET".into(),
            path: format!("/obp/{}/signal/channels", crate::obp::client::API_VERSION),
            operation_id: "OBPv6.0.0-getSignalChannels".into(),
            summary: "Get Signal Channels".into(),
        },
        DiscoveredEndpoint {
            method: "POST".into(),
            path: format!("/obp/{}/signal/channels/CHANNEL_NAME/messages", crate::obp::client::API_VERSION),
            operation_id: "OBPv6.0.0-publishSignalMessage".into(),
            summary: "Publish Signal Message".into(),
        },
        DiscoveredEndpoint {
            method: "GET".into(),
            path: format!("/obp/{}/signal/channels/CHANNEL_NAME/messages", crate::obp::client::API_VERSION),
            operation_id: "OBPv6.0.0-getSignalMessages".into(),
            summary: "Get Signal Messages".into(),
        },
        DiscoveredEndpoint {
            method: "GET".into(),
            path: format!("/obp/{}/signal/channels/CHANNEL_NAME/info", crate::obp::client::API_VERSION),
            operation_id: "OBPv6.0.0-getSignalChannelInfo".into(),
            summary: "Get Signal Channel Info".into(),
        },
        DiscoveredEndpoint {
            method: "DELETE".into(),
            path: format!("/obp/{}/signal/channels/CHANNEL_NAME", crate::obp::client::API_VERSION),
            operation_id: "OBPv6.0.0-deleteSignalChannel".into(),
            summary: "Delete Signal Channel".into(),
        },
    ]
}

/// Get the MAC address of the first non-loopback network interface.
/// Returns a string like "aa:bb:cc:dd:ee:ff" or "unknown".
fn get_mac_address() -> String {
    // Read from /sys/class/net on Linux
    if let Ok(entries) = std::fs::read_dir("/sys/class/net") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name == "lo" { continue; } // skip loopback
            let path = format!("/sys/class/net/{}/address", name);
            if let Ok(mac) = std::fs::read_to_string(&path) {
                let mac = mac.trim().to_string();
                if !mac.is_empty() && mac != "00:00:00:00:00:00" {
                    return mac;
                }
            }
        }
    }
    "unknown".into()
}

/// Publish a message to an OBP signal channel via direct HTTP POST.
async fn publish_signal_message_via_http(
    obp: &ObpClient,
    channel_name: &str,
    payload: &serde_json::Value,
) -> Result<serde_json::Value> {
    let path = format!(
        "/obp/{}/signal/channels/{}/messages",
        crate::obp::client::API_VERSION,
        channel_name
    );
    tracing::info!(
        channel = channel_name,
        "Signal POST {}{} payload={}",
        obp.base_url(), path, payload
    );
    let response = obp.post(&path, payload).await?;
    tracing::info!(
        channel = channel_name,
        "Signal POST response: {}",
        response
    );
    Ok(response)
}

/// Read messages from an OBP signal channel via direct HTTP GET.
async fn read_signal_messages_via_http(
    obp: &ObpClient,
    channel_name: &str,
) -> Result<serde_json::Value> {
    let path = format!(
        "/obp/{}/signal/channels/{}/messages",
        crate::obp::client::API_VERSION,
        channel_name
    );
    tracing::info!(
        channel = channel_name,
        "Signal GET {}{}",
        obp.base_url(), path
    );
    let response = obp.get(&path).await?;
    let msg_count = response.get("messages")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    tracing::info!(
        channel = channel_name,
        count = msg_count,
        "Signal GET response: {} message(s): {}",
        msg_count, response
    );
    Ok(response)
}

/// Back-off parameters for the UDP announce loop.
const INITIAL_INTERVAL_MS: u64 = 15000;
const MAX_INTERVAL_MS: u64 = 60000;
const BACKOFF_MULTIPLIER: f64 = 1.5;

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

/// Derive a fixed second from an agent name using FNV-1a hash.
/// Returns one of 0, 10, 20, 30, 40, 50 — giving 6 slots per minute,
/// each 10 seconds apart, so agents never transmit at the same time.
fn hello_second_from_name(name: &str) -> u32 {
    let mut hash: u32 = 2166136261;
    for byte in name.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    (hash % 6) * 10
}

/// Run the audio announce loop — says "hello" by broadcasting an agent-specific
/// chirp signature followed by binary data (port + capabilities).
///
/// Each broadcast consists of:
///   [agent-specific CALL chirp] + [gap] + [binary chirp message]
///
/// The CALL chirp makes each agent audibly distinctive. The binary data
/// carries the connection details (port, capabilities) so the peer can
/// establish a TCP connection immediately upon decoding.
///
/// Timing: each agent is assigned a fixed second within the minute (derived
/// from its name) so that agents never transmit at the same time. Before
/// contact, the agent transmits every minute at that second. After contact,
/// it slows to every 10 minutes. After a TCP handshake, every 40 minutes.
async fn run_announce_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
    is_transmitting: Arc<AtomicBool>,
    peer_found: Arc<AtomicBool>,
    handshake_done: Arc<AtomicBool>,
    chirp_sig: crate::audio::chirp::AgentChirpSignature,
    agent_name: String,
) {
    use crate::audio::chirp;
    use std::time::{SystemTime, UNIX_EPOCH};

    let sample_rate = crate::audio::modulator::SAMPLE_RATE;
    let section_gap = vec![0.0f32; (sample_rate as f32 * 0.08) as usize]; // 80ms gap

    let my_second = hello_second_from_name(&agent_name);
    tracing::info!(
        second = my_second,
        "TX: Agent '{}' will say hello at second {} of each minute", agent_name, my_second
    );

    let mut announce_count: u64 = 0;
    let mut minutes_between_hellos: u64 = 1; // every minute initially
    let mut made_contact = false;

    loop {
        // Wait until our designated second within the minute
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let current_second = (now_secs % 60) as u32;
        let wait_secs = if current_second <= my_second {
            my_second - current_second
        } else {
            60 - current_second + my_second
        };

        // First hello: broadcast immediately so we're audible on startup.
        // Subsequent hellos: wait for our designated second within the minute.
        let total_wait = if announce_count == 0 {
            0
        } else {
            // Wait for (minutes_between_hellos - 1) full minutes + wait_secs
            (minutes_between_hellos - 1) * 60 + wait_secs as u64
        };

        if total_wait > 0 {
            tracing::debug!(
                wait_s = total_wait,
                target_second = my_second,
                "TX: Waiting {}s for next hello slot", total_wait
            );
            tokio::time::sleep(std::time::Duration::from_secs(total_wait)).await;
        }

        announce_count += 1;

        // Check if a peer was found — send a quick reply then slow down
        if peer_found.swap(false, Ordering::AcqRel) {
            if !made_contact {
                made_contact = true;
                minutes_between_hellos = 10;
                tracing::info!(
                    "TX: Heard a peer! Will slow hellos to every {} minutes",
                    minutes_between_hellos
                );
            }
        }

        // After TCP handshake: slow down even more
        if handshake_done.swap(false, Ordering::AcqRel) {
            minutes_between_hellos = 40;
            tracing::info!(
                "TX: Handshake done, slowing hellos to every {} minutes",
                minutes_between_hellos
            );
        }

        // Progressive amplitude ramp
        let amplitude = chirp::tx_amplitude(announce_count);

        // Build the "hello" message: signature chirp + binary data
        let mgr = manager.lock().await;
        let port = mgr.config().agent_address
            .rsplit(':')
            .next()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(0);
        let caps = mgr.config().capabilities.0;
        drop(mgr);

        let caps_desc = crate::protocol::message::Capabilities(caps).describe();

        let binary_msg = chirp::encode_chirp_message(port, caps, sample_rate);
        let call_samples = chirp_sig.call_chirp(sample_rate, amplitude);

        let mut samples_to_send = Vec::with_capacity(
            call_samples.len() + section_gap.len() + binary_msg.len(),
        );
        samples_to_send.extend_from_slice(&call_samples);
        samples_to_send.extend_from_slice(&section_gap);
        samples_to_send.extend_from_slice(&binary_msg);

        let total_duration_secs = samples_to_send.len() as f32 / sample_rate as f32;

        tracing::info!(
            announce = announce_count,
            port,
            caps = %caps_desc,
            amplitude = format!("{:.2}", amplitude),
            duration_ms = format!("{:.0}", total_duration_secs * 1000.0),
            every_mins = minutes_between_hellos,
            second = my_second,
            "TX: Saying hello at second {} (port={} caps=0x{:02X} [{}])",
            my_second, port, caps, caps_desc
        );

        // Mute RX while transmitting so the mic doesn't pick up our own signal
        is_transmitting.store(true, Ordering::Release);

        if let Err(e) = engine.send_samples(samples_to_send) {
            tracing::error!("TX: Failed to send hello: {}", e);
        }

        // Wait for the tone to finish playing, plus a small margin for echo decay
        let tx_duration_ms = (total_duration_secs * 1000.0) as u64 + 200;
        tokio::time::sleep(std::time::Duration::from_millis(tx_duration_ms)).await;

        is_transmitting.store(false, Ordering::Release);
    }
}

/// Run the audio receive loop — listens for "hello" binary chirp data.
///
/// Continuously buffers mic audio and attempts to decode binary chirp
/// messages. When a message is decoded, the peer's port and capabilities
/// are extracted and registered with the discovery manager.
///
/// Self-echo rejection: if the decoded port matches our own listening port,
/// the message is our own echo and is discarded.
async fn run_receive_loop(
    manager: Arc<tokio::sync::Mutex<DiscoveryManager>>,
    engine: Arc<AudioEngine>,
    is_transmitting: Arc<AtomicBool>,
    peer_found: Arc<AtomicBool>,
    own_port: u16,
) {
    use crate::audio::chirp;

    let sample_rate = crate::audio::modulator::SAMPLE_RATE;
    let mut sample_buffer = Vec::new();
    let mut peak: f32 = 0.0;
    let mut last_peak_log = std::time::Instant::now();
    let mut decode_attempts: u64 = 0;
    let mut hellos_heard: u64 = 0;

    // Post-TX cooldown: suppress decoding for 300ms after own TX finishes
    let mut post_tx_cooldown = std::time::Instant::now() - std::time::Duration::from_secs(10);
    let mut was_transmitting = false;

    // Minimum buffer: a binary chirp message is 44 × 100ms ≈ 4.4s
    let min_msg_samples = (sample_rate as f32 * 4.0) as usize;

    tracing::info!(
        own_port,
        "RX: Listening for hello messages (binary chirp data in 1.5-2.5kHz band)"
    );

    loop {
        // Track peak level and buffer samples (only when not transmitting)
        while let Some(chunk) = engine.try_recv_samples() {
            for &s in &chunk {
                let abs = s.abs();
                if abs > peak {
                    peak = abs;
                }
            }
            if !is_transmitting.load(Ordering::Acquire) {
                sample_buffer.extend_from_slice(&chunk);
            }
        }

        // Log audio stats every 10 seconds
        if last_peak_log.elapsed() > std::time::Duration::from_secs(10) {
            tracing::info!(
                peak = format!("{:.4}", peak),
                buffer = sample_buffer.len(),
                decodes = decode_attempts,
                hellos = hellos_heard,
                "RX: Audio stats (last 10s)"
            );
            peak = 0.0;
            last_peak_log = std::time::Instant::now();
        }

        // If we're currently transmitting, clear buffer and wait
        let currently_transmitting = is_transmitting.load(Ordering::Acquire);
        if currently_transmitting {
            was_transmitting = true;
            sample_buffer.clear();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            continue;
        }
        // Post-TX cooldown: discard samples from the transition period
        if was_transmitting {
            was_transmitting = false;
            post_tx_cooldown = std::time::Instant::now();
            sample_buffer.clear();
            tracing::debug!("RX: Post-TX cooldown (300ms)");
        }

        // Don't try decoding during cooldown
        if post_tx_cooldown.elapsed() < std::time::Duration::from_millis(300) {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            continue;
        }

        // Try to decode a binary chirp message when we have enough audio
        if sample_buffer.len() >= min_msg_samples {
            decode_attempts += 1;
            if let Some((port, caps)) = chirp::decode_chirp_message_sweep(&sample_buffer, sample_rate) {
                // Self-echo check: if the decoded port is our own, discard
                if port == own_port {
                    tracing::debug!("RX: Decoded own port {} → self-echo, ignoring", port);
                    sample_buffer.clear();
                    continue;
                }

                hellos_heard += 1;
                let address = format!("127.0.0.1:{}", port);
                let caps_desc = crate::protocol::message::Capabilities(caps).describe();
                tracing::info!(
                    port,
                    capabilities = caps,
                    caps_desc = %caps_desc,
                    hellos = hellos_heard,
                    "RX: Heard hello! Peer at {} — port={} caps=0x{:02X} [{}]",
                    address, port, caps, caps_desc
                );

                // Register the peer
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
                peer_found.store(true, Ordering::Release);
                sample_buffer.clear();
                continue;
            } else {
                // Trim old samples to avoid unbounded growth
                if sample_buffer.len() > min_msg_samples * 2 {
                    let trim = min_msg_samples / 2;
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
