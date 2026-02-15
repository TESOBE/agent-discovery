/// OBP Dynamic Entity helpers for agent discovery.
///
/// Entity definitions are NO LONGER created at startup. Instead, agents
/// discover the management endpoints and create entities collaboratively
/// during the post-handshake exploration protocol.

use anyhow::Result;
use serde_json::{json, Value};

use crate::obp::client::{ObpClient, API_VERSION};
use crate::obp::exploration::DiscoveredEndpoint;

/// The entity name used for handshake records.
pub const HANDSHAKE_ENTITY: &str = "agent_handshake";

/// Dynamic entity definition for recording agent handshakes.
pub fn agent_handshake_entity() -> Value {
    json!({
        "entity_name": HANDSHAKE_ENTITY,
        "has_personal_entity": false,
        "schema": {
            "description": "Records of agent-to-agent discovery handshakes",
            "required": ["agent_id", "peer_id"],
            "properties": {
                "agent_id": {
                    "type": "string",
                    "example": "550e8400-e29b-41d4-a716-446655440000",
                    "description": "UUID of the agent recording this handshake"
                },
                "agent_name": {
                    "type": "string",
                    "example": "agent-alpha",
                    "description": "Human-readable name of the agent"
                },
                "peer_id": {
                    "type": "string",
                    "example": "660e8400-e29b-41d4-a716-446655440001",
                    "description": "UUID of the discovered peer"
                },
                "peer_address": {
                    "type": "string",
                    "example": "192.168.1.42:9000",
                    "description": "Network address of the peer"
                },
                "discovery_method": {
                    "type": "string",
                    "example": "audio-chirp",
                    "description": "How the peer was discovered (audio-chirp, udp, etc.)"
                },
                "timestamp": {
                    "type": "string",
                    "example": "2024-01-01T00:00:00Z",
                    "description": "When the handshake occurred (epoch seconds)"
                }
            }
        }
    })
}

/// Query resource-docs to discover auto-generated CRUD endpoints for a dynamic entity.
///
/// Fetches `GET /resource-docs/{API_VERSION}/obp` and filters for entries
/// whose `request_url` contains `/{entity_name}`.
pub async fn discover_entity_endpoints_via_http(
    obp_client: &ObpClient,
    entity_name: &str,
) -> Result<Vec<DiscoveredEndpoint>> {
    let path = format!(
        "/resource-docs/{}/obp",
        API_VERSION
    );
    tracing::info!("Querying resource-docs for entity '{}' endpoints", entity_name);

    let response = obp_client.get(&path).await?;

    let mut endpoints = Vec::new();
    let search_fragment = format!("/{}", entity_name);

    if let Some(resource_docs) = response.get("resource_docs").and_then(|r| r.as_array()) {
        for doc in resource_docs {
            let request_url = doc.get("request_url").and_then(|u| u.as_str()).unwrap_or("");
            if request_url.contains(&search_fragment) {
                let method = doc.get("request_verb").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let operation_id = doc.get("operation_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let summary = doc.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string();
                endpoints.push(DiscoveredEndpoint {
                    method,
                    path: request_url.to_string(),
                    operation_id,
                    summary,
                });
            }
        }
    }

    tracing::info!(
        entity = entity_name,
        count = endpoints.len(),
        "Discovered {} CRUD endpoints via resource-docs",
        endpoints.len()
    );
    for ep in &endpoints {
        tracing::info!("  {} {} ({})", ep.method, ep.path, ep.operation_id);
    }

    Ok(endpoints)
}

/// Record a handshake event in the OBP agent_handshake entity via direct HTTP.
pub async fn record_handshake_via_http(
    obp_client: &ObpClient,
    agent_id: &str,
    agent_name: &str,
    peer_id: &str,
    peer_address: &str,
    discovery_method: &str,
) -> Result<Value> {
    let bank_id = obp_client.bank_id().to_string();
    let body = json!({
        "agent_id": agent_id,
        "agent_name": agent_name,
        "peer_id": peer_id,
        "peer_address": peer_address,
        "discovery_method": discovery_method,
        "timestamp": iso_now()
    });

    let path = format!("/banks/{}/{}", bank_id, HANDSHAKE_ENTITY);
    tracing::info!(
        agent = agent_name,
        peer = peer_id,
        "Recording handshake to OBP: POST {}",
        path
    );

    obp_client.post(&path, &body).await
}

/// Simple ISO-8601-ish timestamp without chrono crate.
pub fn iso_now() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    format!("{}Z", secs)
}
