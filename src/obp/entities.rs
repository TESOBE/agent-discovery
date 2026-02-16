/// OBP Dynamic Entity helpers for agent discovery.
///
/// Entity definitions are NO LONGER created at startup. Instead, agents
/// discover the management endpoints and create entities collaboratively
/// during the post-handshake exploration protocol.
///
/// CRUD endpoints are discovered via HATEOAS: after creating the entity,
/// the agent calls `GET /obp/v6.0.0/my/dynamic-entities` and follows
/// the `_links` to find available operations.

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
        "has_personal_entity": true,
        "has_public_access": true,
        "has_community_access": true,
        "personal_requires_role": true,
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

/// Discover CRUD endpoints for a personal dynamic entity via HATEOAS.
///
/// Calls `GET /obp/{API_VERSION}/my/dynamic-entities` and inspects the
/// `_links` for entries whose href contains `/{entity_name}`.
/// Returns the discovered endpoints (method + path pairs).
pub async fn discover_entity_endpoints_via_http(
    obp_client: &ObpClient,
    entity_name: &str,
) -> Result<Vec<DiscoveredEndpoint>> {
    let path = format!(
        "/obp/{}/my/dynamic-entities",
        API_VERSION
    );
    tracing::info!("Querying personal dynamic entities for '{}' CRUD links", entity_name);

    let response = obp_client.get(&path).await?;
    tracing::info!("Personal dynamic entities response: {}", response);

    let mut endpoints = Vec::new();
    let search_fragment = format!("/{}", entity_name);

    // Look for _links in the response — could be top-level or nested
    // inside a matching entity object
    extract_links_from_value(&response, &search_fragment, &mut endpoints);

    tracing::info!(
        entity = entity_name,
        count = endpoints.len(),
        "Discovered {} CRUD endpoints via _links",
        endpoints.len()
    );
    for ep in &endpoints {
        tracing::info!("  {} {} ({})", ep.method, ep.path, ep.summary);
    }

    Ok(endpoints)
}

/// Recursively extract `_links` entries whose href contains the search fragment.
fn extract_links_from_value(
    val: &Value,
    search_fragment: &str,
    endpoints: &mut Vec<DiscoveredEndpoint>,
) {
    match val {
        Value::Object(map) => {
            // Check if this object has _links
            if let Some(links) = map.get("_links") {
                extract_links_entries(links, search_fragment, endpoints);
            }
            // Recurse into all values
            for (_, v) in map {
                extract_links_from_value(v, search_fragment, endpoints);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                extract_links_from_value(item, search_fragment, endpoints);
            }
        }
        _ => {}
    }
}

/// Parse _links and extract endpoints matching the search fragment.
///
/// OBP uses the structure `{"_links": {"related": [{"href": "...", "method": "...", "rel": "..."}]}}`.
/// We handle both direct arrays and objects with named link groups (e.g. "related").
fn extract_links_entries(
    links: &Value,
    search_fragment: &str,
    endpoints: &mut Vec<DiscoveredEndpoint>,
) {
    match links {
        Value::Array(arr) => {
            // Direct array of link objects
            for link in arr {
                try_extract_link(link, search_fragment, endpoints);
            }
        }
        Value::Object(map) => {
            // Object with named link groups, e.g. {"related": [...]}
            for (_key, v) in map {
                if let Value::Array(arr) = v {
                    for link in arr {
                        try_extract_link(link, search_fragment, endpoints);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Try to extract a single link entry if its href matches the search fragment.
fn try_extract_link(
    link: &Value,
    search_fragment: &str,
    endpoints: &mut Vec<DiscoveredEndpoint>,
) {
    let href = link.get("href").and_then(|h| h.as_str()).unwrap_or("");
    if !href.contains(search_fragment) {
        return;
    }
    let method = link.get("method").and_then(|m| m.as_str()).unwrap_or("GET").to_string();
    let rel = link.get("rel").and_then(|r| r.as_str()).unwrap_or("").to_string();
    // Avoid duplicates
    let already_have = endpoints.iter().any(|e| e.path == href && e.method == method);
    if !already_have {
        endpoints.push(DiscoveredEndpoint {
            method,
            path: href.to_string(),
            operation_id: rel.clone(),
            summary: rel,
        });
    }
}

/// Find the CRUD endpoint path for creating a record, from discovered endpoints.
/// Returns the href for the POST endpoint, if found.
pub fn find_create_record_path(endpoints: &[DiscoveredEndpoint]) -> Option<&str> {
    endpoints.iter()
        .find(|ep| ep.method.eq_ignore_ascii_case("POST"))
        .map(|ep| ep.path.as_str())
}

/// Find the CRUD endpoint path for listing records, from discovered endpoints.
/// Returns the href for the GET endpoint, if found.
pub fn find_list_records_path(endpoints: &[DiscoveredEndpoint]) -> Option<&str> {
    endpoints.iter()
        .find(|ep| ep.method.eq_ignore_ascii_case("GET"))
        .map(|ep| ep.path.as_str())
}

/// Record a handshake event using a discovered CRUD endpoint path.
pub async fn record_handshake_via_http(
    obp_client: &ObpClient,
    create_path: &str,
    agent_id: &str,
    agent_name: &str,
    peer_id: &str,
    peer_address: &str,
    discovery_method: &str,
) -> Result<Value> {
    let body = json!({
        "agent_id": agent_id,
        "agent_name": agent_name,
        "peer_id": peer_id,
        "peer_address": peer_address,
        "discovery_method": discovery_method,
        "timestamp": iso_now()
    });

    tracing::info!(
        agent = agent_name,
        peer = peer_id,
        "Recording handshake to OBP: POST {}",
        create_path
    );

    obp_client.post(create_path, &body).await
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
