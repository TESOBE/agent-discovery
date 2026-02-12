/// OBP Dynamic Entity definitions for agent registry and messaging.
///
/// Uses the v6.0.0 snake_case format with entity_name and example fields.

use anyhow::Result;
use serde_json::{json, Value};

use crate::mcp::client::StdioMcpClient;
use crate::obp::client::ObpClient;

/// Dynamic entity definition for the agent registry.
/// Agents register themselves here so others can find them.
pub fn agent_registry_entity() -> Value {
    json!({
        "entity_name": "AgentRegistryEntry",
        "example": {
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "agent_name": "agent-alpha",
            "address": "192.168.1.100:9000",
            "capabilities": 15,
            "status": "online",
            "last_seen": "2024-01-01T00:00:00Z",
            "metadata": "{}"
        }
    })
}

/// Dynamic entity definition for agent-to-agent messages.
/// Agents post messages here for other agents to read.
pub fn agent_messages_entity() -> Value {
    json!({
        "entity_name": "AgentMessage",
        "example": {
            "from_agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "to_agent_id": "660e8400-e29b-41d4-a716-446655440001",
            "message_type": "negotiation",
            "payload": "{}",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    })
}

/// Set up dynamic entities on the OBP instance via MCP.
pub async fn setup_entities_via_mcp(
    mcp_client: &mut StdioMcpClient,
    bank_id: &str,
) -> Result<()> {
    tracing::info!("Setting up OBP dynamic entities via MCP");

    // Create agent registry entity
    let registry_entity = agent_registry_entity();
    let result = mcp_client
        .call_tool(
            "call_obp_api",
            json!({
                "endpoint_id": "OBPv4.0.0-createBankLevelDynamicEntity",
                "path_params": {"BANK_ID": bank_id},
                "body": registry_entity
            }),
        )
        .await;

    match result {
        Ok(val) => tracing::info!("Agent registry entity created: {}", val),
        Err(e) => tracing::warn!("Agent registry entity setup: {} (may already exist)", e),
    }

    // Create agent messages entity
    let messages_entity = agent_messages_entity();
    let result = mcp_client
        .call_tool(
            "call_obp_api",
            json!({
                "endpoint_id": "OBPv4.0.0-createBankLevelDynamicEntity",
                "path_params": {"BANK_ID": bank_id},
                "body": messages_entity
            }),
        )
        .await;

    match result {
        Ok(val) => tracing::info!("Agent messages entity created: {}", val),
        Err(e) => tracing::warn!("Agent messages entity setup: {} (may already exist)", e),
    }

    Ok(())
}

/// Set up dynamic entities on the OBP instance via direct HTTP (fallback).
pub async fn setup_entities_via_http(obp_client: &ObpClient) -> Result<()> {
    tracing::info!("Setting up OBP dynamic entities via HTTP");
    let bank_id = obp_client.bank_id();

    let registry_entity = agent_registry_entity();
    let path = format!("/management/banks/{}/dynamic-entities", bank_id);
    match obp_client.post(&path, &registry_entity).await {
        Ok(val) => tracing::info!("Agent registry entity created: {}", val),
        Err(e) => tracing::warn!("Agent registry entity setup: {} (may already exist)", e),
    }

    let messages_entity = agent_messages_entity();
    match obp_client.post(&path, &messages_entity).await {
        Ok(val) => tracing::info!("Agent messages entity created: {}", val),
        Err(e) => tracing::warn!("Agent messages entity setup: {} (may already exist)", e),
    }

    Ok(())
}

/// Register this agent in the OBP agent registry via MCP.
pub async fn register_agent_via_mcp(
    mcp_client: &mut StdioMcpClient,
    bank_id: &str,
    agent_id: &str,
    agent_name: &str,
    address: &str,
    capabilities: u8,
) -> Result<Value> {
    let body = json!({
        "agent_id": agent_id,
        "agent_name": agent_name,
        "address": address,
        "capabilities": capabilities,
        "status": "online",
        "last_seen": chrono_now(),
        "metadata": "{}"
    });

    mcp_client
        .call_tool(
            "call_obp_api",
            json!({
                "endpoint_id": "OBPv4.0.0-createBankLevelDynamicEntity_AgentRegistryEntry",
                "path_params": {"BANK_ID": bank_id},
                "body": body
            }),
        )
        .await
}

/// Send a message to another agent via OBP dynamic entities.
pub async fn send_message_via_mcp(
    mcp_client: &mut StdioMcpClient,
    bank_id: &str,
    from_agent_id: &str,
    to_agent_id: &str,
    message_type: &str,
    payload: &Value,
) -> Result<Value> {
    let body = json!({
        "from_agent_id": from_agent_id,
        "to_agent_id": to_agent_id,
        "message_type": message_type,
        "payload": serde_json::to_string(payload)?,
        "timestamp": chrono_now()
    });

    mcp_client
        .call_tool(
            "call_obp_api",
            json!({
                "endpoint_id": "OBPv4.0.0-createBankLevelDynamicEntity_AgentMessage",
                "path_params": {"BANK_ID": bank_id},
                "body": body
            }),
        )
        .await
}

/// Simple ISO-8601 timestamp without chrono crate.
fn chrono_now() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple UTC timestamp (not perfect but avoids chrono dependency)
    format!("{}Z", secs)
}
