/// Exploration protocol messages exchanged over TCP after handshake.
///
/// Two agents collaboratively discover how OBP dynamic entities work,
/// starting from zero: find management endpoints, learn the schema,
/// create an entity, discover auto-generated CRUD, and test it.
///
/// Phase 7 extends the protocol to discover and verify OBP signal channels
/// for ephemeral agent-to-agent messaging.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single discovered CRUD endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredEndpoint {
    pub method: String,
    pub path: String,
    pub operation_id: String,
    pub summary: String,
}

/// All messages exchanged during the post-handshake OBP exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExplorationMsg {
    // --- Phase 1: Discover management endpoints ---
    ExploreStart {
        agent_name: String,
        /// The OBP API base URL used by the initiating agent.
        #[serde(default)]
        obp_api_base_url: String,
    },
    ExploreAck {
        agent_name: String,
        /// The OBP API base URL used by the responding agent.
        #[serde(default)]
        obp_api_base_url: String,
    },
    FoundManagementEndpoint {
        endpoint_id: String,
        method: String,
        path: String,
        summary: String,
    },

    // --- Phase 2: Discover entity creation format ---
    FoundEntityFormat {
        endpoint_id: String,
        schema_description: String,
    },

    // --- Phase 3: Create the entity ---
    EntityCreated {
        entity_name: String,
        request_body: Value,
        response: Value,
    },

    // --- Phase 4: Discover auto-generated CRUD ---
    DiscoveredCrudEndpoints {
        entity_name: String,
        endpoints: Vec<DiscoveredEndpoint>,
    },
    EndpointsConfirmed {
        entity_name: String,
        confirmed: bool,
        our_endpoints: Vec<DiscoveredEndpoint>,
    },

    // --- Phase 5: Test record CRUD ---
    RecordCreated {
        entity_name: String,
        endpoint_used: String,
        record: Value,
    },
    RecordVerified {
        entity_name: String,
        record: Value,
        matches: bool,
    },

    // --- Phase 7: Signal channel discovery and verification ---
    FoundSignalEndpoints {
        endpoints: Vec<DiscoveredEndpoint>,
    },
    SignalChannelTested {
        channel_name: String,
        message_id: String,
        payload: Value,
    },
    SignalChannelVerified {
        channel_name: String,
        verified: bool,
    },

    // --- Diagnostics ---
    McpDiagnosis {
        findings: Vec<String>,
    },

    // --- General ---
    Acknowledged {
        phase: String,
    },
    ExplorationComplete {
        summary: String,
        success: bool,
    },
    ExplorationError {
        phase: String,
        error: String,
    },
}
