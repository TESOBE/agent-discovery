/// MCP client: connects to a running MCP server via HTTP (streamable HTTP transport).
/// Communicates via JSON-RPC.

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::Config;

/// Represents an MCP tool discovered from the server.
#[derive(Debug, Clone)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// MCP client that connects to a running MCP server via HTTP.
pub struct McpClient {
    server_url: String,
    http_client: reqwest::Client,
    request_id: u64,
    tools: Vec<McpTool>,
}

impl McpClient {
    /// Connect to a running MCP server. Requires `OBP_MCP_SERVER_URL` to be set.
    pub async fn new(config: &Config) -> Result<Self> {
        let server_url = config
            .obp_mcp_server_url
            .as_ref()
            .context("OBP_MCP_SERVER_URL not set - point it at your running MCP server")?
            .trim_end_matches('/')
            .to_string();

        tracing::info!(url = %server_url, "Connecting to MCP server");

        let mut client = Self {
            server_url,
            http_client: reqwest::Client::new(),
            request_id: 0,
            tools: Vec::new(),
        };

        client.initialize().await?;
        client.discover_tools().await?;

        Ok(client)
    }

    async fn initialize(&mut self) -> Result<()> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "agent-discovery",
                    "version": "0.1.0"
                }
            }
        });

        let response = self.send_request(&request).await?;
        tracing::info!("MCP initialized: {}", response);

        // Send initialized notification
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        self.send_notification(&notification).await?;

        Ok(())
    }

    async fn discover_tools(&mut self) -> Result<()> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "tools/list",
            "params": {}
        });

        let response = self.send_request(&request).await?;

        if let Some(result) = response.get("result") {
            if let Some(tools) = result.get("tools").and_then(|t| t.as_array()) {
                self.tools = tools
                    .iter()
                    .filter_map(|t| {
                        Some(McpTool {
                            name: t.get("name")?.as_str()?.to_string(),
                            description: t
                                .get("description")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .to_string(),
                            input_schema: t
                                .get("inputSchema")
                                .cloned()
                                .unwrap_or(Value::Object(serde_json::Map::new())),
                        })
                    })
                    .collect();

                tracing::info!("Discovered {} MCP tools", self.tools.len());
                for tool in &self.tools {
                    tracing::debug!("  Tool: {} - {}", tool.name, tool.description);
                }
            }
        }

        Ok(())
    }

    /// Call an MCP tool by name with the given arguments.
    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        });

        let response = self.send_request(&request).await?;

        if let Some(error) = response.get("error") {
            anyhow::bail!("MCP tool error: {}", error);
        }

        Ok(response
            .get("result")
            .cloned()
            .unwrap_or(Value::Null))
    }

    /// Get the list of discovered tools.
    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    /// Convert MCP tools to Claude tool-use format.
    pub fn tools_as_claude_tools(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                })
            })
            .collect()
    }

    async fn send_request(&mut self, msg: &Value) -> Result<Value> {
        let response = self
            .http_client
            .post(&self.server_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(msg)
            .send()
            .await
            .context("Failed to send MCP request")?;

        let status = response.status();
        let body = response.text().await.context("Failed to read MCP response")?;

        if !status.is_success() {
            anyhow::bail!("MCP HTTP error ({}): {}", status, body);
        }

        serde_json::from_str(&body).context("Failed to parse MCP JSON-RPC response")
    }

    async fn send_notification(&mut self, msg: &Value) -> Result<()> {
        let _ = self
            .http_client
            .post(&self.server_url)
            .header("Content-Type", "application/json")
            .json(msg)
            .send()
            .await;
        Ok(())
    }

    fn next_id(&mut self) -> u64 {
        self.request_id += 1;
        self.request_id
    }
}
