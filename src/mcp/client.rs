/// MCP client: spawns the OBP MCP server as a subprocess (stdio transport)
/// and communicates via JSON-RPC over stdin/stdout.

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

/// MCP client that communicates with the OBP MCP server via stdio JSON-RPC.
pub struct StdioMcpClient {
    child: tokio::process::Child,
    stdin: tokio::io::BufWriter<tokio::process::ChildStdin>,
    stdout: tokio::io::BufReader<tokio::process::ChildStdout>,
    request_id: u64,
    tools: Vec<McpTool>,
}

impl StdioMcpClient {
    /// Spawn the OBP MCP server subprocess and initialize the MCP connection.
    pub async fn new(config: &Config) -> Result<Self> {
        use tokio::io::{BufReader, BufWriter};
        use tokio::process::Command;

        tracing::info!(
            command = %config.obp_mcp_server_command,
            args = ?config.obp_mcp_server_args,
            "Spawning OBP MCP server"
        );

        let mut child = Command::new(&config.obp_mcp_server_command)
            .args(&config.obp_mcp_server_args)
            .env("OBP_BASE_URL", &config.obp_base_url)
            .env("OBP_API_VERSION", &config.obp_api_version)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .context("Failed to spawn OBP MCP server")?;

        let stdin = child.stdin.take().context("No stdin on child")?;
        let stdout = child.stdout.take().context("No stdout on child")?;

        let mut client = Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            request_id: 0,
            tools: Vec::new(),
        };

        // Initialize the MCP connection
        client.initialize().await?;
        client.discover_tools().await?;

        Ok(client)
    }

    /// Send the MCP initialize request.
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

        self.send_message(&request).await?;
        let response = self.read_response().await?;
        tracing::info!("MCP initialized: {}", response);

        // Send initialized notification
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        self.send_message(&notification).await?;

        Ok(())
    }

    /// Discover available tools from the MCP server.
    async fn discover_tools(&mut self) -> Result<()> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_id(),
            "method": "tools/list",
            "params": {}
        });

        self.send_message(&request).await?;
        let response = self.read_response().await?;

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

        self.send_message(&request).await?;
        let response = self.read_response().await?;

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

    async fn send_message(&mut self, msg: &Value) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        let serialized = serde_json::to_string(msg)?;
        self.stdin
            .write_all(serialized.as_bytes())
            .await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_response(&mut self) -> Result<Value> {
        use tokio::io::AsyncBufReadExt;
        let mut line = String::new();
        self.stdout.read_line(&mut line).await?;
        let response: Value = serde_json::from_str(line.trim())?;
        Ok(response)
    }

    fn next_id(&mut self) -> u64 {
        self.request_id += 1;
        self.request_id
    }
}

impl Drop for StdioMcpClient {
    fn drop(&mut self) {
        // Try to kill the child process
        let _ = self.child.start_kill();
    }
}
