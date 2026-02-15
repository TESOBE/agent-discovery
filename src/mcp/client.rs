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

/// Diagnose why an MCP connection failed. Runs a series of checks against
/// the configured URL and returns a human-readable report of findings.
pub async fn diagnose_mcp(config: &Config) -> Vec<String> {
    let mut findings: Vec<String> = Vec::new();

    // 1. Check if URL is configured
    let url_str = match config.obp_mcp_server_url.as_deref() {
        Some(url) if !url.is_empty() => {
            findings.push(format!("MCP URL configured: {}", url));
            url.to_string()
        }
        _ => {
            findings.push("OBP_MCP_SERVER_URL is not set or empty".into());
            findings.push("Hint: set OBP_MCP_SERVER_URL to point at your running MCP server (e.g. http://localhost:9100/mcp)".into());
            return findings;
        }
    };

    // 2. Parse the URL to extract host and port
    let (host, port) = match parse_host_port(&url_str) {
        Some(hp) => {
            findings.push(format!("Parsed host={}, port={}", hp.0, hp.1));
            hp
        }
        None => {
            findings.push(format!("Could not parse host/port from URL: {}", url_str));
            return findings;
        }
    };

    // 3. Check TCP connectivity (can we reach the port at all?)
    let addr = format!("{}:{}", host, port);
    match std::net::TcpStream::connect_timeout(
        &addr.parse().unwrap_or_else(|_| {
            // If addr doesn't parse as SocketAddr, try resolving
            use std::net::ToSocketAddrs;
            addr.to_socket_addrs()
                .ok()
                .and_then(|mut addrs| addrs.next())
                .unwrap_or_else(|| "127.0.0.1:0".parse().unwrap())
        }),
        std::time::Duration::from_secs(5),
    ) {
        Ok(_) => {
            findings.push(format!("TCP connection to {} succeeded", addr));
        }
        Err(e) => {
            findings.push(format!("TCP connection to {} FAILED: {}", addr, e));
            match e.kind() {
                std::io::ErrorKind::ConnectionRefused => {
                    findings.push(format!(
                        "Hint: connection refused — is the MCP server running? Try: python -m obp_mcp_server or check that port {} is listening",
                        port
                    ));
                }
                std::io::ErrorKind::TimedOut => {
                    findings.push("Hint: connection timed out — check firewall rules or if the host is reachable".into());
                }
                _ => {
                    findings.push(format!("Hint: network error ({}). Check that {} is reachable.", e.kind(), addr));
                }
            }
            return findings;
        }
    }

    // 4. Try an HTTP request to see if it speaks HTTP
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let probe_body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-probe", "version": "0.1.0"}
        }
    });

    match http_client
        .post(&url_str)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .json(&probe_body)
        .send()
        .await
    {
        Ok(response) => {
            let status = response.status();
            findings.push(format!("HTTP POST {} returned status {}", url_str, status));

            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let preview = &body[..body.len().min(300)];
                findings.push(format!("Response body: {}", preview));

                if status.as_u16() == 404 {
                    findings.push("Hint: 404 Not Found — check if the URL path is correct (e.g. /mcp)".into());
                } else if status.as_u16() == 405 {
                    findings.push("Hint: 405 Method Not Allowed — server exists but may not accept POST at this path".into());
                } else if status.is_server_error() {
                    findings.push("Hint: server error — the MCP server may be crashing. Check its logs.".into());
                }
                return findings;
            }

            // 5. Check if the response is valid JSON-RPC
            let body = response.text().await.unwrap_or_default();
            match serde_json::from_str::<Value>(&body) {
                Ok(json) => {
                    if json.get("jsonrpc").is_some() {
                        findings.push("Server speaks JSON-RPC 2.0 — MCP protocol looks correct".into());
                        if let Some(result) = json.get("result") {
                            if let Some(name) = result.get("serverInfo").and_then(|s| s.get("name")) {
                                findings.push(format!("MCP server identifies as: {}", name));
                            }
                            findings.push("MCP initialize handshake succeeded — server appears healthy".into());
                            findings.push("Hint: if McpClient::new() still failed, the error may be in tools/list. Try restarting the MCP server.".into());
                        }
                        if let Some(error) = json.get("error") {
                            findings.push(format!("JSON-RPC error from server: {}", error));
                        }
                    } else {
                        findings.push(format!("Response is JSON but not JSON-RPC: {}", &body[..body.len().min(200)]));
                        findings.push("Hint: the server may not be an MCP server. Check the URL.".into());
                    }
                }
                Err(e) => {
                    findings.push(format!("Response is not valid JSON: {}", e));
                    findings.push(format!("Body preview: {}", &body[..body.len().min(200)]));
                    findings.push("Hint: server is responding but not with JSON — likely not an MCP server at this URL".into());
                }
            }
        }
        Err(e) => {
            findings.push(format!("HTTP request to {} FAILED: {}", url_str, e));
            if e.is_timeout() {
                findings.push("Hint: request timed out — server accepted TCP but didn't respond to HTTP in time".into());
            } else if e.is_connect() {
                findings.push("Hint: connection error at HTTP level".into());
            } else {
                findings.push(format!("Hint: reqwest error: {}", e));
            }
        }
    }

    findings
}

/// Extract host and port from a URL string.
fn parse_host_port(url: &str) -> Option<(String, u16)> {
    // Simple parser: handle http://host:port/path
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);

    let host_port = without_scheme.split('/').next()?;

    if let Some(colon_pos) = host_port.rfind(':') {
        let host = &host_port[..colon_pos];
        let port = host_port[colon_pos + 1..].parse::<u16>().ok()?;
        // Treat 0.0.0.0 as localhost for connection probing
        let host = if host == "0.0.0.0" { "127.0.0.1" } else { host };
        Some((host.to_string(), port))
    } else {
        let default_port = if url.starts_with("https://") { 443 } else { 80 };
        Some((host_port.to_string(), default_port))
    }
}
