/// Claude Sonnet API client with tool-use support for agent negotiation.
///
/// Implements the tool-use loop: send message to Claude, if Claude requests tool calls,
/// execute them via MCP, return results, and let Claude continue.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Config;
use crate::discovery::peer::Peer;
use crate::mcp::client::McpClient;

const CLAUDE_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// A message in the Claude conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

/// Message content can be a simple string or structured content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block in a Claude message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// Response from the Claude Messages API.
#[derive(Debug, Deserialize)]
pub struct ClaudeResponse {
    pub id: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub model: String,
    pub usage: Option<Value>,
}

/// The negotiation result from Claude.
#[derive(Debug, Clone)]
pub struct NegotiationResult {
    pub chosen_channel: String, // "tcp", "obp", "http", etc.
    pub details: Value,         // channel-specific configuration
    pub reasoning: String,      // Claude's reasoning
}

/// Claude API client for negotiation.
pub struct ClaudeNegotiator {
    api_key: String,
    model: String,
    http_client: reqwest::Client,
}

impl ClaudeNegotiator {
    pub fn new(config: &Config) -> Self {
        Self {
            api_key: config.claude_api_key.clone(),
            model: config.claude_model.clone(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Negotiate communication with a discovered peer.
    ///
    /// Claude receives context about both agents and available tools (from MCP),
    /// then decides how they should communicate.
    pub async fn negotiate(
        &self,
        our_agent_id: &str,
        our_address: &str,
        peer: &Peer,
        mut mcp_client: Option<&mut McpClient>,
    ) -> Result<NegotiationResult> {
        let system_prompt = format!(
            r#"You are an autonomous agent negotiation system. Two AI agents have discovered each other via audio FSK tones and need to decide how to communicate going forward.

Your agent: ID={our_agent_id}, Address={our_address}
Discovered peer: ID={peer_id}, Address={peer_address}, Capabilities={capabilities:?}

Your task:
1. Analyze both agents' capabilities
2. If OBP tools are available, you may use them to set up shared storage (dynamic entities) for the agents to communicate through
3. Decide the best communication channel (TCP direct, OBP shared storage, or HTTP)
4. Return your decision as a JSON object with fields: channel (string), config (object), reasoning (string)

Available capabilities:
- TCP (bit 0): Direct TCP connection between agents
- HTTP (bit 1): HTTP-based communication
- OBP (bit 2): Open Bank Project API for shared storage via dynamic entities
- Claude (bit 3): Both agents support Claude-mediated negotiation

Prefer TCP for low-latency direct communication when both support it.
Use OBP dynamic entities as a shared bulletin board when direct connection isn't possible.
"#,
            our_agent_id = our_agent_id,
            our_address = our_address,
            peer_id = peer.agent_id,
            peer_address = peer.address,
            capabilities = peer.capabilities,
        );

        // Build tools list from MCP if available
        let tools: Vec<Value> = if let Some(mcp) = &mcp_client {
            mcp.tools_as_claude_tools()
        } else {
            vec![]
        };

        let mut messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text(
                "Please negotiate a communication channel with the discovered peer. \
                 Analyze capabilities and decide the best approach. \
                 If OBP tools are available, consider setting up dynamic entities for shared storage. \
                 Return your final decision as JSON."
                    .to_string(),
            ),
        }];

        // Tool-use loop
        let max_iterations = 10;
        for iteration in 0..max_iterations {
            let response = self
                .call_claude(&system_prompt, &messages, &tools)
                .await
                .context("Claude API call failed")?;

            tracing::info!(
                iteration,
                stop_reason = ?response.stop_reason,
                "Claude response received"
            );

            // Check if Claude wants to use tools
            let has_tool_use = response
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }));

            if has_tool_use && mcp_client.is_some() {
                // Add assistant message with all content blocks
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(response.content.clone()),
                });

                // Execute each tool call and collect results
                let mut tool_results = Vec::new();
                for block in &response.content {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        tracing::info!(tool = %name, "Executing MCP tool call");

                        let result = if let Some(ref mut mcp) = mcp_client {
                            match mcp.call_tool(name, input.clone()).await {
                                Ok(val) => serde_json::to_string_pretty(&val)
                                    .unwrap_or_else(|_| val.to_string()),
                                Err(e) => format!("Error calling tool: {}", e),
                            }
                        } else {
                            "MCP client not available".to_string()
                        };

                        tool_results.push(ContentBlock::ToolResult {
                            tool_use_id: id.clone(),
                            content: result,
                        });
                    }
                }

                // Add tool results as user message
                messages.push(Message {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(tool_results),
                });
            } else {
                // Claude is done - extract the final decision
                return self.extract_negotiation_result(&response.content);
            }
        }

        anyhow::bail!("Negotiation exceeded maximum iterations")
    }

    /// Call the Claude Messages API.
    async fn call_claude(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Value],
    ) -> Result<ClaudeResponse> {
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        });

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
        }

        let response = self
            .http_client
            .post(CLAUDE_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Claude API")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Claude API error ({}): {}", status, error_text);
        }

        let claude_response: ClaudeResponse = response
            .json()
            .await
            .context("Failed to parse Claude API response")?;

        Ok(claude_response)
    }

    /// Extract a NegotiationResult from Claude's final response.
    fn extract_negotiation_result(
        &self,
        content: &[ContentBlock],
    ) -> Result<NegotiationResult> {
        let mut full_text = String::new();
        for block in content {
            if let ContentBlock::Text { text } = block {
                full_text.push_str(text);
            }
        }

        // Try to extract JSON from the response
        let json_result = extract_json_from_text(&full_text);

        match json_result {
            Some(json) => {
                let channel = json
                    .get("channel")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tcp")
                    .to_string();
                let reasoning = json
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&full_text)
                    .to_string();
                let config = json.get("config").cloned().unwrap_or(Value::Object(serde_json::Map::new()));

                Ok(NegotiationResult {
                    chosen_channel: channel,
                    details: config,
                    reasoning,
                })
            }
            None => {
                // Fallback: use text as reasoning and default to TCP
                Ok(NegotiationResult {
                    chosen_channel: "tcp".to_string(),
                    details: Value::Object(serde_json::Map::new()),
                    reasoning: full_text,
                })
            }
        }
    }
}

/// Try to extract a JSON object from text that may contain markdown code blocks.
fn extract_json_from_text(text: &str) -> Option<Value> {
    // Try the whole text first
    if let Ok(val) = serde_json::from_str::<Value>(text) {
        if val.is_object() {
            return Some(val);
        }
    }

    // Try to find JSON in code blocks
    for block in text.split("```") {
        let trimmed = block.trim().strip_prefix("json").unwrap_or(block.trim());
        if let Ok(val) = serde_json::from_str::<Value>(trimmed.trim()) {
            if val.is_object() {
                return Some(val);
            }
        }
    }

    // Try to find JSON by braces
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if start < end {
                if let Ok(val) = serde_json::from_str::<Value>(&text[start..=end]) {
                    if val.is_object() {
                        return Some(val);
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_plain() {
        let text = r#"{"channel": "tcp", "config": {"port": 9000}, "reasoning": "Both support TCP"}"#;
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result["channel"], "tcp");
    }

    #[test]
    fn test_extract_json_from_markdown() {
        let text = "Here's my decision:\n```json\n{\"channel\": \"obp\", \"reasoning\": \"test\"}\n```\n";
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result["channel"], "obp");
    }

    #[test]
    fn test_extract_json_from_mixed_text() {
        let text = "After analysis, I recommend: {\"channel\": \"tcp\", \"reasoning\": \"direct\"} as the best option.";
        let result = extract_json_from_text(text).unwrap();
        assert_eq!(result["channel"], "tcp");
    }
}
