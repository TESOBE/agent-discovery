use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub agent_name: String,
    pub agent_listen_port: u16,
    pub claude_api_key: String,
    pub claude_model: String,
    pub obp_api_base_url: String,
    pub obp_username: String,
    pub obp_password: String,
    pub obp_consumer_key: String,
    /// URL of a running MCP server (streamable HTTP transport).
    pub obp_mcp_server_url: Option<String>,
    /// Directory for log files. Defaults to /tmp/agent-discovery.
    pub log_dir: String,
    /// Comma-separated list of OBP user IDs allowed to instruct this agent.
    /// If empty, all task requests are rejected (closed by default).
    pub instructor_user_ids: Vec<String>,
}

impl Config {
    /// Path to this agent's log file.
    pub fn log_file(&self) -> String {
        format!("{}/{}.log", self.log_dir, self.agent_name)
    }

    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Self {
            agent_name: std::env::var("AGENT_NAME")
                .unwrap_or_else(|_| "agent-default".into()),
            agent_listen_port: std::env::var("AGENT_LISTEN_PORT")
                .unwrap_or_else(|_| "7312".into())
                .parse()
                .context("AGENT_LISTEN_PORT must be a valid port number")?,
            claude_api_key: std::env::var("CLAUDE_API_KEY")
                .unwrap_or_default(),
            claude_model: std::env::var("CLAUDE_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".into()),
            obp_api_base_url: std::env::var("OBP_API_BASE_URL").unwrap_or_default(),
            obp_username: std::env::var("OBP_USERNAME").unwrap_or_default(),
            obp_password: std::env::var("OBP_PASSWORD").unwrap_or_default(),
            obp_consumer_key: std::env::var("OBP_CONSUMER_KEY").unwrap_or_default(),
            obp_mcp_server_url: Some(std::env::var("OBP_MCP_SERVER_URL")
                .unwrap_or_else(|_| "http://0.0.0.0:9100/mcp".into())),
            log_dir: std::env::var("LOG_DIR")
                .unwrap_or_else(|_| "/tmp/agent-discovery".into()),
            instructor_user_ids: std::env::var("INSTRUCTOR_USER_IDS")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        })
    }
}
