use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub agent_name: String,
    pub agent_listen_port: u16,
    pub claude_api_key: String,
    pub claude_model: String,
    pub obp_base_url: String,
    pub obp_api_version: String,
    pub obp_username: String,
    pub obp_password: String,
    pub obp_consumer_key: String,
    pub obp_bank_id: String,
    pub obp_mcp_server_command: String,
    pub obp_mcp_server_args: Vec<String>,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let mcp_args_str =
            std::env::var("OBP_MCP_SERVER_ARGS").unwrap_or_else(|_| "obp-mcp-server".into());
        let obp_mcp_server_args: Vec<String> =
            mcp_args_str.split_whitespace().map(String::from).collect();

        Ok(Self {
            agent_name: std::env::var("AGENT_NAME")
                .unwrap_or_else(|_| "agent-default".into()),
            agent_listen_port: std::env::var("AGENT_LISTEN_PORT")
                .unwrap_or_else(|_| "9000".into())
                .parse()
                .context("AGENT_LISTEN_PORT must be a valid port number")?,
            claude_api_key: std::env::var("CLAUDE_API_KEY")
                .unwrap_or_default(),
            claude_model: std::env::var("CLAUDE_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".into()),
            obp_base_url: std::env::var("OBP_BASE_URL")
                .unwrap_or_else(|_| "https://apisandbox.openbankproject.com".into()),
            obp_api_version: std::env::var("OBP_API_VERSION")
                .unwrap_or_else(|_| "v6.0.0".into()),
            obp_username: std::env::var("OBP_USERNAME").unwrap_or_default(),
            obp_password: std::env::var("OBP_PASSWORD").unwrap_or_default(),
            obp_consumer_key: std::env::var("OBP_CONSUMER_KEY").unwrap_or_default(),
            obp_bank_id: std::env::var("OBP_BANK_ID")
                .unwrap_or_else(|_| "gh.29.uk".into()),
            obp_mcp_server_command: std::env::var("OBP_MCP_SERVER_COMMAND")
                .unwrap_or_else(|_| "uvx".into()),
            obp_mcp_server_args,
        })
    }
}
