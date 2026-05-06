use anyhow::{Context, Result};

/// One OBP host the agent can talk to. Read from env vars suffixed by `label`
/// uppercased (e.g. label "a" → OBP_API_BASE_URL_A, OBP_USERNAME_A, ...).
#[derive(Debug, Clone)]
pub struct ObpHost {
    pub label: String,
    pub base_url: String,
    pub username: String,
    pub password: String,
    pub consumer_key: String,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub agent_name: String,
    pub agent_listen_port: u16,
    pub claude_api_key: String,
    pub claude_model: String,
    /// OBP hosts in priority order. The agent starts on `obp_hosts[0]` (home)
    /// and switches to the next host on transport failure. Hosts with an
    /// empty base URL in the env are filtered out.
    pub obp_hosts: Vec<ObpHost>,
    /// URL of a running MCP server (streamable HTTP transport).
    pub obp_mcp_server_url: Option<String>,
    /// Directory for log files. Defaults to /tmp/agent-discovery.
    pub log_dir: String,
    /// Comma-separated list of OBP user IDs allowed to instruct this agent.
    /// If empty, all task requests are rejected (closed by default).
    pub instructor_user_ids: Vec<String>,
    /// NetworkManager connection profile name used as a guaranteed fallback
    /// (typically the phone hotspot). Empty disables the watchdog.
    pub hotspot_profile_name: String,
    /// Consecutive watchdog probe failures before forcing the hotspot.
    pub watchdog_fail_threshold: u32,
    /// Seconds between watchdog reachability probes.
    pub watchdog_probe_interval_secs: u64,
    /// systemd unit name running the ffmpeg stream.
    pub stream_service_name: String,
    /// Seconds between `system-commands` channel polls.
    pub system_command_poll_interval_secs: u64,
}

impl Config {
    /// Path to this agent's log file.
    pub fn log_file(&self) -> String {
        format!("{}/{}.log", self.log_dir, self.agent_name)
    }

    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let obp_hosts = ["a", "b"]
            .into_iter()
            .filter_map(|label| {
                let upper = label.to_uppercase();
                let base_url = std::env::var(format!("OBP_API_BASE_URL_{}", upper))
                    .unwrap_or_default();
                if base_url.is_empty() {
                    return None;
                }
                Some(ObpHost {
                    label: label.into(),
                    base_url: base_url.trim_end_matches('/').to_string(),
                    username: std::env::var(format!("OBP_USERNAME_{}", upper))
                        .unwrap_or_default(),
                    password: std::env::var(format!("OBP_PASSWORD_{}", upper))
                        .unwrap_or_default(),
                    consumer_key: std::env::var(format!("OBP_CONSUMER_KEY_{}", upper))
                        .unwrap_or_default(),
                })
            })
            .collect();

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
            obp_hosts,
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
            hotspot_profile_name: std::env::var("HOTSPOT_PROFILE_NAME").unwrap_or_default(),
            watchdog_fail_threshold: std::env::var("WATCHDOG_FAIL_THRESHOLD")
                .unwrap_or_else(|_| "5".into())
                .parse()
                .context("WATCHDOG_FAIL_THRESHOLD must be a positive integer")?,
            watchdog_probe_interval_secs: std::env::var("WATCHDOG_PROBE_INTERVAL_SECS")
                .unwrap_or_else(|_| "60".into())
                .parse()
                .context("WATCHDOG_PROBE_INTERVAL_SECS must be a positive integer")?,
            stream_service_name: std::env::var("STREAM_SERVICE_NAME")
                .unwrap_or_else(|_| "stream.service".into()),
            system_command_poll_interval_secs: std::env::var("SYSTEM_COMMAND_POLL_INTERVAL_SECS")
                .unwrap_or_else(|_| "10".into())
                .parse()
                .context("SYSTEM_COMMAND_POLL_INTERVAL_SECS must be a positive integer")?,
        })
    }
}
