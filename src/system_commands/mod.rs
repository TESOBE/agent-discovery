/// Direct, non-Claude dispatch for hard system operations sent over a
/// dedicated OBP signal channel (`system-commands` by default).
///
/// Why this exists alongside the Claude-mediated `task-requests` poller:
/// switching WiFi, starting/stopping the stream, and similar operations
/// should run against a strict typed schema with no interpretation step.
/// See `docs/streaming-pi-setup.md`.
pub mod stream;
pub mod watchdog;
pub mod wifi;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::Config;
use crate::obp::client::{ObpClient, API_VERSION};
use crate::obp::entities::iso_now;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum SystemCommand {
    WifiConnect {
        ssid: String,
        password: String,
        priority: Option<i32>,
    },
    WifiForget {
        ssid: String,
    },
    StreamStart,
    StreamStop,
    StreamStatus,
}

pub async fn run_system_command_poller(
    obp: Arc<ObpClient>,
    config: Config,
    agent_name: String,
) {
    let channel_in = "system-commands";
    let channel_out = "system-command-responses";
    let mut seen: HashSet<String> = HashSet::new();

    tokio::time::sleep(Duration::from_secs(5)).await;

    tracing::info!(
        "system-commands poller: watching '{}' every {}s",
        channel_in,
        config.system_command_poll_interval_secs
    );

    loop {
        match read_signal(&obp, channel_in).await {
            Ok(response) => {
                if let Some(messages) = response.get("messages").and_then(|m| m.as_array()) {
                    for msg in messages {
                        let msg_id = msg
                            .get("message_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if msg_id.is_empty() || seen.contains(&msg_id) {
                            continue;
                        }
                        seen.insert(msg_id.clone());

                        let sender = msg
                            .get("sender_user_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !config.instructor_user_ids.iter().any(|id| id == sender) {
                            tracing::warn!(
                                message_id = %msg_id,
                                sender = %sender,
                                "system-commands: ignoring message from non-instructor"
                            );
                            continue;
                        }

                        let payload = msg.get("payload").cloned().unwrap_or(Value::Null);
                        handle_message(
                            &obp,
                            &config,
                            &agent_name,
                            channel_out,
                            &msg_id,
                            payload,
                        )
                        .await;
                    }
                }
            }
            Err(e) => {
                tracing::debug!(
                    "system-commands poller: read failed on '{}': {}",
                    channel_in,
                    e
                );
            }
        }

        tokio::time::sleep(Duration::from_secs(
            config.system_command_poll_interval_secs,
        ))
        .await;
    }
}

async fn handle_message(
    obp: &ObpClient,
    config: &Config,
    agent_name: &str,
    channel_out: &str,
    msg_id: &str,
    payload: Value,
) {
    let result = match serde_json::from_value::<SystemCommand>(payload.clone()) {
        Ok(cmd) => dispatch(cmd, config).await,
        Err(e) => Err(anyhow::anyhow!("invalid payload: {}", e)),
    };

    let response = match result {
        Ok(value) => json!({
            "payload": {
                "type": "system-command-response",
                "from": agent_name,
                "in_reply_to": msg_id,
                "status": "ok",
                "result": value,
                "timestamp": iso_now(),
            }
        }),
        Err(e) => {
            tracing::warn!(message_id = %msg_id, "system-commands: error: {}", e);
            json!({
                "payload": {
                    "type": "system-command-response",
                    "from": agent_name,
                    "in_reply_to": msg_id,
                    "status": "error",
                    "error": e.to_string(),
                    "timestamp": iso_now(),
                }
            })
        }
    };

    if let Err(e) = publish_signal(obp, channel_out, &response).await {
        tracing::warn!(
            "system-commands: failed to publish response for {}: {}",
            msg_id,
            e
        );
    }
}

async fn dispatch(cmd: SystemCommand, config: &Config) -> Result<Value> {
    match cmd {
        SystemCommand::WifiConnect {
            ssid,
            password,
            priority,
        } => wifi::connect(&ssid, &password, priority, &config.obp_api_base_url).await,
        SystemCommand::WifiForget { ssid } => wifi::forget(&ssid).await,
        SystemCommand::StreamStart => stream::start(&config.stream_service_name).await,
        SystemCommand::StreamStop => stream::stop(&config.stream_service_name).await,
        SystemCommand::StreamStatus => stream::status(&config.stream_service_name).await,
    }
}

async fn publish_signal(obp: &ObpClient, channel: &str, payload: &Value) -> Result<Value> {
    let path = format!("/obp/{}/signal/channels/{}/messages", API_VERSION, channel);
    obp.post(&path, payload)
        .await
        .context("signal POST failed")
}

async fn read_signal(obp: &ObpClient, channel: &str) -> Result<Value> {
    let path = format!("/obp/{}/signal/channels/{}/messages", API_VERSION, channel);
    obp.get(&path).await.context("signal GET failed")
}
