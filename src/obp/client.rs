/// OBP HTTP client with DirectLogin authentication.
/// Fallback for when MCP server is unavailable.
///
/// One client per OBP host. The agent constructs one of these per configured
/// host and runs them concurrently — pollers, presence publishers, and
/// handshake routes all attach to a specific client and therefore a specific
/// host. There is no cross-host failover; if a host is unreachable, that
/// host's loops just retry on their own cadence while the other hosts keep
/// running.
///
/// Auth tokens are acquired lazily on first use and cached.
///
/// Callers provide the full API path (e.g. `/obp/v6.0.0/management/...`).

use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::{Method, StatusCode};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::config::ObpHost;

/// The OBP API version used for management and resource-docs endpoints.
pub const API_VERSION: &str = "v6.0.0";

/// Check if an OBP API base URL is reachable by hitting /obp/v6.0.0/root.
/// Returns Ok(()) if the endpoint responds with a success status, Err otherwise.
pub async fn check_obp_reachable(base_url: &str) -> Result<()> {
    let url = format!("{}/obp/{}/root", base_url.trim_end_matches('/'), API_VERSION);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("Failed to build HTTP client for reachability check")?;

    let response = client
        .get(&url)
        .send()
        .await
        .context(format!("OBP reachability check failed for {}", url))?;

    if response.status().is_success() {
        Ok(())
    } else {
        anyhow::bail!(
            "OBP reachability check got status {} from {}",
            response.status(),
            url
        )
    }
}

/// OBP API client bound to a single host.
pub struct ObpClient {
    host: ObpHost,
    token: RwLock<Option<String>>,
    http: reqwest::Client,
}

impl ObpClient {
    pub fn new(host: ObpHost) -> Self {
        Self {
            host,
            token: RwLock::new(None),
            http: reqwest::Client::new(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.host.base_url
    }

    pub fn label(&self) -> &str {
        &self.host.label
    }

    pub async fn is_authenticated(&self) -> bool {
        self.token.read().await.is_some()
    }

    /// Authenticate this host. Idempotent: if already authenticated, returns
    /// Ok immediately. Errors propagate to the caller; the next call retries.
    pub async fn authenticate(&self) -> Result<()> {
        self.ensure_authenticated().await
    }

    pub async fn get(&self, path: &str) -> Result<Value> {
        let resp = self.send(Method::GET, path, None).await?;
        parse_response("GET", path, resp).await
    }

    pub async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let resp = self.send(Method::POST, path, Some(body)).await?;
        parse_response("POST", path, resp).await
    }

    pub async fn put(&self, path: &str, body: &Value) -> Result<Value> {
        let resp = self.send(Method::PUT, path, Some(body)).await?;
        parse_response("PUT", path, resp).await
    }

    pub async fn delete(&self, path: &str) -> Result<Value> {
        let resp = self.send(Method::DELETE, path, None).await?;
        let status = resp.status();
        if status.is_success() {
            // DELETE may return an empty body
            match resp.json().await {
                Ok(body) => Ok(body),
                Err(_) => Ok(Value::Null),
            }
        } else {
            let body: Value = resp
                .json()
                .await
                .unwrap_or(Value::String("Unknown error".into()));
            anyhow::bail!("OBP DELETE {} failed ({}): {}", path, status, body);
        }
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<reqwest::Response> {
        self.ensure_authenticated().await?;

        let token = self.token.read().await.clone();
        let url = format!("{}{}", self.host.base_url, path);

        let mut req = self.http.request(method, &url);
        if let Some(t) = &token {
            req = req.header("Authorization", format!("DirectLogin token=\"{}\"", t));
        }
        if let Some(b) = body {
            req = req.json(b);
        }

        req.send()
            .await
            .with_context(|| format!("OBP request failed on host {} ({})", self.host.label, url))
    }

    /// Authenticate if not already authenticated. Idempotent and safe under
    /// concurrent calls (double-checked locking).
    async fn ensure_authenticated(&self) -> Result<()> {
        {
            let token = self.token.read().await;
            if token.is_some() {
                return Ok(());
            }
        }

        let mut token = self.token.write().await;
        if token.is_some() {
            return Ok(());
        }

        let host = &self.host;
        if host.username.is_empty() || host.password.is_empty() || host.consumer_key.is_empty() {
            anyhow::bail!(
                "OBP credentials not configured for host {} (OBP_USERNAME_{}, OBP_PASSWORD_{}, OBP_CONSUMER_KEY_{})",
                host.label,
                host.label.to_uppercase(),
                host.label.to_uppercase(),
                host.label.to_uppercase(),
            );
        }

        let auth_header = format!(
            "DirectLogin username=\"{}\",password=\"{}\",consumer_key=\"{}\"",
            host.username, host.password, host.consumer_key
        );

        let url = format!("{}/obp/{}/my/logins/direct", host.base_url, API_VERSION);
        let response = self
            .http
            .post(&url)
            .header("Authorization", &auth_header)
            .header("Content-Type", "application/json")
            .send()
            .await
            .with_context(|| format!("DirectLogin request failed for host {}", host.label))?;

        let status = response.status();
        let body: Value = response
            .json()
            .await
            .context("Failed to parse DirectLogin response")?;

        if !status.is_success() {
            anyhow::bail!(
                "DirectLogin failed for host {} ({}): {}",
                host.label,
                status,
                body
            );
        }

        let token_str = body
            .get("token")
            .and_then(|t| t.as_str())
            .context("No token in DirectLogin response")?
            .to_string();

        tracing::info!("OBP DirectLogin successful for host {}", host.label);
        *token = Some(token_str);
        Ok(())
    }
}

/// Parse a JSON response and surface non-success statuses as anyhow errors.
async fn parse_response(method: &str, path: &str, resp: reqwest::Response) -> Result<Value> {
    let status: StatusCode = resp.status();
    let body: Value = resp.json().await.context("Failed to parse OBP response")?;
    if !status.is_success() {
        anyhow::bail!("OBP {} {} failed ({}): {}", method, path, status, body);
    }
    Ok(body)
}
