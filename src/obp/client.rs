/// OBP HTTP client with DirectLogin authentication.
/// Fallback for when MCP server is unavailable.
///
/// Holds a list of OBP hosts (`a`, `b`, ...) and an "active" pointer
/// indicating which one is currently `home`. Requests go to home; on a
/// transport failure (connect error / timeout) or an auth failure, the client
/// flips home to the next host and retries. Application errors from a
/// successful HTTP exchange (4xx / 5xx on the data path) do NOT flip home —
/// the server replied coherently, the host is up.
///
/// Auth tokens are stored per host and acquired lazily on first use.
///
/// Callers provide the full API path (e.g. `/obp/v6.0.0/management/...`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::{Method, StatusCode};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::config::{Config, ObpHost};

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

/// One configured host plus its lazily-acquired auth token.
struct HostState {
    host: ObpHost,
    token: RwLock<Option<String>>,
}

/// OBP API client: multi-host with sticky-home failover.
pub struct ObpClient {
    /// Hosts in priority order, each carrying its own lazy auth token.
    hosts: Vec<HostState>,
    /// Index into `hosts` of the host currently considered home.
    active: AtomicUsize,
    http: reqwest::Client,
}

impl ObpClient {
    /// Build a client from the configured OBP hosts. Returns Err if no hosts
    /// are configured (all entries had empty base URLs).
    pub fn new(config: &Config) -> Result<Self> {
        if config.obp_hosts.is_empty() {
            anyhow::bail!("No OBP hosts configured (set OBP_API_BASE_URL_A and/or _B)");
        }
        let hosts = config
            .obp_hosts
            .iter()
            .map(|h| HostState {
                host: h.clone(),
                token: RwLock::new(None),
            })
            .collect();
        Ok(Self {
            hosts,
            active: AtomicUsize::new(0),
            http: reqwest::Client::new(),
        })
    }

    /// Authenticate at startup by walking the host list. Returns Ok as soon
    /// as any host successfully authenticates; that host becomes home. Any
    /// failure flips home to the next candidate. Returns Err only if every
    /// host failed.
    pub async fn authenticate(&self) -> Result<()> {
        let n = self.hosts.len();
        let mut last_err: Option<anyhow::Error> = None;
        for _ in 0..n {
            let idx = self.active.load(Ordering::Acquire);
            let label = self.hosts[idx].host.label.clone();
            match self.ensure_authenticated(idx).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!("OBP auth failed for host {}: {}", label, e);
                    self.flip_home_from(idx);
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("all OBP hosts failed auth")))
    }

    /// Base URL of the host currently considered home.
    pub fn base_url(&self) -> &str {
        let idx = self.active.load(Ordering::Acquire);
        &self.hosts[idx].host.base_url
    }

    /// Label ("a", "b", ...) of the host currently considered home.
    pub fn active_label(&self) -> &str {
        let idx = self.active.load(Ordering::Acquire);
        &self.hosts[idx].host.label
    }

    /// Base URLs of all configured hosts, in priority order.
    pub fn host_base_urls(&self) -> Vec<String> {
        self.hosts.iter().map(|h| h.host.base_url.clone()).collect()
    }

    /// True if the home host has a cached auth token.
    pub async fn is_authenticated(&self) -> bool {
        let idx = self.active.load(Ordering::Acquire);
        self.hosts[idx].token.read().await.is_some()
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

    /// Send a request to the home host, with sticky failover to the next host
    /// on transport or auth failure. Returns the raw response (caller checks
    /// status for application-level errors).
    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<reqwest::Response> {
        let n = self.hosts.len();
        let mut last_err: Option<anyhow::Error> = None;

        for _ in 0..n {
            let idx = self.active.load(Ordering::Acquire);
            let host = &self.hosts[idx].host;

            // Auth precondition: any failure here (transport or HTTP) flips
            // home — we can't use this host without a token, so move on.
            if let Err(e) = self.ensure_authenticated(idx).await {
                tracing::warn!("OBP auth failed on host {}: {}", host.label, e);
                self.flip_home_from(idx);
                last_err = Some(e);
                continue;
            }

            let token = self.hosts[idx].token.read().await.clone();
            let url = format!("{}{}", host.base_url, path);

            let mut req = self.http.request(method.clone(), &url);
            if let Some(t) = &token {
                req = req.header("Authorization", format!("DirectLogin token=\"{}\"", t));
            }
            if let Some(b) = body {
                req = req.json(b);
            }

            match req.send().await {
                Ok(resp) => return Ok(resp),
                Err(e) if is_transport_failure(&e) => {
                    tracing::warn!(
                        "OBP transport failure on host {} ({}): {}",
                        host.label,
                        url,
                        e
                    );
                    self.flip_home_from(idx);
                    last_err = Some(anyhow::anyhow!(
                        "transport failure on host {}: {}",
                        host.label,
                        e
                    ));
                    continue;
                }
                Err(e) => return Err(anyhow::anyhow!("OBP request failed: {}", e)),
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("all OBP hosts failed for {}", path)))
    }

    /// Authenticate host `idx` if not already authenticated. Idempotent and
    /// safe under concurrent calls (double-checked locking).
    async fn ensure_authenticated(&self, idx: usize) -> Result<()> {
        {
            let token = self.hosts[idx].token.read().await;
            if token.is_some() {
                return Ok(());
            }
        }

        let mut token = self.hosts[idx].token.write().await;
        if token.is_some() {
            return Ok(());
        }

        let host = &self.hosts[idx].host;
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

    /// Advance `active` from `observed_idx` to the next host. CAS so that
    /// concurrent failures only flip home once per failure event.
    fn flip_home_from(&self, observed_idx: usize) {
        let n = self.hosts.len();
        if n < 2 {
            return;
        }
        let next = (observed_idx + 1) % n;
        let from = self.hosts[observed_idx].host.label.clone();
        let to = self.hosts[next].host.label.clone();
        if self
            .active
            .compare_exchange(observed_idx, next, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            tracing::warn!("OBP home: {} → {}", from, to);
        }
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

/// True if the reqwest error indicates the host is unreachable
/// (connect refused, DNS failure, timeout) rather than the host replying
/// with an unexpected payload or status.
fn is_transport_failure(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout()
}
