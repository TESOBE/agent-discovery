/// OBP HTTP client with DirectLogin authentication.
/// Fallback for when MCP server is unavailable.
///
/// All requests target the v6.0.0 API exclusively.

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::Config;

/// The only OBP API version we use.
pub const API_VERSION: &str = "v6.0.0";

/// OBP API client using DirectLogin authentication.
pub struct ObpClient {
    base_url: String,
    bank_id: String,
    http_client: reqwest::Client,
    auth_token: Option<String>,
    username: String,
    password: String,
    consumer_key: String,
}

impl ObpClient {
    pub fn new(config: &Config) -> Self {
        Self {
            base_url: config.obp_base_url.clone(),
            bank_id: config.obp_bank_id.clone(),
            http_client: reqwest::Client::new(),
            auth_token: None,
            username: config.obp_username.clone(),
            password: config.obp_password.clone(),
            consumer_key: config.obp_consumer_key.clone(),
        }
    }

    /// Authenticate via DirectLogin and store the token.
    pub async fn authenticate(&mut self) -> Result<()> {
        if self.username.is_empty() || self.password.is_empty() || self.consumer_key.is_empty() {
            anyhow::bail!("OBP credentials not configured (OBP_USERNAME, OBP_PASSWORD, OBP_CONSUMER_KEY)");
        }

        let auth_header = format!(
            "DirectLogin username=\"{}\",password=\"{}\",consumer_key=\"{}\"",
            self.username, self.password, self.consumer_key
        );

        let url = format!("{}/my/logins/direct", self.base_url);
        let response = self
            .http_client
            .post(&url)
            .header("Authorization", &auth_header)
            .header("Content-Type", "application/json")
            .send()
            .await
            .context("DirectLogin request failed")?;

        let status = response.status();
        let body: Value = response.json().await.context("Failed to parse DirectLogin response")?;

        if !status.is_success() {
            anyhow::bail!("DirectLogin failed ({}): {}", status, body);
        }

        let token = body
            .get("token")
            .and_then(|t| t.as_str())
            .context("No token in DirectLogin response")?
            .to_string();

        tracing::info!("OBP DirectLogin successful");
        self.auth_token = Some(token);
        Ok(())
    }

    /// Make an authenticated GET request to the OBP API.
    pub async fn get(&self, path: &str) -> Result<Value> {
        let url = format!("{}/obp/{}{}", self.base_url, API_VERSION, path);
        let mut request = self.http_client.get(&url);

        if let Some(ref token) = self.auth_token {
            request = request.header("Authorization", format!("DirectLogin token=\"{}\"", token));
        }

        let response = request.send().await.context("OBP GET request failed")?;
        let status = response.status();
        let body: Value = response.json().await.context("Failed to parse OBP response")?;

        if !status.is_success() {
            anyhow::bail!("OBP GET {} failed ({}): {}", path, status, body);
        }

        Ok(body)
    }

    /// Make an authenticated POST request to the OBP API.
    pub async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}/obp/{}{}", self.base_url, API_VERSION, path);
        let mut request = self.http_client.post(&url).json(body);

        if let Some(ref token) = self.auth_token {
            request = request.header("Authorization", format!("DirectLogin token=\"{}\"", token));
        }

        let response = request.send().await.context("OBP POST request failed")?;
        let status = response.status();
        let response_body: Value = response.json().await.context("Failed to parse OBP response")?;

        if !status.is_success() {
            anyhow::bail!("OBP POST {} failed ({}): {}", path, status, response_body);
        }

        Ok(response_body)
    }

    /// Make an authenticated PUT request to the OBP API.
    pub async fn put(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}/obp/{}{}", self.base_url, API_VERSION, path);
        let mut request = self.http_client.put(&url).json(body);

        if let Some(ref token) = self.auth_token {
            request = request.header("Authorization", format!("DirectLogin token=\"{}\"", token));
        }

        let response = request.send().await.context("OBP PUT request failed")?;
        let status = response.status();
        let response_body: Value = response.json().await.context("Failed to parse OBP response")?;

        if !status.is_success() {
            anyhow::bail!("OBP PUT {} failed ({}): {}", path, status, response_body);
        }

        Ok(response_body)
    }

    /// Make an authenticated DELETE request to the OBP API.
    pub async fn delete(&self, path: &str) -> Result<Value> {
        let url = format!("{}/obp/{}{}", self.base_url, API_VERSION, path);
        let mut request = self.http_client.delete(&url);

        if let Some(ref token) = self.auth_token {
            request = request.header("Authorization", format!("DirectLogin token=\"{}\"", token));
        }

        let response = request.send().await.context("OBP DELETE request failed")?;
        let status = response.status();

        if status.is_success() {
            // Some DELETE responses may be empty
            match response.json().await {
                Ok(body) => Ok(body),
                Err(_) => Ok(Value::Null),
            }
        } else {
            let body: Value = response
                .json()
                .await
                .unwrap_or(Value::String("Unknown error".into()));
            anyhow::bail!("OBP DELETE {} failed ({}): {}", path, status, body);
        }
    }

    pub fn bank_id(&self) -> &str {
        &self.bank_id
    }

    pub fn is_authenticated(&self) -> bool {
        self.auth_token.is_some()
    }
}
