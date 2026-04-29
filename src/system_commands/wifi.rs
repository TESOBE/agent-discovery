/// WiFi connection management via `nmcli` with confirm-or-revert.
///
/// Switching WiFi from a remote command is dangerous: a wrong SSID or
/// password leaves the Pi offline and unreachable. `connect` snapshots the
/// currently active connection, attempts the new one, probes OBP, and falls
/// back to the previous network if the probe fails.
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::process::Command;

use crate::obp::client::check_obp_reachable;

const NMCLI_UP_TIMEOUT_SECS: u64 = 20;
const PROBE_ATTEMPTS: u32 = 3;
const PROBE_INTERVAL_SECS: u64 = 10;

/// Attempt to join `ssid` using `password`. On success, return a
/// `wifi-connected` status. On failure, revert to the previously active
/// connection and return `wifi-reverted` so the caller can report back.
pub async fn connect(
    ssid: &str,
    password: &str,
    priority: Option<i32>,
    obp_base_url: &str,
) -> Result<Value> {
    let previous = snapshot_active_wifi().await.unwrap_or_else(|e| {
        tracing::warn!("wifi: could not snapshot active connection: {}", e);
        None
    });
    tracing::info!(
        ssid = ssid,
        previous = previous.as_deref().unwrap_or("<none>"),
        "wifi: starting confirm-or-revert switch"
    );

    upsert_profile(ssid, password, priority).await?;

    if let Err(e) = nmcli_up(ssid, Duration::from_secs(NMCLI_UP_TIMEOUT_SECS)).await {
        tracing::warn!("wifi: `nmcli c up {}` failed: {}", ssid, e);
        return revert_and_report(ssid, previous.as_deref(), "nmcli-up-failed").await;
    }

    if probe_obp(obp_base_url, PROBE_ATTEMPTS, Duration::from_secs(PROBE_INTERVAL_SECS))
        .await
        .is_ok()
    {
        tracing::info!(ssid = ssid, "wifi: probe succeeded, switch confirmed");
        Ok(json!({
            "status": "wifi-connected",
            "ssid": ssid,
            "previous": previous,
        }))
    } else {
        tracing::warn!(ssid = ssid, "wifi: probe failed, reverting");
        revert_and_report(ssid, previous.as_deref(), "probe-failed").await
    }
}

/// Remove a saved WiFi profile. Does not touch the active connection unless
/// the deleted profile happens to be the current one.
pub async fn forget(ssid: &str) -> Result<Value> {
    let output = Command::new("nmcli")
        .args(["connection", "delete", ssid])
        .output()
        .await
        .context("failed to exec nmcli connection delete")?;
    if output.status.success() {
        Ok(json!({"status": "wifi-forgotten", "ssid": ssid}))
    } else {
        Ok(json!({
            "status": "wifi-forget-failed",
            "ssid": ssid,
            "stderr": String::from_utf8_lossy(&output.stderr).trim(),
        }))
    }
}

/// Force-join a named connection profile. Used by the watchdog to re-attach
/// to a pinned hotspot when the agent detects extended OBP unreachability.
pub async fn force_up(profile: &str) -> Result<()> {
    nmcli_up(profile, Duration::from_secs(NMCLI_UP_TIMEOUT_SECS)).await
}

async fn revert_and_report(
    attempted: &str,
    previous: Option<&str>,
    reason: &str,
) -> Result<Value> {
    let _ = Command::new("nmcli")
        .args(["connection", "down", attempted])
        .output()
        .await;

    let mut reverted_to: Option<String> = None;
    if let Some(prev) = previous {
        match nmcli_up(prev, Duration::from_secs(NMCLI_UP_TIMEOUT_SECS)).await {
            Ok(()) => reverted_to = Some(prev.to_string()),
            Err(e) => tracing::error!("wifi: revert to {} failed: {}", prev, e),
        }
    }

    Ok(json!({
        "status": "wifi-reverted",
        "ssid": attempted,
        "reason": reason,
        "reverted_to": reverted_to,
    }))
}

async fn snapshot_active_wifi() -> Result<Option<String>> {
    let output = Command::new("nmcli")
        .args(["-t", "-f", "NAME,TYPE", "connection", "show", "--active"])
        .output()
        .await
        .context("failed to exec nmcli connection show --active")?;
    if !output.status.success() {
        anyhow::bail!(
            "nmcli snapshot exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let mut parts = line.splitn(2, ':');
        let name = parts.next().unwrap_or("");
        let ty = parts.next().unwrap_or("");
        if ty == "802-11-wireless" && !name.is_empty() {
            return Ok(Some(name.to_string()));
        }
    }
    Ok(None)
}

async fn upsert_profile(ssid: &str, password: &str, priority: Option<i32>) -> Result<()> {
    let exists = Command::new("nmcli")
        .args(["-t", "-f", "NAME", "connection", "show"])
        .output()
        .await
        .context("failed to exec nmcli connection show")?;
    let already = String::from_utf8_lossy(&exists.stdout)
        .lines()
        .any(|l| l == ssid);

    let prio = priority.unwrap_or(50).to_string();

    if already {
        let status = Command::new("nmcli")
            .args([
                "connection", "modify", ssid,
                "wifi-sec.key-mgmt", "wpa-psk",
                "wifi-sec.psk", password,
                "connection.autoconnect", "yes",
                "connection.autoconnect-priority", &prio,
            ])
            .status()
            .await
            .context("failed to exec nmcli connection modify")?;
        if !status.success() {
            anyhow::bail!("nmcli connection modify exited {}", status);
        }
    } else {
        let status = Command::new("nmcli")
            .args([
                "connection", "add",
                "type", "wifi",
                "con-name", ssid,
                "ssid", ssid,
                "wifi-sec.key-mgmt", "wpa-psk",
                "wifi-sec.psk", password,
                "connection.autoconnect", "yes",
                "connection.autoconnect-priority", &prio,
            ])
            .status()
            .await
            .context("failed to exec nmcli connection add")?;
        if !status.success() {
            anyhow::bail!("nmcli connection add exited {}", status);
        }
    }
    Ok(())
}

async fn nmcli_up(name: &str, timeout: Duration) -> Result<()> {
    let child = Command::new("nmcli")
        .args(["connection", "up", name])
        .output();
    match tokio::time::timeout(timeout, child).await {
        Ok(Ok(output)) if output.status.success() => Ok(()),
        Ok(Ok(output)) => anyhow::bail!(
            "nmcli connection up {} exited {}: {}",
            name,
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ),
        Ok(Err(e)) => Err(e).context("failed to exec nmcli connection up"),
        Err(_) => anyhow::bail!("nmcli connection up {} timed out", name),
    }
}

async fn probe_obp(base_url: &str, attempts: u32, interval: Duration) -> Result<()> {
    let mut last_err: Option<anyhow::Error> = None;
    for i in 0..attempts {
        match check_obp_reachable(base_url).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                tracing::debug!("wifi: probe attempt {} failed: {}", i + 1, e);
                last_err = Some(e);
                tokio::time::sleep(interval).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("probe failed with no error")))
}
