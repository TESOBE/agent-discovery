/// ffmpeg streaming service control via `systemctl`.
///
/// The streaming unit is a plain systemd service (name configurable) that
/// runs the ffmpeg command documented in `docs/streaming-pi-setup.md`.
use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::process::Command;

pub async fn start(service: &str) -> Result<Value> {
    run_systemctl(&["start", service]).await?;
    let active = is_active(service).await?;
    Ok(json!({
        "status": "stream-start-issued",
        "service": service,
        "active": active,
    }))
}

pub async fn stop(service: &str) -> Result<Value> {
    run_systemctl(&["stop", service]).await?;
    let active = is_active(service).await?;
    Ok(json!({
        "status": "stream-stop-issued",
        "service": service,
        "active": active,
    }))
}

pub async fn status(service: &str) -> Result<Value> {
    let active = is_active(service).await?;
    Ok(json!({
        "status": "stream-status",
        "service": service,
        "active": active,
    }))
}

async fn is_active(service: &str) -> Result<bool> {
    let output = Command::new("systemctl")
        .args(["is-active", service])
        .output()
        .await
        .context("failed to exec systemctl is-active")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.trim() == "active")
}

async fn run_systemctl(args: &[&str]) -> Result<()> {
    let status = Command::new("systemctl")
        .args(args)
        .status()
        .await
        .context("failed to exec systemctl")?;
    if !status.success() {
        anyhow::bail!("systemctl {:?} exited {}", args, status);
    }
    Ok(())
}
