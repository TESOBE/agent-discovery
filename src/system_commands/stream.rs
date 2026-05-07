/// ffmpeg streaming service control via `systemctl`.
///
/// The streaming unit is a plain systemd service (name configurable) that
/// runs the ffmpeg command documented in `docs/streaming-pi-setup.md`.
use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::fs;
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

/// Read RTMP_URL from `.stream-env` (relative to cwd). Stream key intentionally
/// omitted. Returns None if the file is missing or has no RTMP_URL.
pub async fn read_rtmp_url() -> Option<String> {
    let body = fs::read_to_string(".stream-env").await.ok()?;
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("RTMP_URL=") {
            let v = rest.trim().trim_matches(|c| c == '"' || c == '\'');
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Latest ffmpeg progress fields scraped from the unit's journal.
#[derive(Debug, Default)]
pub struct ProgressStats {
    pub frame: Option<String>,
    pub fps: Option<String>,
    pub bitrate: Option<String>,
    pub time: Option<String>,
    pub speed: Option<String>,
    pub size: Option<String>,
}

impl ProgressStats {
    pub fn is_empty(&self) -> bool {
        self.frame.is_none()
            && self.fps.is_none()
            && self.bitrate.is_none()
            && self.time.is_none()
            && self.speed.is_none()
            && self.size.is_none()
    }
}

/// Pull the most recent ffmpeg progress line from `journalctl -u <service>`
/// and parse it. Returns None on any failure or if no progress line found.
pub async fn read_recent_progress(service: &str) -> Option<ProgressStats> {
    let output = Command::new("journalctl")
        .args(["-u", service, "-n", "200", "--no-pager", "--output=cat"])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    // ffmpeg may use \r to overwrite progress lines; journald captures those
    // as line-ends in some setups, but if not, split on \r as well.
    let last = stdout
        .split(|c| c == '\n' || c == '\r')
        .rev()
        .find(|l| l.contains("bitrate=") && l.contains("frame="))?
        .to_string();

    Some(parse_progress(&last))
}

// ffmpeg's progress format pads with spaces around values (e.g.
// "frame= 1234 fps= 30"); split_whitespace + per-token '=' split handles both
// "key=val" and "key=" "val" forms.
fn parse_progress(line: &str) -> ProgressStats {
    let mut s = ProgressStats::default();
    let toks: Vec<&str> = line.split_whitespace().collect();
    let mut i = 0;
    while i < toks.len() {
        let tok = toks[i];
        if let Some(idx) = tok.find('=') {
            let key = &tok[..idx];
            let mut val = tok[idx + 1..].to_string();
            // If value is empty, the actual value is the next token
            // (e.g. "frame=" "1234").
            if val.is_empty() && i + 1 < toks.len() {
                val = toks[i + 1].to_string();
                i += 1;
            }
            match key {
                "frame" => s.frame = Some(val),
                "fps" => s.fps = Some(val),
                "bitrate" => s.bitrate = Some(val),
                "time" => s.time = Some(val),
                "speed" => s.speed = Some(val),
                "size" => s.size = Some(val),
                _ => {}
            }
        }
        i += 1;
    }
    s
}
