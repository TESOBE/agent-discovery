/// Reachability watchdog — last line of defence for WiFi robustness.
///
/// Probes every configured OBP host at a steady interval. After N consecutive
/// rounds in which *no* host is reachable, forcibly re-joins the pinned
/// hotspot profile so the Pi regains a connection the agent can be commanded
/// over. Runs even if the command poller is wedged.
use std::time::Duration;

use crate::obp::client::check_obp_reachable;
use crate::system_commands::wifi;

pub async fn run_reachability_watchdog(
    obp_base_urls: Vec<String>,
    hotspot_profile: String,
    fail_threshold: u32,
    probe_interval: Duration,
) {
    if hotspot_profile.is_empty() {
        tracing::info!("watchdog: HOTSPOT_PROFILE_NAME not set — watchdog disabled");
        return;
    }
    if obp_base_urls.is_empty() {
        tracing::info!("watchdog: no OBP hosts configured — watchdog disabled");
        return;
    }

    tracing::info!(
        "watchdog: starting (probe every {}s, threshold {}, hotspot '{}', {} OBP host(s))",
        probe_interval.as_secs(),
        fail_threshold,
        hotspot_profile,
        obp_base_urls.len()
    );

    let mut consecutive_failures: u32 = 0;
    loop {
        tokio::time::sleep(probe_interval).await;

        let mut any_reachable = false;
        let mut errors: Vec<String> = Vec::new();
        for url in &obp_base_urls {
            match check_obp_reachable(url).await {
                Ok(()) => {
                    any_reachable = true;
                    break;
                }
                Err(e) => errors.push(format!("{}: {}", url, e)),
            }
        }

        if any_reachable {
            if consecutive_failures > 0 {
                tracing::info!(
                    "watchdog: OBP reachable again after {} failures",
                    consecutive_failures
                );
            }
            consecutive_failures = 0;
        } else {
            consecutive_failures += 1;
            tracing::warn!(
                "watchdog: no OBP host reachable ({}/{}): [{}]",
                consecutive_failures,
                fail_threshold,
                errors.join("; ")
            );
            if consecutive_failures >= fail_threshold {
                tracing::error!(
                    "watchdog: threshold reached, forcing hotspot '{}'",
                    hotspot_profile
                );
                match wifi::force_up(&hotspot_profile).await {
                    Ok(()) => tracing::info!(
                        "watchdog: hotspot '{}' re-joined",
                        hotspot_profile
                    ),
                    Err(e) => tracing::error!(
                        "watchdog: failed to force hotspot: {}",
                        e
                    ),
                }
                consecutive_failures = 0;
            }
        }
    }
}
