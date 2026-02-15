#![allow(dead_code)]

mod agent;
mod audio;
mod comms;
mod config;
mod discovery;
mod mcp;
mod negotiation;
mod obp;
mod protocol;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::Config;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser)]
#[command(name = "agent-discovery", about = "Agent discovery via audio FSK tones")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the agent (discover peers, negotiate, communicate)
    Run {
        /// Use UDP immediately (without this flag, agents try chirp discovery
        /// for 10 minutes before falling back to UDP)
        #[arg(long)]
        udp: bool,
    },
    /// List available audio devices
    ListDevices,
    /// Play a test FSK tone encoding a message
    TestTone {
        /// Message to encode and play
        #[arg(short, long, default_value = "HELLO")]
        message: String,
    },
    /// Listen for and decode FSK tones from the microphone
    Decode {
        /// Duration to listen in seconds
        #[arg(short, long, default_value = "10")]
        duration: u64,
    },
    /// Play a tone and try to decode it from the mic (tests the acoustic path)
    TestRoundtrip,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::from_env()?;

    // Set up logging: stderr + rolling log file in log_dir.
    // Log file is named after the agent, e.g. agent-alpha.log
    let log_dir = &config.log_dir;
    std::fs::create_dir_all(log_dir)?;

    // Clear the previous log file so each run starts fresh
    let log_file_path = format!("{}/{}.log", log_dir, config.agent_name);
    let _ = std::fs::write(&log_file_path, b"");

    let file_appender = tracing_appender::rolling::never(log_dir, format!("{}.log", config.agent_name));
    let (file_writer, _guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("agent_discovery=info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(file_writer),
        )
        .init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        agent_name = %config.agent_name,
        log_file = %format!("{}/{}.log", log_dir, config.agent_name),
        "Starting agent-discovery"
    );

    match cli.command {
        Commands::Run { udp } => {
            agent::run(config, udp).await?;
        }
        Commands::ListDevices => {
            audio::device::list_devices()?;
        }
        Commands::TestTone { message } => {
            audio::device::play_test_tone(&config, &message)?;
        }
        Commands::Decode { duration } => {
            audio::device::decode_from_mic(&config, duration)?;
        }
        Commands::TestRoundtrip => {
            audio::device::test_roundtrip()?;
        }
    }

    Ok(())
}
