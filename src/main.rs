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

#[derive(Parser)]
#[command(name = "agent-discovery", about = "Agent discovery via audio FSK tones")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the agent (discover peers, negotiate, communicate)
    Run,
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("agent_discovery=info")),
        )
        .init();

    let config = Config::from_env()?;
    tracing::info!(agent_name = %config.agent_name, "Starting agent-discovery");

    match cli.command {
        Commands::Run => {
            agent::run(config).await?;
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
    }

    Ok(())
}
