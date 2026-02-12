/// CommChannel trait abstraction for swappable transport layers.

use anyhow::Result;

/// Trait for a bidirectional communication channel between agents.
pub trait CommChannel: Send + Sync {
    /// Send a message to the peer.
    fn send_message(&self, data: &[u8]) -> Result<()>;

    /// Receive a message from the peer (blocking).
    fn recv_message(&self) -> Result<Vec<u8>>;

    /// Close the channel.
    fn close(&self) -> Result<()>;

    /// Get a description of this channel.
    fn description(&self) -> String;
}
