/// TCP client/server for post-discovery direct communication.

use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use super::channel::CommChannel;

/// A TCP-based communication channel.
pub struct TcpChannel {
    stream: Arc<Mutex<TcpStream>>,
    peer_addr: String,
}

impl TcpChannel {
    /// Connect to a peer at the given address.
    pub fn connect(addr: &str) -> Result<Self> {
        tracing::info!("Connecting to peer at {}", addr);
        let stream = TcpStream::connect(addr)
            .with_context(|| format!("Failed to connect to {}", addr))?;
        let peer_addr = addr.to_string();

        Ok(Self {
            stream: Arc::new(Mutex::new(stream)),
            peer_addr,
        })
    }

    /// Set a read timeout on the underlying stream.
    pub fn set_read_timeout(&self, duration: Option<std::time::Duration>) -> Result<()> {
        let stream = self.stream.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        stream.set_read_timeout(duration)?;
        Ok(())
    }

    /// Create from an accepted connection.
    pub fn from_stream(stream: TcpStream) -> Result<Self> {
        let peer_addr = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        Ok(Self {
            stream: Arc::new(Mutex::new(stream)),
            peer_addr,
        })
    }
}

impl CommChannel for TcpChannel {
    fn send_message(&self, data: &[u8]) -> Result<()> {
        let mut stream = self.stream.lock().map_err(|e| anyhow::anyhow!("{}", e))?;

        // Simple framing: 4-byte length prefix (big-endian) + data
        let len = data.len() as u32;
        stream.write_all(&len.to_be_bytes())?;
        stream.write_all(data)?;
        stream.flush()?;

        Ok(())
    }

    fn recv_message(&self) -> Result<Vec<u8>> {
        let mut stream = self.stream.lock().map_err(|e| anyhow::anyhow!("{}", e))?;

        // Read 4-byte length prefix
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;

        // Sanity check
        if len > 1024 * 1024 {
            anyhow::bail!("Message too large: {} bytes", len);
        }

        let mut data = vec![0u8; len];
        stream.read_exact(&mut data)?;

        Ok(data)
    }

    fn close(&self) -> Result<()> {
        let stream = self.stream.lock().map_err(|e| anyhow::anyhow!("{}", e))?;
        stream.shutdown(std::net::Shutdown::Both)?;
        Ok(())
    }

    fn description(&self) -> String {
        format!("TCP channel to {}", self.peer_addr)
    }
}

/// TCP listener that accepts incoming peer connections.
pub struct TcpCommListener {
    listener: TcpListener,
}

impl TcpCommListener {
    /// Start listening on the given port.
    pub fn bind(port: u16) -> Result<Self> {
        let addr = format!("0.0.0.0:{}", port);
        let listener = TcpListener::bind(&addr)
            .with_context(|| format!("Failed to bind to {}", addr))?;
        tracing::info!("TCP listener bound to {}", addr);

        Ok(Self { listener })
    }

    /// Try to bind to `preferred_port`. If it is already in use, try up to
    /// `max_attempts` consecutive ports before giving up.
    pub fn bind_with_fallback(preferred_port: u16, max_attempts: u16) -> Result<Self> {
        for offset in 0..max_attempts {
            let port = preferred_port.wrapping_add(offset);
            match Self::bind(port) {
                Ok(listener) => {
                    if offset > 0 {
                        tracing::info!(
                            preferred = preferred_port,
                            actual = port,
                            "Preferred port in use, bound to fallback port"
                        );
                    }
                    return Ok(listener);
                }
                Err(e) => {
                    if offset + 1 < max_attempts {
                        tracing::debug!(
                            port,
                            "Port unavailable, trying next: {}",
                            e
                        );
                    } else {
                        return Err(e).with_context(|| {
                            format!(
                                "Could not bind to any port in range {}..{}",
                                preferred_port,
                                preferred_port.wrapping_add(max_attempts)
                            )
                        });
                    }
                }
            }
        }
        unreachable!()
    }

    /// Accept a single connection (blocking).
    pub fn accept(&self) -> Result<TcpChannel> {
        let (stream, addr) = self.listener.accept().context("Failed to accept connection")?;
        tracing::info!("Accepted connection from {}", addr);
        TcpChannel::from_stream(stream)
    }

    /// Get the local address.
    pub fn local_addr(&self) -> Result<std::net::SocketAddr> {
        self.listener.local_addr().context("Failed to get local address")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_tcp_channel_roundtrip() {
        let listener = TcpCommListener::bind(0).unwrap(); // port 0 = OS picks
        let addr = listener.local_addr().unwrap();

        let server = thread::spawn(move || {
            let channel = listener.accept().unwrap();
            let msg = channel.recv_message().unwrap();
            channel.send_message(&msg).unwrap(); // echo back
            msg
        });

        let client = TcpChannel::connect(&addr.to_string()).unwrap();
        let test_data = b"Hello from agent!";
        client.send_message(test_data).unwrap();
        let response = client.recv_message().unwrap();

        assert_eq!(test_data.as_slice(), response.as_slice());

        let echoed = server.join().unwrap();
        assert_eq!(test_data.as_slice(), echoed.as_slice());
    }
}
