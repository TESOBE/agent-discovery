/// Discovery message types.
///
/// Binary payload format:
/// [Version 1B] [MsgType 1B] [AgentID 16B (UUID)] [AddrLen 1B] [Address NB (UTF-8)] [Capabilities 1B]

use uuid::Uuid;

/// Protocol version
pub const PROTOCOL_VERSION: u8 = 1;

/// Message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    Announce = 0x01,
    Ack = 0x02,
    Goodbye = 0x03,
}

impl MessageType {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Announce),
            0x02 => Some(Self::Ack),
            0x03 => Some(Self::Goodbye),
            _ => None,
        }
    }
}

/// Capability bits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities(pub u8);

impl Capabilities {
    pub const TCP: u8 = 0b0000_0001;
    pub const HTTP: u8 = 0b0000_0010;
    pub const OBP: u8 = 0b0000_0100;
    pub const CLAUDE_NEGOTIATION: u8 = 0b0000_1000;
    pub const AUDIO: u8 = 0b0001_0000;

    pub fn new() -> Self {
        Self(0)
    }

    pub fn with_tcp(mut self) -> Self {
        self.0 |= Self::TCP;
        self
    }

    pub fn with_http(mut self) -> Self {
        self.0 |= Self::HTTP;
        self
    }

    pub fn with_obp(mut self) -> Self {
        self.0 |= Self::OBP;
        self
    }

    pub fn with_claude(mut self) -> Self {
        self.0 |= Self::CLAUDE_NEGOTIATION;
        self
    }

    pub fn with_audio(mut self) -> Self {
        self.0 |= Self::AUDIO;
        self
    }

    pub fn has_tcp(self) -> bool {
        self.0 & Self::TCP != 0
    }

    pub fn has_http(self) -> bool {
        self.0 & Self::HTTP != 0
    }

    pub fn has_obp(self) -> bool {
        self.0 & Self::OBP != 0
    }

    pub fn has_claude(self) -> bool {
        self.0 & Self::CLAUDE_NEGOTIATION != 0
    }

    pub fn has_audio(self) -> bool {
        self.0 & Self::AUDIO != 0
    }

    /// Human-readable description of enabled capabilities.
    pub fn describe(self) -> String {
        let mut parts = Vec::new();
        if self.has_tcp() { parts.push("TCP"); }
        if self.has_http() { parts.push("HTTP"); }
        if self.has_obp() { parts.push("OBP"); }
        if self.has_claude() { parts.push("Claude"); }
        if self.has_audio() { parts.push("audio"); }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join("+")
        }
    }
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::new()
    }
}

/// A discovery message.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveryMessage {
    pub version: u8,
    pub msg_type: MessageType,
    pub agent_id: Uuid,
    pub address: String,
    pub capabilities: Capabilities,
}

impl DiscoveryMessage {
    /// Create a new Announce message.
    pub fn announce(agent_id: Uuid, address: &str, capabilities: Capabilities) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            msg_type: MessageType::Announce,
            agent_id,
            address: address.to_string(),
            capabilities,
        }
    }

    /// Create a new Ack message.
    pub fn ack(agent_id: Uuid, address: &str, capabilities: Capabilities) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            msg_type: MessageType::Ack,
            agent_id,
            address: address.to_string(),
            capabilities,
        }
    }

    /// Create a new Goodbye message.
    pub fn goodbye(agent_id: Uuid) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            msg_type: MessageType::Goodbye,
            agent_id,
            address: String::new(),
            capabilities: Capabilities::new(),
        }
    }
}
