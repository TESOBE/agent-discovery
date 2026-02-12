/// Binary serialization/deserialization of discovery messages.

use super::message::{Capabilities, DiscoveryMessage, MessageType, PROTOCOL_VERSION};
use uuid::Uuid;

/// Serialize a DiscoveryMessage to bytes.
pub fn encode_message(msg: &DiscoveryMessage) -> Vec<u8> {
    let addr_bytes = msg.address.as_bytes();
    let mut buf = Vec::with_capacity(1 + 1 + 16 + 1 + addr_bytes.len() + 1);

    buf.push(msg.version);
    buf.push(msg.msg_type as u8);
    buf.extend_from_slice(msg.agent_id.as_bytes());
    buf.push(addr_bytes.len() as u8);
    buf.extend_from_slice(addr_bytes);
    buf.push(msg.capabilities.0);

    buf
}

/// Deserialize a DiscoveryMessage from bytes.
pub fn decode_message(data: &[u8]) -> Option<DiscoveryMessage> {
    // Minimum: version(1) + type(1) + uuid(16) + addr_len(1) + capabilities(1) = 20
    if data.len() < 20 {
        return None;
    }

    let version = data[0];
    if version != PROTOCOL_VERSION {
        tracing::warn!("Unknown protocol version: {}", version);
        return None;
    }

    let msg_type = MessageType::from_byte(data[1])?;

    let agent_id = Uuid::from_slice(&data[2..18]).ok()?;

    let addr_len = data[18] as usize;
    if data.len() < 19 + addr_len + 1 {
        return None;
    }

    let address = std::str::from_utf8(&data[19..19 + addr_len]).ok()?.to_string();

    let capabilities = Capabilities(data[19 + addr_len]);

    Some(DiscoveryMessage {
        version,
        msg_type,
        agent_id,
        address,
        capabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_roundtrip_announce() {
        let id = Uuid::new_v4();
        let msg = DiscoveryMessage::announce(
            id,
            "192.168.1.100:9000",
            Capabilities::new().with_tcp().with_claude(),
        );

        let encoded = encode_message(&msg);
        let decoded = decode_message(&encoded).expect("Failed to decode");

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_message_roundtrip_ack() {
        let id = Uuid::new_v4();
        let msg = DiscoveryMessage::ack(
            id,
            "10.0.0.1:8080",
            Capabilities::new().with_http().with_obp(),
        );

        let encoded = encode_message(&msg);
        let decoded = decode_message(&encoded).expect("Failed to decode");

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_message_roundtrip_goodbye() {
        let id = Uuid::new_v4();
        let msg = DiscoveryMessage::goodbye(id);

        let encoded = encode_message(&msg);
        let decoded = decode_message(&encoded).expect("Failed to decode");

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_decode_truncated_data() {
        assert!(decode_message(&[0x01, 0x01]).is_none());
    }

    #[test]
    fn test_decode_bad_version() {
        let id = Uuid::new_v4();
        let mut msg = DiscoveryMessage::announce(id, "x", Capabilities::new());
        msg.version = 99;
        let encoded = encode_message(&msg);
        assert!(decode_message(&encoded).is_none());
    }

    #[test]
    fn test_capabilities_bitmask() {
        let caps = Capabilities::new()
            .with_tcp()
            .with_obp()
            .with_claude();

        assert!(caps.has_tcp());
        assert!(!caps.has_http());
        assert!(caps.has_obp());
        assert!(caps.has_claude());
        assert_eq!(caps.0, 0b0000_1101);
    }

    #[test]
    fn test_full_pipeline_frame_and_message() {
        use crate::protocol::frame;

        let id = Uuid::new_v4();
        let msg = DiscoveryMessage::announce(
            id,
            "192.168.1.42:9000",
            Capabilities::new().with_tcp().with_obp().with_claude(),
        );

        // Encode message -> frame -> decode frame -> decode message
        let msg_bytes = encode_message(&msg);
        let frame_bytes = frame::encode_frame(&msg_bytes);
        let payload = frame::decode_frame(&frame_bytes).expect("Frame decode failed");
        let decoded_msg = decode_message(&payload).expect("Message decode failed");

        assert_eq!(msg, decoded_msg);
    }
}
