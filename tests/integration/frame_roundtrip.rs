/// Integration test: Frame protocol round-trip.

use agent_discovery::audio::{demodulator, modulator};
use agent_discovery::protocol::{codec, frame, message::*};
use uuid::Uuid;

#[test]
fn test_frame_through_fsk() {
    let payload = b"Test payload through FSK";

    // Encode: payload -> frame -> FSK samples -> demodulate -> decode frame
    let frame_bytes = frame::encode_frame(payload);
    let samples = modulator::modulate_bytes(&frame_bytes);
    let recovered_bytes = demodulator::demodulate_to_bytes(&samples);
    let decoded = frame::decode_frame(&recovered_bytes).expect("Frame decode failed");

    assert_eq!(payload.as_slice(), decoded.as_slice());
}

#[test]
fn test_discovery_message_through_full_pipeline() {
    let agent_id = Uuid::new_v4();
    let msg = DiscoveryMessage::announce(
        agent_id,
        "192.168.1.100:9000",
        Capabilities::new().with_tcp().with_obp().with_claude(),
        "https://apisandbox.openbankproject.com",
    );

    // Encode message -> frame -> FSK -> demodulate -> decode frame -> decode message
    let msg_bytes = codec::encode_message(&msg);
    let frame_bytes = frame::encode_frame(&msg_bytes);
    let samples = modulator::modulate_bytes(&frame_bytes);
    let recovered_bytes = demodulator::demodulate_to_bytes(&samples);
    let payload = frame::decode_frame(&recovered_bytes).expect("Frame decode failed");
    let decoded_msg = codec::decode_message(&payload).expect("Message decode failed");

    assert_eq!(msg.agent_id, decoded_msg.agent_id);
    assert_eq!(msg.msg_type, decoded_msg.msg_type);
    assert_eq!(msg.address, decoded_msg.address);
    assert_eq!(msg.capabilities, decoded_msg.capabilities);
}

#[test]
fn test_multiple_message_types_through_pipeline() {
    let agent_id = Uuid::new_v4();

    for msg in [
        DiscoveryMessage::announce(agent_id, "10.0.0.1:8080", Capabilities::new().with_tcp(), ""),
        DiscoveryMessage::ack(agent_id, "10.0.0.1:8080", Capabilities::new().with_http(), ""),
        DiscoveryMessage::goodbye(agent_id),
    ] {
        let msg_bytes = codec::encode_message(&msg);
        let frame_bytes = frame::encode_frame(&msg_bytes);
        let samples = modulator::modulate_bytes(&frame_bytes);
        let recovered_bytes = demodulator::demodulate_to_bytes(&samples);
        let payload = frame::decode_frame(&recovered_bytes).expect("Frame decode failed");
        let decoded = codec::decode_message(&payload).expect("Message decode failed");

        assert_eq!(msg.agent_id, decoded.agent_id);
        assert_eq!(msg.msg_type, decoded.msg_type);
    }
}
