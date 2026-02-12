/// Frame encoding/decoding for the FSK protocol.
///
/// Frame format:
/// [Preamble 2B: 0xAAAA] [Sync 2B: 0x7E5A] [Length 1B] [Payload NB] [CRC-32 4B] [Tail 1B]

use crc32fast::Hasher;

pub const PREAMBLE: [u8; 2] = [0xAA, 0xAA];
pub const SYNC_WORD: [u8; 2] = [0x7E, 0x5A];
pub const TAIL: u8 = 0x55;
pub const MAX_PAYLOAD_LEN: usize = 255;

/// Encode a payload into a complete frame.
pub fn encode_frame(payload: &[u8]) -> Vec<u8> {
    assert!(
        payload.len() <= MAX_PAYLOAD_LEN,
        "Payload too large: {} > {}",
        payload.len(),
        MAX_PAYLOAD_LEN
    );

    let mut frame = Vec::with_capacity(2 + 2 + 1 + payload.len() + 4 + 1);

    // Preamble
    frame.extend_from_slice(&PREAMBLE);
    // Sync word
    frame.extend_from_slice(&SYNC_WORD);
    // Length
    frame.push(payload.len() as u8);
    // Payload
    frame.extend_from_slice(payload);
    // CRC-32 over length + payload
    let crc = compute_crc(payload.len() as u8, payload);
    frame.extend_from_slice(&crc.to_be_bytes());
    // Tail
    frame.push(TAIL);

    frame
}

/// Compute CRC-32 over the length byte and payload.
fn compute_crc(length: u8, payload: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(&[length]);
    hasher.update(payload);
    hasher.finalize()
}

/// Frame decoder state machine.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DecoderState {
    Hunting,    // Looking for preamble/sync
    Sync1,      // Found first sync byte (0x7E), looking for second (0x5A)
    Receiving,  // Reading length + payload + CRC
}

/// Attempt to decode a frame from raw bytes. Returns the payload if valid.
///
/// Scans for the sync word, then reads length, payload, CRC, and validates.
pub fn decode_frame(data: &[u8]) -> Option<Vec<u8>> {
    let mut state = DecoderState::Hunting;
    let mut i = 0;

    while i < data.len() {
        match state {
            DecoderState::Hunting => {
                if data[i] == SYNC_WORD[0] {
                    state = DecoderState::Sync1;
                }
                i += 1;
            }
            DecoderState::Sync1 => {
                if data[i] == SYNC_WORD[1] {
                    state = DecoderState::Receiving;
                    i += 1;
                } else if data[i] == SYNC_WORD[0] {
                    // Stay in Sync1 (could be repeated 0x7E)
                    i += 1;
                } else {
                    state = DecoderState::Hunting;
                    i += 1;
                }
            }
            DecoderState::Receiving => {
                // Need at least: length(1) + payload(N) + crc(4) + tail(1)
                if i >= data.len() {
                    return None;
                }

                let length = data[i] as usize;
                i += 1;

                // Check we have enough data remaining
                if i + length + 4 + 1 > data.len() {
                    return None; // Incomplete frame
                }

                let payload = &data[i..i + length];
                i += length;

                let crc_bytes = &data[i..i + 4];
                let received_crc =
                    u32::from_be_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
                i += 4;

                let _tail = data[i];
                // We don't strictly enforce tail value

                let computed_crc = compute_crc(length as u8, payload);

                if received_crc == computed_crc {
                    return Some(payload.to_vec());
                } else {
                    tracing::warn!(
                        "CRC mismatch: received={:#010X}, computed={:#010X}",
                        received_crc,
                        computed_crc
                    );
                    // Continue scanning for another sync word
                    state = DecoderState::Hunting;
                }
            }
        }
    }

    None
}

/// Streaming frame decoder that accumulates bytes and emits complete frames.
pub struct FrameDecoder {
    buffer: Vec<u8>,
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(512),
        }
    }

    /// Feed bytes into the decoder. Returns any complete frames found.
    pub fn feed(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        self.buffer.extend_from_slice(data);
        let mut frames = Vec::new();

        loop {
            // Try to find and decode a frame
            match decode_frame(&self.buffer) {
                Some(payload) => {
                    frames.push(payload);
                    // Remove consumed bytes up through this frame
                    // Find the frame end by scanning for sync word + length + payload + crc + tail
                    if let Some(sync_pos) = find_sync(&self.buffer) {
                        let length = self.buffer[sync_pos + 2] as usize;
                        let frame_end = sync_pos + 2 + 1 + length + 4 + 1;
                        if frame_end <= self.buffer.len() {
                            self.buffer.drain(..frame_end);
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }

        // Prevent unbounded buffer growth: trim old data
        if self.buffer.len() > 4096 {
            let trim = self.buffer.len() - 2048;
            self.buffer.drain(..trim);
        }

        frames
    }
}

/// Find the position of the sync word in the buffer.
fn find_sync(data: &[u8]) -> Option<usize> {
    data.windows(2)
        .position(|w| w[0] == SYNC_WORD[0] && w[1] == SYNC_WORD[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_roundtrip() {
        let payload = b"Hello, Agent!";
        let frame = encode_frame(payload);
        let decoded = decode_frame(&frame).expect("Failed to decode frame");
        assert_eq!(payload.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_frame_roundtrip_empty_payload() {
        let payload = b"";
        let frame = encode_frame(payload);
        let decoded = decode_frame(&frame).expect("Failed to decode frame");
        assert_eq!(payload.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_frame_crc_corruption() {
        let payload = b"Test data";
        let mut frame = encode_frame(payload);
        // Corrupt a byte in the payload area
        let payload_start = 2 + 2 + 1; // preamble + sync + length
        frame[payload_start] ^= 0xFF;
        assert!(decode_frame(&frame).is_none(), "Should reject corrupted frame");
    }

    #[test]
    fn test_frame_with_leading_garbage() {
        let payload = b"FindMe";
        let frame = encode_frame(payload);
        let mut with_garbage = vec![0x12, 0x34, 0x56, 0x78, 0x9A];
        with_garbage.extend_from_slice(&frame);
        let decoded = decode_frame(&with_garbage).expect("Should find frame in garbage");
        assert_eq!(payload.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_streaming_decoder() {
        let mut decoder = FrameDecoder::new();
        let payload1 = b"First";
        let payload2 = b"Second";

        let frame1 = encode_frame(payload1);
        let frame2 = encode_frame(payload2);

        let mut combined = frame1;
        combined.extend_from_slice(&frame2);

        let frames = decoder.feed(&combined);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], payload1.as_slice());
        assert_eq!(frames[1], payload2.as_slice());
    }

    #[test]
    fn test_frame_max_payload() {
        let payload = vec![0x42u8; MAX_PAYLOAD_LEN];
        let frame = encode_frame(&payload);
        let decoded = decode_frame(&frame).expect("Should decode max-size frame");
        assert_eq!(payload, decoded);
    }
}
