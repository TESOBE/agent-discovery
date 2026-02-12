/// Integration test: FSK modulation round-trip through the full pipeline.

use agent_discovery::audio::demodulator;
use agent_discovery::audio::modulator;

#[test]
fn test_fsk_roundtrip_short_message() {
    let message = b"HELLO";
    let samples = modulator::modulate_bytes(message);
    let recovered = demodulator::demodulate_to_bytes(&samples);
    assert_eq!(message.as_slice(), &recovered[..message.len()]);
}

#[test]
fn test_fsk_roundtrip_binary_data() {
    let data: Vec<u8> = (0..=255).collect();
    let samples = modulator::modulate_bytes(&data);
    let recovered = demodulator::demodulate_to_bytes(&samples);
    assert_eq!(data, recovered);
}

#[test]
fn test_fsk_roundtrip_with_noise() {
    let message = b"SIGNAL";
    let mut samples = modulator::modulate_bytes(message);

    // Add small amount of noise
    for (i, s) in samples.iter_mut().enumerate() {
        let noise = ((i as f32 * 0.1).sin()) * 0.05;
        *s += noise;
    }

    let recovered = demodulator::demodulate_to_bytes(&samples);
    assert_eq!(message.as_slice(), &recovered[..message.len()]);
}
