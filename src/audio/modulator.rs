/// FSK Modulator: converts bits to audio samples using Bell 202 frequencies.
///
/// - Mark (1): 1200 Hz
/// - Space (0): 2200 Hz
/// - Sample rate: 48000 Hz
/// - Baud rate: 300 bps (160 samples per bit)
/// - Continuous-phase modulation

use std::f32::consts::TAU;

pub const SAMPLE_RATE: u32 = 48000;
pub const BAUD_RATE: u32 = 300;
pub const SAMPLES_PER_BIT: u32 = SAMPLE_RATE / BAUD_RATE; // 160
pub const MARK_FREQ: f32 = 1200.0; // binary 1
pub const SPACE_FREQ: f32 = 2200.0; // binary 0

/// Modulate a slice of bits into audio samples using continuous-phase FSK.
pub fn modulate_bits(bits: &[u8]) -> Vec<f32> {
    let mut samples = Vec::with_capacity(bits.len() * SAMPLES_PER_BIT as usize);
    let mut phase: f32 = 0.0;

    for &bit in bits {
        let freq = if bit != 0 { MARK_FREQ } else { SPACE_FREQ };
        let phase_inc = TAU * freq / SAMPLE_RATE as f32;

        for _ in 0..SAMPLES_PER_BIT {
            samples.push(phase.sin() * 0.8); // amplitude 0.8 to avoid clipping
            phase += phase_inc;
            if phase >= TAU {
                phase -= TAU;
            }
        }
    }

    samples
}

/// Convenience: modulate raw bytes (MSB first per byte).
pub fn modulate_bytes(data: &[u8]) -> Vec<f32> {
    let bits = bytes_to_bits(data);
    modulate_bits(&bits)
}

/// Convert bytes to a bit vector (MSB first).
pub fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for &byte in data {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1);
        }
    }
    bits
}

/// Convert a bit vector back to bytes (MSB first). Pads with zeros if not a multiple of 8.
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= (bit & 1) << (7 - i);
            }
            byte
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_bits_roundtrip() {
        let data = vec![0xAA, 0x55, 0xFF, 0x00];
        let bits = bytes_to_bits(&data);
        let recovered = bits_to_bytes(&bits);
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_modulate_produces_correct_sample_count() {
        let bits = vec![1, 0, 1, 1, 0];
        let samples = modulate_bits(&bits);
        assert_eq!(samples.len(), 5 * SAMPLES_PER_BIT as usize);
    }

    #[test]
    fn test_modulate_amplitude_within_bounds() {
        let bits = vec![1, 0, 1, 0];
        let samples = modulate_bits(&bits);
        for &s in &samples {
            assert!(s >= -1.0 && s <= 1.0, "Sample out of bounds: {}", s);
        }
    }

    #[test]
    fn test_continuous_phase() {
        // Check that the phase is continuous at bit boundaries
        let bits = vec![1, 0]; // transition from mark to space
        let samples = modulate_bits(&bits);
        let boundary = SAMPLES_PER_BIT as usize;
        // The sample at the boundary should be close to the sample that would
        // result from continuous phase (no discontinuity)
        let diff = (samples[boundary] - samples[boundary - 1]).abs();
        // At 48kHz, adjacent samples should differ by at most the max slope
        assert!(diff < 0.2, "Phase discontinuity detected: diff={}", diff);
    }
}
