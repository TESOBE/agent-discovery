/// FSK Demodulator: converts audio samples back to bits using the Goertzel algorithm.
///
/// Uses Goertzel to measure energy at mark (1200 Hz) and space (2200 Hz) frequencies,
/// then decides each bit based on which frequency has more energy.

use super::modulator::{MARK_FREQ, SAMPLE_RATE, SAMPLES_PER_BIT, SPACE_FREQ};

/// Compute the Goertzel magnitude for a specific frequency over a block of samples.
fn goertzel_mag(samples: &[f32], target_freq: f32, sample_rate: u32) -> f32 {
    let n = samples.len() as f32;
    let k = (target_freq * n / sample_rate as f32).round();
    let w = 2.0 * std::f32::consts::PI * k / n;
    let coeff = 2.0 * w.cos();

    let mut s0: f32;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;

    for &sample in samples {
        s0 = sample + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    // Magnitude squared
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

/// Demodulate audio samples into bits.
///
/// Processes samples in blocks of `SAMPLES_PER_BIT`, using Goertzel to determine
/// whether each block contains a mark (1) or space (0) tone.
pub fn demodulate_samples(samples: &[f32]) -> Vec<u8> {
    let block_size = SAMPLES_PER_BIT as usize;
    let mut bits = Vec::new();

    for chunk in samples.chunks(block_size) {
        if chunk.len() < block_size {
            break; // discard partial final block
        }

        let mark_energy = goertzel_mag(chunk, MARK_FREQ, SAMPLE_RATE);
        let space_energy = goertzel_mag(chunk, SPACE_FREQ, SAMPLE_RATE);

        bits.push(if mark_energy > space_energy { 1 } else { 0 });
    }

    bits
}

/// Demodulate samples and convert to bytes.
pub fn demodulate_to_bytes(samples: &[f32]) -> Vec<u8> {
    let bits = demodulate_samples(samples);
    super::modulator::bits_to_bytes(&bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::modulator::{modulate_bits, modulate_bytes};

    #[test]
    fn test_fsk_roundtrip_bits() {
        let original_bits: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let samples = modulate_bits(&original_bits);
        let recovered_bits = demodulate_samples(&samples);
        assert_eq!(
            original_bits, recovered_bits,
            "Bit round-trip failed: expected {:?}, got {:?}",
            original_bits, recovered_bits
        );
    }

    #[test]
    fn test_fsk_roundtrip_bytes() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let samples = modulate_bytes(&original);
        let recovered = demodulate_to_bytes(&samples);
        assert_eq!(
            original, recovered,
            "Byte round-trip failed: expected {:02X?}, got {:02X?}",
            original, recovered
        );
    }

    #[test]
    fn test_fsk_roundtrip_all_zeros() {
        let original = vec![0x00, 0x00];
        let samples = modulate_bytes(&original);
        let recovered = demodulate_to_bytes(&samples);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_fsk_roundtrip_all_ones() {
        let original = vec![0xFF, 0xFF];
        let samples = modulate_bytes(&original);
        let recovered = demodulate_to_bytes(&samples);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_goertzel_detects_correct_frequency() {
        // Generate a pure 1200 Hz tone
        let n = SAMPLES_PER_BIT as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (2.0 * std::f32::consts::PI * MARK_FREQ * t).sin()
            })
            .collect();

        let mark_energy = goertzel_mag(&samples, MARK_FREQ, SAMPLE_RATE);
        let space_energy = goertzel_mag(&samples, SPACE_FREQ, SAMPLE_RATE);

        assert!(
            mark_energy > space_energy * 2.0,
            "Mark energy ({}) should be much greater than space energy ({})",
            mark_energy,
            space_energy
        );
    }
}
