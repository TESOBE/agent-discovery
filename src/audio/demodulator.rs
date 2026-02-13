/// FSK Demodulator: converts audio samples back to bits using the Goertzel algorithm.
///
/// Supports configurable frequency pairs so agents can listen on a
/// different channel than they transmit on.

use super::modulator::{FskFreqs, CHANNEL_1, SAMPLE_RATE, SAMPLES_PER_BIT};

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

/// Demodulate audio samples into bits using the given frequency pair, starting at a sample offset.
pub fn demodulate_samples_freq_at(samples: &[f32], freqs: &FskFreqs, offset: usize) -> Vec<u8> {
    if offset >= samples.len() {
        return Vec::new();
    }
    let samples = &samples[offset..];
    let block_size = SAMPLES_PER_BIT as usize;
    let mut bits = Vec::new();

    for chunk in samples.chunks(block_size) {
        if chunk.len() < block_size {
            break;
        }

        let mark_energy = goertzel_mag(chunk, freqs.mark, SAMPLE_RATE);
        let space_energy = goertzel_mag(chunk, freqs.space, SAMPLE_RATE);

        bits.push(if mark_energy > space_energy { 1 } else { 0 });
    }

    bits
}

/// Demodulate audio samples into bits using the given frequency pair (offset 0).
pub fn demodulate_samples_freq(samples: &[f32], freqs: &FskFreqs) -> Vec<u8> {
    demodulate_samples_freq_at(samples, freqs, 0)
}

/// Demodulate audio samples into bits using default Bell 202 frequencies.
pub fn demodulate_samples(samples: &[f32]) -> Vec<u8> {
    demodulate_samples_freq(samples, &CHANNEL_1)
}

/// Demodulate samples to bytes using the given frequency pair.
pub fn demodulate_to_bytes_freq(samples: &[f32], freqs: &FskFreqs) -> Vec<u8> {
    let bits = demodulate_samples_freq(samples, freqs);
    super::modulator::bits_to_bytes(&bits)
}

/// Demodulate samples to bytes using default Bell 202 frequencies.
pub fn demodulate_to_bytes(samples: &[f32]) -> Vec<u8> {
    let bits = demodulate_samples(samples);
    super::modulator::bits_to_bytes(&bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::modulator::{modulate_bits, modulate_bytes, modulate_bytes_freq, CHANNEL_2, MARK_FREQ, SPACE_FREQ};

    #[test]
    fn test_fsk_roundtrip_bits() {
        let original_bits: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let samples = modulate_bits(&original_bits);
        let recovered_bits = demodulate_samples(&samples);
        assert_eq!(original_bits, recovered_bits);
    }

    #[test]
    fn test_fsk_roundtrip_bytes() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let samples = modulate_bytes(&original);
        let recovered = demodulate_to_bytes(&samples);
        assert_eq!(original, recovered);
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

    #[test]
    fn test_channel2_roundtrip() {
        let original = vec![0xCA, 0xFE];
        let samples = modulate_bytes_freq(&original, &CHANNEL_2);
        let recovered = demodulate_to_bytes_freq(&samples, &CHANNEL_2);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_channels_dont_interfere() {
        // Modulate on channel 1, try to demodulate on channel 2 - should NOT match
        let original = vec![0xAB, 0xCD];
        let samples = modulate_bytes_freq(&original, &CHANNEL_1);
        let recovered = demodulate_to_bytes_freq(&samples, &CHANNEL_2);
        assert_ne!(original, recovered, "Channels should not cross-decode");
    }

    #[test]
    fn test_all_bands_roundtrip() {
        use crate::audio::modulator::BANDS;
        let original = vec![0xDE, 0xAD];
        for (i, band) in BANDS.iter().enumerate() {
            let samples = modulate_bytes_freq(&original, band);
            let recovered = demodulate_to_bytes_freq(&samples, band);
            assert_eq!(original, recovered, "Band {} failed roundtrip", i);
        }
    }
}
