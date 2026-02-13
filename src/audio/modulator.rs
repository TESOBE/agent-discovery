/// FSK Modulator: converts bits to audio samples using continuous-phase FSK.
///
/// Each agent transmits on one of 9 frequency bands (chosen by name).
/// All agents listen on all bands so they can hear everyone.

use std::f32::consts::TAU;

pub const SAMPLE_RATE: u32 = 44100;
pub const BAUD_RATE: u32 = 300;
pub const SAMPLES_PER_BIT: u32 = SAMPLE_RATE / BAUD_RATE; // 147

/// Default Bell 202 frequencies (used by BANDS[0]).
pub const MARK_FREQ: f32 = 800.0;
pub const SPACE_FREQ: f32 = 1800.0;

/// A pair of FSK frequencies.
#[derive(Debug, Clone, Copy)]
pub struct FskFreqs {
    pub mark: f32,  // binary 1
    pub space: f32, // binary 0
}

/// 9 well-separated frequency bands. Each has 1000Hz mark/space separation
/// and 200Hz guard band between adjacent bands.
pub const BANDS: [FskFreqs; 9] = [
    FskFreqs { mark:   800.0, space:  1800.0 }, // Band 0
    FskFreqs { mark:  2000.0, space:  3000.0 }, // Band 1
    FskFreqs { mark:  3200.0, space:  4200.0 }, // Band 2
    FskFreqs { mark:  4400.0, space:  5400.0 }, // Band 3
    FskFreqs { mark:  5600.0, space:  6600.0 }, // Band 4
    FskFreqs { mark:  6800.0, space:  7800.0 }, // Band 5
    FskFreqs { mark:  8000.0, space:  9000.0 }, // Band 6
    FskFreqs { mark:  9200.0, space: 10200.0 }, // Band 7
    FskFreqs { mark: 10400.0, space: 11400.0 }, // Band 8
];

pub const NUM_BANDS: usize = BANDS.len();

/// Pick a band index from an agent name: use the last digit found, mod 9.
/// If no digit, hash the name.
pub fn band_from_name(name: &str) -> usize {
    name.chars()
        .rev()
        .find(|c| c.is_ascii_digit())
        .map(|c| (c as usize - '0' as usize) % NUM_BANDS)
        .unwrap_or_else(|| {
            let hash: usize = name.bytes().map(|b| b as usize).sum();
            hash % NUM_BANDS
        })
}

/// Legacy aliases for tests.
pub const CHANNEL_1: FskFreqs = BANDS[0];
pub const CHANNEL_2: FskFreqs = BANDS[1];

/// Modulate bits using the given frequency pair.
pub fn modulate_bits_freq(bits: &[u8], freqs: &FskFreqs) -> Vec<f32> {
    let mut samples = Vec::with_capacity(bits.len() * SAMPLES_PER_BIT as usize);
    let mut phase: f32 = 0.0;

    for &bit in bits {
        let freq = if bit != 0 { freqs.mark } else { freqs.space };
        let phase_inc = TAU * freq / SAMPLE_RATE as f32;

        for _ in 0..SAMPLES_PER_BIT {
            samples.push(phase.sin() * 0.8);
            phase += phase_inc;
            if phase >= TAU {
                phase -= TAU;
            }
        }
    }

    samples
}

/// Modulate bits using default Bell 202 frequencies.
pub fn modulate_bits(bits: &[u8]) -> Vec<f32> {
    modulate_bits_freq(bits, &CHANNEL_1)
}

/// Modulate bytes using the given frequency pair.
pub fn modulate_bytes_freq(data: &[u8], freqs: &FskFreqs) -> Vec<f32> {
    let bits = bytes_to_bits(data);
    modulate_bits_freq(&bits, freqs)
}

/// Modulate bytes using default Bell 202 frequencies.
pub fn modulate_bytes(data: &[u8]) -> Vec<f32> {
    modulate_bytes_freq(data, &CHANNEL_1)
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
        let bits = vec![1, 0];
        let samples = modulate_bits(&bits);
        let boundary = SAMPLES_PER_BIT as usize;
        let diff = (samples[boundary] - samples[boundary - 1]).abs();
        assert!(diff < 0.2, "Phase discontinuity detected: diff={}", diff);
    }

    #[test]
    fn test_channel2_produces_correct_sample_count() {
        let bits = vec![1, 0, 1];
        let samples = modulate_bits_freq(&bits, &CHANNEL_2);
        assert_eq!(samples.len(), 3 * SAMPLES_PER_BIT as usize);
    }

    #[test]
    fn test_band_from_name() {
        assert_eq!(band_from_name("mumma1"), 1);
        assert_eq!(band_from_name("mumma2"), 2);
        assert_eq!(band_from_name("mumma9"), 0); // 9 % 9 = 0
        assert_eq!(band_from_name("mumma0"), 0);
        assert_eq!(band_from_name("agent42"), 2); // last digit is 2
        // No digit - should use hash, just check it doesn't panic
        let _ = band_from_name("alpha");
        let _ = band_from_name("");
    }

    #[test]
    fn test_all_bands_produce_correct_sample_count() {
        let bits = vec![1, 0, 1];
        for (i, band) in BANDS.iter().enumerate() {
            let samples = modulate_bits_freq(&bits, band);
            assert_eq!(
                samples.len(),
                3 * SAMPLES_PER_BIT as usize,
                "Band {} produced wrong sample count",
                i
            );
        }
    }
}
