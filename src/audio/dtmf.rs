/// DTMF (Dual-Tone Multi-Frequency) audio encoding/decoding.
///
/// Each symbol is two simultaneous sine waves from different frequency groups.
/// This is the same encoding used by telephone keypads — extremely robust
/// over acoustic paths because detection only needs to identify which 2 of 8
/// frequencies are present, not decode individual bits.
///
/// We use DTMF to transmit the agent's listen port and capabilities as
/// a short tone sequence (~1.5 seconds).

use std::f32::consts::TAU;

/// Low frequency group (rows of the DTMF keypad).
const LOW_FREQS: [f32; 4] = [697.0, 770.0, 852.0, 941.0];
/// High frequency group (columns of the DTMF keypad).
const HIGH_FREQS: [f32; 4] = [1209.0, 1336.0, 1477.0, 1633.0];

/// Standard DTMF symbol grid: low_freq × high_freq.
/// Row 0: 1, 2, 3, A
/// Row 1: 4, 5, 6, B
/// Row 2: 7, 8, 9, C
/// Row 3: *, 0, #, D
const DTMF_GRID: [[char; 4]; 4] = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D'],
];

/// Duration of each DTMF tone in seconds.
/// Longer than standard (40ms) for robustness over acoustic path with reverb.
const TONE_DURATION: f32 = 0.18; // 180ms
/// Silence gap between tones in seconds.
/// Longer gap lets reverb die down before next tone.
const GAP_DURATION: f32 = 0.12; // 120ms
/// Amplitude of each frequency component.
const TONE_AMPLITUDE: f32 = 0.4;

/// Preamble sequence to mark start of DTMF data.
pub const PREAMBLE: &[char] = &['*', '#', '*', '#'];
/// Postamble sequence to mark end of DTMF data.
pub const POSTAMBLE: &[char] = &['#', '*', '#', '*'];

/// Look up the frequency pair for a DTMF symbol.
fn symbol_to_freqs(symbol: char) -> Option<(f32, f32)> {
    for (row, low_freq) in LOW_FREQS.iter().enumerate() {
        for (col, high_freq) in HIGH_FREQS.iter().enumerate() {
            if DTMF_GRID[row][col] == symbol {
                return Some((*low_freq, *high_freq));
            }
        }
    }
    None
}

/// Look up the symbol for a frequency pair (by row/col index).
fn indices_to_symbol(row: usize, col: usize) -> Option<char> {
    if row < 4 && col < 4 {
        Some(DTMF_GRID[row][col])
    } else {
        None
    }
}

/// Generate DTMF audio for a sequence of symbols.
pub fn encode_dtmf(symbols: &[char], sample_rate: u32) -> Vec<f32> {
    let tone_samples = (TONE_DURATION * sample_rate as f32) as usize;
    let gap_samples = (GAP_DURATION * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(symbols.len() * (tone_samples + gap_samples));

    for &symbol in symbols {
        if let Some((low, high)) = symbol_to_freqs(symbol) {
            // Generate dual-tone
            for i in 0..tone_samples {
                let t = i as f32 / sample_rate as f32;
                let env = tone_envelope(i, tone_samples);
                let sample = ((TAU * low * t).sin() + (TAU * high * t).sin()) * TONE_AMPLITUDE * env;
                samples.push(sample);
            }
        }
        // Gap (silence)
        for _ in 0..gap_samples {
            samples.push(0.0);
        }
    }

    samples
}

/// Smooth envelope for each tone to avoid clicks.
fn tone_envelope(i: usize, total: usize) -> f32 {
    let fade = (total as f32 * 0.08) as usize; // 8% fade
    if fade == 0 {
        return 1.0;
    }
    if i < fade {
        i as f32 / fade as f32
    } else if i > total - fade {
        (total - i) as f32 / fade as f32
    } else {
        1.0
    }
}

/// Encode a discovery payload as DTMF symbols: preamble + port digits + D + caps digits + postamble.
/// Port and capabilities are both encoded as plain decimal digits, separated by 'D'.
pub fn encode_discovery_payload(port: u16, capabilities: u8) -> Vec<char> {
    let mut symbols = Vec::new();

    // Preamble
    symbols.extend_from_slice(PREAMBLE);

    // Port as decimal digits
    for ch in port.to_string().chars() {
        symbols.push(ch);
    }

    // Separator
    symbols.push('D');

    // Capabilities as decimal digits (0-255)
    for ch in capabilities.to_string().chars() {
        symbols.push(ch);
    }

    // Postamble
    symbols.extend_from_slice(POSTAMBLE);

    symbols
}

/// Decode a DTMF symbol sequence back into port and capabilities.
pub fn decode_discovery_payload(symbols: &[char]) -> Option<(u16, u8)> {
    // Find preamble
    let preamble_len = PREAMBLE.len();
    let postamble_len = POSTAMBLE.len();

    let preamble_pos = symbols
        .windows(preamble_len)
        .position(|w| w == PREAMBLE)?;

    let data_start = preamble_pos + preamble_len;

    // Find postamble after preamble
    let postamble_pos = symbols[data_start..]
        .windows(postamble_len)
        .position(|w| w == POSTAMBLE)
        .map(|p| p + data_start)?;

    let data = &symbols[data_start..postamble_pos];

    // Split on 'D' separator
    let sep_pos = data.iter().position(|&c| c == 'D')?;
    let port_chars = &data[..sep_pos];
    let caps_chars = &data[sep_pos + 1..];

    // Parse port
    let port_str: String = port_chars.iter().collect();
    let port: u16 = port_str.parse().ok()?;

    // Parse capabilities
    let caps_str: String = caps_chars.iter().collect();
    let capabilities: u8 = caps_str.parse().ok()?;

    Some((port, capabilities))
}

/// Map a nibble (0-15) to a DTMF symbol.
/// 0-9 → '0'-'9', 10 → 'A', 11 → 'B', 12 → 'C', 13 → '*', 14 → '#', 15 → '1'+'A' (rare)
/// Goertzel magnitude for a specific frequency over a block of samples.
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

    (s1 * s1 + s2 * s2 - coeff * s1 * s2).sqrt()
}

/// Decode DTMF symbols from audio samples.
/// Returns the sequence of detected symbols.
pub fn decode_dtmf(samples: &[f32], sample_rate: u32) -> Vec<char> {
    let tone_samples = (TONE_DURATION * sample_rate as f32) as usize;
    let gap_samples = (GAP_DURATION * sample_rate as f32) as usize;
    let step = tone_samples + gap_samples;

    // Minimum energy threshold to consider a tone present.
    // Lower threshold for acoustic path robustness.
    let energy_threshold = 0.15;

    // We don't know exactly where tones start, so try multiple offsets
    // within one step period (like we do for FSK)
    let best_symbols = try_decode_at_offset(samples, sample_rate, 0, tone_samples, step, energy_threshold);

    // Try multiple offsets and pick the one that produces the most valid symbols.
    // More offsets = better chance of aligning with the acoustic signal.
    let mut best = best_symbols;
    for frac in 1..16 {
        let offset = step * frac / 16;
        if offset >= samples.len() {
            break;
        }
        let candidate = try_decode_at_offset(samples, sample_rate, offset, tone_samples, step, energy_threshold);
        if candidate.len() > best.len() {
            best = candidate;
        }
    }

    best
}

fn try_decode_at_offset(
    samples: &[f32],
    sample_rate: u32,
    start_offset: usize,
    tone_samples: usize,
    step: usize,
    energy_threshold: f32,
) -> Vec<char> {
    let mut symbols = Vec::new();
    let mut pos = start_offset;

    while pos + tone_samples <= samples.len() {
        let block = &samples[pos..pos + tone_samples];

        // Check energy level - skip if too quiet
        let rms: f32 = (block.iter().map(|s| s * s).sum::<f32>() / block.len() as f32).sqrt();
        if rms < 0.01 {
            pos += step;
            continue;
        }

        // Compute Goertzel magnitude for all 8 DTMF frequencies
        let mut low_mags = [0.0f32; 4];
        let mut high_mags = [0.0f32; 4];

        for (i, &freq) in LOW_FREQS.iter().enumerate() {
            low_mags[i] = goertzel_mag(block, freq, sample_rate);
        }
        for (i, &freq) in HIGH_FREQS.iter().enumerate() {
            high_mags[i] = goertzel_mag(block, freq, sample_rate);
        }

        // Find the strongest in each group
        let (best_low_idx, best_low_mag) = low_mags
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let (best_high_idx, best_high_mag) = high_mags
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Both must exceed threshold
        if *best_low_mag > energy_threshold && *best_high_mag > energy_threshold {
            // Check that the strongest is significantly stronger than the second-strongest
            // (twist check — prevents false positives from broadband noise)
            let mut low_sorted: Vec<f32> = low_mags.to_vec();
            low_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let mut high_sorted: Vec<f32> = high_mags.to_vec();
            high_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

            let low_ratio = if low_sorted[1] > 0.0 { low_sorted[0] / low_sorted[1] } else { 100.0 };
            let high_ratio = if high_sorted[1] > 0.0 { high_sorted[0] / high_sorted[1] } else { 100.0 };

            // Require 1.5x dominance (relaxed for acoustic path)
            if low_ratio > 1.5 && high_ratio > 1.5 {
                if let Some(symbol) = indices_to_symbol(best_low_idx, best_high_idx) {
                    symbols.push(symbol);
                }
            }
        }

        pos += step;
    }

    symbols
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::modulator::SAMPLE_RATE;

    #[test]
    fn test_symbol_to_freqs_all_symbols() {
        // Verify all 16 symbols have frequency mappings
        for row in &DTMF_GRID {
            for &symbol in row {
                assert!(
                    symbol_to_freqs(symbol).is_some(),
                    "Symbol '{}' should have frequencies",
                    symbol
                );
            }
        }
    }

    #[test]
    fn test_encode_decode_single_tone() {
        let symbols = vec!['5'];
        let samples = encode_dtmf(&symbols, SAMPLE_RATE);
        let decoded = decode_dtmf(&samples, SAMPLE_RATE);
        assert_eq!(decoded, symbols, "Single tone roundtrip failed");
    }

    #[test]
    fn test_encode_decode_digits() {
        let symbols: Vec<char> = "1234567890".chars().collect();
        let samples = encode_dtmf(&symbols, SAMPLE_RATE);
        let decoded = decode_dtmf(&samples, SAMPLE_RATE);
        assert_eq!(decoded, symbols, "Digit sequence roundtrip failed");
    }

    #[test]
    fn test_encode_decode_all_symbols() {
        let symbols: Vec<char> = "123A456B789C*0#D".chars().collect();
        let samples = encode_dtmf(&symbols, SAMPLE_RATE);
        let decoded = decode_dtmf(&samples, SAMPLE_RATE);
        assert_eq!(decoded, symbols, "All symbols roundtrip failed");
    }

    #[test]
    fn test_discovery_payload_roundtrip() {
        let symbols = encode_discovery_payload(9001, 0x0F);
        assert!(symbols.starts_with(PREAMBLE));
        assert!(symbols.ends_with(POSTAMBLE));

        let (port, caps) = decode_discovery_payload(&symbols).expect("Should decode");
        assert_eq!(port, 9001);
        assert_eq!(caps, 0x0F);
    }

    #[test]
    fn test_discovery_payload_audio_roundtrip() {
        let symbols = encode_discovery_payload(7312, 0x0B);
        let samples = encode_dtmf(&symbols, SAMPLE_RATE);
        let decoded_symbols = decode_dtmf(&samples, SAMPLE_RATE);
        let (port, caps) = decode_discovery_payload(&decoded_symbols).expect("Should decode");
        assert_eq!(port, 7312);
        assert_eq!(caps, 0x0B);
    }

    #[test]
    fn test_discovery_payload_various_ports() {
        for port in [80, 443, 7312, 9000, 9999, 65535] {
            let symbols = encode_discovery_payload(port, 0x0F);
            let samples = encode_dtmf(&symbols, SAMPLE_RATE);
            let decoded_symbols = decode_dtmf(&samples, SAMPLE_RATE);
            let result = decode_discovery_payload(&decoded_symbols);
            assert!(
                result.is_some(),
                "Should decode port {}. Symbols: {:?}, Decoded: {:?}",
                port, symbols, decoded_symbols
            );
            let (decoded_port, _) = result.unwrap();
            assert_eq!(decoded_port, port, "Port mismatch for {}", port);
        }
    }
}
