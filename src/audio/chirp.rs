/// Chirp-based audio discovery and binary chirp messaging.
///
/// A chirp sweeps from one frequency to another over a short time. Detection uses
/// normalized cross-correlation with a known template, which is much more robust
/// than FSK bit-level decoding for noisy acoustic paths.
///
/// **Discovery chirp:** Long sweep (150ms) for call-and-response presence detection.
///
/// **Binary chirp messaging:** After the chirp handshake, data is sent as a
/// sequence of short chirps: up-chirp (800→3200 Hz) = bit 1, down-chirp
/// (3200→800 Hz) = bit 0. This encodes discovery data (port, capabilities)
/// using the same robust chirp detection that works for presence detection.

use std::f32::consts::TAU;

/// Chirp parameters for agent discovery (CALL chirp).
pub const CHIRP_START_FREQ: f32 = 800.0;
pub const CHIRP_END_FREQ: f32 = 3200.0;
pub const CHIRP_DURATION_SECS: f32 = 0.15;
pub const CHIRP_AMPLITUDE: f32 = 0.5;

/// Response chirp frequency band (distinct from CALL to avoid confusion).
pub const RESPONSE_START_FREQ: f32 = 4000.0;
pub const RESPONSE_END_FREQ: f32 = 6400.0;

/// Progressive volume ramp for TX chirps.
/// Starts at a moderate level for faster discovery.
pub const CHIRP_AMPLITUDE_MIN: f32 = 0.50;
pub const CHIRP_AMPLITUDE_MAX: f32 = 0.80;
pub const CHIRP_RAMP_STEPS: u64 = 3;

/// Minimum floor for chirp detection correlation.
/// The adaptive threshold (see `AdaptiveThreshold`) tracks the noise floor
/// and sets the effective threshold above it. This constant is the absolute
/// minimum — the adaptive threshold will never go below this value.
pub const DETECTION_THRESHOLD: f32 = 0.08;

/// Higher threshold used during the initial warmup period before the adaptive
/// threshold has enough noise-floor data to calibrate. This prevents false
/// positives from ambient noise (which typically produces correlations of
/// ~0.08–0.10 in the chirp bands) triggering premature handshake confirmation.
pub const STARTUP_THRESHOLD: f32 = 0.20;

/// Number of noise-floor samples required before switching from
/// `STARTUP_THRESHOLD` to the computed adaptive threshold.
const MIN_WARMUP_SAMPLES: usize = 20;

/// Generate a linear chirp (frequency sweep) signal.
pub fn generate_chirp(
    start_freq: f32,
    end_freq: f32,
    duration_secs: f32,
    sample_rate: u32,
    amplitude: f32,
) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Phase for a linear chirp: 2π * (f0*t + (f1-f0)*t²/(2*T))
        let phase = TAU * (start_freq * t + (end_freq - start_freq) * t * t / (2.0 * duration_secs));
        let env = smooth_envelope(i, num_samples);
        samples.push(phase.sin() * amplitude * env);
    }

    samples
}

/// Generate the standard discovery chirp at the given sample rate.
pub fn discovery_chirp(sample_rate: u32) -> Vec<f32> {
    generate_chirp(
        CHIRP_START_FREQ,
        CHIRP_END_FREQ,
        CHIRP_DURATION_SECS,
        sample_rate,
        CHIRP_AMPLITUDE,
    )
}

/// Compute the TX amplitude for the given announce count.
/// Linearly ramps from CHIRP_AMPLITUDE_MIN to CHIRP_AMPLITUDE_MAX
/// over CHIRP_RAMP_STEPS announces, then holds at MAX.
pub fn tx_amplitude(announce_count: u64) -> f32 {
    if announce_count >= CHIRP_RAMP_STEPS {
        return CHIRP_AMPLITUDE_MAX;
    }
    let t = announce_count as f32 / CHIRP_RAMP_STEPS as f32;
    CHIRP_AMPLITUDE_MIN + t * (CHIRP_AMPLITUDE_MAX - CHIRP_AMPLITUDE_MIN)
}

/// Generate a CALL chirp (800-3200 Hz) at a specific amplitude.
pub fn discovery_call_chirp(sample_rate: u32, amplitude: f32) -> Vec<f32> {
    generate_chirp(
        CHIRP_START_FREQ,
        CHIRP_END_FREQ,
        CHIRP_DURATION_SECS,
        sample_rate,
        amplitude,
    )
}

/// Generate a RESPONSE chirp (4000-6400 Hz) at a specific amplitude for TX.
pub fn discovery_response_chirp(sample_rate: u32, amplitude: f32) -> Vec<f32> {
    generate_chirp(
        RESPONSE_START_FREQ,
        RESPONSE_END_FREQ,
        CHIRP_DURATION_SECS,
        sample_rate,
        amplitude,
    )
}

/// Generate a RESPONSE chirp template (4000-6400 Hz) for RX detection.
/// Uses the standard CHIRP_AMPLITUDE for correlation (amplitude is normalized out).
pub fn response_chirp_template(sample_rate: u32) -> Vec<f32> {
    generate_chirp(
        RESPONSE_START_FREQ,
        RESPONSE_END_FREQ,
        CHIRP_DURATION_SECS,
        sample_rate,
        CHIRP_AMPLITUDE,
    )
}

/// Detect a chirp in audio samples using normalized cross-correlation.
/// Returns `(peak_correlation, offset)` where offset is the sample position
/// of the best match. This allows the caller to trim only the consumed
/// portion of the buffer instead of discarding everything.
///
/// **Important:** This finds the FIRST (leftmost) match above
/// `DETECTION_THRESHOLD`, not the global maximum. This is critical for
/// double-chirp detection: when two chirps are in the buffer, we must
/// find chirp 1 first so that chirp 2 is preserved for the next detection
/// cycle.
pub fn detect_chirp(samples: &[f32], template: &[f32]) -> (f32, usize) {
    detect_chirp_adaptive(samples, template, DETECTION_THRESHOLD)
}

/// Like `detect_chirp` but with a caller-supplied threshold
/// (e.g., from `AdaptiveThreshold`).
pub fn detect_chirp_adaptive(samples: &[f32], template: &[f32], threshold: f32) -> (f32, usize) {
    if samples.len() < template.len() || template.is_empty() {
        return (0.0, 0);
    }

    let template_energy: f32 = template.iter().map(|s| s * s).sum();
    if template_energy == 0.0 {
        return (0.0, 0);
    }

    let coarse_step = (template.len() / 4).max(1);
    let max_offset = samples.len() - template.len();

    // Scan left-to-right to find the FIRST coarse region above threshold.
    // This ensures we find chirp 1 before chirp 2 in a double-chirp.
    let mut first_candidate_offset: Option<usize> = None;
    let mut first_candidate_corr: f32 = 0.0;

    // Also track the global best for the return value when nothing
    // exceeds the threshold (so callers can still see peak_corr).
    let mut global_best_corr: f32 = 0.0;
    let mut global_best_offset: usize = 0;

    for offset in (0..=max_offset).step_by(coarse_step) {
        let corr = normalized_correlation(samples, template, offset, template_energy);
        if corr > global_best_corr {
            global_best_corr = corr;
            global_best_offset = offset;
        }
        if corr > threshold && first_candidate_offset.is_none() {
            first_candidate_offset = Some(offset);
            first_candidate_corr = corr;
            break; // Stop at the first coarse region above threshold
        }
    }

    // If no coarse region exceeded the threshold, refine the global best
    if first_candidate_offset.is_none() {
        // Fine scan around global best to get accurate peak_corr
        let fine_start = global_best_offset.saturating_sub(coarse_step);
        let fine_end = (global_best_offset + coarse_step).min(max_offset);
        for offset in fine_start..=fine_end {
            let corr = normalized_correlation(samples, template, offset, template_energy);
            if corr > global_best_corr {
                global_best_corr = corr;
                global_best_offset = offset;
            }
        }
        return (global_best_corr, global_best_offset);
    }

    // Fine scan around the first candidate to get the exact offset
    let candidate = first_candidate_offset.unwrap();
    let mut best_corr = first_candidate_corr;
    let mut best_offset = candidate;

    let fine_start = candidate.saturating_sub(coarse_step);
    let fine_end = (candidate + coarse_step).min(max_offset);
    for offset in fine_start..=fine_end {
        let corr = normalized_correlation(samples, template, offset, template_energy);
        if corr > best_corr {
            best_corr = corr;
            best_offset = offset;
        }
    }

    (best_corr, best_offset)
}

/// Compute normalized cross-correlation at a specific offset.
fn normalized_correlation(
    samples: &[f32],
    template: &[f32],
    offset: usize,
    template_energy: f32,
) -> f32 {
    let window = &samples[offset..offset + template.len()];
    let cross: f32 = window.iter().zip(template.iter()).map(|(a, b)| a * b).sum();
    let window_energy: f32 = window.iter().map(|s| s * s).sum();

    if window_energy > 0.0 {
        cross / (window_energy.sqrt() * template_energy.sqrt())
    } else {
        0.0
    }
}

/// Smooth envelope: 5% fade-in and 5% fade-out to avoid clicks.
fn smooth_envelope(i: usize, total: usize) -> f32 {
    let fade = (total as f32 * 0.05) as usize;
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

// ---------------------------------------------------------------------------
// Bandpass pre-filter for chirp detection
// ---------------------------------------------------------------------------

/// Second-order IIR (biquad) filter coefficients, pre-normalized by a0.
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl Biquad {
    /// Design a 2nd-order Butterworth low-pass filter.
    fn lowpass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let omega = TAU * cutoff_hz / sample_rate;
        let cos_w = omega.cos();
        let sin_w = omega.sin();
        let alpha = sin_w / 2.0_f32.sqrt(); // Q = 1/sqrt(2) for Butterworth
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 - cos_w) / 2.0) / a0,
            b1: (1.0 - cos_w) / a0,
            b2: ((1.0 - cos_w) / 2.0) / a0,
            a1: (-2.0 * cos_w) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    /// Design a 2nd-order Butterworth high-pass filter.
    fn highpass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let omega = TAU * cutoff_hz / sample_rate;
        let cos_w = omega.cos();
        let sin_w = omega.sin();
        let alpha = sin_w / 2.0_f32.sqrt();
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 + cos_w) / 2.0) / a0,
            b1: (-(1.0 + cos_w)) / a0,
            b2: ((1.0 + cos_w) / 2.0) / a0,
            a1: (-2.0 * cos_w) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    /// Apply the filter to a sample buffer.
    fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        let (mut x1, mut x2) = (0.0f32, 0.0f32);
        let (mut y1, mut y2) = (0.0f32, 0.0f32);
        for &x in input {
            let y = self.b0 * x + self.b1 * x1 + self.b2 * x2
                   - self.a1 * y1 - self.a2 * y2;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
            out.push(y);
        }
        out
    }
}

/// Apply a bandpass filter (cascaded high-pass + low-pass) to audio samples.
///
/// Removes energy outside `[low_hz, high_hz]`, improving chirp detection
/// by reducing out-of-band noise that dilutes the correlation denominator.
pub fn bandpass_filter(samples: &[f32], low_hz: f32, high_hz: f32, sample_rate: f32) -> Vec<f32> {
    let hp = Biquad::highpass(low_hz, sample_rate);
    let lp = Biquad::lowpass(high_hz, sample_rate);
    lp.apply(&hp.apply(samples))
}

// ---------------------------------------------------------------------------
// Adaptive detection threshold
// ---------------------------------------------------------------------------

/// Tracks the noise floor and computes an adaptive detection threshold.
///
/// Records recent correlation measurements (when no chirp was detected)
/// and sets the threshold to `mean + 3 * stddev`, ensuring it stays above
/// the absolute minimum `DETECTION_THRESHOLD`.
pub struct AdaptiveThreshold {
    buffer: Vec<f32>,
    pos: usize,
    count: usize,
}

impl AdaptiveThreshold {
    /// Create a new tracker with the given ring-buffer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            pos: 0,
            count: 0,
        }
    }

    /// Record a noise-floor correlation measurement.
    pub fn record(&mut self, corr: f32) {
        self.buffer[self.pos] = corr;
        self.pos = (self.pos + 1) % self.buffer.len();
        if self.count < self.buffer.len() {
            self.count += 1;
        }
    }

    /// Compute the adaptive threshold.
    ///
    /// During warmup (fewer than `MIN_WARMUP_SAMPLES` noise measurements),
    /// returns `STARTUP_THRESHOLD` (0.20) to avoid false positives from
    /// ambient noise before the noise floor is established.
    ///
    /// After warmup, returns `max(DETECTION_THRESHOLD, mean + 3 * stddev)`.
    pub fn threshold(&self) -> f32 {
        if self.count < MIN_WARMUP_SAMPLES {
            return STARTUP_THRESHOLD;
        }
        let n = if self.count >= self.buffer.len() {
            self.buffer.len()
        } else {
            self.count
        };
        let data = &self.buffer[..n];
        let mean = data.iter().sum::<f32>() / n as f32;
        let variance = data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / n as f32;
        let stddev = variance.sqrt();
        (mean + 3.0 * stddev).max(DETECTION_THRESHOLD)
    }
}

// ---------------------------------------------------------------------------
// Binary chirp messaging
// ---------------------------------------------------------------------------

/// Duration of a single data chirp (shorter than discovery chirp for throughput).
pub const DATA_CHIRP_DURATION: f32 = 0.08; // 80ms
/// Silence gap between consecutive data chirps.
pub const DATA_CHIRP_GAP: f32 = 0.04; // 40ms
/// Total duration of one bit slot: chirp + gap = 120ms (~8.3 bps).
pub const BIT_SLOT_DURATION: f32 = DATA_CHIRP_DURATION + DATA_CHIRP_GAP;
/// Amplitude for data chirps.
pub const DATA_CHIRP_AMPLITUDE: f32 = 0.5;
/// Detection threshold for data chirps.
pub const DATA_DETECTION_THRESHOLD: f32 = 0.15;

/// Preamble pattern: alternating 1/0 for clock recovery (8 bits).
const PREAMBLE: [u8; 8] = [1, 0, 1, 0, 1, 0, 1, 0];
/// Sync word: four 1-bits in a row signals "data starts next".
const SYNC: [u8; 4] = [1, 1, 1, 1];

/// Generate an up-chirp (800→3200 Hz) representing bit=1.
pub fn data_up_chirp(sample_rate: u32) -> Vec<f32> {
    generate_chirp(
        CHIRP_START_FREQ,
        CHIRP_END_FREQ,
        DATA_CHIRP_DURATION,
        sample_rate,
        DATA_CHIRP_AMPLITUDE,
    )
}

/// Generate a down-chirp (3200→800 Hz) representing bit=0.
pub fn data_down_chirp(sample_rate: u32) -> Vec<f32> {
    generate_chirp(
        CHIRP_END_FREQ,
        CHIRP_START_FREQ,
        DATA_CHIRP_DURATION,
        sample_rate,
        DATA_CHIRP_AMPLITUDE,
    )
}

/// Encode a port number and capabilities byte into a binary chirp audio message.
///
/// Message format:
///   [preamble: 10101010] [sync: 1111] [port: 16 bits MSB] [caps: 8 bits] [checksum: 8 bits]
///   Total: 44 bit slots = ~5.3 seconds of audio
pub fn encode_chirp_message(port: u16, capabilities: u8, sample_rate: u32) -> Vec<f32> {
    let up = data_up_chirp(sample_rate);
    let down = data_down_chirp(sample_rate);
    let gap_samples = (DATA_CHIRP_GAP * sample_rate as f32) as usize;
    let gap = vec![0.0f32; gap_samples];

    // Build the bit sequence: preamble + sync + port (16) + caps (8) + checksum (8)
    let port_bytes = port.to_be_bytes();
    let checksum = port_bytes[0] ^ port_bytes[1] ^ capabilities;

    let mut bits: Vec<u8> = Vec::with_capacity(44);
    bits.extend_from_slice(&PREAMBLE);
    bits.extend_from_slice(&SYNC);
    // Port: 16 bits, MSB first
    for byte in &port_bytes {
        for bit_pos in (0..8).rev() {
            bits.push((byte >> bit_pos) & 1);
        }
    }
    // Capabilities: 8 bits
    for bit_pos in (0..8).rev() {
        bits.push((capabilities >> bit_pos) & 1);
    }
    // Checksum: 8 bits
    for bit_pos in (0..8).rev() {
        bits.push((checksum >> bit_pos) & 1);
    }

    // Encode each bit as up-chirp (1) or down-chirp (0) + gap
    let slot_samples = up.len() + gap_samples;
    let mut samples = Vec::with_capacity(bits.len() * slot_samples);
    for &bit in &bits {
        if bit == 1 {
            samples.extend_from_slice(&up);
        } else {
            samples.extend_from_slice(&down);
        }
        samples.extend_from_slice(&gap);
    }

    samples
}

/// Decode a binary chirp message from audio samples.
///
/// Correlates each bit-slot against up-chirp and down-chirp templates,
/// looks for the preamble+sync pattern, then extracts port and capabilities.
pub fn decode_chirp_message(samples: &[f32], sample_rate: u32) -> Option<(u16, u8)> {
    let up_template = data_up_chirp(sample_rate);
    let down_template = data_down_chirp(sample_rate);
    let chirp_len = up_template.len();
    let gap_samples = (DATA_CHIRP_GAP * sample_rate as f32) as usize;
    let slot_len = chirp_len + gap_samples;

    // We need at least 44 bit slots
    let needed = 44 * slot_len;
    if samples.len() < needed / 2 {
        return None;
    }

    // Pre-compute template energies
    let up_energy: f32 = up_template.iter().map(|s| s * s).sum();
    let down_energy: f32 = down_template.iter().map(|s| s * s).sum();

    if up_energy == 0.0 || down_energy == 0.0 {
        return None;
    }

    // Try multiple start offsets to find the best alignment
    // Search in steps of chirp_len/4 for coarse alignment
    let search_step = (chirp_len / 4).max(1);
    let max_search = samples.len().saturating_sub(needed).min(slot_len * 12);

    for start_offset in (0..=max_search).step_by(search_step) {
        if let Some(result) = try_decode_at_offset(
            samples,
            start_offset,
            &up_template,
            &down_template,
            up_energy,
            down_energy,
            slot_len,
            chirp_len,
        ) {
            return Some(result);
        }
    }

    None
}

/// Try to decode a chirp message starting at a specific sample offset.
fn try_decode_at_offset(
    samples: &[f32],
    start: usize,
    up_template: &[f32],
    down_template: &[f32],
    up_energy: f32,
    down_energy: f32,
    slot_len: usize,
    chirp_len: usize,
) -> Option<(u16, u8)> {
    // Decode enough bits for preamble + sync + data
    let total_bits = 44;
    let needed = start + total_bits * slot_len;
    if samples.len() < needed {
        return None;
    }

    let mut bits = Vec::with_capacity(total_bits);
    for i in 0..total_bits {
        let offset = start + i * slot_len;
        if offset + chirp_len > samples.len() {
            return None;
        }
        let window = &samples[offset..offset + chirp_len];

        let up_corr = {
            let cross: f32 = window.iter().zip(up_template.iter()).map(|(a, b)| a * b).sum();
            let win_energy: f32 = window.iter().map(|s| s * s).sum();
            if win_energy > 0.0 {
                cross / (win_energy.sqrt() * up_energy.sqrt())
            } else {
                0.0
            }
        };

        let down_corr = {
            let cross: f32 = window.iter().zip(down_template.iter()).map(|(a, b)| a * b).sum();
            let win_energy: f32 = window.iter().map(|s| s * s).sum();
            if win_energy > 0.0 {
                cross / (win_energy.sqrt() * down_energy.sqrt())
            } else {
                0.0
            }
        };

        // Both correlations must be above a minimum threshold
        let max_corr = up_corr.max(down_corr);
        if max_corr < DATA_DETECTION_THRESHOLD {
            return None; // Not a chirp signal
        }

        bits.push(if up_corr > down_corr { 1u8 } else { 0u8 });
    }

    // Verify preamble: 10101010
    if bits[0..8] != PREAMBLE {
        return None;
    }
    // Verify sync: 1111
    if bits[8..12] != SYNC {
        return None;
    }

    // Extract data bits (starting at bit 12)
    let data_bits = &bits[12..];
    // data_bits: 16 (port) + 8 (caps) + 8 (checksum) = 32 bits

    let port_hi = bits_to_byte(&data_bits[0..8]);
    let port_lo = bits_to_byte(&data_bits[8..16]);
    let caps = bits_to_byte(&data_bits[16..24]);
    let checksum = bits_to_byte(&data_bits[24..32]);

    let expected_checksum = port_hi ^ port_lo ^ caps;
    if checksum != expected_checksum {
        return None;
    }

    let port = u16::from_be_bytes([port_hi, port_lo]);
    Some((port, caps))
}

/// Convert 8 bits (as u8 array of 0/1) to a byte.
fn bits_to_byte(bits: &[u8]) -> u8 {
    let mut byte = 0u8;
    for (i, &bit) in bits.iter().enumerate() {
        byte |= bit << (7 - i);
    }
    byte
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::modulator::SAMPLE_RATE;

    #[test]
    fn test_chirp_generation() {
        let chirp = discovery_chirp(SAMPLE_RATE);
        let expected_len = (CHIRP_DURATION_SECS * SAMPLE_RATE as f32) as usize;
        assert_eq!(chirp.len(), expected_len);
        // All samples within amplitude bounds
        for &s in &chirp {
            assert!(s.abs() <= CHIRP_AMPLITUDE + 0.01, "Sample out of bounds: {}", s);
        }
    }

    #[test]
    fn test_chirp_self_detection() {
        let chirp = discovery_chirp(SAMPLE_RATE);
        let (corr, _offset) = detect_chirp(&chirp, &chirp);
        assert!(
            corr > 0.9,
            "Self-correlation should be near 1.0, got {}",
            corr
        );
    }

    #[test]
    fn test_chirp_detection_in_silence() {
        let chirp = discovery_chirp(SAMPLE_RATE);
        let silence = vec![0.0f32; chirp.len() * 3];
        let (corr, _) = detect_chirp(&silence, &chirp);
        assert!(
            corr < 0.01,
            "Silence should not trigger detection, got {}",
            corr
        );
    }

    #[test]
    fn test_chirp_detection_with_padding() {
        let chirp = discovery_chirp(SAMPLE_RATE);
        // Embed chirp in silence with random offset
        let mut padded = vec![0.0f32; chirp.len()]; // silence before
        padded.extend_from_slice(&chirp);
        padded.extend(vec![0.0f32; chirp.len()]); // silence after

        let (corr, offset) = detect_chirp(&padded, &chirp);
        assert!(
            corr > 0.8,
            "Should detect chirp in padded signal, got {}",
            corr
        );
        // Offset should be near chirp.len() (where we placed it)
        let expected_offset = chirp.len();
        assert!(
            (offset as isize - expected_offset as isize).unsigned_abs() < chirp.len() / 4,
            "Offset {} should be near {}", offset, expected_offset
        );
    }

    #[test]
    fn test_chirp_detection_with_noise() {
        let chirp = discovery_chirp(SAMPLE_RATE);
        // Add noise to the chirp (simple deterministic "noise")
        let noisy: Vec<f32> = chirp
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let noise = ((i as f32 * 7.3).sin() * 0.1) + ((i as f32 * 13.7).cos() * 0.05);
                s + noise
            })
            .collect();

        let (corr, _) = detect_chirp(&noisy, &chirp);
        assert!(
            corr > DETECTION_THRESHOLD,
            "Should detect chirp with noise, got {}",
            corr
        );
    }

    #[test]
    fn test_wrong_chirp_low_correlation() {
        let template = discovery_chirp(SAMPLE_RATE);
        // A different chirp (reversed sweep)
        let wrong = generate_chirp(
            CHIRP_END_FREQ,
            CHIRP_START_FREQ,
            CHIRP_DURATION_SECS,
            SAMPLE_RATE,
            CHIRP_AMPLITUDE,
        );
        let (corr, _) = detect_chirp(&wrong, &template);
        // Reversed chirp should have lower correlation than threshold
        assert!(
            corr < 0.7,
            "Reversed chirp should have low correlation, got {}",
            corr
        );
    }

    // --- Binary chirp messaging tests ---

    #[test]
    fn test_chirp_message_roundtrip() {
        let port = 7312u16;
        let caps = 0x0Bu8;
        let samples = encode_chirp_message(port, caps, SAMPLE_RATE);
        assert!(!samples.is_empty());

        let result = decode_chirp_message(&samples, SAMPLE_RATE);
        assert_eq!(result, Some((port, caps)), "Roundtrip decode should match");
    }

    #[test]
    fn test_chirp_message_various_ports() {
        for &(port, caps) in &[(80, 0x01), (9001, 0x0F), (65535, 0xFF), (0, 0x00)] {
            let samples = encode_chirp_message(port, caps, SAMPLE_RATE);
            let result = decode_chirp_message(&samples, SAMPLE_RATE);
            assert_eq!(
                result,
                Some((port, caps)),
                "Failed for port={} caps={}",
                port,
                caps
            );
        }
    }

    #[test]
    fn test_chirp_message_with_leading_silence() {
        let port = 7312u16;
        let caps = 0x0Bu8;
        let msg = encode_chirp_message(port, caps, SAMPLE_RATE);

        // Pad with silence before the message
        let mut padded = vec![0.0f32; SAMPLE_RATE as usize]; // 1 second of silence
        padded.extend_from_slice(&msg);

        let result = decode_chirp_message(&padded, SAMPLE_RATE);
        assert_eq!(result, Some((port, caps)), "Should decode with leading silence");
    }

    #[test]
    fn test_chirp_message_with_noise() {
        let port = 9001u16;
        let caps = 0x07u8;
        let samples = encode_chirp_message(port, caps, SAMPLE_RATE);

        // Add mild noise
        let noisy: Vec<f32> = samples
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let noise = (i as f32 * 7.3).sin() * 0.05;
                s + noise
            })
            .collect();

        let result = decode_chirp_message(&noisy, SAMPLE_RATE);
        assert_eq!(result, Some((port, caps)), "Should decode with mild noise");
    }

    #[test]
    fn test_data_up_down_chirps_distinguishable() {
        let up = data_up_chirp(SAMPLE_RATE);
        let down = data_down_chirp(SAMPLE_RATE);

        // Up-chirp should correlate well with up template, poorly with down
        let (up_up, _) = detect_chirp(&up, &up);
        let (up_down, _) = detect_chirp(&up, &down);
        assert!(
            up_up > up_down + 0.2,
            "Up-chirp should correlate better with up template: up_up={} up_down={}",
            up_up,
            up_down
        );

        // Down-chirp should correlate well with down template, poorly with up
        let (down_down, _) = detect_chirp(&down, &down);
        let (down_up, _) = detect_chirp(&down, &up);
        assert!(
            down_down > down_up + 0.2,
            "Down-chirp should correlate better with down template: down_down={} down_up={}",
            down_down,
            down_up
        );
    }

    // --- Progressive volume + response chirp tests ---

    #[test]
    fn test_tx_amplitude_ramp() {
        // At count 0, amplitude should be MIN
        let a0 = tx_amplitude(0);
        assert!(
            (a0 - CHIRP_AMPLITUDE_MIN).abs() < 0.001,
            "At count 0, expected {}, got {}",
            CHIRP_AMPLITUDE_MIN,
            a0
        );

        // At count >= RAMP_STEPS, amplitude should be MAX
        let a_max = tx_amplitude(CHIRP_RAMP_STEPS);
        assert!(
            (a_max - CHIRP_AMPLITUDE_MAX).abs() < 0.001,
            "At count {}, expected {}, got {}",
            CHIRP_RAMP_STEPS,
            CHIRP_AMPLITUDE_MAX,
            a_max
        );
        let a_beyond = tx_amplitude(CHIRP_RAMP_STEPS + 100);
        assert!(
            (a_beyond - CHIRP_AMPLITUDE_MAX).abs() < 0.001,
            "Beyond ramp, expected {}, got {}",
            CHIRP_AMPLITUDE_MAX,
            a_beyond
        );

        // Monotonically increasing
        let mut prev = tx_amplitude(0);
        for i in 1..=CHIRP_RAMP_STEPS {
            let curr = tx_amplitude(i);
            assert!(
                curr >= prev,
                "Amplitude should be monotonically increasing: step {} ({}) < step {} ({})",
                i,
                curr,
                i - 1,
                prev
            );
            prev = curr;
        }
    }

    #[test]
    fn test_response_chirp_generation() {
        let chirp = discovery_response_chirp(SAMPLE_RATE, CHIRP_AMPLITUDE);
        let expected_len = (CHIRP_DURATION_SECS * SAMPLE_RATE as f32) as usize;
        assert_eq!(chirp.len(), expected_len);
        for &s in &chirp {
            assert!(
                s.abs() <= CHIRP_AMPLITUDE + 0.01,
                "Response chirp sample out of bounds: {}",
                s
            );
        }
    }

    #[test]
    fn test_response_chirp_self_detection() {
        let template = response_chirp_template(SAMPLE_RATE);
        let (corr, _) = detect_chirp(&template, &template);
        assert!(
            corr > 0.9,
            "Response chirp self-correlation should be near 1.0, got {}",
            corr
        );
    }

    #[test]
    fn test_call_response_cross_isolation() {
        // CALL and RESPONSE chirps should have low cross-correlation
        let call_template = discovery_chirp(SAMPLE_RATE);
        let response_template = response_chirp_template(SAMPLE_RATE);

        // Check CALL signal against RESPONSE template
        let (cross_cr, _) = detect_chirp(&call_template, &response_template);
        assert!(
            cross_cr < DETECTION_THRESHOLD,
            "CALL vs RESPONSE template should be below threshold ({}), got {}",
            DETECTION_THRESHOLD,
            cross_cr
        );

        // Check RESPONSE signal against CALL template
        let (cross_rc, _) = detect_chirp(&response_template, &call_template);
        assert!(
            cross_rc < DETECTION_THRESHOLD,
            "RESPONSE vs CALL template should be below threshold ({}), got {}",
            DETECTION_THRESHOLD,
            cross_rc
        );
    }

    #[test]
    fn test_amplitude_scaling_does_not_affect_detection() {
        let template = discovery_chirp(SAMPLE_RATE);

        // Generate chirps at various amplitudes and check correlation
        for &amp in &[CHIRP_AMPLITUDE_MIN, 0.3, CHIRP_AMPLITUDE_MAX] {
            let chirp = discovery_call_chirp(SAMPLE_RATE, amp);
            let (corr, _) = detect_chirp(&chirp, &template);
            assert!(
                corr > 0.9,
                "Chirp at amplitude {} should still correlate highly (got {})",
                amp,
                corr
            );
        }

        // Same for response chirps
        let resp_template = response_chirp_template(SAMPLE_RATE);
        for &amp in &[CHIRP_AMPLITUDE_MIN, 0.3, CHIRP_AMPLITUDE_MAX] {
            let chirp = discovery_response_chirp(SAMPLE_RATE, amp);
            let (corr, _) = detect_chirp(&chirp, &resp_template);
            assert!(
                corr > 0.9,
                "Response chirp at amplitude {} should still correlate highly (got {})",
                amp,
                corr
            );
        }
    }
}
