/// Audio device enumeration and stream setup using cpal.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender};

use crate::audio::demodulator;
use crate::audio::modulator;
use crate::config::Config;
use crate::protocol::frame;

/// List all available audio input and output devices.
pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();

    println!("Audio host: {:?}", host.id());

    println!("\n--- Output Devices ---");
    if let Ok(devices) = host.output_devices() {
        for (i, device) in devices.enumerate() {
            let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into());
            println!("  [{}] {}", i, name);
            if let Ok(config) = device.default_output_config() {
                println!("      Default config: {:?}", config);
            }
        }
    }

    println!("\n--- Input Devices ---");
    if let Ok(devices) = host.input_devices() {
        for (i, device) in devices.enumerate() {
            let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into());
            println!("  [{}] {}", i, name);
            if let Ok(config) = device.default_input_config() {
                println!("      Default config: {:?}", config);
            }
        }
    }

    Ok(())
}

/// Play a test FSK tone encoding the given message.
pub fn play_test_tone(_config: &Config, message: &str) -> Result<()> {
    tracing::info!("Encoding message as FSK: {:?}", message);

    // Build a frame around the message bytes
    let frame_bytes = frame::encode_frame(message.as_bytes());
    let samples = modulator::modulate_bytes(&frame_bytes);

    tracing::info!(
        "Generated {} samples ({:.2}s at {}Hz)",
        samples.len(),
        samples.len() as f32 / modulator::SAMPLE_RATE as f32,
        modulator::SAMPLE_RATE
    );

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("No output audio device available")?;
    tracing::info!("Using output device: {}", device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into()));

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: modulator::SAMPLE_RATE,
        buffer_size: cpal::BufferSize::Default,
    };

    let samples = std::sync::Arc::new(samples);
    let sample_idx = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let samples_clone = samples.clone();
    let idx_clone = sample_idx.clone();
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_clone = done.clone();

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for sample in data.iter_mut() {
                let idx = idx_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if idx < samples_clone.len() {
                    *sample = samples_clone[idx];
                } else {
                    *sample = 0.0;
                    done_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        },
        |err| {
            tracing::error!("Audio output error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    tracing::info!("Playing FSK tone...");

    // Wait until all samples are played
    while !done.load(std::sync::atomic::Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    // Small tail to let audio buffer flush
    std::thread::sleep(std::time::Duration::from_millis(200));

    tracing::info!("Done playing.");
    Ok(())
}

/// Listen on the default microphone and attempt to decode FSK frames.
pub fn decode_from_mic(_config: &Config, duration_secs: u64) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input audio device available")?;
    tracing::info!("Using input device: {}", device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into()));

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: modulator::SAMPLE_RATE,
        buffer_size: cpal::BufferSize::Default,
    };

    let (tx, rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(1024);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let _ = tx.try_send(data.to_vec());
        },
        |err| {
            tracing::error!("Audio input error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    tracing::info!("Listening for FSK frames for {}s...", duration_secs);

    let start = std::time::Instant::now();
    let mut all_samples = Vec::new();
    let timeout = std::time::Duration::from_secs(duration_secs);

    while start.elapsed() < timeout {
        if let Ok(chunk) = rx.recv_timeout(std::time::Duration::from_millis(100)) {
            all_samples.extend_from_slice(&chunk);
        }
    }

    tracing::info!("Collected {} samples, decoding...", all_samples.len());

    let bits = demodulator::demodulate_samples(&all_samples);
    let bytes = modulator::bits_to_bytes(&bits);

    match frame::decode_frame(&bytes) {
        Some(payload) => {
            tracing::info!("Decoded frame payload ({} bytes)", payload.len());
            if let Ok(msg) = std::str::from_utf8(&payload) {
                println!("Decoded message: {}", msg);
            } else {
                println!("Decoded payload (hex): {:02X?}", payload);
            }
        }
        None => {
            tracing::warn!("No valid frame found in received audio");
        }
    }

    Ok(())
}

/// Play a tone through the speaker and simultaneously record from the mic,
/// then try to decode it. Tests whether the acoustic path works.
pub fn test_roundtrip() -> Result<()> {
    use crate::audio::modulator::BANDS;

    let test_payload = b"ROUNDTRIP_OK";
    let frame_bytes = frame::encode_frame(test_payload);

    println!("=== Audio Roundtrip Test ===");
    println!("Will play a tone on each of the 9 bands and try to decode from mic.\n");

    let host = cpal::default_host();
    let output_device = host.default_output_device().context("No output device")?;
    let input_device = host.default_input_device().context("No input device")?;

    let out_name = output_device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
    let in_name = input_device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
    println!("Output: {}", out_name);
    println!("Input:  {}", in_name);

    let stream_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: modulator::SAMPLE_RATE,
        buffer_size: cpal::BufferSize::Default,
    };

    // Set up recording
    let (rec_tx, rec_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(4096);
    let input_stream = input_device.build_input_stream(
        &stream_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let _ = rec_tx.try_send(data.to_vec());
        },
        |err| eprintln!("Input error: {}", err),
        None,
    )?;
    input_stream.play()?;

    // Test each band
    for (band_idx, band) in BANDS.iter().enumerate() {
        let samples = modulator::modulate_bytes_freq(&frame_bytes, band);
        let duration_secs = samples.len() as f32 / modulator::SAMPLE_RATE as f32;

        print!("Band {} ({}/{}Hz): ", band_idx, band.mark, band.space);

        // Drain any old mic data
        while rec_rx.try_recv().is_ok() {}

        // Small pre-recording silence to establish baseline
        std::thread::sleep(std::time::Duration::from_millis(200));
        while rec_rx.try_recv().is_ok() {} // drain again

        // Play the tone
        let samples_arc = std::sync::Arc::new(samples);
        let idx = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let s = samples_arc.clone();
        let i = idx.clone();
        let d = done.clone();

        let output_stream = output_device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for sample in data.iter_mut() {
                    let cur = i.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if cur < s.len() {
                        *sample = s[cur];
                    } else {
                        *sample = 0.0;
                        d.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            },
            |err| eprintln!("Output error: {}", err),
            None,
        )?;
        output_stream.play()?;

        // Wait for playback to finish
        while !done.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Extra time for echo/tail
        std::thread::sleep(std::time::Duration::from_millis(300));
        drop(output_stream);

        // Collect recorded samples
        let mut recorded = Vec::new();
        while let Ok(chunk) = rec_rx.try_recv() {
            recorded.extend_from_slice(&chunk);
        }

        // Measure RMS
        let rms: f32 = if recorded.is_empty() {
            0.0
        } else {
            let sum_sq: f32 = recorded.iter().map(|s| s * s).sum();
            (sum_sq / recorded.len() as f32).sqrt()
        };

        // Try to decode on this band
        let bits = demodulator::demodulate_samples_freq(&recorded, band);
        let bytes = modulator::bits_to_bytes(&bits);
        let decoded = frame::decode_frame(&bytes);

        match decoded {
            Some(payload) if payload == test_payload => {
                println!("OK (rms={:.4}, {:.1}s, {} samples recorded)", rms, duration_secs, recorded.len());
            }
            Some(payload) => {
                println!("DECODED BUT WRONG (rms={:.4}, got {} bytes)", rms, payload.len());
            }
            None => {
                println!("FAILED (rms={:.4}, {} samples recorded)", rms, recorded.len());
            }
        }
    }

    println!("\nIf all bands show FAILED with rms near 0, the mic is not picking up the speaker.");
    println!("If rms > 0 but FAILED, the signal quality is too low for decoding.");
    println!("If some bands show OK, audio discovery should work on those bands.");

    Ok(())
}

/// Play a friendly two-tone chime to signal a successful handshake.
/// Uses a rising major third (C5 → E5) with a soft envelope.
pub fn play_handshake_chime() -> Result<()> {
    let sample_rate = modulator::SAMPLE_RATE;
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("No output audio device for chime")?;

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    // Two notes: C5 (523Hz) for 120ms, then E5 (659Hz) for 150ms
    let note1_freq = 523.25_f32;
    let note2_freq = 659.25_f32;
    let note1_samples = (sample_rate as f32 * 0.12) as usize;
    let note2_samples = (sample_rate as f32 * 0.15) as usize;
    let gap_samples = (sample_rate as f32 * 0.03) as usize; // 30ms gap
    let total = note1_samples + gap_samples + note2_samples;

    let tau = std::f32::consts::TAU;
    let mut samples = Vec::with_capacity(total);

    // Note 1 with fade-in/fade-out envelope
    for i in 0..note1_samples {
        let t = i as f32 / sample_rate as f32;
        let env = smooth_envelope(i, note1_samples);
        samples.push((tau * note1_freq * t).sin() * 0.35 * env);
    }
    // Silent gap
    for _ in 0..gap_samples {
        samples.push(0.0);
    }
    // Note 2 with fade-in/fade-out envelope
    for i in 0..note2_samples {
        let t = i as f32 / sample_rate as f32;
        let env = smooth_envelope(i, note2_samples);
        samples.push((tau * note2_freq * t).sin() * 0.35 * env);
    }

    let samples = std::sync::Arc::new(samples);
    let idx = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let s = samples.clone();
    let i = idx.clone();
    let d = done.clone();

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for sample in data.iter_mut() {
                let cur = i.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if cur < s.len() {
                    *sample = s[cur];
                } else {
                    *sample = 0.0;
                    d.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        },
        |err| tracing::error!("Chime output error: {}", err),
        None,
    )?;

    stream.play()?;
    while !done.load(std::sync::atomic::Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    std::thread::sleep(std::time::Duration::from_millis(50));

    Ok(())
}

/// Smooth envelope: quick fade-in (10%) and fade-out (20%) to avoid clicks.
fn smooth_envelope(i: usize, total: usize) -> f32 {
    let fade_in = (total as f32 * 0.1) as usize;
    let fade_out = (total as f32 * 0.2) as usize;
    if i < fade_in {
        i as f32 / fade_in as f32
    } else if i > total - fade_out {
        (total - i) as f32 / fade_out as f32
    } else {
        1.0
    }
}

/// AudioEngine manages sending and receiving FSK-modulated audio via cpal.
pub struct AudioEngine {
    tx_sender: Sender<Vec<f32>>,
    rx_receiver: Receiver<Vec<f32>>,
    _output_stream: cpal::Stream,
    _input_stream: cpal::Stream,
}

impl AudioEngine {
    /// Create a new AudioEngine using the default audio devices.
    ///
    /// Adapts to each device's native channel count: output duplicates mono
    /// samples across all channels, input mixes down to mono.
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();

        let output_device = host
            .default_output_device()
            .context("No output audio device")?;
        let input_device = host
            .default_input_device()
            .context("No input audio device")?;

        let out_name = output_device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into());
        let in_name = input_device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into());
        tracing::info!("Audio output device: {}", out_name);
        tracing::info!("Audio input device: {}", in_name);

        let out_default = output_device.default_output_config()
            .context("No default output config")?;
        let in_default = input_device.default_input_config()
            .context("No default input config")?;
        tracing::info!("Output default config: {:?}", out_default);
        tracing::info!("Input default config: {:?}", in_default);

        let out_channels = out_default.channels();
        let in_channels = in_default.channels();

        // Use each device's native channel count to avoid ALSA errors on
        // devices that only support stereo (common on USB audio on Pi).
        let out_config = cpal::StreamConfig {
            channels: out_channels,
            sample_rate: modulator::SAMPLE_RATE,
            buffer_size: cpal::BufferSize::Default,
        };
        let in_config = cpal::StreamConfig {
            channels: in_channels,
            sample_rate: modulator::SAMPLE_RATE,
            buffer_size: cpal::BufferSize::Default,
        };
        tracing::info!("Output stream: {}ch @ {}Hz", out_channels, modulator::SAMPLE_RATE);
        tracing::info!("Input stream: {}ch @ {}Hz", in_channels, modulator::SAMPLE_RATE);

        // TX: samples to play (mono from our modulator, duplicated to all output channels)
        let (tx_sender, tx_receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(64);
        let tx_buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
        let tx_buf_clone = tx_buffer.clone();

        let output_stream = output_device.build_output_stream(
            &out_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // Drain any queued sample chunks
                while let Ok(chunk) = tx_receiver.try_recv() {
                    tx_buf_clone.lock().unwrap().extend_from_slice(&chunk);
                }
                let mut buf = tx_buf_clone.lock().unwrap();
                for frame in data.chunks_mut(out_channels as usize) {
                    let sample = if buf.is_empty() {
                        0.0
                    } else {
                        buf.remove(0)
                    };
                    // Duplicate mono sample to all channels
                    for ch in frame.iter_mut() {
                        *ch = sample;
                    }
                }
            },
            |err| tracing::error!("Output stream error: {}", err),
            None,
        )?;

        // RX: received samples (mix interleaved multi-channel input down to mono)
        let (rx_sender, rx_receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(1024);

        let input_stream = input_device.build_input_stream(
            &in_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = data
                    .chunks(in_channels as usize)
                    .map(|frame| frame.iter().sum::<f32>() / in_channels as f32)
                    .collect();
                let _ = rx_sender.try_send(mono);
            },
            |err| tracing::error!("Input stream error: {}", err),
            None,
        )?;

        output_stream.play()?;
        input_stream.play()?;

        Ok(Self {
            tx_sender,
            rx_receiver,
            _output_stream: output_stream,
            _input_stream: input_stream,
        })
    }

    /// Queue samples for transmission.
    pub fn send_samples(&self, samples: Vec<f32>) -> Result<()> {
        self.tx_sender
            .send(samples)
            .map_err(|e| anyhow::anyhow!("Failed to queue TX samples: {}", e))
    }

    /// Receive a chunk of samples from the microphone (blocks until available).
    pub fn recv_samples(&self) -> Result<Vec<f32>> {
        self.rx_receiver
            .recv()
            .map_err(|e| anyhow::anyhow!("Failed to receive RX samples: {}", e))
    }

    /// Try to receive a chunk of samples without blocking.
    pub fn try_recv_samples(&self) -> Option<Vec<f32>> {
        self.rx_receiver.try_recv().ok()
    }
}

/// LoopbackAudioEngine connects TX output directly to RX input in-memory.
/// Used for testing without audio hardware.
pub struct LoopbackAudioEngine {
    tx_sender: Sender<Vec<f32>>,
    rx_receiver: Receiver<Vec<f32>>,
}

impl LoopbackAudioEngine {
    pub fn new() -> Self {
        let (tx, rx) = bounded(1024);
        Self {
            tx_sender: tx,
            rx_receiver: rx,
        }
    }

    pub fn send_samples(&self, samples: Vec<f32>) -> Result<()> {
        self.tx_sender
            .send(samples)
            .map_err(|e| anyhow::anyhow!("Loopback send failed: {}", e))
    }

    pub fn recv_samples(&self) -> Result<Vec<f32>> {
        self.rx_receiver
            .recv()
            .map_err(|e| anyhow::anyhow!("Loopback recv failed: {}", e))
    }

    pub fn try_recv_samples(&self) -> Option<Vec<f32>> {
        self.rx_receiver.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loopback_engine_roundtrip() {
        let engine = LoopbackAudioEngine::new();
        let original = vec![0xCA, 0xFE];
        let frame_bytes = frame::encode_frame(&original);
        let samples = modulator::modulate_bytes(&frame_bytes);

        engine.send_samples(samples.clone()).unwrap();
        let received = engine.recv_samples().unwrap();

        assert_eq!(samples, received);
    }
}
