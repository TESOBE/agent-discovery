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

    if let Some(dev) = host.default_output_device() {
        let name = dev.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
        println!("Default output device: {}", name);
    }
    if let Some(dev) = host.default_input_device() {
        let name = dev.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
        println!("Default input device: {}", name);
    }

    println!("\n--- Output Devices ---");
    if let Ok(devices) = host.output_devices() {
        for (i, device) in devices.enumerate() {
            let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into());
            println!("  [{}] {}", i, name);
            if let Ok(config) = device.default_output_config() {
                println!("      Default: {:?}", config);
            }
            if let Ok(configs) = device.supported_output_configs() {
                for cfg in configs {
                    println!("      Supported: ch={} rate={}-{} fmt={:?}",
                        cfg.channels(), cfg.min_sample_rate(), cfg.max_sample_rate(),
                        cfg.sample_format());
                }
            }
        }
    }

    println!("\n--- Input Devices ---");
    if let Ok(devices) = host.input_devices() {
        for (i, device) in devices.enumerate() {
            let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into());
            println!("  [{}] {}", i, name);
            if let Ok(config) = device.default_input_config() {
                println!("      Default: {:?}", config);
            }
            if let Ok(configs) = device.supported_input_configs() {
                for cfg in configs {
                    println!("      Supported: ch={} rate={}-{} fmt={:?}",
                        cfg.channels(), cfg.min_sample_rate(), cfg.max_sample_rate(),
                        cfg.sample_format());
                }
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

/// Generate handshake chime samples — a rising major third (C5 → E5).
pub fn generate_handshake_chime() -> Vec<f32> {
    let sample_rate = modulator::SAMPLE_RATE;

    let note1_freq = 523.25_f32;
    let note2_freq = 659.25_f32;
    let note1_samples = (sample_rate as f32 * 0.12) as usize;
    let note2_samples = (sample_rate as f32 * 0.15) as usize;
    let gap_samples = (sample_rate as f32 * 0.03) as usize;
    let total = note1_samples + gap_samples + note2_samples;

    let tau = std::f32::consts::TAU;
    let mut samples = Vec::with_capacity(total);

    for i in 0..note1_samples {
        let t = i as f32 / sample_rate as f32;
        let env = smooth_envelope(i, note1_samples);
        samples.push((tau * note1_freq * t).sin() * 0.35 * env);
    }
    for _ in 0..gap_samples {
        samples.push(0.0);
    }
    for i in 0..note2_samples {
        let t = i as f32 / sample_rate as f32;
        let env = smooth_envelope(i, note2_samples);
        samples.push((tau * note2_freq * t).sin() * 0.35 * env);
    }

    samples
}

/// Generate OBP-unreachable alert samples — a descending minor interval (E4 → C4).
pub fn generate_no_obp_tone() -> Vec<f32> {
    let sample_rate = modulator::SAMPLE_RATE;

    let note1_freq = 329.63_f32;
    let note2_freq = 261.63_f32;
    let note1_samples = (sample_rate as f32 * 0.15) as usize;
    let note2_samples = (sample_rate as f32 * 0.20) as usize;
    let gap_samples = (sample_rate as f32 * 0.04) as usize;
    let total = note1_samples + gap_samples + note2_samples;

    let tau = std::f32::consts::TAU;
    let mut samples = Vec::with_capacity(total);

    for i in 0..note1_samples {
        let t = i as f32 / sample_rate as f32;
        let env = smooth_envelope(i, note1_samples);
        samples.push((tau * note1_freq * t).sin() * 0.35 * env);
    }
    for _ in 0..gap_samples {
        samples.push(0.0);
    }
    for i in 0..note2_samples {
        let t = i as f32 / sample_rate as f32;
        let env = smooth_envelope(i, note2_samples);
        samples.push((tau * note2_freq * t).sin() * 0.35 * env);
    }

    samples
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

/// Collect audio device info and post it to OBP signal channel 'audio-diagnostics'.
pub async fn diagnose_audio_to_signal(config: &Config) -> Result<()> {
    let host = cpal::default_host();
    let mut info = String::new();

    info.push_str(&format!("host: {:?}\n", host.id()));

    if let Some(dev) = host.default_output_device() {
        let name = dev.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
        info.push_str(&format!("default_output: {}\n", name));
        if let Ok(cfg) = dev.default_output_config() {
            info.push_str(&format!("  config: ch={} rate={} fmt={:?}\n", cfg.channels(), cfg.sample_rate(), cfg.sample_format()));
        }
    } else {
        info.push_str("default_output: NONE\n");
    }

    if let Some(dev) = host.default_input_device() {
        let name = dev.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
        info.push_str(&format!("default_input: {}\n", name));
        if let Ok(cfg) = dev.default_input_config() {
            info.push_str(&format!("  config: ch={} rate={} fmt={:?}\n", cfg.channels(), cfg.sample_rate(), cfg.sample_format()));
        }
    } else {
        info.push_str("default_input: NONE\n");
    }

    info.push_str("\nall_outputs:\n");
    if let Ok(devices) = host.output_devices() {
        for (i, device) in devices.enumerate() {
            let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
            info.push_str(&format!("  [{}] {}\n", i, name));
            if let Ok(cfg) = device.default_output_config() {
                info.push_str(&format!("    default: ch={} rate={} fmt={:?}\n", cfg.channels(), cfg.sample_rate(), cfg.sample_format()));
            }
            if let Ok(configs) = device.supported_output_configs() {
                for cfg in configs {
                    info.push_str(&format!("    supported: ch={} rate={}-{} fmt={:?}\n",
                        cfg.channels(), cfg.min_sample_rate(), cfg.max_sample_rate(), cfg.sample_format()));
                }
            }
        }
    }

    info.push_str("\nall_inputs:\n");
    if let Ok(devices) = host.input_devices() {
        for (i, device) in devices.enumerate() {
            let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
            info.push_str(&format!("  [{}] {}\n", i, name));
            if let Ok(cfg) = device.default_input_config() {
                info.push_str(&format!("    default: ch={} rate={} fmt={:?}\n", cfg.channels(), cfg.sample_rate(), cfg.sample_format()));
            }
            if let Ok(configs) = device.supported_input_configs() {
                for cfg in configs {
                    info.push_str(&format!("    supported: ch={} rate={}-{} fmt={:?}\n",
                        cfg.channels(), cfg.min_sample_rate(), cfg.max_sample_rate(), cfg.sample_format()));
                }
            }
        }
    }

    println!("{}", info);

    // Post to OBP signal channel
    let mut obp = crate::obp::client::ObpClient::new(config);
    match obp.authenticate().await {
        Ok(()) => {
            let payload = serde_json::json!({
                "payload": {
                    "agent_name": config.agent_name,
                    "type": "audio-diagnostics",
                    "info": info,
                }
            });
            let path = format!("/obp/{}/signal/channels/audio-diagnostics/messages",
                crate::obp::client::API_VERSION);
            match obp.post(&path, &payload).await {
                Ok(val) => println!("Posted to signal channel: {}", val),
                Err(e) => println!("Failed to post to signal channel: {}", e),
            }
        }
        Err(e) => println!("OBP auth failed (skipping signal post): {:?}", e),
    }

    Ok(())
}

/// An ALSA device parsed from `arecord -l` or `aplay -l` output.
#[derive(Debug, Clone)]
struct AlsaDevice {
    card_num: u32,
    card_name: String,
    device_num: u32,
    description: String,
}

impl AlsaDevice {
    /// ALSA device string suitable for aplay/arecord -D flag.
    fn plughw(&self) -> String {
        format!("plughw:{},{}", self.card_name, self.device_num)
    }
}

/// Parse `arecord -l` or `aplay -l` output into a list of ALSA devices.
/// Lines look like: `card 2: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]`
fn parse_alsa_devices(output: &str) -> Vec<AlsaDevice> {
    let re_card = regex_lite::Regex::new(
        r"^card (\d+): (\S+) \[([^\]]*)\], device (\d+):"
    ).expect("valid regex");

    let mut devices = Vec::new();
    for line in output.lines() {
        if let Some(caps) = re_card.captures(line) {
            let card_num: u32 = caps[1].parse().unwrap_or(0);
            // Skip card 0 (typically onboard/dummy on Pi)
            if card_num == 0 {
                continue;
            }
            devices.push(AlsaDevice {
                card_num,
                card_name: caps[2].to_string(),
                device_num: caps[4].parse().unwrap_or(0),
                description: caps[3].to_string(),
            });
        }
    }
    devices
}

/// Compute RMS of raw S16_LE bytes interpreted as mono i16 samples.
fn rms_from_raw_bytes(bytes: &[u8]) -> f64 {
    if bytes.len() < 2 {
        return 0.0;
    }
    let samples: Vec<f64> = bytes
        .chunks_exact(2)
        .map(|pair| {
            let val = i16::from_le_bytes([pair[0], pair[1]]);
            val as f64 / i16::MAX as f64
        })
        .collect();
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Generate 1 second of 1 kHz sine wave as raw S16_LE bytes at 44100 Hz mono.
fn generate_sine_tone_bytes(freq_hz: f64, duration_secs: f64, sample_rate: u32) -> Vec<u8> {
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    let mut bytes = Vec::with_capacity(num_samples * 2);
    let tau = std::f64::consts::TAU;
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (tau * freq_hz * t).sin() * 0.8; // 80% amplitude
        let i16_val = (sample * i16::MAX as f64) as i16;
        bytes.extend_from_slice(&i16_val.to_le_bytes());
    }
    bytes
}

/// Auto-detect USB audio devices by enumerating ALSA devices, testing mics
/// for ambient noise, testing speaker-mic pairs with a loopback tone, and
/// writing `~/.asoundrc` with the working pair using card names.
pub fn detect_audio() -> Result<()> {
    use std::process::{Command, Stdio};

    println!("=== Audio Device Detection ===\n");

    // Step 1: Enumerate ALSA devices
    println!("Step 1: Enumerating ALSA devices...");

    let capture_output = Command::new("arecord")
        .args(["-l"])
        .output()
        .context("Failed to run arecord -l — is alsa-utils installed?")?;
    let capture_list = String::from_utf8_lossy(&capture_output.stdout);

    let playback_output = Command::new("aplay")
        .args(["-l"])
        .output()
        .context("Failed to run aplay -l — is alsa-utils installed?")?;
    let playback_list = String::from_utf8_lossy(&playback_output.stdout);

    let capture_devices = parse_alsa_devices(&capture_list);
    let playback_devices = parse_alsa_devices(&playback_list);

    println!("  Capture devices (skipping card 0):");
    for dev in &capture_devices {
        println!("    card {}: {} [{}] device {} -> {}",
            dev.card_num, dev.card_name, dev.description, dev.device_num, dev.plughw());
    }
    println!("  Playback devices (skipping card 0):");
    for dev in &playback_devices {
        println!("    card {}: {} [{}] device {} -> {}",
            dev.card_num, dev.card_name, dev.description, dev.device_num, dev.plughw());
    }

    if capture_devices.is_empty() {
        anyhow::bail!("No USB capture devices found (arecord -l showed nothing beyond card 0)");
    }
    if playback_devices.is_empty() {
        anyhow::bail!("No USB playback devices found (aplay -l showed nothing beyond card 0)");
    }

    // Step 2: Test each capture device for ambient noise
    println!("\nStep 2: Testing capture devices for ambient noise (1s each)...");

    struct MicResult {
        device: AlsaDevice,
        ambient_rms: f64,
        is_live: bool,
    }

    let mut mic_results: Vec<MicResult> = Vec::new();

    for dev in &capture_devices {
        let hw = dev.plughw();
        print!("  Testing {} ... ", hw);

        let result = Command::new("arecord")
            .args(["-D", &hw, "-d", "1", "-f", "S16_LE", "-r", "44100", "-c", "1", "-t", "raw", "-q"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();

        match result {
            Ok(output) if output.status.success() => {
                let rms = rms_from_raw_bytes(&output.stdout);
                let is_live = rms > 0.001;
                println!("RMS={:.6} {}", rms, if is_live { "LIVE" } else { "silent" });
                mic_results.push(MicResult {
                    device: dev.clone(),
                    ambient_rms: rms,
                    is_live,
                });
            }
            Ok(output) => {
                println!("FAILED (exit code: {:?})", output.status.code());
            }
            Err(e) => {
                println!("ERROR: {}", e);
            }
        }
    }

    let live_mics: Vec<&MicResult> = mic_results.iter().filter(|m| m.is_live).collect();
    if live_mics.is_empty() {
        // Fall back to all mics that at least responded, even if silent
        println!("\n  No mics picked up ambient noise — will try all responsive mics for loopback test.");
    }

    // Use live mics preferentially, but fall back to all responsive mics
    let test_mics: Vec<&MicResult> = if live_mics.is_empty() {
        mic_results.iter().collect()
    } else {
        live_mics
    };

    if test_mics.is_empty() {
        anyhow::bail!("No capture devices responded successfully");
    }

    // Step 3: Test each (speaker, mic) pair with a loopback tone
    println!("\nStep 3: Testing speaker-mic pairs with 1 kHz loopback tone...");

    let tone_bytes = generate_sine_tone_bytes(1000.0, 1.0, 44100);
    let mut found_pair: Option<(&AlsaDevice, &AlsaDevice)> = None;

    'outer: for mic in &test_mics {
        for spk in &playback_devices {
            let mic_hw = mic.device.plughw();
            let spk_hw = spk.plughw();
            print!("  Testing speaker={} mic={} ... ", spk_hw, mic_hw);

            // Start recording (2 seconds) in background
            let mut arecord = match Command::new("arecord")
                .args(["-D", &mic_hw, "-d", "2", "-f", "S16_LE", "-r", "44100", "-c", "1", "-t", "raw", "-q"])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(child) => child,
                Err(e) => {
                    println!("arecord spawn failed: {}", e);
                    continue;
                }
            };

            // Wait for arecord to initialize
            std::thread::sleep(std::time::Duration::from_millis(200));

            // Play the tone
            let mut aplay = match Command::new("aplay")
                .args(["-D", &spk_hw, "-f", "S16_LE", "-r", "44100", "-c", "1", "-t", "raw", "-q"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(child) => child,
                Err(e) => {
                    println!("aplay spawn failed: {}", e);
                    let _ = arecord.kill();
                    continue;
                }
            };

            // Pipe tone data to aplay
            if let Some(ref mut stdin) = aplay.stdin {
                use std::io::Write;
                let _ = stdin.write_all(&tone_bytes);
            }
            drop(aplay.stdin.take()); // Close stdin so aplay finishes
            let _ = aplay.wait();

            // Wait for arecord to finish
            let rec_output = match arecord.wait_with_output() {
                Ok(output) => output,
                Err(e) => {
                    println!("arecord wait failed: {}", e);
                    continue;
                }
            };

            let rec_rms = rms_from_raw_bytes(&rec_output.stdout);
            let threshold = if mic.ambient_rms > 0.0 {
                mic.ambient_rms * 5.0
            } else {
                0.005 // absolute threshold if ambient was zero
            };

            let is_loopback = rec_rms > threshold;
            println!("RMS={:.6} (ambient={:.6}, threshold={:.6}) {}",
                rec_rms, mic.ambient_rms, threshold,
                if is_loopback { "LOOPBACK OK" } else { "no loopback" });

            if is_loopback {
                found_pair = Some((&spk, &mic.device));
                break 'outer;
            }
        }
    }

    // Step 4: Write ~/.asoundrc
    let (spk, mic) = match found_pair {
        Some(pair) => pair,
        None => {
            // If no loopback detected, use the first playback + first responsive mic
            println!("\n  No loopback detected — falling back to first playback + first capture device.");
            (&playback_devices[0], &test_mics[0].device)
        }
    };

    println!("\nStep 4: Writing ~/.asoundrc...");
    println!("  Speaker: {} [{}]", spk.card_name, spk.description);
    println!("  Mic:     {} [{}]", mic.card_name, mic.description);

    let asoundrc_content = format!(
        r#"pcm.!default {{
    type asym
    playback.pcm "plughw:{spk_name},0"
    capture.pcm "plughw:{mic_name},0"
}}
ctl.!default {{
    type hw
    card {spk_name}
}}
"#,
        spk_name = spk.card_name,
        mic_name = mic.card_name,
    );

    let home = std::env::var("HOME").context("HOME environment variable not set")?;
    let asoundrc_path = format!("{}/.asoundrc", home);

    // Back up existing .asoundrc if present
    if std::path::Path::new(&asoundrc_path).exists() {
        let backup_path = format!("{}.bak", asoundrc_path);
        std::fs::copy(&asoundrc_path, &backup_path)
            .context("Failed to back up existing ~/.asoundrc")?;
        println!("  Backed up existing ~/.asoundrc to ~/.asoundrc.bak");
    }

    std::fs::write(&asoundrc_path, &asoundrc_content)
        .context("Failed to write ~/.asoundrc")?;

    println!("  Written to {}", asoundrc_path);
    println!("\n=== Detection complete ===");
    println!("Test with: arecord -d 2 -f cd test.wav && aplay test.wav");

    Ok(())
}

/// AudioEngine manages sending and receiving FSK-modulated audio.
///
/// Tries cpal first (works on desktop). If cpal fails (common on Pi with
/// USB audio), falls back to spawning `aplay`/`arecord` subprocesses which
/// use ALSA read/write mode and work reliably with USB devices.
///
/// TODO: Add a self-discovery / self-test mode. On startup, the engine should
/// play a short known tone through the speaker and simultaneously listen on
/// the mic. If the mic picks up the tone (RMS above a threshold), the current
/// audio config is working. If not, try the next config in a ranked list:
///   1. subprocess aplay/arecord at 44100Hz mono
///   2. subprocess at 48000Hz mono
///   3. subprocess at 48000Hz stereo
///   4. cpal with I16 format
///   5. cpal with F32 format
///   6. cpal with different device indices (enumerate all output/input pairs)
/// Once a working config is found, persist it (e.g. to a local file) so
/// subsequent startups skip the probing. Log the result to the OBP
/// audio-diagnostics signal channel for remote visibility.
pub struct AudioEngine {
    tx_sender: Sender<Vec<f32>>,
    rx_receiver: Receiver<Vec<f32>>,
    // Hold onto resources so they aren't dropped:
    _backend: AudioBackend,
}

enum AudioBackend {
    Cpal {
        _output_stream: cpal::Stream,
        _input_stream: cpal::Stream,
    },
    Subprocess {
        _output_child: std::process::Child,
        _input_child: std::process::Child,
        _output_thread: Option<std::thread::JoinHandle<()>>,
        _input_thread: Option<std::thread::JoinHandle<()>>,
    },
}

/// The sample rate used by the subprocess backend.
/// Matches modulator::SAMPLE_RATE so no resampling is needed.
const SUBPROCESS_RATE: u32 = 44100;

/// Maximum consecutive subprocess restart failures before giving up.
const MAX_SUBPROCESS_RESTARTS: u32 = 5;

/// Spawn an aplay subprocess that reads raw S16_LE mono from stdin.
fn spawn_aplay() -> Result<std::process::Child> {
    use std::process::{Command, Stdio};
    let rate_str = SUBPROCESS_RATE.to_string();
    Command::new("aplay")
        .args(["-D", "default", "-f", "S16_LE", "-r", &rate_str, "-c", "1", "-t", "raw", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn aplay — is alsa-utils installed?")
}

/// Spawn an arecord subprocess that writes raw S16_LE mono to stdout.
fn spawn_arecord() -> Result<std::process::Child> {
    use std::process::{Command, Stdio};
    let rate_str = SUBPROCESS_RATE.to_string();
    Command::new("arecord")
        .args(["-D", "default", "-f", "S16_LE", "-r", &rate_str, "-c", "1", "-t", "raw", "-q"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn arecord — is alsa-utils installed?")
}

/// Convert f32 samples to raw S16_LE bytes for aplay.
fn samples_to_s16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_val = (clamped * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&i16_val.to_le_bytes());
    }
    bytes
}

impl AudioEngine {
    /// Create a new AudioEngine.
    ///
    /// On Linux, tries aplay/arecord subprocesses first — these use ALSA
    /// read/write mode which works reliably with USB audio on Pi (cpal uses
    /// MMAP mode which causes POLLERR on USB devices).
    /// On other platforms, uses cpal directly.
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            match Self::try_subprocess() {
                Ok(engine) => {
                    tracing::info!("Audio engine: using aplay/arecord subprocess backend");
                    return Ok(engine);
                }
                Err(sub_err) => {
                    tracing::warn!("Subprocess backend failed: {}. Trying cpal.", sub_err);
                }
            }
        }

        match Self::try_cpal() {
            Ok(engine) => {
                tracing::info!("Audio engine: using cpal backend");
                Ok(engine)
            }
            Err(cpal_err) => {
                anyhow::bail!("No audio backend available: {}", cpal_err);
            }
        }
    }

    /// Try to create an engine using cpal.
    fn try_cpal() -> Result<Self> {
        let host = cpal::default_host();

        let output_device = host.default_output_device()
            .context("No output audio device")?;
        let input_device = host.default_input_device()
            .context("No input audio device")?;

        let out_name = output_device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into());
        let in_name = input_device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "Unknown".into());
        tracing::info!("cpal output device: {}", out_name);
        tracing::info!("cpal input device: {}", in_name);

        let out_default = output_device.default_output_config()
            .context("No default output config")?;
        let in_default = input_device.default_input_config()
            .context("No default input config")?;

        let out_channels = out_default.channels();
        let in_channels = in_default.channels();
        let out_rate = out_default.sample_rate();
        let in_rate = in_default.sample_rate();
        let out_fmt = out_default.sample_format();
        let in_fmt = in_default.sample_format();

        let out_config = cpal::StreamConfig {
            channels: out_channels,
            sample_rate: out_rate,
            buffer_size: cpal::BufferSize::Default,
        };
        let in_config = cpal::StreamConfig {
            channels: in_channels,
            sample_rate: in_rate,
            buffer_size: cpal::BufferSize::Default,
        };
        tracing::info!("cpal output: {}ch @ {}Hz fmt={:?}", out_channels, out_rate, out_fmt);
        tracing::info!("cpal input: {}ch @ {}Hz fmt={:?}", in_channels, in_rate, in_fmt);

        let (tx_sender, tx_receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(64);
        let tx_buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));

        let output_stream = {
            let tx_buf = tx_buffer.clone();
            output_device.build_output_stream(
                &out_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    while let Ok(chunk) = tx_receiver.try_recv() {
                        tx_buf.lock().unwrap().extend_from_slice(&chunk);
                    }
                    let mut buf = tx_buf.lock().unwrap();
                    for frame in data.chunks_mut(out_channels as usize) {
                        let sample = if buf.is_empty() { 0.0 } else { buf.remove(0) };
                        for ch in frame.iter_mut() {
                            *ch = sample;
                        }
                    }
                },
                |err| tracing::error!("Output stream error: {}", err),
                None,
            )?
        };

        let (rx_sender, rx_receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(1024);

        let input_stream = {
            input_device.build_input_stream(
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
            )?
        };

        output_stream.play()?;
        input_stream.play()?;

        Ok(Self {
            tx_sender,
            rx_receiver,
            _backend: AudioBackend::Cpal {
                _output_stream: output_stream,
                _input_stream: input_stream,
            },
        })
    }

    /// Create an engine using aplay/arecord subprocesses.
    /// Works reliably on Pi with USB audio because these tools use
    /// ALSA read/write mode (not MMAP) and respect ~/.asoundrc.
    fn try_subprocess() -> Result<Self> {
        use std::io::{Read as IoRead, Write as IoWrite};

        let mut output_child = spawn_aplay()?;
        let mut input_child = spawn_arecord()?;

        tracing::info!("Subprocess backend: aplay/arecord at {}Hz mono S16_LE", SUBPROCESS_RATE);

        let (tx_sender, tx_receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(64);
        let (rx_sender, rx_receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(1024);

        // Output thread: receive f32 samples, convert to i16, write to aplay stdin.
        // Auto-restarts aplay if the subprocess dies (USB audio glitch, device reset, etc.)
        let mut aplay_stdin = output_child.stdin.take().context("No aplay stdin")?;
        let output_thread = std::thread::Builder::new()
            .name("aplay-writer".into())
            .spawn(move || {
                let mut consecutive_failures: u32 = 0;
                while let Ok(samples) = tx_receiver.recv() {
                    let bytes = samples_to_s16_bytes(&samples);
                    if aplay_stdin.write_all(&bytes).is_ok() {
                        consecutive_failures = 0;
                        continue;
                    }
                    // aplay died — attempt restart
                    consecutive_failures += 1;
                    tracing::warn!(
                        "aplay subprocess died (failure {}/{}), attempting restart...",
                        consecutive_failures, MAX_SUBPROCESS_RESTARTS
                    );
                    if consecutive_failures >= MAX_SUBPROCESS_RESTARTS {
                        tracing::error!("aplay: max restart attempts reached, giving up");
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    match spawn_aplay() {
                        Ok(mut child) => match child.stdin.take() {
                            Some(new_stdin) => {
                                aplay_stdin = new_stdin;
                                tracing::info!("aplay restarted successfully");
                                // Retry the write with the new process
                                if aplay_stdin.write_all(&bytes).is_err() {
                                    tracing::error!("Write failed immediately after aplay restart");
                                    // Don't break; let the next recv() iteration detect the failure
                                }
                            }
                            None => {
                                tracing::error!("Restarted aplay had no stdin, giving up");
                                break;
                            }
                        },
                        Err(e) => {
                            tracing::error!("Failed to restart aplay: {}, giving up", e);
                            break;
                        }
                    }
                }
            })
            .context("Failed to spawn aplay writer thread")?;

        // Input thread: read i16 from arecord stdout, convert to f32, send.
        // Auto-restarts arecord if the subprocess dies.
        let mut arecord_stdout = input_child.stdout.take().context("No arecord stdout")?;
        let input_thread = std::thread::Builder::new()
            .name("arecord-reader".into())
            .spawn(move || {
                let mut buf = [0u8; 2048]; // 1024 i16 samples
                let mut consecutive_failures: u32 = 0;
                loop {
                    match arecord_stdout.read(&mut buf) {
                        Ok(0) | Err(_) => {
                            // arecord died — attempt restart
                            consecutive_failures += 1;
                            tracing::warn!(
                                "arecord subprocess died (failure {}/{}), attempting restart...",
                                consecutive_failures, MAX_SUBPROCESS_RESTARTS
                            );
                            if consecutive_failures >= MAX_SUBPROCESS_RESTARTS {
                                tracing::error!("arecord: max restart attempts reached, giving up");
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(500));
                            match spawn_arecord() {
                                Ok(mut child) => match child.stdout.take() {
                                    Some(new_stdout) => {
                                        arecord_stdout = new_stdout;
                                        tracing::info!("arecord restarted successfully");
                                        continue;
                                    }
                                    None => {
                                        tracing::error!("Restarted arecord had no stdout, giving up");
                                        break;
                                    }
                                },
                                Err(e) => {
                                    tracing::error!("Failed to restart arecord: {}, giving up", e);
                                    break;
                                }
                            }
                        }
                        Ok(n) => {
                            consecutive_failures = 0;
                            let samples: Vec<f32> = buf[..n]
                                .chunks_exact(2)
                                .map(|pair| {
                                    let i16_val = i16::from_le_bytes([pair[0], pair[1]]);
                                    i16_val as f32 / i16::MAX as f32
                                })
                                .collect();
                            if rx_sender.try_send(samples).is_err() {
                                // Channel full, drop oldest
                            }
                        }
                    }
                }
            })
            .context("Failed to spawn arecord reader thread")?;

        Ok(Self {
            tx_sender,
            rx_receiver,
            _backend: AudioBackend::Subprocess {
                _output_child: output_child,
                _input_child: input_child,
                _output_thread: Some(output_thread),
                _input_thread: Some(input_thread),
            },
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
