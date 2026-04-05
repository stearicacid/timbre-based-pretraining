import logging
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List

import ddsp
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _build_ddsp_components(sample_rate: int, n_samples: int):
    harmonic_synth = ddsp.synths.Harmonic(
        n_samples=n_samples,
        sample_rate=sample_rate,
        scale_fn=None,
        name="harmonic",
    )
    noise_synth = ddsp.synths.FilteredNoise(n_samples=n_samples, name="noise", initial_bias=0.0)
    reverb_effect = ddsp.effects.Reverb(name="reverb")
    return harmonic_synth, noise_synth, reverb_effect


def normalize_harmonic_distribution(harmonic_dist: np.ndarray) -> np.ndarray:
    harmonic_dist = np.maximum(harmonic_dist, 0.0)
    total = np.sum(harmonic_dist)
    if total <= 0:
        return np.ones_like(harmonic_dist) / len(harmonic_dist)
    return harmonic_dist / total


def make_decay_ir(sample_rate: int, seconds: float = 2.0) -> np.ndarray:
    ir_size = max(1, int(sample_rate * seconds))
    ir = 0.01 * np.random.randn(ir_size).astype(np.float32)
    n_fade_in = min(32, ir_size // 16)
    if n_fade_in > 0:
        ir[:n_fade_in] *= np.linspace(0.0, 1.0, n_fade_in, dtype=np.float32)
    decay = np.exp(-np.linspace(0.0, 6.0, ir_size, dtype=np.float32))
    ir *= decay
    ir /= np.max(np.abs(ir)) + 1e-6
    return ir[np.newaxis, :]


def synthesize_audio_from_harmonic(
    harmonic_dist: np.ndarray,
    f0_hz: float,
    amplitude: float,
    sample_rate: int,
    n_samples: int,
    hop_size: int = 64,
    noise_level: float = 0.0,
    reverb_level: float = 0.0,
    n_filter_banks: int = 20,
    ir_seconds: float = 2.0,
) -> np.ndarray:
    noise_level = float(np.clip(noise_level, 0.0, 0.01))
    reverb_level = float(np.clip(reverb_level, 0.0, 0.5))

    n_frames = max(1, n_samples // hop_size)
    n_samples = n_frames * hop_size

    if harmonic_dist.shape != (45,):
        raise ValueError(f"Expected harmonic_dist shape (45,), got {harmonic_dist.shape}")

    harmonic_synth, noise_synth, reverb_effect = _build_ddsp_components(sample_rate, n_samples)

    harmonic_dist_td = np.tile(harmonic_dist[np.newaxis, :], (n_frames, 1))

    # Create control signals
    f0_hz_control = np.ones(n_frames, dtype=np.float32) * f0_hz
    amps_control = np.ones(n_frames, dtype=np.float32) * amplitude

    # Convert to TensorFlow tensors
    harmonic_dist_tf = tf.convert_to_tensor(harmonic_dist_td, dtype=tf.float32)[tf.newaxis, ...]  # [1, n_frames, 45]
    f0_hz_tf = tf.convert_to_tensor(f0_hz_control, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]   # [1, n_frames, 1]
    amps_tf = tf.convert_to_tensor(amps_control, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]     # [1, n_frames, 1]

    # Noise magnitudes: constant over time & bands; scale with amplitude
    magnitudes_tf = tf.broadcast_to(amps_tf, [1, n_frames, n_filter_banks])

    # IR for reverb (fixed shape; reverb amount controlled by wet mix)
    ir_np = make_decay_ir(sample_rate, seconds=ir_seconds).astype(np.float32)  # [1, ir_size]
    ir_tf = tf.convert_to_tensor(ir_np, dtype=tf.float32)

    logger.debug(f"Tensor shapes:")
    logger.debug(f"  harmonic_dist_tf: {harmonic_dist_tf.shape}")
    logger.debug(f"  f0_hz_tf: {f0_hz_tf.shape}")
    logger.debug(f"  amps_tf: {amps_tf.shape}")
    logger.debug(f"  magnitudes_tf: {magnitudes_tf.shape}")

    # Synthesize harmonic and noise
    # Keras requires the first call argument to be positional.
    audio_harm = harmonic_synth(amps_tf, harmonic_dist_tf, f0_hz_tf)  # [1, n_samples]

    audio_noise = noise_synth(magnitudes_tf)  # [1, n_samples]
    audio_dry = audio_harm + noise_level * audio_noise

    # Apply reverb (convolution). ddsp.effects.Reverb expects (audio, ir) or (ir, audio)
    try:
        audio_wet = reverb_effect(audio_dry, ir_tf)
    except Exception:
        audio_wet = reverb_effect(ir_tf, audio_dry)

    audio_out = (1.0 - reverb_level) * audio_dry + reverb_level * audio_wet
    return audio_out.numpy().flatten()


def apply_fade(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0 or len(audio) <= 2 * fade_samples:
        return audio
    fade_window = np.linspace(0, 1, fade_samples)
    out = audio.copy()
    out[:fade_samples] *= fade_window
    out[-fade_samples:] *= fade_window[::-1]
    return out


def synthesize_note_ddsp(
    harmonic_dist: np.ndarray,
    f0_hz: float,
    amplitude: float,
    duration: float,
    sample_rate: int = 16000,
    hop_size: int = 64,
    magnitudes_level: float = 0.1,
    ir_level: float = 0.1,
) -> np.ndarray:
    duration_samp = int(round(duration * sample_rate))
    n_frames = max(1, duration_samp // hop_size)
    n_samples = n_frames * hop_size
    if n_samples <= 0:
        return np.array([])

    harmonic_dist = normalize_harmonic_distribution(harmonic_dist)
    audio = synthesize_audio_from_harmonic(
        harmonic_dist=harmonic_dist,
        f0_hz=f0_hz,
        amplitude=amplitude,
        sample_rate=sample_rate,
        n_samples=n_samples,
        hop_size=hop_size,
        noise_level=magnitudes_level,
        reverb_level=ir_level,
        n_filter_banks=20,
        ir_seconds=2.0,
    )
    return apply_fade(audio, int(0.005 * sample_rate))


def synthesize_midi_track(
    notes: List[Dict[str, float]],
    program_harmonic_dists: Dict[int, np.ndarray],
    program_cluster_params: Dict[int, tuple],
    sample_rate: int = 16000,
) -> np.ndarray:
    if not notes:
        return np.array([])

    track_samples = int(max(note["offset"] for note in notes) * sample_rate)
    audio_track = np.zeros(track_samples)

    notes_by_program: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for note in notes:
        notes_by_program[note["program"]].append(note)

    for program, program_notes in notes_by_program.items():
        if program not in program_cluster_params or program not in program_harmonic_dists:
            continue
        _, magnitudes_level, ir_level = program_cluster_params[program]
        harmonic_dist = program_harmonic_dists[program]

        for note in program_notes:
            duration = note["offset"] - note["onset"]
            if duration <= 0:
                continue

            f0_hz = 440.0 * (2.0 ** ((note["pitch"] - 69) / 12.0))
            amplitude = note["velocity"] / 127.0 * 0.5
            note_audio = synthesize_note_ddsp(
                harmonic_dist,
                f0_hz,
                amplitude,
                duration,
                sample_rate=sample_rate,
                hop_size=64,
                magnitudes_level=magnitudes_level,
                ir_level=ir_level,
            )
            if len(note_audio) == 0:
                continue

            start_sample = int(note["onset"] * sample_rate)
            end_sample = min(track_samples, start_sample + len(note_audio))
            audio_track[start_sample:end_sample] += note_audio[: end_sample - start_sample]

    peak = np.max(np.abs(audio_track))
    if peak > 1.0:
        audio_track = np.tanh(audio_track / peak) * 0.95
    return audio_track


def cut_into_segments(
    audio: np.ndarray,
    midi_notes: List[Dict[str, float]],
    segment_duration: float = 10.0,
    sample_rate: int = 16000,
):
    segment_samples = int(segment_duration * sample_rate)
    total_samples = len(audio)
    segments = []

    for start_sample in range(0, total_samples, segment_samples):
        end_sample = start_sample + segment_samples
        if end_sample > total_samples:
            break

        audio_segment = audio[start_sample:end_sample]
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate

        midi_segment = []
        for note in midi_notes:
            if note["onset"] < end_time and note["offset"] > start_time:
                adjusted = note.copy()
                adjusted["onset"] = max(0, note["onset"] - start_time)
                adjusted["offset"] = min(segment_duration, note["offset"] - start_time)
                if adjusted["offset"] > adjusted["onset"]:
                    midi_segment.append(adjusted)

        segments.append(
            {
                "audio": audio_segment,
                "midi": midi_segment,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    return segments
