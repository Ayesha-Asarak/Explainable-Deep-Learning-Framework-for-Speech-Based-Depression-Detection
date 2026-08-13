"""Acoustic feature extraction for speech-based depression detection."""

import numpy as np
import librosa

from .config import (
    SAMPLE_RATE,
    N_MELS,
    N_MFCC,
    HOP_LENGTH,
    N_FFT,
    FEATURE_NAMES,
)

def preprocess_audio(
    y: np.ndarray,
    trim: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Optionally trim and peak-normalize an audio array."""
    y = np.asarray(y, dtype=np.float32)
    if trim:
        y = librosa.effects.trim(y, top_db=25)[0]
    if len(y) == 0:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)
    if normalize:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
    return y.astype(np.float32)


def load_audio(
    path,
    max_duration=None,
    trim: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Load audio as 16 kHz mono, then optionally trim and normalize."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
    return preprocess_audio(y, trim=trim, normalize=normalize)


def segment_audio(y: np.ndarray, segment_duration: float, overlap: float) -> list[np.ndarray]:
    """Split audio into overlapping fixed-length segments."""
    return [s["audio"] for s in segment_audio_with_times(y, segment_duration, overlap)]

def segment_audio_with_times(
    y: np.ndarray, segment_duration: float, overlap: float
) -> list[dict]:
    """Split audio into segments with start/end times in seconds."""
    seg_len = int(segment_duration * SAMPLE_RATE)
    hop = int(seg_len * (1 - overlap))
    if len(y) < seg_len:
        padded = np.pad(y, (0, seg_len - len(y)))
        duration = len(y) / SAMPLE_RATE
        return [{"audio": padded, "start_sec": 0.0, "end_sec": max(duration, segment_duration)}]
    segments = []
    for start in range(0, len(y) - seg_len + 1, hop):
        start_sec = start / SAMPLE_RATE
        end_sec = (start + seg_len) / SAMPLE_RATE
        segments.append({
            "audio": y[start : start + seg_len],
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
        })
    return segments


def compute_full_mel_spectrogram(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return log-mel spectrogram (n_mels, time) and time axis in seconds."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(np.arange(log_mel.shape[1]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    return log_mel.astype(np.float32), times.astype(np.float32)


def compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """Return log-mel spectrogram shaped (1, n_mels, time)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel[np.newaxis, :, :].astype(np.float32)


def extract_acoustic_features(y: np.ndarray) -> dict[str, float]:
    """Extract interpretable acoustic features from a speech segment."""
    if len(y) < SAMPLE_RATE * 0.5:
        return {name: 0.0 for name in FEATURE_NAMES}

    # piptrack is much faster than pyin for real-time inference
    pitches, _ = librosa.piptrack(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    pitch_vals = []
    for t in range(pitches.shape[1]):
        col = pitches[:, t]
        idx = col.argmax()
        if col[idx] > 0:
            pitch_vals.append(col[idx])
    voiced_f0 = np.array(pitch_vals) if pitch_vals else np.array([0.0])

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)

    energy_threshold = np.percentile(rms, 20)
    speech_frames = rms > energy_threshold
    speech_rate = float(np.mean(speech_frames))
    pause_ratio = float(1.0 - speech_rate)

    features = {
        "pitch_mean_hz": float(np.mean(voiced_f0)) if len(voiced_f0) else 0.0,
        "pitch_std_hz": float(np.std(voiced_f0)) if len(voiced_f0) else 0.0,
        "energy_mean": float(np.mean(rms)),
        "energy_std": float(np.std(rms)),
        "speech_rate": speech_rate,
        "pause_ratio": pause_ratio,
        "zero_crossing_rate": float(np.mean(zcr)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
    }
    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfccs[i]))
    return features


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)


def augment_audio(y: np.ndarray) -> np.ndarray:
    """Light augmentation for small datasets."""
    choice = np.random.randint(0, 4)
    if choice == 0:
        rate = np.random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(y, rate=rate)
    if choice == 1:
        steps = np.random.uniform(-2, 2)
        return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=steps)
    if choice == 2:
        noise = np.random.randn(len(y)).astype(np.float32) * 0.005
        return y + noise
    return y
