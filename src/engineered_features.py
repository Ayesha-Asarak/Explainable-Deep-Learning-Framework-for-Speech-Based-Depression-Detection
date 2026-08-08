"""Standard eGeMAPS and temporal prosody feature engineering."""

from functools import lru_cache

import librosa
import numpy as np
import opensmile

from .config import HOP_LENGTH, SAMPLE_RATE


@lru_cache(maxsize=1)
def _egemaps_extractor():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def _pause_run_durations(silent_frames, frame_seconds):
    padded = np.pad(silent_frames.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return (stops - starts).astype(np.float32) * frame_seconds


def extract_temporal_prosody(y):
    """Measure energy dynamics and pauses using a fixed relative VAD."""
    y = np.asarray(y, dtype=np.float32)
    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )[0]
    if not len(rms):
        return np.zeros(12, dtype=np.float32)

    # Frames below -35 dB relative to the recording peak are treated as
    # silence. Unlike the previous 20th-percentile rule, this ratio can vary.
    db = librosa.amplitude_to_db(rms, ref=np.max)
    silent = db < -35.0
    frame_seconds = HOP_LENGTH / SAMPLE_RATE
    pause_durations = _pause_run_durations(silent, frame_seconds)
    meaningful_pauses = pause_durations[pause_durations >= 0.2]
    duration_minutes = max(len(y) / SAMPLE_RATE / 60.0, 1e-6)

    x = np.arange(len(rms), dtype=np.float32)
    slope = (
        float(np.polyfit(x, rms, 1)[0])
        if len(rms) >= 2
        else 0.0
    )
    delta_std = float(np.std(np.diff(rms))) if len(rms) >= 2 else 0.0
    return np.asarray(
        [
            len(y) / SAMPLE_RATE,
            np.percentile(rms, 10),
            np.percentile(rms, 50),
            np.percentile(rms, 90),
            np.std(rms),
            np.mean(~silent),
            np.mean(silent),
            len(meaningful_pauses) / duration_minutes,
            np.mean(meaningful_pauses) if len(meaningful_pauses) else 0.0,
            np.max(meaningful_pauses) if len(meaningful_pauses) else 0.0,
            slope,
            delta_std,
        ],
        dtype=np.float32,
    )


def extract_engineered_features(y):
    """Return eGeMAPSv02 functionals plus custom temporal prosody."""
    y = np.asarray(y, dtype=np.float32)
    frame = _egemaps_extractor().process_signal(y, SAMPLE_RATE)
    egemaps = frame.to_numpy(dtype=np.float32).reshape(-1)
    temporal = extract_temporal_prosody(y)
    return np.concatenate([egemaps, temporal]).astype(np.float32)

