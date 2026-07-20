"""Dataset loading for DAIC-WOZ style interview recordings."""

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    DATA_DIR,
    DEPRESSED_DIR,
    NON_DEPRESSED_DIR,
    SEGMENT_DURATION,
    SEGMENT_OVERLAP,
)
from .features import (
    load_audio,
    segment_audio,
    compute_mel_spectrogram,
    extract_acoustic_features,
    features_to_vector,
    augment_audio,
)


def discover_audio_files(data_dir: Path) -> list[tuple[Path, int, str]]:
    """Return (audio_path, label, participant_id) tuples. label: 1=depressed, 0=non-depressed."""
    samples = []
    for label, folder in [(1, DEPRESSED_DIR), (0, NON_DEPRESSED_DIR)]:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
        for participant_dir in sorted(folder_path.iterdir()):
            if not participant_dir.is_dir():
                continue
            audio_files = list(participant_dir.glob("*_AUDIO.wav"))
            if audio_files:
                pid = participant_dir.name.replace("_P", "")
                samples.append((audio_files[0], label, pid))
    return samples


def build_segment_dataset(
    data_dir: Path,
    augment: bool = False,
    augment_factor: int = 3,
    max_duration: float = 180.0,
    max_segments_per_file=None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """
    Build mel spectrogram segments and acoustic feature vectors.
    Returns (spectrograms, labels, metadata_per_segment).
    """
    spectrograms, labels, metadata = [], [], []
    for audio_path, label, pid in discover_audio_files(data_dir):
        y = load_audio(str(audio_path), max_duration=max_duration)
        segments = segment_audio(y, SEGMENT_DURATION, SEGMENT_OVERLAP)
        if max_segments_per_file and len(segments) > max_segments_per_file:
            indices = np.linspace(0, len(segments) - 1, max_segments_per_file, dtype=int)
            segments = [segments[i] for i in indices]

        variants = [segments]
        if augment:
            for _ in range(augment_factor):
                aug_segments = []
                for seg in segments:
                    aug_y = augment_audio(seg)
                    aug_segments.append(aug_y[: len(seg)] if len(aug_y) >= len(seg) else np.pad(aug_y, (0, len(seg) - len(aug_y))))
                variants.append(aug_segments)

        for variant_idx, segs in enumerate(variants):
            for seg_idx, seg in enumerate(segs):
                spec = compute_mel_spectrogram(seg)
                feats = extract_acoustic_features(seg)
                spectrograms.append(spec)
                labels.append(label)
                metadata.append({
                    "participant_id": pid,
                    "audio_path": str(audio_path),
                    "segment_idx": seg_idx,
                    "variant": variant_idx,
                    "label": label,
                })
    return spectrograms, labels, metadata


class MelSpectrogramDataset(Dataset):
    def __init__(self, spectrograms: list[np.ndarray], labels: list[int]):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.spectrograms[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def participant_level_split(
    metadata: list[dict], test_ratio: float = 0.22
) -> tuple[list[int], list[int]]:
    """Split indices by participant to avoid data leakage."""
    participants = {}
    for i, m in enumerate(metadata):
        participants.setdefault(m["participant_id"], []).append(i)

    pids = sorted(participants.keys())
    n_test = max(1, int(len(pids) * test_ratio))
    rng = np.random.RandomState(42)
    test_pids = set(rng.choice(pids, size=n_test, replace=False))

    train_idx, test_idx = [], []
    for pid, indices in participants.items():
        if pid in test_pids:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)
    return train_idx, test_idx


def save_training_metadata(path: Path, info: dict) -> None:
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
