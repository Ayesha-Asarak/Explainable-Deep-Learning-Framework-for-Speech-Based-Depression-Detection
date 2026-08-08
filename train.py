#!/usr/bin/env python3
"""Train depression detection models from extracted WAVs and ZIP archives."""

import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import DATA_DIR, MODEL_PATH, SCALER_PATH, METADATA_PATH, SEGMENT_DURATION, SEGMENT_OVERLAP
from src.data import (
    build_segment_dataset,
    MelSpectrogramDataset,
    participant_level_split,
    save_training_metadata,
    discover_audio_sources,
)
from src.model import DepressionCNN, FeatureClassifier

MAX_DURATION = 60.0
MAX_SEGMENTS = 12
CNN_EPOCHS = 40
PARTICIPANT_BATCH_SIZE = 8


class ParticipantBagDataset(Dataset):
    """
    One item = all sampled speech segments for one participant.

    This avoids claiming that every 3-second segment independently proves
    depression. The loss is applied after averaging segment probabilities.
    """

    def __init__(
        self,
        spectrograms,
        metadata,
        participant_ids,
        augment=False,
    ):
        self.spectrograms = spectrograms
        self.metadata = metadata
        self.augment = augment
        grouped = {}
        for index, item in enumerate(metadata):
            pid = item["participant_id"]
            if pid in participant_ids:
                grouped.setdefault(
                    pid,
                    {"indices": [], "label": item["label"]},
                )
                grouped[pid]["indices"].append(index)
        self.participants = sorted(grouped.items())

    def __len__(self):
        return len(self.participants)

    def _spec_augment(self, x):
        """Light frequency/time masking; zero is the normalized mean."""
        x = x.copy()
        _, n_mels, n_frames = x.shape
        if np.random.rand() < 0.5:
            width = np.random.randint(1, min(9, n_mels) + 1)
            start = np.random.randint(0, n_mels - width + 1)
            x[:, start:start + width, :] = 0
        if np.random.rand() < 0.5:
            width = np.random.randint(1, min(7, n_frames) + 1)
            start = np.random.randint(0, n_frames - width + 1)
            x[:, :, start:start + width] = 0
        return x

    def __getitem__(self, index):
        pid, info = self.participants[index]
        specs = []
        for segment_index in info["indices"]:
            spec = self.spectrograms[segment_index]
            specs.append(self._spec_augment(spec) if self.augment else spec)
        x = torch.from_numpy(np.stack(specs)).float()
        y = torch.tensor(info["label"], dtype=torch.float32)
        return x, y, pid


def aggregate_segment_logits(segment_logits, batch_size, segments_per_person):
    """Match inference: sigmoid each segment, then average probabilities."""
    segment_probs = torch.sigmoid(
        segment_logits.view(batch_size, segments_per_person)
    )
    mean_probs = segment_probs.mean(dim=1, keepdim=True).clamp(1e-6, 1 - 1e-6)
    return torch.logit(mean_probs), mean_probs


def train_bag_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        batch_size, segment_count = x.shape[:2]
        optimizer.zero_grad()
        segment_logits = model(x.flatten(0, 1))
        participant_logits, _ = aggregate_segment_logits(
            segment_logits, batch_size, segment_count
        )
        loss = criterion(participant_logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        prob = torch.sigmoid(model(x.to(device))).cpu().numpy().flatten()
        preds.extend((prob >= 0.5).astype(int).tolist())
        labels.extend(y.numpy().astype(int).tolist())
    if not labels:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


@torch.no_grad()
def evaluate_bags(model, loader, device):
    """Evaluate one aggregated decision per held-out participant."""
    model.eval()
    labels, preds, probabilities = [], [], []
    for x, y, _ in loader:
        x = x.to(device)
        batch_size, segment_count = x.shape[:2]
        segment_logits = model(x.flatten(0, 1))
        _, mean_probs = aggregate_segment_logits(
            segment_logits, batch_size, segment_count
        )
        probabilities.extend(mean_probs.cpu().numpy().flatten().tolist())
        labels.extend(y.numpy().astype(int).tolist())
        preds.extend((mean_probs.cpu().numpy().flatten() >= 0.5).astype(int).tolist())

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "n_participants": len(labels),
        "mean_probability": float(np.mean(probabilities)),
    }


def train_feature_model(X, y, n_features, device, epochs=30):
    model = FeatureClassifier(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    positives = max(float(np.sum(y == 1)), 1.0)
    negatives = max(float(np.sum(y == 0)), 1.0)
    pos_weight = torch.tensor([negatives / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().unsqueeze(1).to(device)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        criterion(model(X_t), y_t).backward()
        optimizer.step()
    return model


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        "Loading extracted WAVs + ZIP archives "
        "(streaming ZIP audio in memory; no extraction)..."
    )

    spectrograms, labels, metadata = build_segment_dataset(
        DATA_DIR,
        augment=False,
        max_duration=MAX_DURATION,
        max_segments_per_file=MAX_SEGMENTS,
        include_zips=True,
        participant_only=True,
    )
    if not labels:
        raise RuntimeError("No usable *_AUDIO.wav files found in folders or ZIP archives")

    participant_ids = {m["participant_id"] for m in metadata}
    print(f"Segments: {len(labels)} | depressed: {sum(labels)} | non-depressed: {len(labels) - sum(labels)}")
    print(f"Unique participants used: {len(participant_ids)}")

    train_idx, test_idx = participant_level_split(metadata, test_ratio=0.22)
    train_pids = {metadata[i]["participant_id"] for i in train_idx}
    test_pids = {metadata[i]["participant_id"] for i in test_idx}
    print(
        f"Participant split: {len(train_pids)} train / {len(test_pids)} test "
        "(no participant overlap)"
    )

    dataset = MelSpectrogramDataset(spectrograms, labels)
    train_bags = ParticipantBagDataset(
        spectrograms, metadata, train_pids, augment=True
    )
    test_bags = ParticipantBagDataset(
        spectrograms, metadata, test_pids, augment=False
    )
    train_loader = DataLoader(
        train_bags,
        batch_size=PARTICIPANT_BATCH_SIZE,
        shuffle=True,
    )
    test_bag_loader = DataLoader(
        test_bags,
        batch_size=PARTICIPANT_BATCH_SIZE,
        shuffle=False,
    )
    # Retained only to report diagnostic segment-level metrics.
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=8)

    model = DepressionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_labels = np.array([labels[i] for i in train_idx])
    train_positives = max(float(np.sum(train_labels == 1)), 1.0)
    train_negatives = max(float(np.sum(train_labels == 0)), 1.0)
    pos_weight = torch.tensor(
        [train_negatives / train_positives],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Positive-class loss weight: {pos_weight.item():.3f}")

    print("Training CNN...")
    for epoch in range(1, CNN_EPOCHS + 1):
        loss = train_bag_epoch(
            model, train_loader, optimizer, criterion, device
        )
        if epoch == 1 or epoch % 4 == 0 or epoch == CNN_EPOCHS:
            print(f"  Epoch {epoch:02d} loss={loss:.4f}")

    segment_metrics = evaluate(model, test_loader, device)
    participant_metrics = evaluate_bags(model, test_bag_loader, device)
    print(
        "Segment-level test:",
        {k: round(v, 3) for k, v in segment_metrics.items()},
    )
    print(
        "Participant-level test:",
        {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in participant_metrics.items()
        },
    )

    print("Saving deployment model...")
    deploy = DepressionCNN().to(device)
    deploy.load_state_dict(model.state_dict())
    full_bags = ParticipantBagDataset(
        spectrograms, metadata, participant_ids, augment=True
    )
    full_loader = DataLoader(
        full_bags,
        batch_size=PARTICIPANT_BATCH_SIZE,
        shuffle=True,
    )
    opt2 = torch.optim.Adam(deploy.parameters(), lr=5e-4)
    full_positive = max(
        float(sum(info["label"] == 1 for _, info in full_bags.participants)),
        1.0,
    )
    full_negative = max(
        float(sum(info["label"] == 0 for _, info in full_bags.participants)),
        1.0,
    )
    deploy_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [full_negative / full_positive],
            dtype=torch.float32,
            device=device,
        )
    )
    for _ in range(10):
        train_bag_epoch(
            deploy, full_loader, opt2, deploy_criterion, device
        )

    torch.save(
        {
            "cnn_state_dict": deploy.state_dict(),
            "n_mels": 128,
            "training_method": "participant_bag_probability_aggregation",
            "segments_per_participant": MAX_SEGMENTS,
        },
        MODEL_PATH,
    )

    print("Training feature model from cached segment features...")
    X = np.array([m["acoustic_features"] for m in metadata])
    y = np.array(labels, dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    feat_model = train_feature_model(X_scaled, y, X.shape[1], device, epochs=30)
    torch.save(
        {"feature_model_state_dict": feat_model.state_dict(), "n_features": X.shape[1]},
        MODEL_PATH.parent / "feature_model.pt",
    )

    sources, conflicts = discover_audio_sources(DATA_DIR, include_zips=True)
    save_training_metadata(METADATA_PATH, {
        "n_participants": len(participant_ids),
        "n_segments": len(labels),
        "n_depressed_participants": len({
            m["participant_id"] for m in metadata if m["label"] == 1
        }),
        "n_non_depressed_participants": len({
            m["participant_id"] for m in metadata if m["label"] == 0
        }),
        "n_zip_sources": sum(s["kind"] == "zip" for s in sources),
        "n_extracted_sources": sum(s["kind"] == "file" for s in sources),
        "excluded_label_conflicts": conflicts,
        "train_participants": len(train_pids),
        "test_participants": len(test_pids),
        "cnn_training_method": "participant_bag_probability_aggregation",
        "cnn_epochs": CNN_EPOCHS,
        "segments_per_participant": MAX_SEGMENTS,
        "participant_only_audio": True,
        "segment_level_test_metrics": segment_metrics,
        "participant_level_test_metrics": participant_metrics,
        # Backward-compatible key used by existing documentation/UI.
        "test_metrics": participant_metrics,
    })
    print(f"Done. Models saved to {MODEL_PATH.parent}/")


if __name__ == "__main__":
    main()
