#!/usr/bin/env python3
"""Train depression detection CNN and feature-based explainer."""

import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import DATA_DIR, MODEL_PATH, SCALER_PATH, METADATA_PATH, SEGMENT_DURATION, SEGMENT_OVERLAP
from src.data import (
    build_segment_dataset,
    MelSpectrogramDataset,
    participant_level_split,
    save_training_metadata,
    discover_audio_files,
)
from src.features import features_to_vector, segment_audio, load_audio, extract_acoustic_features
from src.model import DepressionCNN, FeatureClassifier

MAX_DURATION = 60.0
MAX_SEGMENTS = 12


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
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


def train_feature_model(X, y, n_features, device, epochs=30):
    model = FeatureClassifier(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().unsqueeze(1).to(device)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        criterion(model(X_t), y_t).backward()
        optimizer.step()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading dataset (fast mode: 60s per file, max 12 segments)...")

    spectrograms, labels, metadata = build_segment_dataset(
        DATA_DIR,
        augment=False,
        max_duration=MAX_DURATION,
        max_segments_per_file=MAX_SEGMENTS,
    )
    print(f"Segments: {len(labels)} | depressed: {sum(labels)} | non-depressed: {len(labels) - sum(labels)}")

    train_idx, test_idx = participant_level_split(metadata, test_ratio=0.22)
    dataset = MelSpectrogramDataset(spectrograms, labels)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=8)

    model = DepressionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Training CNN...")
    for epoch in range(1, 13):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % 4 == 0:
            m = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch:02d} loss={loss:.4f} f1={m['f1']:.3f}")

    metrics = evaluate(model, test_loader, device)
    print("Test:", {k: round(v, 3) for k, v in metrics.items()})

    print("Saving deployment model...")
    deploy = DepressionCNN().to(device)
    deploy.load_state_dict(model.state_dict())
    full_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    opt2 = torch.optim.Adam(deploy.parameters(), lr=5e-4)
    for _ in range(8):
        train_epoch(deploy, full_loader, opt2, criterion, device)

    torch.save({"cnn_state_dict": deploy.state_dict(), "n_mels": 128}, MODEL_PATH)

    print("Training feature model...")
    all_features, all_labels = [], []
    for audio_path, label, _ in discover_audio_files(DATA_DIR):
        y = load_audio(str(audio_path), max_duration=MAX_DURATION)
        segs = segment_audio(y, SEGMENT_DURATION, SEGMENT_OVERLAP)
        if len(segs) > MAX_SEGMENTS:
            idx = np.linspace(0, len(segs) - 1, MAX_SEGMENTS, dtype=int)
            segs = [segs[i] for i in idx]
        for seg in segs:
            all_features.append(features_to_vector(extract_acoustic_features(seg)))
            all_labels.append(label)

    X = np.array(all_features)
    y = np.array(all_labels, dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    feat_model = train_feature_model(X_scaled, y, X.shape[1], device, epochs=30)
    torch.save(
        {"feature_model_state_dict": feat_model.state_dict(), "n_features": X.shape[1]},
        MODEL_PATH.parent / "feature_model.pt",
    )

    save_training_metadata(METADATA_PATH, {
        "n_participants": len(discover_audio_files(DATA_DIR)),
        "n_segments": len(labels),
        "test_metrics": metrics,
    })
    print(f"Done. Models saved to {MODEL_PATH.parent}/")


if __name__ == "__main__":
    main()
