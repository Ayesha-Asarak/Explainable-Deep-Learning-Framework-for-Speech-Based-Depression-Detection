"""WavLM embedding extraction and participant-level SSL classifiers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    EMBEDDING_CACHE_DIR,
    MODEL_DIR,
    SAMPLE_RATE,
    SSL_EMBEDDING_DIM,
    SSL_MAX_DURATION,
    SSL_MAX_SEGMENTS,
    SSL_MODEL_ID,
    SSL_SEGMENT_DURATION,
    SSL_SEGMENT_OVERLAP,
)
from .data import load_audio_source
from .features import segment_audio_with_times


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class WavLMEmbedder:
    """Frozen WavLM encoder that maps waveforms to pooled embeddings."""

    def __init__(self, model_id: str = SSL_MODEL_ID, device=None):
        from transformers import AutoFeatureExtractor, WavLMModel

        cache_dir = str(MODEL_DIR / "huggingface_cache")
        self.device = device or pick_device()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model = WavLMModel.from_pretrained(model_id, cache_dir=cache_dir)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def embed_waveforms(self, waveforms: list[np.ndarray]) -> np.ndarray:
        if not waveforms:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        pooled_list = []
        # Process one clip at a time to avoid padding contamination in mean pooling.
        for wave in waveforms:
            wave = np.asarray(wave, dtype=np.float32)
            if wave.ndim > 1:
                wave = wave.mean(axis=-1)
            inputs = self.feature_extractor(
                wave,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=False,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state  # (1, T, H)
            pooled_list.append(hidden.mean(dim=1).squeeze(0))
        return torch.stack(pooled_list, dim=0).detach().cpu().numpy().astype(np.float32)


def aggregate_segment_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Convert variable-length segment embeddings into one participant vector.
    mean + std + max captures central tendency, variability, and peaks.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if len(x) == 0:
        dim = SSL_EMBEDDING_DIM
        return np.zeros(dim * 3, dtype=np.float32)
    return np.concatenate(
        [x.mean(axis=0), x.std(axis=0), x.max(axis=0)],
        axis=0,
    ).astype(np.float32)


def embedding_cache_path(fingerprint: str) -> Path:
    return EMBEDDING_CACHE_DIR / f"{fingerprint}.npz"


def extract_participant_embeddings(
    source: dict,
    embedder: WavLMEmbedder,
    max_duration: float = SSL_MAX_DURATION,
    segment_duration: float = SSL_SEGMENT_DURATION,
    overlap: float = SSL_SEGMENT_OVERLAP,
    max_segments: int = SSL_MAX_SEGMENTS,
    use_cache: bool = True,
) -> dict:
    """Extract and optionally cache segment embeddings for one participant."""
    fingerprint = source.get("source_fingerprint")
    if not fingerprint:
        from .data import _source_fingerprint

        fingerprint = _source_fingerprint(source)

    cache_file = embedding_cache_path(fingerprint)
    if use_cache and cache_file.exists():
        payload = np.load(cache_file, allow_pickle=False)
        return {
            "participant_id": source["participant_id"],
            "label": int(source["label"]),
            "fingerprint": fingerprint,
            "segment_embeddings": payload["segment_embeddings"],
            "starts": payload["starts"],
            "ends": payload["ends"],
            "participant_vector": payload["participant_vector"],
        }

    audio = load_audio_source(
        source,
        max_duration=max_duration,
        participant_only=True,
    )
    timed = segment_audio_with_times(audio, segment_duration, overlap)
    if max_segments and len(timed) > max_segments:
        indices = np.linspace(0, len(timed) - 1, max_segments, dtype=int)
        timed = [timed[i] for i in indices]

    waveforms = [item["audio"] for item in timed]
    segment_embeddings = embedder.embed_waveforms(waveforms)
    starts = np.array([item["start_sec"] for item in timed], dtype=np.float32)
    ends = np.array([item["end_sec"] for item in timed], dtype=np.float32)
    participant_vector = aggregate_segment_embeddings(segment_embeddings)

    if use_cache:
        np.savez_compressed(
            cache_file,
            segment_embeddings=segment_embeddings,
            starts=starts,
            ends=ends,
            participant_vector=participant_vector,
        )

    return {
        "participant_id": source["participant_id"],
        "label": int(source["label"]),
        "fingerprint": fingerprint,
        "segment_embeddings": segment_embeddings,
        "starts": starts,
        "ends": ends,
        "participant_vector": participant_vector,
    }


def candidate_searches(cv):
    return {
        "logistic_regression": GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=4000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]},
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "rbf_svm": GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            class_weight="balanced",
                            probability=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0, 50.0],
                "model__gamma": ["scale", 0.001, 0.01, 0.1],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "random_forest": GridSearchCV(
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "extra_trees": GridSearchCV(
            ExtraTreesClassifier(
                n_estimators=600,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
    }


def select_threshold(y_true, probabilities, metric: str = "balanced_accuracy"):
    """Pick a decision threshold from validation predictions only."""
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.2, 0.8, 61):
        preds = (probabilities >= threshold).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def evaluate_predictions(y_true, probabilities, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    preds = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
        "n_participants": int(len(y_true)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def bootstrap_accuracy_ci(y_true, probabilities, threshold, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        preds = (probabilities[idx] >= threshold).astype(int)
        scores.append(accuracy_score(y_true[idx], preds))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {
        "mean": float(np.mean(scores)),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
    }


class EmbeddingBagMLP(nn.Module):
    """Small MLP on aggregated participant embeddings."""

    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_embedding_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 80,
    lr: float = 1e-3,
    patience: int = 12,
):
    device = pick_device()
    model = EmbeddingBagMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_tr = torch.from_numpy(X_train).float().to(device)
    y_tr = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
    x_va = torch.from_numpy(X_val).float().to(device)
    y_va = np.asarray(y_val)

    best_state = None
    best_score = -np.inf
    wait = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_tr), y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(x_va)).cpu().numpy().ravel()
        threshold, score = select_threshold(y_va, probs)
        if score > best_score:
            best_score = score
            best_state = {
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "threshold": threshold,
                "score": score,
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state["state_dict"])
    model.eval()
    return model, best_state["threshold"], float(best_state["score"])


def save_ssl_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(artifact, handle)


def load_ssl_artifact(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)
