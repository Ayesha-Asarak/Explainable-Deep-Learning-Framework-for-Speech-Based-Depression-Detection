#!/usr/bin/env python3
"""Train semantic transcript + WavLM participant classifiers."""

import json
import pickle

import numpy as np
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    DATA_DIR,
    EMBEDDING_CACHE_DIR,
    MANIFEST_PATH,
    MODEL_DIR,
    MULTIMODAL_METADATA_PATH,
    MULTIMODAL_MODEL_PATH,
    SPLIT_PATH,
)
from src.data import (
    discover_audio_sources,
    load_json,
    load_participant_transcript,
)
from src.ssl_model import (
    bootstrap_accuracy_ci,
    evaluate_predictions,
    pick_device,
    select_threshold,
)

TEXT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_CACHE_DIR = MODEL_DIR / "text_embedding_cache"
TEXT_CACHE_DIR.mkdir(exist_ok=True)


class SemanticTextEmbedder:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        cache_dir = str(MODEL_DIR / "huggingface_cache")
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL_ID, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            TEXT_MODEL_ID, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text):
        words = text.split()
        chunks = [
            " ".join(words[start : start + 180])
            for start in range(0, len(words), 180)
        ]
        if len(chunks) > 8:
            indices = np.linspace(0, len(chunks) - 1, 8, dtype=int)
            chunks = [chunks[index] for index in indices]
        if not chunks:
            chunks = [""]
        inputs = self.tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        hidden = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors = pooled.cpu().numpy()
        return np.concatenate(
            [vectors.mean(axis=0), vectors.std(axis=0)]
        ).astype(np.float32)


def _candidate_search(cv):
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(random_state=42)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )
    svm = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(random_state=42)),
            (
                "model",
                SVC(
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    return {
        "pca_logistic": GridSearchCV(
            logistic,
            {
                "pca__n_components": [8, 16, 32],
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "pca_svm": GridSearchCV(
            svm,
            {
                "pca__n_components": [8, 16, 32],
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.1],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
    }


def _oof(estimator, X, y, cv):
    probabilities = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        model = clone(estimator).fit(X[train_idx], y[train_idx])
        probabilities[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return probabilities


def main():
    split = load_json(SPLIT_PATH)
    manifest = load_json(MANIFEST_PATH)["participants"]
    by_pid = {item["participant_id"]: item for item in manifest}
    sources, conflicts = discover_audio_sources(DATA_DIR, include_zips=True)

    embedder = None
    rows = []
    for number, source in enumerate(sources, 1):
        pid = source["participant_id"]
        fingerprint = by_pid[pid]["source_fingerprint"]
        speech_path = EMBEDDING_CACHE_DIR / f"{fingerprint}.npz"
        text_path = TEXT_CACHE_DIR / f"{fingerprint}.npy"
        text = load_participant_transcript(source)
        if not text or not speech_path.exists():
            continue
        if text_path.exists():
            text_vector = np.load(text_path)
        else:
            if embedder is None:
                print(f"Loading semantic text encoder: {TEXT_MODEL_ID}")
                embedder = SemanticTextEmbedder()
            print(f"[{number}/{len(sources)}] encoding transcript {pid}")
            text_vector = embedder.encode(text)
            np.save(text_path, text_vector)
        speech_vector = np.load(speech_path)["participant_vector"]
        rows.append(
            {
                "pid": pid,
                "label": int(source["label"]),
                "text": text_vector.astype(np.float32),
                "speech": speech_vector.astype(np.float32),
            }
        )

    train_ids = set(split["train_participant_ids"])
    test_ids = set(split["test_participant_ids"])
    train = [row for row in rows if row["pid"] in train_ids]
    test = [row for row in rows if row["pid"] in test_ids]
    y_train = np.asarray([row["label"] for row in train])
    y_test = np.asarray([row["label"] for row in test])
    families = {
        "semantic_text": (
            np.stack([row["text"] for row in train]),
            np.stack([row["text"] for row in test]),
        ),
        "wavlm_speech": (
            np.stack([row["speech"] for row in train]),
            np.stack([row["speech"] for row in test]),
        ),
        "semantic_text_wavlm": (
            np.stack(
                [
                    np.concatenate([row["text"], row["speech"]])
                    for row in train
                ]
            ),
            np.stack(
                [
                    np.concatenate([row["text"], row["speech"]])
                    for row in test
                ]
            ),
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = []
    comparison = {}
    for family, (X_train, X_test) in families.items():
        for model_name, search in _candidate_search(cv).items():
            name = f"{family}:{model_name}"
            print(f"Cross-validating {name}...")
            search.fit(X_train, y_train)
            score = float(search.best_score_)
            comparison[name] = {
                "cv_balanced_accuracy": score,
                "best_params": search.best_params_,
            }
            candidates.append(
                (score, name, family, search.best_estimator_, X_train, X_test)
            )
            print(f"  CV balanced accuracy={score:.3f}")

    score, name, family, estimator, X_train, X_test = max(
        candidates, key=lambda item: item[0]
    )
    oof = _oof(estimator, X_train, y_train, cv)
    threshold, oof_score = select_threshold(y_train, oof)
    model = clone(estimator).fit(X_train, y_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_metrics = evaluate_predictions(y_train, train_prob, threshold)
    test_metrics = evaluate_predictions(y_test, test_prob, threshold)
    test_metrics["accuracy_bootstrap_ci95"] = bootstrap_accuracy_ci(
        y_test, test_prob, threshold
    )
    print(
        "HELD-OUT participant test:",
        {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in test_metrics.items()
            if key not in {"confusion_matrix", "accuracy_bootstrap_ci95"}
        },
    )

    all_labels = np.asarray([row["label"] for row in rows])
    if family == "semantic_text":
        X_all = np.stack([row["text"] for row in rows])
    elif family == "wavlm_speech":
        X_all = np.stack([row["speech"] for row in rows])
    else:
        X_all = np.stack(
            [
                np.concatenate([row["text"], row["speech"]])
                for row in rows
            ]
        )
    deploy_model = clone(estimator).fit(X_all, all_labels)
    artifact = {
        "model_type": "semantic_multimodal",
        "text_model_id": TEXT_MODEL_ID,
        "feature_family": family,
        "selected_model": name,
        "model": deploy_model,
        "threshold": float(threshold),
        "training_cv_balanced_accuracy": float(score),
        "training_oof_balanced_accuracy": float(oof_score),
        "held_out_test_metrics": test_metrics,
        "requires_transcript": family != "wavlm_speech",
    }
    with open(MULTIMODAL_MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)
    MULTIMODAL_METADATA_PATH.write_text(
        json.dumps(
            {
                "n_participants": len(rows),
                "selected_model": name,
                "feature_family": family,
                "threshold": float(threshold),
                "training_cv_balanced_accuracy": float(score),
                "training_oof_balanced_accuracy": float(oof_score),
                "train_metrics": train_metrics,
                "held_out_test_metrics": test_metrics,
                "candidate_comparison": comparison,
                "target_accuracy_range": [0.75, 0.85],
                "target_met": bool(
                    0.75 <= test_metrics["accuracy"] <= 0.85
                ),
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved semantic multimodal model to {MULTIMODAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
