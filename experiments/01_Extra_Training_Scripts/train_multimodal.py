#!/usr/bin/env python3
"""Train a leakage-safe speech + participant-transcript classifier."""

import json
import pickle

import numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import (
    DATA_DIR,
    EMBEDDING_CACHE_DIR,
    MANIFEST_PATH,
    MULTIMODAL_METADATA_PATH,
    MULTIMODAL_MODEL_PATH,
    SPLIT_PATH,
    SSL_MODEL_ID,
)
from src.data import (
    discover_audio_sources,
    load_json,
    load_participant_transcript,
)
from src.ssl_model import (
    bootstrap_accuracy_ci,
    evaluate_predictions,
    select_threshold,
)


def _probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return expit(model.decision_function(X))


def _oof_probabilities(estimator, X, y, cv):
    probabilities = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in cv.split(np.arange(len(y)), y):
        model = clone(estimator)
        train_x = [X[i] for i in train_idx] if isinstance(X, list) else X[train_idx]
        val_x = [X[i] for i in val_idx] if isinstance(X, list) else X[val_idx]
        model.fit(train_x, y[train_idx])
        probabilities[val_idx] = _probabilities(model, val_x)
    return probabilities


def _text_candidates(cv):
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        max_features=30000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        ngram_range=(3, 5),
        max_features=30000,
    )
    return {
        "word_logistic": GridSearchCV(
            Pipeline(
                [
                    ("tfidf", word),
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
            {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0]},
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "word_char_logistic": GridSearchCV(
            Pipeline(
                [
                    ("tfidf", FeatureUnion([("word", word), ("char", char)])),
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
            {"model__C": [0.1, 0.3, 1.0, 3.0]},
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "word_char_svm": GridSearchCV(
            Pipeline(
                [
                    ("tfidf", FeatureUnion([("word", word), ("char", char)])),
                    (
                        "model",
                        LinearSVC(
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.03, 0.1, 0.3, 1.0]},
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
    }


def _speech_search(cv):
    pipeline = Pipeline(
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
    return GridSearchCV(
        pipeline,
        {
            "pca__n_components": [8, 16, 32],
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )


def main():
    split = load_json(SPLIT_PATH)
    manifest = load_json(MANIFEST_PATH)["participants"]
    manifest_by_pid = {item["participant_id"]: item for item in manifest}
    sources, conflicts = discover_audio_sources(DATA_DIR, include_zips=True)

    rows = []
    for source in sources:
        pid = source["participant_id"]
        transcript = load_participant_transcript(source)
        fingerprint = manifest_by_pid[pid]["source_fingerprint"]
        cache_path = EMBEDDING_CACHE_DIR / f"{fingerprint}.npz"
        if not transcript or not cache_path.exists():
            print(f"Warning: skipping {pid}; transcript or embedding missing")
            continue
        speech = np.load(cache_path)["participant_vector"].astype(np.float32)
        rows.append(
            {
                "participant_id": pid,
                "label": int(source["label"]),
                "text": transcript,
                "speech": speech,
            }
        )

    train_ids = set(split["train_participant_ids"])
    test_ids = set(split["test_participant_ids"])
    train_rows = [row for row in rows if row["participant_id"] in train_ids]
    test_rows = [row for row in rows if row["participant_id"] in test_ids]
    train_texts = [row["text"] for row in train_rows]
    test_texts = [row["text"] for row in test_rows]
    train_speech = np.stack([row["speech"] for row in train_rows])
    test_speech = np.stack([row["speech"] for row in test_rows])
    y_train = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    y_test = np.asarray([row["label"] for row in test_rows], dtype=np.int64)

    print(
        f"Multimodal data: {len(rows)} participants | "
        f"{len(train_rows)} train / {len(test_rows)} test"
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_text_name = None
    best_text_search = None
    best_text_cv = -np.inf
    comparison = {}
    for name, search in _text_candidates(cv).items():
        print(f"Cross-validating {name}...")
        search.fit(train_texts, y_train)
        score = float(search.best_score_)
        comparison[name] = {
            "cv_balanced_accuracy": score,
            "best_params": search.best_params_,
        }
        print(f"  CV balanced accuracy={score:.3f}")
        if score > best_text_cv:
            best_text_name = name
            best_text_search = search
            best_text_cv = score

    print("Cross-validating frozen WavLM speech classifier...")
    speech_search = _speech_search(cv)
    speech_search.fit(train_speech, y_train)
    speech_cv = float(speech_search.best_score_)
    comparison["wavlm_speech"] = {
        "cv_balanced_accuracy": speech_cv,
        "best_params": speech_search.best_params_,
    }
    print(f"  CV balanced accuracy={speech_cv:.3f}")

    text_oof = _oof_probabilities(
        best_text_search.best_estimator_, train_texts, y_train, cv
    )
    speech_oof = _oof_probabilities(
        speech_search.best_estimator_, train_speech, y_train, cv
    )

    candidates = []
    for text_weight in np.linspace(0.0, 1.0, 11):
        probabilities = text_weight * text_oof + (1 - text_weight) * speech_oof
        threshold, score = select_threshold(y_train, probabilities)
        candidates.append((score, text_weight, threshold))
    blend_score, text_weight, threshold = max(candidates, key=lambda item: item[0])
    print(
        f"Selected blend from train OOF only: text={text_weight:.1f}, "
        f"speech={1-text_weight:.1f}, threshold={threshold:.2f}, "
        f"balanced accuracy={blend_score:.3f}"
    )

    text_model = clone(best_text_search.best_estimator_).fit(
        train_texts, y_train
    )
    speech_model = clone(speech_search.best_estimator_).fit(
        train_speech, y_train
    )
    train_prob = (
        text_weight * _probabilities(text_model, train_texts)
        + (1 - text_weight) * _probabilities(speech_model, train_speech)
    )
    test_prob = (
        text_weight * _probabilities(text_model, test_texts)
        + (1 - text_weight) * _probabilities(speech_model, test_speech)
    )

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

    # Refit deployment components on all participants after metrics are locked.
    all_texts = [row["text"] for row in rows]
    all_speech = np.stack([row["speech"] for row in rows])
    all_labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    deploy_text = clone(best_text_search.best_estimator_).fit(
        all_texts, all_labels
    )
    deploy_speech = clone(speech_search.best_estimator_).fit(
        all_speech, all_labels
    )
    artifact = {
        "model_type": "multimodal_wavlm_tfidf",
        "ssl_model_id": SSL_MODEL_ID,
        "text_model_name": best_text_name,
        "text_model": deploy_text,
        "speech_model": deploy_speech,
        "text_weight": float(text_weight),
        "speech_weight": float(1 - text_weight),
        "threshold": float(threshold),
        "held_out_test_metrics": test_metrics,
        "training_oof_balanced_accuracy": float(blend_score),
        "requires_transcript": True,
        "train_participant_ids": [
            row["participant_id"] for row in train_rows
        ],
        "test_participant_ids": [
            row["participant_id"] for row in test_rows
        ],
    }
    with open(MULTIMODAL_MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)

    metadata = {
        "n_participants": len(rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "text_model": best_text_name,
        "text_weight": float(text_weight),
        "speech_weight": float(1 - text_weight),
        "threshold": float(threshold),
        "training_oof_balanced_accuracy": float(blend_score),
        "train_metrics": train_metrics,
        "held_out_test_metrics": test_metrics,
        "candidate_comparison": comparison,
        "target_accuracy_range": [0.75, 0.85],
        "target_met": bool(0.75 <= test_metrics["accuracy"] <= 0.85),
        "excluded_label_conflicts": conflicts,
        "note": (
            "Text contains participant utterances only. Model and threshold "
            "selection used train participants only."
        ),
    }
    MULTIMODAL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved multimodal model to {MULTIMODAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
