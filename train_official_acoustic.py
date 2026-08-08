#!/usr/bin/env python3
"""Train deployable acoustic classifiers from raw participant speech."""

import json
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    DATA_DIR,
    MODEL_DIR,
    SSL_MAX_DURATION,
    SSL_MAX_SEGMENTS,
    SSL_SEGMENT_DURATION,
    SSL_SEGMENT_OVERLAP,
)
from src.data import discover_audio_sources, load_audio_source
from src.features import (
    extract_acoustic_features,
    features_to_vector,
    segment_audio,
)
from src.ssl_model import (
    bootstrap_accuracy_ci,
    evaluate_predictions,
    select_threshold,
)

MODEL_PATH = MODEL_DIR / "depression_acoustic_refreshed_candidate.pkl"
METADATA_PATH = MODEL_DIR / "acoustic_refreshed_candidate_metadata.json"


def aggregate(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    return np.concatenate(
        [
            values.mean(axis=0),
            values.std(axis=0),
            np.median(values, axis=0),
            np.percentile(values, 25, axis=0),
            np.percentile(values, 75, axis=0),
        ]
    ).astype(np.float32)


def searches(cv):
    scaled = lambda model: Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("model", model),
        ]
    )
    return {
        "logistic": GridSearchCV(
            scaled(
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=42,
                )
            ),
            {
                "pca__n_components": [8, 16, 32],
                "model__C": [0.01, 0.1, 1, 10],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "svm": GridSearchCV(
            scaled(
                SVC(
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                )
            ),
            {
                "pca__n_components": [8, 16, 32],
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", 0.01, 0.1],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "extra_trees": GridSearchCV(
            ExtraTreesClassifier(
                n_estimators=700,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "random_forest": GridSearchCV(
            RandomForestClassifier(
                n_estimators=700,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
    }


def main():
    sources, conflicts = discover_audio_sources(
        DATA_DIR, include_zips=True
    )
    rows = []
    for number, source in enumerate(sources, 1):
        pid = source["participant_id"]
        print(f"[{number}/{len(sources)}] extracting acoustic {pid}")
        audio = load_audio_source(
            source,
            max_duration=SSL_MAX_DURATION,
            participant_only=True,
        )
        segments = segment_audio(
            audio, SSL_SEGMENT_DURATION, SSL_SEGMENT_OVERLAP
        )
        if len(segments) > SSL_MAX_SEGMENTS:
            indices = np.linspace(
                0, len(segments) - 1, SSL_MAX_SEGMENTS, dtype=int
            )
            segments = [segments[index] for index in indices]
        vectors = [
            features_to_vector(extract_acoustic_features(segment))
            for segment in segments
        ]
        rows.append(
            {
                "participant_id": pid,
                "label": int(source["label"]),
                "split": source["official_split"],
                "vector": aggregate(vectors),
            }
        )

    development = [
        row for row in rows if row["split"] in {"train", "dev"}
    ]
    test = [row for row in rows if row["split"] == "test"]
    X_train = np.stack([row["vector"] for row in development])
    y_train = np.asarray([row["label"] for row in development])
    X_test = np.stack([row["vector"] for row in test])
    y_test = np.asarray([row["label"] for row in test])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_name, best_search, best_cv = None, None, -np.inf
    comparison = {}
    for name, search in searches(cv).items():
        print(f"Cross-validating {name}...")
        search.fit(X_train, y_train)
        score = float(search.best_score_)
        comparison[name] = {
            "cv_balanced_accuracy": score,
            "best_params": search.best_params_,
        }
        print(f"  CV balanced accuracy={score:.3f}")
        if score > best_cv:
            best_name, best_search, best_cv = name, search, score

    oof = np.zeros(len(y_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        model = clone(best_search.best_estimator_).fit(
            X_train[train_idx], y_train[train_idx]
        )
        oof[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]
    threshold, oof_score = select_threshold(y_train, oof)
    selected = clone(best_search.best_estimator_).fit(X_train, y_train)
    train_prob = selected.predict_proba(X_train)[:, 1]
    test_prob = selected.predict_proba(X_test)[:, 1]
    train_metrics = evaluate_predictions(
        y_train, train_prob, threshold
    )
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

    X_all = np.stack([row["vector"] for row in rows])
    y_all = np.asarray([row["label"] for row in rows])
    deployment = clone(best_search.best_estimator_).fit(X_all, y_all)
    artifact = {
        "model_type": "raw_acoustic_official_phq",
        "selected_model": best_name,
        "model": deployment,
        "threshold": float(threshold),
        "aggregation": ["mean", "std", "median", "p25", "p75"],
        "segment_duration": SSL_SEGMENT_DURATION,
        "segment_overlap": SSL_SEGMENT_OVERLAP,
        "max_duration": SSL_MAX_DURATION,
        "max_segments": SSL_MAX_SEGMENTS,
        "training_cv_balanced_accuracy": best_cv,
        "training_oof_balanced_accuracy": oof_score,
        "held_out_test_metrics": test_metrics,
        "official_labels": True,
    }
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "n_participants": len(rows),
                "selected_model": best_name,
                "threshold": float(threshold),
                "training_cv_balanced_accuracy": best_cv,
                "training_oof_balanced_accuracy": oof_score,
                "train_metrics": train_metrics,
                "held_out_test_metrics": test_metrics,
                "candidate_comparison": comparison,
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved raw-acoustic candidate to {MODEL_PATH}")


if __name__ == "__main__":
    main()
