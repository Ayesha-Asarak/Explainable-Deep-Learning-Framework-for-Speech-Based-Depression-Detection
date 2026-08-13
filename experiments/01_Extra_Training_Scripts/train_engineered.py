#!/usr/bin/env python3
"""Evaluate eGeMAPS, temporal prosody, and raw acoustic feature fusion."""

import json
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.config import (
    DATA_DIR,
    ENGINEERED_CACHE_DIR,
    ENGINEERED_MODEL_PATH,
    MANIFEST_PATH,
    MODEL_DIR,
    SSL_MAX_DURATION,
    SSL_MAX_SEGMENTS,
    SSL_SEGMENT_DURATION,
    SSL_SEGMENT_OVERLAP,
)
from src.data import discover_audio_sources, load_audio_source, load_json
from src.engineered_features import extract_engineered_features
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
from train_official_acoustic import aggregate, searches

METADATA_PATH = MODEL_DIR / "engineered_candidate_metadata.json"


def main():
    manifest = {
        item["participant_id"]: item
        for item in load_json(MANIFEST_PATH)["participants"]
    }
    sources, conflicts = discover_audio_sources(
        DATA_DIR, include_zips=True
    )
    rows = []
    for number, source in enumerate(sources, 1):
        pid = source["participant_id"]
        fingerprint = manifest[pid]["source_fingerprint"]
        cache_path = ENGINEERED_CACHE_DIR / f"{fingerprint}.npz"
        print(f"[{number}/{len(sources)}] engineered features {pid}")
        if cache_path.exists():
            cached = np.load(cache_path)
            engineered = cached["engineered"]
            raw_acoustic = cached["raw_acoustic"]
        else:
            audio = load_audio_source(
                source,
                max_duration=SSL_MAX_DURATION,
                participant_only=True,
            )
            engineered = extract_engineered_features(audio)
            segments = segment_audio(
                audio,
                SSL_SEGMENT_DURATION,
                SSL_SEGMENT_OVERLAP,
            )
            if len(segments) > SSL_MAX_SEGMENTS:
                indices = np.linspace(
                    0, len(segments) - 1, SSL_MAX_SEGMENTS, dtype=int
                )
                segments = [segments[index] for index in indices]
            raw_acoustic = aggregate(
                [
                    features_to_vector(
                        extract_acoustic_features(segment)
                    )
                    for segment in segments
                ]
            )
            np.savez_compressed(
                cache_path,
                engineered=engineered,
                raw_acoustic=raw_acoustic,
            )
        rows.append(
            {
                "participant_id": pid,
                "label": int(source["label"]),
                "split": source["official_split"],
                "engineered": engineered.astype(np.float32),
                "raw_acoustic": raw_acoustic.astype(np.float32),
            }
        )

    development = [
        row for row in rows if row["split"] in {"train", "dev"}
    ]
    test = [row for row in rows if row["split"] == "test"]
    y_train = np.asarray([row["label"] for row in development])
    y_test = np.asarray([row["label"] for row in test])
    families = {
        "egemaps_temporal": (
            np.stack([row["engineered"] for row in development]),
            np.stack([row["engineered"] for row in test]),
        ),
        "raw_acoustic": (
            np.stack([row["raw_acoustic"] for row in development]),
            np.stack([row["raw_acoustic"] for row in test]),
        ),
        "engineered_raw_fusion": (
            np.stack(
                [
                    np.concatenate(
                        [row["engineered"], row["raw_acoustic"]]
                    )
                    for row in development
                ]
            ),
            np.stack(
                [
                    np.concatenate(
                        [row["engineered"], row["raw_acoustic"]]
                    )
                    for row in test
                ]
            ),
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = []
    comparison = {}
    for family, (X_train, X_test) in families.items():
        for model_name, search in searches(cv).items():
            name = f"{family}:{model_name}"
            print(f"Cross-validating {name}...")
            search.fit(X_train, y_train)
            score = float(search.best_score_)
            comparison[name] = {
                "cv_balanced_accuracy": score,
                "best_params": search.best_params_,
            }
            candidates.append(
                (
                    score,
                    name,
                    family,
                    search.best_estimator_,
                    X_train,
                    X_test,
                )
            )
            print(f"  CV balanced accuracy={score:.3f}")

    score, name, family, estimator, X_train, X_test = max(
        candidates, key=lambda item: item[0]
    )
    oof = np.zeros(len(y_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        model = clone(estimator).fit(
            X_train[train_idx], y_train[train_idx]
        )
        oof[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]
    threshold, oof_score = select_threshold(y_train, oof)
    selected = clone(estimator).fit(X_train, y_train)
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

    if family == "egemaps_temporal":
        X_all = np.stack([row["engineered"] for row in rows])
    elif family == "raw_acoustic":
        X_all = np.stack([row["raw_acoustic"] for row in rows])
    else:
        X_all = np.stack(
            [
                np.concatenate(
                    [row["engineered"], row["raw_acoustic"]]
                )
                for row in rows
            ]
        )
    y_all = np.asarray([row["label"] for row in rows])
    deployment = clone(estimator).fit(X_all, y_all)
    artifact = {
        "model_type": "engineered_acoustic_official_phq",
        "feature_family": family,
        "selected_model": name,
        "model": deployment,
        "threshold": float(threshold),
        "training_cv_balanced_accuracy": float(score),
        "training_oof_balanced_accuracy": float(oof_score),
        "held_out_test_metrics": test_metrics,
        "official_labels": True,
        "max_duration": SSL_MAX_DURATION,
        "max_segments": SSL_MAX_SEGMENTS,
    }
    with open(ENGINEERED_MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "n_participants": len(rows),
                "feature_family": family,
                "selected_model": name,
                "threshold": float(threshold),
                "training_cv_balanced_accuracy": float(score),
                "training_oof_balanced_accuracy": float(oof_score),
                "train_metrics": train_metrics,
                "held_out_test_metrics": test_metrics,
                "candidate_comparison": comparison,
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved engineered candidate to {ENGINEERED_MODEL_PATH}")


if __name__ == "__main__":
    main()
