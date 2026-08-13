#!/usr/bin/env python3
"""Blend binary depression and PHQ severity models with recall constraints."""

import json
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import (
    DATA_DIR,
    ENGINEERED_CACHE_DIR,
    MANIFEST_PATH,
    MODEL_DIR,
)
from src.data import discover_audio_sources, load_json
from src.ssl_model import (
    bootstrap_accuracy_ci,
    evaluate_predictions,
)

MODEL_PATH = MODEL_DIR / "depression_hybrid_candidate.pkl"
METADATA_PATH = MODEL_DIR / "hybrid_candidate_metadata.json"
MINIMUM_OOF_RECALL = 0.65


def _binary_model():
    return ExtraTreesClassifier(
        n_estimators=700,
        max_depth=6,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _severity_model():
    return ExtraTreesRegressor(
        n_estimators=800,
        max_depth=8,
        max_features="sqrt",
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )


def _severity_probability(model, features):
    return np.clip(model.predict(features) / 24.0, 0.0, 1.0)


def _choose_blend(y_true, binary_probability, severity_probability):
    choices = []
    for binary_weight in np.linspace(0.0, 1.0, 21):
        probability = (
            binary_weight * binary_probability
            + (1.0 - binary_weight) * severity_probability
        )
        for threshold in np.linspace(0.2, 0.7, 101):
            predictions = (probability >= threshold).astype(int)
            recall = recall_score(
                y_true, predictions, zero_division=0
            )
            if recall < MINIMUM_OOF_RECALL:
                continue
            choices.append(
                {
                    "binary_weight": float(binary_weight),
                    "threshold": float(threshold),
                    "accuracy": float(
                        accuracy_score(y_true, predictions)
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            y_true, predictions
                        )
                    ),
                    "recall": float(recall),
                }
            )
    if not choices:
        raise RuntimeError("No blend satisfies the OOF recall constraint")
    return max(
        choices,
        key=lambda item: (
            item["accuracy"],
            item["balanced_accuracy"],
            item["recall"],
        ),
    )


def main():
    manifest = {
        item["participant_id"]: item
        for item in load_json(MANIFEST_PATH)["participants"]
    }
    sources, conflicts = discover_audio_sources(
        DATA_DIR, include_zips=True
    )
    rows = []
    for source in sources:
        participant_id = source["participant_id"]
        score = source.get("phq_score")
        if score is None or participant_id not in manifest:
            continue
        fingerprint = manifest[participant_id]["source_fingerprint"]
        cache_path = ENGINEERED_CACHE_DIR / f"{fingerprint}.npz"
        if not cache_path.exists():
            continue
        cached = np.load(cache_path)
        gender = source.get("gender")
        gender_vector = np.asarray(
            [
                float(gender == 0),
                float(gender == 1),
                float(gender is None),
            ],
            dtype=np.float32,
        )
        rows.append(
            {
                "participant_id": participant_id,
                "label": int(source["label"]),
                "score": float(score),
                "split": source["official_split"],
                "raw": cached["raw_acoustic"].astype(np.float32),
                "severity": np.concatenate(
                    [
                        cached["engineered"].astype(np.float32),
                        gender_vector,
                    ]
                ),
            }
        )

    development = [
        row for row in rows if row["split"] in {"train", "dev"}
    ]
    test = [row for row in rows if row["split"] == "test"]
    raw_train = np.stack([row["raw"] for row in development])
    severity_train = np.stack(
        [row["severity"] for row in development]
    )
    labels_train = np.asarray(
        [row["label"] for row in development]
    )
    scores_train = np.asarray(
        [row["score"] for row in development]
    )
    raw_test = np.stack([row["raw"] for row in test])
    severity_test = np.stack([row["severity"] for row in test])
    labels_test = np.asarray([row["label"] for row in test])

    binary_oof = np.zeros(len(development))
    severity_oof = np.zeros(len(development))
    folds = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    for train_index, validation_index in folds.split(
        raw_train, labels_train
    ):
        binary = clone(_binary_model()).fit(
            raw_train[train_index], labels_train[train_index]
        )
        severity = clone(_severity_model()).fit(
            severity_train[train_index], scores_train[train_index]
        )
        binary_oof[validation_index] = binary.predict_proba(
            raw_train[validation_index]
        )[:, 1]
        severity_oof[validation_index] = _severity_probability(
            severity, severity_train[validation_index]
        )

    selection = _choose_blend(
        labels_train, binary_oof, severity_oof
    )
    binary_weight = selection["binary_weight"]
    severity_weight = 1.0 - binary_weight
    threshold = selection["threshold"]
    print("OOF selection:", selection)

    binary = _binary_model().fit(raw_train, labels_train)
    severity = _severity_model().fit(severity_train, scores_train)
    train_probability = (
        binary_weight * binary.predict_proba(raw_train)[:, 1]
        + severity_weight
        * _severity_probability(severity, severity_train)
    )
    test_probability = (
        binary_weight * binary.predict_proba(raw_test)[:, 1]
        + severity_weight
        * _severity_probability(severity, severity_test)
    )
    train_metrics = evaluate_predictions(
        labels_train, train_probability, threshold
    )
    test_metrics = evaluate_predictions(
        labels_test, test_probability, threshold
    )
    test_metrics["accuracy_bootstrap_ci95"] = bootstrap_accuracy_ci(
        labels_test, test_probability, threshold
    )
    print(
        "HELD-OUT participant test:",
        {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in test_metrics.items()
            if key not in {"confusion_matrix", "accuracy_bootstrap_ci95"}
        },
    )

    target_met = (
        test_metrics["accuracy"] >= 0.688
        and test_metrics["recall"] >= 0.636
    )
    artifact = {
        "model_type": "hybrid_binary_phq_official",
        "binary_model": binary,
        "severity_model": severity,
        "binary_weight": binary_weight,
        "severity_weight": severity_weight,
        "threshold": threshold,
        "minimum_oof_recall": MINIMUM_OOF_RECALL,
        "held_out_test_metrics": test_metrics,
        "official_labels": True,
    }
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "n_participants": len(rows),
                "selection": selection,
                "train_metrics": train_metrics,
                "held_out_test_metrics": test_metrics,
                "target_accuracy": 0.688,
                "target_recall": 0.636,
                "target_met": target_met,
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved hybrid candidate to {MODEL_PATH}")
    print(f"TARGET_MET={target_met}")


if __name__ == "__main__":
    main()
