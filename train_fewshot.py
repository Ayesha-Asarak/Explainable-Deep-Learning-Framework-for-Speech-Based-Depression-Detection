#!/usr/bin/env python3
"""Evaluate few-shot prototype models without replacing a stronger model."""

import json
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_DIR,
    EMBEDDING_CACHE_DIR,
    ENGINEERED_CACHE_DIR,
    MANIFEST_PATH,
    MODEL_DIR,
)
from src.data import discover_audio_sources, load_json
from src.fewshot import PrototypeClassifier
from src.ssl_model import (
    bootstrap_accuracy_ci,
    evaluate_predictions,
    select_threshold,
)

CANDIDATE_PATH = MODEL_DIR / "depression_fewshot_candidate.pkl"
METADATA_PATH = MODEL_DIR / "fewshot_candidate_metadata.json"


def _oof_probabilities(estimator, X, y, cv):
    probabilities = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        model = clone(estimator).fit(X[train_idx], y[train_idx])
        probabilities[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return probabilities


def _search(cv, dimensions):
    components = sorted(
        set(min(value, dimensions) for value in (4, 8, 16, 32))
    )
    return GridSearchCV(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(random_state=42)),
                ("prototype", PrototypeClassifier()),
            ]
        ),
        {
            "pca__n_components": components,
            "prototype__metric": ["cosine", "euclidean"],
            "prototype__temperature": [0.1, 0.3, 1.0, 3.0],
            "prototype__shrinkage": [0.0, 0.1, 0.25],
        },
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
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
        pid = source["participant_id"]
        fingerprint = manifest[pid]["source_fingerprint"]
        wavlm_path = EMBEDDING_CACHE_DIR / f"{fingerprint}.npz"
        engineered_path = ENGINEERED_CACHE_DIR / f"{fingerprint}.npz"
        if not wavlm_path.exists() or not engineered_path.exists():
            print(f"Warning: missing cached features for {pid}")
            continue
        wavlm = np.load(wavlm_path)["participant_vector"].astype(
            np.float32
        )
        engineered_cache = np.load(engineered_path)
        engineered = engineered_cache["engineered"].astype(np.float32)
        raw = engineered_cache["raw_acoustic"].astype(np.float32)
        rows.append(
            {
                "participant_id": pid,
                "label": int(source["label"]),
                "split": source["official_split"],
                "wavlm": wavlm,
                "engineered": engineered,
                "raw": raw,
            }
        )

    development = [
        row for row in rows if row["split"] in {"train", "dev"}
    ]
    test = [row for row in rows if row["split"] == "test"]
    y_train = np.asarray([row["label"] for row in development])
    y_test = np.asarray([row["label"] for row in test])
    families = {
        "raw_acoustic": (
            np.stack([row["raw"] for row in development]),
            np.stack([row["raw"] for row in test]),
        ),
        "egemaps_temporal": (
            np.stack([row["engineered"] for row in development]),
            np.stack([row["engineered"] for row in test]),
        ),
        "wavlm": (
            np.stack([row["wavlm"] for row in development]),
            np.stack([row["wavlm"] for row in test]),
        ),
        "raw_egemaps": (
            np.stack(
                [
                    np.concatenate([row["raw"], row["engineered"]])
                    for row in development
                ]
            ),
            np.stack(
                [
                    np.concatenate([row["raw"], row["engineered"]])
                    for row in test
                ]
            ),
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = []
    comparison = {}
    for family, (X_train, X_test) in families.items():
        print(f"Cross-validating few-shot prototypes: {family}...")
        search = _search(cv, X_train.shape[1])
        search.fit(X_train, y_train)
        score = float(search.best_score_)
        comparison[family] = {
            "cv_balanced_accuracy": score,
            "best_params": search.best_params_,
        }
        candidates.append(
            (
                score,
                family,
                search.best_estimator_,
                X_train,
                X_test,
            )
        )
        print(f"  CV balanced accuracy={score:.3f}")

    score, family, estimator, X_train, X_test = max(
        candidates, key=lambda item: item[0]
    )
    oof = _oof_probabilities(estimator, X_train, y_train, cv)
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

    artifact = {
        "model_type": "prototype_fewshot_official_phq",
        "feature_family": family,
        "selected_model": "prototype_classifier",
        "model": selected,
        "threshold": float(threshold),
        "training_cv_balanced_accuracy": float(score),
        "training_oof_balanced_accuracy": float(oof_score),
        "held_out_test_metrics": test_metrics,
        "official_labels": True,
    }
    with open(CANDIDATE_PATH, "wb") as handle:
        pickle.dump(artifact, handle)

    current = {
        "accuracy": 0.59375,
        "balanced_accuracy": 0.6038961038961039,
        "recall": 0.6363636363636364,
        "f1": 0.5185185185185185,
    }
    deploy_eligible = (
        test_metrics["accuracy"] > current["accuracy"]
        and test_metrics["balanced_accuracy"]
        >= current["balanced_accuracy"]
        and test_metrics["recall"] >= current["recall"]
        and test_metrics["f1"] >= current["f1"]
    )
    metadata = {
        "n_participants": len(rows),
        "feature_family": family,
        "selected_model": "prototype_classifier",
        "threshold": float(threshold),
        "training_cv_balanced_accuracy": float(score),
        "training_oof_balanced_accuracy": float(oof_score),
        "train_metrics": train_metrics,
        "held_out_test_metrics": test_metrics,
        "current_deployed_metrics": current,
        "deploy_eligible": deploy_eligible,
        "candidate_comparison": comparison,
        "excluded_label_conflicts": conflicts,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved few-shot candidate to {CANDIDATE_PATH}")
    print(f"DEPLOY_ELIGIBLE={deploy_eligible}")


if __name__ == "__main__":
    main()
