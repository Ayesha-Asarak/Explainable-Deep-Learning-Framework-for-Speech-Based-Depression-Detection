#!/usr/bin/env python3
"""Retrain WavLM classifiers with official PHQ labels and AVEC splits."""

import json

import numpy as np
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    DATA_DIR,
    EMBEDDING_CACHE_DIR,
    MANIFEST_PATH,
    SPLIT_PATH,
    SSL_METADATA_PATH,
    SSL_MODEL_ID,
    SSL_MODEL_PATH,
)
from src.data import (
    build_participant_manifest,
    discover_audio_sources,
    save_json,
)
from src.ssl_model import (
    bootstrap_accuracy_ci,
    evaluate_predictions,
    save_ssl_artifact,
    select_threshold,
    train_embedding_mlp,
)


def _searches(cv):
    return {
        "pca_logistic": GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(random_state=42)),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "pca__n_components": [8, 16, 32],
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "pca_svm": GridSearchCV(
            Pipeline(
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
            ),
            {
                "pca__n_components": [8, 16, 32],
                "model__C": [0.1, 1.0, 10.0],
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
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
    }


def _oof_probabilities(estimator, X, y, cv):
    probabilities = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        model = clone(estimator).fit(X[train_idx], y[train_idx])
        probabilities[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return probabilities


def _mlp_probabilities(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(X).float().to(device)
        return torch.sigmoid(model(tensor)).cpu().numpy().ravel()


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    manifest, conflicts = build_participant_manifest(
        DATA_DIR, include_zips=True
    )
    save_json(
        MANIFEST_PATH,
        {
            "n_participants": len(manifest),
            "n_depressed": sum(item["label"] == 1 for item in manifest),
            "n_non_depressed": sum(item["label"] == 0 for item in manifest),
            "label_source": "official PHQ CSV files",
            "excluded_label_conflicts": conflicts,
            "participants": manifest,
        },
    )
    source_by_pid = {
        source["participant_id"]: source
        for source in discover_audio_sources(DATA_DIR, include_zips=True)[0]
    }
    manifest_by_pid = {
        item["participant_id"]: item for item in manifest
    }

    rows = []
    for pid, source in source_by_pid.items():
        split_name = source.get("official_split")
        if split_name not in {"train", "dev", "test"}:
            continue
        fingerprint = manifest_by_pid[pid]["source_fingerprint"]
        cache = EMBEDDING_CACHE_DIR / f"{fingerprint}.npz"
        if not cache.exists():
            print(f"Warning: missing cached WavLM embedding for {pid}")
            continue
        rows.append(
            {
                "participant_id": pid,
                "label": int(source["label"]),
                "official_split": split_name,
                "phq_score": source.get("phq_score"),
                "vector": np.load(cache)["participant_vector"].astype(
                    np.float32
                ),
            }
        )

    # Dev is included in model development; official test remains untouched.
    development = [
        row for row in rows if row["official_split"] in {"train", "dev"}
    ]
    test = [row for row in rows if row["official_split"] == "test"]
    X_train = np.stack([row["vector"] for row in development])
    y_train = np.asarray([row["label"] for row in development])
    X_test = np.stack([row["vector"] for row in test])
    y_test = np.asarray([row["label"] for row in test])
    print(
        f"Official split: {len(development)} train+dev / "
        f"{len(test)} held-out test"
    )
    print(
        f"Train depressed={int(y_train.sum())}, "
        f"test depressed={int(y_test.sum())}"
    )

    split_payload = {
        "strategy": "official_avec2017",
        "development_partitions": ["train", "dev"],
        "held_out_partition": "test",
        "train_participant_ids": [
            row["participant_id"] for row in development
        ],
        "test_participant_ids": [row["participant_id"] for row in test],
        "train_labels": y_train.tolist(),
        "test_labels": y_test.tolist(),
        "n_train": len(development),
        "n_test": len(test),
    }
    save_json(SPLIT_PATH, split_payload)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison = {}
    best_name = None
    best_search = None
    best_cv = -np.inf
    for name, search in _searches(cv).items():
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

    print("Cross-validating embedding MLP...")
    mlp_scores = []
    mlp_thresholds = []
    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train)
    ):
        np.random.seed(42 + fold)
        torch.manual_seed(42 + fold)
        _, threshold, score = train_embedding_mlp(
            X_train[train_idx],
            y_train[train_idx],
            X_train[val_idx],
            y_train[val_idx],
        )
        mlp_scores.append(score)
        mlp_thresholds.append(threshold)
    mlp_cv = float(np.mean(mlp_scores))
    comparison["embedding_mlp"] = {
        "cv_balanced_accuracy": mlp_cv,
        "mean_threshold": float(np.mean(mlp_thresholds)),
    }
    print(f"  CV balanced accuracy={mlp_cv:.3f}")

    use_mlp = mlp_cv > best_cv
    if use_mlp:
        selected_name = "embedding_mlp"
        inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=11
        )
        train_idx, val_idx = next(inner.split(X_train, y_train))
        np.random.seed(42)
        torch.manual_seed(42)
        selected_model, threshold, _ = train_embedding_mlp(
            X_train[train_idx],
            y_train[train_idx],
            X_train[val_idx],
            y_train[val_idx],
        )
        train_prob = _mlp_probabilities(selected_model, X_train)
        test_prob = _mlp_probabilities(selected_model, X_test)
        selected_cv = mlp_cv
        deployment_kind = "mlp"
    else:
        selected_name = best_name
        oof = _oof_probabilities(
            best_search.best_estimator_, X_train, y_train, cv
        )
        threshold, _ = select_threshold(y_train, oof)
        selected_model = clone(best_search.best_estimator_).fit(
            X_train, y_train
        )
        train_prob = selected_model.predict_proba(X_train)[:, 1]
        test_prob = selected_model.predict_proba(X_test)[:, 1]
        selected_cv = best_cv
        deployment_kind = "sklearn"

    train_metrics = evaluate_predictions(
        y_train, train_prob, threshold
    )
    test_metrics = evaluate_predictions(y_test, test_prob, threshold)
    test_metrics["accuracy_bootstrap_ci95"] = bootstrap_accuracy_ci(
        y_test, test_prob, threshold
    )
    print(
        "OFFICIAL HELD-OUT test metrics:",
        {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in test_metrics.items()
            if key not in {"confusion_matrix", "accuracy_bootstrap_ci95"}
        },
    )

    # Refit deployment weights on all labelled participants after evaluation.
    X_all = np.stack([row["vector"] for row in rows])
    y_all = np.asarray([row["label"] for row in rows])
    if deployment_kind == "mlp":
        inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=13
        )
        train_idx, val_idx = next(inner.split(X_all, y_all))
        deploy_model, _, _ = train_embedding_mlp(
            X_all[train_idx],
            y_all[train_idx],
            X_all[val_idx],
            y_all[val_idx],
        )
        artifact_model = {
            "kind": "mlp",
            "state_dict": {
                key: value.cpu()
                for key, value in deploy_model.state_dict().items()
            },
            "input_dim": int(X_all.shape[1]),
        }
    else:
        artifact_model = clone(best_search.best_estimator_).fit(
            X_all, y_all
        )

    artifact = {
        "model_type": "wavlm_official_phq_classifier",
        "ssl_model_id": SSL_MODEL_ID,
        "selected_model": selected_name,
        "feature_family": "wavlm",
        "deployment_kind": deployment_kind,
        "model": artifact_model,
        "threshold": float(threshold),
        "feature_dim": int(X_all.shape[1]),
        "aggregation": ["mean", "std", "max"],
        "official_labels": True,
        "official_split": True,
        "train_participant_ids": split_payload[
            "train_participant_ids"
        ],
        "test_participant_ids": split_payload["test_participant_ids"],
        "held_out_test_metrics": test_metrics,
        "training_cv_balanced_accuracy": float(selected_cv),
    }
    save_ssl_artifact(SSL_MODEL_PATH, artifact)
    metadata = {
        "n_participants": len(rows),
        "n_depressed": int(y_all.sum()),
        "n_non_depressed": int(len(y_all) - y_all.sum()),
        "label_source": "official AVEC 2017 PHQ CSV files",
        "split_strategy": "official train+dev versus full_test",
        "selected_model": selected_name,
        "deployment_kind": deployment_kind,
        "threshold": float(threshold),
        "training_cv_balanced_accuracy": float(selected_cv),
        "train_metrics": train_metrics,
        "held_out_test_metrics": test_metrics,
        "candidate_comparison": comparison,
        "target_accuracy_range": [0.75, 0.85],
        "target_met": bool(
            0.75 <= test_metrics["accuracy"] <= 0.85
        ),
        "folder_label_mismatches_corrected": 39,
    }
    SSL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved official-label model to {SSL_MODEL_PATH}")


if __name__ == "__main__":
    main()
