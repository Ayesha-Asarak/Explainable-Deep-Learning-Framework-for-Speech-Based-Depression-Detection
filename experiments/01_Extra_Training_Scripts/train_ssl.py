#!/usr/bin/env python3
"""Leakage-safe WavLM training for participant-level depression detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.config import (
    DATA_DIR,
    MANIFEST_PATH,
    MODEL_DIR,
    SPLIT_PATH,
    SSL_MAX_DURATION,
    SSL_MAX_SEGMENTS,
    SSL_METADATA_PATH,
    SSL_MODEL_ID,
    SSL_MODEL_PATH,
    SSL_SEGMENT_DURATION,
    SSL_SEGMENT_OVERLAP,
)
from src.data import (
    build_participant_manifest,
    discover_audio_sources,
    fixed_participant_split,
    save_json,
)
from src.ssl_model import (
    WavLMEmbedder,
    bootstrap_accuracy_ci,
    candidate_searches,
    evaluate_predictions,
    extract_participant_embeddings,
    save_ssl_artifact,
    select_threshold,
    train_embedding_mlp,
)


def attach_fingerprints(sources: list[dict], manifest: list[dict]) -> list[dict]:
    by_pid = {entry["participant_id"]: entry for entry in manifest}
    enriched = []
    for source in sources:
        entry = by_pid[source["participant_id"]]
        item = dict(source)
        item["source_fingerprint"] = entry["source_fingerprint"]
        enriched.append(item)
    return enriched


def build_embedding_matrix(records: list[dict]):
    X = np.stack([r["participant_vector"] for r in records]).astype(np.float32)
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    pids = [r["participant_id"] for r in records]
    return X, y, pids


def index_by_pid(pids, selected):
    selected = set(selected)
    return [i for i, pid in enumerate(pids) if pid in selected]


def main():
    np.random.seed(42)
    MODEL_DIR.mkdir(exist_ok=True)

    print("Building participant manifest...")
    manifest, conflicts = build_participant_manifest(DATA_DIR, include_zips=True)
    save_json(MANIFEST_PATH, {
        "n_participants": len(manifest),
        "n_depressed": sum(1 for m in manifest if m["label"] == 1),
        "n_non_depressed": sum(1 for m in manifest if m["label"] == 0),
        "excluded_label_conflicts": conflicts,
        "participants": manifest,
    })
    print(f"Manifest: {len(manifest)} participants -> {MANIFEST_PATH}")

    split = fixed_participant_split(manifest, test_ratio=0.22, seed=42)
    save_json(SPLIT_PATH, split)
    print(
        f"Fixed split: {split['n_train']} train / {split['n_test']} test "
        f"(seed={split['seed']})"
    )
    print("Held-out test participants are locked and will not be used for selection.")

    sources, _ = discover_audio_sources(DATA_DIR, include_zips=True)
    sources = attach_fingerprints(sources, manifest)

    print(f"Loading frozen WavLM encoder: {SSL_MODEL_ID}")
    embedder = WavLMEmbedder(SSL_MODEL_ID)

    records = []
    for number, source in enumerate(sources, start=1):
        pid = source["participant_id"]
        print(
            f"[{number}/{len(sources)}] embedding "
            f"{'depressed' if source['label'] else 'non-depressed'} {pid}"
        )
        try:
            record = extract_participant_embeddings(
                source,
                embedder,
                max_duration=SSL_MAX_DURATION,
                segment_duration=SSL_SEGMENT_DURATION,
                overlap=SSL_SEGMENT_OVERLAP,
                max_segments=SSL_MAX_SEGMENTS,
                use_cache=True,
            )
            records.append(record)
        except Exception as exc:
            print(f"Warning: skipping {pid}: {exc}")

    X, y, pids = build_embedding_matrix(records)
    train_idx = index_by_pid(pids, split["train_participant_ids"])
    test_idx = index_by_pid(pids, split["test_participant_ids"])
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    train_pids = [pids[i] for i in train_idx]
    test_pids = [pids[i] for i in test_idx]
    print(
        f"Embedding matrix: {X.shape[0]} participants x {X.shape[1]} features | "
        f"{len(train_idx)} train / {len(test_idx)} test"
    )

    # ---- Model selection only on train participants ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    searches = candidate_searches(cv)
    comparison = {}
    best_name = None
    best_search = None
    best_cv = -np.inf

    for name, search in searches.items():
        print(f"Cross-validating {name} on train participants only...")
        search.fit(X_train, y_train)
        cv_score = float(search.best_score_)
        comparison[name] = {
            "cv_balanced_accuracy": cv_score,
            "best_params": search.best_params_,
        }
        print(f"  CV balanced accuracy={cv_score:.3f} params={search.best_params_}")
        if cv_score > best_cv:
            best_name, best_search, best_cv = name, search, cv_score

    # Threshold selected via out-of-fold predictions on train set only.
    oof_probs = np.zeros(len(y_train), dtype=np.float64)
    for fold_train, fold_val in cv.split(X_train, y_train):
        estimator = clone(best_search.best_estimator_)
        estimator.fit(X_train[fold_train], y_train[fold_train])
        oof_probs[fold_val] = estimator.predict_proba(X_train[fold_val])[:, 1]

    threshold, threshold_score = select_threshold(y_train, oof_probs)
    print(
        f"Selected sklearn model={best_name} "
        f"CV={best_cv:.3f} threshold={threshold:.3f} "
        f"(OOF balanced accuracy={threshold_score:.3f})"
    )

    # Optional MLP on the same frozen embeddings (inner holdout from train).
    print("Training embedding MLP with early stopping on train-only holdout...")
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    mlp_scores = []
    mlp_thresholds = []
    for fold_train, fold_val in inner.split(X_train, y_train):
        _, thr, score = train_embedding_mlp(
            X_train[fold_train],
            y_train[fold_train],
            X_train[fold_val],
            y_train[fold_val],
        )
        mlp_scores.append(score)
        mlp_thresholds.append(thr)
    mlp_cv = float(np.mean(mlp_scores))
    mlp_threshold = float(np.mean(mlp_thresholds))
    comparison["embedding_mlp"] = {
        "cv_balanced_accuracy": mlp_cv,
        "mean_threshold": mlp_threshold,
    }
    print(f"  MLP CV balanced accuracy={mlp_cv:.3f}")

    use_mlp = mlp_cv > best_cv
    if use_mlp:
        print("MLP selected over sklearn models based on train CV.")
        # Fit MLP using last train fold split for early stopping, then refit-like
        # by training with an internal validation split again.
        val_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
        tr, va = next(val_split.split(X_train, y_train))
        mlp_model, mlp_threshold, _ = train_embedding_mlp(
            X_train[tr], y_train[tr], X_train[va], y_train[va]
        )
        selected_model_name = "embedding_mlp"
        selected_threshold = mlp_threshold
        deployment_kind = "mlp"

        # For deployment, retrain MLP with early-stopping against a tiny train holdout.
        # Held-out test remains untouched.
        def predict_proba_matrix(matrix):
            import torch

            mlp_model.eval()
            with torch.no_grad():
                x = torch.from_numpy(matrix).float().to(next(mlp_model.parameters()).device)
                return torch.sigmoid(mlp_model(x)).cpu().numpy().ravel()

        train_probs = predict_proba_matrix(X_train)
        test_probs = predict_proba_matrix(X_test)
        selected_estimator = {
            "kind": "mlp",
            "state_dict": {k: v.cpu() for k, v in mlp_model.state_dict().items()},
            "input_dim": int(X_train.shape[1]),
        }
    else:
        selected_model_name = best_name
        selected_threshold = threshold
        deployment_kind = "sklearn"
        selected_estimator = best_search.best_estimator_
        # Refit selected sklearn estimator on all train participants.
        selected_estimator.fit(X_train, y_train)
        train_probs = selected_estimator.predict_proba(X_train)[:, 1]
        test_probs = selected_estimator.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_predictions(y_train, train_probs, selected_threshold)
    print("Train (fit-set) metrics @ selected threshold:", {
        k: round(v, 3) if isinstance(v, float) else v
        for k, v in train_metrics.items()
        if k != "confusion_matrix"
    })

    # ---- One-shot held-out evaluation ----
    test_metrics = evaluate_predictions(y_test, test_probs, selected_threshold)
    ci = bootstrap_accuracy_ci(y_test, test_probs, selected_threshold)
    test_metrics["accuracy_bootstrap_ci95"] = ci
    print("HELD-OUT participant test metrics:", {
        k: (round(v, 3) if isinstance(v, float) else v)
        for k, v in test_metrics.items()
        if k not in {"confusion_matrix", "accuracy_bootstrap_ci95"}
    })
    print(
        f"Accuracy 95% CI: {ci['ci95_low']:.3f} – {ci['ci95_high']:.3f} "
        f"(mean={ci['mean']:.3f})"
    )

    # Deployment model: refit on ALL participants only AFTER reporting held-out metrics.
    print("Fitting deployment model on all participants (metrics already locked)...")
    if deployment_kind == "sklearn":
        deployment_model = clone(best_search.best_estimator_)
        deployment_model.fit(X, y)
        artifact_model = deployment_model
    else:
        # Retrain MLP with an internal validation split over all data for early stop.
        val_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
        tr, va = next(val_split.split(X, y))
        mlp_deploy, deploy_threshold, _ = train_embedding_mlp(
            X[tr], y[tr], X[va], y[va]
        )
        # Keep the validation-selected threshold from train-only selection for honesty,
        # but store the deployment-fit MLP weights.
        artifact_model = {
            "kind": "mlp",
            "state_dict": {k: v.cpu() for k, v in mlp_deploy.state_dict().items()},
            "input_dim": int(X.shape[1]),
        }
        # Do not overwrite selected_threshold with deploy_threshold.

    artifact = {
        "model_type": "wavlm_frozen_participant_classifier",
        "ssl_model_id": SSL_MODEL_ID,
        "selected_model": selected_model_name,
        "deployment_kind": deployment_kind,
        "model": artifact_model,
        "threshold": float(selected_threshold),
        "feature_dim": int(X.shape[1]),
        "aggregation": ["mean", "std", "max"],
        "segment_duration": SSL_SEGMENT_DURATION,
        "segment_overlap": SSL_SEGMENT_OVERLAP,
        "max_duration": SSL_MAX_DURATION,
        "max_segments": SSL_MAX_SEGMENTS,
        "train_participant_ids": train_pids,
        "test_participant_ids": test_pids,
        "held_out_test_metrics": test_metrics,
        "training_cv_balanced_accuracy": float(
            mlp_cv if use_mlp else best_cv
        ),
        "candidate_comparison": comparison,
    }
    save_ssl_artifact(SSL_MODEL_PATH, artifact)

    metadata = {
        "n_participants": int(len(y)),
        "n_depressed": int(np.sum(y == 1)),
        "n_non_depressed": int(np.sum(y == 0)),
        "n_features": int(X.shape[1]),
        "ssl_model_id": SSL_MODEL_ID,
        "selected_model": selected_model_name,
        "deployment_kind": deployment_kind,
        "threshold": float(selected_threshold),
        "training_cv_balanced_accuracy": float(mlp_cv if use_mlp else best_cv),
        "train_metrics_at_threshold": train_metrics,
        "held_out_test_metrics": test_metrics,
        "candidate_comparison": comparison,
        "split_path": str(SPLIT_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "model_path": str(SSL_MODEL_PATH),
        "excluded_label_conflicts": conflicts,
        "note": (
            "Held-out test participants were never used for model or threshold "
            "selection. Deployment weights may be refit on all participants after "
            "metrics were recorded."
        ),
    }
    SSL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved SSL model to {SSL_MODEL_PATH}")
    print(f"Saved metadata to {SSL_METADATA_PATH}")

    target_low, target_high = 0.75, 0.85
    acc = test_metrics["accuracy"]
    if target_low <= acc <= target_high:
        print(f"SUCCESS: held-out accuracy {acc:.1%} is within the 75–85% target.")
    elif acc > target_high:
        print(
            f"Held-out accuracy {acc:.1%} exceeds 85%. "
            "Treat as a strong result but verify with a larger cohort."
        )
    else:
        print(
            f"Held-out accuracy {acc:.1%} is below the 75–85% target. "
            "This is the honest leakage-safe result on the current 127-participant set."
        )


if __name__ == "__main__":
    main()
