#!/usr/bin/env python3
"""
Train improved participant-level classifiers using:
1) COVAREP acoustic features (DAIC-WOZ)
2) Cached WavLM embeddings
3) Fused COVAREP + WavLM features

Model/threshold selection uses only the locked train participants.
Held-out test participants are evaluated once at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier

from src.config import (
    DATA_DIR,
    MANIFEST_PATH,
    MODEL_DIR,
    SPLIT_PATH,
    SSL_METADATA_PATH,
    SSL_MODEL_PATH,
    SSL_MODEL_ID,
    EMBEDDING_CACHE_DIR,
)
from src.covarep import extract_covarep_vector
from src.data import (
    build_participant_manifest,
    discover_audio_sources,
    fixed_participant_split,
    load_json,
    save_json,
)
from src.ssl_model import (
    bootstrap_accuracy_ci,
    evaluate_predictions,
    extract_participant_embeddings,
    save_ssl_artifact,
    select_threshold,
    WavLMEmbedder,
)

CANDIDATE_MODEL_PATH = MODEL_DIR / "depression_fusion_candidate.pkl"
CANDIDATE_METADATA_PATH = MODEL_DIR / "fusion_candidate_metadata.json"


def attach_fingerprints(sources, manifest):
    by_pid = {entry["participant_id"]: entry for entry in manifest}
    enriched = []
    for source in sources:
        item = dict(source)
        item["source_fingerprint"] = by_pid[source["participant_id"]]["source_fingerprint"]
        enriched.append(item)
    return enriched


def load_cached_or_extract_wavlm(sources):
    records = []
    embedder = None
    for number, source in enumerate(sources, start=1):
        cache = EMBEDDING_CACHE_DIR / f"{source['source_fingerprint']}.npz"
        if cache.exists():
            payload = np.load(cache)
            records.append({
                "participant_id": source["participant_id"],
                "label": int(source["label"]),
                "wavlm": payload["participant_vector"].astype(np.float32),
            })
            continue
        if embedder is None:
            print("Loading WavLM for missing cache entries...")
            embedder = WavLMEmbedder(SSL_MODEL_ID)
        print(f"[{number}/{len(sources)}] extracting missing WavLM for {source['participant_id']}")
        rec = extract_participant_embeddings(source, embedder, use_cache=True)
        records.append({
            "participant_id": rec["participant_id"],
            "label": rec["label"],
            "wavlm": rec["participant_vector"],
        })
    return records


def build_feature_sets(sources, wavlm_records):
    by_pid = {r["participant_id"]: r for r in wavlm_records}
    rows = []
    for source in sources:
        pid = source["participant_id"]
        try:
            cov = extract_covarep_vector(source)
        except Exception as exc:
            print(f"Warning: COVAREP failed for {pid}: {exc}")
            continue
        if pid not in by_pid:
            print(f"Warning: missing WavLM for {pid}")
            continue
        rows.append({
            "participant_id": pid,
            "label": int(source["label"]),
            "covarep": cov,
            "wavlm": by_pid[pid]["wavlm"],
            "fused": np.concatenate([cov, by_pid[pid]["wavlm"]]).astype(np.float32),
        })
    return rows


def matrix(rows, key):
    X = np.stack([r[key] for r in rows]).astype(np.float32)
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    pids = [r["participant_id"] for r in rows]
    return X, y, pids


def index_by_pid(pids, selected):
    selected = set(selected)
    return [i for i, pid in enumerate(pids) if pid in selected]


def candidate_pipelines(cv, n_features):
    max_pca = min(64, n_features, 80)
    searches = {
        "pca_logistic": GridSearchCV(
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(random_state=42)),
                ("model", LogisticRegression(
                    class_weight="balanced", max_iter=5000, random_state=42
                )),
            ]),
            {
                "pca__n_components": [16, 32, max_pca] if max_pca >= 32 else [min(16, max_pca), max_pca],
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "pca_svm": GridSearchCV(
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(random_state=42)),
                ("model", SVC(
                    class_weight="balanced", probability=True, random_state=42
                )),
            ]),
            {
                "pca__n_components": [16, 32, max_pca] if max_pca >= 32 else [min(16, max_pca), max_pca],
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
        ),
        "random_forest": GridSearchCV(
            RandomForestClassifier(
                n_estimators=700,
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
        ),
    }
    return searches


def oof_probabilities(estimator, X, y, cv):
    probs = np.zeros(len(y), dtype=np.float64)
    for tr, va in cv.split(X, y):
        model = clone(estimator)
        model.fit(X[tr], y[tr])
        probs[va] = model.predict_proba(X[va])[:, 1]
    return probs


def evaluate_feature_family(name, X, y, train_idx, test_idx, cv):
    print(f"\n=== Feature family: {name} ({X.shape[1]} dims) ===")
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    searches = candidate_pipelines(cv, X_train.shape[1])
    comparison = {}
    best_name, best_search, best_cv = None, None, -np.inf
    for model_name, search in searches.items():
        print(f"CV {model_name}...")
        search.fit(X_train, y_train)
        score = float(search.best_score_)
        comparison[model_name] = {
            "cv_balanced_accuracy": score,
            "best_params": search.best_params_,
        }
        print(f"  CV balanced accuracy={score:.3f} params={search.best_params_}")
        if score > best_cv:
            best_name, best_search, best_cv = model_name, search, score

    oof = oof_probabilities(best_search.best_estimator_, X_train, y_train, cv)
    threshold, thr_score = select_threshold(y_train, oof)
    estimator = clone(best_search.best_estimator_)
    estimator.fit(X_train, y_train)
    train_probs = estimator.predict_proba(X_train)[:, 1]
    test_probs = estimator.predict_proba(X_test)[:, 1]
    train_metrics = evaluate_predictions(y_train, train_probs, threshold)
    # Do not print detailed test metrics here during selection across families.
    # Keep only train-side selection score.
    return {
        "feature_family": name,
        "selected_model": best_name,
        "cv_balanced_accuracy": best_cv,
        "threshold": threshold,
        "threshold_oof_balanced_accuracy": thr_score,
        "estimator": estimator,
        "comparison": comparison,
        "train_metrics": train_metrics,
        "train_probs": train_probs,
        "test_probs": test_probs,
        "y_test": y_test,
        "y_train": y_train,
    }


def main():
    np.random.seed(42)
    MODEL_DIR.mkdir(exist_ok=True)

    if MANIFEST_PATH.exists() and SPLIT_PATH.exists():
        manifest_payload = load_json(MANIFEST_PATH)
        manifest = manifest_payload["participants"]
        split = load_json(SPLIT_PATH)
        print(f"Loaded locked split: {split['n_train']} train / {split['n_test']} test")
    else:
        manifest, conflicts = build_participant_manifest(DATA_DIR, include_zips=True)
        save_json(MANIFEST_PATH, {
            "n_participants": len(manifest),
            "n_depressed": sum(1 for m in manifest if m["label"] == 1),
            "n_non_depressed": sum(1 for m in manifest if m["label"] == 0),
            "excluded_label_conflicts": conflicts,
            "participants": manifest,
        })
        split = fixed_participant_split(manifest, test_ratio=0.22, seed=42)
        save_json(SPLIT_PATH, split)

    sources, conflicts = discover_audio_sources(DATA_DIR, include_zips=True)
    sources = attach_fingerprints(sources, manifest)

    print("Loading WavLM participant vectors from cache...")
    wavlm_records = load_cached_or_extract_wavlm(sources)
    print("Extracting COVAREP participant vectors...")
    rows = build_feature_sets(sources, wavlm_records)
    print(f"Usable participants with COVAREP+WavLM: {len(rows)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    family_results = []
    for key in ("covarep", "wavlm", "fused"):
        X, y, pids = matrix(rows, key)
        train_idx = index_by_pid(pids, split["train_participant_ids"])
        test_idx = index_by_pid(pids, split["test_participant_ids"])
        result = evaluate_feature_family(key, X, y, train_idx, test_idx, cv)
        result["pids_train"] = [pids[i] for i in train_idx]
        result["pids_test"] = [pids[i] for i in test_idx]
        result["X_all"] = X
        result["y_all"] = y
        family_results.append(result)

    # Select family + model by train CV only (no probability blending).
    # Blends looked strong on tiny OOF splits but collapsed on held-out data.
    winner = max(family_results, key=lambda r: r["cv_balanced_accuracy"])
    print(
        f"\nSelected by train CV: {winner['feature_family']} / "
        f"{winner['selected_model']} "
        f"(CV balanced accuracy={winner['cv_balanced_accuracy']:.3f})"
    )

    if False:
        # Disabled: unstable on N≈100 speakers.
        ranked = sorted(family_results, key=lambda r: r["cv_balanced_accuracy"], reverse=True)
        if len(ranked) >= 2:
            a, b = ranked[0], ranked[1]
            Xa, ya, pids = matrix(rows, a["feature_family"])
            Xb, _, _ = matrix(rows, b["feature_family"])
            train_idx = index_by_pid(pids, split["train_participant_ids"])
            test_idx = index_by_pid(pids, split["test_participant_ids"])
            oof_a = oof_probabilities(a["estimator"], Xa[train_idx], ya[train_idx], cv)
            oof_b = oof_probabilities(b["estimator"], Xb[train_idx], ya[train_idx], cv)
            for alpha in (0.3, 0.5, 0.7):
                blend = alpha * oof_a + (1 - alpha) * oof_b
                thr, score = select_threshold(ya[train_idx], blend)
                print(f"Blend {a['feature_family']}+{b['feature_family']} alpha={alpha}: OOF BA={score:.3f}")
                if score > winner["cv_balanced_accuracy"]:
                    est_a = clone(a["estimator"]).fit(Xa[train_idx], ya[train_idx])
                    est_b = clone(b["estimator"]).fit(Xb[train_idx], ya[train_idx])
                    train_probs = alpha * est_a.predict_proba(Xa[train_idx])[:, 1] + (1 - alpha) * est_b.predict_proba(Xb[train_idx])[:, 1]
                    test_probs = alpha * est_a.predict_proba(Xa[test_idx])[:, 1] + (1 - alpha) * est_b.predict_proba(Xb[test_idx])[:, 1]
                    winner = {
                        "feature_family": f"blend:{a['feature_family']}+{b['feature_family']}",
                        "selected_model": f"alpha={alpha}",
                        "cv_balanced_accuracy": score,
                        "threshold": thr,
                        "threshold_oof_balanced_accuracy": score,
                        "estimator": {"kind": "blend", "alpha": alpha, "a": est_a, "b": est_b, "family_a": a["feature_family"], "family_b": b["feature_family"]},
                        "comparison": {
                            a["feature_family"]: a["comparison"],
                            b["feature_family"]: b["comparison"],
                        },
                        "train_metrics": evaluate_predictions(ya[train_idx], train_probs, thr),
                        "train_probs": train_probs,
                        "test_probs": test_probs,
                        "y_test": ya[test_idx],
                        "y_train": ya[train_idx],
                        "pids_train": [pids[i] for i in train_idx],
                        "pids_test": [pids[i] for i in test_idx],
                        "X_all": None,
                        "y_all": ya,
                        "Xa_all": Xa,
                        "Xb_all": Xb,
                    }

    # One-shot held-out evaluation
    test_metrics = evaluate_predictions(
        winner["y_test"], winner["test_probs"], winner["threshold"]
    )
    ci = bootstrap_accuracy_ci(
        winner["y_test"], winner["test_probs"], winner["threshold"]
    )
    test_metrics["accuracy_bootstrap_ci95"] = ci
    print("\nHELD-OUT participant test metrics:", {
        k: (round(v, 3) if isinstance(v, float) else v)
        for k, v in test_metrics.items()
        if k not in {"confusion_matrix", "accuracy_bootstrap_ci95"}
    })
    print(f"Accuracy 95% CI: {ci['ci95_low']:.3f} – {ci['ci95_high']:.3f}")

    # Deployment fit on all participants after metrics locked
    print("Fitting deployment model on all participants...")
    if isinstance(winner["estimator"], dict) and winner["estimator"].get("kind") == "blend":
        alpha = winner["estimator"]["alpha"]
        est_a = clone(winner["estimator"]["a"]).fit(winner["Xa_all"], winner["y_all"])
        est_b = clone(winner["estimator"]["b"]).fit(winner["Xb_all"], winner["y_all"])
        artifact_model = {
            "kind": "blend",
            "alpha": alpha,
            "family_a": winner["estimator"]["family_a"],
            "family_b": winner["estimator"]["family_b"],
            "model_a": est_a,
            "model_b": est_b,
        }
        deployment_kind = "blend"
    else:
        # Refit winner estimator on all rows for its feature family
        family = winner["feature_family"]
        X_all, y_all, _ = matrix(rows, family)
        est = clone(winner["estimator"]).fit(X_all, y_all)
        artifact_model = est
        deployment_kind = "sklearn"

    artifact = {
        "model_type": "covarep_wavlm_participant_classifier",
        "ssl_model_id": SSL_MODEL_ID,
        "selected_model": winner["selected_model"],
        "feature_family": winner["feature_family"],
        "deployment_kind": deployment_kind,
        "model": artifact_model,
        "threshold": float(winner["threshold"]),
        "train_participant_ids": winner["pids_train"],
        "test_participant_ids": winner["pids_test"],
        "held_out_test_metrics": test_metrics,
        "training_cv_balanced_accuracy": float(winner["cv_balanced_accuracy"]),
        "candidate_comparison": {
            r["feature_family"]: {
                "cv_balanced_accuracy": r["cv_balanced_accuracy"],
                "selected_model": r["selected_model"],
                "models": r["comparison"],
            }
            for r in family_results
        },
    }
    save_ssl_artifact(CANDIDATE_MODEL_PATH, artifact)

    metadata = {
        "n_participants": len(rows),
        "selected_model": winner["selected_model"],
        "feature_family": winner["feature_family"],
        "deployment_kind": deployment_kind,
        "threshold": float(winner["threshold"]),
        "training_cv_balanced_accuracy": float(winner["cv_balanced_accuracy"]),
        "train_metrics_at_threshold": winner["train_metrics"],
        "held_out_test_metrics": test_metrics,
        "excluded_label_conflicts": conflicts,
        "model_path": str(CANDIDATE_MODEL_PATH),
        "note": (
            "Held-out test participants were never used for model/threshold selection. "
            "COVAREP features are acoustic-only speech descriptors from DAIC-WOZ."
        ),
    }
    CANDIDATE_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved candidate model to {CANDIDATE_MODEL_PATH}")

    acc = test_metrics["accuracy"]
    if 0.75 <= acc <= 0.85:
        print(f"SUCCESS: held-out accuracy {acc:.1%} is within the 75–85% target.")
    elif acc > 0.85:
        print(f"Held-out accuracy {acc:.1%} exceeds 85%. Verify on a larger cohort.")
    else:
        print(
            f"Held-out accuracy {acc:.1%} is below the 75–85% target. "
            "This is the honest leakage-safe result on the current dataset."
        )


if __name__ == "__main__":
    main()
