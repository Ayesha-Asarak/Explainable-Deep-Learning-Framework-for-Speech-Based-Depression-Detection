#!/usr/bin/env python3
"""Predict PHQ-8 severity, then derive depression risk leakage-safely."""

import json
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

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
    select_threshold,
)

MODEL_PATH = MODEL_DIR / "depression_phq_regression_candidate.pkl"
METADATA_PATH = MODEL_DIR / "phq_regression_candidate_metadata.json"
MAX_PHQ8_SCORE = 24.0


def _candidates():
    candidates = []
    for params in ParameterGrid(
        {
            "max_depth": [4, 8, None],
            "min_samples_leaf": [2, 4, 8],
            "max_features": ["sqrt", 0.5],
        }
    ):
        candidates.append(
            (
                "extra_trees",
                params,
                ExtraTreesRegressor(
                    n_estimators=800,
                    random_state=42,
                    n_jobs=-1,
                    **params,
                ),
            )
        )
    for params in ParameterGrid(
        {
            "max_depth": [4, 8, None],
            "min_samples_leaf": [2, 5],
            "max_features": ["sqrt", 0.5],
        }
    ):
        candidates.append(
            (
                "random_forest",
                params,
                RandomForestRegressor(
                    n_estimators=800,
                    random_state=42,
                    n_jobs=-1,
                    **params,
                ),
            )
        )
    for params in ParameterGrid(
        {
            "n_estimators": [100, 250],
            "max_depth": [1, 2],
            "learning_rate": [0.02, 0.05],
        }
    ):
        candidates.append(
            (
                "gradient_boosting",
                params,
                GradientBoostingRegressor(
                    min_samples_leaf=5,
                    loss="huber",
                    random_state=42,
                    **params,
                ),
            )
        )
    for params in ParameterGrid(
        {
            "n_estimators": [150, 300],
            "max_depth": [1, 2, 3],
            "learning_rate": [0.02, 0.05],
        }
    ):
        candidates.append(
            (
                "xgboost",
                params,
                XGBRegressor(
                    objective="reg:squarederror",
                    min_child_weight=8,
                    subsample=0.8,
                    colsample_bytree=0.75,
                    reg_alpha=0.2,
                    reg_lambda=5.0,
                    random_state=42,
                    n_jobs=-1,
                    **params,
                ),
            )
        )
    for alpha in (1.0, 10.0, 100.0):
        for dimensions in (8, 16, 32):
            candidates.append(
                (
                    "pca_ridge",
                    {"alpha": alpha, "dimensions": dimensions},
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "pca",
                                PCA(
                                    n_components=dimensions,
                                    random_state=42,
                                ),
                            ),
                            ("model", Ridge(alpha=alpha)),
                        ]
                    ),
                )
            )
    for c_value in (0.1, 1.0, 10.0):
        candidates.append(
            (
                "svr",
                {"C": c_value},
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            SVR(
                                C=c_value,
                                epsilon=1.0,
                                kernel="rbf",
                            ),
                        ),
                    ]
                ),
            )
        )
    return candidates


def _probabilities(scores):
    return np.clip(np.asarray(scores) / MAX_PHQ8_SCORE, 0.0, 1.0)


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
        score = source.get("phq_score")
        if score is None or pid not in manifest:
            continue
        fingerprint = manifest[pid]["source_fingerprint"]
        cache_path = ENGINEERED_CACHE_DIR / f"{fingerprint}.npz"
        if not cache_path.exists():
            continue
        cache = np.load(cache_path)
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
                "participant_id": pid,
                "label": int(source["label"]),
                "score": float(score),
                "split": source["official_split"],
                "raw": cache["raw_acoustic"].astype(np.float32),
                "engineered": cache["engineered"].astype(np.float32),
                "gender": gender_vector,
            }
        )

    development = [
        row for row in rows if row["split"] in {"train", "dev"}
    ]
    test = [row for row in rows if row["split"] == "test"]
    y_train = np.asarray([row["label"] for row in development])
    scores_train = np.asarray([row["score"] for row in development])
    y_test = np.asarray([row["label"] for row in test])
    families = {
        "raw_gender": (
            np.stack(
                [
                    np.concatenate([row["raw"], row["gender"]])
                    for row in development
                ]
            ),
            np.stack(
                [
                    np.concatenate([row["raw"], row["gender"]])
                    for row in test
                ]
            ),
        ),
        "engineered_gender": (
            np.stack(
                [
                    np.concatenate(
                        [row["engineered"], row["gender"]]
                    )
                    for row in development
                ]
            ),
            np.stack(
                [
                    np.concatenate(
                        [row["engineered"], row["gender"]]
                    )
                    for row in test
                ]
            ),
        ),
        "raw_engineered_gender": (
            np.stack(
                [
                    np.concatenate(
                        [row["raw"], row["engineered"], row["gender"]]
                    )
                    for row in development
                ]
            ),
            np.stack(
                [
                    np.concatenate(
                        [row["raw"], row["engineered"], row["gender"]]
                    )
                    for row in test
                ]
            ),
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for family, (X_train, X_test) in families.items():
        for name, params, estimator in _candidates():
            oof_scores = np.zeros(len(development))
            for train_index, validation_index in cv.split(
                X_train, y_train
            ):
                model = clone(estimator).fit(
                    X_train[train_index],
                    scores_train[train_index],
                )
                oof_scores[validation_index] = model.predict(
                    X_train[validation_index]
                )
            oof_probability = _probabilities(oof_scores)
            threshold, balanced = select_threshold(
                y_train, oof_probability
            )
            mae = float(np.mean(np.abs(oof_scores - scores_train)))
            results.append(
                {
                    "family": family,
                    "name": name,
                    "params": params,
                    "estimator": estimator,
                    "X_train": X_train,
                    "X_test": X_test,
                    "threshold": threshold,
                    "oof_balanced_accuracy": balanced,
                    "oof_mae": mae,
                }
            )
            print(
                f"{family}:{name} {params} -> "
                f"balanced={balanced:.3f}, PHQ MAE={mae:.2f}"
            )

    winner = max(
        results,
        key=lambda item: (
            item["oof_balanced_accuracy"],
            -item["oof_mae"],
        ),
    )
    selected = clone(winner["estimator"]).fit(
        winner["X_train"], scores_train
    )
    train_probability = _probabilities(
        selected.predict(winner["X_train"])
    )
    test_probability = _probabilities(
        selected.predict(winner["X_test"])
    )
    threshold = winner["threshold"]
    train_metrics = evaluate_predictions(
        y_train, train_probability, threshold
    )
    test_metrics = evaluate_predictions(
        y_test, test_probability, threshold
    )
    test_metrics["accuracy_bootstrap_ci95"] = bootstrap_accuracy_ci(
        y_test, test_probability, threshold
    )
    print(
        "Selected:",
        f"{winner['family']}:{winner['name']}",
        winner["params"],
        f"OOF={winner['oof_balanced_accuracy']:.3f}",
    )
    print(
        "HELD-OUT participant test:",
        {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in test_metrics.items()
            if key not in {"confusion_matrix", "accuracy_bootstrap_ci95"}
        },
    )

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
    artifact = {
        "model_type": "phq_regression_official",
        "model": selected,
        "feature_family": winner["family"],
        "selected_model": winner["name"],
        "threshold": float(threshold),
        "max_phq8_score": MAX_PHQ8_SCORE,
        "training_oof_balanced_accuracy": winner[
            "oof_balanced_accuracy"
        ],
        "held_out_test_metrics": test_metrics,
        "official_labels": True,
    }
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "n_participants": len(rows),
                "selected_model": winner["name"],
                "feature_family": winner["family"],
                "best_params": winner["params"],
                "threshold": float(threshold),
                "training_oof_balanced_accuracy": winner[
                    "oof_balanced_accuracy"
                ],
                "training_oof_phq_mae": winner["oof_mae"],
                "train_metrics": train_metrics,
                "held_out_test_metrics": test_metrics,
                "current_deployed_metrics": current,
                "deploy_eligible": deploy_eligible,
                "candidate_comparison": [
                    {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "estimator",
                            "X_train",
                            "X_test",
                        }
                    }
                    for item in results
                ],
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved PHQ regression candidate to {MODEL_PATH}")
    print(f"DEPLOY_ELIGIBLE={deploy_eligible}")


if __name__ == "__main__":
    main()
