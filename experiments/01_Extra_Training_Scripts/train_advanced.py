#!/usr/bin/env python3
"""Advanced regularized boosting and validation-selected model ensembles."""

import json
import pickle

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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

CANDIDATE_PATH = MODEL_DIR / "depression_advanced_candidate.pkl"
METADATA_PATH = MODEL_DIR / "advanced_candidate_metadata.json"


def _oof(estimator, X, y, cv):
    probabilities = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        model = clone(estimator).fit(X[train_idx], y[train_idx])
        probabilities[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return probabilities


def _searches(cv, dimensions, class_ratio):
    feature_counts = sorted(
        set(min(value, dimensions) for value in (10, 20, 40, 80))
    )
    return {
        "selected_logistic": GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("select", SelectKBest(f_classif)),
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
                "select__k": feature_counts,
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "shrinkage_lda": GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LinearDiscriminantAnalysis(solver="lsqr"),
                    ),
                ]
            ),
            {
                "model__shrinkage": [
                    "auto",
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                ]
            },
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
        ),
        "xgboost": RandomizedSearchCV(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=class_ratio,
            ),
            {
                "n_estimators": randint(80, 450),
                "max_depth": randint(1, 5),
                "learning_rate": loguniform(0.01, 0.2),
                "min_child_weight": randint(2, 12),
                "subsample": uniform(0.65, 0.35),
                "colsample_bytree": uniform(0.45, 0.5),
                "reg_alpha": loguniform(1e-3, 2.0),
                "reg_lambda": loguniform(0.5, 10.0),
                "gamma": uniform(0.0, 1.0),
            },
            n_iter=36,
            scoring="balanced_accuracy",
            cv=cv,
            random_state=42,
            n_jobs=-1,
        ),
    }


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
        cache_path = ENGINEERED_CACHE_DIR / f"{fingerprint}.npz"
        if not cache_path.exists():
            print(f"Warning: missing feature cache for {pid}")
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
    positive = max(1, int(np.sum(y_train == 1)))
    negative = max(1, int(np.sum(y_train == 0)))
    class_ratio = negative / positive
    candidates = []
    comparison = {}
    for family, (X_train, X_test) in families.items():
        for model_name, search in _searches(
            cv, X_train.shape[1], class_ratio
        ).items():
            name = f"{family}:{model_name}"
            print(f"Cross-validating {name}...")
            search.fit(X_train, y_train)
            score = float(search.best_score_)
            estimator = search.best_estimator_
            probabilities = _oof(estimator, X_train, y_train, cv)
            threshold, oof_score = select_threshold(
                y_train, probabilities
            )
            comparison[name] = {
                "cv_balanced_accuracy": score,
                "oof_threshold_balanced_accuracy": oof_score,
                "best_params": search.best_params_,
            }
            candidates.append(
                {
                    "name": name,
                    "family": family,
                    "estimator": estimator,
                    "X_train": X_train,
                    "X_test": X_test,
                    "oof": probabilities,
                    "cv_score": score,
                    "threshold": threshold,
                    "oof_score": oof_score,
                }
            )
            print(
                f"  CV balanced accuracy={score:.3f}, "
                f"OOF threshold score={oof_score:.3f}"
            )

    # Select individual or two-model blend using development OOF only.
    options = []
    for candidate in candidates:
        options.append(
            {
                "kind": "single",
                "members": [candidate],
                "weights": [1.0],
                "oof": candidate["oof"],
            }
        )
    ranked = sorted(
        candidates, key=lambda item: item["oof_score"], reverse=True
    )[:5]
    for first_index in range(len(ranked)):
        for second_index in range(first_index + 1, len(ranked)):
            first = ranked[first_index]
            second = ranked[second_index]
            for first_weight in (0.25, 0.5, 0.75):
                blend = (
                    first_weight * first["oof"]
                    + (1 - first_weight) * second["oof"]
                )
                options.append(
                    {
                        "kind": "blend",
                        "members": [first, second],
                        "weights": [
                            first_weight,
                            1 - first_weight,
                        ],
                        "oof": blend,
                    }
                )

    for option in options:
        threshold, score = select_threshold(y_train, option["oof"])
        option["threshold"] = threshold
        option["score"] = score
    winner = max(options, key=lambda item: item["score"])
    print(
        "Selected on development OOF:",
        [member["name"] for member in winner["members"]],
        winner["weights"],
        f"score={winner['score']:.3f}",
    )

    fitted_members = []
    train_prob = np.zeros(len(y_train))
    test_prob = np.zeros(len(y_test))
    for member, weight in zip(
        winner["members"], winner["weights"]
    ):
        model = clone(member["estimator"]).fit(
            member["X_train"], y_train
        )
        fitted_members.append(
            {
                "name": member["name"],
                "family": member["family"],
                "weight": weight,
                "model": model,
            }
        )
        train_prob += weight * model.predict_proba(
            member["X_train"]
        )[:, 1]
        test_prob += weight * model.predict_proba(
            member["X_test"]
        )[:, 1]

    threshold = winner["threshold"]
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
        "model_type": "advanced_acoustic_official_phq",
        "selected_models": fitted_members,
        "threshold": float(threshold),
        "training_oof_balanced_accuracy": float(winner["score"]),
        "held_out_test_metrics": test_metrics,
        "official_labels": True,
    }
    with open(CANDIDATE_PATH, "wb") as handle:
        pickle.dump(artifact, handle)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "n_participants": len(rows),
                "selected_models": [
                    {
                        "name": member["name"],
                        "family": member["family"],
                        "weight": member["weight"],
                    }
                    for member in fitted_members
                ],
                "threshold": float(threshold),
                "training_oof_balanced_accuracy": float(
                    winner["score"]
                ),
                "train_metrics": train_metrics,
                "held_out_test_metrics": test_metrics,
                "current_deployed_metrics": current,
                "deploy_eligible": deploy_eligible,
                "candidate_comparison": comparison,
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved advanced candidate to {CANDIDATE_PATH}")
    print(f"DEPLOY_ELIGIBLE={deploy_eligible}")


if __name__ == "__main__":
    main()
