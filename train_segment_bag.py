#!/usr/bin/env python3
"""Train local-window acoustic models with participant-grouped validation."""

import json
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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

CACHE_DIR = MODEL_DIR / "segment_acoustic_cache"
MODEL_PATH = MODEL_DIR / "depression_segment_bag_candidate.pkl"
METADATA_PATH = MODEL_DIR / "segment_bag_candidate_metadata.json"


def _segments(source):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{source['participant_id']}.npz"
    if cache_path.exists():
        return np.load(cache_path)["vectors"]
    audio = load_audio_source(
        source,
        max_duration=SSL_MAX_DURATION,
        participant_only=True,
    )
    windows = segment_audio(
        audio, SSL_SEGMENT_DURATION, SSL_SEGMENT_OVERLAP
    )
    if len(windows) > SSL_MAX_SEGMENTS:
        indices = np.linspace(
            0, len(windows) - 1, SSL_MAX_SEGMENTS, dtype=int
        )
        windows = [windows[index] for index in indices]
    vectors = np.stack(
        [
            features_to_vector(extract_acoustic_features(window))
            for window in windows
        ]
    ).astype(np.float32)
    np.savez_compressed(cache_path, vectors=vectors)
    return vectors


def _participant_probability(probabilities, aggregation):
    if aggregation == "mean":
        return float(np.mean(probabilities))
    if aggregation == "median":
        return float(np.median(probabilities))
    if aggregation == "top_quartile":
        count = max(1, int(np.ceil(len(probabilities) * 0.25)))
        return float(np.mean(np.sort(probabilities)[-count:]))
    raise ValueError(aggregation)


def _fit(model, rows):
    X = np.concatenate([row["vectors"] for row in rows])
    y = np.concatenate(
        [
            np.full(len(row["vectors"]), row["label"], dtype=int)
            for row in rows
        ]
    )
    weights = np.concatenate(
        [
            np.full(
                len(row["vectors"]),
                1.0 / len(row["vectors"]),
                dtype=np.float32,
            )
            for row in rows
        ]
    )
    if isinstance(model, Pipeline):
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def _predict(model, rows, aggregation):
    return np.asarray(
        [
            _participant_probability(
                model.predict_proba(row["vectors"])[:, 1],
                aggregation,
            )
            for row in rows
        ]
    )


def _models(class_ratio):
    models = {}
    for leaf in (2, 5, 10):
        models[f"extra_trees_leaf{leaf}"] = ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=leaf,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    for depth in (1, 2, 3):
        for rate in (0.02, 0.05):
            models[f"xgb_depth{depth}_rate{rate}"] = XGBClassifier(
                n_estimators=250,
                max_depth=depth,
                learning_rate=rate,
                min_child_weight=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=4.0,
                scale_pos_weight=class_ratio,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
    for regularization in (0.01, 0.1, 1.0):
        models[f"logistic_c{regularization}"] = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=regularization,
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=42,
                    ),
                ),
            ]
        )
    return models


def main():
    sources, conflicts = discover_audio_sources(
        DATA_DIR, include_zips=True
    )
    rows = []
    for number, source in enumerate(sources, 1):
        print(
            f"[{number}/{len(sources)}] segment features "
            f"{source['participant_id']}"
        )
        rows.append(
            {
                "participant_id": source["participant_id"],
                "label": int(source["label"]),
                "split": source["official_split"],
                "vectors": _segments(source),
            }
        )
    development = [
        row for row in rows if row["split"] in {"train", "dev"}
    ]
    test = [row for row in rows if row["split"] == "test"]
    y_train = np.asarray([row["label"] for row in development])
    y_test = np.asarray([row["label"] for row in test])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    class_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

    candidates = []
    for name, model in _models(class_ratio).items():
        fold_probabilities = {
            aggregation: np.zeros(len(development))
            for aggregation in ("mean", "median", "top_quartile")
        }
        for train_indices, val_indices in cv.split(
            np.zeros(len(development)), y_train
        ):
            train_rows = [development[index] for index in train_indices]
            val_rows = [development[index] for index in val_indices]
            fitted = _fit(pickle.loads(pickle.dumps(model)), train_rows)
            for aggregation in fold_probabilities:
                fold_probabilities[aggregation][val_indices] = _predict(
                    fitted, val_rows, aggregation
                )
        for aggregation, probabilities in fold_probabilities.items():
            threshold, score = select_threshold(y_train, probabilities)
            candidates.append(
                {
                    "name": name,
                    "model": model,
                    "aggregation": aggregation,
                    "threshold": threshold,
                    "oof_score": score,
                }
            )
            print(
                f"{name}/{aggregation}: OOF balanced "
                f"accuracy={score:.3f}"
            )

    winner = max(candidates, key=lambda item: item["oof_score"])
    selected = _fit(winner["model"], development)
    train_prob = _predict(
        selected, development, winner["aggregation"]
    )
    test_prob = _predict(selected, test, winner["aggregation"])
    threshold = winner["threshold"]
    train_metrics = evaluate_predictions(
        y_train, train_prob, threshold
    )
    test_metrics = evaluate_predictions(y_test, test_prob, threshold)
    test_metrics["accuracy_bootstrap_ci95"] = bootstrap_accuracy_ci(
        y_test, test_prob, threshold
    )
    print(
        "Selected:",
        winner["name"],
        winner["aggregation"],
        f"OOF={winner['oof_score']:.3f}",
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
        "model_type": "segment_bag_official_phq",
        "model": selected,
        "selected_model": winner["name"],
        "aggregation": winner["aggregation"],
        "threshold": float(threshold),
        "segment_duration": SSL_SEGMENT_DURATION,
        "segment_overlap": SSL_SEGMENT_OVERLAP,
        "max_duration": SSL_MAX_DURATION,
        "max_segments": SSL_MAX_SEGMENTS,
        "training_oof_balanced_accuracy": winner["oof_score"],
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
                "aggregation": winner["aggregation"],
                "threshold": float(threshold),
                "training_oof_balanced_accuracy": winner["oof_score"],
                "train_metrics": train_metrics,
                "held_out_test_metrics": test_metrics,
                "current_deployed_metrics": current,
                "deploy_eligible": deploy_eligible,
                "candidate_comparison": [
                    {
                        "name": item["name"],
                        "aggregation": item["aggregation"],
                        "oof_balanced_accuracy": item["oof_score"],
                    }
                    for item in candidates
                ],
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved segment-bag candidate to {MODEL_PATH}")
    print(f"DEPLOY_ELIGIBLE={deploy_eligible}")


if __name__ == "__main__":
    main()
