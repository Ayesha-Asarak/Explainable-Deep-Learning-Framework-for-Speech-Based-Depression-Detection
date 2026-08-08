#!/usr/bin/env python3
"""Train participant models from speech sampled across complete interviews."""

import json
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    DATA_DIR,
    MODEL_DIR,
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

CACHE_DIR = MODEL_DIR / "full_interview_feature_cache"
MODEL_PATH = MODEL_DIR / "depression_full_interview_candidate.pkl"
METADATA_PATH = MODEL_DIR / "full_interview_candidate_metadata.json"
MAX_SEGMENTS = 64


def _aggregate(vectors):
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


def _uniform_subset(vectors, count):
    if len(vectors) <= count:
        return vectors
    indices = np.linspace(0, len(vectors) - 1, count, dtype=int)
    return [vectors[index] for index in indices]


def _extract(source):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{source['participant_id']}.npz"
    if cache_path.exists():
        return np.load(cache_path)["vectors"]
    audio = load_audio_source(
        source,
        max_duration=None,
        participant_only=True,
    )
    segments = segment_audio(
        audio, SSL_SEGMENT_DURATION, SSL_SEGMENT_OVERLAP
    )
    segments = _uniform_subset(segments, MAX_SEGMENTS)
    vectors = np.stack(
        [
            features_to_vector(extract_acoustic_features(segment))
            for segment in segments
        ]
    ).astype(np.float32)
    np.savez_compressed(cache_path, vectors=vectors)
    return vectors


def _views(vectors, mode):
    if mode.startswith("uniform_"):
        count = int(mode.split("_")[1])
        return [_aggregate(_uniform_subset(vectors, count))]
    if mode == "interleaved_4x16":
        sampled = _uniform_subset(vectors, 64)
        return [
            _aggregate(sampled[offset::4])
            for offset in range(4)
            if len(sampled[offset::4])
        ]
    raise ValueError(f"Unknown view mode: {mode}")


def _models():
    models = {}
    for depth in (6, None):
        for leaf in (2, 4):
            label = "none" if depth is None else str(depth)
            models[f"extra_trees_d{label}_l{leaf}"] = (
                ExtraTreesClassifier(
                    n_estimators=800,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
            )
    for depth in (1, 2, 3):
        models[f"xgboost_d{depth}"] = XGBClassifier(
            n_estimators=300,
            max_depth=depth,
            learning_rate=0.03,
            min_child_weight=6,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=5.0,
            scale_pos_weight=1.55,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    for components in (8, 16, 32):
        models[f"logistic_pca{components}"] = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=components,
                        random_state=42,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=42,
                    ),
                ),
            ]
        )
    return models


def _fit(model, rows, mode):
    views = []
    labels = []
    weights = []
    for row in rows:
        participant_views = _views(row["vectors"], mode)
        views.extend(participant_views)
        labels.extend([row["label"]] * len(participant_views))
        weights.extend(
            [1.0 / len(participant_views)] * len(participant_views)
        )
    X = np.stack(views)
    y = np.asarray(labels)
    sample_weight = np.asarray(weights)
    if isinstance(model, Pipeline):
        model.fit(X, y, model__sample_weight=sample_weight)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _predict(model, rows, mode):
    probabilities = []
    for row in rows:
        participant_views = np.stack(_views(row["vectors"], mode))
        view_probabilities = model.predict_proba(
            participant_views
        )[:, 1]
        probabilities.append(float(np.mean(view_probabilities)))
    return np.asarray(probabilities)


def main():
    sources, conflicts = discover_audio_sources(
        DATA_DIR, include_zips=True
    )
    rows = []
    for number, source in enumerate(sources, 1):
        print(
            f"[{number}/{len(sources)}] full interview "
            f"{source['participant_id']}"
        )
        rows.append(
            {
                "participant_id": source["participant_id"],
                "label": int(source["label"]),
                "split": source["official_split"],
                "vectors": _extract(source),
            }
        )

    development = [
        row for row in rows if row["split"] in {"train", "dev"}
    ]
    test = [row for row in rows if row["split"] == "test"]
    labels_train = np.asarray(
        [row["label"] for row in development]
    )
    labels_test = np.asarray([row["label"] for row in test])
    folds = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    modes = [
        "uniform_16",
        "uniform_32",
        "uniform_48",
        "uniform_64",
        "interleaved_4x16",
    ]
    candidates = []
    for mode in modes:
        for model_name, estimator in _models().items():
            oof = np.zeros(len(development))
            for train_index, validation_index in folds.split(
                np.zeros(len(development)), labels_train
            ):
                train_rows = [
                    development[index] for index in train_index
                ]
                validation_rows = [
                    development[index] for index in validation_index
                ]
                fitted = _fit(
                    clone(estimator), train_rows, mode
                )
                oof[validation_index] = _predict(
                    fitted, validation_rows, mode
                )
            threshold, balanced_accuracy = select_threshold(
                labels_train, oof
            )
            metrics = evaluate_predictions(
                labels_train, oof, threshold
            )
            candidates.append(
                {
                    "mode": mode,
                    "model_name": model_name,
                    "estimator": estimator,
                    "threshold": threshold,
                    "oof_metrics": metrics,
                }
            )
            print(
                f"{mode}:{model_name} -> OOF balanced="
                f"{balanced_accuracy:.3f}, recall="
                f"{metrics['recall']:.3f}"
            )

    eligible = [
        candidate
        for candidate in candidates
        if candidate["oof_metrics"]["recall"] >= 0.60
    ]
    winner = max(
        eligible or candidates,
        key=lambda candidate: (
            candidate["oof_metrics"]["balanced_accuracy"],
            candidate["oof_metrics"]["accuracy"],
        ),
    )
    selected = _fit(
        clone(winner["estimator"]),
        development,
        winner["mode"],
    )
    train_probability = _predict(
        selected, development, winner["mode"]
    )
    test_probability = _predict(selected, test, winner["mode"])
    threshold = winner["threshold"]
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
        "Selected:",
        winner["mode"],
        winner["model_name"],
        winner["oof_metrics"],
    )
    print(
        "HELD-OUT participant test:",
        {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in test_metrics.items()
            if key not in {"confusion_matrix", "accuracy_bootstrap_ci95"}
        },
    )

    deploy_eligible = (
        test_metrics["accuracy"] > 0.59375
        and test_metrics["balanced_accuracy"]
        >= 0.6038961038961039
        and test_metrics["recall"] >= 0.6363636363636364
        and test_metrics["f1"] >= 0.5185185185185185
    )
    artifact = {
        "model_type": "full_interview_acoustic_official",
        "model": selected,
        "selected_model": winner["model_name"],
        "view_mode": winner["mode"],
        "threshold": float(threshold),
        "segment_duration": SSL_SEGMENT_DURATION,
        "segment_overlap": SSL_SEGMENT_OVERLAP,
        "max_duration": None,
        "max_segments": MAX_SEGMENTS,
        "held_out_test_metrics": test_metrics,
        "official_labels": True,
    }
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "n_participants": len(rows),
                "selected_model": winner["model_name"],
                "view_mode": winner["mode"],
                "threshold": float(threshold),
                "oof_metrics": winner["oof_metrics"],
                "train_metrics": train_metrics,
                "held_out_test_metrics": test_metrics,
                "deploy_eligible": deploy_eligible,
                "candidate_comparison": [
                    {
                        "mode": candidate["mode"],
                        "model_name": candidate["model_name"],
                        "threshold": candidate["threshold"],
                        "oof_metrics": candidate["oof_metrics"],
                    }
                    for candidate in candidates
                ],
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved full-interview candidate to {MODEL_PATH}")
    print(f"DEPLOY_ELIGIBLE={deploy_eligible}")


if __name__ == "__main__":
    main()
