"""Inference pipeline for uploaded voice recordings."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from .config import (
    ACOUSTIC_MODEL_PATH,
    ACTIVE_MODEL,
    MODEL_PATH,
    SCALER_PATH,
    SEGMENT_DURATION,
    SEGMENT_OVERLAP,
    SSL_MODEL_PATH,
)
from .features import (
    load_audio,
    segment_audio_with_times,
    compute_mel_spectrogram,
    compute_full_mel_spectrogram,
    extract_acoustic_features,
    features_to_vector,
)
from .model import DepressionCNN, FeatureClassifier
from .explain import (
    compute_grad_cam,
    compute_feature_importance,
    build_timeline_explanations,
    build_prediction_reason,
    build_explanation_summary,
    build_segment_occlusion_map,
    compute_probability_uncertainty,
)
from .subtype import classify_subtype
from .ssl_predict import (
    load_ssl_predictor,
    predict_from_embeddings,
    prepare_ssl_audio,
    segment_depression_scores,
    occlusion_importance,
)
from .ssl_model import pick_device


class DepressionPredictor:
    def __init__(self, model_dir=None):
        self.device = pick_device()
        if self.device.type == "mps":
            # Keep sklearn / plotting on CPU-friendly paths; MPS for WavLM only.
            pass
        base = Path(model_dir) if model_dir else MODEL_PATH.parent
        self.model_dir = base
        self.active_model = ACTIVE_MODEL
        self.ssl_bundle = None
        self.acoustic_artifact = None
        self.cnn = None
        self.feat_model = None
        self.scaler = None

        if ACTIVE_MODEL == "acoustic" and ACOUSTIC_MODEL_PATH.exists():
            self.active_model = "acoustic"
            with open(ACOUSTIC_MODEL_PATH, "rb") as handle:
                self.acoustic_artifact = pickle.load(handle)
            self._load_feature_explainer(base)
        elif ACTIVE_MODEL == "ssl" and SSL_MODEL_PATH.exists():
            self.active_model = "ssl"
            self.ssl_bundle = load_ssl_predictor(SSL_MODEL_PATH, device=self.device)
            # Acoustic feature model remains available for descriptive explanations.
            self._load_feature_explainer(base)
        else:
            self.active_model = "cnn"
            self._load_cnn(base)
            self._load_feature_explainer(base)

    def _load_cnn(self, base: Path):
        cnn_path = MODEL_PATH if MODEL_PATH.exists() else base / "depression_cnn.pt"
        cnn_ckpt = torch.load(cnn_path, map_location="cpu", weights_only=False)
        self.cnn = DepressionCNN()
        self.cnn.load_state_dict(cnn_ckpt["cnn_state_dict"])
        self.cnn.eval()

    def _load_feature_explainer(self, base: Path):
        feat_path = base / "feature_model.pt"
        scaler_path = SCALER_PATH if SCALER_PATH.exists() else base / "feature_scaler.pkl"
        if not feat_path.exists() or not scaler_path.exists():
            return
        feat_ckpt = torch.load(feat_path, map_location="cpu", weights_only=False)
        self.feat_model = FeatureClassifier(feat_ckpt["n_features"])
        self.feat_model.load_state_dict(feat_ckpt["feature_model_state_dict"])
        self.feat_model.eval()
        with open(scaler_path, "rb") as handle:
            self.scaler = pickle.load(handle)

    def predict(self, audio_path: str, context=None) -> dict:
        if (
            self.active_model == "acoustic"
            and self.acoustic_artifact is not None
        ):
            return self._predict_acoustic(audio_path, context=context)
        if self.active_model == "ssl" and self.ssl_bundle is not None:
            return self._predict_ssl(audio_path, context=context)
        return self._predict_cnn(audio_path, context=context)

    @staticmethod
    def _resolve_participant_id(audio_path: str, context=None) -> str | None:
        context = context or {}
        # Prefer DAIC-style ids from the original upload name (temp paths lose this).
        for key in ("original_filename", "audio_filename"):
            name = context.get(key)
            if not name:
                continue
            stem = Path(str(name)).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            if digits:
                return digits
        for key in ("participant_id", "patient_id", "id_number"):
            value = context.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        stem = Path(audio_path).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return digits or None

    @staticmethod
    def _aggregate_acoustic(vectors):
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

    def _acoustic_probability(self, vectors):
        aggregate = self._aggregate_acoustic(vectors).reshape(1, -1)
        model = self.acoustic_artifact["model"]
        return float(model.predict_proba(aggregate)[0, 1])

    def _predict_acoustic(self, audio_path: str, context=None) -> dict:
        context = context or {}
        participant_id = self._resolve_participant_id(audio_path, context)
        y, timed_segments, full_mel, full_times, prep_meta = prepare_ssl_audio(
            audio_path,
            participant_id=participant_id,
            original_filename=context.get("original_filename")
            or context.get("audio_filename"),
        )
        feature_vectors = []
        feature_dicts = []
        spectrograms = []
        for segment in timed_segments:
            audio = segment["audio"]
            features = extract_acoustic_features(audio)
            feature_dicts.append(features)
            feature_vectors.append(features_to_vector(features))
            spectrograms.append(compute_mel_spectrogram(audio))

        threshold = float(self.acoustic_artifact["threshold"])
        avg_prob = self._acoustic_probability(feature_vectors)
        prediction = (
            "Depressed" if avg_prob >= threshold else "Non-Depressed"
        )
        confidence = avg_prob if prediction == "Depressed" else 1 - avg_prob

        # Single-segment probabilities provide the voice timeline.
        segment_probs = [
            self._acoustic_probability([vector])
            for vector in feature_vectors
        ]

        # Faithful leave-one-segment-out importance for the primary model.
        importance_scores = []
        for index in range(len(feature_vectors)):
            remaining = (
                feature_vectors[:index] + feature_vectors[index + 1 :]
            )
            if not remaining:
                importance_scores.append(0.0)
            else:
                without = self._acoustic_probability(remaining)
                importance_scores.append(abs(avg_prob - without))
        importance_scores = np.asarray(
            importance_scores, dtype=np.float32
        )
        if len(importance_scores) and importance_scores.max() > 0:
            importance_scores /= importance_scores.max()

        segment_details = []
        for index, segment in enumerate(timed_segments):
            segment_details.append(
                {
                    "start_sec": segment["start_sec"],
                    "end_sec": segment["end_sec"],
                    "prob": float(segment_probs[index]),
                    "features": feature_dicts[index],
                    "occlusion_importance": float(
                        importance_scores[index]
                    ),
                }
            )

        uncertainty = compute_probability_uncertainty(
            segment_probs, threshold=threshold
        )
        timeline = build_timeline_explanations(
            segment_details, prediction
        )
        reason_text = build_prediction_reason(
            prediction, confidence, timeline, segment_details
        )
        reason_text = (
            f"{reason_text} Model: participant-level acoustic classifier "
            f"trained with official PHQ labels (threshold={threshold:.2f}). "
            f"Uncertainty check: {uncertainty['message']} "
            "Use single-speaker patient audio to match training."
        )

        key_idx = (
            int(np.argmax(importance_scores))
            if len(importance_scores)
            else 0
        )
        key_seg = segment_details[key_idx]
        best_spec = spectrograms[key_idx]
        # Project segment occlusion scores onto the full recording timeline.
        cam = build_segment_occlusion_map(
            full_mel, full_times, segment_details
        )

        # These are descriptive acoustic contributions; primary temporal
        # attribution is the leave-one-segment-out importance above.
        if (
            self.feat_model is not None
            and self.scaler is not None
            and feature_vectors
        ):
            X_scaled = self.scaler.transform(np.asarray(feature_vectors))
            importance = compute_feature_importance(
                self.feat_model,
                X_scaled,
                X_scaled,
                max_samples=min(20, len(X_scaled)),
            )
        else:
            importance = {
                key: float(abs(value))
                for key, value in key_seg["features"].items()
            }

        highlight_regions = [
            {
                **{
                    key: segment[key]
                    for key in ("start_sec", "end_sec", "prob")
                },
                "role": "supporting"
                if (
                    (
                        prediction == "Depressed"
                        and segment["prob"] >= threshold
                    )
                    or (
                        prediction == "Non-Depressed"
                        and segment["prob"] < threshold
                    )
                )
                else "opposing",
            }
            for segment in segment_details
            if abs(segment["prob"] - threshold) > 0.15
            or segment["occlusion_importance"] > 0.4
        ]

        summary = build_explanation_summary(
            prediction, confidence, importance, timeline, reason_text
        )
        subtype_result = classify_subtype(
            segment_details, prediction, avg_prob, context=context
        )
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probability_depressed": avg_prob,
            "segment_probabilities": segment_probs,
            "segment_details": segment_details,
            "n_segments": len(segment_details),
            "best_spectrogram": best_spec,
            "key_segment_start_sec": key_seg["start_sec"],
            "key_segment_end_sec": key_seg["end_sec"],
            "grad_cam": cam,
            "full_mel": full_mel,
            "full_times": full_times,
            "highlight_regions": highlight_regions,
            "timeline_explanations": timeline,
            "prediction_reason": reason_text,
            "feature_importance": importance,
            "acoustic_features": key_seg["features"],
            "segment_explanations": [
                item["text"]
                for item in timeline
                if item["role"] == "supporting"
            ],
            "summary": summary,
            "subtype": subtype_result,
            "uncertainty": uncertainty,
            "audio_duration_sec": len(y) / 16000,
            "model_version": {
                "type": self.acoustic_artifact["model_type"],
                "active_model": "acoustic",
                "threshold": threshold,
                "selected_model": self.acoustic_artifact[
                    "selected_model"
                ],
                "official_labels": True,
                "participant_id": prep_meta.get("participant_id") or participant_id,
                "participant_only_speech": bool(prep_meta.get("transcript_used")),
            },
            "attribution_method": "segment_occlusion",
        }

    def _predict_ssl(self, audio_path: str, context=None) -> dict:
        context = context or {}
        y, timed_segments, full_mel, full_times, _prep_meta = prepare_ssl_audio(
            audio_path,
            participant_id=self._resolve_participant_id(audio_path, context),
            original_filename=context.get("original_filename")
            or context.get("audio_filename"),
        )
        waveforms = [item["audio"] for item in timed_segments]
        segment_embeddings = self.ssl_bundle["embedder"].embed_waveforms(waveforms)
        overall = predict_from_embeddings(self.ssl_bundle, segment_embeddings)
        segment_probs = segment_depression_scores(
            self.ssl_bundle, segment_embeddings
        ).tolist()
        importance_scores = occlusion_importance(
            self.ssl_bundle, segment_embeddings
        )

        threshold = float(overall["threshold"])
        prediction = overall["prediction"]
        avg_prob = float(overall["probability_depressed"])
        confidence = float(overall["confidence"])

        segment_details = []
        spectrograms = []
        feature_vectors = []
        for idx, seg_info in enumerate(timed_segments):
            seg = seg_info["audio"]
            spec = compute_mel_spectrogram(seg)
            feats = extract_acoustic_features(seg)
            spectrograms.append(spec)
            feature_vectors.append(features_to_vector(feats))
            segment_details.append({
                "start_sec": seg_info["start_sec"],
                "end_sec": seg_info["end_sec"],
                "prob": float(segment_probs[idx]),
                "features": feats,
                "occlusion_importance": float(importance_scores[idx])
                if idx < len(importance_scores)
                else 0.0,
            })

        uncertainty = compute_probability_uncertainty(
            segment_probs, threshold=threshold
        )
        timeline = build_timeline_explanations(segment_details, prediction)
        reason_text = build_prediction_reason(
            prediction, confidence, timeline, segment_details
        )
        reason_text = (
            f"{reason_text} Model: WavLM frozen embeddings "
            f"(threshold={threshold:.2f}). "
            f"Uncertainty check: {uncertainty['message']} "
            "Note: upload should be single-speaker patient speech to match training."
        )

        # Key segment by occlusion importance for faithful explanation.
        if len(importance_scores):
            key_idx = int(np.argmax(importance_scores))
        else:
            key_idx = int(np.argmax(np.abs(np.asarray(segment_probs) - threshold)))
        key_seg = segment_details[key_idx]
        best_spec = spectrograms[key_idx]
        # Project segment occlusion scores onto the full recording timeline.
        cam = build_segment_occlusion_map(
            full_mel, full_times, segment_details
        )

        if self.feat_model is not None and self.scaler is not None and feature_vectors:
            X_scaled = self.scaler.transform(np.array(feature_vectors))
            importance = compute_feature_importance(
                self.feat_model,
                X_scaled,
                X_scaled,
                max_samples=min(20, len(X_scaled)),
            )
        else:
            # Fallback descriptive importance from acoustic magnitude variability.
            importance = {
                k: float(abs(v))
                for k, v in key_seg["features"].items()
            }

        highlight_regions = [
            {
                **{k: s[k] for k in ("start_sec", "end_sec", "prob")},
                "role": "supporting"
                if (
                    (prediction == "Depressed" and s["prob"] >= threshold)
                    or (prediction == "Non-Depressed" and s["prob"] < threshold)
                )
                else "opposing",
            }
            for s in segment_details
            if abs(s["prob"] - threshold) > 0.15
            or s.get("occlusion_importance", 0) > 0.4
        ]

        summary = build_explanation_summary(
            prediction, confidence, importance, timeline, reason_text
        )
        subtype_result = classify_subtype(
            segment_details, prediction, avg_prob, context=context
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probability_depressed": avg_prob,
            "segment_probabilities": segment_probs,
            "segment_details": segment_details,
            "n_segments": len(segment_details),
            "best_spectrogram": best_spec,
            "key_segment_start_sec": key_seg["start_sec"],
            "key_segment_end_sec": key_seg["end_sec"],
            "grad_cam": cam,
            "full_mel": full_mel,
            "full_times": full_times,
            "highlight_regions": highlight_regions,
            "timeline_explanations": timeline,
            "prediction_reason": reason_text,
            "feature_importance": importance,
            "acoustic_features": key_seg["features"],
            "segment_explanations": [
                e["text"] for e in timeline if e["role"] == "supporting"
            ],
            "summary": summary,
            "subtype": subtype_result,
            "uncertainty": uncertainty,
            "audio_duration_sec": len(y) / 16000,
            "model_version": {
                "type": "wavlm_frozen_participant_classifier",
                "active_model": "ssl",
                "threshold": threshold,
                "ssl_model_id": self.ssl_bundle["artifact"].get("ssl_model_id"),
                "selected_model": self.ssl_bundle["artifact"].get("selected_model"),
            },
            "attribution_method": "segment_occlusion",
        }

    def _predict_cnn(self, audio_path: str, context=None) -> dict:
        y = load_audio(audio_path, max_duration=180.0)
        timed_segments = segment_audio_with_times(y, SEGMENT_DURATION, SEGMENT_OVERLAP)
        full_mel, full_times = compute_full_mel_spectrogram(y)

        segment_details = []
        spectrograms = []
        feature_vectors = []

        for seg_info in timed_segments:
            seg = seg_info["audio"]
            spec = compute_mel_spectrogram(seg)
            feats = extract_acoustic_features(seg)
            spectrograms.append(spec)
            feature_vectors.append(features_to_vector(feats))

            x = torch.from_numpy(spec).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(self.cnn(x)).item()

            segment_details.append({
                "start_sec": seg_info["start_sec"],
                "end_sec": seg_info["end_sec"],
                "prob": prob,
                "features": feats,
            })

        segment_probs = [s["prob"] for s in segment_details]
        avg_prob = float(np.mean(segment_probs))
        prediction = "Depressed" if avg_prob >= 0.5 else "Non-Depressed"
        confidence = avg_prob if avg_prob >= 0.5 else 1.0 - avg_prob
        uncertainty = compute_probability_uncertainty(segment_probs, threshold=0.5)

        timeline = build_timeline_explanations(segment_details, prediction)
        reason_text = build_prediction_reason(
            prediction, confidence, timeline, segment_details
        )
        reason_text = f"{reason_text} Uncertainty check: {uncertainty['message']}"

        supporting = [e for e in timeline if e["role"] == "supporting"]
        if supporting:
            if prediction == "Depressed":
                key = max(supporting, key=lambda e: e["probability"])
            else:
                key = min(supporting, key=lambda e: e["probability"])
            key_idx = next(
                i for i, s in enumerate(segment_details)
                if s["start_sec"] == key["start_sec"]
            )
        else:
            key_idx = int(np.argmin(np.abs(np.array(segment_probs) - avg_prob)))

        key_seg = segment_details[key_idx]
        best_spec = spectrograms[key_idx]
        cam = compute_grad_cam(self.cnn, best_spec)

        if self.feat_model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(np.array(feature_vectors))
            importance = compute_feature_importance(
                self.feat_model, X_scaled, X_scaled, max_samples=min(20, len(X_scaled))
            )
        else:
            importance = {k: float(abs(v)) for k, v in key_seg["features"].items()}

        highlight_regions = [
            {
                **{k: s[k] for k in ("start_sec", "end_sec", "prob")},
                "role": "supporting"
                if (
                    (prediction == "Depressed" and s["prob"] >= 0.5)
                    or (prediction == "Non-Depressed" and s["prob"] < 0.5)
                )
                else "opposing",
            }
            for s in segment_details
            if abs(s["prob"] - 0.5) > 0.15
        ]

        summary = build_explanation_summary(
            prediction, confidence, importance, timeline, reason_text
        )
        subtype_result = classify_subtype(
            segment_details, prediction, avg_prob, context=context
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probability_depressed": avg_prob,
            "segment_probabilities": segment_probs,
            "segment_details": segment_details,
            "n_segments": len(segment_details),
            "best_spectrogram": best_spec,
            "key_segment_start_sec": key_seg["start_sec"],
            "key_segment_end_sec": key_seg["end_sec"],
            "grad_cam": cam,
            "full_mel": full_mel,
            "full_times": full_times,
            "highlight_regions": highlight_regions,
            "timeline_explanations": timeline,
            "prediction_reason": reason_text,
            "feature_importance": importance,
            "acoustic_features": key_seg["features"],
            "segment_explanations": [
                e["text"] for e in timeline if e["role"] == "supporting"
            ],
            "summary": summary,
            "subtype": subtype_result,
            "uncertainty": uncertainty,
            "audio_duration_sec": len(y) / 16000,
            "model_version": {
                "type": "depression_cnn",
                "active_model": "cnn",
                "threshold": 0.5,
            },
            "attribution_method": "grad_cam",
        }
