"""Inference pipeline for uploaded voice recordings."""

import pickle

import numpy as np
import torch

from .config import MODEL_PATH, SCALER_PATH, SEGMENT_DURATION, SEGMENT_OVERLAP
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
)
from .subtype import classify_subtype


class DepressionPredictor:
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = model_dir or MODEL_PATH.parent

        cnn_ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.cnn = DepressionCNN().to(self.device)
        self.cnn.load_state_dict(cnn_ckpt["cnn_state_dict"])
        self.cnn.eval()

        feat_path = base / "feature_model.pt"
        feat_ckpt = torch.load(feat_path, map_location=self.device, weights_only=False)
        self.feat_model = FeatureClassifier(feat_ckpt["n_features"]).to(self.device)
        self.feat_model.load_state_dict(feat_ckpt["feature_model_state_dict"])
        self.feat_model.eval()

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, audio_path: str, context=None) -> dict:
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

            x = torch.from_numpy(spec).unsqueeze(0).to(self.device)
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

        timeline = build_timeline_explanations(segment_details, prediction)
        reason_text = build_prediction_reason(prediction, confidence, timeline, segment_details)

        # Key segment for Grad-CAM: highest-impact supporting segment
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

        X_scaled = self.scaler.transform(np.array(feature_vectors))
        importance = compute_feature_importance(
            self.feat_model, X_scaled, X_scaled, max_samples=min(20, len(X_scaled))
        )

        highlight_regions = [
            {**{k: s[k] for k in ("start_sec", "end_sec", "prob")},
             "role": "supporting" if (
                 (prediction == "Depressed" and s["prob"] >= 0.5)
                 or (prediction == "Non-Depressed" and s["prob"] < 0.5)
             ) else "opposing"}
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
            "segment_explanations": [e["text"] for e in timeline if e["role"] == "supporting"],
            "summary": summary,
            "subtype": subtype_result,
            "audio_duration_sec": len(y) / 16000,
        }
