"""SSL-backed inference helpers and segment occlusion attributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import SSL_SEGMENT_DURATION, SSL_SEGMENT_OVERLAP, SSL_MAX_SEGMENTS, SSL_MAX_DURATION
from .features import (
    compute_full_mel_spectrogram,
    compute_mel_spectrogram,
    extract_acoustic_features,
    features_to_vector,
    load_audio,
    segment_audio_with_times,
)
from .ssl_model import (
    EmbeddingBagMLP,
    WavLMEmbedder,
    aggregate_segment_embeddings,
    load_ssl_artifact,
)


def load_ssl_predictor(artifact_path, device=None):
    artifact = load_ssl_artifact(artifact_path)
    kind = artifact.get("deployment_kind", "sklearn")
    threshold = float(artifact.get("threshold", 0.5))
    model = artifact["model"]

    if kind == "wavlm_finetune":
        # Lazy import to avoid loading the heavy training module unless needed.
        from transformers import AutoFeatureExtractor, WavLMModel
        from .config import MODEL_DIR, SSL_MODEL_ID
        import torch.nn as nn

        class _Finetuned(nn.Module):
            def __init__(self):
                super().__init__()
                cache_dir = str(MODEL_DIR / "huggingface_cache")
                model_id = artifact.get("ssl_model_id", SSL_MODEL_ID)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_id, cache_dir=cache_dir
                )
                self.encoder = WavLMModel.from_pretrained(model_id, cache_dir=cache_dir)
                hidden = self.encoder.config.hidden_size
                self.attn = nn.Linear(hidden, 1)
                self.head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(hidden, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),
                )

            def forward_participant(self, segments: torch.Tensor) -> torch.Tensor:
                from .config import SAMPLE_RATE
                arrays = [w.detach().cpu().numpy() for w in segments]
                inputs = self.feature_extractor(
                    arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
                )
                device = next(self.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                hidden = self.encoder(**inputs).last_hidden_state
                weights = torch.softmax(self.attn(hidden).squeeze(-1), dim=1)
                emb = (hidden * weights.unsqueeze(-1)).sum(dim=1)
                pooled = emb.mean(dim=0, keepdim=True)
                return self.head(pooled).squeeze(0)

        finetuned = _Finetuned()
        finetuned.load_state_dict(model["state_dict"])
        finetuned.eval()
        if device is not None:
            finetuned.to(device)
        return {
            "artifact": artifact,
            "embedder": None,
            "kind": kind,
            "threshold": threshold,
            "mlp": None,
            "sklearn_model": None,
            "finetuned": finetuned,
        }

    embedder = WavLMEmbedder(artifact.get("ssl_model_id"), device=device)
    mlp = None
    sklearn_model = None
    if kind == "mlp":
        mlp = EmbeddingBagMLP(model["input_dim"])
        mlp.load_state_dict(model["state_dict"])
        mlp.eval()
        if device is not None:
            mlp.to(device)
    else:
        sklearn_model = model

    return {
        "artifact": artifact,
        "embedder": embedder,
        "kind": kind,
        "threshold": threshold,
        "mlp": mlp,
        "sklearn_model": sklearn_model,
        "finetuned": None,
    }


def predict_from_embeddings(bundle, segment_embeddings: np.ndarray) -> dict:
    vector = aggregate_segment_embeddings(segment_embeddings)
    X = vector.reshape(1, -1)
    if bundle["kind"] == "mlp":
        device = next(bundle["mlp"].parameters()).device
        with torch.no_grad():
            logit = bundle["mlp"](torch.from_numpy(X).float().to(device))
            prob = float(torch.sigmoid(logit).item())
    else:
        prob = float(bundle["sklearn_model"].predict_proba(X)[0, 1])

    threshold = bundle["threshold"]
    prediction = "Depressed" if prob >= threshold else "Non-Depressed"
    confidence = abs(prob - threshold) + 0.5
    confidence = float(min(0.99, max(0.5, confidence)))
    # Prefer probability-based confidence matching prior API.
    confidence = prob if prediction == "Depressed" else (1.0 - prob)
    return {
        "probability_depressed": prob,
        "prediction": prediction,
        "confidence": float(confidence),
        "threshold": threshold,
        "participant_vector": vector,
    }


def segment_depression_scores(bundle, segment_embeddings: np.ndarray) -> np.ndarray:
    """
    Score each segment by replacing the participant mean with that segment only.
    This is a faithful occlusion-style attribution for the SSL classifier.
    """
    scores = []
    for i in range(len(segment_embeddings)):
        one = segment_embeddings[i : i + 1]
        result = predict_from_embeddings(bundle, one)
        scores.append(result["probability_depressed"])
    return np.asarray(scores, dtype=np.float32)


def occlusion_importance(bundle, segment_embeddings: np.ndarray) -> np.ndarray:
    """
    Importance = |P(all) - P(without segment i)|.
    Higher values mean the segment more strongly influenced the decision.
    """
    if len(segment_embeddings) == 0:
        return np.zeros(0, dtype=np.float32)
    full = predict_from_embeddings(bundle, segment_embeddings)["probability_depressed"]
    importances = []
    for i in range(len(segment_embeddings)):
        keep = np.concatenate(
            [segment_embeddings[:i], segment_embeddings[i + 1 :]],
            axis=0,
        )
        if len(keep) == 0:
            importances.append(0.0)
            continue
        without = predict_from_embeddings(bundle, keep)["probability_depressed"]
        importances.append(abs(full - without))
    arr = np.asarray(importances, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def prepare_ssl_audio(
    audio_path: str,
    max_duration: float = SSL_MAX_DURATION,
    transcript_path: str | None = None,
    participant_id: str | None = None,
    original_filename: str | None = None,
):
    """
    Load audio for acoustic/SSL inference.

    Training used participant-only speech from transcripts. When a matching
    DAIC-WOZ transcript is available (by path, participant id, or original
    upload filename like 325_AUDIO.wav), interviewer turns are removed.
    """
    from .data import (
        apply_participant_only_from_transcript,
        find_transcript_for_participant,
    )

    resolved_transcript = transcript_path
    used_participant_id = participant_id

    def _try(pid):
        nonlocal resolved_transcript, used_participant_id
        if resolved_transcript or not pid:
            return
        hit = find_transcript_for_participant(pid)
        if hit is not None:
            resolved_transcript = str(hit)
            used_participant_id = pid

    _try(participant_id)
    if original_filename:
        stem = Path(original_filename).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        _try(digits)
    # Temp upload paths rarely contain the DAIC id; still try.
    stem = Path(audio_path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    _try(digits)

    if resolved_transcript:
        y = load_audio(audio_path, max_duration=None, trim=False, normalize=False)
        y = apply_participant_only_from_transcript(
            y, resolved_transcript, max_duration=max_duration
        )
    else:
        y = load_audio(audio_path, max_duration=max_duration)

    timed = segment_audio_with_times(
        y,
        SSL_SEGMENT_DURATION,
        SSL_SEGMENT_OVERLAP,
    )
    if SSL_MAX_SEGMENTS and len(timed) > SSL_MAX_SEGMENTS:
        indices = np.linspace(0, len(timed) - 1, SSL_MAX_SEGMENTS, dtype=int)
        timed = [timed[i] for i in indices]
    full_mel, full_times = compute_full_mel_spectrogram(y)
    return y, timed, full_mel, full_times, {
        "transcript_used": bool(resolved_transcript),
        "participant_id": used_participant_id,
        "transcript_path": resolved_transcript,
    }
