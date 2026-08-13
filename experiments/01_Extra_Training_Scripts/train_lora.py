#!/usr/bin/env python3
"""LoRA-tune the final WavLM layers using official PHQ partitions."""

import json
import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import AutoFeatureExtractor, WavLMModel

from src.config import (
    DATA_DIR,
    MODEL_DIR,
    SAMPLE_RATE,
    SSL_MODEL_ID,
    SSL_SEGMENT_DURATION,
)
from src.data import discover_audio_sources, load_audio_source
from src.features import segment_audio
from src.ssl_model import (
    bootstrap_accuracy_ci,
    evaluate_predictions,
    pick_device,
    select_threshold,
)

CANDIDATE_PATH = MODEL_DIR / "depression_lora_candidate.pkl"
METADATA_PATH = MODEL_DIR / "lora_candidate_metadata.json"
MAX_SEGMENTS = 4
TRAIN_SEGMENTS_PER_EPOCH = 2


class WavLMLoRAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        cache_dir = str(MODEL_DIR / "huggingface_cache")
        self.processor = AutoFeatureExtractor.from_pretrained(
            SSL_MODEL_ID, cache_dir=cache_dir
        )
        encoder = WavLMModel.from_pretrained(
            SSL_MODEL_ID, cache_dir=cache_dir
        )
        encoder.feature_extractor._freeze_parameters()
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=[10, 11],
            layers_pattern="layers",
            bias="none",
        )
        self.encoder = get_peft_model(encoder, config)
        hidden = encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1),
        )

    def forward(self, waveforms):
        arrays = [
            np.asarray(waveform, dtype=np.float32)
            for waveform in waveforms
        ]
        inputs = self.processor(
            arrays,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        inputs = {
            key: value.to(device) for key, value in inputs.items()
        }
        hidden = self.encoder(**inputs).last_hidden_state
        segment_embeddings = hidden.mean(dim=1)
        participant_embedding = segment_embeddings.mean(
            dim=0, keepdim=True
        )
        return self.head(participant_embedding).squeeze(0)


def build_records():
    sources, conflicts = discover_audio_sources(
        DATA_DIR, include_zips=True
    )
    records = []
    for number, source in enumerate(sources, 1):
        print(f"[{number}/{len(sources)}] LoRA audio {source['participant_id']}")
        audio = load_audio_source(
            source,
            max_duration=90.0,
            participant_only=True,
        )
        segments = segment_audio(audio, SSL_SEGMENT_DURATION, 0.5)
        if len(segments) > MAX_SEGMENTS:
            indices = np.linspace(
                0, len(segments) - 1, MAX_SEGMENTS, dtype=int
            )
            segments = [segments[index] for index in indices]
        records.append(
            {
                "participant_id": source["participant_id"],
                "label": int(source["label"]),
                "split": source["official_split"],
                "segments": segments,
            }
        )
    return records, conflicts


def predict_records(model, records):
    model.eval()
    labels = []
    probabilities = []
    with torch.no_grad():
        for record in records:
            probability = torch.sigmoid(
                model(record["segments"])
            ).item()
            labels.append(record["label"])
            probabilities.append(probability)
    return np.asarray(labels), np.asarray(probabilities)


def train_epoch(model, records, optimizer, criterion, device):
    model.train()
    order = np.random.permutation(len(records))
    losses = []
    for index in order:
        record = records[index]
        segments = record["segments"]
        if len(segments) > TRAIN_SEGMENTS_PER_EPOCH:
            chosen = np.random.choice(
                len(segments),
                TRAIN_SEGMENTS_PER_EPOCH,
                replace=False,
            )
            segments = [segments[position] for position in chosen]
        augmented = []
        for segment in segments:
            waveform = np.asarray(segment, dtype=np.float32).copy()
            if np.random.rand() < 0.5:
                waveform *= np.random.uniform(0.85, 1.15)
            if np.random.rand() < 0.4:
                waveform += np.random.normal(
                    0, 0.003, waveform.shape
                ).astype(np.float32)
            augmented.append(waveform)

        target = torch.tensor(
            [record["label"]],
            dtype=torch.float32,
            device=device,
        )
        optimizer.zero_grad()
        logit = model(augmented)
        loss = criterion(logit, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = pick_device()
    print(f"LoRA device: {device}")
    records, conflicts = build_records()
    train = [record for record in records if record["split"] == "train"]
    dev = [record for record in records if record["split"] == "dev"]
    test = [record for record in records if record["split"] == "test"]
    print(
        f"Official partitions: {len(train)} train / "
        f"{len(dev)} dev / {len(test)} test"
    )

    model = WavLMLoRAClassifier().to(device)
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    print(
        f"Trainable LoRA/head parameters: "
        f"{sum(parameter.numel() for parameter in trainable):,}"
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        weight_decay=0.01,
    )
    train_labels = np.asarray([record["label"] for record in train])
    positive = max(1, int(np.sum(train_labels == 1)))
    negative = max(1, int(np.sum(train_labels == 0)))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negative / positive],
            dtype=torch.float32,
            device=device,
        )
    )

    best = None
    best_score = -np.inf
    wait = 0
    for epoch in range(1, 7):
        loss = train_epoch(
            model, train, optimizer, criterion, device
        )
        dev_labels, dev_prob = predict_records(model, dev)
        threshold, score = select_threshold(dev_labels, dev_prob)
        dev_metrics = evaluate_predictions(
            dev_labels, dev_prob, threshold
        )
        print(
            f"Epoch {epoch}: loss={loss:.4f}, "
            f"dev_BA={score:.3f}, dev_acc={dev_metrics['accuracy']:.3f}, "
            f"threshold={threshold:.2f}"
        )
        if score > best_score:
            best_score = score
            best = {
                "adapter_state": {
                    key: value.detach().cpu().clone()
                    for key, value in get_peft_model_state_dict(
                        model.encoder
                    ).items()
                },
                "head_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.head.state_dict().items()
                },
                "full_state": deepcopy(model.state_dict()),
                "threshold": float(threshold),
                "dev_metrics": dev_metrics,
                "epoch": epoch,
            }
            wait = 0
        else:
            wait += 1
            if wait >= 2:
                break

    model.load_state_dict(best["full_state"])
    model.to(device)
    development = train + dev
    development_labels, development_prob = predict_records(
        model, development
    )
    test_labels, test_prob = predict_records(model, test)
    threshold = best["threshold"]
    development_metrics = evaluate_predictions(
        development_labels, development_prob, threshold
    )
    test_metrics = evaluate_predictions(
        test_labels, test_prob, threshold
    )
    test_metrics["accuracy_bootstrap_ci95"] = bootstrap_accuracy_ci(
        test_labels, test_prob, threshold
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
        "model_type": "wavlm_lora_official_phq",
        "ssl_model_id": SSL_MODEL_ID,
        "selected_model": "wavlm_lora_final_two_layers",
        "adapter_state": best["adapter_state"],
        "head_state": best["head_state"],
        "threshold": threshold,
        "lora_config": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.1,
            "layers": [10, 11],
            "target_modules": ["q_proj", "v_proj"],
        },
        "best_epoch": best["epoch"],
        "dev_metrics": best["dev_metrics"],
        "held_out_test_metrics": test_metrics,
        "official_labels": True,
    }
    with open(CANDIDATE_PATH, "wb") as handle:
        pickle.dump(artifact, handle)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "n_participants": len(records),
                "partitions": {
                    "train": len(train),
                    "dev": len(dev),
                    "test": len(test),
                },
                "best_epoch": best["epoch"],
                "threshold": threshold,
                "dev_metrics": best["dev_metrics"],
                "development_metrics": development_metrics,
                "held_out_test_metrics": test_metrics,
                "current_deployed_metrics": current,
                "deploy_eligible": deploy_eligible,
                "excluded_label_conflicts": conflicts,
            },
            indent=2,
        )
    )
    print(f"Saved LoRA candidate to {CANDIDATE_PATH}")
    print(f"DEPLOY_ELIGIBLE={deploy_eligible}")


if __name__ == "__main__":
    main()
