"""
Depression subtype profile classifier.

Maps acoustic speech patterns to clinically-inspired depression subtypes.
Research prototype only — NOT a clinical subtype diagnosis.
"""

import numpy as np

SUBTYPES = {
    "MDD": {
        "id": "MDD",
        "name": "Major Depressive Disorder (MDD)",
        "short": "MDD",
        "description": "Persistent sadness, loss of interest, fatigue",
        "symptoms": ["Persistent sadness", "Loss of interest", "Fatigue", "Low motivation"],
    },
    "Dysthymia": {
        "id": "Dysthymia",
        "name": "Persistent Depressive Disorder (Dysthymia)",
        "short": "Dysthymia",
        "description": "Long-term depression lasting 2+ years",
        "symptoms": ["Chronic low mood", "Milder but persistent symptoms", "Long duration"],
    },
    "Bipolar": {
        "id": "Bipolar",
        "name": "Bipolar Depression",
        "short": "Bipolar",
        "description": "Depressive episodes occurring in bipolar disorder",
        "symptoms": ["Depressive episodes", "Mood instability", "Energy fluctuations"],
    },
    "SAD": {
        "id": "SAD",
        "name": "Seasonal Affective Disorder (SAD)",
        "short": "SAD",
        "description": "Depression related to seasonal changes",
        "symptoms": ["Seasonal pattern", "Winter lethargy", "Reduced daylight response"],
    },
    "Postpartum": {
        "id": "Postpartum",
        "name": "Postpartum Depression",
        "short": "Postpartum",
        "description": "Depression after childbirth",
        "symptoms": ["Post-birth onset", "Fatigue", "Emotional exhaustion", "Withdrawal"],
    },
    "Psychotic": {
        "id": "Psychotic",
        "name": "Psychotic Depression",
        "short": "Psychotic",
        "description": "Depression with hallucinations or delusions",
        "symptoms": ["Severe depression", "Disorganized speech patterns", "Flat affect"],
    },
    "Situational": {
        "id": "Situational",
        "name": "Situational / Reactive Depression",
        "short": "Situational",
        "description": "Triggered by stressful life events",
        "symptoms": ["Stress-triggered", "Emotional reactivity", "Event-linked onset"],
    },
}

# Acoustic profile: (feature_key, ideal_value, tolerance, weight)
# ideal_value is the target; score falls off with distance from ideal
ACOUSTIC_PROFILES = {
    "MDD": [
        ("energy_mean", 0.008, 0.012, 1.4),
        ("pitch_std_hz", 25.0, 40.0, 1.2),
        ("pause_ratio", 0.42, 0.2, 1.3),
        ("speech_rate", 0.48, 0.2, 1.1),
        ("zero_crossing_rate", 0.04, 0.03, 0.8),
    ],
    "Dysthymia": [
        ("energy_mean", 0.012, 0.01, 1.0),
        ("pitch_std_hz", 40.0, 35.0, 0.9),
        ("pause_ratio", 0.32, 0.18, 1.0),
        ("speech_rate", 0.58, 0.18, 0.9),
        ("spectral_centroid", 1400.0, 600.0, 0.7),
    ],
    "Bipolar": [
        ("pitch_std_hz", 90.0, 70.0, 1.3),
        ("energy_std", 0.012, 0.01, 1.2),
        ("speech_rate", 0.55, 0.25, 0.8),
        ("energy_mean", 0.015, 0.012, 0.7),
        ("zero_crossing_rate", 0.07, 0.04, 0.9),
    ],
    "SAD": [
        ("energy_mean", 0.007, 0.01, 1.2),
        ("pitch_mean_hz", 130.0, 80.0, 1.0),
        ("speech_rate", 0.45, 0.2, 1.1),
        ("pause_ratio", 0.45, 0.2, 1.0),
        ("spectral_centroid", 1200.0, 500.0, 0.8),
    ],
    "Postpartum": [
        ("energy_mean", 0.006, 0.008, 1.4),
        ("pause_ratio", 0.48, 0.2, 1.2),
        ("speech_rate", 0.42, 0.18, 1.1),
        ("pitch_std_hz", 30.0, 35.0, 0.9),
        ("zero_crossing_rate", 0.035, 0.025, 0.8),
    ],
    "Psychotic": [
        ("pitch_std_hz", 15.0, 20.0, 1.3),
        ("energy_std", 0.004, 0.006, 1.1),
        ("speech_rate", 0.4, 0.2, 1.0),
        ("pause_ratio", 0.5, 0.2, 1.0),
        ("zero_crossing_rate", 0.03, 0.02, 0.9),
    ],
    "Situational": [
        ("pitch_std_hz", 100.0, 80.0, 1.2),
        ("energy_std", 0.015, 0.012, 1.1),
        ("speech_rate", 0.62, 0.2, 0.9),
        ("pause_ratio", 0.28, 0.15, 0.7),
        ("spectral_bandwidth", 1900.0, 700.0, 0.8),
    ],
}

CONTEXT_BOOSTS = {
    "chronic": {"Dysthymia": 0.25},
    "recent_stress": {"Situational": 0.30},
    "postpartum": {"Postpartum": 0.35},
    "seasonal": {"SAD": 0.30},
    "mood_swings": {"Bipolar": 0.25},
}


def _feature_score(actual: float, ideal: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    dist = abs(actual - ideal) / tolerance
    return float(np.exp(-0.5 * dist ** 2))


def _aggregate_features(segment_details: list[dict], depressed_only: bool = True) -> dict:
    segs = segment_details
    if depressed_only:
        segs = [s for s in segment_details if s["prob"] >= 0.5]
    if not segs:
        segs = segment_details

    keys = segs[0]["features"].keys()
    return {k: float(np.mean([s["features"][k] for s in segs])) for k in keys}


def classify_subtype(
    segment_details: list[dict],
    prediction: str,
    depression_prob: float,
    context=None,
) -> dict:
    """
    Classify depression subtype profile from acoustic patterns.
    Returns ranked subtypes with scores and explanations.
    """
    context = context or {}

    if prediction != "Depressed":
        return {
            "primary_type": None,
            "primary_name": "Not Applicable",
            "confidence": 0.0,
            "applicable": False,
            "message": (
                "Depression subtype classification applies only when depression is detected. "
                "Your voice was classified as Non-Depressed."
            ),
            "rankings": [],
            "matched_symptoms": [],
            "disclaimer": _disclaimer(),
        }

    agg = _aggregate_features(segment_details, depressed_only=True)
    raw_scores = {}

    for subtype_id, profile in ACOUSTIC_PROFILES.items():
        score = 0.0
        weight_sum = 0.0
        for feat_key, ideal, tol, weight in profile:
            actual = agg.get(feat_key, 0.0)
            score += _feature_score(actual, ideal, tol) * weight
            weight_sum += weight
        raw_scores[subtype_id] = score / max(weight_sum, 1e-8)

    for ctx_key, boosts in CONTEXT_BOOSTS.items():
        if context.get(ctx_key):
            for subtype_id, boost in boosts.items():
                raw_scores[subtype_id] = raw_scores.get(subtype_id, 0) + boost

    # Weight by overall depression probability
    for sid in raw_scores:
        raw_scores[sid] *= 0.6 + 0.4 * depression_prob

    scores = np.array(list(raw_scores.values()))
    ids = list(raw_scores.keys())
    exp_scores = np.exp(scores - scores.max())
    probs = exp_scores / exp_scores.sum()

    rankings = []
    for i, sid in enumerate(ids):
        info = SUBTYPES[sid]
        rankings.append({
            "id": sid,
            "name": info["name"],
            "short": info["short"],
            "description": info["description"],
            "symptoms": info["symptoms"],
            "score": round(float(raw_scores[sid]), 4),
            "probability": round(float(probs[i]), 4),
        })
    rankings.sort(key=lambda x: x["probability"], reverse=True)

    primary = rankings[0]
    matched = _match_symptoms(agg, primary["id"])

    return {
        "primary_type": primary["id"],
        "primary_name": primary["name"],
        "primary_description": primary["description"],
        "confidence": primary["probability"],
        "applicable": True,
        "message": (
            f"Based on acoustic patterns in your voice, the profile most similar to "
            f"**{primary['name']}** ({primary['probability']:.0%} match). "
            f"This reflects speech symptom patterns, not a clinical diagnosis."
        ),
        "rankings": rankings,
        "matched_symptoms": matched,
        "acoustic_summary": _acoustic_summary(agg),
        "disclaimer": _disclaimer(),
    }


def _match_symptoms(features: dict, subtype_id: str) -> list[str]:
    """List observed acoustic signs that align with the matched subtype."""
    signs = []
    energy = features.get("energy_mean", 0)
    pitch_std = features.get("pitch_std_hz", 0)
    pause = features.get("pause_ratio", 0)
    speech = features.get("speech_rate", 0)
    energy_std = features.get("energy_std", 0)

    if energy < 0.012:
        signs.append("Low vocal energy (fatigue indicator)")
    if pitch_std < 35:
        signs.append("Monotone speech (reduced emotional expression)")
    if pause > 0.38:
        signs.append("Frequent pauses (psychomotor slowing)")
    if speech < 0.52:
        signs.append("Slow speech rate")
    if pitch_std > 75:
        signs.append("High pitch variability (emotional fluctuation)")
    if energy_std > 0.012:
        signs.append("Variable vocal energy (mood instability pattern)")
    if pitch_std < 20:
        signs.append("Very flat affect (severe reduction in expressiveness)")

    subtype_signs = {
        "MDD": ["Low vocal energy (fatigue indicator)", "Monotone speech (reduced emotional expression)"],
        "Dysthymia": ["Mild but persistent low energy patterns"],
        "Bipolar": ["Variable vocal energy (mood instability pattern)", "High pitch variability (emotional fluctuation)"],
        "SAD": ["Low vocal energy (fatigue indicator)", "Slow speech rate"],
        "Postpartum": ["Low vocal energy (fatigue indicator)", "Frequent pauses (psychomotor slowing)"],
        "Psychotic": ["Very flat affect (severe reduction in expressiveness)", "Monotone speech (reduced emotional expression)"],
        "Situational": ["High pitch variability (emotional fluctuation)", "Variable vocal energy (mood instability pattern)"],
    }
    aligned = [s for s in signs if s in subtype_signs.get(subtype_id, [])]
    return aligned or signs[:3] or ["General depression-linked acoustic patterns"]


def _acoustic_summary(features: dict) -> dict:
    return {
        "energy_level": "Low" if features.get("energy_mean", 0) < 0.012 else "Normal",
        "pitch_variation": "Flat" if features.get("pitch_std_hz", 0) < 35 else "Variable",
        "speech_pace": "Slow" if features.get("speech_rate", 0) < 0.52 else "Normal",
        "pause_frequency": "High" if features.get("pause_ratio", 0) > 0.38 else "Normal",
    }


def _disclaimer() -> str:
    return (
        "Subtype profiles are estimated from speech acoustics only. "
        "Clinical diagnosis requires a qualified mental health professional, "
        "patient history, and DSM-5 assessment. Do not use this as medical diagnosis."
    )
