#!/usr/bin/env python3
"""Streamlit app: upload voice recording → depression prediction + explanation."""

import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import MODEL_PATH, ROOT_DIR
from src.predict import DepressionPredictor
from src.explain import plot_grad_cam

st.set_page_config(
    page_title="Speech Depression Detection",
    page_icon="🎙️",
    layout="wide",
)

st.title("Explainable Deep Learning for Speech-Based Depression Detection")
st.markdown(
    """
    Upload a voice recording to analyze speech patterns and receive a **depressed / non-depressed**
    prediction with **explainable AI** insights (Grad-CAM + acoustic feature importance).

    > ⚠️ **Disclaimer:** This system is a research decision-support tool only.
    > It does **not** replace professional clinical diagnosis.
    """
)

@st.cache_resource
def load_predictor():
    if not MODEL_PATH.exists():
        return None
    return DepressionPredictor()


predictor = load_predictor()

if predictor is None:
    st.error(
        "Model not found. Please run training first:\n\n"
        "```bash\npip install -r requirements.txt\npython train.py\n```"
    )
    st.stop()

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload a voice recording (.wav, .mp3, .flac, .ogg)",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
    )

with col_info:
    st.info(
        "**Dataset:** DAIC-WOZ style interviews\n\n"
        "**Model:** CNN on mel spectrograms\n\n"
        "**Explainability:** Grad-CAM + SHAP features"
    )

if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    with st.spinner("Analyzing speech patterns..."):
        try:
            result = predictor.predict(tmp_path)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    pred = result["prediction"]
    conf = result["confidence"]
    prob = result["probability_depressed"]

    st.divider()

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        if pred == "Depressed":
            st.markdown(f"### 🔴 Result: **{pred}**")
        else:
            st.markdown(f"### 🟢 Result: **{pred}**")
    with res_col2:
        st.metric("Confidence", f"{conf:.1%}")
    with res_col3:
        st.metric("Depression Probability", f"{prob:.1%}")

    st.progress(min(max(prob, 0.0), 1.0))

    tab_summary, tab_viz, tab_features = st.tabs(
        ["Explanation Summary", "Grad-CAM Visualization", "Acoustic Features"]
    )

    with tab_summary:
        st.markdown(result["summary"])

        st.subheader("Segment-level analysis")
        st.caption(
            f"Recording duration: {result['audio_duration_sec']:.1f}s | "
            f"Analyzed {result['n_segments']} segments of 3 seconds each"
        )
        seg_probs = result["segment_probabilities"]
        fig_seg, ax = plt.subplots(figsize=(10, 3))
        ax.plot(seg_probs, marker="o", color="#3498db")
        ax.axhline(0.5, color="gray", linestyle="--", label="Decision threshold")
        ax.set_xlabel("Segment index")
        ax.set_ylabel("P(Depressed)")
        ax.set_title("Depression probability across recording")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig_seg)
        plt.close(fig_seg)

        for exp in result["segment_explanations"]:
            st.markdown(f"- {exp}")

    with tab_viz:
        st.subheader("Grad-CAM: Which time-frequency regions influenced the prediction?")
        fig_cam = plot_grad_cam(result["best_spectrogram"], result["grad_cam"])
        st.pyplot(fig_cam)
        plt.close(fig_cam)
        st.caption(
            "Red/yellow regions indicate mel-frequency bands and time frames "
            "that most strongly influenced the model's decision."
        )

    with tab_features:
        st.subheader("Acoustic feature importance (SHAP)")
        importance = result["feature_importance"]
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        feat_names = [k.replace("_", " ").title() for k, _ in sorted_feats[:10]]
        feat_vals = [v for _, v in sorted_feats[:10]]

        fig_bar, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(feat_names))
        colors = ["#e74c3c" if pred == "Depressed" else "#27ae60"] * len(feat_names)
        ax.barh(y_pos, feat_vals, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top 10 contributing acoustic features")
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.subheader("Raw acoustic measurements (representative segment)")
        acoustic = result["acoustic_features"]
        ac_cols = st.columns(3)
        display_feats = [
            ("Pitch Mean (Hz)", "pitch_mean_hz"),
            ("Pitch Std (Hz)", "pitch_std_hz"),
            ("Energy Mean", "energy_mean"),
            ("Pause Ratio", "pause_ratio"),
            ("Speech Rate", "speech_rate"),
            ("Spectral Centroid", "spectral_centroid"),
        ]
        for i, (label, key) in enumerate(display_feats):
            with ac_cols[i % 3]:
                val = acoustic.get(key, 0)
                if key in ("pause_ratio", "speech_rate"):
                    st.metric(label, f"{val:.0%}")
                elif "hz" in key.lower():
                    st.metric(label, f"{val:.1f}")
                else:
                    st.metric(label, f"{val:.4f}")

else:
    st.markdown("---")
    st.subheader("How it works")
    st.markdown(
        """
        1. **Preprocessing** — noise trimming, normalization, 3-second segmentation
        2. **Feature extraction** — MFCCs, mel spectrograms, pitch, energy, pauses
        3. **Deep learning** — CNN learns spatial-temporal patterns from spectrograms
        4. **Explainability** — Grad-CAM highlights important time regions; SHAP ranks acoustic features
        5. **Output** — Binary classification (Depressed / Non-Depressed) with confidence and explanations
        """
    )

    sample_dirs = [
        ROOT_DIR / "depressed",
        ROOT_DIR / "non depressed",
    ]
    st.subheader("Sample recordings in your dataset")
    for d in sample_dirs:
        if d.exists():
            wavs = list(d.glob("*/*_AUDIO.wav"))[:2]
            label = "Depressed" if "non" not in d.name.lower() else "Non-Depressed"
            for w in wavs:
                st.text(f"[{label}] {w.parent.name}")
