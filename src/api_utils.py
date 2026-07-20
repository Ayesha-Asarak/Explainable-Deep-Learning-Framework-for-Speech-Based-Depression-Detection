"""Serialize prediction results for the web API."""

import base64
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .explain import plot_grad_cam, plot_full_spectrogram, plot_timeline_chart


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f1419")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


def result_to_json(result: dict) -> dict:
    importance = result["feature_importance"]
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    fig_spec = plot_full_spectrogram(
        result["full_mel"],
        result["full_times"],
        result["highlight_regions"],
        result["prediction"],
    )
    spectrogram_b64 = fig_to_base64(fig_spec)

    fig_cam = plot_grad_cam(
        result["best_spectrogram"],
        result["grad_cam"],
        segment_start_sec=result["key_segment_start_sec"],
        title=f"Grad-CAM at {_fmt(result['key_segment_start_sec'])} – {_fmt(result['key_segment_end_sec'])}",
    )
    grad_cam_b64 = fig_to_base64(fig_cam)

    fig_timeline = plot_timeline_chart(result["segment_details"], result["prediction"])
    timeline_b64 = fig_to_base64(fig_timeline)

    fig_bar, ax = plt.subplots(figsize=(8, 5), facecolor="#0f1419")
    ax.set_facecolor("#1a2332")
    names = [k.replace("_", " ").title() for k, _ in sorted_feats]
    vals = [v for _, v in sorted_feats]
    color = "#e85d6f" if result["prediction"] == "Depressed" else "#3ecf8e"
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=color, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="#a8b8d0", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", color="#a8b8d0")
    ax.set_title("Top Acoustic Features (SHAP)", color="#e8edf5", fontsize=11)
    ax.tick_params(colors="#a8b8d0")
    for spine in ax.spines.values():
        spine.set_color("#2d3a4f")
    feat_chart_b64 = fig_to_base64(fig_bar)

    subtype = result.get("subtype", {})
    subtype_chart_b64 = None
    if subtype.get("rankings"):
        fig_sub, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1419")
        ax.set_facecolor("#1a2332")
        names = [r["short"] for r in subtype["rankings"]]
        probs = [r["probability"] * 100 for r in subtype["rankings"]]
        colors = ["#e85d6f" if i == 0 else "#5b8def" for i in range(len(names))]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, probs, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, color="#a8b8d0", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Profile Match (%)", color="#a8b8d0")
        ax.set_title("Depression Type Profile Classification", color="#e8edf5", fontsize=11)
        ax.set_xlim(0, max(probs) * 1.15 if probs else 100)
        ax.tick_params(colors="#a8b8d0")
        for spine in ax.spines.values():
            spine.set_color("#2d3a4f")
        subtype_chart_b64 = fig_to_base64(fig_sub)

    acoustic = result["acoustic_features"]
    return {
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 4),
        "probability_depressed": round(result["probability_depressed"], 4),
        "audio_duration_sec": round(result["audio_duration_sec"], 1),
        "n_segments": result["n_segments"],
        "prediction_reason": result["prediction_reason"],
        "key_segment": {
            "start_sec": result["key_segment_start_sec"],
            "end_sec": result["key_segment_end_sec"],
        },
        "timeline_explanations": result["timeline_explanations"],
        "segment_probabilities": [round(p, 4) for p in result["segment_probabilities"]],
        "segment_explanations": result["segment_explanations"],
        "summary": result["summary"],
        "acoustic_features": {k: round(v, 4) for k, v in acoustic.items()},
        "feature_importance": [
            {"name": k.replace("_", " ").title(), "key": k, "value": round(v, 4)}
            for k, v in sorted_feats
        ],
        "subtype": {
            "primary_type": subtype.get("primary_type"),
            "primary_name": subtype.get("primary_name"),
            "primary_description": subtype.get("primary_description", ""),
            "confidence": round(subtype.get("confidence", 0), 4),
            "applicable": subtype.get("applicable", False),
            "message": subtype.get("message", ""),
            "rankings": subtype.get("rankings", []),
            "matched_symptoms": subtype.get("matched_symptoms", []),
            "acoustic_summary": subtype.get("acoustic_summary", {}),
            "disclaimer": subtype.get("disclaimer", ""),
        },
        "charts": {
            "spectrogram": spectrogram_b64,
            "grad_cam": grad_cam_b64,
            "timeline": timeline_b64,
            "features": feat_chart_b64,
            "subtype": subtype_chart_b64,
        },
    }


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s}s" if m else f"{s}s"
