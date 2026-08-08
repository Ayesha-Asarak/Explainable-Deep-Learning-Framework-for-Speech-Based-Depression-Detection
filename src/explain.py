"""Explainability: Grad-CAM, spectrograms, and time-stamped voice explanations."""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")

from .config import FEATURE_NAMES, SAMPLE_RATE, HOP_LENGTH, SEGMENT_DURATION
from .model import DepressionCNN


def compute_grad_cam(model: DepressionCNN, spectrogram: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(spectrogram).unsqueeze(0)
    x.requires_grad_(True)
    logits = model(x)
    model.zero_grad()
    logits.backward()

    activations = model._grad_cam_activations
    gradients = model._grad_cam_gradients
    if activations is None or gradients is None:
        return np.zeros(spectrogram.shape[-2:], dtype=np.float32)

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=spectrogram.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def _format_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def explain_voice_at_time(
    start_sec: float,
    end_sec: float,
    prob: float,
    features: dict,
    prediction: str,
) -> str:
    """One sentence explaining what was detected at an exact time in the voice."""
    time_str = f"{_format_time(start_sec)} – {_format_time(end_sec)}"
    cues = []

    pitch_std = features.get("pitch_std_hz", 0)
    pitch_mean = features.get("pitch_mean_hz", 0)
    energy = features.get("energy_mean", 0)
    pause = features.get("pause_ratio", 0)
    speech = features.get("speech_rate", 0)

    if pitch_std < 30:
        cues.append("flat/monotone pitch")
    if pitch_mean < 150 and pitch_mean > 0:
        cues.append("low vocal pitch")
    if energy < 0.015:
        cues.append("low voice energy (quiet speech)")
    if pause > 0.4:
        cues.append(f"long pauses ({pause:.0%} silence)")
    if speech < 0.5:
        cues.append(f"slow speech rate ({speech:.0%} active)")
    if pitch_std > 80:
        cues.append("high pitch variation")
    if energy > 0.04:
        cues.append("strong vocal energy")
    if speech > 0.75 and pause < 0.25:
        cues.append("active, fluent speech")

    if not cues:
        cues.append("typical acoustic rhythm and tone")

    cue_text = ", ".join(cues)
    score = prob if prediction == "Depressed" else (1 - prob)
    direction = "toward Depressed" if prob >= 0.5 else "toward Non-Depressed"

    return (
        f"At {time_str} in your voice: detected {cue_text}. "
        f"This segment scored {prob:.0%} depressed ({direction}, weight {score:.0%})."
    )


def build_timeline_explanations(
    segment_details: list[dict],
    prediction: str,
    top_n: int = 6,
) -> list[dict]:
    """
    Build time-stamped explanations for the most influential voice regions.
    segment_details: list of {start_sec, end_sec, prob, features}
    """
    is_depressed = prediction == "Depressed"

    if is_depressed:
        ranked = sorted(segment_details, key=lambda s: s["prob"], reverse=True)
        supporting = [s for s in ranked if s["prob"] >= 0.5][:top_n]
        opposing = sorted(
            [s for s in segment_details if s["prob"] < 0.5],
            key=lambda s: s["prob"],
        )[:2]
    else:
        ranked = sorted(segment_details, key=lambda s: s["prob"])
        supporting = [s for s in ranked if s["prob"] < 0.5][:top_n]
        opposing = sorted(
            [s for s in segment_details if s["prob"] >= 0.5],
            key=lambda s: s["prob"],
            reverse=True,
        )[:2]

    explanations = []
    for seg in supporting:
        explanations.append({
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "time_label": f"{_format_time(seg['start_sec'])} – {_format_time(seg['end_sec'])}",
            "probability": round(seg["prob"], 4),
            "role": "supporting",
            "text": explain_voice_at_time(
                seg["start_sec"], seg["end_sec"], seg["prob"], seg["features"], prediction
            ),
            "cues": _extract_cues(seg["features"]),
        })

    for seg in opposing:
        explanations.append({
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "time_label": f"{_format_time(seg['start_sec'])} – {_format_time(seg['end_sec'])}",
            "probability": round(seg["prob"], 4),
            "role": "opposing",
            "text": explain_voice_at_time(
                seg["start_sec"], seg["end_sec"], seg["prob"], seg["features"], prediction
            ),
            "cues": _extract_cues(seg["features"]),
        })

    explanations.sort(key=lambda e: e["start_sec"])
    return explanations


def _extract_cues(features: dict) -> list[str]:
    cues = []
    if features.get("pitch_std_hz", 0) < 30:
        cues.append("Monotone pitch")
    if features.get("energy_mean", 0) < 0.015:
        cues.append("Low energy")
    if features.get("pause_ratio", 0) > 0.4:
        cues.append("Long pauses")
    if features.get("speech_rate", 0) < 0.5:
        cues.append("Slow speech")
    if features.get("pitch_std_hz", 0) > 80:
        cues.append("Varied pitch")
    if features.get("energy_mean", 0) > 0.04:
        cues.append("Strong energy")
    if features.get("speech_rate", 0) > 0.75:
        cues.append("Fluent speech")
    return cues or ["Normal patterns"]


def build_prediction_reason(
    prediction: str,
    confidence: float,
    timeline: list[dict],
    segment_details: list[dict],
) -> str:
    """Plain-language summary citing exact places in the voice."""
    n_depressed = sum(1 for s in segment_details if s["prob"] >= 0.5)
    n_total = len(segment_details)
    pct = n_depressed / max(n_total, 1)

    supporting = [e for e in timeline if e["role"] == "supporting"][:3]
    time_refs = ", ".join(e["time_label"] for e in supporting)

    if prediction == "Depressed":
        return (
            f"We classified your voice as **Depressed** ({confidence:.0%} confidence) because "
            f"{n_depressed} of {n_total} segments ({pct:.0%}) showed depression-linked patterns. "
            f"The strongest evidence was at **{time_refs}** — where the model found "
            f"reduced energy, altered pitch, or longer pauses in your speech."
        )
    return (
        f"We classified your voice as **Non-Depressed** ({confidence:.0%} confidence) because "
        f"{n_total - n_depressed} of {n_total} segments ({1 - pct:.0%}) showed healthy speech patterns. "
        f"The clearest normal speech was at **{time_refs}** — with active rhythm and typical vocal energy."
    )


def plot_full_spectrogram(
    log_mel: np.ndarray,
    times: np.ndarray,
    highlighted_regions: list[dict],
    prediction: str,
) -> plt.Figure:
    """Full recording spectrogram with highlighted voice regions in seconds."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f1419")
    ax.set_facecolor("#1a2332")

    im = ax.imshow(
        log_mel, aspect="auto", origin="lower", cmap="magma",
        extent=[times[0], times[-1], 0, log_mel.shape[0]],
    )
    ax.set_xlabel("Time in recording (seconds)", color="#a8b8d0", fontsize=11)
    ax.set_ylabel("Mel frequency bands", color="#a8b8d0", fontsize=11)
    ax.set_title(
        "Full Voice Spectrogram — Highlighted regions show where the model looked",
        color="#e8edf5", fontsize=12, pad=12,
    )
    ax.tick_params(colors="#a8b8d0")
    cbar = plt.colorbar(im, ax=ax, fraction=0.02)
    cbar.ax.yaxis.set_tick_params(color="#a8b8d0")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#a8b8d0")

    for region in highlighted_regions:
        start = region["start_sec"]
        end = region["end_sec"]
        prob = region.get("prob", 0.5)
        role = region.get("role", "supporting")

        if prediction == "Depressed":
            color = "#e85d6f" if prob >= 0.5 else "#3ecf8e"
        else:
            color = "#3ecf8e" if prob < 0.5 else "#e85d6f"

        alpha = 0.25 + 0.35 * abs(prob - 0.5)
        rect = mpatches.Rectangle(
            (start, 0), end - start, log_mel.shape[0],
            linewidth=2, edgecolor=color, facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)
        mid = (start + end) / 2
        ax.annotate(
            f"{prob:.0%}",
            xy=(mid, log_mel.shape[0] * 0.92),
            ha="center", fontsize=8, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85),
        )

    for spine in ax.spines.values():
        spine.set_color("#2d3a4f")

    legend_patches = [
        mpatches.Patch(color="#e85d6f", alpha=0.6, label="Depression signal"),
        mpatches.Patch(color="#3ecf8e", alpha=0.6, label="Non-depression signal"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", facecolor="#1a2332",
              edgecolor="#2d3a4f", labelcolor="#a8b8d0", fontsize=9)

    plt.tight_layout()
    return fig


def build_segment_occlusion_map(
    full_mel: np.ndarray,
    times: np.ndarray,
    segment_details: list[dict],
) -> np.ndarray:
    """
    Project leave-one-segment-out importance onto the full-recording time axis.

    Overlapping segments use the max importance at each frame. A mild energy
    weighting adds vertical structure for display only; the temporal pattern
    is the faithful attribution signal.
    """
    n_mels, n_frames = full_mel.shape
    if n_frames == 0 or len(times) == 0:
        return np.zeros((n_mels, max(n_frames, 1)), dtype=np.float32)

    frame_imp = np.zeros(n_frames, dtype=np.float32)
    for segment in segment_details:
        importance = float(segment.get("occlusion_importance", 0.0))
        start = float(segment["start_sec"])
        end = float(segment["end_sec"])
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            # Fallback for last segment / floating-point edge cases.
            mask = (times >= start) & (times <= end)
        if np.any(mask):
            frame_imp[mask] = np.maximum(frame_imp[mask], importance)

    # Relative mel energy per frame (0–1) so the overlay is not a solid block.
    energy = full_mel - full_mel.min(axis=0, keepdims=True)
    energy = energy / (energy.max(axis=0, keepdims=True) + 1e-8)
    cam = frame_imp[np.newaxis, :] * (0.35 + 0.65 * energy)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def plot_occlusion_map(
    full_mel: np.ndarray,
    times: np.ndarray,
    occlusion_map: np.ndarray,
    segment_details: list[dict],
    key_start_sec: float = 0.0,
    key_end_sec: float = 0.0,
    title: str = "Leave-one-segment-out occlusion importance",
) -> plt.Figure:
    """
    Full-recording occlusion view: spectrogram overlay + importance over time.
    """
    t0 = float(times[0]) if len(times) else 0.0
    t1 = float(times[-1]) if len(times) else 1.0
    n_mels = full_mel.shape[0]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        facecolor="#0f1419",
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.32},
        sharex=True,
        constrained_layout=True,
    )
    for ax in axes:
        ax.set_facecolor("#1a2332")
        ax.tick_params(colors="#a8b8d0")
        for spine in ax.spines.values():
            spine.set_color("#2d3a4f")

    axes[0].imshow(
        full_mel,
        aspect="auto",
        origin="lower",
        cmap="gray",
        alpha=0.55,
        extent=[t0, t1, 0, n_mels],
    )
    im = axes[0].imshow(
        occlusion_map,
        aspect="auto",
        origin="lower",
        cmap="jet",
        alpha=0.7,
        extent=[t0, t1, 0, n_mels],
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_ylabel("Mel bands", color="#a8b8d0")
    axes[0].set_title(title, color="#e8edf5", fontsize=12, pad=10)
    cbar = plt.colorbar(im, ax=axes[0], fraction=0.02, pad=0.01)
    cbar.set_label("Occlusion importance", color="#a8b8d0")
    cbar.ax.yaxis.set_tick_params(color="#a8b8d0")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#a8b8d0")

    if key_end_sec > key_start_sec:
        axes[0].axvspan(
            key_start_sec,
            key_end_sec,
            color="white",
            alpha=0.12,
            linewidth=0,
        )
        axes[0].axvline(key_start_sec, color="white", linestyle="--", linewidth=1.2, alpha=0.85)
        axes[0].axvline(key_end_sec, color="white", linestyle="--", linewidth=1.2, alpha=0.85)

    # Max importance over time (handles 50% segment overlap cleanly).
    if segment_details:
        starts = [float(s["start_sec"]) for s in segment_details]
        ends = [float(s["end_sec"]) for s in segment_details]
        vals = [float(s.get("occlusion_importance", 0.0)) for s in segment_details]
        n_pts = max(200, int(max(t1 - t0, 1.0) * 20))
        t_dense = np.linspace(t0, t1, n_pts)
        y_dense = np.zeros_like(t_dense)
        for start, end, val in zip(starts, ends, vals):
            mask = (t_dense >= start) & (t_dense < end)
            if not np.any(mask):
                mask = (t_dense >= start) & (t_dense <= end)
            y_dense[mask] = np.maximum(y_dense[mask], val)
        axes[1].fill_between(t_dense, y_dense, color="#5b8def", alpha=0.35)
        axes[1].plot(t_dense, y_dense, color="#8eb6ff", linewidth=1.8)
        peak_idx = int(np.argmax(vals))
        peak_t = 0.5 * (starts[peak_idx] + ends[peak_idx])
        peak_v = vals[peak_idx]
        axes[1].scatter([peak_t], [peak_v], color="#e85d6f", zorder=5, s=36)
        axes[1].annotate(
            f"Peak {_format_time(peak_t)} ({peak_v:.0%})",
            xy=(peak_t, peak_v),
            xytext=(8, 8),
            textcoords="offset points",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#e85d6f", alpha=0.85),
        )
    else:
        axes[1].plot([t0, t1], [0, 0], color="#5b8def")

    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_xlim(t0, t1)
    axes[1].set_xlabel("Time in recording (seconds)", color="#a8b8d0")
    axes[1].set_ylabel("Segment importance", color="#a8b8d0")
    axes[1].set_title(
        "How much each voice segment changed the participant-level prediction",
        color="#e8edf5",
        fontsize=11,
    )

    return fig


def plot_grad_cam(
    spectrogram: np.ndarray,
    cam: np.ndarray,
    segment_start_sec: float = 0.0,
    title: str = "Grad-CAM at Key Voice Region",
) -> plt.Figure:
    """Grad-CAM with time axis in seconds (absolute position in recording)."""
    spec_2d = spectrogram[0]
    n_frames = spec_2d.shape[1]
    frame_times = segment_start_sec + np.arange(n_frames) * (SEGMENT_DURATION / n_frames)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0f1419")
    for ax in axes:
        ax.set_facecolor("#1a2332")
        ax.tick_params(colors="#a8b8d0")
        for spine in ax.spines.values():
            spine.set_color("#2d3a4f")

    im0 = axes[0].imshow(
        spec_2d, aspect="auto", origin="lower", cmap="magma",
        extent=[frame_times[0], frame_times[-1], 0, spec_2d.shape[0]],
    )
    axes[0].set_title("Mel Spectrogram (this voice segment)", color="#e8edf5")
    axes[0].set_xlabel("Time in recording (seconds)", color="#a8b8d0")
    axes[0].set_ylabel("Mel bands", color="#a8b8d0")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046)
    plt.setp(plt.getp(cbar0.ax.axes, "yticklabels"), color="#a8b8d0")

    axes[1].imshow(
        spec_2d, aspect="auto", origin="lower", cmap="gray", alpha=0.5,
        extent=[frame_times[0], frame_times[-1], 0, spec_2d.shape[0]],
    )
    im1 = axes[1].imshow(
        cam, aspect="auto", origin="lower", cmap="jet", alpha=0.65,
        extent=[frame_times[0], frame_times[-1], 0, spec_2d.shape[0]],
    )
    axes[1].set_title(title, color="#e8edf5")
    axes[1].set_xlabel("Time in recording (seconds)", color="#a8b8d0")
    axes[1].set_ylabel("Mel bands", color="#a8b8d0")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.setp(plt.getp(cbar1.ax.axes, "yticklabels"), color="#a8b8d0")

    frame_importance = cam.mean(axis=0)
    peak_frame = int(np.argmax(frame_importance))
    peak_time = frame_times[peak_frame]
    axes[1].axvline(peak_time, color="white", linestyle="--", linewidth=1.5, alpha=0.9)
    axes[1].annotate(
        f"Peak at {_format_time(peak_time)}",
        xy=(peak_time, spec_2d.shape[0] * 0.5),
        color="white", fontsize=9, ha="center",
        bbox=dict(boxstyle="round", facecolor="#e85d6f", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def compute_probability_uncertainty(segment_probs: list[float], threshold: float = 0.5) -> dict:
    """
    Decision uses mean vs 0.5 threshold.
    Uncertainty uses mean ± 1 standard deviation across segments.
    """
    probs = np.asarray(segment_probs, dtype=np.float64)
    if len(probs) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "threshold": threshold,
            "stability": "insufficient_data",
            "stability_label": "Insufficient data",
            "message": "Not enough segments to estimate uncertainty.",
        }

    mean = float(np.mean(probs))
    std = float(np.std(probs, ddof=0)) if len(probs) > 1 else 0.0
    lower = float(max(0.0, mean - std))
    upper = float(min(1.0, mean + std))

    if lower >= threshold:
        stability = "stable_depressed"
        stability_label = "Stable Depressed"
        message = (
            f"Mean {mean:.0%} with ±1 SD band [{lower:.0%} – {upper:.0%}] "
            f"stays entirely above the {threshold:.0%} threshold."
        )
    elif upper <= threshold:
        stability = "stable_non_depressed"
        stability_label = "Stable Non-Depressed"
        message = (
            f"Mean {mean:.0%} with ±1 SD band [{lower:.0%} – {upper:.0%}] "
            f"stays entirely below the {threshold:.0%} threshold."
        )
    else:
        stability = "uncertain"
        stability_label = "Uncertain / Needs Review"
        message = (
            f"Mean {mean:.0%} with ±1 SD band [{lower:.0%} – {upper:.0%}] "
            f"crosses the {threshold:.0%} threshold — prediction is less stable."
        )

    return {
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
        "threshold": threshold,
        "stability": stability,
        "stability_label": stability_label,
        "message": message,
    }


def plot_timeline_chart(
    segment_details: list[dict],
    prediction: str,
    uncertainty: dict = None,
) -> plt.Figure:
    """Bar chart of depression probability vs time, with 0.5 threshold and mean ± 1 SD."""
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="#0f1419")
    ax.set_facecolor("#1a2332")

    starts = [s["start_sec"] for s in segment_details]
    probs = [s["prob"] for s in segment_details]
    widths = [s["end_sec"] - s["start_sec"] for s in segment_details]
    colors = ["#e85d6f" if p >= 0.5 else "#3ecf8e" for p in probs]

    if uncertainty is None:
        uncertainty = compute_probability_uncertainty(probs)

    mean = uncertainty["mean"]
    lower = uncertainty["lower"]
    upper = uncertainty["upper"]
    std = uncertainty["std"]

    # Mean ± 1 SD uncertainty band
    ax.axhspan(
        lower, upper,
        color="#5b8def", alpha=0.18,
        label=f"Mean ± 1 SD  [{lower:.0%} – {upper:.0%}]",
        zorder=1,
    )
    ax.axhline(
        mean, color="#5b8def", linestyle="-", linewidth=2,
        label=f"Mean μ = {mean:.0%}  (σ = {std:.0%})",
        zorder=3,
    )
    ax.axhline(
        0.5, color="#f0c040", linestyle="--", linewidth=1.5,
        label="Decision threshold = 0.5",
        zorder=3,
    )

    ax.bar(
        starts, probs, width=widths, align="edge",
        color=colors, alpha=0.85, edgecolor="#2d3a4f", zorder=2,
    )

    ax.set_xlabel("Time in voice recording (seconds)", color="#a8b8d0")
    ax.set_ylabel("Depression probability", color="#a8b8d0")
    title = (
        f"Voice Timeline — decision at 0.5 | uncertainty: {uncertainty['stability_label']}"
    )
    ax.set_title(title, color="#e8edf5", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.tick_params(colors="#a8b8d0")
    for spine in ax.spines.values():
        spine.set_color("#2d3a4f")
    ax.legend(
        facecolor="#1a2332", edgecolor="#2d3a4f", labelcolor="#a8b8d0",
        loc="upper right", fontsize=8,
    )

    plt.tight_layout()
    return fig


def compute_feature_importance(
    feature_model: torch.nn.Module,
    feature_vectors: np.ndarray,
    background: np.ndarray,
    max_samples: int = 50,
    fast: bool = True,
) -> dict:
    """Gradient importance by default for fast inference; set fast=False for SHAP."""
    if fast:
        return _gradient_feature_importance(feature_model, feature_vectors)

    try:
        import shap
    except ImportError:
        return _gradient_feature_importance(feature_model, feature_vectors)

    feature_model.eval()
    n_bg = min(10, len(background))
    n_explain = min(10, max_samples, len(feature_vectors))

    def model_fn(x):
        with torch.no_grad():
            return torch.sigmoid(feature_model(torch.from_numpy(x).float())).numpy()

    explainer = shap.KernelExplainer(model_fn, background[:n_bg])
    shap_values = explainer.shap_values(feature_vectors[:n_explain], nsamples=30)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {FEATURE_NAMES[i]: float(mean_abs[i]) for i in range(len(FEATURE_NAMES))}


def _gradient_feature_importance(feature_model, feature_vectors):
    feature_model.eval()
    x = torch.from_numpy(feature_vectors).float().requires_grad_(True)
    torch.sigmoid(feature_model(x)).sum().backward()
    grads = x.grad.abs().mean(dim=0).detach().numpy()
    return {FEATURE_NAMES[i]: float(grads[i]) for i in range(len(FEATURE_NAMES))}


def explain_segment_features(features: dict, label: str) -> list:
    return [explain_voice_at_time(0, 3, 0.5, features, label)]


def build_explanation_summary(
    prediction: str,
    confidence: float,
    feature_importance: dict,
    timeline: list[dict],
    reason_text: str,
) -> str:
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    top_lines = "\n".join(
        f"  • {name.replace('_', ' ').title()}: {score:.3f}"
        for name, score in top_features
    )
    time_lines = "\n".join(
        f"  • [{e['time_label']}] {e['text']}"
        for e in timeline if e["role"] == "supporting"
    )[:2000]

    return f"""## Prediction: {prediction}
**Confidence:** {confidence:.1%}

### Why {prediction}?
{reason_text}

### Exact places in your voice
{time_lines}

### Top acoustic features
{top_lines}

---
*Decision-support tool only — not a clinical diagnosis.*
"""
