const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const clearBtn = document.getElementById("clearBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const filePreview = document.getElementById("filePreview");
const dropContent = dropZone.querySelector(".drop-content");
const fileName = document.getElementById("fileName");
const resultsSection = document.getElementById("resultsSection");
const statusBadge = document.getElementById("statusBadge");
const toast = document.getElementById("toast");

let selectedFile = null;

const ACOUSTIC_DISPLAY = [
  ["Pitch Mean", "pitch_mean_hz", "hz"],
  ["Pitch Std", "pitch_std_hz", "hz"],
  ["Energy Mean", "energy_mean", "num"],
  ["Pause Ratio", "pause_ratio", "pct"],
  ["Speech Rate", "speech_rate", "pct"],
  ["Spectral Centroid", "spectral_centroid", "num"],
];

function showToast(msg, isError = false) {
  toast.textContent = msg;
  toast.classList.toggle("error", isError);
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 4000);
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.model_ready) {
      statusBadge.textContent = "● Model ready";
      statusBadge.className = "status-badge ready";
    } else {
      statusBadge.textContent = "○ Model not trained";
      statusBadge.className = "status-badge error";
      showToast("Run 'python train.py' to train the model first.", true);
    }
  } catch {
    statusBadge.textContent = "○ Server offline";
    statusBadge.className = "status-badge error";
  }
}

function setFile(file) {
  selectedFile = file;
  if (file) {
    fileName.textContent = file.name;
    filePreview.classList.remove("hidden");
    dropContent.classList.add("hidden");
    analyzeBtn.disabled = false;
  } else {
    filePreview.classList.add("hidden");
    dropContent.classList.remove("hidden");
    analyzeBtn.disabled = true;
    fileInput.value = "";
  }
}

browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
clearBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  setFile(null);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

function formatValue(val, type) {
  if (type === "pct") return `${(val * 100).toFixed(1)}%`;
  if (type === "hz") return `${val.toFixed(1)} Hz`;
  return val.toFixed(4);
}

function renderTimelineCards(timeline, prediction) {
  const container = document.getElementById("timelineCards");
  const isDepressed = prediction === "Depressed";

  container.innerHTML = timeline.map((item) => {
    const isHigh = item.probability >= 0.5;
    const cardClass = item.role === "supporting"
      ? `supporting ${isDepressed ? "depressed" : "healthy"}`
      : "opposing";
    const probClass = isHigh ? "high" : "low";
    const roleLabel = item.role === "supporting" ? "Key evidence" : "Counter-signal";
    const cues = (item.cues || []).map((c) => `<span class="cue-tag">${c}</span>`).join("");

    return `
      <div class="timeline-card ${cardClass}">
        <div class="timeline-card-header">
          <span class="time-badge">⏱ ${item.time_label}</span>
          <span class="prob-badge ${probClass}">${(item.probability * 100).toFixed(0)}% depressed</span>
          <span class="role-badge">${roleLabel}</span>
        </div>
        <p class="timeline-card-text">${item.text}</p>
        <div class="cue-tags">${cues}</div>
      </div>`;
  }).join("");
}

function renderSubtype(subtype) {
  const resultEl = document.getElementById("subtypeResult");
  const rankingsEl = document.getElementById("subtypeRankings");
  const chartImg = document.getElementById("subtypeImg");

  if (!subtype.applicable) {
    resultEl.innerHTML = `<div class="subtype-na">${subtype.message}</div>`;
    rankingsEl.innerHTML = "";
    chartImg.classList.add("hidden");
    return;
  }

  const symptoms = (subtype.matched_symptoms || [])
    .map((s) => `<li>${s}</li>`).join("");

  resultEl.innerHTML = `
    <div class="subtype-primary">
      <div class="type-name">${subtype.primary_name}</div>
      <div class="type-desc">${subtype.primary_description}</div>
      <div class="type-match">${(subtype.confidence * 100).toFixed(1)}% profile match</div>
      <p style="margin-top:0.75rem;font-size:0.9rem;color:var(--text)">
        ${subtype.message.replace(/\*\*/g, "")}
      </p>
      ${symptoms ? `<ul class="symptom-list">${symptoms}</ul>` : ""}
    </div>`;

  rankingsEl.innerHTML = `
    <h4 class="section-label">All type profiles ranked</h4>
    ${(subtype.rankings || []).map((r, i) => `
      <div class="subtype-row ${i === 0 ? "top" : ""}">
        <span class="rank-name">${r.name}</span>
        <div class="rank-bar-wrap">
          <div class="rank-bar" style="width:${r.probability * 100}%"></div>
        </div>
        <span class="rank-pct">${(r.probability * 100).toFixed(0)}%</span>
      </div>`).join("")}`;

}

function renderResults(data) {
  const isDepressed = data.prediction === "Depressed";
  const badge = document.getElementById("resultBadge");
  badge.className = `result-badge ${isDepressed ? "depressed" : "non-depressed"}`;
  badge.textContent = `${isDepressed ? "🔴" : "🟢"} ${data.prediction}`;

  document.getElementById("confidenceVal").textContent = `${(data.confidence * 100).toFixed(1)}%`;
  document.getElementById("probVal").textContent = `${(data.probability_depressed * 100).toFixed(1)}%`;
  document.getElementById("durationVal").textContent = `${data.audio_duration_sec}s`;
  document.getElementById("segmentsVal").textContent = data.n_segments;
  document.getElementById("probBar").style.width = `${data.probability_depressed * 100}%`;

  document.getElementById("predictionLabel").textContent = data.prediction;
  document.getElementById("predictionReason").textContent =
    data.prediction_reason.replace(/\*\*/g, "");

  renderTimelineCards(data.timeline_explanations || [], data.prediction);
  renderSubtype(data.subtype || {});

  if (data.charts.subtype) {
    document.getElementById("subtypeImg").src = `data:image/png;base64,${data.charts.subtype}`;
    document.getElementById("subtypeImg").classList.remove("hidden");
  }

  document.getElementById("spectrogramImg").src = `data:image/png;base64,${data.charts.spectrogram}`;
  document.getElementById("timelineImg").src = `data:image/png;base64,${data.charts.timeline}`;
  document.getElementById("gradCamImg").src = `data:image/png;base64,${data.charts.grad_cam}`;
  document.getElementById("featuresImg").src = `data:image/png;base64,${data.charts.features}`;

  const grid = document.getElementById("acousticGrid");
  grid.innerHTML = ACOUSTIC_DISPLAY.map(([label, key, type]) => {
    const val = data.acoustic_features[key] ?? 0;
    return `
      <div class="acoustic-item">
        <div class="label">${label}</div>
        <div class="value">${formatValue(val, type)}</div>
      </div>`;
  }).join("");

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const btnText = analyzeBtn.querySelector(".btn-text");
  const loader = analyzeBtn.querySelector(".btn-loader");
  btnText.textContent = "Analyzing…";
  loader.classList.remove("hidden");
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("chronic", document.getElementById("ctxChronic").checked);
  formData.append("recent_stress", document.getElementById("ctxStress").checked);
  formData.append("postpartum", document.getElementById("ctxPostpartum").checked);
  formData.append("seasonal", document.getElementById("ctxSeasonal").checked);
  formData.append("mood_swings", document.getElementById("ctxMoodSwings").checked);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed");
    renderResults(data);
    showToast("Analysis complete!");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    btnText.textContent = "Analyze Recording";
    loader.classList.add("hidden");
    analyzeBtn.disabled = !selectedFile;
  }
});

checkHealth();
