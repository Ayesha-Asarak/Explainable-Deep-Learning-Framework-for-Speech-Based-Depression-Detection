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
const refreshRecordsBtn = document.getElementById("refreshRecordsBtn");
const searchPatientId = document.getElementById("searchPatientId");

let selectedFile = null;
let lookupTimer = null;
let autofillLock = false;

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

function getPatientForm() {
  return {
    name: document.getElementById("patientName").value.trim(),
    age: document.getElementById("patientAge").value.trim(),
    patient_id: document.getElementById("patientId").value.trim(),
    id_number: document.getElementById("idNumber").value.trim(),
    gender: document.getElementById("patientGender").value,
    phone: document.getElementById("patientPhone").value.trim(),
    notes: document.getElementById("patientNotes").value.trim(),
    save_record: document.getElementById("saveRecord").checked,
  };
}

function validatePatientForm(patient) {
  if (!patient.name) return "Please enter patient name";
  if (!patient.age) return "Please enter patient age";
  const ageNum = Number(patient.age);
  if (!Number.isFinite(ageNum) || ageNum < 1 || ageNum > 120) {
    return "Age must be between 1 and 120";
  }
  if (!patient.patient_id && !patient.id_number) {
    return "Please enter Patient ID or ID number";
  }
  return null;
}

function setMatchStatus(message, found = true) {
  const el = document.getElementById("patientMatchStatus");
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.classList.remove("hidden");
  el.classList.toggle("not-found", !found);
  el.textContent = message;
}

function fillPatientForm(patient, { overwrite = true } = {}) {
  if (!patient) return;
  autofillLock = true;
  const setVal = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (!overwrite && el.value.trim()) return;
    el.value = value ?? "";
  };
  setVal("patientName", patient.name || "");
  setVal("patientAge", patient.age ?? "");
  setVal("patientId", patient.patient_id || "");
  setVal("idNumber", patient.id_number || "");
  setVal("patientGender", patient.gender || "");
  setVal("patientPhone", patient.phone || "");
  setVal("patientNotes", patient.notes || "");
  autofillLock = false;
}

async function lookupPatient({ fromField } = {}) {
  if (autofillLock) return;

  const patientId = document.getElementById("patientId").value.trim();
  const idNumber = document.getElementById("idNumber").value.trim();
  const query = fromField === "id_number" ? idNumber : patientId;
  if (!query) {
    setMatchStatus("");
    return;
  }

  const params = new URLSearchParams();
  if (fromField === "id_number") params.set("id_number", idNumber);
  else params.set("patient_id", patientId);

  try {
    const res = await fetch(`/api/patients/lookup?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lookup failed");

    if (!data.found) {
      setMatchStatus("No saved patient found for this ID. New patient will be created on save.", false);
      return;
    }

    // Keep Patient ID and ID number linked for the same person
    fillPatientForm(data.patient, { overwrite: true });
    const when = data.saved_at ? new Date(data.saved_at).toLocaleString() : "";
    setMatchStatus(
      `Existing patient matched (${data.match_field}). ` +
      `Loaded ${data.patient.name || "profile"} · Patient ID ${data.patient.patient_id || "—"} · ` +
      `ID No. ${data.patient.id_number || "—"}` +
      (when ? ` · last saved ${when}` : "")
    );
    showToast("Patient details loaded automatically");
  } catch (err) {
    setMatchStatus(err.message, false);
  }
}

function scheduleLookup(fromField) {
  clearTimeout(lookupTimer);
  lookupTimer = setTimeout(() => lookupPatient({ fromField }), 400);
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

function renderPatientSummary(patient) {
  const el = document.getElementById("patientSummary");
  if (!patient) {
    el.innerHTML = "";
    return;
  }
  const fields = [
    ["Name", patient.name || "—"],
    ["Age", patient.age ?? "—"],
    ["Patient ID", patient.patient_id || "—"],
    ["ID Number", patient.id_number || "—"],
    ["Gender", patient.gender || "—"],
    ["Phone", patient.phone || "—"],
  ];
  el.innerHTML = fields.map(([label, value]) => `
    <div>
      <span class="ps-label">${label}</span>
      <div class="ps-value">${value}</div>
    </div>`).join("");
  if (patient.notes) {
    el.innerHTML += `
      <div style="grid-column:1/-1">
        <span class="ps-label">Notes</span>
        <div class="ps-value" style="font-weight:400;font-size:0.88rem">${patient.notes}</div>
      </div>`;
  }
}

function renderSaveStatus(saved) {
  const el = document.getElementById("saveStatus");
  if (!saved) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.classList.remove("hidden");
  el.textContent = `Saved · Record ID ${saved.record_id} · ${new Date(saved.saved_at).toLocaleString()}`;
}

function renderUncertainty(uncertainty) {
  const box = document.getElementById("uncertaintyBox");
  if (!uncertainty) {
    box.innerHTML = "";
    return;
  }

  const cls = uncertainty.stability || "uncertain";
  box.innerHTML = `
    <div class="u-title ${cls}">${uncertainty.stability_label || "Uncertainty"}</div>
    <div class="uncertainty-stats">
      <div class="u-stat">
        <span class="u-label">Mean (μ)</span>
        <div class="u-value">${(uncertainty.mean * 100).toFixed(1)}%</div>
      </div>
      <div class="u-stat">
        <span class="u-label">Std Dev (σ)</span>
        <div class="u-value">${(uncertainty.std * 100).toFixed(1)}%</div>
      </div>
      <div class="u-stat">
        <span class="u-label">μ − σ</span>
        <div class="u-value">${(uncertainty.lower * 100).toFixed(1)}%</div>
      </div>
      <div class="u-stat">
        <span class="u-label">μ + σ</span>
        <div class="u-value">${(uncertainty.upper * 100).toFixed(1)}%</div>
      </div>
      <div class="u-stat">
        <span class="u-label">Threshold</span>
        <div class="u-value">${((uncertainty.threshold ?? 0.5) * 100).toFixed(0)}%</div>
      </div>
    </div>
    <p class="u-msg">${uncertainty.message || ""}</p>
  `;
}

function renderResults(data) {
  const isDepressed = data.prediction === "Depressed";
  const badge = document.getElementById("resultBadge");
  badge.className = `result-badge ${isDepressed ? "depressed" : "non-depressed"}`;
  badge.textContent = `${isDepressed ? "🔴" : "🟢"} ${data.prediction}`;

  renderPatientSummary(data.patient);
  renderSaveStatus(data.saved_record);

  document.getElementById("confidenceVal").textContent = `${((data.confidence || 0) * 100).toFixed(1)}%`;
  document.getElementById("probVal").textContent = `${((data.probability_depressed || 0) * 100).toFixed(1)}%`;
  document.getElementById("durationVal").textContent = `${data.audio_duration_sec ?? "—"}s`;
  document.getElementById("segmentsVal").textContent = data.n_segments ?? "—";
  document.getElementById("probBar").style.width = `${(data.probability_depressed || 0) * 100}%`;

  document.getElementById("predictionLabel").textContent = data.prediction || "—";
  document.getElementById("predictionReason").textContent =
    (data.prediction_reason || "").replace(/\*\*/g, "");

  renderTimelineCards(data.timeline_explanations || [], data.prediction);
  renderSubtype(data.subtype || {});
  renderUncertainty(data.uncertainty || null);

  const charts = data.charts || {};
  const setChart = (imgId, b64) => {
    const img = document.getElementById(imgId);
    if (b64) {
      img.src = `data:image/png;base64,${b64}`;
      img.classList.remove("hidden");
    } else {
      img.removeAttribute("src");
      img.classList.add("hidden");
    }
  };

  if (charts.subtype) {
    setChart("subtypeImg", charts.subtype);
  } else {
    document.getElementById("subtypeImg").classList.add("hidden");
  }
  setChart("spectrogramImg", charts.spectrogram);
  setChart("timelineImg", charts.timeline);
  setChart("gradCamImg", charts.grad_cam);
  setChart("featuresImg", charts.features);

  const useOcclusion = (data.attribution_method || "") === "segment_occlusion";
  const gradcamTabBtn = document.getElementById("gradcamTabBtn");
  const gradcamHeading = document.getElementById("gradcamHeading");
  const gradcamDesc = document.getElementById("gradcamDesc");
  const gradCamImg = document.getElementById("gradCamImg");
  if (gradcamTabBtn) {
    gradcamTabBtn.textContent = useOcclusion ? "Occlusion map" : "Grad-CAM";
  }
  if (gradcamHeading) {
    gradcamHeading.textContent = useOcclusion
      ? "Occlusion map across the recording"
      : "Grad-CAM at key voice region";
  }
  if (gradcamDesc) {
    gradcamDesc.textContent = useOcclusion
      ? "Colour intensity shows how much each voice segment changed the participant-level prediction when removed (leave-one-segment-out). The curve below is the faithful segment importance; white dashed lines mark the peak segment."
      : "Shows the exact seconds and frequency bands that most influenced the CNN prediction.";
  }
  if (gradCamImg) {
    gradCamImg.alt = useOcclusion
      ? "Occlusion map visualization"
      : "Grad-CAM visualization";
  }

  const grid = document.getElementById("acousticGrid");
  grid.innerHTML = ACOUSTIC_DISPLAY.map(([label, key, type]) => {
    const val = (data.acoustic_features || {})[key] ?? 0;
    return `
      <div class="acoustic-item">
        <div class="label">${label}</div>
        <div class="value">${formatValue(val, type)}</div>
      </div>`;
  }).join("");

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

  // Open explanation tab by default when loading a saved record
  if (data.from_saved_record) {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    const summaryBtn = document.querySelector('.tab-btn[data-tab="summary"]');
    const summaryPanel = document.getElementById("tab-summary");
    if (summaryBtn) summaryBtn.classList.add("active");
    if (summaryPanel) summaryPanel.classList.add("active");
  }

  if (!data.from_saved_record) {
    loadRecords();
  }
}

async function openSavedRecord(recordId) {
  try {
    showToast("Loading saved analysis…");
    const res = await fetch(`/api/records/${encodeURIComponent(recordId)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to load record");

    renderResults(data);
    if (data.has_charts) {
      showToast("Saved explainability loaded");
    } else {
      showToast("Record loaded. Charts were not saved for older analyses — run a new analysis to store them.", true);
    }
  } catch (err) {
    showToast(err.message, true);
  }
}

async function deleteSavedRecord(recordId, patientLabel) {
  const ok = window.confirm(
    `Delete this saved analysis${patientLabel ? ` for ${patientLabel}` : ""}?\n\nThis cannot be undone.`
  );
  if (!ok) return;
  try {
    const res = await fetch(`/api/records/${encodeURIComponent(recordId)}`, {
      method: "DELETE",
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to delete record");
    showToast(`Deleted record ${recordId}`);
    await loadRecords();
  } catch (err) {
    showToast(err.message, true);
  }
}

async function deleteAllForPatient(patientId, patientLabel) {
  if (!patientId) {
    showToast("No patient ID available to delete", true);
    return;
  }
  const ok = window.confirm(
    `Delete ALL saved records for ${patientLabel || patientId}?\n\nThis removes patient details and all analyses for this ID.`
  );
  if (!ok) return;
  try {
    const res = await fetch(`/api/patients/${encodeURIComponent(patientId)}`, {
      method: "DELETE",
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to delete patient records");
    showToast(data.message || "Patient records deleted");
    await loadRecords();
  } catch (err) {
    showToast(err.message, true);
  }
}

async function loadRecords() {
  const list = document.getElementById("recordsList");
  const q = (searchPatientId.value || "").trim();
  const url = q ? `/api/patients?patient_id=${encodeURIComponent(q)}&limit=30` : "/api/patients?limit=30";

  try {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to load records");

    if (!data.records || data.records.length === 0) {
      list.innerHTML = `<p class="records-empty">No saved records found.</p>`;
      return;
    }

    list.innerHTML = data.records.map((r) => {
      const p = r.patient || {};
      const a = r.analysis || {};
      const pred = a.prediction || "—";
      const predClass = pred === "Depressed" ? "depressed" : "healthy";
      const conf = a.confidence != null ? `${(a.confidence * 100).toFixed(0)}%` : "—";
      const when = r.saved_at ? new Date(r.saved_at).toLocaleString() : "—";
      const subtype = a.subtype && a.subtype.primary_name ? a.subtype.primary_name : "—";
      const hasCharts = a.charts_available && Object.keys(a.charts_available).length > 0;
      const patientKey = p.patient_id || p.id_number || "";
      const patientLabel = p.name || patientKey || "this patient";
      const safeLabel = String(patientLabel).replace(/"/g, "&quot;");
      return `
        <div class="record-card" data-record-id="${r.record_id}">
          <button type="button" class="record-open clickable" data-record-id="${r.record_id}">
            <div class="record-card-top">
              <div class="record-name">${p.name || "Unknown"} · ${p.patient_id || p.id_number || "No ID"}</div>
              <span class="record-pred ${predClass}">${pred} (${conf})</span>
            </div>
            <div class="record-meta">
              Age ${p.age ?? "—"} · ${p.gender || "Gender n/a"} · Record ${r.record_id} · ${when}
            </div>
            <div class="record-meta">Subtype profile: ${subtype}</div>
            ${p.notes ? `<div class="record-meta">Notes: ${p.notes}</div>` : ""}
            <div class="record-meta record-action">
              ${hasCharts ? "Click to view full explainability" : "Click to view saved details"}
            </div>
          </button>
          <div class="record-actions-row">
            <button type="button" class="btn-delete-record" data-record-id="${r.record_id}" data-label="${safeLabel}">
              Delete record
            </button>
            <button type="button" class="btn-delete-patient" data-patient-id="${patientKey}" data-label="${safeLabel}" ${patientKey ? "" : "disabled"}>
              Delete patient
            </button>
          </div>
        </div>`;
    }).join("");

    list.querySelectorAll(".record-open[data-record-id]").forEach((card) => {
      card.addEventListener("click", () => openSavedRecord(card.dataset.recordId));
    });
    list.querySelectorAll(".btn-delete-record").forEach((btn) => {
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteSavedRecord(btn.dataset.recordId, btn.dataset.label);
      });
    });
    list.querySelectorAll(".btn-delete-patient").forEach((btn) => {
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteAllForPatient(btn.dataset.patientId, btn.dataset.label);
      });
    });
  } catch (err) {
    list.innerHTML = `<p class="records-empty">${err.message}</p>`;
  }
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const patient = getPatientForm();
  const formError = validatePatientForm(patient);
  if (formError) {
    showToast(formError, true);
    return;
  }

  const btnText = analyzeBtn.querySelector(".btn-text");
  const loader = analyzeBtn.querySelector(".btn-loader");
  btnText.textContent = "Analyzing…";
  loader.classList.remove("hidden");
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("name", patient.name);
  formData.append("age", patient.age);
  formData.append("patient_id", patient.patient_id);
  formData.append("id_number", patient.id_number);
  formData.append("gender", patient.gender);
  formData.append("phone", patient.phone);
  formData.append("notes", patient.notes);
  formData.append("save_record", patient.save_record);
  formData.append("chronic", document.getElementById("ctxChronic").checked);
  formData.append("recent_stress", document.getElementById("ctxStress").checked);
  formData.append("postpartum", document.getElementById("ctxPostpartum").checked);
  formData.append("seasonal", document.getElementById("ctxSeasonal").checked);
  formData.append("mood_swings", document.getElementById("ctxMoodSwings").checked);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg).join(", ")
        : (data.detail || "Analysis failed");
      throw new Error(detail);
    }
    renderResults(data);
    showToast(data.saved_record ? "Analysis complete and saved!" : "Analysis complete!");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    btnText.textContent = "Analyze Recording";
    loader.classList.add("hidden");
    analyzeBtn.disabled = !selectedFile;
  }
});

refreshRecordsBtn.addEventListener("click", loadRecords);
searchPatientId.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadRecords();
});

const patientIdInput = document.getElementById("patientId");
const idNumberInput = document.getElementById("idNumber");

patientIdInput.addEventListener("input", () => scheduleLookup("patient_id"));
patientIdInput.addEventListener("blur", () => lookupPatient({ fromField: "patient_id" }));
idNumberInput.addEventListener("input", () => scheduleLookup("id_number"));
idNumberInput.addEventListener("blur", () => lookupPatient({ fromField: "id_number" }));

checkHealth();
loadRecords();
