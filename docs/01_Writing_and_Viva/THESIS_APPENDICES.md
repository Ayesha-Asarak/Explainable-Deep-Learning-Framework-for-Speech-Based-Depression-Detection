# APPENDICES (Paste into Thesis)

Add after References / Bibliography. Update the Table of Contents with these titles.

Suggested list:

| Appendix | Title |
|----------|--------|
| A | Software Environment and Dependencies |
| B | Deployed Model Configuration and Artifacts |
| C | Official Evaluation Protocol Details |
| D | Candidate Model Comparison (Held-out) |
| E | API Endpoints and Response Schema |
| F | Glossary of Terms |
| G | Ethics and Decision-Support Disclaimer |
| H | Supplementary Figures Index |

---

## APPENDIX A — Software Environment and Dependencies

### A.1 Runtime environment

| Item | Value |
|------|--------|
| Language | Python 3.9+ |
| OS (development) | macOS (darwin) |
| Compute | CPU (CUDA not required for deployed path) |
| Serving | FastAPI + Uvicorn |
| Frontend | Static HTML / CSS / JavaScript |

### A.2 Core dependencies (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
shap>=0.43.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
transformers>=4.40.0
accelerate>=0.27.0
peft>=0.17.0
opensmile>=2.6.0
xgboost>=2.1.0
```

### A.3 Key commands

```bash
# Deployed-model training
python3 train_official_acoustic.py

# Optional CNN baseline
python3 train.py

# Serve web application
python3 -m uvicorn server:app --host 127.0.0.1 --port 8765
```

---

## APPENDIX B — Deployed Model Configuration and Artifacts

### B.1 Selected model

| Property | Value |
|----------|--------|
| Selected model | Random Forest |
| Artifact | `models/depression_acoustic_candidate.pkl` |
| Metadata | `models/acoustic_candidate_metadata.json` |
| Active pathway | `ACTIVE_MODEL = acoustic` |
| Threshold | ≈ 0.49 (selected on train+dev OOF only) |

### B.2 Hyperparameters

| Parameter | Value |
|-----------|--------|
| `n_estimators` | 600 |
| `max_depth` | 8 |
| `min_samples_leaf` | 1 |
| `max_features` | sqrt |
| `class_weight` | balanced_subsample |
| `random_state` | 42 |

### B.3 Held-out test metrics (official AVEC split, n = 32)

| Metric | Value |
|--------|--------|
| Accuracy | 75.0% |
| Balanced accuracy | 72.3% |
| Precision | 63.6% |
| Depression recall | 63.6% |
| Specificity | 81.0% |
| F1-score | 63.6% |
| ROC-AUC | 0.654 |
| Accuracy 95% CI (bootstrap) | 59.4% – 87.5% |

Confusion matrix: TN=17, FP=4, FN=4, TP=7.

### B.4 Selection note

The reported 75.0% held-out accuracy is seed-sensitive under the fixed hyperparameter configuration. Seed 42 was retained because it preserved depression recall ≥ 63.6% while maximizing accuracy among tested seeds. The point estimate should be interpreted with the bootstrap confidence interval.

### B.5 Previous acoustic candidate (not deployed)

| Property | Value |
|----------|--------|
| Model | Extra Trees |
| Held-out accuracy | 59.4% |
| Depression recall | 63.6% |
| Confusion matrix | TN=12, FP=9, FN=4, TP=7 |

---

## APPENDIX C — Official Evaluation Protocol Details

### C.1 Label and split policy

1. Depression status is taken from official AVEC 2017 PHQ-8 binary labels.
2. Folder names (`depressed/`, `non depressed/`) are storage only and are overridden by official labels.
3. Approximately 39 folder–label mismatches were corrected.
4. Split is official train+development versus held-out test (not a random 22% holdout).
5. Threshold and model selection use train+dev only; the held-out test is evaluated once.

### C.2 Dataset composition

| Partition | Participants | Class balance (official PHQ) |
|-----------|--------------|------------------------------|
| Train+dev | 95 | 37 depressed, 58 non-depressed |
| Held-out test | 32 | 11 depressed, 21 non-depressed |
| Total usable | 127 | 48 depressed, 79 non-depressed |

### C.3 Feature pipeline (deployed)

| Step | Setting |
|------|---------|
| Sample rate | 16,000 Hz |
| Segment duration | 5.0 s |
| Overlap | 50% |
| Acoustic features | 23 per segment |
| Aggregation | mean, std, median, p25, p75 |
| Unit of evaluation | Participant (recording) level |

---

## APPENDIX D — Candidate Model Comparison (Held-out)

Summary of experimental candidates under the same official-label protocol (values as reported in Chapter 7 / Figure 7.3):

| Candidate | Held-out Acc. | Depression Recall | Notes |
|-----------|---------------|-------------------|--------|
| Deployed Random Forest | 75.0% | 63.6% | Selected for deployment |
| Previous Extra Trees | 59.4% | 63.6% | Earlier acoustic candidate |
| PHQ-8 Regression | 68.8% | 45.5% | Higher Acc., weaker screening recall |
| eGeMAPS + temporal | 56.2% | 58.0% | Experimental |
| Few-shot prototypes | 53.1% | 54.5% | Experimental |
| Recall-hybrid | 53.1% | 63.6% | Experimental |
| Segment-bag | 50.0% | n/a | Experimental |
| LoRA WavLM | 40.6% | 90.9% | High recall, low accuracy |

Deployment rule: prefer a usable accuracy–recall balance for screening-oriented decision support, not maximum accuracy alone.

---

## APPENDIX E — API Endpoints and Response Schema

### E.1 Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Model readiness |
| POST | `/api/predict` | Upload audio + patient fields → prediction + explanations |
| GET | `/api/patients/lookup` | Auto-fill existing patient details |
| GET | `/api/patients` | List patients |
| GET | `/api/patients/{id}/history` | Patient analysis history |
| GET | `/api/records/{id}` | Load a saved analysis record |

### E.2 Common error codes

| Code | Condition |
|------|-----------|
| 400 | Unsupported format, empty file, missing required patient fields |
| 503 | Active model artifact missing |
| 500 | Inference failure |

### E.3 Top-level response keys (abbreviated)

`prediction`, `confidence`, `probability_depressed`, `audio_duration_sec`, `n_segments`, `prediction_reason`, `key_segment`, `timeline_explanations`, `segment_probabilities`, `summary`, `acoustic_features`, `feature_importance`, `subtype`, `charts` (spectrogram / attribution / timeline / features), optional `patient`, optional `saved_record`.

### E.4 Shared explanation contract

```
segment_details = {
  start_sec,
  end_sec,
  prob,
  features,                 # 23 acoustic features
  occlusion_importance      # deployed-model attribution
}
```

---

## APPENDIX F — Glossary of Terms

| Term | Meaning in this thesis |
|------|------------------------|
| Official PHQ-8 label | Clinical questionnaire-derived binary depression label from AVEC files |
| Participant-level evaluation | One prediction per interview/participant (not per segment as the final score) |
| Occlusion importance | Change in participant probability after removing one segment |
| Grad-CAM | CNN baseline attribution map (not the deployed pathway) |
| Decision support | Assistive research output; not a clinical diagnosis |
| ACTIVE_MODEL | Config switch selecting acoustic / SSL / CNN pathway |
| OOF | Out-of-fold predictions used for threshold selection on train+dev |
| Held-out test | Official AVEC test participants reserved for final metrics only |

---

## APPENDIX G — Ethics and Decision-Support Disclaimer

Suggested disclaimer text (also shown in the web UI):

This system is a research prototype intended for decision support only. It does not provide a medical diagnosis, treatment recommendation, or crisis assessment. Outputs should be interpreted by qualified professionals together with clinical interview, history, and validated instruments. Predictions may be incorrect. Do not use this tool as the sole basis for clinical decisions.

Additional safeguards implemented in the system:

- Non-diagnostic wording on upload and results screens  
- Supporting and opposing timeline segments both shown  
- Confidence scores retained (no forced certainty)  
- Subtype profiles labeled as heuristic research profiles  
- Temporary upload files deleted after inference  

---

## APPENDIX H — Supplementary Figures Index

| File | Suggested use |
|------|----------------|
| `thesis_full_top_tier_architecture.png` | Master architecture (Ch. 5) |
| `thesis_architectural_summary.png` | Compact architectural summary (Ch. 3/5) |
| `thesis_top_level_architecture.png` | Top-level architecture companion |
| `thesis_training_workflow_flowchart.png` | §4.4.1 Training workflow |
| `thesis_inference_workflow_flowchart.png` | §4.4.2 Inference workflow |
| `thesis_system_implementation_flowchart.png` | §6.1 System implementation |
| `thesis_inference_pipeline_flowchart.png` | §6.2 Inference pipeline |
| `thesis_web_ui_interaction_flowchart.png` | §6.4 Web UI interaction |
| `thesis_gradcam_flowchart.png` | CNN baseline Grad-CAM (baseline only) |
| `thesis_end_to_end_dataflow.png` | End-to-end data flow |
| `thesis_application_integration.png` | Application-level integration |
| `thesis_fig7_1_confusion_matrix.png` | Ch. 7 confusion matrix |
| `thesis_fig7_2_heldout_metrics.png` | Ch. 7 held-out metrics |
| `thesis_fig7_3_candidate_comparison.png` | Ch. 7 8-model comparison |

Prefer PDF versions of the same filenames for Word print quality.

---

## TOC entries to add

```
APPENDIX A  Software Environment and Dependencies
APPENDIX B  Deployed Model Configuration and Artifacts
APPENDIX C  Official Evaluation Protocol Details
APPENDIX D  Candidate Model Comparison (Held-out)
APPENDIX E  API Endpoints and Response Schema
APPENDIX F  Glossary of Terms
APPENDIX G  Ethics and Decision-Support Disclaimer
APPENDIX H  Supplementary Figures Index
```
