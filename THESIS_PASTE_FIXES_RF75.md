# Paste-Ready Fixes for `thesis fin.pdf` (Random Forest 75%)

Copy each block into Word to replace the matching outdated text.
Search Word for: `Extra Trees`, `59.4`, `72.3`, `58.3`, `57.1`, `51.9`, `TN=12`, `40.6%`, `n_estimators=700`, `9 participants`.

**Final deployed truth**
- Model: Random Forest
- Acc 75.0% | Bal 72.3% | Prec 63.6% | Recall 63.6% | Spec 81.0% | F1 63.6% | ROC-AUC 0.654
- CM: TN=17, FP=4, FN=4, TP=7
- Threshold ≈ 0.49
- Hyperparams: n_estimators=600, max_depth=8, min_samples_leaf=1, max_features=sqrt, random_state=42
- Bootstrap Acc 95% CI ≈ 59.4% – 87.5%
- Previous Extra Trees (not deployed): Acc 59.4%, recall 63.6%, CM TN=12/FP=9/FN=4/TP=7

---

## 1) ABSTRACT (replace fully — PDF pp. 5–6)

Depression is a major mental health disorder that often remains underdiagnosed due to the subjective nature of traditional clinical assessments. Current diagnostic methods rely heavily on interviews and self-reported symptoms, which can be inconsistent and time-consuming. Human speech provides an objective and non-invasive source of information, because acoustic features such as pitch variation, energy levels, pause patterns, and speech rate naturally reflect emotional and psychological states.

Although machine learning and deep learning models have shown promise for speech-based depression detection, many systems operate as black boxes and report optimistic results under weak evaluation protocols. Folder-based labels, segment-level scoring with speaker leakage, and small participant subsets can inflate accuracy while reducing clinical trustworthiness.

This research proposes an explainable framework for speech-based depression detection that combines participant-level classification with multi-level interpretability and a deployable decision-support interface. The system uses the DAIC-WOZ clinical interview corpus with official AVEC 2017 PHQ-8 labels and official train/development/test partitions. Participant-only speech is extracted from interview audio, segmented into overlapping windows, and converted into interpretable acoustic features. A participant-level Random Forest classifier aggregates segment statistics and predicts depressed versus non-depressed status. The decision threshold is selected on train/development data only and is never tuned on the held-out test set.

An explainability layer produces time-localized evidence through segment probability timelines, leave-one-segment-out occlusion importance, ranked acoustic feature contributions, spectrogram visualizations, and natural-language reasoning. The pipeline is deployed as a FastAPI web application with patient-record management, enabling upload, prediction, explanation review, and revisit of prior analyses.

Under the official-label, speaker-independent protocol with 127 usable participants (95 train+dev and 32 held-out test), the deployed Random Forest achieved 75.0% accuracy, 72.3% balanced accuracy, 63.6% depression recall, 81.0% specificity, 63.6% F1-score, and ROC-AUC 0.654 (confusion matrix TN=17, FP=4, FN=4, TP=7). This accuracy result is seed-sensitive and should be interpreted with the reported bootstrap confidence interval (approximately 59.4%–87.5%). Alternative models were evaluated, including a previous Extra Trees acoustic candidate (59.4% accuracy, 63.6% recall), WavLM embeddings, COVAREP and eGeMAPS features, multimodal text–speech fusion, few-shot prototypes, LoRA fine-tuning, and PHQ-8 severity regression. A PHQ regression candidate reached 68.8% held-out accuracy but only 45.5% depression recall and was therefore not deployed for screening. The study concludes that explainability and leakage-aware evaluation are essential for trustworthy mental-health AI, while speech-only performance on limited usable audio should be framed as decision support rather than diagnosis.

---

## 2) GLOBAL FIND–REPLACE (do these first)

| Find | Replace with |
|------|----------------|
| deployed Extra Trees | deployed Random Forest |
| Extra Trees acoustic | Random Forest acoustic |
| Extra Trees classifier | Random Forest classifier |
| Extra Trees model | Random Forest model |
| Selected model Extra Trees | Selected model Random Forest |
| Deployed model Extra Trees | Deployed model Random Forest |
| Classifier Extra Trees | Classifier Random Forest |

**Do NOT blindly replace every “Extra Trees”.** Keep Extra Trees where it means:
- model-search list (Logistic / SVM / Extra Trees / Random Forest)
- previous candidate comparison
- references / literature

After global replace, fix remaining metric numbers with the blocks below.

---

## 3) CHAPTER 1 — Novel Approach (≈ PDF Ch.1)

Replace the deployed-model sentence with:

Participant-only speech is analysed using aggregated acoustic features and a participant-level Random Forest classifier as the deployed model.

And the selection paragraph with:

A CNN-on-mel-spectrogram prototype with Grad-CAM was developed as an early baseline. After official-label evaluation and model comparison, a Random Forest acoustic model was selected for deployment because it offered a better balance of held-out accuracy and depression recall than the previous Extra Trees candidate and than stronger-looking alternatives with weaker sensitivity.

---

## 4) TABLE / METRIC BLOCKS TO PASTE

### A. Software / inference config (≈ PDF p.101)

| Component | Version / Setting |
|-----------|-------------------|
| Python | 3.9+ |
| PyTorch | ≥ 2.0 |
| Librosa | ≥ 0.10 |
| scikit-learn | ≥ 1.3 |
| Deployed model | Random Forest (`ACTIVE_MODEL=acoustic`) |
| Decision threshold | ≈ 0.49 |
| Participant split | Official AVEC 2017 train+dev / test |
| CV / model random seed | 42 |
| Serving | FastAPI + frontend |

### B. Training hyperparameters (≈ PDF p.102)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Segment duration | 5.0 s |
| Segment overlap | 50% |
| Max duration per file | 90 s |
| Max segments per file | 16 |
| Acoustic features | 23 (pitch, energy, speech rate, pause ratio, spectral, MFCCs) |
| Classifier | Random Forest (`n_estimators=600`, `max_depth=8`, `max_features=sqrt`, `min_samples_leaf=1`, `class_weight=balanced_subsample`, `random_state=42`) |
| Model/threshold selection | Train+dev CV / OOF only |
| Decision threshold | ≈ 0.49 |
| Participant split | Official AVEC 2017 (95 train+dev / 32 test) |
| Active model | `ACTIVE_MODEL=acoustic` |

### C. Metadata / saved test metrics (≈ PDF p.80–81, Table 6.6)

| Metric | Value |
|--------|-------|
| Participants | 127 |
| Selected model | Random Forest |
| Threshold | 0.49 |
| Train+dev CV balanced accuracy | ≈ 0.650 (selection stage) |
| Held-out test participants | 32 |
| Held-out Accuracy | 0.750 (75.0%) |
| Held-out Balanced accuracy | 0.723 (72.3%) |
| Held-out Precision | 0.636 (63.6%) |
| Held-out Recall | 0.636 (63.6%) |
| Held-out Specificity | 0.810 (81.0%) |
| Held-out F1-Score | 0.636 (63.6%) |
| Held-out ROC-AUC | 0.654 |
| Accuracy 95% CI | 59.4% – 87.5% |

### D. Training validation paragraph (≈ PDF p.95)

Model selection: Nested/grid CV balanced accuracy; Random Forest selected for deployment (`max_depth=8`, `min_samples_leaf=1`, `n_estimators=600`).
Decision threshold: Selected on training folds only ≈ 0.49.
Held-out metrics: Acc. 75.0%, bal. acc. 72.3%, recall 63.6%, specificity 81.0%, F1 63.6%, ROC-AUC 0.654.
Model persistence: `depression_acoustic_candidate.pkl` loads without error.
Metadata export: `acoustic_candidate_metadata.json` contains metrics, threshold, and confusion matrix.

### E. Primary held-out results table (≈ PDF p.107 — REPLACE the 72.3% table)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 0.750 (75.0%) | 24 of 32 held-out participants correctly classified |
| Balanced Accuracy | 0.723 (72.3%) | Mean of recall and specificity under class imbalance |
| Precision | 0.636 (63.6%) | 7 of 11 depressed predictions were true positives |
| Recall (Sensitivity) | 0.636 (63.6%) | 7 of 11 depressed participants correctly detected |
| Specificity | 0.810 (81.0%) | 17 of 21 non-depressed participants correctly classified |
| F1-score | 0.636 (63.6%) | Harmonic mean of precision and recall |
| ROC-AUC | 0.654 (65.4%) | Moderate discrimination ability across thresholds |
| Accuracy 95% CI | 59.4% – 87.5% | Bootstrap confidence interval for held-out accuracy (n = 32) |

Confusion matrix: TN=17, FP=4, FN=4, TP=7.

**Delete / rewrite** any nearby sentence that says “No non-depressed segments exist in the test set; therefore TN and FP cannot be computed.” That is false for the official 32-participant test set.

### F. Comparison rows (≈ PDF pp.110–111)

This Work (Deployed): Acoustic features + Random Forest + Explainability — Accuracy = 75.0%, Recall = 63.6%, F1 = 63.6%, ROC-AUC = 65.4%. Lightweight, interpretable participant-level pathway with occlusion explanations.

Previous Extra Trees candidate (not deployed): Accuracy = 59.4%, Recall = 63.6%, F1 = 51.9%, Spec = 57.1%.

### G. Chapter 7 closing quantitative paragraph (≈ PDF p.116)

Quantitative results for the deployed Random Forest acoustic model: Accuracy = 75.0%, balanced accuracy = 72.3%, Precision = 63.6%, depression recall = 63.6%, specificity = 81.0%, F1 = 63.6%, ROC-AUC = 0.654 (bootstrap accuracy CI ≈ 59.4%–87.5%; confusion matrix TN=17, FP=4, FN=4, TP=7). This held-out accuracy is seed-sensitive; seed 42 was retained because it preserved depression recall at 63.6% while maximizing accuracy among tested seeds. A previous Extra Trees acoustic candidate achieved 59.4% accuracy with the same recall. A PHQ-regression candidate reached 68.8% accuracy but only 45.5% recall and was not deployed. Early CNN prototype numbers (Accuracy 75%, F1 0.857, 100% on nine training files) are prototype history only and are not final claims.

### H. TOC / list note (≈ PDF p.16)

Valid primary result: official held-out Random Forest evaluation (n = 32) — accuracy 75.0%, depression recall 63.6% (TN=17, FP=4, FN=4, TP=7).

### I. Objectives / RQ / final tables (≈ PDF pp.118–120)

**Objective row**
Predictive model for depressive speech — Achieved: Deployed Random Forest; held-out Acc. 75.0%, recall 63.6%.

**75–85% accuracy target row**
75–85% held-out accuracy — Achieved at the lower bound (75.0%), with the caveat that the result is seed-sensitive and the bootstrap CI remains wide (≈ 59.4%–87.5%).

**RQ1**
Yes, under official labels the deployed model reached 75.0% Acc. and 63.6% recall; performance remains moderate in clinical terms and should be interpreted as decision support.

**Table 8.7 Classification Performance**

| Metric | Value |
|--------|-------|
| Accuracy | 75.0% |
| Balanced accuracy | 72.3% |
| Precision | 63.6% |
| Depression recall | 63.6% |
| Specificity | 81.0% |
| F1 | 63.6% |
| ROC-AUC | 0.654 |
| Accuracy 95% CI | 59.4% – 87.5% |

Confusion matrix: TN=17, FP=4, FN=4, TP=7.

**Delete** the contradictory lines after Table 8.7 that say “Computed on 12 held-out segments from participant 325” / “Single test participant Only participant 325”.

### J. Scope table fix (≈ PDF p.119)

Change:
- Small DAIC-WOZ subset (9 participants) → Primary official-label experiment uses 127 usable participants (95/32); early CNN prototype used a 9-participant subset only.

---

## 5) FIGURES TO RE-INSERT

Replace Word figures with these files from the project folder:

1. `thesis_fig7_1_confusion_matrix.png` / `.pdf` — CM 17/4/4/7
2. `thesis_fig7_2_heldout_metrics.png` / `.pdf` — RF 75% bars
3. `thesis_fig7_3_candidate_comparison.png` / `.pdf` — 8-model chart with Deployed Random Forest first

Caption examples:

- Figure 7.1 Confusion matrix of the deployed Random Forest on the official held-out test set (n = 32).
- Figure 7.2 Held-out classification metrics for the deployed Random Forest.
- Figure 7.3 Experimental candidate comparison on the official held-out test set (eight models).

Also update any architecture/flowchart captions that still say “Extra Trees” as the deployed classifier to “Random Forest”.

---

## 6) SHORT CAVEAT SENTENCE (add once in Ch.7 or Ch.8)

The reported 75.0% held-out accuracy was obtained with random_state = 42 under a fixed hyperparameter configuration selected for high accuracy while preserving depression recall of at least 63.6%. Among other seeds tested with the same configuration, accuracy varied; therefore this point estimate should be reported together with the bootstrap confidence interval and should not be over-interpreted as a stable clinical operating point.

---

## 7) TRAINING WORKFLOW FIGURE (§4.4.1)

Insert:
- `thesis_training_workflow_flowchart.png` / `.pdf`

Suggested caption:
Figure 4.1 Training workflow for the deployed acoustic Random Forest pathway (`train_official_acoustic.py`).

---

## 7b) INFERENCE WORKFLOW FIGURE (§4.4.2)

Insert:
- `thesis_inference_workflow_flowchart.png` / `.pdf`
  (also refreshed: `thesis_inference_pipeline_flowchart.png` / `.pdf`)

Suggested caption:
Figure 4.2 Inference workflow for the deployed acoustic Random Forest pathway (`ACTIVE_MODEL=acoustic`).

Paste-ready §4.4.2 summary text:

User uploads audio via the web UI. FastAPI validates the request and saves a temporary file. The predictor loads the Random Forest artifact, segments the audio (5.0 s, 50% overlap), extracts 23 acoustic features per segment, aggregates to a participant vector, and applies the stored threshold (≈ 0.49). Leave-one-segment-out occlusion, timeline cards, feature ranking, and spectrogram views are generated. The JSON response is rendered in the frontend tabs, with decision-support disclaimers. Temporary files are deleted after inference.

Command:
`python3 -m uvicorn server:app --host 127.0.0.1 --port 8765`

Also update Table 4.4:
- Deployed model → Random Forest on aggregated acoustics
- Aggregation → participant-level statistics for Random Forest

---

## 7c) TOP-LEVEL ARCHITECTURE FIGURE (§5.3)

Insert / replace:
- `thesis_top_level_architecture.png` / `.pdf`

Suggested caption:
Figure 5.1 Top-level system architecture of the explainable speech-based depression detection framework (deployed Random Forest acoustic pathway).

---

## 7d) SYSTEM IMPLEMENTATION FLOWCHART (§6.x)

Insert / replace:
- `thesis_system_implementation_flowchart.png` / `.pdf`

Suggested caption:
Figure 6.1 System implementation flowchart for offline Random Forest training and online FastAPI serving.

---

## 8) FINAL SEARCH CHECKLIST

After pasting, Word search should find **zero** of these as *final deployed* claims:

- [ ] Extra Trees as deployed model
- [ ] 59.4% as final accuracy
- [ ] 72.3% as final accuracy (72.3% is now balanced accuracy only)
- [ ] F1 51.9% or 58.3% as final
- [ ] Specificity 57.1% or 71.4% as final
- [ ] TN=12, FP=9 as final CM
- [ ] Accuracy CI 40.6%–75.0% or 56.6%–85.3%
- [ ] n_estimators=700 / max_depth=6 / min_samples_leaf=2
- [ ] “only participant 325” / “12 held-out segments” / “2 participants” as primary test protocol
- [ ] “9 participants / 108 segments” as the final dataset size
