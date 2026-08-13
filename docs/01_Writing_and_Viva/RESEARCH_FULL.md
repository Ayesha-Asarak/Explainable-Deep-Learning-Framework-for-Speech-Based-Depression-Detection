# An Explainable Framework for Speech-Based Depression Detection

**Full Research Document (Aligned to Deployed System)**

**Author:** Ayesha Asarak (215509V)  
**Programme:** BSc(Hons) in Artificial Intelligence  
**Department:** Computational Mathematics, Faculty of Information Technology  
**University:** University of Moratuwa, Sri Lanka  
**Year:** 2026  

---

## Abstract

Depression is a major mental health disorder that often remains underdiagnosed because traditional assessment depends on interviews and self-reported symptoms. Speech provides a non-invasive behavioural signal: depressed speakers frequently show lower energy, flatter pitch, slower rate, and longer pauses. Machine learning can analyse these cues, but many systems are black boxes and some reported results are inflated by weak labels or speaker leakage.

This research develops an explainable speech-based depression detection framework. Using the DAIC-WOZ corpus with official AVEC 2017 PHQ-8 labels and official train/development/test splits, participant-only speech is segmented and converted into 23 interpretable acoustic features. Features are aggregated to participant level and classified by a Random Forest model. The decision threshold (≈ 0.49) is selected on train+dev only. Explanations include leave-one-segment-out occlusion importance, timeline cards, ranked acoustic features, spectrogram views, and—when the prediction is Depressed—an optional research subtype/category profile. The system is deployed as a FastAPI web application with patient-record support.

On 127 usable participants (95 train+dev / 32 held-out test), the deployed Random Forest achieved **75.0% accuracy**, **72.3% balanced accuracy**, **63.6% depression recall**, **81.0% specificity**, **63.6% F1**, and **ROC-AUC 0.654** (confusion matrix TN=17, FP=4, FN=4, TP=7; bootstrap Acc. CI ≈ 59.4%–87.5%). The result is seed-sensitive and is framed as **decision support, not diagnosis**.

**Keywords:** depression detection, speech acoustics, explainable AI, DAIC-WOZ, PHQ-8, Random Forest, occlusion importance, decision support

---

## 1. Introduction

Depression affects mood, cognition, and daily functioning worldwide. Early identification supports timely care, yet clinical workflows still rely heavily on interviews and questionnaires such as PHQ-8/PHQ-9. These methods are valuable but can be subjective, time-consuming, and inconsistent across settings.

Speech is an attractive complementary signal. Psychomotor changes associated with depression often appear as reduced pitch variability, lower vocal energy, slower articulation, and increased pausing. Artificial intelligence can quantify these patterns from interview recordings. However, two barriers limit clinical usefulness:

1. **Opacity** — many models output only a probability without saying which time regions or acoustic cues mattered.  
2. **Unreliable evaluation** — folder labels, segment-level scoring with speaker leakage, or tiny participant subsets can produce optimistic numbers that do not generalise.

This project addresses both barriers by building an end-to-end explainable pipeline: official clinical labels, participant-level prediction, multi-level explanations, and a deployable web interface intended for research decision support rather than diagnosis.

### 1.1 Problem (brief)

Depression is often missed; speech AI usually gives a score without a clear **Depressed / Non-Depressed** decision and without a clear explanation of why.

### 1.2 Solution (brief)

A speech system that:
- predicts **Depressed** or **Non-Depressed**
- if Depressed, also suggests a **depression category/profile** (research heuristic)
- explains the decision with time regions and acoustic cues
- runs as a web tool for decision support (not diagnosis)

---

## 2. Objectives

1. Critically review speech-based depression detection and identify gaps in interpretability and evaluation rigor.  
2. Study relevant methods in acoustic analysis, machine learning, deep learning, and explainable AI.  
3. Design an explainable framework that combines prediction with time-local and feature-level explanations.  
4. Implement a leakage-aware training/evaluation pipeline using official DAIC-WOZ / AVEC PHQ labels.  
5. Compare multiple modelling approaches and select a deployable model using accuracy and depression recall.  
6. Deploy the system as a web-based decision-support prototype with visualisation and patient records.  
7. Evaluate honestly and discuss contributions and limitations for healthcare AI.

---

## 3. Literature Review

### 3.1 Speech and depression

Prior work shows that paralinguistic features (pitch, energy, pause, rate, spectral cues, MFCCs) carry depression-related information. Corpora such as DAIC-WOZ and challenges such as AVEC provide interview audio with clinical questionnaire labels.

### 3.2 Modelling trends

| Approach | Strength | Limitation |
|----------|----------|------------|
| Handcrafted acoustics + classical ML | Interpretable features; strong on small data | May miss complex spectro-temporal patterns |
| CNN / RNN on spectrograms | Learns time–frequency patterns | Black-box; needs careful evaluation |
| Pretrained speech transformers (e.g. WavLM) | Strong representations | Data-hungry; less transparent |
| Multimodal (audio+text+video) | Often highest published scores | Harder privacy/deployment; not speech-only |

### 3.3 Explainable AI

Grad-CAM is useful for CNNs. For non-CNN tabular/ensemble models, occlusion, feature attribution, and timeline visualisations are more appropriate. Mental-health tools additionally need clinician-readable language, not only heatmaps.

### 3.4 Summary of literature gaps

Few systems jointly provide (a) official-label speaker-independent evaluation, (b) multi-level time-local explanations, and (c) a usable speech-only deployment interface.

---

## 4. Problem Definition (Research Gap)

| Gap | Description |
|-----|-------------|
| Representational opacity | Predictions without time/feature evidence |
| Weak evaluation protocols | Folder labels, leakage, tiny subsets |
| Dual-path disconnect | Deep models vs interpretable acoustics rarely integrated |
| Deployment/trust gap | Many prototypes stop at accuracy tables |
| Semantic gap | Explanations not mapped to clinical speech cues |

**Research questions**

- **RQ1:** Can speech acoustics support honest binary detection under official labels?  
- **RQ2:** Can multi-level explanations make predictions inspectable?  
- **RQ3:** Does XAI remain post-hoc (not replacing the predictive decision rule)?  

---

## 5. Technology Adopted

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| Audio | Librosa, SoundFile |
| Classical ML | scikit-learn (Random Forest, Extra Trees, SVM, Logistic) |
| Deep baseline | PyTorch CNN + Grad-CAM |
| Experiments | WavLM, LoRA/PEFT, openSMILE eGeMAPS, XGBoost |
| XAI | Occlusion importance, feature ranking, timelines, spectrograms |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Data | DAIC-WOZ interview audio + AVEC 2017 PHQ-8 CSVs |

**Primary commands**

```bash
python3 train_official_acoustic.py
python3 -m uvicorn server:app --host 127.0.0.1 --port 8765
```

---

## 6. Novel Approach

1. **Official PHQ-8 labels** and official AVEC partitions replace folder-name labels (~39 mismatches corrected).  
2. **Participant-level Random Forest** on aggregated acoustic statistics is the deployed classifier.  
3. **Multi-level explainability:** occlusion importance, timeline cards, feature ranks, spectrograms, natural-language reasons.  
4. **Category/profile output:** if prediction = Depressed, a heuristic subtype ranking is shown (research profile, not supervised diagnosis).  
5. **Web decision-support deployment** with patient identity and analysis history.  

**Dual pathway**

- **Deployed:** Random Forest + occlusion  
- **Baseline / prototype:** CNN + Grad-CAM  

---

## 7. Design

### 7.1 Top-level architecture

**Offline training**  
DAIC-WOZ + official labels → participant-only speech → 5 s segments → 23 features → aggregate → CV model search → threshold on train+dev → save artifact  

**Online inference**  
Upload → FastAPI → preprocess → Random Forest → explanations → UI tabs (+ optional patient save)  

### 7.2 Dataset design

| Item | Value |
|------|--------|
| Corpus | DAIC-WOZ / AVEC 2017 |
| Usable participants | 127 |
| Train+dev | 95 (37 depressed, 58 non-depressed) |
| Held-out test | 32 (11 depressed, 21 non-depressed) |
| Label source | Official PHQ-8 binary |
| Evaluation unit | Participant level |

### 7.3 Feature design

Per segment (examples): pitch mean/std, energy, speech rate, pause ratio, spectral descriptors, MFCCs.  
Aggregation: mean, standard deviation, median, 25th and 75th percentiles → one participant vector.

### 7.4 Explainability design

| Output | Role |
|--------|------|
| Occlusion importance | Which segments change P(depressed) most when removed |
| Timeline cards | Time-stamped supporting/opposing evidence |
| Feature ranking | Which acoustics contributed most |
| Spectrogram views | Visual context of the recording |
| Subtype profile | Optional category ranking if Depressed |

### 7.5 Design principles

Separation of concerns · leakage-aware evaluation · model-faithful XAI · decision-support framing (not diagnosis)

---

## 8. Implementation

### 8.1 Project structure (core)

```
Data/
├── train_official_acoustic.py      # deployed training
├── train.py                        # CNN baseline (optional)
├── server.py                       # FastAPI
├── frontend/                       # Web UI
├── models/
│   ├── depression_acoustic_candidate.pkl
│   └── acoustic_candidate_metadata.json
└── src/
    ├── config.py
    ├── data.py
    ├── features.py
    ├── predict.py
    ├── explain.py
    ├── subtype.py
    ├── patient_store.py
    └── api_utils.py
```

### 8.2 Deployed model configuration

| Property | Value |
|----------|--------|
| Model | Random Forest |
| `n_estimators` | 600 |
| `max_depth` | 8 |
| `min_samples_leaf` | 1 |
| `max_features` | sqrt |
| `class_weight` | balanced_subsample |
| `random_state` | 42 |
| Threshold | ≈ 0.49 |
| ACTIVE_MODEL | acoustic |

### 8.3 Inference workflow

1. User uploads audio + patient fields in the web UI.  
2. FastAPI validates and saves a temporary file.  
3. Audio is loaded at 16 kHz, segmented (5 s, 50% overlap).  
4. 23 acoustic features are extracted and aggregated.  
5. Random Forest outputs P(depressed); threshold yields label + confidence.  
6. Occlusion, timeline, features, and charts are generated.  
7. If Depressed → optional subtype/category ranking.  
8. JSON + charts returned to UI; temp file deleted.

### 8.4 UI tabs

Why This Result · Depression Type · Spectrogram · Voice Timeline · Attribution · Features  

---

## 9. Evaluation

### 9.1 Protocol

- Official AVEC PHQ-8 labels  
- Speaker-independent official split (95 / 32)  
- Threshold and model selection on train+dev only  
- Metrics: Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, Confusion Matrix, Bootstrap CI  

### 9.2 Deployed Random Forest results (held-out n = 32)

| Metric | Value |
|--------|--------|
| Accuracy | **75.0%** |
| Balanced accuracy | 72.3% |
| Precision | 63.6% |
| Depression recall (sensitivity) | **63.6%** |
| Specificity | 81.0% |
| F1-score | 63.6% |
| ROC-AUC | 0.654 |
| Accuracy 95% CI | 59.4% – 87.5% |

**Confusion matrix:** TN=17, FP=4, FN=4, TP=7  

**Caveat:** held-out accuracy is seed-sensitive; report with the confidence interval.

### 9.3 Candidate comparison (summary)

| Candidate | Acc. | Recall | Deployed? |
|-----------|------|--------|-----------|
| Random Forest | 75.0% | 63.6% | **Yes** |
| Previous Extra Trees | 59.4% | 63.6% | No |
| PHQ-8 Regression | 68.8% | 45.5% | No (weak recall) |
| eGeMAPS + temporal | 56.2% | 58.0% | No |
| Few-shot prototypes | 53.1% | 54.5% | No |
| Recall-hybrid | 53.1% | 63.6% | No |
| Segment-bag | 50.0% | n/a | No |
| LoRA WavLM | 40.6% | 90.9% | No (low Acc.) |

**Selection rule:** best usable balance of accuracy and depression recall for screening-oriented decision support.

### 9.4 Answers to research questions

- **RQ1:** Yes, moderately — 75.0% Acc. and 63.6% recall under official labels.  
- **RQ2:** Yes, qualitatively — time-local occlusion, cues, and feature ranks make predictions inspectable.  
- **RQ3:** Yes — XAI is post-hoc; it does not replace the Random Forest decision rule.

### 9.5 Limitations

- Small held-out set (n = 32) and wide CI  
- Seed-sensitive accuracy  
- Speech-only (no text/video in deployed path)  
- Single corpus; no clinician-rated explanation study  
- Subtype module is heuristic, not supervised diagnosis  
- Early CNN 9-participant prototype is not the final claim  

---

## 10. Conclusion and Further Work

### 10.1 Conclusion

This research delivered an end-to-end explainable speech depression framework with official-label evaluation and web deployment. The deployed Random Forest acoustic model reaches **75.0% held-out accuracy** and **63.6% depression recall**, with occlusion-based and feature-level explanations. Performance remains moderate in clinical terms and must be presented as **decision support**, not diagnosis.

### 10.2 Contributions

1. Integrated explainable speech pipeline (predict + explain + deploy)  
2. Dual pathway: deployed RF + CNN/Grad-CAM baseline  
3. Time-local clinical grounding of explanations  
4. Leakage-aware official-label evaluation protocol  
5. Deployable FastAPI decision-support application  
6. Responsible-AI safeguards (disclaimers, uncertainty, opposing evidence)

### 10.3 Further work

- Larger and multi-corpus validation (e.g. E-DAIC)  
- Clinician user studies for explanation usefulness  
- Fairness analysis by gender/age/accent  
- Better calibration and uncertainty communication  
- Human-in-the-loop screening workflows  

---

## 11. References (selected)

Full bibliography is maintained in the thesis PDF. Core references include:

1. Gratch et al. — DAIC-WOZ / distress interview corpus  
2. AVEC 2017 depression challenge materials and PHQ-8 label partitions  
3. Cummins et al. — surveys on speech and depression  
4. Selvaraju et al. — Grad-CAM  
5. Breiman — Random Forests  
6. Related work on acoustic features, WavLM/transformers, multimodal depression detection, and XAI for healthcare (see thesis References chapter)

---

## Appendix snapshot (quick facts)

| Item | Value |
|------|--------|
| Deployed model | Random Forest |
| Dataset usable | 127 (95 / 32) |
| Threshold | ≈ 0.49 |
| Held-out Acc. | 75.0% |
| Depression recall | 63.6% |
| Specificity | 81.0% |
| F1 | 63.6% |
| ROC-AUC | 0.654 |
| CM | TN17 / FP4 / FN4 / TP7 |
| XAI (deployed) | Occlusion + timeline + features |
| Baseline XAI | Grad-CAM (CNN only) |
| Framing | Decision support, not diagnosis |

---

## Related project files

| File | Purpose |
|------|---------|
| `THESIS_PRESENTATION_SLIDES.md` | Slide content |
| `THESIS_APPENDICES.md` | Appendices A–H |
| `THESIS_PASTE_FIXES_RF75.md` | Thesis edit checklist |
| `thesis_full_top_tier_architecture.png` | Master architecture figure |
| `thesis_fig7_1_confusion_matrix.png` | Confusion matrix |
| `thesis_fig7_2_heldout_metrics.png` | Held-out metrics |
| `thesis_fig7_3_candidate_comparison.png` | 8-model comparison |

---

*End of research document. This file matches the deployed Random Forest system (not the older Extra Trees 59.4% story, and not the early CNN 9-participant prototype as the final claim).*
