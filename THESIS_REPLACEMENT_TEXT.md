# Thesis Replacement Text (Match Current System)

Use this document to replace outdated thesis content.
Keep Declaration, Dedication, Acknowledgement, and most of Chapter 2 literature.
Replace Abstract and Chapters 1, 3–8 content below where marked.

---

## ABSTRACT (REPLACE FULLY)

Depression is a major mental health disorder that often remains underdiagnosed due to the subjective nature of traditional clinical assessments. Current diagnostic methods rely heavily on interviews and self-reported symptoms, which can be inconsistent and time-consuming. Human speech provides an objective and non-invasive source of information, because acoustic features such as pitch variation, energy levels, pause patterns, and speech rate naturally reflect emotional and psychological states.

Although machine learning and deep learning models have shown promise for speech-based depression detection, many systems operate as black boxes and report optimistic results under weak evaluation protocols. In particular, folder-based labels, segment-level scoring with speaker leakage, and small participant subsets can inflate accuracy while reducing clinical trustworthiness.

This research proposes an explainable framework for speech-based depression detection that combines participant-level classification with multi-level interpretability and a deployable decision-support interface. The system uses the DAIC-WOZ clinical interview corpus with official AVEC 2017 PHQ-8 labels and official train/development/test partitions. Participant-only speech is extracted from interview audio, segmented into overlapping windows, and converted into interpretable acoustic features. A participant-level Extra Trees classifier aggregates segment statistics and predicts depressed versus non-depressed status. The decision threshold is selected on train/development data only, using balanced-accuracy criteria, and is never tuned on the held-out test set.

An explainability layer produces time-localized evidence through segment probability timelines, leave-one-segment-out occlusion importance, ranked acoustic feature contributions, spectrogram visualizations, and natural-language reasoning. The pipeline is deployed as a FastAPI web application with patient-record management, enabling upload, prediction, explanation review, and revisit of prior analyses.

Under the official-label, speaker-independent protocol with 127 usable participants (95 train+dev and 32 held-out test), the deployed model achieved 59.4% accuracy, 60.4% balanced accuracy, 63.6% depression recall, 57.1% specificity, 51.9% F1-score, and ROC-AUC 0.654. Alternative models were evaluated, including WavLM embeddings, COVAREP and eGeMAPS features, multimodal text–speech fusion, few-shot prototypes, LoRA fine-tuning, and PHQ-8 severity regression. A PHQ regression candidate reached 68.8% held-out accuracy but only 45.5% depression recall and was therefore not deployed for screening. The study concludes that explainability and leakage-aware evaluation are essential for trustworthy mental-health AI, while honest speech-only performance on limited usable audio remains moderate and should be framed as decision support rather than diagnosis.

---

## CHAPTER 1 — REPLACE THESE SECTIONS

### 1.1 Introduction (REPLACE)

Depression is one of the most prevalent mental health disorders worldwide and significantly affects emotional, cognitive, and social functioning. Early identification supports timely intervention, yet traditional assessment depends largely on clinical interviews and self-report questionnaires. These methods are valuable but can be time-consuming, subjective, and inconsistent across settings.

Speech provides a complementary behavioural signal. Depressed speakers often exhibit psychomotor retardation expressed as reduced pitch variability, lower vocal energy, slower speech rate, and increased pausing. Artificial intelligence can analyse these acoustic patterns automatically. However, many published systems focus on predictive accuracy alone and either lack clinician-usable explanations or evaluate performance with protocols that risk speaker leakage and label noise.

This thesis develops and evaluates an explainable speech-based depression detection framework. The final deployed system uses official PHQ-8 labels, participant-level evaluation, acoustic feature aggregation, and multi-level explanations delivered through a web application intended as decision support rather than diagnosis.

### 1.2 Objectives (REPLACE)

1. To critically review speech-based depression detection and identify gaps in interpretability and evaluation rigor.
2. To study relevant methods in acoustic analysis, machine learning, deep learning, and explainable AI.
3. To design an explainable depression-detection framework that combines prediction with time-localized and feature-level explanations.
4. To implement a leakage-aware training and evaluation pipeline using official DAIC-WOZ / AVEC PHQ labels and speaker-independent splits.
5. To compare multiple modelling approaches and select a deployable model based on both accuracy and depression recall.
6. To deploy the system as a web-based decision-support prototype with visualization and patient-record support.
7. To evaluate the framework honestly and discuss its contribution and limitations for healthcare AI.

### 1.3 Problem in Brief (REPLACE)

Deep learning and classical machine learning can detect depression-related patterns in speech, but two problems remain. First, many models are black boxes and do not explain which time regions or acoustic characteristics drove a prediction. Second, reported performance is often unreliable because of speaker leakage, segment-level inflation, and inconsistent labels. These issues create a gap between technical prototypes and clinically trustworthy decision-support tools.

### 1.4 Background and Motivation (REPLACE)

Speech-based depression detection has grown with corpora such as DAIC-WOZ and challenges such as AVEC. Early systems used handcrafted acoustics with classical classifiers; later systems used CNNs on spectrograms, recurrent models, and pretrained speech transformers. Multimodal audio–text–video systems often score highest, but speech-only tools remain attractive for privacy and deployment.

Motivation for this work comes from three needs: (1) transparent explanations for clinicians and researchers; (2) official-label, participant-level evaluation that avoids inflated claims; and (3) a usable software pipeline from audio upload to explained prediction.

### 1.5 Novel Approach (REPLACE)

The proposed approach extends standard speech depression classification in four ways:

1. Official PHQ-8 labels and official AVEC partitions replace folder-name labels.
2. Participant-only speech is analysed using aggregated acoustic features and a participant-level Extra Trees classifier as the deployed model.
3. Explainability is integrated into inference through occlusion importance, acoustic feature ranking, timeline cards, and spectrogram visualization.
4. The system is deployed as a web decision-support application with patient identity and analysis history.

A CNN-on-mel-spectrogram prototype with Grad-CAM was developed as an early baseline. After official-label evaluation and model comparison, the acoustic Extra Trees model was selected for deployment because it offered a better balance of held-out accuracy and depression recall than stronger-looking alternatives with weaker sensitivity.

### 1.6 Resource Requirements (REPLACE)

Hardware: development workstation with adequate storage for interview audio and model caches; GPU optional.
Software: Python 3.9+, PyTorch, Librosa, scikit-learn, XGBoost, Transformers/PEFT (experimental models), openSMILE, FastAPI, Uvicorn, Matplotlib.
Data: DAIC-WOZ-style interview audio, transcripts for participant-only extraction, and official AVEC 2017 PHQ label CSVs.

### 1.7 Structure of the Thesis (REPLACE)

Chapter 2 reviews related literature.
Chapter 3 presents theoretical foundations for the extension.
Chapter 4 describes the approach, hypotheses, and workflows.
Chapter 5 presents analysis and design.
Chapter 6 details implementation.
Chapter 7 reports evaluation results and discussion.
Chapter 8 concludes and outlines future work.

### 1.8 Summary (REPLACE)

This chapter introduced the need for explainable, rigorously evaluated speech-based depression detection. The thesis develops a deployable framework that prioritizes official labels, speaker-independent evaluation, acoustic interpretability, and clinical decision-support framing.

---

## CHAPTER 2 — KEEP MOST CONTENT; REPLACE ONLY THESE PARTS

### Replace Section 2.6.1 Scope Boundaries With:

Scope boundaries:
- Binary classification (depressed vs non-depressed); not a clinical diagnosis.
- Primary deployed pathway is speech-only acoustic analysis.
- Official AVEC 2017 PHQ-8 labels and official splits are used for final evaluation.
- Usable audio for the main reported experiment: 127 participants (48 depressed, 79 non-depressed).
- CNN + Grad-CAM is retained as a baseline/prototype pathway; the deployed pathway uses Extra Trees + occlusion explanations.
- Subtype matching is heuristic profile matching, not supervised subtype diagnosis.

### Replace Final Paragraph of 2.7 With:

The following chapters describe the proposed explainable framework, beginning with a CNN+Grad-CAM prototype and culminating in an official-label participant-level acoustic classifier with multi-level explanations and web deployment.

---

## CHAPTER 3 — REPLACE KEY SECTIONS

### 3.1 Introduction (UPDATE FINAL PARAGRAPH)

This chapter transitions from literature critique to the conceptual blueprint of the extension. The extension begins from CNN-based spectrogram classification and Grad-CAM theory, then expands to participant-level acoustic modelling, occlusion-based attribution, official-label evaluation, and decision-support deployment.

### 3.2.4 Machine Learning Theory (ADD AFTER CNN PARAGRAPH)

For small clinical corpora, participant-level tabular classifiers remain competitive. Aggregating segment acoustics into mean, standard deviation, median, and percentile statistics produces a fixed participant vector. Ensemble methods such as Extra Trees can learn non-linear interactions among pitch, energy, pause, spectral, and MFCC statistics while remaining comparatively robust under limited sample size. This thesis therefore treats CNN spectrogram modelling as a baseline pathway and participant-level Extra Trees acoustic classification as the final deployed pathway.

### 3.2.6 Explainable AI Theory (ADD)

When the primary model is not a CNN, Grad-CAM is not directly applicable. Faithful alternatives include leave-one-segment-out occlusion importance: a segment is important if removing it changes the participant-level depression probability. Feature-level gradients or magnitude-based acoustic rankings provide complementary semantic explanations. Timeline cards convert timestamps and acoustic thresholds into clinician-readable language.

### 3.3 Gap Statement (REPLACE FORMAL GAP)

There exists a practical gap for an integrated speech-based depression framework that (a) uses official clinical labels and speaker-independent evaluation, (b) provides multi-level time-localized explanations suitable for decision support, and (c) remains deployable as a speech-only research prototype.

### 3.5 Theoretical Foundations (REPLACE 3.5.1–3.5.2 EMPHASIS)

3.5.1 Baseline Classifier: CNN on Mel-Spectrograms
Retained as the initial deep learning baseline and Grad-CAM demonstration pathway.

3.5.2 Deployed Classifier: Participant-Level Acoustic Extra Trees
Participant speech is segmented; 23 acoustic features are extracted per segment; statistics are aggregated; Extra Trees predicts depression probability. Threshold selection uses train/development out-of-fold predictions.

3.5.3 Grad-CAM Extension
Used for the CNN pathway.

3.5.4 Occlusion Importance Extension
Used for the deployed acoustic pathway: leave-one-segment-out importance identifies influential time windows.

3.5.5 Acoustic Feature Pathway and Feature Importance
Unchanged in purpose: pitch, energy, pauses, spectral measures, and MFCCs provide semantic grounding.

3.5.6 Timeline Explanation Layer
Unchanged: overlapping windows, ranked supporting/opposing segments, natural-language cues.

3.5.7 Recording-Level Aggregation
Participant-level mean probability and thresholded decision.

3.5.8 Subtype Profile Matcher
Optional heuristic only.

3.5.9 Deployment Layer
FastAPI + web UI + patient records.

### Table 3.3 Characteristics (REPLACE TABLE CONTENT)

| Characteristic | Description |
|---|---|
| Dual pathway | CNN+Grad-CAM baseline; acoustic Extra Trees deployed |
| Official labels | AVEC PHQ-8 binary labels override folder names |
| Speaker-independent split | Official train+dev vs test; no participant overlap |
| Multi-level explanation | Occlusion/Grad-CAM, features, timeline, summary |
| Time-localized output | Absolute seconds |
| Decision-support framing | Non-diagnostic disclaimers |
| Deployable system | Training scripts + API + web UI + patient store |

### 3.7 Bridging Table (UPDATE ROWS)

| Research Gap | Extension Component |
|---|---|
| Representational opacity | Occlusion importance + Grad-CAM baseline |
| Temporal localization deficiency | Timeline cards + peak/key segment highlighting |
| Label and split unreliability | Official PHQ labels + official partitions |
| Dual-path disconnect | Acoustic semantics + model attribution |
| Deployment and trust gap | Web app with explanations and patient history |

### 3.8 Summary (REPLACE LAST SENTENCE)

Chapter 4 translates these foundations into the practical approach used in this thesis, covering both the CNN prototype and the final official-label acoustic deployment.

---

## CHAPTER 4 — REPLACE MAJOR CONTENT

### 4.1 Introduction (REPLACE)

This chapter describes the approach used to design, implement, and deploy the explainable speech-based depression detection framework. The work progressed from a CNN+Grad-CAM prototype to a final official-label participant-level acoustic classifier with occlusion-based explanations and web deployment.

### 4.2 Hypotheses (REPLACE)

H1: Officially labelled participant-level acoustic features from clinical interview speech contain discriminative signal for binary depression classification under speaker-independent evaluation.

H2: Multi-level explanations (timeline, acoustic features, and occlusion/Grad-CAM attributions) can make predictions inspectable without requiring the model to be a black box.

H3: Segmenting interviews and aggregating segment evidence supports time-localized explanations in absolute seconds.

H4: Packaging the pipeline as a web application enables practical decision-support demonstration and review of patient analyses.

Null expectation: with limited usable DAIC-WOZ audio, clinical-grade accuracy is not hypothesized. Target ranges such as 75–85% may be aspirational and must not be claimed unless achieved under the locked official test protocol.

### 4.3 Inputs and Outputs (REPLACE TABLES)

Inputs:
- Voice recording (WAV/MP3/FLAC/OGG/M4A/WebM)
- Optional patient metadata (name, age, ID, gender, notes)
- Optional clinical context checkboxes for subtype boosting
- Training labels from official AVEC PHQ-8 CSVs
- Transcripts for participant-only speech extraction

Outputs:
- Prediction: Depressed / Non-Depressed
- Probability and confidence
- Threshold used
- Timeline explanations with exact seconds
- Occlusion/Grad-CAM importance visualizations
- Acoustic feature importance
- Natural-language prediction reason
- Optional subtype profile ranking
- Optional saved patient analysis record

### 4.4 Training Workflow (REPLACE)

1. Discover audio from depressed/ and non depressed/ folders and ZIP archives.
2. Load official AVEC train/dev/test PHQ labels.
3. Override folder labels with official PHQ binary labels; exclude unresolved conflicts.
4. Extract participant-only speech using transcripts.
5. Segment speech (prototype: 3 s; SSL experiments: 5 s; deployed acoustic: 5 s windows with max duration/segment caps).
6. Extract acoustic features and aggregate to participant vectors.
7. Train candidate classifiers on official train+dev only.
8. Select model and threshold using cross-validation / out-of-fold metrics.
9. Evaluate once on official held-out test participants.
10. Deploy selected artifact through the web API.

Command for deployed model:
`python3 train_official_acoustic.py`

### 4.4.2 Inference Workflow (REPLACE STEP 3–5)

3. Per-segment acoustic feature extraction.
4. Participant-level aggregation and Extra Trees probability.
5. Thresholded decision.
6. Explainability: timeline cards, occlusion importance, feature ranking, spectrogram views, optional subtype.
7. Optional patient-record save and later recall.

### 4.5 Technologies (UPDATE / EXTEND)

Add:
- scikit-learn Extra Trees / logistic / SVM / forests
- XGBoost (experimental)
- Transformers / WavLM / PEFT LoRA (experimental)
- openSMILE eGeMAPS (experimental)
- FastAPI patient-record endpoints

Dataset row:
- Official AVEC label CSVs
- 127 usable participants in main official-label experiment

### 4.6 Features of Extension (REPLACE 4.6.2)

Dual-pathway architecture:
- Baseline: DepressionCNN + Grad-CAM
- Deployed: Extra Trees acoustic classifier + occlusion importance
- Shared: acoustic feature semantics, timeline explanations, web deployment

### 4.9 Summary (REPLACE)

The approach combines official-label evaluation, participant-level acoustic classification, multi-level explainability, and web deployment. The next chapters present design, implementation, and results for this final system, while retaining the CNN prototype as historical baseline.

---

## CHAPTER 5 — REPLACE ARCHITECTURE EMPHASIS

### 5.1 Introduction (REPLACE MODULE LIST)

The design comprises:
1. Preprocessing module: audio loading, participant-only extraction, segmentation, feature extraction.
2. ML engine: baseline CNN pathway and deployed acoustic Extra Trees pathway.
3. Extended module: explanations, subtype profiling, API serialization, patient records, web UI.

### 5.3 Top-Level Architecture (REPLACE ML ENGINE DESCRIPTION)

ML Engine:
- Baseline: DepressionCNN on mel spectrograms.
- Deployed: participant-level Extra Trees on aggregated acoustic features (`depression_acoustic_candidate.pkl`).
- Active model selected by `ACTIVE_MODEL` in `src/config.py` (`acoustic` for deployment).

### 5.3.2.4 Inference Aggregation (REPLACE)

For deployed acoustic model:
1. Extract per-segment acoustic vectors.
2. Aggregate with mean, std, median, p25, p75.
3. Predict depression probability with Extra Trees.
4. Apply validation-selected threshold (approximately 0.49).
5. Compute segment-level probabilities and occlusion importance for explanation.

### 5.3.3 Extended Module (ADD)

Occlusion importance:
- Remove one segment at a time from the participant bag.
- Measure absolute change in depression probability.
- Normalize importance scores for visualization and key-segment selection.

Patient store:
- Resolve patient identity by patient ID / ID number.
- Save analysis JSON and enable later review of prediction and explanations.

### 5.6 Summary (REPLACE)

The design supports a baseline CNN explainability pathway and a final deployed acoustic pathway under official labels, with shared explanation and deployment services.

---

## CHAPTER 6 — REPLACE IMPLEMENTATION FACTS

### 6.2.1 Project Structure (REPLACE MODELS / TRAIN FILES)

```
Data/
├── depressed/
├── non depressed/
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
├── full_test_split.csv
├── models/
│   ├── depression_acoustic_candidate.pkl
│   ├── acoustic_candidate_metadata.json
│   ├── depression_cnn.pt                  # baseline
│   ├── feature_model.pt                   # descriptive explanations
│   ├── participant_manifest.json
│   └── ... experimental candidates ...
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── explain.py
│   ├── predict.py
│   ├── patient_store.py
│   ├── ssl_model.py / ssl_predict.py
│   └── ...
├── train.py                      # CNN baseline
├── train_official_acoustic.py    # deployed model training
├── train_official.py             # WavLM official-label experiments
├── train_*.py                    # other experimental trainers
├── server.py
└── frontend/
```

### 6.4.5 Training Module (REPLACE)

Primary deployed training script: `train_official_acoustic.py`
- Official PHQ labels
- Participant-only speech
- Acoustic aggregation
- Extra Trees selected by CV balanced accuracy
- Threshold from OOF predictions
- One-shot held-out test evaluation

CNN baseline remains in `train.py` for Grad-CAM demonstration.

### 6.4.8 Inference Module (REPLACE)

`DepressionPredictor`:
- If `ACTIVE_MODEL == "acoustic"`: load Extra Trees artifact and run occlusion-based explanations.
- If `ACTIVE_MODEL == "ssl"`: WavLM pathway.
- Else: CNN + Grad-CAM pathway.

### 6.5 Algorithms (ADD ALGORITHM)

ALGORITHM: ParticipantAcousticInference(audio)
1. Load and preprocess audio; prefer participant-only speech when transcript available.
2. Segment into overlapping windows.
3. Extract 23 acoustic features per segment.
4. Aggregate feature statistics across segments.
5. probability ← ExtraTrees.predict_proba(aggregate)
6. prediction ← Depressed if probability ≥ threshold else Non-Depressed
7. For each segment i: importance_i ← |P_full − P_without_i|
8. Build timeline, feature ranking, and visualizations.
9. Return prediction + explanations.

### 6.8.2 Training Validation (REPLACE METRICS TABLE)

| Item | Value |
|---|---|
| Usable participants | 127 |
| Train+dev | 95 |
| Held-out test | 32 |
| Selected model | Extra Trees |
| Threshold | 0.49 |
| Held-out accuracy | 59.4% |
| Balanced accuracy | 60.4% |
| Depression recall | 63.6% |
| Specificity | 57.1% |
| F1 | 51.9% |
| ROC-AUC | 0.654 |

Do not present the early CNN 75% / 100% recording-level prototype as the final result.

### 6.9 Summary (REPLACE)

Implementation covers baseline CNN explainability and the final official-label acoustic Extra Trees deployment with occlusion explanations, API serving, frontend visualization, and patient-record management.

---

## CHAPTER 7 — REPLACE FULLY (EVALUATION)

### 7.1 Introduction

This chapter evaluates the final system under an official-label, speaker-independent protocol and compares it with experimental alternatives. Research questions are answered using the locked held-out official test participants.

RQ1: Can speech-based models classify depression under official PHQ labels and participant-level splits?
RQ2: Can explanations identify time regions and acoustic properties aligned with known depressive markers?
RQ3: Can explainability be integrated without replacing the predictive model’s decision logic?

### 7.2 Evaluation Strategy

Objectives:
1. Measure participant-level classification on official held-out test data.
2. Prevent leakage by keeping test participants out of model/threshold selection.
3. Compare alternative feature families and learning methods.
4. Assess explanation usefulness qualitatively.
5. Verify web deployment functionality.

Protocol:
- Labels: official AVEC 2017 PHQ-8 binary.
- Split: official train+dev vs full test.
- Model/threshold selection: train+dev only.
- Final metrics: one evaluation on held-out test.
- Folder names never used as ground truth.

### 7.3 Experimental Setup

Hardware: CPU workstation (GPU optional for WavLM/LoRA experiments).
Software: Python 3.9+, PyTorch, Librosa, scikit-learn, XGBoost, Transformers/PEFT, openSMILE, FastAPI.
Deployed inference: acoustic Extra Trees artifact; threshold ≈ 0.49.

### 7.4 Datasets and Test Cases

Dataset: DAIC-WOZ interview audio with official PHQ labels.
Main reported experiment: 127 usable participants (48 depressed, 79 non-depressed).
Partition: 95 train+dev / 32 official test.
Important data finding: approximately 39 folder labels conflicted with official PHQ labels and were corrected.

Test cases:
1. Official held-out participant classification (primary quantitative result).
2. Experimental candidate comparison (WavLM, COVAREP, eGeMAPS, multimodal, few-shot, LoRA, PHQ regression, hybrids).
3. Qualitative explanation inspection on sample recordings.
4. Web API and UI functional checks.

### 7.5 Participants

Interview subjects: officially labelled DAIC-WOZ participants with usable audio.
No formal clinician user study was conducted; evaluation of explanations is developer/researcher qualitative review.

### 7.6 Metrics

Accuracy, balanced accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, confusion matrix, bootstrap accuracy CI.
For screening, depression recall is treated as clinically important alongside accuracy.

### 7.7 Results and Analysis

#### 7.7.1 Deployed Model Held-Out Results (PRIMARY TABLE)

| Metric | Value |
|---|---:|
| Accuracy | 59.4% |
| Balanced accuracy | 60.4% |
| Precision | 43.8% |
| Depression recall | 63.6% |
| Specificity | 57.1% |
| F1-score | 51.9% |
| ROC-AUC | 0.654 |
| Accuracy 95% CI | 40.6% – 75.0% |
| Test participants | 32 |
| Confusion matrix | TN=12, FP=9, FN=4, TP=7 |

Interpretation:
The deployed model correctly classified 19 of 32 held-out participants and detected 7 of 11 depressed participants. Performance is moderate. It is preferable to high-accuracy candidates that miss most depressed cases.

#### 7.7.2 Invalid High-Accuracy Trap

An alternative thresholding regime produced 65.6% accuracy by predicting almost all participants as non-depressed, with 0% depression recall. That result is rejected as clinically invalid.

#### 7.7.3 Experimental Candidate Comparison

| Approach | Held-out accuracy | Notes |
|---|---:|---|
| Deployed Extra Trees acoustic | 59.4% | Best deployable balance; recall 63.6% |
| PHQ-8 severity regression | 68.8% | Higher accuracy but recall only 45.5%; not deployed |
| eGeMAPS + temporal | 56.2% | Strong CV, weaker test |
| Few-shot prototypes | 53.1% | Not better |
| Recall-constrained hybrid | 53.1% | Kept recall, lost accuracy |
| Segment-bag model | 50.0% | Overfit |
| Advanced gender-aware ensemble | 43.8% | Overfit |
| LoRA WavLM | 40.6% | High recall, very low specificity |
| Full-interview resampling | 39.5% | Did not generalize |

#### 7.7.4 Explainability Results

For the deployed acoustic model:
- Timeline cards show supporting/opposing seconds with acoustic cues.
- Occlusion importance identifies influential segments.
- Feature rankings commonly surface energy, pause-related, pitch-variability, and spectral/MFCC contributors.
- Spectrograms provide visual context.
- Subtype profiles remain research-only heuristics.

Clinical alignment is qualitative: depressed predictions more often cite low energy, longer pauses, and flatter pitch; non-depressed predictions more often cite stronger energy and more fluent timing.

#### 7.7.5 Web Application Evaluation

Functional checks passed for health endpoint, audio upload, prediction response, explanation tabs, and patient save/recall.

### 7.8 Comparison With Existing Methods

Published full-corpus multimodal systems often report higher scores, but differ in data scale, modalities, and evaluation details. This thesis prioritizes:
- official labels and locked splits,
- speech-only deployment practicality,
- integrated explanations,
- honest reporting of moderate performance.

The early CNN prototype on a tiny subset produced higher offline numbers (including segment-level F1 around 0.86 and perfect recording-level accuracy on 9 files). Those results are retained only as prototype history and are not the final claim of this thesis.

### 7.9 Discussion

RQ1: Yes, but only moderately under rigorous official-label evaluation (59.4% accuracy, 63.6% recall).
RQ2: Yes, qualitatively; explanations are time-localized and acoustically grounded.
RQ3: Yes; explanation methods are post-hoc wrappers around the predictive model.

Key finding: improving raw accuracy and preserving depression recall simultaneously was not achieved with the available usable audio. The PHQ regression candidate shows the trade-off clearly (68.8% accuracy vs 45.5% recall).

Limitations:
- 127 usable participants versus 189 official labels.
- Corrupted/unreadable archives (e.g., 440_P.zip).
- Speech-only scope.
- No prospective clinician study.
- Repeated experimentation on the same official test set increases caution about over-interpreting small gains.

### 7.10 Summary

Under official PHQ labels and speaker-independent evaluation, the deployed explainable acoustic model achieved 59.4% held-out accuracy with 63.6% depression recall. Explainability and deployment objectives were achieved. The 75–85% accuracy target was not honestly reached.

---

## CHAPTER 8 — REPLACE FULLY (CONCLUSION)

### 8.1 Introduction

This thesis designed, implemented, and evaluated an explainable framework for speech-based depression detection. The final contribution is not a claim of clinical-grade accuracy. It is a leakage-aware, explainable, deployable speech decision-support prototype grounded in official PHQ labels.

### 8.2 Contributions

1. Integrated explainable inference with timeline, feature, and attribution views.
2. Dual pathway: CNN+Grad-CAM baseline and deployed acoustic Extra Trees model.
3. Official-label correction and speaker-independent AVEC evaluation protocol.
4. Systematic comparison of alternative models and rejection of clinically invalid high-accuracy shortcuts.
5. Web deployment with patient-record management.
6. Responsible AI framing with disclaimers and uncertainty-aware presentation.

### 8.3 Achievement of Objectives

| Objective | Status |
|---|---|
| Literature and gap analysis | Achieved |
| Explainable framework design | Achieved |
| Official-label leakage-aware pipeline | Achieved |
| Model comparison and deployment | Achieved |
| Honest evaluation | Achieved |
| 75–85% held-out accuracy target | Not achieved |

### 8.4 Quantitative Summary (FINAL CLAIM)

| Item | Result |
|---|---|
| Deployed model | Extra Trees acoustic classifier |
| Participants (main experiment) | 127 |
| Held-out accuracy | 59.4% |
| Balanced accuracy | 60.4% |
| Depression recall | 63.6% |
| Specificity | 57.1% |
| F1 | 51.9% |
| ROC-AUC | 0.654 |
| Best non-deployed accuracy candidate | 68.8% (recall 45.5%) |

### 8.5 Limitations

Dataset size and missing/unreadable audio; speech-only modality; moderate discriminative signal; no clinician-rated explanation study; binary PHQ threshold oversimplifies severity; subtype module is heuristic.

### 8.6 Challenges

Label conflicts between folders and official PHQ CSVs; overfitting of high CV scores; accuracy–recall trade-offs; compute and dependency issues for WavLM/LoRA/XGBoost experiments; need to keep explanations faithful to the active model.

### 8.7 Future Work

1. Recover additional officially labelled audio and repair corrupted archives.
2. Nested cross-validation and external corpora.
3. Clinician evaluation of explanation usefulness.
4. Careful multimodal fusion with privacy constraints.
5. Severity-aware modelling that preserves screening recall.
6. Prospective decision-support study.

### 8.8 Summary

This thesis demonstrates that an explainable, officially evaluated, speech-only depression detection system can be built and deployed, but that honest held-out performance remains moderate with currently usable data. Transparency, leakage-aware evaluation, and clinical framing are therefore as important as accuracy claims.

---

## GLOBAL SEARCH-AND-REPLACE LIST

Replace these old claims everywhere:

| Old text | Replace with |
|---|---|
| 9 participants | 127 usable participants (main official-label experiment) |
| 108 segments as final dataset size | segment features aggregated to participant level; 127 participants |
| Primary model is CNN | Deployed model is Extra Trees acoustic classifier; CNN is baseline |
| Grad-CAM as only/main XAI | Occlusion importance for deployed model; Grad-CAM for CNN baseline |
| Accuracy 75% / F1 0.857 as final | Accuracy 59.4% / F1 51.9% held-out |
| Recording-level 100% on 9 files as final | Do not use as final claim |
| Threshold 0.5 as final | Threshold ≈ 0.49 selected on train/dev |
| Folder labels as ground truth | Official AVEC PHQ-8 labels |
| train.py as only trainer | train_official_acoustic.py for deployed model |
| Target 75–85% achieved | Target not achieved |

---

## SHORT EXAMINER-SAFE PARAGRAPH (OPTIONAL INSERT)

The research process began with a CNN-based mel-spectrogram prototype and Grad-CAM explanations on a small participant subset. After integrating official AVEC PHQ labels and speaker-independent evaluation, label noise and leakage-sensitive metrics became apparent. Multiple modern alternatives were tested. The final deployed system is a participant-level acoustic Extra Trees classifier with occlusion-based and feature-level explanations. Its held-out accuracy is 59.4% with 63.6% depression recall. Higher raw-accuracy candidates were available but reduced depression detection and were not deployed for screening use.
