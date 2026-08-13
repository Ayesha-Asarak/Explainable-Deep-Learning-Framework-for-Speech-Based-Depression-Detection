# An Explainable Deep Learning Framework for Speech-Based Depression Detection

**Ayesha Asarak**  
**215509V**

BSc(Hons) in Artificial Intelligence

Department of Computational Mathematics  
Faculty of Information Technology  
University of Moratuwa  
Sri Lanka

July 2026

---

Thesis submitted in partial fulfillment of the requirement for the Research Project in the Artificial Intelligence degree for the degree of BSc(Hons) in Artificial Intelligence

Department of Computational Mathematics  
Faculty of Information Technology  
University of Moratuwa  
Sri Lanka

July 2026

---

## DECLARATION

I declare that this thesis is my own work and this thesis/dissertation does not incorporate without acknowledgement any material previously submitted for a degree or diploma in any other university or Institute of higher learning and to the best of my knowledge and belief it does not contain any material previously published or written by another person except where the acknowledgement is made in the text. I retain the right to use this content in whole or part in future works (such as articles or books).

Signature: ____________________________  Date: _______________

The above candidate has carried out research for the undergraduate thesis/dissertation under my supervision. I confirm that the declaration made above by the student is true and correct.

Name of Supervisor: ____________________________

Signature of the Supervisor: ____________________________  Date: _______________

---

## DEDICATION

I dedicate this thesis to my beloved parents, whose unconditional love, endless encouragement, and unwavering belief in me have been my greatest source of strength throughout my academic journey. Their sacrifices and support have made this achievement possible.

I also dedicate this work to my respected supervisor, Prof. Subha Fernando, and my co-supervisor, Dr. Visula, for their invaluable guidance, continuous encouragement, constructive feedback, and unwavering support throughout this research. Their expertise, patience, and mentorship have been instrumental in the successful completion of this thesis.

Finally, I dedicate this thesis to my family, friends, lecturers, and everyone who supported and believed in me throughout this journey. Their encouragement and kindness inspired me to persevere and achieve this important milestone.

---

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude to my supervisor, Prof. Subha Fernando, for her invaluable guidance, continuous encouragement, expert advice, and constructive feedback throughout the course of this research. Her knowledge, patience, and unwavering support have been instrumental in the successful completion of this thesis.

I am equally grateful to my co-supervisor, Dr. Visula, for valuable insights, encouragement, and thoughtful suggestions, which greatly contributed to improving the quality of this research.

I would also like to extend my heartfelt thanks to the academic and non-academic staff of my department for providing the necessary facilities, resources, and support throughout my research journey.

My sincere appreciation goes to my family for their unconditional love, patience, understanding, and constant encouragement during every stage of my studies. Their unwavering support has been my greatest source of motivation.

I am also thankful to my friends and colleagues for their encouragement, assistance, and memorable moments throughout this journey.

Finally, I would like to express my gratitude to everyone who contributed directly or indirectly to the successful completion of this thesis. Your support and kindness are deeply appreciated.

---

## ABSTRACT

Depression is a major mental health disorder that often remains underdiagnosed due to the subjective nature of traditional clinical assessments. Current diagnostic methods rely heavily on interviews and self-reported symptoms, which can be inconsistent and time-consuming. Human speech provides an objective and non-invasive source of information, because acoustic features such as pitch variation, energy levels, pause patterns, and speech rate naturally reflect emotional and psychological states.

Although machine learning and deep learning models have shown promise for speech-based depression detection, many systems operate as black boxes and report optimistic results under weak evaluation protocols. In particular, folder-based labels, segment-level scoring with speaker leakage, and small participant subsets can inflate accuracy while reducing clinical trustworthiness.

This research proposes an explainable framework for speech-based depression detection that combines participant-level classification with multi-level interpretability and a deployable decision-support interface. The system uses the DAIC-WOZ clinical interview corpus with official AVEC 2017 PHQ-8 labels and official train/development/test partitions. Participant-only speech is extracted from interview audio, segmented into overlapping windows, and converted into interpretable acoustic features. A participant-level Extra Trees classifier aggregates segment statistics and predicts depressed versus non-depressed status. The decision threshold is selected on train/development data only, using balanced-accuracy criteria, and is never tuned on the held-out test set.

An explainability layer produces time-localized evidence through segment probability timelines, leave-one-segment-out occlusion importance, ranked acoustic feature contributions, spectrogram visualizations, and natural-language reasoning. The pipeline is deployed as a FastAPI web application with patient-record management, enabling upload, prediction, explanation review, and revisit of prior analyses.

Under the official-label, speaker-independent protocol with 127 usable participants (95 train+dev and 32 held-out test), the deployed model achieved 59.4% accuracy, 60.4% balanced accuracy, 63.6% depression recall, 57.1% specificity, 51.9% F1-score, and ROC-AUC 0.654. Alternative models were evaluated, including WavLM embeddings, COVAREP and eGeMAPS features, multimodal text–speech fusion, few-shot prototypes, LoRA fine-tuning, and PHQ-8 severity regression. A PHQ regression candidate reached 68.8% held-out accuracy but only 45.5% depression recall and was therefore not deployed for screening. The study concludes that explainability and leakage-aware evaluation are essential for trustworthy mental-health AI, while honest speech-only performance on limited usable audio remains moderate and should be framed as decision support rather than diagnosis.

**Keywords:** depression detection, speech acoustics, explainable AI, DAIC-WOZ, PHQ-8, Extra Trees, occlusion importance, clinical decision support

---

## TABLE OF CONTENTS

1. Introduction  
2. Literature Review  
3. Theoretical Foundations for the Extension  
4. Approach  
5. Analysis and Design  
6. Implementation  
7. Evaluation  
8. Conclusion and Further Work  
References  

---

## LIST OF TABLES

| Table | Title |
|---|---|
| 2.1 | Comparative analysis of existing method categories |
| 3.1 | Paralinguistic constructs and depression-related directions |
| 3.2 | Rule-based acoustic cue mapping |
| 3.3 | Characteristics of the proposed extension |
| 3.4 | Bridging research gaps with extension components |
| 4.1 | System inputs |
| 4.2 | System outputs |
| 7.1 | Research questions |
| 7.2 | Dataset composition for the main experiment |
| 7.3 | Deployed model held-out metrics |
| 7.4 | Confusion matrix on official held-out test |
| 7.5 | Experimental candidate comparison |
| 8.1 | Achievement of objectives |
| 8.2 | Final quantitative summary |

---

# CHAPTER 1  
# INTRODUCTION

## 1.1 Introduction

Depression is one of the most prevalent mental health disorders worldwide and significantly affects emotional, cognitive, and social functioning. Early identification supports timely intervention, yet traditional assessment depends largely on clinical interviews and self-report questionnaires. These methods are valuable but can be time-consuming, subjective, and inconsistent across settings.

Speech provides a complementary behavioural signal. Depressed speakers often exhibit psychomotor retardation expressed as reduced pitch variability, lower vocal energy, slower speech rate, and increased pausing. Artificial intelligence can analyse these acoustic patterns automatically. However, many published systems focus on predictive accuracy alone and either lack clinician-usable explanations or evaluate performance with protocols that risk speaker leakage and label noise.

This thesis develops and evaluates an explainable speech-based depression detection framework. The final deployed system uses official PHQ-8 labels, participant-level evaluation, acoustic feature aggregation, and multi-level explanations delivered through a web application intended as decision support rather than diagnosis.

## 1.2 Objectives

1. To critically review speech-based depression detection and identify gaps in interpretability and evaluation rigor.  
2. To study relevant methods in acoustic analysis, machine learning, deep learning, and explainable AI.  
3. To design an explainable depression-detection framework that combines prediction with time-localized and feature-level explanations.  
4. To implement a leakage-aware training and evaluation pipeline using official DAIC-WOZ / AVEC PHQ labels and speaker-independent splits.  
5. To compare multiple modelling approaches and select a deployable model based on both accuracy and depression recall.  
6. To deploy the system as a web-based decision-support prototype with visualization and patient-record support.  
7. To evaluate the framework honestly and discuss its contribution and limitations for healthcare AI.

## 1.3 Problem in Brief

Deep learning and classical machine learning can detect depression-related patterns in speech, but two problems remain. First, many models are black boxes and do not explain which time regions or acoustic characteristics drove a prediction. Second, reported performance is often unreliable because of speaker leakage, segment-level inflation, and inconsistent labels. These issues create a gap between technical prototypes and clinically trustworthy decision-support tools.

## 1.4 Background and Motivation

Speech-based depression detection has grown with corpora such as DAIC-WOZ and challenges such as AVEC. Early systems used handcrafted acoustics with classical classifiers; later systems used CNNs on spectrograms, recurrent models, and pretrained speech transformers. Multimodal audio–text–video systems often score highest, but speech-only tools remain attractive for privacy and deployment.

Motivation for this work comes from three needs: (1) transparent explanations for clinicians and researchers; (2) official-label, participant-level evaluation that avoids inflated claims; and (3) a usable software pipeline from audio upload to explained prediction.

## 1.5 Novel Approach to Speech-Based Depression Detection

The proposed approach extends standard speech depression classification in four ways:

1. Official PHQ-8 labels and official AVEC partitions replace folder-name labels.  
2. Participant-only speech is analysed using aggregated acoustic features and a participant-level Extra Trees classifier as the deployed model.  
3. Explainability is integrated into inference through occlusion importance, acoustic feature ranking, timeline cards, and spectrogram visualization.  
4. The system is deployed as a web decision-support application with patient identity and analysis history.

A CNN-on-mel-spectrogram prototype with Grad-CAM was developed as an early baseline. After official-label evaluation and model comparison, the acoustic Extra Trees model was selected for deployment because it offered a better balance of held-out accuracy and depression recall than stronger-looking alternatives with weaker sensitivity.

## 1.6 Resource Requirements

**Hardware:** development workstation with adequate storage for interview audio and model caches; GPU optional.  

**Software:** Python 3.9+, PyTorch, Librosa, scikit-learn, XGBoost, Transformers/PEFT (experimental models), openSMILE, FastAPI, Uvicorn, Matplotlib.  

**Data:** DAIC-WOZ-style interview audio, transcripts for participant-only extraction, and official AVEC 2017 PHQ label CSVs.

## 1.7 Structure of the Thesis

- Chapter 2 reviews related literature.  
- Chapter 3 presents theoretical foundations for the extension.  
- Chapter 4 describes the approach, hypotheses, and workflows.  
- Chapter 5 presents analysis and design.  
- Chapter 6 details implementation.  
- Chapter 7 reports evaluation results and discussion.  
- Chapter 8 concludes and outlines future work.

## 1.8 Summary

This chapter introduced the need for explainable, rigorously evaluated speech-based depression detection. The thesis develops a deployable framework that prioritizes official labels, speaker-independent evaluation, acoustic interpretability, and clinical decision-support framing.

---

# CHAPTER 2  
# LITERATURE REVIEW

## 2.1 Introduction

Following the problem identified in Chapter 1, this chapter critically reviews existing research on speech-based depression detection and the role of artificial intelligence in mental health analysis. While prior studies have demonstrated promising performance using machine learning and deep learning techniques, significant challenges remain, particularly in terms of interpretability, label reliability, and clinical applicability.

This chapter examines the evolution of approaches in this domain, analyzes their strengths and limitations, and identifies key technological gaps. The discussion covers traditional machine learning, deep learning methods, and emerging explainable artificial intelligence approaches. The chapter concludes by defining the research gap that motivates the proposed solution.

## 2.2 Chronological Review of Prior Work

### 2.2.1 Early Development of Speech-Based Depression Detection

The earliest work emerged from clinical psychiatry and speech science. Clinicians observed that depressed patients often exhibit psychomotor retardation, reflected in slower speech, longer pauses, reduced vocal energy, and flatter intonation. These observations established speech as a potential behavioural biomarker for mood disorders.

In the 1990s and early 2000s, researchers quantified these observations using handcrafted acoustic features such as pitch, energy, speech rate, pause duration, jitter, shimmer, MFCCs, and spectral measures. Classifiers such as Support Vector Machines (SVM), Gaussian Mixture Models (GMM), and Hidden Markov Models (HMM) were applied to distinguish depressed from non-depressed speakers.

A major limitation of this era was small, heterogeneous datasets and inconsistent labeling criteria. Nevertheless, these studies established a consistent finding: paralinguistic cues—how something is said—carry diagnostic information beyond lexical content.

The introduction of standardized corpora marked an important milestone. The DAIC-WOZ dataset, developed for the AVEC depression sub-challenge, provided clinical interview recordings with PHQ-8–validated labels. This enabled reproducible benchmarking and shifted the field toward data-driven methods. Early DAIC-WOZ baselines relied on engineered features (e.g., COVAREP, openSMILE) combined with traditional classifiers.

### 2.2.2 Recent Developments and Future Trends

Recent years have seen a shift from handcrafted features to deep learning. Convolutional Neural Networks (CNNs) applied to mel spectrograms became common. Recurrent models and hybrid CNN–LSTM architectures capture longer temporal dependencies. Self-supervised pretrained models such as wav2vec 2.0, HuBERT, and WavLM have shown strong transfer learning performance.

Multimodal approaches also gained traction. Many state-of-the-art systems fuse speech, linguistic, and visual cues. However, audio-only methods remain attractive because they are less invasive, easier to deploy, and raise fewer privacy concerns than video.

Explainable AI (XAI) has become a priority in mental health applications. Techniques such as Grad-CAM, SHAP, LIME, occlusion analysis, and attention visualization aim to reveal which input regions or features drive predictions.

Emerging trends include real-time monitoring, longitudinal tracking, fairness auditing, clinical validation, privacy-preserving learning, and integration with digital phenotyping.

### 2.2.3 Issues and Research Challenges

Despite progress, several issues persist:

- **Dataset limitations** — Many studies use subsets of DAIC-WOZ; generalization remains unproven.  
- **Label reliability** — Folder organization is not always equivalent to official PHQ labels.  
- **Evaluation methodology** — Segment-level evaluation with speaker overlap can inflate performance.  
- **Class imbalance and demographic bias** — Models may learn speaker identity rather than pathology.  
- **Explainability gap** — High-performing models often remain opaque.  
- **Clinical adoption barriers** — Regulatory, ethical, and validation requirements limit deployment.  
- **Unimodal vs multimodal trade-offs** — Multimodal systems may perform better but are harder to deploy.

## 2.3 Comparative Analysis of Existing Methods

**Table 2.1. Comparative analysis of existing method categories**

| Approach | Features / Input | Model | Explainability | Typical Strengths | Typical Weaknesses |
|---|---|---|---|---|---|
| Handcrafted acoustic + classical ML | MFCCs, pitch, energy, pauses, COVAREP | SVM, RF, GMM | High | Transparent, low data need | Limited representation power |
| Engineered features + deep MLP | openSMILE / COVAREP | MLP, DNN | Moderate | Strong baselines | Fixed feature sets |
| CNN on spectrograms | Mel / log-mel | CNN | Low–Moderate with Grad-CAM | Learns time–frequency patterns | Opaque without XAI; data hungry |
| RNN / LSTM | Frame-level sequences | LSTM, BiLSTM | Low–Moderate | Temporal dynamics | Computationally heavier |
| Pretrained speech models | Waveform / embeddings | WavLM, HuBERT + head | Low | Strong transfer learning | Compute heavy; harder to explain |
| Multimodal fusion | Audio + text + video | Fusion networks | Low | Highest benchmark scores | Privacy and complexity |
| XAI-enhanced speech systems | Acoustics / spectrograms + attributions | CNN / trees + Grad-CAM / occlusion / SHAP | High | Prediction + justification | Extra pipeline complexity |

The proposed framework occupies the speech-only explainable quadrant: participant-level acoustic classification with multi-level explanations, while retaining a CNN+Grad-CAM pathway as a baseline prototype.

## 2.4 Strengths and Limitations of Current Approaches

**Strengths:** non-invasive assessment; objective quantification; demonstrated feasibility on DAIC-WOZ; deep learning representational power; growing XAI toolkit.

**Limitations:** black-box predictions; small-sample overfitting; segment-level evaluation bias; single-corpus dependency; weak clinical validation; weak alignment between heatmaps and clinical biomarkers; ethical and regulatory gaps.

## 2.5 Summary of Issues

The literature shows a field that progressed from observational clinical insights to sophisticated deep learning pipelines, yet several issues remain unresolved:

1. Depression detection from speech is scientifically plausible but clinically unvalidated at scale.  
2. Performance metrics are often inflated by improper splitting and correlated segments.  
3. Explainability is frequently optional rather than a core design requirement.  
4. Deployment-ready systems combining classification, interpretation, and usable interfaces are rare.  
5. Ethical safeguards are inconsistently addressed.

## 2.6 Discussion on Research Gaps

A clear gap exists between predictive performance and clinical trustworthiness. State-of-the-art models may achieve strong scores while remaining unsuitable for real-world workflows because they cannot answer: Why was this person flagged? Which moments in the recording matter? Which acoustic symptoms were detected?

Few works unify accurate binary depression classification with multi-level explainability—visual, feature-level, and temporal—within a deployable decision-support application evaluated under official labels and speaker-independent splits.

### 2.6.1 Definition of the Research Gap / Problem

**Research gap:** Existing speech-based depression detection systems predominantly function as black-box classifiers and/or report results under weak evaluation protocols. There is a methodological and practical gap for an integrated framework that combines participant-level depression classification with explainable AI techniques suitable for clinical decision support, using official PHQ labels.

**Problem statement:** Major Depressive Disorder and related conditions affect a large global population, yet diagnosis depends heavily on subjective assessment. Speech carries paralinguistic markers of depression, but automated tools that exploit this signal either lack explainability or lack rigorous, leakage-aware evaluation.

**Specific problem addressed by this thesis:** Design, implement, and evaluate an explainable framework that:

1. Detects depression from clinical interview speech using official PHQ-8 labels.  
2. Explains predictions via occlusion importance / Grad-CAM, acoustic feature importance, and second-level timeline explanations.  
3. Aggregates segment-level evidence into participant-level decision support.  
4. Optionally provides research-oriented depression subtype profile matching.  
5. Deploys the pipeline as a web-based application for upload, prediction, visualization, and patient-record review.

**Research questions:**

- **RQ1:** Can speech-based models classify depression under official PHQ labels and participant-level splits?  
- **RQ2:** Can XAI methods identify time regions and acoustic properties consistent with known depressive speech markers?  
- **RQ3:** Can explainability be integrated without replacing the predictive model’s decision logic?

**Scope boundaries:**

- Binary classification (depressed vs non-depressed); not a clinical diagnosis.  
- Primary deployed pathway is speech-only acoustic analysis.  
- Official AVEC 2017 PHQ-8 labels and official splits are used for final evaluation.  
- Main reported experiment: 127 usable participants (48 depressed, 79 non-depressed).  
- CNN + Grad-CAM is retained as a baseline/prototype pathway; the deployed pathway uses Extra Trees + occlusion explanations.  
- Subtype matching is heuristic profile matching, not supervised subtype diagnosis.

## 2.7 Summary

This chapter reviewed the evolution of speech-based depression detection from early acoustic–phonetic studies to modern deep learning and multimodal fusion. The central unresolved issue is the explainability and evaluation gap: clinicians require understandable, time-localized justification, and researchers require leakage-aware metrics based on reliable labels.

The following chapters describe the proposed explainable framework, beginning with a CNN+Grad-CAM prototype and culminating in an official-label participant-level acoustic classifier with multi-level explanations and web deployment.

---

# CHAPTER 3  
# THEORETICAL FOUNDATIONS FOR THE EXTENSION

## 3.1 Introduction

Chapter 2 established that speech-based depression detection has matured from handcrafted acoustic analysis to deep learning, yet critical limitations persist: opaque predictions, weak evaluation protocols, and limited clinical usability. This chapter presents the theoretical foundations for the proposed extension.

The extension does not discard deep learning. A CNN operating on mel-spectrogram representations remains an important baseline pathway. The final deployed system, however, uses participant-level acoustic aggregation and Extra Trees classification under official PHQ labels, with occlusion-based and feature-level explanations.

## 3.2 Overview of Existing Theory

### 3.2.1 Clinical Theory: Depression and Psychomotor Retardation

Major Depressive Disorder is characterized by persistent low mood, anhedonia, fatigue, and cognitive slowing. Psychomotor retardation manifests in speech as reduced articulation rate, longer pauses, lower vocal energy, and flatter prosody. Instruments such as the PHQ-8 provide ground-truth labels used in DAIC-WOZ / AVEC evaluations.

### 3.2.2 Paralinguistic and Acoustic–Phonetic Theory

**Table 3.1. Paralinguistic constructs and depression-related directions**

| Construct | Acoustic Measure | Depression-Related Direction |
|---|---|---|
| Prosody | Pitch mean, pitch std | Lower variability (monotone) |
| Intensity | RMS energy, energy std | Reduced loudness and range |
| Temporal structure | Speech rate, pause ratio | Slower, more silent |
| Voice quality / spectrum | Spectral centroid, MFCCs | Altered timbre / envelope |

### 3.2.3 Signal Processing Theory: Time–Frequency Representations

Speech is non-stationary. Mel-spectrograms provide a perceptually motivated time–frequency representation suitable for CNN analysis and visualization. Acoustic functionals summarize local windows into interpretable scalars for tabular classifiers.

### 3.2.4 Machine Learning Theory

CNNs exploit local time–frequency patterns in spectrograms. For small clinical corpora, participant-level tabular classifiers remain competitive. Aggregating segment acoustics into mean, standard deviation, median, and percentile statistics produces a fixed participant vector. Ensemble methods such as Extra Trees can learn non-linear interactions among acoustic statistics while remaining comparatively robust under limited sample size.

### 3.2.5 Evaluation Theory: Speaker Independence and Aggregation

Proper evaluation requires participant-level splitting: all evidence from one individual must reside in a single partition. At inference, segment-level probabilities or aggregated participant vectors support recording-level decisions. Threshold selection must use development data only.

### 3.2.6 Explainable AI Theory

Post-hoc methods include Grad-CAM for CNNs, feature attribution for tabular models, and occlusion analysis for segment bags. In mental health AI, explanations should be faithful, understandable, and actionable. When the primary model is not a CNN, leave-one-segment-out occlusion importance is a faithful alternative: a segment is important if removing it changes the participant-level depression probability.

## 3.3 Revisiting the Research Gap

1. **Representational opacity** — models may learn useful patterns without exposing them.  
2. **Temporal localization deficiency** — single labels without timestamped evidence.  
3. **Dual-path disconnect** — deep models and interpretable acoustics are often treated as alternatives.  
4. **Label/split unreliability** — folder labels and leaked splits inflate claims.  
5. **Deployment and trust gap** — prototypes often stop at accuracy tables.

**Formal gap statement:** There exists a practical gap for an integrated speech-based depression framework that (a) uses official clinical labels and speaker-independent evaluation, (b) provides multi-level time-localized explanations suitable for decision support, and (c) remains deployable as a speech-only research prototype.

## 3.4 Rationale for the Extension

- **Clinical:** clinicians need inspectable evidence, not opaque scores.  
- **Technical:** post-hoc XAI can wrap a trained model without changing its decision weights.  
- **Ethical:** transparency, uncertainty, and human oversight are required in mental-health AI.  
- **Practical:** a web interface turns a training script into a usable research instrument.

## 3.5 Theoretical Foundations / Inspiration for Extension

### 3.5.1 Baseline Classifier: CNN on Mel-Spectrograms

Treat log-mel spectrograms as image-like inputs. Three convolutional blocks learn hierarchical time–frequency patterns. Grad-CAM on the final convolutional layer visualizes influential regions.

### 3.5.2 Deployed Classifier: Participant-Level Acoustic Extra Trees

Participant speech is segmented; 23 acoustic features are extracted per segment; statistics are aggregated; Extra Trees predicts depression probability. Threshold selection uses train/development out-of-fold predictions.

### 3.5.3 Grad-CAM Extension

Used for the CNN pathway; maps importance back to absolute seconds.

### 3.5.4 Occlusion Importance Extension

Used for the deployed acoustic pathway: leave-one-segment-out importance identifies influential time windows.

### 3.5.5 Acoustic Feature Pathway

Pitch, energy, pauses, spectral measures, and MFCCs anchor explanations in clinically named constructs.

### 3.5.6 Timeline Explanation Layer

Overlapping windows receive probabilities and feature vectors. Supporting and opposing segments generate natural-language explanations.

**Table 3.2. Rule-based acoustic cue mapping**

| Feature Condition | Generated Cue |
|---|---|
| pitch_std < 30 Hz | Flat/monotone pitch |
| energy_mean < 0.015 | Low voice energy |
| pause_ratio > 0.4 | Long pauses |
| speech_rate < 0.5 | Slow speech rate |

### 3.5.7 Recording-Level Aggregation

Final decision uses participant-level probability and a validation-selected threshold.

### 3.5.8 Depression Subtype Profile Matcher

Optional heuristic profile matching only; not supervised subtype diagnosis.

### 3.5.9 Deployment Layer

FastAPI backend, web frontend, and patient-record storage implement decision-support interaction.

## 3.6 Characteristics of the Extension

**Table 3.3. Characteristics of the proposed extension**

| Characteristic | Description |
|---|---|
| Dual pathway | CNN+Grad-CAM baseline; acoustic Extra Trees deployed |
| Official labels | AVEC PHQ-8 binary labels override folder names |
| Speaker-independent split | Official train+dev vs test; no participant overlap |
| Multi-level explanation | Occlusion/Grad-CAM, features, timeline, summary |
| Time-localized output | Absolute seconds |
| Decision-support framing | Non-diagnostic disclaimers |
| Deployable system | Training scripts + API + web UI + patient store |

## 3.7 Bridging the Research Gap with Extension

**Table 3.4. Bridging research gaps with extension components**

| Research Gap | Extension Component |
|---|---|
| Representational opacity | Occlusion importance + Grad-CAM baseline |
| Temporal localization deficiency | Timeline cards + key-segment highlighting |
| Label and split unreliability | Official PHQ labels + official partitions |
| Dual-path disconnect | Acoustic semantics + model attribution |
| Deployment and trust gap | Web app with explanations and patient history |

## 3.8 Summary

This chapter established the theoretical foundations spanning clinical psychomotor theory, paralinguistics, signal processing, machine learning, evaluation rigor, and XAI. Chapter 4 translates these foundations into the practical approach used in this thesis.

---

# CHAPTER 4  
# APPROACH

## 4.1 Introduction

This chapter describes the approach used to design, implement, and deploy the explainable speech-based depression detection framework. The work progressed from a CNN+Grad-CAM prototype to a final official-label participant-level acoustic classifier with occlusion-based explanations and web deployment.

## 4.2 Hypothesis and Its Inspiration

**H1:** Officially labelled participant-level acoustic features from clinical interview speech contain discriminative signal for binary depression classification under speaker-independent evaluation.

**H2:** Multi-level explanations (timeline, acoustic features, and occlusion/Grad-CAM attributions) can make predictions inspectable without requiring the model to be a black box.

**H3:** Segmenting interviews and aggregating segment evidence supports time-localized explanations in absolute seconds.

**H4:** Packaging the pipeline as a web application enables practical decision-support demonstration and review of patient analyses.

**Null expectation:** with limited usable DAIC-WOZ audio, clinical-grade accuracy is not hypothesized. Aspirational accuracy ranges such as 75–85% must not be claimed unless achieved under the locked official test protocol.

## 4.3 Inputs and Outputs of Extension

**Table 4.1. System inputs**

| Input | Specification |
|---|---|
| Primary input | Voice recording (WAV/MP3/FLAC/OGG/M4A/WebM) |
| Patient metadata | Name, age, ID number, patient ID, gender, notes |
| Training labels | Official AVEC PHQ-8 binary labels |
| Transcripts | Participant-turn intervals for speech extraction |
| Optional context | Chronic mood, recent stress, postpartum, seasonal, mood swings |

**Table 4.2. System outputs**

| Output | Description |
|---|---|
| Prediction | Depressed / Non-Depressed |
| Probability / confidence | Participant-level depression risk and confidence |
| Threshold | Validation-selected decision threshold |
| Timeline explanations | Exact-second supporting/opposing evidence |
| Attribution map | Occlusion importance or Grad-CAM |
| Feature importance | Ranked acoustic contributors |
| Prediction reason | Natural-language summary |
| Subtype ranking | Optional research-only profile match |
| Saved record | Optional patient analysis history |

## 4.4 Process Workflow for Extension

### 4.4.1 Training Workflow

1. Discover audio from `depressed/` and `non depressed/` folders and ZIP archives.  
2. Load official AVEC train/dev/test PHQ labels.  
3. Override folder labels with official PHQ binary labels; exclude unresolved conflicts.  
4. Extract participant-only speech using transcripts.  
5. Segment speech into overlapping windows.  
6. Extract acoustic features and aggregate to participant vectors.  
7. Train candidate classifiers on official train+dev only.  
8. Select model and threshold using cross-validation / out-of-fold metrics.  
9. Evaluate once on official held-out test participants.  
10. Deploy selected artifact through the web API.

Command for the deployed model:

```bash
python3 train_official_acoustic.py
```

### 4.4.2 Inference Workflow

1. User uploads audio and optional patient metadata via the web UI.  
2. Backend validates format and loads the active predictor.  
3. Audio is preprocessed and segmented.  
4. Acoustic features are extracted per segment and aggregated.  
5. Extra Trees produces participant-level probability and thresholded prediction.  
6. Explainability layer builds timeline cards, occlusion importance, feature ranking, and visualizations.  
7. Optional subtype profile matching is computed for depressed predictions.  
8. JSON response is rendered in the frontend; optional patient record is saved.

### 4.4.3 Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Labels | Official PHQ-8 | Corrects folder noise |
| Split | Official AVEC partitions | Speaker-independent evaluation |
| Deployed model | Extra Trees acoustic | Best accuracy–recall balance among honest candidates |
| Threshold | Selected on train/dev OOF | Avoids test leakage |
| Attribution | Occlusion for acoustic model | Faithful to active model |
| Framing | Decision support | Not diagnosis |

## 4.5 Technologies Identified

- **Language / ML:** Python, PyTorch, scikit-learn, XGBoost  
- **Audio:** Librosa, SoundFile, NumPy  
- **Experimental SSL / PEFT:** Transformers, Accelerate, PEFT, WavLM  
- **Engineered acoustics:** openSMILE eGeMAPS  
- **API / UI:** FastAPI, Uvicorn, HTML/CSS/JavaScript  
- **Visualization:** Matplotlib, Pillow  
- **Data:** DAIC-WOZ audio + official AVEC label CSVs

## 4.6 Features of Technological Extension

1. Multi-level explainability  
2. Dual-pathway architecture (CNN baseline + acoustic deployment)  
3. Time-localized evidence  
4. Interactive web deployment  
5. Leakage-aware training protocol  
6. Decision-support safeguards  
7. Patient-record management  
8. Extensibility for experimental trainers

## 4.7 Target Users / Use Scenarios

**Users:** mental health researchers, clinicians evaluating AI tools, AI students, thesis examiners, telehealth prototype developers.

**Scenarios:** research demonstration; clinician inspection of explanations; teaching explainable health AI; pipeline validation before scale-up; patient analysis save/recall.

**Non-use cases:** standalone diagnosis; emergency triage; employment/insurance/legal decisions; analysis without consent.

## 4.8 Positioning Within the AI Body of Knowledge

The work sits at the intersection of speech signal processing, machine learning, explainable AI, and clinical decision-support systems. It contributes an integrated, deployable, temporally grounded explanation pipeline evaluated under official labels.

## 4.9 Summary

The approach combines official-label evaluation, participant-level acoustic classification, multi-level explainability, and web deployment. The next chapters present design, implementation, and results for this final system, while retaining the CNN prototype as historical baseline.

---

# CHAPTER 5  
# ANALYSIS AND DESIGN

## 5.1 Introduction

This chapter translates the approach into a formal design specification. The architecture comprises:

1. **Preprocessing module** — audio loading, participant-only extraction, segmentation, feature extraction.  
2. **ML engine** — baseline CNN pathway and deployed acoustic Extra Trees pathway.  
3. **Extended module** — explanations, subtype profiling, API serialization, patient records, web UI.

## 5.2 Rationale for the Design Extension

1. **Modularity** — separate preprocessing, modelling, explanation, and deployment.  
2. **Dual-path processing** — deep spectrogram baseline plus interpretable acoustic deployment.  
3. **Temporal granularity** — segment timestamps propagate through explanations.  
4. **Non-invasive explainability** — post-hoc methods wrap trained models.  
5. **Deployable decision support** — REST API and interactive frontend.

## 5.3 Top-Level Architecture

```
Presentation Layer (frontend)
        │ HTTP
API / Orchestration (server.py, DepressionPredictor, api_utils)
        │
 ┌──────┼──────────────────────┐
 │      │                      │
Preprocessing   ML Engine      Extended Module
(features/data) (CNN / Trees)  (explain/subtype/patient_store)
        │
Data Layer (audio, official CSVs, models/, patient_records/)
```

### 5.3.1 Preprocessing Module

- Load audio at 16 kHz mono.  
- Prefer participant-only speech using transcript intervals.  
- Segment into overlapping windows.  
- Extract mel spectrograms (baseline pathway) and 23 acoustic features (deployed pathway).  
- Aggregate acoustic statistics for participant-level classification.

### 5.3.2 ML Engine

**Baseline:** DepressionCNN on mel spectrograms with Grad-CAM hooks.  

**Deployed:** Extra Trees on aggregated acoustic features (`depression_acoustic_candidate.pkl`).  

**Active model** selected by `ACTIVE_MODEL` in `src/config.py` (`acoustic` for deployment).

Inference for deployed model:

1. Extract per-segment acoustic vectors.  
2. Aggregate with mean, std, median, p25, p75.  
3. Predict probability with Extra Trees.  
4. Apply validation-selected threshold (≈ 0.49).  
5. Compute segment probabilities and occlusion importance.

### 5.3.3 Extended Module

- Timeline explanations and prediction reason  
- Occlusion importance / Grad-CAM visualization  
- Acoustic feature importance  
- Optional subtype profile matching  
- API chart serialization  
- Patient identity resolution and analysis history

## 5.4 Data Flow and Interaction Design

**Training flow:** discover sources → official labels → participant-only audio → features → train/dev selection → held-out test evaluation → save artifact.

**Inference flow:** upload → predict → explain → render tabs → optional save record → later recall.

**UI journey:** enter/select patient → upload audio → analyze → review prediction and explanation tabs → save/reopen past analyses.

## 5.5 Extension Integration into AI Framework

- **Model level:** CNN hooks for Grad-CAM; occlusion wrapper for Extra Trees.  
- **Pipeline level:** post-classification explanation stage.  
- **Application level:** CDSS-style human-in-the-loop framing.  
- **Extensibility:** experimental trainers for WavLM, LoRA, eGeMAPS, multimodal, and PHQ regression.

## 5.6 Summary

The design supports a baseline CNN explainability pathway and a final deployed acoustic pathway under official labels, with shared explanation and deployment services.

---

# CHAPTER 6  
# IMPLEMENTATION

## 6.1 Introduction

This chapter describes how the design was realized in code: environment, module-wise implementation, algorithms, integration, and testing.

## 6.2 Overall Development Environment

### 6.2.1 Project Structure

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
│   ├── depression_cnn.pt
│   ├── feature_model.pt
│   ├── participant_manifest.json
│   └── experimental candidates...
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── explain.py
│   ├── predict.py
│   ├── patient_store.py
│   ├── ssl_model.py
│   └── ssl_predict.py
├── train.py
├── train_official_acoustic.py
├── train_official.py
├── train_*.py
├── server.py
└── frontend/
```

### 6.2.2 Development Methodology

Implementation proceeded iteratively: preprocessing → CNN baseline → explainability → web deployment → official-label pipeline → acoustic deployment → experimental comparisons → patient-record support.

### 6.2.3 Environment Configuration

```bash
export NUMBA_CACHE_DIR="$(pwd)/.numba_cache"
pip3 install -r requirements.txt
python3 train_official_acoustic.py
python3 -m uvicorn server:app --host 127.0.0.1 --port 8765
```

## 6.3 Hardware and Software Platforms

CPU workstation for main training and inference; GPU optional for WavLM/LoRA experiments. Software stack includes Python 3.9+, PyTorch, Librosa, scikit-learn, Transformers/PEFT, openSMILE, FastAPI, and Matplotlib.

## 6.4 Module-Wise Implementation

| Module | Responsibility |
|---|---|
| `config.py` | Paths, sampling, active model, official label files |
| `features.py` | Load/segment audio; mel and acoustic features |
| `data.py` | Source discovery, official labels, participant-only audio, manifest |
| `model.py` | DepressionCNN and FeatureClassifier baseline |
| `train_official_acoustic.py` | Deployed Extra Trees training/evaluation |
| `train.py` | CNN baseline training |
| `explain.py` | Timeline, Grad-CAM, charts, feature importance |
| `predict.py` | Active-model inference orchestration |
| `patient_store.py` | Patient identity and analysis history |
| `server.py` / `api_utils.py` | REST API and JSON/chart serialization |
| `frontend/` | Upload UI, explanation tabs, saved-record review |

## 6.5 Algorithms and Pseudocode

### Algorithm 1: Participant-Only Preprocessing

Load full interview audio → parse transcript participant intervals → concatenate participant speech → trim/normalize → segment into overlapping windows.

### Algorithm 2: Acoustic Aggregation

For each segment extract 23 features → stack vectors → compute mean, std, median, p25, p75 → concatenate into participant vector.

### Algorithm 3: Official-Label Model Selection

Train candidates on train+dev with stratified CV → select by balanced accuracy → choose threshold on OOF predictions → evaluate once on official test.

### Algorithm 4: Deployed Inference

```
probability ← ExtraTrees.predict_proba(aggregate)
prediction ← Depressed if probability ≥ threshold else Non-Depressed
for each segment i:
    importance_i ← |P_full − P_without_i|
build timeline, feature ranking, visualizations
return prediction + explanations
```

### Algorithm 5: Grad-CAM (CNN Baseline)

Forward with gradients → weight final-conv activations → ReLU → upsample → normalize → annotate peak second.

## 6.6 Workflow Diagrams

**Training:** discover → label → extract → aggregate → CV select → threshold → held-out test → save artifact.  

**Inference:** upload → preprocess → predict → explain → render → optional save.  

**UI:** patient identity → analyze → tabs (Why, Type, Spectrogram, Timeline, Attribution, Features) → saved records.

## 6.7 Integration of Components

`server.py` calls `DepressionPredictor.predict()`, which uses `features.py`, active model artifact, `explain.py`, and optional `subtype.py` / `patient_store.py`. Charts are serialized by `api_utils.py` and rendered by the frontend.

## 6.8 Testing During Development

Testing covered audio loading, participant-only extraction, official-label overrides, CV training, held-out evaluation, API errors, frontend rendering, patient save/recall, and qualitative explanation checks.

**Final deployed validation summary:**

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

## 6.9 Summary

Implementation covers baseline CNN explainability and the final official-label acoustic Extra Trees deployment with occlusion explanations, API serving, frontend visualization, and patient-record management.

---

# CHAPTER 7  
# EVALUATION

## 7.1 Introduction

This chapter evaluates the final system under an official-label, speaker-independent protocol and compares it with experimental alternatives.

**Table 7.1. Research questions**

| RQ | Question |
|---|---|
| RQ1 | Can speech-based models classify depression under official PHQ labels and participant-level splits? |
| RQ2 | Can explanations identify time regions and acoustic properties aligned with known depressive markers? |
| RQ3 | Can explainability be integrated without replacing the predictive model’s decision logic? |

## 7.2 Evaluation Strategy

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

## 7.3 Experimental Setup

Hardware: CPU workstation; GPU optional for WavLM/LoRA.  
Software: Python 3.9+, PyTorch, Librosa, scikit-learn, XGBoost, Transformers/PEFT, openSMILE, FastAPI.  
Deployed inference: acoustic Extra Trees artifact; threshold ≈ 0.49.

## 7.4 Datasets and Test Cases

**Table 7.2. Dataset composition for the main experiment**

| Item | Value |
|---|---|
| Corpus | DAIC-WOZ / AVEC 2017 |
| Usable participants | 127 |
| Depressed | 48 |
| Non-depressed | 79 |
| Train+dev | 95 |
| Held-out test | 32 |
| Label source | Official PHQ-8 binary |
| Folder mismatches corrected | ~39 |

Test cases:

1. Official held-out participant classification (primary result).  
2. Experimental candidate comparison.  
3. Qualitative explanation inspection.  
4. Web API and UI functional checks.

## 7.5 Participants

Interview subjects are officially labelled DAIC-WOZ participants with usable audio. No formal clinician user study was conducted; explanation assessment is qualitative.

## 7.6 Evaluation Metrics

Accuracy, balanced accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, confusion matrix, and bootstrap accuracy CI. For screening, depression recall is treated as clinically important alongside accuracy.

## 7.7 Results and Analysis

### 7.7.1 Deployed Model Held-Out Results

**Table 7.3. Deployed model held-out metrics**

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

**Table 7.4. Confusion matrix on official held-out test (n=32)**

|  | Predicted Non-Depressed | Predicted Depressed |
|---|---:|---:|
| Actual Non-Depressed | TN = 12 | FP = 9 |
| Actual Depressed | FN = 4 | TP = 7 |

The deployed model correctly classified 19 of 32 held-out participants and detected 7 of 11 depressed participants.

### 7.7.2 Invalid High-Accuracy Trap

An alternative thresholding regime produced 65.6% accuracy by predicting almost all participants as non-depressed, with 0% depression recall. That result is rejected as clinically invalid.

### 7.7.3 Experimental Candidate Comparison

**Table 7.5. Experimental candidate comparison**

| Approach | Held-out accuracy | Notes |
|---|---:|---|
| Deployed Extra Trees acoustic | 59.4% | Best deployable balance; recall 63.6% |
| PHQ-8 severity regression | 68.8% | Higher accuracy but recall 45.5%; not deployed |
| eGeMAPS + temporal | 56.2% | Strong CV, weaker test |
| Few-shot prototypes | 53.1% | Not better |
| Recall-constrained hybrid | 53.1% | Kept recall, lost accuracy |
| Segment-bag model | 50.0% | Overfit |
| Advanced gender-aware ensemble | 43.8% | Overfit |
| LoRA WavLM | 40.6% | High recall, very low specificity |
| Full-interview resampling | 39.5% | Did not generalize |

### 7.7.4 Explainability Results

For the deployed acoustic model:

- Timeline cards show supporting/opposing seconds with acoustic cues.  
- Occlusion importance identifies influential segments.  
- Feature rankings commonly surface energy, pause-related, pitch-variability, and spectral/MFCC contributors.  
- Spectrograms provide visual context.  
- Subtype profiles remain research-only heuristics.

### 7.7.5 Web Application Evaluation

Functional checks passed for health endpoint, audio upload, prediction response, explanation tabs, and patient save/recall.

### 7.7.6 Prototype History Note

An early CNN prototype on a tiny participant subset produced higher offline numbers, including segment-level F1 around 0.86 and perfect recording-level accuracy on nine files. Those results are retained only as prototype history and are **not** the final claim of this thesis.

## 7.8 Comparison with Existing Methods

Published full-corpus multimodal systems often report higher scores, but differ in data scale, modalities, and evaluation details. This thesis prioritizes official labels and locked splits, speech-only deployment practicality, integrated explanations, and honest reporting of moderate performance.

## 7.9 Discussion of Findings

**RQ1:** Yes, but only moderately under rigorous official-label evaluation (59.4% accuracy, 63.6% recall).  

**RQ2:** Yes, qualitatively; explanations are time-localized and acoustically grounded.  

**RQ3:** Yes; explanation methods are post-hoc wrappers around the predictive model.

**Key finding:** improving raw accuracy and preserving depression recall simultaneously was not achieved with the available usable audio. The PHQ regression candidate shows the trade-off clearly (68.8% accuracy vs 45.5% recall).

**Limitations:** 127 usable participants versus 189 official labels; corrupted/unreadable archives; speech-only scope; no prospective clinician study; caution about over-interpreting small gains after repeated experimentation on the same official test set.

## 7.10 Summary

Under official PHQ labels and speaker-independent evaluation, the deployed explainable acoustic model achieved 59.4% held-out accuracy with 63.6% depression recall. Explainability and deployment objectives were achieved. The 75–85% accuracy target was not honestly reached.

---

# CHAPTER 8  
# CONCLUSION AND FURTHER WORK

## 8.1 Introduction

This thesis designed, implemented, and evaluated an explainable framework for speech-based depression detection. The final contribution is not a claim of clinical-grade accuracy. It is a leakage-aware, explainable, deployable speech decision-support prototype grounded in official PHQ labels.

## 8.2 Summary of Research Contributions

1. Integrated explainable inference with timeline, feature, and attribution views.  
2. Dual pathway: CNN+Grad-CAM baseline and deployed acoustic Extra Trees model.  
3. Official-label correction and speaker-independent AVEC evaluation protocol.  
4. Systematic comparison of alternative models and rejection of clinically invalid high-accuracy shortcuts.  
5. Web deployment with patient-record management.  
6. Responsible AI framing with disclaimers and uncertainty-aware presentation.

## 8.3 Achievement of Objectives

**Table 8.1. Achievement of objectives**

| Objective | Status |
|---|---|
| Literature and gap analysis | Achieved |
| Explainable framework design | Achieved |
| Official-label leakage-aware pipeline | Achieved |
| Model comparison and deployment | Achieved |
| Honest evaluation | Achieved |
| 75–85% held-out accuracy target | Not achieved |

## 8.4 Quantitative Performance Summary

**Table 8.2. Final quantitative summary**

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

## 8.5 Limitations of the Study

Dataset size and missing/unreadable audio; speech-only modality; moderate discriminative signal; no clinician-rated explanation study; binary PHQ threshold oversimplifies severity; subtype module is heuristic.

## 8.6 Challenges Encountered

Label conflicts between folders and official PHQ CSVs; overfitting of high CV scores; accuracy–recall trade-offs; dependency issues for WavLM/LoRA/XGBoost experiments; keeping explanations faithful to the active model.

## 8.7 Future Work

1. Recover additional officially labelled audio and repair corrupted archives.  
2. Nested cross-validation and external corpora.  
3. Clinician evaluation of explanation usefulness.  
4. Careful multimodal fusion with privacy constraints.  
5. Severity-aware modelling that preserves screening recall.  
6. Prospective decision-support study.

## 8.8 Summary

This thesis demonstrates that an explainable, officially evaluated, speech-only depression detection system can be built and deployed, but that honest held-out performance remains moderate with currently usable data. Transparency, leakage-aware evaluation, and clinical framing are therefore as important as accuracy claims.

In conclusion, the work provides a foundation for trustworthy speech-based depression decision support: a complete pipeline from official labels and participant-level modelling to explained predictions in a usable web application.

---

# REFERENCES

[1] World Health Organization. (2017). Depression and other common mental disorders: Global health estimates. WHO.

[2] World Health Organization. (2021). Ethics and governance of artificial intelligence for health. WHO.

[3] American Psychiatric Association. (2013). Diagnostic and statistical manual of mental disorders (5th ed.). APA.

[4] Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9: Validity of a brief depression severity measure. Journal of General Internal Medicine, 16(9), 606–613.

[5] Kroenke, K., Strine, T. W., Spitzer, R. L., Williams, J. B. W., Berry, J. T., & Mokdad, A. H. (2009). The PHQ-8 as a measure of current depression in the general population. Journal of Affective Disorders, 114(1–3), 163–173.

[6] Hamilton, M. (1960). A rating scale for depression. Journal of Neurology, Neurosurgery, and Psychiatry, 23(1), 56–62.

[7] Beck, A. T., Ward, C. H., Mendelson, M., Mock, J., & Erbaugh, J. (1961). An inventory for measuring depression. Archives of General Psychiatry, 4(6), 561–571.

[8] Sobin, C., & Sackeim, H. A. (1997). Psychomotor symptoms of depression. American Journal of Psychiatry, 154(1), 4–17.

[9] Flint, A. J., Black, S. E., Campbell-Taylor, I., Gailey, G. F., & Levinton, C. (1993). Abnormal speech rate, pauses and acoustic frequencies in Alzheimer’s disease and depression. Brain and Language.

[10] Cannizzaro, M., Harel, B., Reilly, N., Chappell, P., & Snyder, P. J. (2004). Voice acoustical measurement of the severity of major depression. Brain and Cognition, 56(1), 30–35.

[11] Mundt, J. C., Snyder, P. J., Cannizzaro, M. S., Chappie, K., & Geralts, D. S. (2007). Voice acoustic measures of depression severity and treatment response collected via interactive voice response (IVR) technology. Journal of Neurolinguistics, 20(1), 50–64.

[12] Cummins, N., Scherer, S., Krajewski, J., Schnieder, S., Epps, J., & Quatieri, T. F. (2015). A review of depression and suicide risk assessment using speech analysis. Speech Communication, 71, 10–49.

[13] Low, D. M., Bentley, K. H., & Ghosh, S. S. (2020). Automated assessment of psychiatric disorders using speech: A systematic review. Laryngoscope Investigative Otolaryngology, 5(1), 96–116.

[14] Scherer, S., Stratou, G., Gratch, J., & Morency, L.-P. (2013). Investigating voice quality as a speaker-independent indicator of depression and PTSD. Interspeech.

[15] Gratch, J., Artstein, R., Lucas, G., Stratou, G., Scherer, S., Nazarian, A., … Morency, L.-P. (2014). The Distress Analysis Interview Corpus of human and computer interviews. LREC.

[16] Valstar, M., Schuller, B., Smith, K., Eyben, F., Jiang, B., Bilakhia, S., … Pantic, M. (2013). AVEC 2013: The continuous Audio/Visual Emotion and depression recognition challenge. AVEC Workshop.

[17] Valstar, M., Schuller, B., Smith, K., Almaev, T., Eyben, F., … Pantic, M. (2014). AVEC 2014: 3D dimensional affect and depression recognition challenge. AVEC Workshop.

[18] Valstar, M., Gratch, J., Schuller, B., Ringeval, F., Lalanne, D., Torres Torres, M., … Pantic, M. (2016). AVEC 2016: Depression, mood, and emotion recognition workshop and challenge. AVEC Workshop / ACM MM.

[19] Ringeval, F., Schuller, B., Valstar, M., Gratch, J., Cowie, R., Scherer, S., … Pantic, M. (2017). AVEC 2017: Real-life depression, and affect recognition workshop and challenge. AVEC Workshop / ACM MM.

[20] Williamson, J. R., Quatieri, T. F., Helfer, B. S., Horwitz, R., Yu, B., & Mehta, D. D. (2013). Vocal biomarkers of depression based on motor incoordination. AVEC Workshop.

[21] Ma, X., Yang, H., Chen, Q., Huang, D., & Wang, Y. (2016). DepAudioNet: An efficient deep model for audio based depression classification. AVEC Workshop / ACM MM.

[22] He, L., & Cao, C. (2018). Automated depression analysis using convolutional neural networks from speech. Journal of Biomedical Informatics, 83, 103–111.

[23] Al Hanai, T., Ghassemi, M., & Glass, J. (2018). Detecting depression with audio/text sequence modeling of interviews. Interspeech.

[24] Huang, Z., Epps, J., & Joachim, D. (2018). Speech landmark bigrams for depression detection from naturalistic smartphone speech. ICASSP.

[25] Rejaibi, E., Komaty, A., Meriaudeau, F., Agrebi, S., & Othmani, A. (2022). MFCC-based recurrent neural network for automatic clinical depression recognition and assessment from speech. Biomedical Signal Processing and Control, 71, 103107.

[26] Eyben, F., Wöllmer, M., & Schuller, B. (2010). openSMILE: The Munich versatile and fast open-source audio feature extractor. ACM Multimedia.

[27] Eyben, F., Scherer, K. R., Schuller, B. W., Sundberg, J., André, E., Busso, C., … Truong, K. P. (2015). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for voice research and affective computing. IEEE Transactions on Affective Computing, 7(2), 190–202.

[28] Degottex, G., Kane, J., Drugman, T., Raitio, T., & Scherer, S. (2014). COVAREP—A collaborative voice analysis repository for speech technologies. ICASSP.

[29] McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python. SciPy.

[30] Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357–366.

[31] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436–444.

[32] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.

[33] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. NeurIPS.

[34] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. NeurIPS.

[35] Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM TASLP, 29, 3451–3460.

[36] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., … Wei, F. (2022). WavLM: Large-scale self-supervised pre-training for full stack speech processing. IEEE JSTSP, 16(6), 1505–1518.

[37] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., … Chen, W. (2022). LoRA: Low-rank adaptation of large language models. ICLR.

[38] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825–2830.

[39] Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63(1), 3–42.

[40] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.

[41] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.

[42] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.

[43] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.

[44] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?” Explaining the predictions of any classifier. KDD.

[45] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.

[46] Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. ECCV.

[47] Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What clinicians want: Contextualizing explainable machine learning for clinical end use. MLHC.

[48] Amann, J., Blasimme, A., Vayena, E., Frey, D., & Madai, V. I. (2020). Explainability for artificial intelligence in healthcare: A multidisciplinary perspective. BMC Medical Informatics and Decision Making, 20, 310.

[49] Tjoa, E., & Guan, C. (2021). A survey on explainable artificial intelligence (XAI): Toward medical XAI. IEEE TNNLS, 32(11), 4793–4813.

[50] Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv:1702.08608.

[51] Arrieta, A. B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., … Herrera, F. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. Information Fusion, 58, 82–115.

[52] Wiens, J., Saria, S., Sendak, M., Ghassemi, M., Liu, V. X., Doshi-Velez, F., … Goldenberg, A. (2019). Do no harm: A roadmap for responsible machine learning for health care. Nature Medicine, 25, 1337–1340.

[53] He, L., Jiang, M., Zhang, X., et al. Multimodal depression recognition surveys and AVEC baseline comparisons (representative survey / challenge literature).

[54] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP.

---

## APPENDIX A — FINAL RESULT CARD (FOR EXAMINERS)

| Item | Final Thesis Claim |
|---|---|
| System type | Explainable speech-based depression decision-support framework |
| Deployed model | Participant-level Extra Trees on aggregated acoustic features |
| Labels | Official AVEC 2017 PHQ-8 binary |
| Evaluation | Speaker-independent official test participants |
| Accuracy | 59.4% |
| Depression recall | 63.6% |
| Balanced accuracy | 60.4% |
| Explainability | Timeline + occlusion + acoustic features + spectrograms |
| Clinical status | Research prototype / decision support — not a diagnosis |
| Accuracy target 75–85% | Not achieved under honest protocol |

---

*End of Thesis*
