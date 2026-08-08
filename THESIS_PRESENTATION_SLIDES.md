# Presentation Slides — Explainable Speech-Based Depression Detection

**Ayesha Asarak · 215509V**  
BSc(Hons) in Artificial Intelligence  
University of Moratuwa  

Use one slide per `##` heading (or split where marked). Keep speaker notes short.

---

## Title

**An Explainable Framework for Speech-Based Depression Detection**

Ayesha Asarak (215509V)  
BSc(Hons) in Artificial Intelligence  
Department of Computational Mathematics  
Faculty of Information Technology  
University of Moratuwa  

---

## Agenda

1. Introduction  
2. Objectives  
3. Literature Review  
4. Problem Definition (Research Gap)  
5. Technology Adopted  
6. Novel Approach  
7. Design  
8. Implementation  
9. Evaluation  
10. Conclusion and Further Work  
11. References  

---

## 1. Introduction

**Problem**
- Depression is common and often underdiagnosed
- Clinical assessment is interview- and questionnaire-based (subjective, time-consuming)
- Speech carries acoustic markers: low energy, flat pitch, slow rate, long pauses

**Opportunity**
- AI can analyse speech automatically
- Clinicians need **explanations**, not only a score

**This work**
- Speech-only, explainable decision-support system
- Official AVEC PHQ-8 labels + speaker-independent evaluation
- Deployed as a FastAPI web application

---

## 2. Objectives

1. Review speech-based depression detection and XAI gaps  
2. Study acoustic, ML/DL, and explainability methods  
3. Design an explainable prediction + explanation framework  
4. Implement a leakage-aware official-label pipeline  
5. Compare candidates; select a deployable model  
6. Deploy a web decision-support prototype  
7. Evaluate honestly and discuss limitations  

**Framing:** decision support — **not** clinical diagnosis

---

## 3. Literature Review (Key Points)

| Theme | What exists | Limitation |
|-------|-------------|------------|
| Classical acoustics | openSMILE / COVAREP + SVM/RF | Often weak XAI |
| Deep learning | CNN / RNN on spectrograms | Black-box decisions |
| Transformers | WavLM / HuBERT | Data-hungry; opaque |
| Multimodal | Audio + text + video | Higher scores, harder to deploy |
| XAI | Grad-CAM, SHAP, attention | Often not clinician-ready |

**Gap:** few systems combine **official labels**, **participant-level evaluation**, **time-local explanations**, and **deployable UI**.

---

## 4. Problem Definition (Research Gap)

1. **Opacity** — predictions without time/feature evidence  
2. **Weak evaluation** — folder labels, segment leakage, tiny subsets  
3. **Dual-path disconnect** — deep models vs interpretable acoustics rarely unified  
4. **Deployment gap** — many theses stop at accuracy tables  
5. **Trust gap** — no decision-support disclaimers / record review  

**Research questions**
- RQ1: Can speech acoustics support honest binary detection under official labels?  
- RQ2: Can multi-level explanations make predictions inspectable?  
- RQ3: Does XAI remain post-hoc (not replacing the decision rule)?  

---

## 5. Technology Adopted (Extended)

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| Audio | Librosa, SoundFile |
| Classical ML | scikit-learn (RF, ET, SVM, Logistic) |
| Deep baseline | PyTorch CNN + Grad-CAM |
| Experiments | WavLM, LoRA/PEFT, openSMILE, XGBoost |
| XAI | Occlusion importance, feature ranking, timelines |
| Serving | FastAPI, Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Data | DAIC-WOZ + AVEC 2017 PHQ-8 CSVs |

---

## 6. Novel Approach

**Four extensions**

1. **Official PHQ-8 labels** + official AVEC splits (not folder names)  
2. **Participant-level Random Forest** on aggregated acoustic features  
3. **Multi-level XAI:** occlusion · timeline · features · spectrograms  
4. **Deployable web app** with patient-record save/recall  

**Dual pathway**
- **Deployed:** Random Forest + occlusion  
- **Baseline:** CNN + Grad-CAM (prototype only)  

---

## 7. Design — Architecture (1)

**Offline**
Data → Features → CV model search → Threshold (train+dev) → Artifact  

**Online**
Upload → FastAPI → Preprocess → Random Forest → Explain → UI tabs  

**Dataset (usable)**
- 127 participants  
- 95 train+dev / 32 held-out test  
- Threshold ≈ 0.49 (never tuned on test)

*(Show: `thesis_full_top_tier_architecture.png` or `thesis_architectural_summary.png`)*

---

## 7. Design — Architecture (2)

**Modules**
- Preprocessing (`features.py`, `data.py`)  
- ML Engine (`train_official_acoustic.py`, RF artifact)  
- Explainability (`explain.py`)  
- API + UI (`server.py`, `frontend/`)  
- Patient store (`patient_store.py`)  

**Key decisions**
- Participant-level aggregation (not final segment scoring)  
- Occlusion for deployed model (Grad-CAM only for CNN)  
- Decision-support disclaimers everywhere  

---

## 8. Implementation

**Training**
```bash
python3 train_official_acoustic.py
```

**Serving**
```bash
python3 -m uvicorn server:app --host 127.0.0.1 --port 8765
```

**Deployed model**
- Random Forest (`n_estimators=600`, `max_depth=8`, seed=42)  
- Artifact: `depression_acoustic_candidate.pkl`  

**UI tabs**
Why This Result · Type · Spectrogram · Timeline · Attribution · Features  

*(Show: system implementation / inference / web UI flowcharts)*

---

## 9. Evaluation — Protocol

| Item | Setting |
|------|---------|
| Labels | Official AVEC PHQ-8 binary |
| Split | 95 train+dev / 32 test |
| Unit | Participant-level |
| Threshold | ≈ 0.49 (train+dev OOF only) |
| Metrics | Acc, bal. Acc, Prec, Rec, Spec, F1, ROC-AUC, CM, bootstrap CI |

**Honesty rules**
- No test-set threshold tuning  
- Folder labels overridden  
- CNN 9-participant prototype ≠ final claim  

---

## 9. Evaluation — Deployed Results

**Random Forest (held-out n = 32)**

| Metric | Value |
|--------|--------|
| Accuracy | **75.0%** |
| Balanced accuracy | 72.3% |
| Precision | 63.6% |
| Depression recall | **63.6%** |
| Specificity | 81.0% |
| F1 | 63.6% |
| ROC-AUC | 0.654 |
| Acc. 95% CI | 59.4% – 87.5% |

**Confusion matrix:** TN=17, FP=4, FN=4, TP=7  

**Caveat:** accuracy is seed-sensitive — report with CI  

*(Show: Fig 7.1 CM, Fig 7.2 metrics, Fig 7.3 8-model chart)*

---

## 9. Evaluation — Candidate Comparison

| Model | Acc. | Recall | Deploy? |
|-------|------|--------|---------|
| **Random Forest (deployed)** | **75.0%** | **63.6%** | **Yes** |
| Extra Trees (previous) | 59.4% | 63.6% | No |
| PHQ regression | 68.8% | 45.5% | No (weak recall) |
| eGeMAPS / few-shot / hybrid | ~50–56% | ~54–64% | No |
| LoRA WavLM | 40.6% | 90.9% | No (low Acc.) |

**Selection rule:** accuracy–recall balance for screening support, not Acc. alone.

---

## 10. Conclusion and Further Work

**Achieved**
- Official-label, leakage-aware speech pipeline  
- Deployed RF + occlusion explanations + web UI  
- Honest held-out result: **75.0% Acc., 63.6% recall**  

**Limits**
- n = 32 test; seed-sensitive Acc.; speech-only; no clinician study  

**Further work**
- Larger / multi-corpus validation  
- Clinician-rated explanation study  
- Fairness by demographics  
- Calibrated uncertainty & human-in-the-loop workflows  

---

## 11. References (selected for slides)

Keep 6–8 on the slide; full list in thesis.

1. Gratch et al. — DAIC-WOZ corpus  
2. AVEC 2017 / PHQ-8 depression challenge materials  
3. Cummins et al. — speech & depression survey  
4. Selvaraju et al. — Grad-CAM  
5. Breiman — Random Forests  
6. Valstar / Ringeval et al. — AVEC challenges  
7. Related acoustic / WavLM / multimodal depression papers from thesis bibliography  

*(Point examiners to full References chapter)*

---

## Closing / Q&A

**Takeaway**
> Explainable, officially evaluated speech AI can support depression screening research — with moderate performance and mandatory decision-support framing.

**Demo (optional)**
Web UI → upload → prediction + occlusion timeline + features  

**Questions?**

---

## Speaker cheat-sheet (30–40 min)

| Section | Time |
|---------|------|
| 1–2 Intro + Objectives | 4 min |
| 3–4 Literature + Gap | 6 min |
| 5–6 Tech + Novel approach | 5 min |
| 7–8 Design + Implementation | 8 min |
| 9 Evaluation | 10 min |
| 10–11 Conclusion + Refs + Q&A | 7 min |

**Do not say**
- 9 participants / 108 segments as final  
- CNN Acc 75% / F1 0.857 as final  
- Extra Trees as deployed model  
- “diagnosis” without “decision support”  

**Do say**
- Official labels, 127 usable, RF deployed, occlusion XAI  
- 75.0% Acc / 63.6% recall / CI wide / seed caveat  
