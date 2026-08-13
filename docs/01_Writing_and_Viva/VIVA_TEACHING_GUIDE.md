# Viva Teaching Guide — From Basics to Your Full Project

**Your project:** Explainable speech-based depression detection  
**Deployed model:** Random Forest  
**Main result:** 75.0% accuracy, 63.6% depression recall (held-out n=32)  
**Framing:** Decision support — **not** diagnosis  

Study this in order. Each section has: **What it means**, **Your project**, **Likely viva Q&A**.

---

# PART A — Basics (start here)

## A1. What is depression (in this project)?

**What it means**  
Depression is a mental health condition. Clinicians often use questionnaires like **PHQ-8 / PHQ-9**. A common cut-off is PHQ ≥ 10 → depressed (binary).

**Your project**  
You do **binary classification**:
- **Depressed** (PHQ binary = 1)
- **Non-Depressed** (PHQ binary = 0)

If Depressed, you also show a **research subtype/profile** (heuristic — not a medical diagnosis).

**Viva Q:** Is your system a doctor?  
**A:** No. It is a **decision-support research tool**. A clinician must interpret it with other information.

---

## A2. Why speech?

**What it means**  
Depressed speech often shows:
- lower energy (quieter)
- flatter pitch (less variation)
- slower rate
- longer pauses

These are **acoustic / paralinguistic** cues (how you sound), not the words alone.

**Your project**  
Speech-only deployed path (no video; text only used to cut interviewer speech from transcripts).

**Viva Q:** Why not only text?  
**A:** Speech acoustics can work even when transcripts are limited; privacy-friendly; matches your deployed acoustic model.

---

## A3. What is machine learning here?

**What it means**  
1. Extract numbers from audio (**features**)  
2. Train a model on labelled examples  
3. Predict new audio  
4. Measure how often it is correct (**metrics**)

**Your project pipeline**
```
Audio → participant speech → segments → 23 acoustic features
→ aggregate to one person vector → Random Forest → Depressed/Non-Depressed
→ explanations (occlusion, timeline, features) → Web UI
```

---

## A4. Dataset basics (DAIC-WOZ / AVEC)

**What it means**  
- **DAIC-WOZ:** clinical interview corpus (participant + virtual interviewer Ellie)  
- **AVEC 2017:** challenge with **official PHQ labels** and train/dev/test splits  

**Your usable data**

| Set | n | Depressed | Non-depressed | Use |
|-----|---|-----------|---------------|-----|
| Train+Dev | 95 | 37 | 58 | Train + choose threshold |
| Held-out test | 32 | 11 | 21 | Final score only |
| **Total usable** | **127** | 48 | 79 | — |

**Critical point**  
Folder names (`depressed/`, `non depressed/`) are **not always correct**. You use **official PHQ labels** (~39 folder mismatches corrected).

**Viva Q:** Why not use folder labels?  
**A:** They are noisy. Official PHQ is the clinical ground truth for AVEC.

---

# PART B — Your method (core story)

## B1. Preprocessing

**Steps**
1. Load audio at **16 kHz**, mono  
2. Prefer **participant-only speech** using transcript (remove Ellie)  
3. Segment into **5 s** windows, **50% overlap**  
4. Cap duration / number of segments for consistency  

**Why participant-only?**  
Training used patient speech only. If you upload full interview (Ellie + patient), the model often wrongly says Non-Depressed.

**Demo tip:** Upload `321_AUDIO.wav` or `325_AUDIO.wav` (filename must contain the ID).

---

## B2. Features (23 acoustics)

**What it means**  
Handcrafted numbers humans can interpret, e.g.:
- pitch mean / std  
- energy  
- speech rate  
- pause ratio  
- spectral features  
- MFCCs  

**Aggregation**  
Many segments → one participant vector using:
**mean, std, median, p25, p75**

**Viva Q:** Why aggregate to participant level?  
**A:** Final clinical label is per person. Segment-level scoring can leak speakers and inflate accuracy.

---

## B3. Model — Random Forest (deployed)

**What it means**  
Random Forest = many decision trees; majority / probability vote. Good for tabular features and smaller clinical datasets.

**Your settings**
- `n_estimators=600`
- `max_depth=8`
- `min_samples_leaf=1`
- `max_features=sqrt`
- `class_weight=balanced_subsample`
- `random_state=42`
- Threshold ≈ **0.49** (chosen on train+dev only, **never** on test)

**Dual pathway**
- **Deployed:** Random Forest + occlusion  
- **Baseline only:** CNN + Grad-CAM  

**Viva Q:** Why not CNN as final?  
**A:** CNN was an early prototype. After official-label comparison, RF had a better deployable accuracy–recall balance. CNN remains for Grad-CAM demo only.

**Viva Q:** Why Extra Trees in your thesis history?  
**A:** Extra Trees was a previous candidate (~59.4% Acc). Final deployed model is **Random Forest (75.0%)**.

---

## B4. Explainability (XAI)

| Method | Used for | Meaning |
|--------|----------|---------|
| **Occlusion** | Deployed RF | Remove one segment; if probability changes a lot → important |
| Timeline cards | Both | Time ranges + plain-language cues |
| Feature ranking | Acoustic | Which features matter |
| Spectrogram | Visual | Show the recording |
| Grad-CAM | CNN baseline only | Heatmap on spectrogram |

**Viva Q:** Does explanation change the prediction?  
**A:** No. XAI is **post-hoc**. The RF decides first; explanations justify it.

**Viva Q:** Is subtype diagnostic?  
**A:** No. Heuristic research profile when prediction = Depressed.

---

## B5. Web system

- **FastAPI** backend (`server.py`)  
- Frontend upload + tabs  
- Patient save / delete records  
- Disclaimer: not a diagnosis  

**Tabs:** Why · Type · Spectrogram · Timeline · Attribution · Features  

---

# PART C — Evaluation (memorise this table)

## C1. Metrics (what each means)

| Metric | Meaning (simple) |
|--------|------------------|
| Accuracy | Overall % correct |
| Balanced accuracy | Average of recall & specificity (fairer if imbalanced) |
| Precision | Of predicted depressed, how many truly depressed |
| Recall / Sensitivity | Of truly depressed, how many you caught |
| Specificity | Of truly non-depressed, how many correctly non-depressed |
| F1 | Balance of precision & recall |
| ROC-AUC | Ranking quality across thresholds |
| Confusion matrix | TN / FP / FN / TP counts |

## C2. Your held-out numbers (FINAL)

**Random Forest, n = 32**

| Metric | Value |
|--------|--------|
| Accuracy | **75.0%** |
| Balanced accuracy | **72.3%** |
| Precision | **63.6%** |
| Depression recall | **63.6%** |
| Specificity | **81.0%** |
| F1 | **63.6%** |
| ROC-AUC | **0.654** |
| Threshold | ≈ **0.49** |
| Acc. 95% CI | **59.4% – 87.5%** |

**CM:** TN=17, FP=4, FN=4, TP=7  

**Caveat:** accuracy is **seed-sensitive**; always mention the CI.

## C3. Why not higher?

- Small test set (32)  
- Speech-only  
- Clinical data is hard / noisy  
- Honest official-label protocol (no cheating with leakage)

## C4. Candidate comparison (one line each)

- **RF deployed:** best usable Acc + recall  
- **Extra Trees:** older, 59.4% Acc  
- **PHQ regression:** 68.8% Acc but only 45.5% recall → bad for screening  
- **LoRA WavLM:** high recall, low accuracy → not deployed  

---

# PART D — Design / Implementation talking points

## D1. One-minute architecture speech

> “Offline we train on official AVEC labels: extract participant speech, acoustic features, aggregate, choose Random Forest and threshold on train+dev. Online, the user uploads audio; FastAPI runs the same features through RF; occlusion and timelines explain the result; the UI shows Depressed/Non-Depressed with confidence and a disclaimer.”

## D2. Key files

| File | Role |
|------|------|
| `train_official_acoustic.py` | Train deployed model |
| `depression_acoustic_candidate.pkl` | Saved RF + threshold |
| `predict.py` | Inference |
| `explain.py` | Occlusion / charts / text |
| `server.py` | API |
| `frontend/` | UI |

## D3. Commands

```bash
python3 train_official_acoustic.py
python3 -m uvicorn server:app --host 127.0.0.1 --port 8765
```

---

# PART E — Research questions & contributions

## E1. RQs

1. Can speech support honest binary detection under official labels? → **Yes, moderately (75% Acc, 63.6% recall)**  
2. Can multi-level explanations make predictions inspectable? → **Yes (occlusion, timeline, features)**  
3. Is XAI post-hoc? → **Yes**

## E2. Contributions (list 5)

1. Official-label, leakage-aware pipeline  
2. Deployed RF acoustic classifier  
3. Multi-level explainability  
4. Web decision-support app + patient records  
5. Honest evaluation + candidate comparison  

## E3. Limitations (say these yourself — examiners like honesty)

1. Test n=32; wide CI  
2. Seed-sensitive accuracy  
3. Speech-only  
4. Single corpus  
5. No clinician user study  
6. Subtype is heuristic  

## E4. Future work

Larger/multi-corpus data · clinician studies · fairness by demographics · better calibration · human-in-the-loop screening  

---

# PART F — Demo script (viva practical)

1. Open http://127.0.0.1:8765  
2. Enter patient details  
3. Upload **`321_AUDIO.wav`** or **`325_AUDIO.wav`**  
4. Analyze → expect **Depressed**  
5. Show Why / Timeline / Features  
6. Upload **`303_AUDIO.wav`** → expect **Non-Depressed**  
7. End: “Decision support, not diagnosis. Held-out 75% Acc, 63.6% recall.”

**Avoid for depressed demo:** 308, 309 (model misses them)

---

# PART G — Rapid-fire Q&A (memorise)

**Q: Input / output?**  
A: Voice recording → Depressed/Non-Depressed + confidence + explanations (+ subtype if depressed).

**Q: Features?**  
A: 23 acoustics aggregated with mean/std/median/percentiles.

**Q: Model?**  
A: Random Forest, threshold ≈ 0.49.

**Q: Dataset size?**  
A: 127 usable; 95 train+dev; 32 test.

**Q: Accuracy?**  
A: 75.0% held-out; recall 63.6%; CI 59.4–87.5%.

**Q: Explainability?**  
A: Occlusion for RF; Grad-CAM only for CNN baseline.

**Q: Why 75% not 95%?**  
A: Honest clinical speech task, small test set, speech-only, no label leakage.

**Q: Can it diagnose?**  
A: No.

**Q: What if full interview uploaded without transcript match?**  
A: Interviewer speech can bias toward Non-Depressed; system tries to use transcript when ID/filename matches DAIC data.

**Q: Overfitting?**  
A: Speaker-independent official split; threshold not tuned on test; report CI.

---

# PART H — Words you must NOT say

- “This diagnoses depression”  
- “100% accurate”  
- “Final dataset is 9 participants / 108 segments”  
- “Final model is CNN with 75% F1 0.857” (that was prototype only)  
- “Deployed model is Extra Trees” (old; now RF)

---

# PART I — 30-second closing speech

> “I built an explainable speech system that classifies Depressed versus Non-Depressed using official AVEC labels, a Random Forest on acoustic features, and occlusion-based explanations in a web app. On 32 held-out participants it reaches 75% accuracy and 63.6% recall. It is intended as decision support, not diagnosis, and future work needs larger clinical validation.”

---

# Study plan (2–3 days)

| Day | Focus |
|-----|--------|
| Day 1 | Parts A–B (basics + method) + say metrics aloud 10 times |
| Day 2 | Parts C–E (evaluation, RQs, limits) + draw architecture from memory |
| Day 3 | Part F demo practice + Part G rapid-fire with a friend |

---

*Aligned to your deployed Random Forest system. Prefer this over older thesis drafts that still say Extra Trees 59.4% as final.*
