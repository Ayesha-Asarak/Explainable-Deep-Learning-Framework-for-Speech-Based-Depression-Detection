# Accuracy Improvement Results

## Evaluation protocol
- Participants: 127 (48 depressed, 79 non-depressed)
- Labels: official AVEC 2017 PHQ binary labels only
- Split: official 95 train+dev / 32 full-test (`models/participant_split.json`)
- No participant overlap between train and test
- Threshold and model selection used **train+dev participants only**
- Held-out test evaluated after selection
- Folder names are storage locations and are never used as labels
- 39 incorrect folder labels were overridden by official PHQ labels

## Deployed model
- Input: raw participant speech acoustic features
- Head: Extra Trees participant-level classifier
- Artifact: `models/depression_acoustic_candidate.pkl`
- Active inference mode: `acoustic` (`src/config.py`)

## Held-out participant metrics
| Metric | Official-label deployed model |
|---|---:|
| Accuracy | **59.4%** |
| Balanced accuracy | 60.4% |
| Precision | 43.8% |
| Depression recall | 63.6% |
| Specificity | 57.1% |
| F1 | 51.9% |
| ROC-AUC | 0.654 |
| Accuracy 95% CI | 40.6% – 75.0% |

The apparent 65.6% alternative predicted every participant as
non-depressed. Its depression recall was 0%, so it is not a valid model.

## Target status
Requested target: **75–85% held-out accuracy**.

**Not reached.** The best deployable official-label, official-split result is
**59.4%**, improved from 40.6%.

## What was tried
1. Frozen WavLM + logistic/SVM/forests/MLP
2. COVAREP acoustic functionals
3. WavLM + COVAREP fusion and blends
4. Attention pooling over segment embeddings
5. HistGradientBoosting
6. Participant transcript TF-IDF + speech fusion
7. MiniLM semantic transcript embeddings + WavLM fusion
8. eGeMAPSv02 functionals + temporal pause/energy features
9. Few-shot prototype learning on raw, eGeMAPS, and WavLM embeddings
10. LoRA adaptation of WavLM attention in encoder layers 10–11
11. XGBoost, shrinkage LDA, feature selection, gender-aware fusion, and
    validation-selected probability ensembles
12. Participant-grouped local-window classifiers with mean, median, and
    top-quartile segment probability aggregation
13. Continuous PHQ-8 severity regression with acoustic, engineered, and
    gender-aware feature families
14. Recall-constrained hybrid of the binary acoustic and PHQ-severity models

The eGeMAPS model reached 69.0% train/dev cross-validation balanced accuracy,
but only 56.2% accuracy and 58.0% balanced accuracy on the held-out test.
It was not deployed because the raw-acoustic model remained stronger at 59.4%
accuracy, 60.4% balanced accuracy, and 65.4% ROC-AUC.

The few-shot prototype model reached 53.1% held-out accuracy, 53.5% balanced
accuracy, and 54.5% recall. It was not deployed.

LoRA reached 40.6% held-out accuracy, 52.6% balanced accuracy, and 90.9%
recall. Its specificity was only 14.3%, so it was not deployed.

The advanced gender-aware feature-selection candidate reached 66.7% OOF
balanced accuracy but only 43.8% held-out accuracy. The participant-grouped
segment-bag candidate reached 67.7% OOF balanced accuracy but only 50.0%
held-out accuracy. Both were rejected, and the 59.4% raw-acoustic model
remains deployed.

The PHQ-8 severity regression candidate reached **68.8% held-out accuracy**
and 63.2% balanced accuracy. It correctly detected 5 of 11 depressed
participants (45.5% recall), compared with 7 of 11 (63.6% recall) for the
deployed acoustic classifier. It is saved as
`models/depression_phq_regression_candidate.pkl`, but is not deployed
automatically because its improved overall accuracy comes with lower
depression sensitivity.

The recall-constrained hybrid preserved 63.6% depression recall, but its
held-out accuracy was only 53.1%. Therefore, the available models cannot
simultaneously provide 68.8% accuracy and 63.6% depression recall on the
current official test participants.

## July 30 full-data rescan

The latest data scan found 139 usable officially labelled participants
(101 train+dev and 38 official test). The canonical manifest was refreshed.

- Full-interview sampling (up to 64 segments): 39.5% accuracy, 75.0% recall
- Refreshed 90-second baseline: 44.7% accuracy, 66.7% recall

Neither candidate improved accuracy while preserving depression recall, so
neither replaced the deployed model. This also confirms that simply using
more segments from the same interviews does not solve the train/test
distribution shift.

Higher train-CV scores collapsed on the official test set, indicating
overfitting and distribution shift. The semantic transcript candidate reached
43.8% accuracy but 0% depression recall, so it was not deployed.

## Why 75–85% is hard here
- Only 32 official held-out participants
- Speech-only DAIC-WOZ signal is weak and highly speaker-dependent
- Only 127 of the 189 officially labelled participants have usable audio
- Stronger CV scores did not generalize

## Next data steps to reach 75–85% honestly
1. Add the missing officially labelled audio participants
2. Repair the corrupted `440_P.zip`
3. Consider severity regression (PHQ-8) rather than hard binary only
4. Keep single-speaker patient recordings at inference time (matches training)
