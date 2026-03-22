# EEG Classification Report — {DATASET} × {MODEL}

**Date:** {DATE}  
**Paradigm:** {PARADIGM}  
**Seed:** 42  

---

## 1. Dataset

| Property        | Value             |
|-----------------|-------------------|
| Name            | {DATASET}         |
| Subjects        | {N_SUBJECTS}      |
| Classes         | {CLASSES}         |
| Channels        | {N_CHANNELS}      |
| Sampling Rate   | {SFREQ} Hz        |
| Trials/Subject  | {TRIALS_PER_SUBJ} |

---

## 2. Pipeline

| Stage          | Details                        | Applied   |
|----------------|--------------------------------|-----------|
| Preprocessing  | {PREPROCESSING}                | {YES/NO}  |
| Model          | {MODEL}                        | always    |
| Trainer        | {TRAINER_CONFIG}               | {NN only} |

### 2.1 Preprocessing Details
```
{PREPROCESSING_DETAILS}
# e.g.:
# BandpassFilter: 8-32 Hz, order 5
# RegressionRemover: EOG channels [22, 23, 24]
# FilterBankTransform: 6 bands (8-32 Hz)
# none
```

### 2.2 Model Hyperparameters
```
{MODEL_HYPERPARAMS}
# e.g.:
# n_channels=22, n_classes=4, eegn_kern_size=64
# n_components=4 (CSP+LDA)
# band_mode=True (RiemannianSVM)
```

### 2.3 Trainer Config *(NN models only)*
```
{TRAINER_CONFIG_DETAILS}
# e.g.:
# n_epochs=200, lr=1e-3, batch_size=64
# l2_scale=2.0, loss_scale=2.0, label_smoothing=0.2
# grad_clip=0.0, scheduler=none, optimizer=adam
# EarlyStoppingCallback(patience=50)
# N/A — classical model
```

---

## 3. Evaluation Protocol

| Property         | Value             |
|------------------|-------------------|
| Type             | {EVAL_TYPE}       |
| Subject split    | {SUBJECT_SPLIT}   |
| Session split    | {SESSION_SPLIT}   |
| Test ratio       | {TEST_RATIO}      |
| Validation ratio | {VAL_RATIO}       |

---

## 4. Results Summary

### 4.1 Core Metrics (all paradigms)

| Metric            | Mean      | Std        |
|-------------------|-----------|------------|
| Accuracy          | {ACC}     | {ACC_STD}  |
| Balanced Accuracy | {BACC}    | {BACC_STD} |
| F1-Macro          | {F1}      | {F1_STD}   |
| Cohen's Kappa     | {KAPPA}   | {KAPPA_STD}|
| MCC               | {MCC}     | {MCC_STD}  |
| Chance Level      | {CHANCE}  | —          |

<!-- ── IMAGINED SPEECH ONLY — remove section if not applicable ── -->
### 4.2 Top-k Accuracy *(imagined speech)*

| k | Top-k Accuracy |
|---|----------------|
| 3 | {TOP3}         |
| 5 | {TOP5}         |

<!-- ── CLINICAL ONLY — remove section if not applicable ── -->
### 4.2 Clinical Metrics *(clinical paradigm)*

| Metric      | Value    |
|-------------|----------|
| ROC-AUC     | {ROCAUC} |
| PR-AUC      | {PRAUC}  |
| Brier Score | {BRIER}  |
| ECE         | {ECE}    |

---

## 5. Per-Subject Results

| Subject  | Accuracy   | Balanced Acc | F1-Macro | Kappa | MCC  |
|----------|------------|--------------|----------|-------|------|
| {S1}     | {ACC_S1}   | {BACC_S1}    | {F1_S1}  | ...   | ...  |
| ...      | ...        | ...          | ...      | ...   | ...  |
| **Mean** | {ACC_MEAN} | {BACC_MEAN}  | {F1_MEAN}| ...   | ...  |
| **Std**  | {ACC_STD}  | {BACC_STD}   | {F1_STD} | ...   | ...  |

---

## 6. Confusion Matrix

![Confusion Matrix](../figures/{DATASET}_{MODEL}_confusion_matrix.png)
```
{CONFUSION_MATRIX_RAW}
```

---

## 7. F1 Per Class *(imagined speech only)*

| Class | F1     | Support |
|-------|--------|---------|
| {C1}  | {F1C1} | {N1}    |
| ...   | ...    | ...     |

---

## 8. Notes & Observations

{NOTES}

---

## 9. References

- Dataset: {DATASET_CITATION}
- Model: {MODEL_CITATION}