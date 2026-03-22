from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    paradigm: str = 'motor_imagery',
    top_k: int = 3,
) -> Dict:
    """
    Compute classification metrics for EEG paradigms.

    Args:
        y_true:   true labels (n_trials,)
        y_pred:   predicted labels (n_trials,)
        y_prob:   predicted probabilities (n_trials, n_classes) — required for ROC/PR/ECE
        paradigm: 'motor_imagery' | 'imagined_speech' | 'clinical'
        top_k:    k for top-k accuracy (imagined_speech only)

    Returns:
        dict with metrics relevant to the paradigm
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    # ── core metrics — all paradigms ─────────────────────────────────────────
    metrics = {
        'accuracy':          float(accuracy_score(y_true, y_pred)),
        'f1_macro':          float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
        'f1_weighted':       float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_per_class':      {
            str(c): float(s)
            for c, s in zip(
                classes,
                f1_score(y_true, y_pred, average=None, zero_division=0, labels=classes)
            )
        },
        'kappa':             float(cohen_kappa_score(y_true, y_pred)),
        'mcc':               float(matthews_corrcoef(y_true, y_pred)),
        'confusion_matrix':  confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        'classes':           classes.tolist(),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred))
    }

    # ── motor imagery extras ──────────────────────────────────────────────────
    if paradigm == 'motor_imagery':
        pass  # core metrics sufficient for MI

    # ── imagined speech extras ────────────────────────────────────────────────
    if paradigm == 'imagined_speech':
        if y_prob is not None and n_classes >= top_k:
            metrics['top_k_accuracy'] = float(
                top_k_accuracy_score(y_true, y_prob, k=top_k, labels=classes)
            )

    # ── clinical extras ───────────────────────────────────────────────────────
    if paradigm == 'clinical':
        if y_prob is not None:
            # binarize for multiclass ROC/PR
            if n_classes == 2:
                pos_class: np.ndarray = classes[1]
                y_prob_pos = y_prob[:, 1]
                y_true_bin = (y_true == pos_class).astype(int)
                metrics['roc_auc']  = float(roc_auc_score(y_true_bin, y_prob_pos))
                metrics['pr_auc']   = float(average_precision_score(y_true_bin, y_prob_pos))
                metrics['brier']    = float(brier_score_loss(y_true_bin, y_prob_pos))
                metrics['ece']      = float(_compute_ece(y_true_bin, y_prob_pos))
            else:
                y_true_bin = label_binarize(y_true, classes=classes)
                metrics['roc_auc']  = float(roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro'))
                metrics['pr_auc']   = float(average_precision_score(y_true_bin, y_prob, average='macro'))
                metrics['brier']    = float(_compute_multiclass_brier(y_true_bin, y_prob))
                metrics['ece']      = float(_compute_ece_multiclass(y_true, y_prob, classes))

    return metrics


def _compute_ece(y_true_bin: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for binary classification."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = float(np.mean(y_true_bin[mask] == 1))
        conf = float(np.mean(y_prob[mask]))
        ece += (mask.sum() / len(y_true_bin)) * abs(acc - conf)
    return ece


def _compute_ece_multiclass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for multiclass — uses max confidence."""
    confidences = y_prob.max(axis=1)
    predictions = classes[y_prob.argmax(axis=1)]
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = float(np.mean(correct[mask]))
        conf = float(np.mean(confidences[mask]))
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return ece


def _compute_multiclass_brier(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score for multiclass — mean squared error over all classes."""
    return float(np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1)))


def metrics_to_row(    
    metrics: Dict,
    subject_id: Optional[int] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None
) -> dict:
    """
    Flatten metrics dict to a single CSV row.
    Excludes confusion_matrix, f1_per_class, classes (too complex for CSV row).
    """
    row = {}
    if dataset_name: row['dataset'] = dataset_name
    if model_name:   row['model']   = model_name
    if subject_id is not None: row['subject_id'] = subject_id

    skip = {'confusion_matrix', 'f1_per_class', 'classes'}
    for k, v in metrics.items():
        if k not in skip:
            row[k] = round(v, 4) if isinstance(v, float) else v
    return row