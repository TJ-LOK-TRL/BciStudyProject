from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute standard classification metrics."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }