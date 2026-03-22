from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np

class EvaluationType(str, Enum):
    INTRA_SUBJECT = 'intra_subject'
    CROSS_SUBJECT = 'cross_subject'

class SplitType(str, Enum):
    NONE        = 'none'           # no session split - random split within subject
    FIXED       = 'fixed_split'    # fixed train/test ratio within subject
    KFOLD       = 'kfold'          # k-fold cross-validation within subject
    SESSION     = 'session'        # Session

@dataclass
class SubjectResult:
    """Per-subject predictions and labels — used to compute detailed metrics."""
    subject_id:  int
    y_true:      List[str]
    y_pred:      List[str]
    accuracy:    float

    def to_arrays(self):
        return np.array(self.y_true), np.array(self.y_pred)


@dataclass
class EvaluationResult:
    """
    Full evaluation result — aggregated metrics + per-subject predictions.
    Use compute_metrics(y_true, y_pred) on per_subject_results to get
    F1, Kappa, MCC, confusion matrix, etc.
    """
    evaluation:           EvaluationType
    split_type:           SplitType
    accuracy_mean:        float
    accuracy_std:         float
    per_subject:          Dict[int, float]           = field(default_factory=dict)
    per_subject_results:  Dict[int, SubjectResult]   = field(default_factory=dict)
    metrics:              Optional[Dict]             = field(default=None)

    def compute_all_metrics(self, paradigm: str = 'motor_imagery') -> Dict:
        """
        Compute full metrics across all subjects combined.
        Requires per_subject_results to be populated.
        """
        from src.evaluation.metrics import compute_metrics

        if not self.per_subject_results:
            raise RuntimeError(
                'per_subject_results is empty — '
                'run evaluation with save_predictions=True'
            )

        y_true_all = np.concatenate([
            r.y_true for r in self.per_subject_results.values()
        ])
        y_pred_all = np.concatenate([
            r.y_pred for r in self.per_subject_results.values()
        ])

        self.metrics = compute_metrics(
            np.array(y_true_all),
            np.array(y_pred_all),
            paradigm=paradigm,
        )
        return self.metrics

    def per_subject_metrics(self, paradigm: str = 'motor_imagery') -> Dict[int, Dict]:
        """Compute metrics per subject individually."""
        from src.evaluation.metrics import compute_metrics

        if not self.per_subject_results:
            raise RuntimeError('per_subject_results is empty')

        return {
            subj: compute_metrics(
                np.array(r.y_true),
                np.array(r.y_pred),
                paradigm=paradigm,
            )
            for subj, r in self.per_subject_results.items()
        }

    def __str__(self) -> str:
        lines = [
            f'[{self.evaluation.value}|{self.split_type.value}] '
            f'Accuracy: {self.accuracy_mean:.3f} ± {self.accuracy_std:.3f}',
        ]
        for subj, acc in self.per_subject.items():
            lines.append(f'  Subject {subj}: {acc:.3f}')
        return '\n'.join(lines)