from dataclasses import dataclass, field
from typing import Dict
from enum import Enum

class EvaluationType(str, Enum):
    INTRA_SUBJECT = 'intra_subject'
    CROSS_SUBJECT = 'cross_subject'

@dataclass
class EvaluationResult:
    evaluation: EvaluationType
    accuracy_mean: float
    accuracy_std: float
    per_subject: Dict[int, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f'[{self.evaluation.value}] Accuracy: {self.accuracy_mean:.3f} ± {self.accuracy_std:.3f}',
        ]
        for subj, acc in self.per_subject.items():
            lines.append(f'  Subject {subj}: {acc:.3f}')
        return '\n'.join(lines)