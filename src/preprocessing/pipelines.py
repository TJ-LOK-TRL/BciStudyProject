from typing import List
import numpy as np
from src.preprocessing.base_preprocessor import BasePreprocessor


class PreprocessingPipeline(BasePreprocessor):
    """Chain of preprocessors applied sequentially."""

    def __init__(self, steps: List[BasePreprocessor]):
        self.steps = steps

    def fit(self, X: np.ndarray) -> 'PreprocessingPipeline':
        for step in self.steps:
            step.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        for step in self.steps:
            X = step.transform(X)
        return X

    def __repr__(self) -> str:
        steps_str = ' -> '.join(str(s) for s in self.steps)
        return f'PreprocessingPipeline([{steps_str}])'


# ─── Pipelines pré-definidos ─────────────────────────────────────────────────

def bci_standard(sfreq: float) -> PreprocessingPipeline:
    """Bandpass 4-38Hz + z-score — standard for MI BCI papers."""
    from src.preprocessing.transforms.filtering import BandpassFilter
    from src.preprocessing.transforms.normalization import ZScoreNormalizer
    return PreprocessingPipeline([
        BandpassFilter(sfreq=sfreq, l_freq=4.0, h_freq=38.0),
        ZScoreNormalizer(),
    ])


def broadband(sfreq: float) -> PreprocessingPipeline:
    """Bandpass 0.5-40Hz + z-score — broader frequency range."""
    from src.preprocessing.transforms.filtering import BandpassFilter
    from src.preprocessing.transforms.normalization import ZScoreNormalizer
    return PreprocessingPipeline([
        BandpassFilter(sfreq=sfreq, l_freq=0.5, h_freq=40.0),
        ZScoreNormalizer(),
    ])