import numpy as np
from src.preprocessing.base_preprocessor import BasePreprocessor


class ZScoreNormalizer(BasePreprocessor):
    """Z-score normalization per trial."""

    def fit(self, X: np.ndarray) -> 'ZScoreNormalizer':
        return self   # stateless — no fitting needed

    def transform(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True) + 1e-8
        return (X - mean) / std

    def __repr__(self) -> str:
        return 'ZScoreNormalizer()'


class MinMaxNormalizer(BasePreprocessor):
    """Min-max normalization per trial."""

    def fit(self, X: np.ndarray) -> 'MinMaxNormalizer':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        min_val = X.min(axis=-1, keepdims=True)
        max_val = X.max(axis=-1, keepdims=True)
        return (X - min_val) / (max_val - min_val + 1e-8)
    
    def __repr__(self) -> str:
        return 'MinMaxNormalizer()'