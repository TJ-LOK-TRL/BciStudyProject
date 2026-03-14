from abc import ABC, abstractmethod
import numpy as np


class BasePreprocessor(ABC):
    """Abstract base class for EEG preprocessing pipelines."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BasePreprocessor':
        """Fit preprocessor parameters on training data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing to data."""
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'