from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class BaseModel(ABC):

    def __init__(self):
        self.model = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit model to training data.
        Optional kwargs (supported by PyTorch models):
            X_val: np.ndarray — validation features for early stopping
            y_val: np.ndarray — validation labels for early stopping
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def clone(self) -> 'BaseModel':
        """Return a new unfitted instance with the same hyperparameters."""
        pass

    def save(self, path: str) -> None:
        """Save model to disk."""
        raise NotImplementedError(f'{self.__class__.__name__} must implement save()')

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        raise NotImplementedError(f'{cls.__name__} must implement load()')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(fitted={self.is_fitted})'