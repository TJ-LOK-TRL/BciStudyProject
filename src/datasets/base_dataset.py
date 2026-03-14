from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np

class BaseDataset(ABC):
    """Abstract base class for all EEG datasets."""

    def __init__(self, data_path: str, subject_ids: Optional[list[int]] = None):
        self.data_path = data_path
        self.subject_ids = subject_ids
        self.X: Optional[np.ndarray] = None  # (n_trials, n_channels, n_times)
        self.y: Optional[np.ndarray] = None  # (n_trials,)
        self.metadata: Dict[str, Any] = {}
        self._is_loaded: bool = False

    @property
    @abstractmethod
    def n_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def sfreq(self) -> float:
        """Sampling frequency in Hz."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load raw data into self.X and self.y."""
        pass

    @abstractmethod
    def preprocess(self) -> None:
        """Apply basic preprocessing (filtering, epoching, etc.)."""
        pass

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess if not done yet, then return X, y."""
        if not self._is_loaded:
            self.load()
            self.preprocess()
            self._is_loaded = True
        return self.X, self.y

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'subjects={self.subject_ids}, '
            f'n_classes={self.n_classes}, '
            f'channels={self.n_channels})'
        )