from abc import ABC, abstractmethod
import numpy as np


class IPredictable(ABC):

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))