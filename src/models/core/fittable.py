from abc import ABC, abstractmethod
import numpy as np

from abc import ABC, abstractmethod
import numpy as np


class IFittable(ABC):
    """
    Interface for all fittable models — sklearn, PyTorch, or otherwise.
    Any model that implements fit + predict can participate in evaluation
    and benchmarking pipelines.
    """

    def __init__(self):
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> None:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(fitted={self.is_fitted})'
