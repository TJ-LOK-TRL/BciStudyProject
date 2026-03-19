# src/input_adapters/base_input_adapter.py
from abc import abstractmethod
import torch
import numpy as np

class BaseInputAdapter:
    @abstractmethod
    def transform(self, X: np.ndarray) -> torch.Tensor:
        pass
