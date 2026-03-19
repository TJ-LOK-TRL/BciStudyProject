from src.input_adapters.registry import register_adapter
from src.input_adapters.base_input_adapter import BaseInputAdapter
import torch
import numpy as np

@register_adapter
class CNN1DAdapter(BaseInputAdapter):
    """(n, C, T) → (n, C, T) - for Conv1D models, no reshape needed"""
    def transform(self, X: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(X)