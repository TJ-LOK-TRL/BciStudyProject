from src.input_adapters.registry import register_adapter
from src.input_adapters.base_input_adapter import BaseInputAdapter
import torch
import numpy as np

@register_adapter
class CNN2DAdapter(BaseInputAdapter):
    """(n, C, T) → (n, 1, C, T) - for Conv2D models"""
    def transform(self, X: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(X).unsqueeze(1)