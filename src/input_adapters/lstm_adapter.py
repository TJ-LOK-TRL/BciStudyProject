from src.input_adapters.registry import register_adapter
from src.input_adapters.base_input_adapter import BaseInputAdapter
import torch
import numpy as np

@register_adapter
class LSTMAdapter(BaseInputAdapter):
    """(n, C, T) → (n, T, C) - for LSTM/GRU models"""
    def transform(self, X: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(X).permute(0, 2, 1)