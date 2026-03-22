from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from src.models.core import IHyperparametrizable


class BaseNN(nn.Module, IHyperparametrizable, ABC):
    """
    Base for all EEG neural network architectures.
    Enforces forward() implementation.
    Training logic lives in NNWrapper — this is pure architecture.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: input tensor — shape depends on input adapter
               CNN2D: (batch, 1, channels, times)
               LSTM:  (batch, times, channels)
               CNN1D: (batch, channels, times)
        Returns:
            logits: (batch, n_classes)
        """
        pass