import torch
from src.models.nn.base_neural_model import BaseNN
from src.models.nn.architectures.shallow_conv_net_architecture import ShallowConvNetArchitecture


class ShallowConvNet(BaseNN):
    """
    ShallowConvNet model.
    Owns the architecture and its hyperparams.
    Training is handled externally by NNWrapper.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.dropout = dropout

        self._arch = ShallowConvNetArchitecture(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._arch(x)

    def get_hyperparams(self) -> dict:
        return {
            'n_channels': self.n_channels,
            'n_times':    self.n_times,
            'n_classes':  self.n_classes,
            'dropout':    self.dropout,
        }