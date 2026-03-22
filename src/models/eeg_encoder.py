from typing import Literal
import torch
from src.models.nn.architectures.eeg_encoder_architecture import EEGEncoderArchitecture
from src.models.nn.base_neural_model import BaseNN


class EEGEncoderModel(BaseNN):
    """
    EEGEncoder model.
    Owns the architecture and its hyperparams.
    Training is handled externally by NNWrapper.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_windows: int = 5,
        eegn_F1: int = 16,
        eegn_D: int = 2,
        eegn_kern_size: int = 64,
        eegn_pool_size: int = 7,
        eegn_dropout: float = 0.5,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 4,
        tcn_filters: int = 32,
        tcn_dropout: float = 0.3,
        fuse: Literal['average', 'concat'] = 'average',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.eegn_F1 = eegn_F1
        self.eegn_D = eegn_D
        self.eegn_kern_size = eegn_kern_size
        self.eegn_pool_size = eegn_pool_size
        self.eegn_dropout = eegn_dropout
        self.tcn_depth = tcn_depth
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_filters = tcn_filters
        self.tcn_dropout = tcn_dropout
        self.fuse = fuse

        self._arch = EEGEncoderArchitecture(
            n_classes=n_classes,
            in_channels=n_channels,
            n_windows=n_windows,
            eegn_F1=eegn_F1,
            eegn_D=eegn_D,
            eegn_kern_size=eegn_kern_size,
            eegn_pool_size=eegn_pool_size,
            eegn_dropout=eegn_dropout,
            tcn_depth=tcn_depth,
            tcn_kernel_size=tcn_kernel_size,
            tcn_filters=tcn_filters,
            tcn_dropout=tcn_dropout,
            fuse=fuse,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._arch(x)

    def get_hyperparams(self) -> dict:
        return {
            'n_channels':      self.n_channels,
            'n_classes':       self.n_classes,
            'n_windows':       self.n_windows,
            'eegn_F1':         self.eegn_F1,
            'eegn_D':          self.eegn_D,
            'eegn_kern_size':  self.eegn_kern_size,
            'eegn_pool_size':  self.eegn_pool_size,
            'eegn_dropout':    self.eegn_dropout,
            'tcn_depth':       self.tcn_depth,
            'tcn_kernel_size': self.tcn_kernel_size,
            'tcn_filters':     self.tcn_filters,
            'tcn_dropout':     self.tcn_dropout,
            'fuse':            self.fuse,
        }
    