import torch
import torch.nn as nn

class ShallowConvNetArchitecture(nn.Module):
    """
    ShallowConvNet architecture (Schirrmeister et al. 2017).
    Temporal conv -> Spatial conv -> Squaring -> Mean pooling -> Log -> FC
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        n_temporal_filters: int = 40,
        temporal_kernel_size: int = 25,
        pool_kernel_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_temporal_filters,
            kernel_size=(1, temporal_kernel_size),
            bias=False,
        )
        self.spatial_conv = nn.Conv2d(
            in_channels=n_temporal_filters,
            out_channels=n_temporal_filters,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(n_temporal_filters)
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_kernel_size),
            stride=(1, pool_stride),
        )
        self.dropout = nn.Dropout(dropout)

        # compute flatten size dynamically
        dummy = torch.zeros(1, 1, n_channels, n_times)
        dummy = self.temporal_conv(dummy)
        dummy = self.spatial_conv(dummy)
        dummy = self.pool(dummy)
        flatten_size = dummy.view(1, -1).shape[1]

        self.fc = nn.Linear(flatten_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, n_times)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = x ** 2                  # squaring activation
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))   # log activation
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
