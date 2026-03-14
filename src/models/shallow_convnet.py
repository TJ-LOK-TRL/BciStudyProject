from typing import Optional, Literal, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from src.models.base_model import BaseModel

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

class ShallowConvNet(BaseModel):
    """
    ShallowConvNet classifier for Motor Imagery EEG.
    Wraps ShallowConvNetArchitecture with sklearn-compatible interface.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 64,
        dropout: float = 0.5,
        device: Optional[Literal['cuda', 'cpu']] = None,
        *,
        logging_indent: str = '',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self._device_str = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self._device_str)
        self._label_encoder = LabelEncoder()
        self.model = ShallowConvNetArchitecture(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            dropout=dropout,
        ).to(self.device)

        self.logging_indent = logging_indent

    def _to_tensor(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert numpy arrays to torch tensors with correct shape."""
        # X: (n_trials, n_channels, n_times) -> (n_trials, 1, n_channels, n_times)
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        if y is not None:
            y_tensor = torch.LongTensor(y).to(self.device)
            return X_tensor, y_tensor
        return X_tensor, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_encoded = self._label_encoder.fit_transform(y)
        X_tensor, y_tensor = self._to_tensor(X, y_encoded)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f'{self.logging_indent}Epoch [{epoch+1}/{self.n_epochs}] loss: {avg_loss:.4f}')

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet, call fit() first')
        self.model.eval()
        with torch.no_grad():
            X_tensor, _ = self._to_tensor(X)
            logits = self.model(X_tensor)
            y_encoded = torch.argmax(logits, dim=1).cpu().numpy()
        return self._label_encoder.inverse_transform(y_encoded)

    def clone(self) -> 'ShallowConvNet':
        return ShallowConvNet(
            n_channels=self.n_channels,
            n_times=self.n_times,
            n_classes=self.n_classes,
            lr=self.lr,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            dropout=self.dropout,
            device=self._device_str
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'label_encoder': self._label_encoder,
            'hyperparams': {
                'n_channels': self.n_channels,
                'n_times': self.n_times,
                'n_classes': self.n_classes,
                'lr': self.lr,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'dropout': self.dropout,
                'device': self._device_str,
            }
        }, path)
        print(f'  Model saved to {path}')

    @classmethod
    def load(cls, path: str) -> 'ShallowConvNet':
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(**checkpoint['hyperparams'])
        model.model.load_state_dict(checkpoint['state_dict'])
        model._label_encoder = checkpoint['label_encoder']
        model.is_fitted = True
        print(f'  Model loaded from {path}')
        return model

    def __repr__(self) -> str:
        return (
            f'ShallowConvNet('
            f'channels={self.n_channels}, '
            f'n_classes={self.n_classes}, '
            f'device={self.device}, '
            f'fitted={self.is_fitted})'
        )
