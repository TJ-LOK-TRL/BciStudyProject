from typing import Optional, Literal
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from transformers import LlamaConfig
from src.models.llama_eeg import EEGLlamaForCausalLM

from src.models.base_model import BaseModel


# ─── Sub-modules ────────────────────────────────────────────────────────────

class LinearL2(nn.Module):
    """Linear layer with L2 regularisation computed manually in training loop."""

    def __init__(self, in_features: int, out_features: int, weight_decay: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def l2_loss(self) -> torch.Tensor:
        return self.weight_decay * torch.sum(self.linear.weight ** 2)


class Conv1dL2(nn.Module):
    """Conv1d with L2 regularisation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        weight_decay: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def l2_loss(self) -> torch.Tensor:
        return self.weight_decay * torch.sum(self.conv.weight ** 2)


class Conv2dL2(nn.Module):
    """Conv2d with L2 regularisation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding = 0,
        dilation: int = 1,
        groups: int = 1,
        weight_decay: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def l2_loss(self) -> torch.Tensor:
        return self.weight_decay * torch.sum(self.conv.weight ** 2)


class Chomp1d(nn.Module):
    """Remove extra padding added for causal convolutions."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class ConvBlock(nn.Module):
    """
    Downsampling Projector — EEGNet-style convolutional block.
    Reduces temporal dimension and extracts spatial-temporal features.
    """

    def __init__(
        self,
        F1: int = 16,
        kern_length: int = 64,
        pool_size: int = 7,
        D: int = 2,
        in_chans: int = 22,
        dropout: float = 0.3,
    ):
        super().__init__()
        F2 = F1 * D
        self.conv1 = Conv2dL2(1, F1, (kern_length, 1), padding='same', weight_decay=0.009)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dL2(F1, F2, (1, in_chans), groups=F1, weight_decay=0.009)
        self.bn2 = nn.BatchNorm2d(F2)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = Conv2dL2(F2, F2, (16 + 1, 1), padding='same', weight_decay=0.009)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((pool_size, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 1, 3, 2)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.depthwise(x))
        x = self.activation(x)
        x = self.dropout1(self.avgpool1(x))
        x = self.bn3(self.conv2(x))
        x = self.activation(x)
        x = self.dropout2(self.avgpool2(x))
        return x


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with dilated causal convolutions.
    Captures local temporal features.
    """

    def __init__(
        self,
        input_dim: int,
        depth: int,
        kernel_size: int,
        filters: int,
        dropout: float,
        weight_decay: float = 0.009,
        max_norm: float = 0.6,
    ):
        super().__init__()
        self.activation = F.silu
        self.downsample = nn.Conv1d(input_dim, filters, 1) if input_dim != filters else None

        self.cn1 = nn.Sequential(
            Conv1dL2(input_dim, filters, kernel_size, weight_decay=weight_decay),
            nn.BatchNorm1d(filters),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.cn2 = nn.Sequential(
            Conv1dL2(filters, filters, kernel_size, weight_decay=weight_decay),
            nn.BatchNorm1d(filters),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList()
        for i in range(depth - 1):
            dilation = 2 ** (i + 1)
            padding = (kernel_size - 1) * dilation
            self.blocks.append(nn.Sequential(
                Conv1dL2(filters if i > 0 else input_dim, filters, kernel_size,
                         padding=padding, dilation=dilation, weight_decay=weight_decay),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                Conv1dL2(filters, filters, kernel_size,
                         padding=padding, dilation=dilation, weight_decay=weight_decay),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        self._init_weights(max_norm)

    def _init_weights(self, max_norm: float) -> None:
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    nn.init.kaiming_uniform_(layer.weight)
                    nn.utils.clip_grad_norm_(layer.parameters(), max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        out = self.cn2(self.cn1(out))
        res = self.downsample(out) if self.downsample is not None else out

        for i, block in enumerate(self.blocks):
            out = block(out) + (res if i == 0 else self.blocks[i - 1](res))
            out = self.activation(out)

        return out.transpose(1, 2)


class StableTransformerBlock(nn.Module):
    """
    Stable Transformer using LLaMA architecture (pre-norm + RMSNorm + SwiGLU).
    Captures global spatial-temporal context.
    """

    def __init__(self, embed_dim: int, num_heads: int = 2):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        config = LlamaConfig(
            hidden_size=embed_dim,
            intermediate_size=embed_dim,
            num_hidden_layers=2,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=500,
            pad_token_id=0,
            vocab_size=32,       # minimum accepted by HuggingFace LLaMA
            dropout_ratio=0.3,    # now is used
            weight_decay=0.5,     # now is used
        )
        self.llama = EEGLlamaForCausalLM(config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.llama(
            inputs_embeds=x,
            output_hidden_states=True,
        ).hidden_states[-1]
        return self.dropout(x + out)


# ─── Main architecture ───────────────────────────────────────────────────────

class EEGEncoderArchitecture(nn.Module):
    """
    EEGEncoder: Downsampling Projector + N parallel DSTS blocks.
    Each DSTS block = dropout + TCN + Stable Transformer (LLaMA).
    """

    def __init__(
        self,
        n_classes: int = 4,
        in_chans: int = 22,
        n_windows: int = 5,
        eegn_F1: int = 16,
        eegn_D: int = 2,
        eegn_kern_size: int = 64,
        eegn_pool_size: int = 7,
        eegn_dropout: float = 0.3,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 4,
        tcn_filters: int = 32,
        tcn_dropout: float = 0.3,
        fuse: Literal['average', 'concat'] = 'average',
    ):
        super().__init__()
        self.n_windows = n_windows
        self.fuse = fuse
        F2 = eegn_F1 * eegn_D

        self.conv_block = ConvBlock(
            F1=eegn_F1, kern_length=eegn_kern_size,
            pool_size=eegn_pool_size, D=eegn_D,
            in_chans=in_chans, dropout=eegn_dropout,
        )
        self.aa_drop = nn.Dropout(0.3)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(F2, tcn_depth, tcn_kernel_size, tcn_filters, tcn_dropout)
            for _ in range(n_windows)
        ])
        self.trm_blocks = nn.ModuleList([
            StableTransformerBlock(embed_dim=F2, num_heads=2)
            for _ in range(n_windows)
        ])
        self.dense_layers = nn.ModuleList([
            LinearL2(tcn_filters, n_classes, weight_decay=0.5)
            for _ in range(n_windows)
        ])
        if fuse == 'concat':
            self.final_dense = LinearL2(n_classes * n_windows, n_classes, weight_decay=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, n_times)
        x = self.conv_block(x)
        x = x[:, :, :, 0].permute(0, 2, 1)    # (batch, seq_len, F2)

        sw_outputs = []
        for i in range(self.n_windows):
            window = self.aa_drop(x)
            tcn_out = self.tcn_blocks[i](window)
            tcn_out = tcn_out[:, -1, :]                                             # (batch, tcn_filters)
            trm_out = self.trm_blocks[i](window).mean(1)                            # (batch, F2)
            fused = tcn_out + F.dropout(trm_out, p=0.3, training=self.training)
            sw_outputs.append(self.dense_layers[i](fused))

        if self.fuse == 'average':
            return torch.mean(torch.stack(sw_outputs, dim=0), dim=0)
        else:
            out = torch.cat(sw_outputs, dim=1)
            return self.final_dense(out)


# ─── BaseModel wrapper ───────────────────────────────────────────────────────

class EEGEncoderModel(BaseModel):
    """
    EEGEncoder classifier — wraps EEGEncoderArchitecture with BaseModel interface.
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
        eegn_dropout: float = 0.3,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 4,
        tcn_filters: int = 32,
        tcn_dropout: float = 0.3,
        fuse: Literal['average', 'concat'] = 'average',
        lr: float = 1e-3,
        n_epochs: int = 500,
        batch_size: int = 64,
        verbose: bool = False,
        device: Optional[Literal['cuda', 'cpu']] = None,
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
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self._device_str = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self._device_str)
        self._label_encoder = LabelEncoder()
        self.model = EEGEncoderArchitecture(
            n_classes=n_classes,
            in_chans=n_channels,
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
        ).to(self.device)

    def _to_tensor(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # X: (n_trials, n_channels, n_times) -> (n_trials, 1, n_channels, n_times)
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        if y is not None:
            return X_tensor, torch.LongTensor(y).to(self.device)
        return X_tensor, None

    def _l2_loss(self) -> torch.Tensor:
        """Sum L2 losses from all Conv/Linear L2 modules."""
        return sum(
            m.l2_loss()
            for m in self.model.modules()
            if hasattr(m, 'l2_loss')
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_encoded = self._label_encoder.fit_transform(y)
        X_tensor, y_tensor = self._to_tensor(X, y_encoded)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
        scaler = torch.amp.GradScaler(self._device_str, enabled=self._device_str == 'cuda')

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self._device_str, enabled=self._device_str == 'cuda'):
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch) + self._l2_loss()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f'    Epoch [{epoch+1}/{self.n_epochs}] loss: {epoch_loss/len(loader):.4f}')

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet, call fit() first')
        self.model.eval()
        with torch.no_grad():
            X_tensor, _ = self._to_tensor(X)
            with torch.amp.autocast(device_type=self._device_str, enabled=self._device_str == 'cuda'):
                logits = self.model(X_tensor)
            y_encoded = torch.argmax(logits, dim=1).cpu().numpy()
        return self._label_encoder.inverse_transform(y_encoded)

    def clone(self) -> 'EEGEncoderModel':
        return EEGEncoderModel(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            n_windows=self.n_windows,
            eegn_F1=self.eegn_F1,
            eegn_D=self.eegn_D,
            eegn_kern_size=self.eegn_kern_size,
            eegn_pool_size=self.eegn_pool_size,
            eegn_dropout=self.eegn_dropout,
            tcn_depth=self.tcn_depth,
            tcn_kernel_size=self.tcn_kernel_size,
            tcn_filters=self.tcn_filters,
            tcn_dropout=self.tcn_dropout,
            fuse=self.fuse,
            lr=self.lr,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            device=self._device_str,
        )

    def save(self, path: str) -> None:
        """Save model weights and hyperparameters."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'label_encoder': self._label_encoder,
            'hyperparams': {
                'n_channels': self.n_channels,
                'n_classes': self.n_classes,
                'n_windows': self.n_windows,
                'eegn_F1': self.eegn_F1,
                'eegn_D': self.eegn_D,
                'eegn_kern_size': self.eegn_kern_size,
                'eegn_pool_size': self.eegn_pool_size,
                'eegn_dropout': self.eegn_dropout,
                'tcn_depth': self.tcn_depth,
                'tcn_kernel_size': self.tcn_kernel_size,
                'tcn_filters': self.tcn_filters,
                'tcn_dropout': self.tcn_dropout,
                'fuse': self.fuse,
                'lr': self.lr,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'device': self._device_str,
            }
        }, path)
        print(f'  Model saved to {path}')

    @classmethod
    def load(cls, path: str) -> 'EEGEncoderModel':
        """Load model from disk."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(**checkpoint['hyperparams'])
        model.model.load_state_dict(checkpoint['state_dict'])
        model._label_encoder = checkpoint['label_encoder']
        model.is_fitted = True
        print(f'  Model loaded from {path}')
        return model

    def __repr__(self) -> str:
        return (
            f'EEGEncoderModel('
            f'channels={self.n_channels}, '
            f'n_classes={self.n_classes}, '
            f'n_windows={self.n_windows}, '
            f'device={self.device}, '
            f'fitted={self.is_fitted})'
        )