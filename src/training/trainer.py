# src/training/trainer.py

from typing import Optional, List, Dict, Type
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from src.training.callbacks import Callback, EarlyStoppingCallback, CheckpointCallback


class Trainer:
    """
    Centralized training loop for all PyTorch EEG models.

    Loss:
        train_loss = loss_scale * CrossEntropy(label_smoothing) + l2_scale * manual_L2
        val_loss   = CrossEntropy(no smoothing, no L2)

    Gradient clipping is applied before every optimizer step to prevent
    loss spikes / gradient explosion (especially with GRU + Transformer).
    """

    def __init__(
        self,
        model_arch: nn.Module,
        device: torch.device,
        n_epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        label_smoothing: float = 0.0,
        loss_scale: float = 1.0,
        l2_scale: float = 0.0,
        grad_clip: float = 0.0,       # max gradient norm - set 0 to disable
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        callbacks: Optional[List[Callback]] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model_arch = model_arch
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.loss_scale = loss_scale
        self.l2_scale = l2_scale
        self.grad_clip = grad_clip
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.callbacks = callbacks or []
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def _l2_loss(self) -> torch.Tensor:
        total = sum(
            m.l2_loss()
            for m in self.model_arch.modules()
            if hasattr(m, 'l2_loss')
        )
        return total if isinstance(total, torch.Tensor) else torch.tensor(0.0, device=self.device)

    def _compute_accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        return (torch.argmax(logits, dim=1) == y).float().mean().item()

    def _to_tensor(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        if y is not None:
            return X_tensor, torch.LongTensor(y).to(self.device)
        return X_tensor, None

    def set_run_id(self, run_id: str) -> None:
        self.current_run_id = run_id
        for cb in self.callbacks:
            if isinstance(cb, CheckpointCallback):
                cb.set_run_id(run_id)

    def _resume_if_exists(self, optimizer, scaler) -> int:
        if self.checkpoint_dir is None:
            return 0
        run_id = getattr(self, 'current_run_id', 'default')
        path = self.checkpoint_dir / f'checkpoint_{run_id}.pt'
        if not path.exists():
            return 0
        print(f'  Resuming from checkpoint: {path}')
        ckpt = torch.load(path, map_location=self.device)
        self.model_arch.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        print(f'  Resumed from epoch {ckpt["epoch"] + 1}')
        return ckpt['epoch'] + 1

    def _restore_best_weights(self) -> None:
        from src.training.callbacks import EarlyStoppingCallback
        try:
            from src.training.callbacks import BestModelCallback
        except ImportError:
            BestModelCallback = None

        for cb in self.callbacks:
            if isinstance(cb, EarlyStoppingCallback) and cb.best_state is not None:
                print(f'  Restoring best weights (val_loss: {cb.best_val_loss:.4f})')
                self.model_arch.load_state_dict(
                    {k: v.to(self.device) for k, v in cb.best_state.items()}
                )
                return

        if BestModelCallback is not None:
            for cb in self.callbacks:
                if isinstance(cb, BestModelCallback) and cb.best_state is not None:
                    print(
                        f'  Restoring best weights '
                        f'(epoch {cb.best_epoch}, {cb.monitor}: {cb._best_value:.4f})'
                    )
                    self.model_arch.load_state_dict(
                        {k: v.to(self.device) for k, v in cb.best_state.items()}
                    )
                    return

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        label_encoder: LabelEncoder,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        for cb in self.callbacks:
            cb.reset()

        X_tensor, y_tensor = self._to_tensor(X_train, y_train)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val_tensor, y_val_tensor = self._to_tensor(X_val, y_val)

        optimizer = self.optimizer_cls(
            self.model_arch.parameters(),
            lr=self.lr,
            **self.optimizer_kwargs,
        )

        train_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        val_criterion   = nn.CrossEntropyLoss()

        is_cuda = self.device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

        start_epoch = self._resume_if_exists(optimizer, scaler)

        # debug on first epoch only
        _debug_done = False

        for epoch in range(start_epoch, self.n_epochs):
            self.model_arch.train()
            train_loss, train_acc = 0.0, 0.0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    logits = self.model_arch(X_batch)
                    l2_loss = self.l2_scale * self._l2_loss()
                    loss = self.loss_scale * train_criterion(logits, y_batch) + l2_loss

                scaler.scale(loss).backward()

                # gradient clipping — prevents loss spikes in GRU/Transformer models
                if self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model_arch.parameters(), max_norm=self.grad_clip
                    )

                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                train_acc  += self._compute_accuracy(logits.detach(), y_batch)

            train_loss /= len(loader)
            train_acc  /= len(loader)

            val_loss, val_acc = None, None
            if use_val:
                self.model_arch.eval()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=is_cuda):
                        val_logits = self.model_arch(X_val_tensor)
                        val_loss = val_criterion(val_logits, y_val_tensor).item()
                        val_acc  = self._compute_accuracy(val_logits, y_val_tensor)
                self.model_arch.train()

            if not _debug_done:
                with torch.no_grad():
                    sample = next(iter(loader))
                    _logits = self.model_arch(sample[0])
                    _ce  = train_criterion(_logits, sample[1]).item()
                    _l2  = self._l2_loss().item()
                print(
                    f'  DEBUG epoch 0 — '
                    f'CE: {_ce:.4f}, L2: {_l2:.4f}, '
                    f'l2_scale: {self.l2_scale}'
                )
                _debug_done = True

            logs: Dict = {
                'train_loss':      train_loss,
                'train_acc':       train_acc,
                'val_loss':        val_loss,
                'val_acc':         val_acc,
                'model_state':     self.model_arch.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state':    scaler.state_dict(),
                'label_encoder':   label_encoder,
            }

            stop = False
            for cb in self.callbacks:
                if cb.on_epoch_end(epoch, logs):
                    stop = True
            if stop:
                break

        self._restore_best_weights()