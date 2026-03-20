# src/training/trainer.py

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Type
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from src.training.callbacks.callbacks import Callback, EarlyStoppingCallback, CheckpointCallback
from src.training.schedulers import build_scheduler
from src.training.trainer_config import TrainerConfig
from src.training.optimizers import build_optimizer


class Trainer:
    """
    Centralized training loop for all PyTorch EEG models.

    Loss:
        train_loss = loss_scale * CrossEntropy(label_smoothing) + l2_scale * manual_L2
        val_loss   = CrossEntropy(no smoothing, no L2)
    """

    def __init__(
        self,
        model_arch: nn.Module,
        device: torch.device,
        config: TrainerConfig,
    ):
        self.model_arch = model_arch
        self.device = device
        self.cfg = config
        self.callbacks = self._build_callbacks()    
        self.checkpoint_dir = (
            Path(config.checkpoint.checkpoint_dir) if config.checkpoint else None
        )

    def _build_callbacks(self) -> List[Callback]:
        from src.training.callbacks import (
            LoggerCallback,
            EarlyStoppingCallback,
            BestModelCallback,
            CheckpointCallback,
        )
        callbacks = []

        if self.cfg.logger is not None:
            callbacks.append(LoggerCallback(
                every_n_epochs=self.cfg.logger.every_n_epochs,
                metrics=self.cfg.logger.metrics,
            ))
        if self.cfg.early_stopping is not None:
            callbacks.append(EarlyStoppingCallback(
                patience=self.cfg.early_stopping.patience,
                min_delta=self.cfg.early_stopping.min_delta,
                monitor=self.cfg.early_stopping.monitor,
                mode=self.cfg.early_stopping.mode,
            ))
        if self.cfg.best_model is not None:
            callbacks.append(BestModelCallback(
                monitor=self.cfg.best_model.monitor,
                mode=self.cfg.best_model.mode,
            ))
        if self.cfg.checkpoint is not None:
            callbacks.append(CheckpointCallback(
                checkpoint_dir=self.cfg.checkpoint.checkpoint_dir,
                every_n_epochs=self.cfg.checkpoint.every_n_epochs,
            ))
        return callbacks

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
        adapter = self.cfg.input_adapter
        X_tensor = adapter.transform(X).to(self.device)
        if y is not None:
            return X_tensor, torch.LongTensor(y).to(self.device)
        return X_tensor, None

    def set_run_id(self, run_id: str) -> None:
        self.current_run_id = run_id
        for cb in self.callbacks:
            if isinstance(cb, CheckpointCallback):
                cb.set_run_id(run_id)

    def _resume_if_exists(
        self, 
        optimizer: torch.optim.Optimizer, 
        scaler: torch.amp.GradScaler
    ) -> int:
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
        from src.training.callbacks.callbacks import BestModelCallback
        for cb in self.callbacks:
            if isinstance(cb, (EarlyStoppingCallback, BestModelCallback)) and cb.best_state is not None:
                print(
                    f'  Restoring best weights '
                    f'(epoch {getattr(cb, "best_epoch", "?")}, '
                    f'{cb.monitor}: {cb._best_value:.4f})'
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
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val_tensor, y_val_tensor = self._to_tensor(X_val, y_val)

        optimizer = build_optimizer(
            self.model_arch.parameters(),
            name=self.cfg.optimizer,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            **self.cfg.optimizer_kwargs,
        )

        # build scheduler
        scheduler = build_scheduler(
            optimizer,
            self.cfg.scheduler,
            T_max=self.cfg.n_epochs,
        )

        train_criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        val_criterion   = nn.CrossEntropyLoss()

        is_cuda = self.device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

        start_epoch = self._resume_if_exists(optimizer, scaler)
        _debug_done = False

        for epoch in range(start_epoch, self.cfg.n_epochs):
            self.model_arch.train()
            train_loss, train_acc = 0.0, 0.0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    logits = self.model_arch(X_batch)
                    l2_loss = self.cfg.l2_scale * self._l2_loss()
                    loss = self.cfg.loss_scale * train_criterion(logits, y_batch) + l2_loss

                scaler.scale(loss).backward()

                if self.cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model_arch.parameters(),
                        max_norm=self.cfg.grad_clip,
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

            # scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if use_val and val_loss is not None:
                        scheduler.step(val_loss)
                else:
                    scheduler.step()

            if not _debug_done:
                with torch.no_grad():
                    sample = next(iter(loader))
                    _logits = self.model_arch(sample[0])
                    _ce  = train_criterion(_logits, sample[1]).item()
                    _l2  = self._l2_loss().item()
                print(
                    f'  DEBUG epoch 0 — '
                    f'CE: {_ce:.4f}, L2: {_l2:.4f}, '
                    f'l2_scale: {self.cfg.l2_scale}, '
                    f'scheduler: {self.cfg.scheduler}'
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