from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Literal
from pathlib import Path
import torch


class Callback(ABC):
    """Base class for training callbacks."""

    def on_epoch_end(self, epoch: int, logs: Dict) -> bool:
        """Called after each epoch. Return True to stop training."""
        return False
    
    def reset(self) -> None:
        """Reset internal state between runs. Override if stateful."""
        pass

    @abstractmethod
    def clone(self) -> 'Callback':
        """Return a new instance with same config but fresh state."""
        pass


class EarlyStoppingCallback(Callback):
    """
    Stop training when a monitored metric stops improving.
 
    Args:
        patience:  Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        monitor:   Metric to monitor. Must be a key in the logs dict.
                   Common values: 'val_loss', 'val_acc', 'train_loss'.
        mode:      'min' — stop when metric stops decreasing (e.g. val_loss)
                   'max' — stop when metric stops increasing (e.g. val_acc)
 
    Examples:
        # Default — monitor val_loss (theoretically more robust)
        EarlyStoppingCallback(patience=50)
 
        # Paper behaviour — monitor val_acc
        EarlyStoppingCallback(patience=50, monitor='val_acc', mode='max')
    """
 
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-4,
        monitor: str = 'val_loss',
        mode: Literal['min', 'max'] = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
 
        self._best_value: float = float('inf') if mode == 'min' else float('-inf')
        self._patience_counter: int = 0
        self.best_state: Optional[Dict] = None
 
        # expose best_val_loss for trainer's restore message — always tracks val_loss
        self.best_val_loss: float = float('inf')
 
    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'min':
            return current < self._best_value - self.min_delta
        else:
            return current > self._best_value + self.min_delta
 
    def on_epoch_end(self, epoch: int, logs: Dict) -> bool:
        value = logs.get(self.monitor)
        if value is None:
            raise ValueError(
                f'EarlyStoppingCallback: \'{self.monitor}\' not found in logs. '
                f'Available keys: {list(logs.keys())}'
            )
 
        # always track val_loss for the restore message, regardless of monitor
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
 
        if self._is_improvement(value):
            self._best_value = value
            self._patience_counter = 0
            self.best_state = {
                k: v.cpu().clone()
                for k, v in logs['model_state'].items()
            }
        else:
            self._patience_counter += 1
 
        if self._patience_counter >= self.patience:
            print(
                f'  Early stopping at epoch {epoch + 1} — '
                f'best {self.monitor}: {self._best_value:.4f}'
            )
            return True
        return False
 
    def reset(self) -> None:
        """Reset state — must be called before each new training run."""
        self._best_value = float('inf') if self.mode == 'min' else float('-inf')
        self._patience_counter = 0
        self.best_state = None
        self.best_val_loss = float('inf')
 
    def clone(self) -> 'EarlyStoppingCallback':
        return EarlyStoppingCallback(
            patience=self.patience,
            min_delta=self.min_delta,
            monitor=self.monitor,
            mode=self.mode,
        )


class BestModelCallback(Callback):
    """
    Tracks the best model weights over the full training run
    without stopping training early.

    At the end of training, the trainer restores these weights
    so predict() uses the best epoch, not the last.

    Args:
        monitor: Metric to track. Must be a key in the logs dict.
                 Common values: 'val_acc' (paper behaviour), 'val_loss'.
        mode:    'max' for accuracy-like metrics, 'min' for loss-like metrics.

    Example:
        # Paper behaviour — best val_acc over 500 epochs, no early stop
        BestModelCallback(monitor='val_acc', mode='max')

        # Alternative — best val_loss
        BestModelCallback(monitor='val_loss', mode='min')
    """

    def __init__(
        self,
        monitor: str = 'val_acc',
        mode: Literal['min', 'max'] = 'max',
    ):
        self.monitor = monitor
        self.mode = mode
        self._best_value: float = float('-inf') if mode == 'max' else float('inf')
        self.best_state: Optional[Dict] = None
        self.best_epoch: int = -1

        # exposed for trainer restore message
        self.best_val_loss: float = float('inf')

    def _is_improvement(self, current: float) -> bool:
        if self.mode == 'max':
            return current > self._best_value
        else:
            return current < self._best_value

    def on_epoch_end(self, epoch: int, logs: Dict) -> bool:
        value = logs.get(self.monitor)
        if value is None:
            raise ValueError(
                f"BestModelCallback: '{self.monitor}' not found in logs. "
                f"Available keys: {list(logs.keys())}"
            )

        # always track val_loss for restore message
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        if self._is_improvement(value):
            self._best_value = value
            self.best_epoch = epoch + 1
            self.best_state = {
                k: v.cpu().clone()
                for k, v in logs['model_state'].items()
            }

        return False  # never stops training

    def reset(self) -> None:
        self._best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.best_state = None
        self.best_epoch = -1
        self.best_val_loss = float('inf')

    def clone(self) -> 'BestModelCallback':
        return BestModelCallback(
            monitor=self.monitor,
            mode=self.mode,
        )


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir: str, every_n_epochs: int = 50):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_n_epochs = every_n_epochs
        self.current_id: str = 'default'

    def set_run_id(self, run_id: str) -> None:
        """Called before each subject/fold to set the checkpoint filename."""
        self.current_id = run_id

    def on_epoch_end(self, epoch: int, logs: Dict) -> bool:
        if (epoch + 1) % self.every_n_epochs == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.checkpoint_dir / f'checkpoint_{self.current_id}.pt'
            torch.save({
                'epoch': epoch,
                'run_id': self.current_id,
                'model_state': logs['model_state'],
                'optimizer_state': logs['optimizer_state'],
                'scaler_state': logs['scaler_state'],
                'label_encoder': logs['label_encoder'],
            }, path)
            print(f'  Checkpoint saved: {path}')
        return False
    
    def clone(self) -> 'CheckpointCallback':
        return CheckpointCallback(
            checkpoint_dir=str(self.checkpoint_dir),
            every_n_epochs=self.every_n_epochs,
        )


class LoggerCallback(Callback):
    """Print training metrics every N epochs."""

    def __init__(self, every_n_epochs: int = 10, metrics: Optional[List[str]] = None):
        self.every_n_epochs = every_n_epochs
        # which metrics to print — extendable without touching trainer
        self.metrics = metrics or ['train_loss', 'val_loss', 'train_acc', 'val_acc']

    def on_epoch_end(self, epoch: int, logs: Dict) -> bool:
        if (epoch + 1) % self.every_n_epochs == 0:
            available = {k: v for k, v in logs.items() if k in self.metrics and v is not None}
            metrics_str = '  '.join(f'{k}: {v:.4f}' for k, v in available.items())
            print(f'  Epoch [{epoch + 1}] {metrics_str}')
        return False
    
    def clone(self) -> 'LoggerCallback':
        return LoggerCallback(
            every_n_epochs=self.every_n_epochs,
            metrics=self.metrics.copy(),
        )