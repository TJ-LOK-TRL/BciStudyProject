from abc import ABC, abstractmethod
from typing import Dict, Optional, List
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
    """Stop training when val_loss stops improving."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss: float = float('inf')
        self.patience_counter: int = 0
        self.best_state: Optional[Dict] = None

    def on_epoch_end(self, epoch: int, logs: Dict) -> bool:
        val_loss = logs.get('val_loss')
        if val_loss is None:
            raise Exception('val_loss is None.')

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_state = {
                k: v.cpu().clone()
                for k, v in logs['model_state'].items()
            }
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f'  Early stopping at epoch {epoch + 1} — best val loss: {self.best_val_loss:.4f}')
            return True
        return False
    
    def reset(self) -> None:
        """Reset state — must be called before each new training run."""
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_state = None

    def clone(self) -> 'EarlyStoppingCallback':
        return EarlyStoppingCallback(
            patience=self.patience,
            min_delta=self.min_delta,
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