from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class LoggerCallbackConfig:
    every_n_epochs: int = 10
    metrics: List[str] = field(
        default_factory=lambda: ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    )


@dataclass
class EarlyStoppingCallbackConfig:
    patience: int = 50
    min_delta: float = 1e-4
    monitor: str = 'val_loss'
    mode: Literal['min', 'max'] = 'min'


@dataclass
class BestModelCallbackConfig:
    monitor: str = 'val_acc'
    mode: Literal['min', 'max'] = 'max'


@dataclass
class CheckpointCallbackConfig:
    checkpoint_dir: str = 'results/checkpoints'
    every_n_epochs: int = 50