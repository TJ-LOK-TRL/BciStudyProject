from typing import Optional
from dataclasses import dataclass, field
from dacite import from_dict as dacite_from_dict
from src.input_adapters import BaseInputAdapter, CNN2DAdapter
from src.training.callbacks import (
    LoggerCallbackConfig, EarlyStoppingCallbackConfig, 
    BestModelCallbackConfig, CheckpointCallbackConfig
)


@dataclass
class TrainerConfig:
    """
    Serializable trainer configuration.
    Pass to Trainer instead of individual kwargs for cleaner experiment definitions.
    """
    n_epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    label_smoothing: float = 0.0
    loss_scale: float = 1.0
    l2_scale: float = 0.0
    grad_clip: float = 0.0
    weight_decay: float = 0.0
    scheduler: str = 'none'
    optimizer: str = 'adam'
    optimizer_kwargs: dict = field(default_factory=dict)
    input_adapter: BaseInputAdapter = CNN2DAdapter() # Default

    # callback configs
    logger: Optional[LoggerCallbackConfig] = field(
        default_factory=LoggerCallbackConfig
    )
    early_stopping: Optional[EarlyStoppingCallbackConfig] = field(
        default_factory=EarlyStoppingCallbackConfig
    )
    best_model: Optional[BestModelCallbackConfig] = None      # disabled by default
    checkpoint: Optional[CheckpointCallbackConfig] = None     # disabled by default

    def to_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        d['input_adapter'] = self.input_adapter.__class__.__name__ # TODO: Maybe use enums in the future
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainerConfig':
        from src.input_adapters import get_adapter  # imports trigger registration
        d = d.copy()
        adapter_name = d.pop('input_adapter', 'CNN2DAdapter')  # remove do dict
        config = dacite_from_dict(data_class=cls, data=d)
        config.input_adapter = get_adapter(adapter_name)
        return config