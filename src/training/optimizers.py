# src/training/optimizers.py
from typing import Dict, Type
import torch
from torch.optim import Optimizer


_OPTIMIZER_REGISTRY: Dict[str, Type[Optimizer]] = {
    'adam':    torch.optim.Adam,
    'adamw':   torch.optim.AdamW,
    'sgd':     torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
}


def build_optimizer(
    params,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    if name not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f'Unknown optimizer: {name!r}. '
            f'Available: {list(_OPTIMIZER_REGISTRY.keys())}'
        )
    return _OPTIMIZER_REGISTRY[name](
        params,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs,
    )