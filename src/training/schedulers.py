# src/training/schedulers.py
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


def build_scheduler(optimizer: Optimizer, name: str, **kwargs):
    """Build learning rate scheduler by name."""
    if name == 'none' or name is None:
        return None
    elif name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 100))
    elif name == 'step':
        return StepLR(optimizer, step_size=kwargs.get('step_size', 50), gamma=kwargs.get('gamma', 0.5))
    elif name == 'plateau':
        return ReduceLROnPlateau(optimizer, patience=kwargs.get('patience', 10))
    else:
        raise ValueError(f'Unknown scheduler: {name}')