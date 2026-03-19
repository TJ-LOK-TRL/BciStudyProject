from typing import Optional, Literal
from pathlib import Path
import importlib
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from src.models.core.fittable import IFittable
from src.models.nn.base_neural_model import BaseNN
from src.models.core.hyperparametrizable import IHyperparametrizable
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig


class NNWrapper(IFittable):
    """
    Generic wrapper — pairs any BaseNN with a Trainer.
    Handles: fit, predict, clone, save, load, label encoding.
    Model handles: forward, get_hyperparams, from_hyperparams.

    save/load is fully automatic for any BaseNN + Hyperparametrizable subclass.

    Usage:
        wrapper = NNWrapper(
            arch=EEGEncoderModel(n_channels=22, n_classes=4),
            config=TrainerConfig.for_eeg_encoder(),
        )
    """

    def __init__(
        self,
        arch: BaseNN,
        config: Optional[TrainerConfig] = None,
        device: Optional[Literal['cuda', 'cpu']] = None,
    ):
        super().__init__()
        self._device_str = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self._device_str)
        self._label_encoder = LabelEncoder()
        self.train_config = config or TrainerConfig()
        self.model = arch.to(self.device)
        self._trainer = Trainer(
            model_arch=self.model,
            device=self.device,
            config=self.train_config,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self._trainer.fit(
            X_train=X,
            y_train=self._label_encoder.fit_transform(y),
            X_val=X_val,
            y_val=self._label_encoder.transform(y_val) if y_val is not None else None,
            label_encoder=self._label_encoder,
        )
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted yet, call fit() first')
        self.model.eval()
        with torch.no_grad():
            X_tensor, _ = self._trainer._to_tensor(X)
            with torch.amp.autocast(
                device_type=self._device_str,
                enabled=self._device_str == 'cuda',
            ):
                logits = self.model(X_tensor)
            y_encoded = torch.argmax(logits, dim=1).cpu().numpy()
        return self._label_encoder.inverse_transform(y_encoded)

    def clone(self) -> 'NNWrapper':
        # reconstruct cleanly from hyperparams
        if not isinstance(self.model, IHyperparametrizable):
            raise RuntimeError(
                f'{self.model.__class__.__name__} must implement Hyperparametrizable '
                f'to support clone()'
            )
        new_arch = self.model.__class__.from_hyperparams(self.model.get_hyperparams())
        return NNWrapper(
            arch=new_arch,
            config=self.train_config,
            device=self._device_str,
        )

    def save(self, path: str) -> None:
        if not isinstance(self.model, IHyperparametrizable):
            raise RuntimeError(
                f'{self.model.__class__.__name__} must implement Hyperparametrizable '
                f'to support save()'
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'arch_class':       self.model.__class__.__name__,
            'arch_module':      self.model.__class__.__module__,
            'arch_hyperparams': self.model.get_hyperparams(),
            'state_dict':       self.model.state_dict(),
            'label_encoder':    self._label_encoder,
            'train_config':     self.train_config.to_dict(),
            'device':           self._device_str,
        }, path)
        print(f'  Model saved to {path}')

    @classmethod
    def load(cls, path: str) -> 'NNWrapper':
        ckpt = torch.load(path, map_location='cpu')
        module = importlib.import_module(ckpt['arch_module'])
        arch_cls = getattr(module, ckpt['arch_class'])
        arch = arch_cls.from_hyperparams(ckpt['arch_hyperparams'])
        config = TrainerConfig.from_dict(ckpt['train_config'])
        wrapper = cls(arch=arch, config=config, device=ckpt['device'])
        wrapper.model.load_state_dict(ckpt['state_dict'])
        wrapper._label_encoder = ckpt['label_encoder']
        wrapper.is_fitted = True
        print(f'  Model loaded from {path}')
        return wrapper

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def __repr__(self) -> str:
        return (
            f'NNWrapper('
            f'arch={self.model.__class__.__name__}, '
            f'device={self.device}, '
            f'fitted={self.is_fitted})'
        )