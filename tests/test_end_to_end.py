import pytest
import torch
from src.training.trainer import Trainer, TrainerConfig
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel

@pytest.mark.parametrize("model_cls", [LSTMModel, GRUModel])
def test_smoke_training(model_cls):
    B, C, T, n_classes = 8, 32, 200, 4
    Xtr = torch.randn(B, T, C)
    ytr = torch.randint(0, n_classes, (B,))
    Xva = torch.randn(B//2, T, C)
    yva = torch.randint(0, n_classes, (B//2,))
    model = model_cls(C, n_classes)
    train_loader = [(Xtr, ytr)]
    val_loader = [(Xva, yva)]
    cfg = TrainerConfig(epochs=1, batch_size=8, lr=1e-3)
    trainer = Trainer(model, train_loader, val_loader, cfg)
    score, hist = trainer.train_and_evaluate(single_epoch=True)
    assert isinstance(score, float)
