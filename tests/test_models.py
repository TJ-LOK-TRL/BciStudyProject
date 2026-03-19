import pytest, torch
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.cnn1d import CNN1DModel
from src.models.liquid import LiquidNNBasic

@pytest.mark.parametrize("model_name", ["lstm", "gru", "cnn1d", "liquid"])
def test_forward_shapes(model_name):
    B, C, T, n_classes = 4, 32, 500, 4
    if model_name in ['lstm','gru','liquid']:
        x = torch.randn(B, T, C)
        if model_name=='lstm':
            model = LSTMModel(C, n_classes)
        elif model_name=='gru':
            model = GRUModel(C, n_classes)
        else:
            model = LiquidNNBasic(C, n_classes)
    else:
        x = torch.randn(B, C, T)
        model = CNN1DModel(C, n_classes)
    logits = model(x)
    assert logits.shape == (B, n_classes)
