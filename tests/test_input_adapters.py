import torch
from src.input_adapters.temporal import to_sequence_input

def test_to_sequence_input():
    x = torch.randn(2, 3, 5)
    y = to_sequence_input(x)
    assert y.shape == (2, 5, 3)
